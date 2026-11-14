"""Summarize a fixed-top-5/adaptive run with paired question bootstrap CIs."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import re
import math

import numpy as np


def load_setting(directory, setting):
    matches = list(directory.glob(f"adaptive_topk_{setting}_*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {setting} result in {directory}, got {len(matches)}")
    return matches[0], json.loads(matches[0].read_text(encoding="utf-8"))


def question_metrics(result, metric):
    repeats = result["samples_by_repeat"]
    ids = [sample["id"] for sample in repeats[0]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs")
    for repeat in repeats:
        if [sample["id"] for sample in repeat] != ids:
            raise ValueError("Question order changed between repeats")
    return ids, np.array([
        statistics.mean(repeat[index]["metric_scores"][metric] for repeat in repeats)
        for index in range(len(ids))
    ])


def summarize(result):
    repeats = result["samples_by_repeat"]
    first = repeats[0]
    if any(sample["selected_k"] != len(sample["retrieved_documents"])
           for repeat in repeats for sample in repeat):
        raise ValueError("Selected k does not match saved document count")
    return {
        "sample_count": result["sample_count"],
        "repeats": result["repeats"],
        "quality": result["mean_metrics"],
        "mean_documents": result["mean_docs_per_query"],
        "mean_generator_input_tokens": result["generator_avg_input_tokens"],
        "selected_k_distribution": dict(sorted(Counter(sample["selected_k"] for sample in first).items())),
        "decision_reasons": dict(Counter(sample["adaptive_decision"]["reason"]
                                          for sample in first if sample["adaptive_decision"])),
        "end_to_end_seconds_by_repeat": result["end_to_end_seconds_by_repeat"],
        "median_end_to_end_seconds": result["median_end_to_end_s"],
        "stage_median_seconds": {key: statistics.median(values)
                                  for key, values in result["stage_seconds_by_repeat"].items() if values},
        "repeat_prediction_disagreements": sum(
            sample["prediction"] != first[index]["prediction"]
            for repeat in repeats[1:] for index, sample in enumerate(repeat)
        ),
    }


def token_log_summary(path, sample_count, batch_size=1):
    """Batch-size-one logs give exact lengths; larger batches give padded slots."""
    if not path.exists():
        return None
    data = path.read_bytes()
    content = data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8", errors="replace")
    measured = {}
    current = None
    for line in content.splitlines():
        if "[experiment]" in line or "[warmup " in line:
            current = None
        marker = re.search(r"\[repeat (\d+)/(\d+), setting=(\w+)\]", line)
        if marker:
            current = (marker.group(3), int(marker.group(1)))
            measured[current] = []
        tokens = re.search(r"\[tokens\] input=(\d+), generated=(\d+)", line)
        if current is not None and tokens:
            measured[current].append((int(tokens.group(1)), int(tokens.group(2))))
    settings = {}
    for (setting, repeat), counts in measured.items():
        expected_batches = math.ceil(sample_count / batch_size)
        if len(counts) != expected_batches:
            raise ValueError(f"Expected {expected_batches} token logs for {setting} repeat {repeat}, got {len(counts)}")
        if batch_size > 1:
            settings.setdefault(setting, []).append({
                "repeat": repeat, "samples": sample_count, "batches": len(counts),
                "input_token_slots_including_padding": sum(
                    row[0] * min(batch_size, sample_count - i * batch_size) for i, row in enumerate(counts)),
                "generated_token_slots_including_eos_padding": sum(
                    row[1] * min(batch_size, sample_count - i * batch_size) for i, row in enumerate(counts)),
                "note": "Output slots include padding after early EOS; they are not useful generated-token counts.",
            })
            continue
        settings.setdefault(setting, []).append({
            "repeat": repeat, "samples": len(counts),
            "mean_input_tokens": statistics.mean(row[0] for row in counts),
            "mean_generated_tokens_including_eos": statistics.mean(row[1] for row in counts),
            "hit_max_32_tokens": sum(row[1] == 32 for row in counts),
        })
    return settings


def summarize_replay(directory):
    """Pair answer-quality changes in a frozen-prompt generation study."""
    import hashlib
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flashrag.evaluator.metrics import ExactMatch, F1_Score, Sub_ExactMatch

    workload = json.loads((directory / "workload.json").read_text(encoding="utf-8"))
    source_path = Path(workload["source"])
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == workload["source_sha256"]
    samples = json.loads(source_path.read_text(encoding="utf-8"))["samples_by_repeat"][0][:len(workload["ids"])]
    assert [s["id"] for s in samples] == workload["ids"]
    rows = json.loads((directory / "measurements.json").read_text(encoding="utf-8"))
    comparison = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    names = list(dict.fromkeys(row["variant"] for row in rows))
    reference = comparison["reference"]
    predictions, scores = {}, {}
    config = {"dataset_name": "nq"}
    metrics = {"em": ExactMatch(config).calculate_em,
               "f1": lambda p, a: F1_Score(config).token_level_scores(p, a)["f1"],
               "acc": Sub_ExactMatch(config).calculate_sub_em}
    for name in names:
        group = [r for r in rows if r["variant"] == name]
        predictions[name] = group[0]["predictions"]
        assert all(r["predictions"] == predictions[name] for r in group), "Predictions differ across repeats"
        assert len(predictions[name]) == len(samples)
        scores[name] = {metric: np.array([function(p, s["golden_answers"])
                                         for p, s in zip(predictions[name], samples)])
                        for metric, function in metrics.items()}
        for row in group:
            for metric in metrics:
                assert np.isclose(scores[name][metric].mean(), row["quality"][metric])
    results = {}
    for name in names:
        result = {"text_differences": sum(a != b for a, b in zip(predictions[reference], predictions[name])),
                  "metrics": {}}
        for metric in metrics:
            delta = scores[name][metric] - scores[reference][metric]
            rng = np.random.default_rng(2024)
            boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
            result["metrics"][metric] = {
                "reference": float(scores[reference][metric].mean()), "variant": float(scores[name][metric].mean()),
                "difference": float(delta.mean()), "better_questions": int((delta > 0).sum()),
                "worse_questions": int((delta < 0).sum()),
                "paired_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
                "question_differences": delta.tolist(),
            }
        results[name] = result
    payload = {"reference": reference, "question_ids": workload["ids"], "comparisons": results,
               "note": "2000 paired question resamples, seed 2024. Intervals concern reused-question quality, not timing uncertainty or an independent quality test."}
    (directory / "quality_comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"texts_changed": result["text_differences"],
                             "metrics": {m: {k: v for k, v in row.items() if k != "question_differences"}
                                         for m, row in result["metrics"].items()}}
                      for name, result in results.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--replay", action="store_true", help="Analyze a frozen-prompt generation replay instead of a full RAG run.")
    args = parser.parse_args()
    if args.replay:
        summarize_replay(args.result_dir)
        return
    fixed_path, fixed = load_setting(args.result_dir, "k5")
    adaptive_path, adaptive = load_setting(args.result_dir, "adaptive_gap")
    if fixed["run_id"] != adaptive["run_id"]:
        raise ValueError("Settings must come from the same run")
    fixed_summary, adaptive_summary = summarize(fixed), summarize(adaptive)
    comparisons = {}
    for metric in ("em", "f1", "acc"):
        fixed_ids, fixed_scores = question_metrics(fixed, metric)
        adaptive_ids, adaptive_scores = question_metrics(adaptive, metric)
        if fixed_ids != adaptive_ids:
            raise ValueError("Settings must evaluate the same questions in the same order")
        differences = adaptive_scores - fixed_scores
        rng = np.random.default_rng(2024)
        bootstrap = np.array([rng.choice(differences, len(differences), replace=True).mean()
                              for _ in range(2000)])
        comparisons[metric] = {
            "adaptive_minus_fixed": float(differences.mean()),
            "paired_bootstrap_95_percent_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "adaptive_better_questions": int((differences > 0).sum()),
            "adaptive_worse_questions": int((differences < 0).sum()),
            "equal_questions": int((differences == 0).sum()),
        }
        if np.isin(fixed_scores, [0, 1]).all() and np.isin(adaptive_scores, [0, 1]).all():
            better = int((differences > 0).sum())
            worse = int((differences < 0).sum())
            discordant = better + worse
            comparisons[metric]["exact_mcnemar_two_sided_p"] = min(
                1.0, 2 * sum(math.comb(discordant, k) for k in range(min(better, worse) + 1))
                / (2 ** discordant)
            )
    payload = {
        "run_id": fixed["run_id"], "fixed_file": fixed_path.name, "adaptive_file": adaptive_path.name,
        "policy": adaptive["policy"], "fixed_top5": fixed_summary, "adaptive": adaptive_summary,
        "documents_reduction_percent": 100 * (1 - adaptive["mean_docs_per_query"] / fixed["mean_docs_per_query"]),
        "input_tokens_reduction_percent": 100 * (1 - adaptive["generator_avg_input_tokens"] / fixed["generator_avg_input_tokens"]),
        "latency_change_percent": 100 * (adaptive["median_end_to_end_s"] / fixed["median_end_to_end_s"] - 1),
        "paired_quality_comparison": comparisons,
        "measured_token_logs": token_log_summary(args.result_dir / "run.log", fixed["sample_count"],
                                                fixed["metadata"].get("generator_batch_size", 1)),
        "bootstrap_note": "2000 paired question resamples, seed 2024; question uncertainty, not latency uncertainty. Percentile intervals can be unreliable for sparse binary differences; also report exact McNemar p-values.",
    }
    destination = args.result_dir / "comparison.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for index, (fixed_sample, adaptive_sample) in enumerate(zip(
            fixed["samples_by_repeat"][0], adaptive["samples_by_repeat"][0])):
        row = {
            "id": fixed_sample["id"], "question": fixed_sample["question"],
            "golden_answers": json.dumps(fixed_sample["golden_answers"], ensure_ascii=False),
            "adaptive_k": adaptive_sample["selected_k"],
            "adaptive_reason": adaptive_sample["adaptive_decision"]["reason"],
            "largest_gap": adaptive_sample["adaptive_decision"]["largest_gap"],
            "fixed_input_tokens": fixed_sample["generator_input_tokens"],
            "adaptive_input_tokens": adaptive_sample["generator_input_tokens"],
            "fixed_prediction_repeat1": fixed_sample["prediction"],
            "adaptive_prediction_repeat1": adaptive_sample["prediction"],
        }
        for label, result in (("fixed", fixed), ("adaptive", adaptive)):
            for metric in ("em", "f1", "acc"):
                row[f"{label}_{metric}"] = statistics.mean(
                    repeat[index]["metric_scores"][metric] for repeat in result["samples_by_repeat"])
        rows.append(row)
    with (args.result_dir / "question_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
