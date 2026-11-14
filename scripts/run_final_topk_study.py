"""Reproducible, development-calibrated Adaptive Top-K final experiment.

Stages: prepare disjoint splits, audit resources, calibrate score-only policies
using development retrieval coverage, freeze them, then evaluate fresh questions.
No test gold answers or LLM outputs enter policy selection. All six settings use
the same generator configuration; formal runs rotate their order by repetition.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temp.replace(path)


def question_key(row):
    return " ".join(row["question"].casefold().split())


def prepare_splits(output, dev_count=200, test_count=500, seed=598):
    source = ROOT / "datasets/flashrag_official/nq/test.jsonl"
    provenance = json.loads((ROOT / "docs/nq_500_manifest.json").read_text(encoding="utf-8"))
    if digest(source) != provenance["source_sha256"]:
        raise ValueError("Official dataset checksum differs from the pinned source")
    used = read_jsonl(ROOT / "examples/quick_start/dataset/nq/test.jsonl")
    used += read_jsonl(ROOT / "datasets/nq_500_heldout/nq/test.jsonl")
    excluded = {question_key(row) for row in used}
    seen = set(excluded)
    eligible = []
    for row in read_jsonl(source):
        key = question_key(row)
        if key not in seen:
            eligible.append(row)
            seen.add(key)
    if dev_count + test_count > len(eligible):
        raise ValueError("Insufficient unused unique questions")
    selected = random.Random(seed).sample(eligible, dev_count + test_count)
    development, evaluation = selected[:dev_count], selected[dev_count:]
    if {r["id"] for r in development} & {r["id"] for r in evaluation}:
        raise ValueError("Development/evaluation ID overlap")
    split_root = ROOT / "datasets/nq_final_study/nq"
    files = {}
    for name, rows in (("dev", development), ("test", evaluation)):
        path = split_root / f"{name}.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        if path.exists() and path.read_text(encoding="utf-8") != payload:
            raise ValueError("Refusing to replace a different frozen split")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        files[name] = {"file": str(path.relative_to(ROOT)), "count": len(rows), "sha256": digest(path),
                       "ids": [r["id"] for r in rows]}
    manifest = {"source_repo": provenance["source_repo"], "source_revision": provenance["source_revision"],
                "source_sha256": digest(source), "seed": seed, "excluded_question_count": len(excluded),
                "eligible_count": len(eligible), "splits": files,
                "note": "Disjoint from every previous 17/500-question experiment. Test answers are used only after policies freeze."}
    save(output / "split_manifest.json", manifest)
    return development, evaluation, manifest


def lexical_hit(documents, answers, normalize):
    # Token-boundary substring proxy. Empty normalized aliases never match.
    needles = [" " + normalize(a) + " " for a in answers if normalize(a)]
    return any(needle in " " + normalize(doc["contents"]) + " " for doc in documents for needle in needles)


def calibrate(documents, scores, records, normalize):
    """Minimize selected docs, subject to no lost dev lexical hits vs fixed-5."""
    import numpy as np
    from flashrag.retriever.adaptive_topk import policy_from_config
    baseline = [lexical_hit(docs[:5], row["golden_answers"], normalize) for docs, row in zip(documents, records)]
    score_grid = sorted(set(float(x) for x in np.quantile(np.asarray(scores)[:, :5], np.linspace(.05, .95, 19))))
    configurations = []
    for minimum in (2, 3, 4):
        for gap in (.003, .005, .008, .01, .015, .02, .03, .05):
            configurations.append(dict(type="score_gap", min_k=minimum, max_k=5, fallback_k=5,
                                       gap_threshold=gap, higher_is_better=True))
        for threshold in score_grid:
            configurations.append(dict(type="similarity_threshold", min_k=minimum, max_k=5, fallback_k=5,
                                       score_threshold=threshold, higher_is_better=True))
    rows = []
    for config in configurations:
        policy = policy_from_config(config)
        counts = [policy.select(s)["selected_k"] for s in scores]
        hits = [lexical_hit(docs[:k], r["golden_answers"], normalize)
                for docs, k, r in zip(documents, counts, records)]
        lost = sum(a and not b for a, b in zip(baseline, hits))
        rows.append({"config": config, "mean_k": statistics.mean(counts), "k_distribution": dict(Counter(counts)),
                     "lexical_hits": sum(hits), "lost_fixed5_hits": lost, "feasible": lost == 0})
    chosen = {}
    for kind in ("score_gap", "similarity_threshold"):
        feasible = [r for r in rows if r["config"]["type"] == kind and r["feasible"]]
        if not feasible:
            # A fixed-5-equivalent safety fallback makes failure explicit.
            config = (dict(type=kind, min_k=5, max_k=5, fallback_k=5, gap_threshold=.05) if kind == "score_gap"
                      else dict(type=kind, min_k=5, max_k=5, fallback_k=5, score_threshold=score_grid[0]))
            chosen[kind] = {"config": config, "fallback_reason": "no feasible adaptive candidate"}
        else:
            chosen[kind] = min(feasible, key=lambda r: (r["mean_k"], -r["config"]["min_k"], json.dumps(r["config"], sort_keys=True)))
    return {"criterion": "No individual development lexical hit lost versus fixed-5; minimize mean k; ties prefer larger min_k then canonical configuration.",
            "development_count": len(records), "fixed5_lexical_hits": sum(baseline), "candidate_count": len(rows),
            "candidates": rows, "chosen": chosen,
            "limitation": "Lexical coverage is not semantic sufficiency or answer quality. No LLM outputs or test answers are used for calibration."}


def build_pipeline(args):
    import torch
    from flashrag.config import Config
    from flashrag.pipeline import SequentialPipeline
    from flashrag.prompt import PromptTemplate
    torch.set_num_threads(8)
    config = Config(config_dict={
        "dataset_name": "nq", "framework": "hf", "disable_save": True,
        "model2path": {"e5": str(args.retriever_path), "qwen2.5-0.5b-instruct": str(args.model_path)},
        "generator_model": "qwen2.5-0.5b-instruct", "retrieval_method": "e5",
        "index_path": str(args.index_path), "corpus_path": str(args.corpus_path),
        "retrieval_topk": 10, "retrieval_use_fp16": False,
        "generator_batch_size": 8, "generation_params": {"do_sample": False, "max_tokens": 32},
        "metrics": ["em", "f1", "acc"], "save_metric_score": False, "save_intermediate_data": False,
    })
    template = PromptTemplate(config,
        system_prompt="Answer the question based on the given document. \
                    Only give me the answer and do not output any other words. \
                    \nThe following are given documents.\n\n{reference}",
        user_prompt="Question: {question}\nAnswer:")
    pipeline = SequentialPipeline(config, prompt_template=template)
    pipeline.generator.generate = partial(pipeline.generator.generate, sort_by_length=False,
                                         reuse_tokenization=False, decode_generated_only=False,
                                         clear_cuda_cache=True, collect_scores=False)
    return pipeline, config


def audit_resources(pipeline, args, records, normalize):
    import numpy as np
    encoder, index, corpus = pipeline.retriever.encoder, pipeline.retriever.index, pipeline.retriever.corpus
    if index.ntotal != len(corpus):
        raise ValueError("Index/corpus count mismatch")
    positions = sorted(random.Random(598).sample(range(len(corpus)), 16))
    stored = np.stack([index.reconstruct(i) for i in positions])
    old_length = encoder.max_length
    audits = []
    try:
        for length in (128, 180, 512):
            encoder.max_length = length
            encoded = encoder.encode([corpus[i]["contents"] for i in positions], is_query=False)
            cos = np.sum(encoded * stored, axis=1) / (np.linalg.norm(encoded, axis=1) * np.linalg.norm(stored, axis=1))
            audits.append({"passage_max_length": length, "cosines": cos.tolist(), "minimum_cosine": float(cos.min()), "mean_cosine": float(cos.mean())})
    finally:
        encoder.max_length = old_length
    all_text = "\n".join(" " + normalize(doc["contents"]) + " " for doc in corpus)
    coverage = [any(" " + normalize(a) + " " in all_text for a in row["golden_answers"] if normalize(a)) for row in records]
    best = max(audits, key=lambda r: r["mean_cosine"])
    return {"corpus_documents": len(corpus), "index_vectors": index.ntotal, "dimension": index.d,
            "corpus_sha256": digest(args.corpus_path), "index_sha256": digest(args.index_path),
            "index_metric": int(index.metric_type), "sample_positions": positions, "mapping_checks": audits,
            "sample_mapping_consistent": best["minimum_cosine"] > .99,
            "development_corpus_lexical_hits": sum(coverage), "development_questions": len(records),
            "note": "Sixteen re-encoded passages check sampled vector/document alignment only. Whole-corpus token-boundary answer-string presence is a proxy, not semantic evidence; final test golds are not inspected here."}


def evaluate(pipeline, config, records, policies, output, repeats, phase):
    import torch
    from flashrag.dataset import Dataset
    from flashrag.evaluator.utils import normalize_answer
    from flashrag.retriever.adaptive_topk import policy_from_config
    settings = {f"fixed_{k}": {"fixed_k": k} for k in (1, 3, 5, 10)}
    settings.update({name: {"policy": value["config"]} for name, value in policies.items()})
    path = output / f"{phase}_measurements.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    for repeat in range(repeats):
        names = list(settings)
        names = names[repeat % len(names):] + names[:repeat % len(names)]
        if repeat % 2:
            names.reverse()
        for name in names:
            if any(r["repeat"] == repeat + 1 and r["setting"] == name for r in rows):
                continue
            setting = settings[name]
            pipeline.retrieval_policy = policy_from_config(setting["policy"]) if "policy" in setting else None
            pipeline.retriever.topk = setting.get("fixed_k", 5)
            pipeline.run(Dataset(config=config, data=records[:8]), do_eval=False)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            data = Dataset(config=config, data=records)
            cpu_start, start = time.process_time(), time.perf_counter()
            result = pipeline.run(data, do_eval=False)
            torch.cuda.synchronize()
            wall, cpu = time.perf_counter() - start, time.process_time() - cpu_start
            peak = torch.cuda.max_memory_allocated() / 2**20
            reserved = torch.cuda.max_memory_reserved() / 2**20
            profile = dict(pipeline.last_profile)
            generation_profile = dict(pipeline.generator.last_generation_profile)
            metrics = pipeline.evaluator.evaluate(result)
            samples = []
            for item in result:
                docs = item.retrieval_result
                ids = pipeline.generator.tokenizer.encode(item.prompt, truncation=True, max_length=pipeline.generator.max_input_len)
                samples.append({"id": item.id, "prediction": item.pred, "selected_k": len(docs),
                                "input_tokens": len(ids), "retrieved_ids": [d["id"] for d in docs],
                                "answer_string_hit": lexical_hit(docs, item.golden_answers, normalize_answer),
                                "metric_scores": item.output.get("metric_score", {}),
                                "decision": item.output.get("adaptive_decision")})
            row = {"phase": phase, "setting": name, "repeat": repeat + 1, "configuration": setting,
                   "wall_seconds": wall, "process_cpu_seconds": cpu, "peak_allocated_mib": peak,
                   "peak_reserved_mib": reserved, "stage_seconds": profile, "generation_profile": generation_profile,
                   "metrics": metrics, "mean_k": statistics.mean(s["selected_k"] for s in samples),
                   "mean_input_tokens": statistics.mean(s["input_tokens"] for s in samples),
                   "retrieval_answer_hit": statistics.mean(s["answer_string_hit"] for s in samples),
                   "k_distribution": dict(sorted(Counter(s["selected_k"] for s in samples).items())), "samples": samples}
            rows.append(row)
            save(path, rows)
            print("[FINAL] " + json.dumps({k: v for k, v in row.items() if k not in ("samples", "generation_profile", "stage_seconds")}), flush=True)
    return rows


def summarize(rows):
    import numpy as np
    names = list(dict.fromkeys(r["setting"] for r in rows))
    ref = next(r for r in rows if r["setting"] == "fixed_5")
    reference_time = statistics.median(r["wall_seconds"] for r in rows if r["setting"] == "fixed_5")
    output = {}
    for name in names:
        group = [r for r in rows if r["setting"] == name]
        first = group[0]
        assert [s["id"] for s in first["samples"]] == [s["id"] for s in ref["samples"]]
        paired = {}
        for metric in ("em", "f1", "acc"):
            delta = np.array([a["metric_scores"][metric] - b["metric_scores"][metric]
                              for a, b in zip(first["samples"], ref["samples"])])
            rng = np.random.default_rng(598)
            boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
            paired[metric] = {"difference": float(delta.mean()), "better": int((delta > 0).sum()),
                              "worse": int((delta < 0).sum()), "bootstrap_95_ci": np.quantile(boot, [.025, .975]).tolist()}
        median = statistics.median(r["wall_seconds"] for r in group)
        output[name] = {"times": [r["wall_seconds"] for r in group], "median_seconds": median,
                        "throughput_qps": len(first["samples"]) / median,
                        "median_process_cpu_seconds": statistics.median(r["process_cpu_seconds"] for r in group),
                        "time_reduction_vs_fixed5_percent": 100 * (1 - median / reference_time),
                        "peak_allocated_mib": max(r["peak_allocated_mib"] for r in group),
                        "peak_reserved_mib": max(r["peak_reserved_mib"] for r in group),
                        "metrics": first["metrics"], "mean_k": first["mean_k"],
                        "mean_input_tokens": first["mean_input_tokens"], "retrieval_answer_hit": first["retrieval_answer_hit"],
                        "k_distribution": first["k_distribution"], "paired_vs_fixed5": paired,
                        "prediction_disagreements_across_repeats": sum(a["prediction"] != b["prediction"] for r in group[1:] for a, b in zip(first["samples"], r["samples"])),
                        "median_stages": {key: statistics.median(r["stage_seconds"][key] for r in group)
                                          for key in ("retrieval_s", "selection_s", "prompt_s", "generation_s")}}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/experiments/final_topk")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--retriever-path", type=Path, default=Path.home() / ".cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd")
    parser.add_argument("--corpus-path", type=Path, default=ROOT / "examples/quick_start/indexes/general_knowledge.jsonl")
    parser.add_argument("--index-path", type=Path, default=ROOT / "examples/quick_start/indexes/e5_Flat.index")
    parser.add_argument("--stage", choices=("develop", "evaluate", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("Final timing requires at least three repeats")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev, test, manifest = prepare_splits(args.output_dir)
    pipeline, config = build_pipeline(args)
    from flashrag.evaluator.utils import normalize_answer
    import torch
    frozen_path = args.output_dir / "frozen_policies.json"
    if args.stage in ("develop", "all") and not frozen_path.exists():
        audit = audit_resources(pipeline, args, dev, normalize_answer)
        save(args.output_dir / "resource_audit.json", audit)
        if not audit["sample_mapping_consistent"]:
            raise ValueError("Index/document audit failed; inspect resource_audit.json before evaluating")
        docs, scores = pipeline.retriever.batch_search([r["question"] for r in dev], num=10, return_score=True)
        calibration = calibrate(docs, scores, dev, normalize_answer)
        save(args.output_dir / "calibration.json", calibration)
        frozen = {"policies": calibration["chosen"], "manifest_sha256": digest(args.output_dir / "split_manifest.json"),
                  "calibration_sha256": digest(args.output_dir / "calibration.json"),
                  "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "development_selection": calibration["criterion"],
                  "evaluation_rule": "All frozen methods reported; no final-test winner is selected or retuned. Match fixed-5 observed EM and F1 for a quality-preserving claim; report all paired differences regardless."}
        save(frozen_path, frozen)
        print("[FROZEN] " + json.dumps(frozen), flush=True)
    if not frozen_path.exists():
        raise ValueError("Run development before final evaluation")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if digest(args.output_dir / "split_manifest.json") != frozen["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    metadata = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name(0), "generator_dtype": str(pipeline.generator.model.dtype),
                "generator_batch_size": 8, "sort_by_length": False, "max_new_tokens": 32, "cpu_threads": torch.get_num_threads(),
                "policy_sha256": digest(frozen_path), "generator_source_sha256": digest(ROOT / "flashrag/generator/generator.py"),
                "policy_source_sha256": digest(ROOT / "flashrag/retriever/adaptive_topk.py"),
                "runner_sha256": digest(__file__), "corpus_sha256": digest(args.corpus_path), "index_sha256": digest(args.index_path)}
    save(args.output_dir / "metadata.json", metadata)
    if args.stage in ("develop", "all"):
        dev_rows = evaluate(pipeline, config, dev, frozen["policies"], args.output_dir, 1, "development")
        save(args.output_dir / "development_summary.json", summarize(dev_rows))
    if args.stage in ("evaluate", "all"):
        rows = evaluate(pipeline, config, test, frozen["policies"], args.output_dir, args.repeats, "evaluation")
        save(args.output_dir / "evaluation_summary.json", summarize(rows))
    print("[COMPLETE] " + str(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
