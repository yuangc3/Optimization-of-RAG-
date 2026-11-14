import argparse
import copy
import csv
import datetime
import json
import statistics
import time
import platform
from functools import partial
from pathlib import Path
from dataclasses import asdict
from collections import Counter

def clone_dataset(dataset, sample_num=None):
    # Copy plain records, avoiding Dataset/Item dynamic attribute lookup during deepcopy.
    items = dataset.data if sample_num is None else dataset.data[:sample_num]
    records = [dict(item.data, output=item.output) for item in items]
    return type(dataset)(config=dataset.config, data=copy.deepcopy(records))


def synchronize(device):
    if device is None:
        return
    device = str(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif (
        device.startswith("mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.synchronize()


parser = argparse.ArgumentParser(description="Compare fixed and score-adaptive retrieval top-k.")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--retriever_path", type=str, required=True)
parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
parser.add_argument("--sample-num", type=int, default=5)
parser.add_argument("--warmups", type=int, default=1)
parser.add_argument("--warmup-sample-num", type=int,
                    help="Use a smaller prefix for warmups only; measured repeats still use sample-num.")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--output-dir", type=Path)
repo_root = Path(__file__).resolve().parents[2]
parser.add_argument("--data-dir", type=Path, default=repo_root / "examples/quick_start/dataset")
parser.add_argument("--index-path", type=Path, default=repo_root / "examples/quick_start/indexes/e5_Flat.index")
parser.add_argument("--corpus-path", type=Path, default=repo_root / "examples/quick_start/indexes/general_knowledge.jsonl")
parser.add_argument("--collect-scores", action="store_true",
                    help="Use the historical token-score path for comparison with text-only generation.")
parser.add_argument("--generator-batch-size", type=int, default=1)
parser.add_argument("--sort-by-length", action="store_true")
parser.add_argument("--reuse-tokenization", action="store_true")
parser.add_argument("--decode-generated-only", action="store_true")
parser.add_argument("--cpu-threads", type=int, help="Set PyTorch intra-op threads before model loading.")
parser.add_argument("--retain-cuda-cache", action="store_true",
                    help="Avoid the historical empty_cache call between generation batches.")
parser.add_argument("--adaptive", action="store_true", help="Also benchmark per-query score-gap selection.")
parser.add_argument("--adaptive-min-k", type=int, default=1)
parser.add_argument("--adaptive-max-k", type=int, default=10)
parser.add_argument("--adaptive-fallback-k", type=int, default=5)
parser.add_argument("--adaptive-gap", type=float, default=0.05)
parser.add_argument("--adaptive-score-order", choices=["higher", "lower"], default="higher",
                    help="Use higher for inner-product similarity, lower for L2 distance.")
args = parser.parse_args()

if any(k <= 0 for k in args.k_values):
    parser.error("all k values must be positive")
if len(set(args.k_values)) != len(args.k_values):
    parser.error("k values must be unique")
if args.sample_num <= 0 or args.warmups < 0 or args.repeats <= 0:
    parser.error("sample-num and repeats must be positive; warmups cannot be negative")
if args.warmup_sample_num is not None and args.warmup_sample_num <= 0:
    parser.error("warmup-sample-num must be positive")
if args.generator_batch_size <= 0:
    parser.error("generator-batch-size must be positive")
if args.cpu_threads is not None and args.cpu_threads <= 0:
    parser.error("cpu-threads must be positive")
for resource in (args.data_dir, args.index_path, args.corpus_path):
    if not resource.exists():
        parser.error(f"required local resource does not exist: {resource}")

import torch
if args.cpu_threads is not None:
    torch.set_num_threads(args.cpu_threads)
import transformers
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from flashrag.retriever.adaptive_topk import ScoreGapPolicy

adaptive_policy = None
if args.adaptive:
    try:
        adaptive_policy = ScoreGapPolicy(
            min_k=args.adaptive_min_k, max_k=args.adaptive_max_k,
            fallback_k=args.adaptive_fallback_k, gap_threshold=args.adaptive_gap,
            higher_is_better=args.adaptive_score_order == "higher",
        )
    except ValueError as error:
        parser.error(str(error))

output_dir = args.output_dir or Path(__file__).resolve().parents[2] / "docs"
output_dir.mkdir(parents=True, exist_ok=True)
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
summary_path = output_dir / f"adaptive_topk_summary_{run_id}.csv"

config_dict = {
    "data_dir": str(args.data_dir.resolve()),
    "index_path": str(args.index_path.resolve()),
    "corpus_path": str(args.corpus_path.resolve()),
    "model2path": {"e5": args.retriever_path, "qwen2.5-0.5b-instruct": args.model_path},
    "generator_model": "qwen2.5-0.5b-instruct",
    "retrieval_method": "e5",
    "metrics": ["em", "f1", "acc", "input_tokens"],
    "retrieval_topk": max(args.k_values),
    "save_intermediate_data": False,
    "save_metric_score": False,
    "framework": "hf",
    "test_sample_num": args.sample_num,
    "random_sample": False,
    "retrieval_use_fp16": False,
    "generator_batch_size": args.generator_batch_size,
    "generation_params": {"do_sample": False, "max_tokens": 32},
}

config = Config(config_dict=config_dict)
print(f"[config] script={Path(__file__).resolve()}")
print(f"[config] k_values={args.k_values}")
print(f"[config] sample_num={args.sample_num}, warmups={args.warmups}, repeats={args.repeats}")
print(f"[config] results_dir={output_dir}")

all_split = get_dataset(config)
test_data = all_split["test"]
if test_data is None or not len(test_data):
    parser.error("test split is missing or empty")
if len(test_data) < args.sample_num:
    print(f"[warning] requested {args.sample_num} samples, only {len(test_data)} available")
prompt_templete = PromptTemplate(
    config,
    system_prompt="Answer the question based on the given document. \
                    Only give me the answer and do not output any other words. \
                    \nThe following are given documents.\n\n{reference}",
    user_prompt="Question: {question}\nAnswer:",
)


pipeline = SequentialPipeline(config, prompt_template=prompt_templete)
if adaptive_policy is not None:
    import faiss
    metric = getattr(getattr(pipeline.retriever, "index", None), "metric_type", None)
    expected_order = {faiss.METRIC_INNER_PRODUCT: "higher", faiss.METRIC_L2: "lower"}.get(metric)
    if expected_order is not None and args.adaptive_score_order != expected_order:
        parser.error(f"index metric requires --adaptive-score-order {expected_order}")
pipeline.generator.generate = partial(pipeline.generator.generate, collect_scores=args.collect_scores,
                                      clear_cuda_cache=not args.retain_cuda_cache,
                                      sort_by_length=args.sort_by_length,
                                      reuse_tokenization=args.reuse_tokenization,
                                      decode_generated_only=args.decode_generated_only)
pipeline.evaluator.save_data_flag = False
pipeline.evaluator.save_metric_flag = False
run_metadata = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "generator_device": str(pipeline.generator.model.device),
    "model_path": args.model_path,
    "retriever_path": args.retriever_path,
    "data_dir": str(args.data_dir.resolve()),
    "index_path": str(args.index_path.resolve()),
    "corpus_path": str(args.corpus_path.resolve()),
    "collect_scores": args.collect_scores,
    "generator_batch_size": args.generator_batch_size,
    "sort_by_length": args.sort_by_length,
    "reuse_tokenization": args.reuse_tokenization,
    "decode_generated_only": args.decode_generated_only,
    "torch_intraop_threads": torch.get_num_threads(),
    "torch_interop_threads": torch.get_num_interop_threads(),
    "clear_cuda_cache": not args.retain_cuda_cache,
    "generation_params": config_dict["generation_params"],
    "seed": config["seed"],
    "warmup_sample_count": min(args.warmup_sample_num or len(test_data), len(test_data)),
    "input_tokens_note": "Evaluator tokenizer count; not necessarily generator tokenizer count",
    "adaptive_policy": asdict(adaptive_policy) if adaptive_policy is not None else None,
}

summary_fields = [
    "k",
    "setting",
    "collect_scores",
    "sample_count",
    "repeat_count",
    "mean_docs_per_query",
    "avg_input_tokens",
    "generator_avg_input_tokens",
    "selected_k_distribution",
    "em",
    "f1",
    "acc",
    "median_retrieval_s",
    "median_selection_s",
    "median_prompt_s",
    "median_generation_s",
    "median_end_to_end_s",
    "mean_end_to_end_s",
    "throughput_qps",
    "result_file",
]

with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
    summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
    summary_writer.writeheader()
    summary_file.flush()

    experiments = [(f"k{k}", k, None) for k in args.k_values]
    if adaptive_policy is not None:
        experiments.append(("adaptive_gap", None, adaptive_policy))
    for setting, k, policy in experiments:
        pipeline.retrieval_policy = policy
        pipeline.retriever.topk = k if policy is None else policy.max_k
        print(f"\n[experiment] setting={setting}")

        for warmup in range(1, args.warmups + 1):
            print(f"[warmup {warmup}/{args.warmups}] excluded from results")
            pipeline.run(clone_dataset(test_data, args.warmup_sample_num), do_eval=False)
            synchronize(getattr(pipeline.generator.model, "device", pipeline.device))

        repeat_times = []
        repeat_metrics = []
        samples_by_repeat = []
        generation_profiles = []
        stage_times = {"retrieval_s": [], "selection_s": [], "prompt_s": [], "generation_s": []}
        final_dataset = None

        for repeat in range(1, args.repeats + 1):
            print(f"[repeat {repeat}/{args.repeats}, setting={setting}]")
            run_dataset = clone_dataset(test_data)
            synchronize(getattr(pipeline.generator.model, "device", pipeline.device))
            total_start = time.perf_counter()
            final_dataset = pipeline.run(run_dataset, do_eval=False)
            synchronize(getattr(pipeline.generator.model, "device", pipeline.device))
            total_s = time.perf_counter() - total_start
            repeat_times.append(total_s)
            generation_profiles.append(copy.deepcopy(pipeline.generator.last_generation_profile))

            profile = pipeline.last_profile
            for stage_name in stage_times:
                stage_value = profile.get(stage_name)
                if stage_value is not None:
                    stage_times[stage_name].append(stage_value)

            metrics = pipeline.evaluator.evaluate(final_dataset)
            repeat_metrics.append(metrics)
            # Match the generator's truncation/tokenizer, outside inference timing.
            generator_input_tokens = [
                len(pipeline.generator.tokenizer.encode(
                    item.prompt, truncation=True, max_length=pipeline.generator.max_input_len
                )) for item in final_dataset
            ]
            samples_by_repeat.append(
                [
                    {
                        "id": item.id,
                        "question": item.question,
                        "golden_answers": item.golden_answers,
                        "prediction": item.pred,
                        "metric_scores": item.output.get("metric_score", {}),
                        "retrieved_documents": item.retrieval_result,
                        "input_tokens": item.output.get("metric_score", {}).get("input_tokens"),
                        "generator_input_tokens": token_count,
                        "selected_k": len(item.retrieval_result),
                        "candidate_scores": item.output.get("retrieval_scores"),
                        "adaptive_decision": item.output.get("adaptive_decision"),
                    }
                    for item, token_count in zip(final_dataset, generator_input_tokens)
                ]
            )
            print(f"[profile] setting={setting}, repeat={repeat}, end_to_end_s={total_s:.4f}")

        median_s = statistics.median(repeat_times)
        mean_s = statistics.mean(repeat_times)
        qps = len(test_data) / median_s if median_s else 0.0
        mean_docs = statistics.mean(
            len(documents) for documents in final_dataset.retrieval_result
        )
        averaged_metrics = {
            metric: statistics.mean(result[metric] for result in repeat_metrics if metric in result)
            for metric in ("em", "f1", "acc", "avg_input_tokens")
            if any(metric in result for result in repeat_metrics)
        }

        avg_generator_tokens = statistics.mean(
            sample["generator_input_tokens"] for repeat_samples in samples_by_repeat for sample in repeat_samples
        )
        k_distribution = dict(sorted(Counter(
            len(item.retrieval_result) for item in final_dataset
        ).items()))
        result_path = output_dir / f"adaptive_topk_{setting}_{run_id}.json"
        result_payload = {
            "metadata": run_metadata,
            "run_id": run_id,
            "k": k,
            "setting": setting,
            "policy": asdict(policy) if policy is not None else None,
            "generator_avg_input_tokens": avg_generator_tokens,
            "selected_k_distribution": k_distribution,
            "sample_count": len(final_dataset),
            "warmups": args.warmups,
            "repeats": args.repeats,
            "metrics_by_repeat": repeat_metrics,
            "mean_metrics": averaged_metrics,
            "mean_docs_per_query": mean_docs,
            "end_to_end_seconds_by_repeat": repeat_times,
            "stage_seconds_by_repeat": stage_times,
            "median_end_to_end_s": median_s,
            "mean_end_to_end_s": mean_s,
            "throughput_qps_at_median": qps,
            "samples_by_repeat": samples_by_repeat,
            "generation_profiles_by_repeat": generation_profiles,
        }
        with result_path.open("w", encoding="utf-8") as result_file:
            json.dump(result_payload, result_file, indent=2, ensure_ascii=False, default=str)

        summary_row = {
            "k": k,
            "setting": setting,
            "collect_scores": args.collect_scores,
            "sample_count": len(final_dataset),
            "repeat_count": args.repeats,
            "mean_docs_per_query": mean_docs,
            "avg_input_tokens": averaged_metrics.get("avg_input_tokens"),
            "generator_avg_input_tokens": avg_generator_tokens,
            "selected_k_distribution": json.dumps(k_distribution),
            "em": averaged_metrics.get("em"),
            "f1": averaged_metrics.get("f1"),
            "acc": averaged_metrics.get("acc"),
            "median_retrieval_s": statistics.median(stage_times["retrieval_s"]) if stage_times["retrieval_s"] else None,
            "median_selection_s": statistics.median(stage_times["selection_s"]) if stage_times["selection_s"] else None,
            "median_prompt_s": statistics.median(stage_times["prompt_s"]) if stage_times["prompt_s"] else None,
            "median_generation_s": statistics.median(stage_times["generation_s"]) if stage_times["generation_s"] else None,
            "median_end_to_end_s": median_s,
            "mean_end_to_end_s": mean_s,
            "throughput_qps": qps,
            "result_file": result_path.name,
        }
        summary_writer.writerow(summary_row)
        summary_file.flush()
        print(f"[saved] {result_path}")
        print(f"[summary] setting={setting}, docs={mean_docs:.2f}, metrics={averaged_metrics}, median_s={median_s:.4f}, qps={qps:.3f}")

print(f"\n[done] summary={summary_path}")
