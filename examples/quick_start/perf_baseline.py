import argparse
import copy
import csv
import datetime
import json
import statistics
import time
from pathlib import Path

import torch

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate


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


parser = argparse.ArgumentParser(description="Compare fixed retrieval top-k values.")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--retriever_path", type=str, required=True)
parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
parser.add_argument("--sample-num", type=int, default=5)
parser.add_argument("--warmups", type=int, default=1)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()

if any(k <= 0 for k in args.k_values):
    parser.error("all k values must be positive")
if len(set(args.k_values)) != len(args.k_values):
    parser.error("k values must be unique")
if args.sample_num <= 0 or args.warmups < 0 or args.repeats <= 0:
    parser.error("sample-num and repeats must be positive; warmups cannot be negative")

output_dir = args.output_dir or Path(__file__).resolve().parents[2] / "docs"
output_dir.mkdir(parents=True, exist_ok=True)
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
summary_path = output_dir / f"adaptive_topk_summary_{run_id}.csv"

config_dict = {
    "data_dir": "dataset/",
    "index_path": "indexes/e5_Flat.index",
    "corpus_path": "indexes/general_knowledge.jsonl",
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
    "generator_batch_size": 1,
    "generation_params": {"do_sample": False, "max_tokens": 32},
}

config = Config(config_dict=config_dict)
print(f"[config] script={Path(__file__).resolve()}")
print(f"[config] k_values={args.k_values}")
print(f"[config] sample_num={args.sample_num}, warmups={args.warmups}, repeats={args.repeats}")
print(f"[config] results_dir={output_dir}")

all_split = get_dataset(config)
test_data = all_split["test"]
prompt_templete = PromptTemplate(
    config,
    system_prompt="Answer the question based on the given document. \
                    Only give me the answer and do not output any other words. \
                    \nThe following are given documents.\n\n{reference}",
    user_prompt="Question: {question}\nAnswer:",
)


pipeline = SequentialPipeline(config, prompt_template=prompt_templete)
pipeline.evaluator.save_data_flag = False
pipeline.evaluator.save_metric_flag = False

summary_fields = [
    "k",
    "sample_count",
    "repeat_count",
    "mean_docs_per_query",
    "avg_input_tokens",
    "em",
    "f1",
    "acc",
    "median_retrieval_s",
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

    for k in args.k_values:
        pipeline.retriever.topk = k
        print(f"\n[experiment] k={k}")

        for warmup in range(1, args.warmups + 1):
            print(f"[warmup {warmup}/{args.warmups}] excluded from results")
            pipeline.run(copy.deepcopy(test_data), do_eval=False)
            synchronize(getattr(pipeline.generator.model, "device", pipeline.device))

        repeat_times = []
        repeat_metrics = []
        samples_by_repeat = []
        stage_times = {"retrieval_s": [], "prompt_s": [], "generation_s": []}
        final_dataset = None

        for repeat in range(1, args.repeats + 1):
            print(f"[repeat {repeat}/{args.repeats}, k={k}]")
            run_dataset = copy.deepcopy(test_data)
            synchronize(getattr(pipeline.generator.model, "device", pipeline.device))
            total_start = time.perf_counter()
            final_dataset = pipeline.run(run_dataset, do_eval=False)
            synchronize(getattr(pipeline.generator.model, "device", pipeline.device))
            total_s = time.perf_counter() - total_start
            repeat_times.append(total_s)

            profile = pipeline.last_profile
            for stage_name in stage_times:
                stage_value = profile.get(stage_name)
                if stage_value is not None:
                    stage_times[stage_name].append(stage_value)

            metrics = pipeline.evaluator.evaluate(final_dataset)
            repeat_metrics.append(metrics)
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
                    }
                    for item in final_dataset
                ]
            )
            print(f"[profile] k={k}, repeat={repeat}, end_to_end_s={total_s:.4f}")

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

        result_path = output_dir / f"adaptive_topk_k{k}_{run_id}.json"
        result_payload = {
            "run_id": run_id,
            "k": k,
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
        }
        with result_path.open("w", encoding="utf-8") as result_file:
            json.dump(result_payload, result_file, indent=2, ensure_ascii=False, default=str)

        summary_row = {
            "k": k,
            "sample_count": len(final_dataset),
            "repeat_count": args.repeats,
            "mean_docs_per_query": mean_docs,
            "avg_input_tokens": averaged_metrics.get("avg_input_tokens"),
            "em": averaged_metrics.get("em"),
            "f1": averaged_metrics.get("f1"),
            "acc": averaged_metrics.get("acc"),
            "median_retrieval_s": statistics.median(stage_times["retrieval_s"]) if stage_times["retrieval_s"] else None,
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
        print(f"[summary] k={k}, metrics={averaged_metrics}, median_s={median_s:.4f}, qps={qps:.3f}")

print(f"\n[done] summary={summary_path}")
