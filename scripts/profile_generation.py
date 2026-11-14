"""Controlled generation profiling and allocator/score ablations on saved RAG prompts."""
import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
import gzip
import json
from pathlib import Path
import platform
import statistics
import shutil
import gc
import os
from collections import Counter
from bisect import bisect_right
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import transformers
from flashrag.config import Config
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_generator
from flashrag.dataset import Dataset
from flashrag.evaluator import Evaluator


@contextmanager
def diagnostics(generator, events, record_cuda_events=True):
    original_tokenizer = generator.tokenizer
    original_forward = generator.model.forward
    original_generate = generator.model.generate
    original_empty = torch.cuda.empty_cache

    class TokenizerProxy:
        def __getattr__(self, name):
            return getattr(original_tokenizer, name)

        def __call__(self, *args, **kwargs):
            with torch.profiler.record_function("rag.tokenization"):
                return original_tokenizer(*args, **kwargs)

        def decode(self, *args, **kwargs):
            with torch.profiler.record_function("rag.text_decode"):
                return original_tokenizer.decode(*args, **kwargs)

        def batch_decode(self, *args, **kwargs):
            with torch.profiler.record_function("rag.text_decode"):
                return original_tokenizer.batch_decode(*args, **kwargs)

        def pad(self, *args, **kwargs):
            with torch.profiler.record_function("rag.tokenization"):
                return original_tokenizer.pad(*args, **kwargs)

    @wraps(original_forward)
    def forward(*args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        phase = "prefill" if ids is not None and ids.shape[-1] > 1 else "decode"
        start = torch.cuda.Event(enable_timing=True) if record_cuda_events else None
        end = torch.cuda.Event(enable_timing=True) if record_cuda_events else None
        with torch.profiler.record_function("rag." + phase):
            if start is not None:
                start.record()
            output = original_forward(*args, **kwargs)
            if end is not None:
                end.record()
        if start is not None:
            events.append((phase, start, end))
        return output

    @wraps(original_generate)
    def generate(*args, **kwargs):
        with torch.profiler.record_function("rag.model_generate"):
            return original_generate(*args, **kwargs)

    def empty_cache():
        with torch.profiler.record_function("rag.empty_cache"):
            return original_empty()

    generator.tokenizer = TokenizerProxy()
    generator.model.forward = forward
    generator.model.generate = generate
    torch.cuda.empty_cache = empty_cache
    try:
        yield
    finally:
        generator.tokenizer = original_tokenizer
        generator.model.forward = original_forward
        generator.model.generate = original_generate
        torch.cuda.empty_cache = original_empty


def interval_union_us(intervals):
    """Time covered by at least one interval, without double-counting overlap."""
    total = 0.0
    right = None
    for start, end in sorted(intervals):
        if right is None or start > right:
            total += end - start
            right = end
        elif end > right:
            total += end - right
            right = end
    return total


def analyze_trace(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            trace = json.load(handle)
    else:
        trace = json.loads(path.read_text(encoding="utf-8"))
    events = trace["traceEvents"]
    window = [e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation"
              and e.get("name") == "rag.profile_window"]
    if len(window) != 1:
        raise ValueError("Expected one host profiling window")
    left, duration = window[0]["ts"], window[0]["dur"]
    right = left + duration
    inside = [e for e in events if e.get("ph") == "X" and e.get("dur", 0) > 0
              and e["ts"] < right and e["ts"] + e["dur"] > left]

    def intervals(selected):
        return [(max(left, e["ts"]), min(right, e["ts"] + e["dur"])) for e in selected]

    kernels = [e for e in inside if e.get("cat") == "kernel"]
    copies = [e for e in inside if e.get("cat") == "gpu_memcpy"]
    memsets = [e for e in inside if e.get("cat") == "gpu_memset"]
    if not kernels:
        raise RuntimeError("No CUDA kernels captured; do not infer GPU time from host annotations")
    runtime = [e for e in inside if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    syncs = [e for e in runtime if "Synchronize" in e["name"]]
    launches = [e for e in runtime if "LaunchKernel" in e["name"] or "LaunchKernel" in e["name"].replace("launch", "Launch")]
    names = Counter(e["name"] for e in runtime)
    hot = {}
    for event in kernels:
        row = hot.setdefault(event["name"], {"calls": 0, "sum_us": 0.0})
        row["calls"] += 1
        row["sum_us"] += event["dur"]
    scopes = [{"name": e["name"], "host_interval_ms": e["dur"] / 1000}
              for e in inside if e.get("cat") == "user_annotation" and e["name"].startswith("rag.")]
    scope_summary = {}
    for scope in scopes:
        row = scope_summary.setdefault(scope["name"], {"calls": 0, "sum_host_interval_ms": 0.0})
        row["calls"] += 1
        row["sum_host_interval_ms"] += scope["host_interval_ms"]
    copy_directions = {}
    for name in {e["name"] for e in copies}:
        selected = [e for e in copies if e["name"] == name]
        copy_directions[name] = {"count": len(selected),
                                 "bytes": sum(e.get("args", {}).get("bytes", 0) for e in selected),
                                 "union_ms": interval_union_us(intervals(selected)) / 1000}
    forward_scopes = sorted(
        (e["ts"], e["ts"] + e["dur"], e["name"]) for e in inside
        if e.get("cat") == "user_annotation" and e["name"] in ("rag.prefill", "rag.decode"))
    forward_starts = [scope[0] for scope in forward_scopes]
    launch_by_correlation = {e.get("args", {}).get("correlation"): e for e in launches
                             if "correlation" in e.get("args", {})}
    by_phase = {}
    for kernel in kernels:
        launch = launch_by_correlation.get(kernel.get("args", {}).get("correlation"))
        phase = "outside_forward_or_unmatched"
        if launch is not None:
            index = bisect_right(forward_starts, launch["ts"]) - 1
            if index >= 0 and launch["ts"] <= forward_scopes[index][1]:
                phase = forward_scopes[index][2]
        row = by_phase.setdefault(phase, {"kernel_count": 0, "kernel_sum_ms": 0.0})
        row["kernel_count"] += 1
        row["kernel_sum_ms"] += kernel["dur"] / 1000
    return {
        "host_window_ms": duration / 1000, "kernel_count": len(kernels),
        "kernel_sum_ms": sum(e["dur"] for e in kernels) / 1000,
        "kernel_union_ms": interval_union_us(intervals(kernels)) / 1000,
        "process_kernel_coverage_percent": 100 * interval_union_us(intervals(kernels)) / duration,
        "kernel_copy_memset_union_ms": interval_union_us(intervals(kernels + copies + memsets)) / 1000,
        "copy_count": len(copies), "copy_bytes": sum(e.get("args", {}).get("bytes", 0) for e in copies),
        "copy_union_ms": interval_union_us(intervals(copies)) / 1000,
        "copies_by_direction": dict(Counter(e["name"] for e in copies)),
        "copy_direction_details": copy_directions,
        "sync_api_count": len(syncs), "sync_api_host_union_ms": interval_union_us(intervals(syncs)) / 1000,
        "launch_api_count": len(launches), "launch_api_host_union_ms": interval_union_us(intervals(launches)) / 1000,
        "runtime_api_counts": dict(names), "host_scopes": scope_summary,
        "top_kernels": [{"name": name, **values} for name, values in sorted(hot.items(), key=lambda item: item[1]["sum_us"], reverse=True)[:12]],
        "kernels_by_forward_phase": by_phase,
        "note": "Kernel coverage is this process's traced activity in an instrumented host window, not global GPU utilization or SM occupancy. Host API intervals include waits, not just CPU execution.",
    }


def cpu_gpu_study(generator, prompts, samples, expected, args):
    rows = []
    for repeat in range(args.repeats):
        batches = [1, 8] if repeat % 2 == 0 else [8, 1]
        for batch in batches:
            torch.cuda.empty_cache()
            generator.generate(prompts[:8], batch_size=batch, clear_cuda_cache=True)
            torch.cuda.synchronize()
            wall_start, cpu_start = time.perf_counter(), time.process_time()
            output = generator.generate(prompts, batch_size=batch, clear_cuda_cache=True)
            torch.cuda.synchronize()
            cpu_s, wall_s = time.process_time() - cpu_start, time.perf_counter() - wall_start
            rows.append({"repeat": repeat + 1, "batch_size": batch, "wall_seconds": wall_s,
                         "process_cpu_seconds": cpu_s, "cpu_core_equivalents": cpu_s / wall_s,
                         "text_mismatches_against_saved": sum(a != b for a, b in zip(output, expected))})
            if len(output) != len(prompts):
                raise RuntimeError("Wrong output count")
            (args.output_dir / "cpu_gpu_measurements.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            print("[cpu-gpu] " + json.dumps(rows[-1]), flush=True)

    lengths = [len(generator.tokenizer.encode(p)) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    indexes = [order[round(i * (len(order)-1)/7)] for i in range(8)]
    selected = [prompts[i] for i in indexes]
    traces = []
    for batch in [1, 8]:
        # Exactly eight output tokens per question isolate decode-work length.
        options = {"batch_size": batch, "clear_cuda_cache": True, "min_new_tokens": 8, "max_new_tokens": 8}
        generator.generate(selected, **options)
        torch.cuda.synchronize()
        with diagnostics(generator, [], record_cuda_events=False):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False, profile_memory=False, with_stack=False) as prof:
                cpu_start = time.process_time()
                with torch.profiler.record_function("rag.profile_window"):
                    generator.generate(selected, **options)
                    torch.cuda.synchronize()
                diagnostic_cpu_s = time.process_time() - cpu_start
        path = args.output_dir / f"batch{batch}_fixed8_trace.json"
        prof.export_chrome_trace(str(path))
        analysis = analyze_trace(path)
        analysis.update({"batch_size": batch, "diagnostic_process_cpu_seconds": diagnostic_cpu_s,
                         "question_ids": [samples[i]["id"] for i in indexes],
                         "input_lengths": [lengths[i] for i in indexes],
                         "fixed_output_tokens_per_question": 8, "logical_output_tokens": 64})
        traces.append(analysis)
        with path.open("rb") as src, gzip.open(str(path) + ".gz", "wb", compresslevel=3) as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()
        (args.output_dir / "cpu_gpu_trace_analysis.json").write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")
        del prof
        gc.collect()
        print("[trace] batch", batch, "kernels", analysis["kernel_count"], "coverage", analysis["process_kernel_coverage_percent"], flush=True)
    summary = {str(batch): {"median_wall_seconds": statistics.median(r["wall_seconds"] for r in rows if r["batch_size"] == batch),
                            "median_process_cpu_seconds": statistics.median(r["process_cpu_seconds"] for r in rows if r["batch_size"] == batch),
                            "runs": [r for r in rows if r["batch_size"] == batch]} for batch in [1, 8]}
    (args.output_dir / "cpu_gpu_study.json").write_text(json.dumps({
        "sample_count": len(prompts), "repeats": args.repeats, "summary": summary, "diagnostics": traces,
        "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
        "dtype": str(generator.model.dtype), "logical_cpu_count": os.cpu_count(),
        "process_cpu_clock": vars(time.get_clock_info("process_time")),
        "note": "Unprofiled formal runs use normal greedy decoding up to 32 tokens. Separate profiled runs force exactly eight tokens on eight prompts. Process CPU seconds sum all process threads and exclude sleep; scope durations are not CPU execution time.",
    }, indent=2) + "\n", encoding="utf-8")
    print("[done] CPU/GPU study", flush=True)


def padding_diagnostics(generator, prompts, samples, destination, variants=None):
    selected = prompts[:32]
    results = []
    custom_variants = variants is not None
    if variants is None:
        variants = {"original": {"sort_by_length": False}, "sorted": {"sort_by_length": True}}
    for name, variant in variants.items():
        options = dict(batch_size=8, clear_cuda_cache=True, min_new_tokens=8, max_new_tokens=8)
        options.update(variant)
        generator.generate(selected, **options)
        torch.cuda.synchronize()
        with diagnostics(generator, [], record_cuda_events=False):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False, profile_memory=False, with_stack=False) as prof:
                with torch.profiler.record_function("rag.profile_window"):
                    generator.generate(selected, **options)
                    torch.cuda.synchronize()
        path = destination / f"{name}_fixed8_trace.json"
        prof.export_chrome_trace(str(path))
        analysis = analyze_trace(path)
        analysis.update({"variant": name, "options": options, "sort_by_length": options["sort_by_length"],
                         "question_ids": [s["id"] for s in samples[:32]],
                         "logical_output_tokens": len(selected) * 8,
                         "generation_profile": generator.last_generation_profile})
        results.append(analysis)
        with path.open("rb") as src, gzip.open(str(path) + ".gz", "wb", compresslevel=3) as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()
        del prof
        gc.collect()
    filename = "execution_trace_analysis.json" if custom_variants else "padding_trace_analysis.json"
    (destination / filename).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def compare_trials(rows, reference_name):
    """Summarize synchronized, unprofiled replay trials against a same-session reference."""
    references = {r["repeat"]: r for r in rows if r["variant"] == reference_name}
    reference_median = statistics.median(r["seconds"] for r in references.values())
    comparisons = {}
    for name in dict.fromkeys(r["variant"] for r in rows):
        group = [r for r in rows if r["variant"] == name]
        if len(group) != len(references):
            raise ValueError("Incomplete reference or variant repetitions")
        median = statistics.median(r["seconds"] for r in group)
        predictions = group[0]["predictions"]
        paired = []
        for row in group:
            ref = references[row["repeat"]]
            if len(row["predictions"]) != len(ref["predictions"]):
                raise ValueError("Variant/reference output counts differ")
            if row["generation_profile"]["logical_input_tokens"] != ref["generation_profile"]["logical_input_tokens"]:
                raise ValueError("Variant/reference logical input counts differ")
            paired.append({"repeat": row["repeat"], "seconds_saved": ref["seconds"] - row["seconds"],
                           "text_differences": sum(a != b for a, b in zip(ref["predictions"], row["predictions"]))})
        comparisons[name] = {
            "seconds": [r["seconds"] for r in group], "median_seconds": median,
            "median_time_reduction_percent": 100 * (1 - median / reference_median),
            "speedup": reference_median / median,
            "median_process_cpu_seconds": statistics.median(r["process_cpu_seconds"] for r in group),
            "peak_allocated_mib": [r["peak_allocated_mib"] for r in group],
            "peak_reserved_mib": [r["peak_reserved_mib"] for r in group],
            "quality_by_repeat": [r["quality"] for r in group], "paired_results": paired,
            "repeat_prediction_disagreements": [sum(a != b for a, b in zip(predictions, r["predictions"])) for r in group],
            "median_generation_profile": {key: statistics.median(r["generation_profile"][key] for r in group)
                                          for key in ("sorting_s", "batch_tokenization_s", "text_decoding_s",
                                                      "logical_input_tokens", "padded_input_token_slots", "generated_token_slots", "batches")},
        }
    return {"reference": reference_name, "sample_count": len(next(iter(references.values()))["predictions"]),
            "comparisons": comparisons,
            "note": "Medians are from one session, with variant order rotated across repeats. Phase wall times may contain synchronization; they are not isolated CPU or GPU execution times."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sample-num", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-sample-num", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--cpu-gpu-study", action="store_true")
    parser.add_argument("--padding-study", action="store_true")
    parser.add_argument("--execution-study", action="store_true",
                        help="Separate fixed-eight-token CPU/CUDA traces for sorted batch 8, batch 32 and combined changes.")
    parser.add_argument("--cpu-threads", type=int, default=8,
                        help="PyTorch intra-op reference thread count; inter-op threads are unchanged.")
    parser.add_argument("--reverse-odd-rounds", action="store_true",
                        help="Reproduce the initial allocator/score study's rotated/reversed order.")
    parser.add_argument("--variants", nargs="+", default=["baseline", "retain_allocator_cache", "score_collection_on"],
                        choices=["baseline", "retain_allocator_cache", "score_collection_on", "batch4", "batch8", "batch8_length_sorted",
                                 "sorted_reuse_tokens", "sorted_decode_new", "sorted_threads1", "sorted_threads2", "sorted_threads4",
                                 "sorted_combined", "sorted_batch32", "sorted_batch32_combined"])
    args = parser.parse_args()
    if args.sample_num < 3 or args.repeats < 3:
        parser.error("Use at least three prompts and three repeats")
    if args.warmup_sample_num < 1:
        parser.error("warmup-sample-num must be positive")
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.source_json.read_text(encoding="utf-8"))
    samples = source["samples_by_repeat"][0][:args.sample_num]
    config = Config(config_dict={
        "model2path": {"qwen2.5-0.5b-instruct": str(args.model_path.resolve())},
        "generator_model": "qwen2.5-0.5b-instruct", "framework": "hf",
        "generator_batch_size": 1, "generation_params": {"do_sample": False, "max_tokens": 32},
        "disable_save": True,
        "metrics": ["em", "f1", "acc"], "save_metric_score": False, "save_intermediate_data": False,
    })
    template = PromptTemplate(config,
        system_prompt="Answer the question based on the given document. \
                    Only give me the answer and do not output any other words. \
                    \nThe following are given documents.\n\n{reference}",
        user_prompt="Question: {question}\nAnswer:")
    prompts = [template.get_string(question=s["question"], retrieval_result=s["retrieved_documents"]) for s in samples]
    generator = get_generator(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this study")
    lengths = [len(generator.tokenizer.encode(p, truncation=True, max_length=generator.max_input_len)) for p in prompts]
    if lengths != [s["generator_input_tokens"] for s in samples]:
        raise RuntimeError("Reconstructed prompts differ from saved generator token counts")
    expected = [s["prediction"] for s in samples]
    evaluator = Evaluator(config)
    evaluation_data = Dataset(config=config, data=[
        {"id": s["id"], "question": s["question"], "golden_answers": s["golden_answers"]} for s in samples])
    (args.output_dir / "workload.json").write_text(json.dumps({
        "source": str(args.source_json), "source_sha256": hashlib.sha256(args.source_json.read_bytes()).hexdigest(),
        "ids": [s["id"] for s in samples], "input_tokens": lengths,
        "prompt_sha256": hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")

    if args.cpu_gpu_study:
        if len(prompts) < 8:
            parser.error("CPU/GPU study requires at least eight prompts")
        cpu_gpu_study(generator, prompts, samples, expected, args)
        return

    # Warm up before profiling, then profile representative short/median/long prompts.
    generator.generate(prompts[:5], clear_cuda_cache=True)
    ordered = sorted(range(len(prompts)), key=lambda i: lengths[i])
    profile_ids = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
    events = []
    annotations = []
    if not args.skip_profile:
        with diagnostics(generator, events):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True, profile_memory=True) as prof:
                generator.generate([prompts[i] for i in profile_ids], clear_cuda_cache=True)
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output_dir / "profile_trace.json"))
        operators = prof.key_averages()
        (args.output_dir / "profile_operators.txt").write_text(
            operators.table(sort_by="self_cuda_time_total", row_limit=40), encoding="utf-8")
        annotations = [{"name": e.key, "calls": e.count, "cpu_total_us": e.cpu_time_total,
                        "device_total_us": getattr(e, "device_time_total", 0)}
                       for e in operators if e.key.startswith("rag.")]
    event_summary = {phase: {"calls": sum(p == phase for p, _, _ in events),
                            "cuda_stream_interval_ms": sum(start.elapsed_time(end) for p, start, end in events if p == phase)}
                     for phase in ("prefill", "decode")}

    variants = {
        "baseline": {"clear_cuda_cache": True, "collect_scores": False},
        "retain_allocator_cache": {"clear_cuda_cache": False, "collect_scores": False},
        "score_collection_on": {"clear_cuda_cache": True, "collect_scores": True},
        "batch4": {"clear_cuda_cache": True, "collect_scores": False, "batch_size": 4},
        "batch8": {"clear_cuda_cache": True, "collect_scores": False, "batch_size": 8},
        "batch8_length_sorted": {"clear_cuda_cache": True, "collect_scores": False, "batch_size": 8, "sort_by_length": True},
    }
    sorted_options = variants["batch8_length_sorted"]
    variants.update({
        "sorted_reuse_tokens": dict(sorted_options, reuse_tokenization=True),
        "sorted_decode_new": dict(sorted_options, decode_generated_only=True),
        "sorted_threads1": dict(sorted_options, torch_threads=1),
        "sorted_threads2": dict(sorted_options, torch_threads=2),
        "sorted_threads4": dict(sorted_options, torch_threads=4),
        "sorted_combined": dict(sorted_options, reuse_tokenization=True, decode_generated_only=True, torch_threads=1),
        "sorted_batch32": dict(sorted_options, batch_size=32),
        "sorted_batch32_combined": dict(sorted_options, batch_size=32, reuse_tokenization=True, decode_generated_only=True),
    })
    execution_variants = {name: variants[name] for name in (
        "batch8_length_sorted", "sorted_batch32", "sorted_batch32_combined")}
    variants = {name: variants[name] for name in args.variants}
    rows = []
    for repeat in range(args.repeats):
        names = list(variants)
        names = names[repeat % len(names):] + names[:repeat % len(names)]
        if args.reverse_odd_rounds and repeat % 2:
            names.reverse()
        for name in names:
            options = dict(variants[name])
            torch.set_num_threads(options.pop("torch_threads", args.cpu_threads))
            torch.cuda.empty_cache()  # Outside timing; each variant then warms its own path.
            generator.generate(prompts[:args.warmup_sample_num], **options)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_stats()
            start = time.perf_counter()
            cpu_start = time.process_time()
            output = generator.generate(prompts, **options)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            cpu_s = time.process_time() - cpu_start
            after = torch.cuda.memory_stats()
            row = {"repeat": repeat + 1, "variant": name, "seconds": elapsed, "process_cpu_seconds": cpu_s,
                   "torch_intraop_threads": torch.get_num_threads(),
                   "torch_interop_threads": torch.get_num_interop_threads(),
                   "text_mismatches_against_saved": sum(x != y for x, y in zip(output, expected)),
                   "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                   "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                   "device_allocations": after.get("num_device_alloc", 0) - before.get("num_device_alloc", 0),
                   "device_frees": after.get("num_device_free", 0) - before.get("num_device_free", 0)}
            if len(output) != len(expected):
                raise RuntimeError("Incorrect output count")
            evaluation_data.update_output("pred", output)
            row["quality"] = evaluator.evaluate(evaluation_data)
            row["predictions"] = output
            row["generation_profile"] = dict(generator.last_generation_profile)
            rows.append(row)
            print("[study] " + json.dumps({key: value for key, value in row.items()
                                           if key not in ("predictions", "generation_profile")}), flush=True)
            (args.output_dir / "measurements.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    summary = {name: {"median_seconds": statistics.median(r["seconds"] for r in rows if r["variant"] == name),
                      "runs": [r for r in rows if r["variant"] == name]} for name in variants}
    payload = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(0), "dtype": str(generator.model.dtype), "sample_count": len(samples),
               "reference_cpu_threads": args.cpu_threads,
               "generator_source_sha256": hashlib.sha256((ROOT / "flashrag/generator/generator.py").read_bytes()).hexdigest(),
               "repeats": args.repeats, "variants": variants, "summary": summary,
               "warmup_sample_count": min(args.warmup_sample_num, len(samples)),
               "profile_question_ids": [samples[i]["id"] for i in profile_ids],
               "profile_input_lengths": [lengths[i] for i in profile_ids],
               "diagnostic_annotations": annotations, "diagnostic_forward_events": event_summary,
               "note": "Profiler/event figures are diagnostic and instrumented; formal timings contain neither profiler nor forward hooks."}
    if args.padding_study:
        payload["padding_diagnostics"] = padding_diagnostics(generator, prompts, samples, args.output_dir)
    if args.execution_study:
        torch.set_num_threads(args.cpu_threads)
        payload["execution_diagnostics"] = padding_diagnostics(
            generator, prompts, samples, args.output_dir, execution_variants)
    (args.output_dir / "study.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "comparison.json").write_text(
        json.dumps(compare_trials(rows, next(iter(variants))), indent=2) + "\n", encoding="utf-8")
    trace_path = args.output_dir / "profile_trace.json"
    if not args.skip_profile and trace_path.is_file():
        with trace_path.open("rb") as src, gzip.open(args.output_dir / "profile_trace.json.gz", "wb", compresslevel=3) as dst:
            shutil.copyfileobj(src, dst)
        trace_path.unlink()
    print("[done] " + json.dumps({k:v["median_seconds"] for k,v in summary.items()}), flush=True)


if __name__ == "__main__":
    main()
