# RAG Performance Profiling Report

**Purpose:** Project meeting discussion draft | **Profiling date:** October 14, 2026 | **Scope:** Recent profiling-related code changes and a CPU-only retrieval measurement on a MacBook

## Executive Summary

- The latest profiling work adds per-stage timing and token-count diagnostics, plus a repeatable baseline script.
- On this MacBook, warm CPU retrieval for five NQ questions took a median of **0.087 s per batch** (**17.5 ms/query**) after two warmups. Retriever initialization took **3.342 s**; the first query batch took **2.278 s**, including first-encode overhead.
- These measurements cover retrieval only. End-to-end generation could not be measured because the Qwen generator weights are not available locally.
- Warm retrieval is not currently the strongest optimization target. The most promising code-level hypothesis is avoidable generation-score computation, but it needs an A/B measurement before calling it a bottleneck.
- Do not compare these Mac results directly with the earlier PC run unless model versions, data, batch size, warmup policy, precision, and timing boundaries match.

## Recent Profiling Changes

Commit `1c8df11` (`Add RAG baseline profiling and token counts`, October 8, 2026) changed three files:

| Area | Change |
| --- | --- |
| `examples/quick_start/perf_baseline.py` | Added a Qwen 2.5 0.5B + E5 baseline runner; uses five test samples, three warmups, ten timed repetitions, median batch latency, and retrieval-only throughput calculation. |
| `flashrag/pipeline/pipeline.py` | Added retrieval, prompt-construction, and generation timers; prints the generator device; added optional query rewriting to `SequentialPipeline`. |
| `flashrag/generator/generator.py` | Prints input/generated token counts and times generation-score post-processing. |

The script configures `retrieval_topk=5`, `generator_batch_size=1`, and up to 32 generated tokens. It disables evaluation during timed iterations and evaluates only after the repetitions.

## MacBook CPU Measurement

| Measurement | Result |
| --- | ---: |
| Runtime environment | Python 3.10.22, PyTorch 2.13.0, Transformers 5.19.0 (`flashrag-clean`) |
| Device used | CPU; the E5 model reported `cpu` |
| Dataset | NQ test split, five sampled questions |
| Corpus and index | 15,000 documents; 44 MB FAISS index; 6.4 MB corpus file |
| Retriever initialization | 3.342 s, measured around retriever construction (corpus, index, and E5 model loading) |
| First five-query search | 2.278 s, including initial query encoding/runtime warmup |
| Warm search protocol | Two warmup calls followed by five measured calls |
| Warm search results | 0.092, 0.084, 0.090, 0.084, 0.087 s per five-query batch |
| Warm median | **0.087 s per batch**, approximately **17.5 ms/query** |
| Approximate warm retrieval throughput | **57 queries/s**, retrieval-only (5 queries / 0.087 s) |
| Generation / end-to-end time | Not measured: no local Qwen generator weights |

These are small-sample local measurements, useful as a starting point rather than a general performance claim. The first query batch is not representative of steady-state latency. The prior PC result was not available in this run, so no cross-device speedup is claimed.

## Optimization Opportunities

| Priority | Hypothesis | Evidence and confidence | Suggested experiment |
| --- | --- | --- | --- |
| 1 | Generation may spend unnecessary time and memory producing token scores when only answer text is requested. | `HFCausalLMGenerator.generate()` always requests `output_scores=True`, stacks per-token vocabulary logits, applies softmax, and gathers generated-token scores. This is a code-path observation, **not yet a measured bottleneck**. | Compare the current path with a path that skips score collection when neither `return_scores` nor `return_dict` is requested. Use identical prompts, model, device, warmups, and repetitions; compare latency, peak memory, and text equality. Keep score-returning API behavior covered by tests. |
| 2 | Mac retrieval may be able to use MPS instead of CPU. | MPS is available in the local environment, but `get_device()` selects CUDA when available and otherwise CPU. Thus the measured E5 run used CPU. | Benchmark CPU and MPS separately with the same E5 checkpoint and query set. Verify numerical compatibility and synchronize around timed MPS work. Keep the CPU path as the baseline; do not assume MPS is faster for this small workload. |
| 3 | Per-stage accelerator timings may under-report asynchronous work. | Pipeline timers use `time.perf_counter()` around calls. The outer baseline synchronizes around the whole run, but the individual retrieval/generation timers do not consistently synchronize at their boundaries. This does not affect the CPU measurement above, but matters for MPS/CUDA comparisons. | Add device-aware synchronization immediately before and after each measured stage, then compare stage totals with synchronized end-to-end time. |
| 4 | Retrieval tuning is lower priority at this sample size. | Warm CPU retrieval is about 17.5 ms/query for top-5 on the included five-question sample. This does not establish retrieval cost at larger batch sizes or corpus sizes. | Revisit retrieval after full-pipeline timings exist, or use a larger representative query set to test batching and throughput. |

## Measurement Caveats

- The full `perf_baseline.py` run was not completed because the local Qwen generator checkpoint is missing. Therefore there is no measured prompt-to-answer latency, generation latency, or end-to-end QPS.
- The committed script calls `torch.mps.synchronize()` unconditionally. That may fail in environments without a usable MPS backend; make synchronization device-aware before treating the script as portable across Mac, Linux, and Windows.
- The pipeline prints stage timings, but evaluation is intentionally excluded from the timed loop. Report inference and evaluation costs separately if the meeting question concerns total experiment runtime.
- The current run used only five questions. It is too small to draw strong conclusions about full-dataset throughput or statistical variance.
- No source-code optimization was applied as part of this profiling report; the listed opportunities are experiments to validate.

## Reproducing the Full Baseline

After making both model directories available locally, run from the repository root:

```bash
cd examples/quick_start
PYTHONPATH=../.. conda run -n flashrag-clean python perf_baseline.py \
  --model_path /path/to/qwen2.5-0.5b-instruct \
  --retriever_path /path/to/e5-base-v2
```

The script performs three warmup runs and ten timed runs over the configured five test questions. For a CPU-versus-MPS or PC-versus-Mac comparison, keep model checkpoints, prompts, generation parameters, sample order, precision, and timing/synchronization policy identical.

## Questions for the Project Meeting

1. Should the next milestone optimize single-query latency, throughput at larger batch sizes, or end-to-end experiment runtime?
2. Is changing the generator to skip score collection on text-only calls acceptable, provided score-returning APIs retain their behavior?
3. Should Mac MPS support be a project goal, or should the benchmark remain CPU/CUDA-focused?
4. Which PC baseline configuration and measurements should be used for a fair comparison?
5. What representative dataset size and latency/quality trade-off should define success?
