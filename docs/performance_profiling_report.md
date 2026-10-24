# RAG Performance Profiling Report

**Purpose:** Profiling and adaptive top-k experiment log | **Updated:** October 24, 2026 | **Scope:** Profiling changes, a CPU-only retrieval measurement, and the fixed top-k comparison setup

## Executive Summary

- The profiling work adds per-stage timing and token-count diagnostics, plus a repeatable fixed top-k comparison runner.
- On this MacBook, warm CPU retrieval for five NQ questions took a median of **0.087 s per batch** (**17.5 ms/query**) after two warmups. Retriever initialization took **3.342 s**; the first query batch took **2.278 s**, including first-encode overhead.
- These historical measurements cover retrieval only. End-to-end generation and the new fixed-k comparison have not yet been measured with valid model paths.
- The current experiment is a fixed-k baseline (`k=1,3,5,10`), not an adaptive policy. Its purpose is to measure the quality/cost trade-off before designing per-question k selection.
- Do not compare these Mac results directly with the earlier PC run unless model versions, data, batch size, warmup policy, precision, and timing boundaries match.

## RAG and Experiment Context

The current pipeline works as follows:

1. E5 encodes the question into a vector.
2. FAISS searches the existing vector index and returns the top `k` documents.
3. The prompt combines the question with those documents.
4. Qwen generates an answer from that prompt.
5. The evaluator scores the answer using EM, F1, and accuracy; the benchmark also records input-token counts and latency.

E5 is the **retriever**; Qwen is the **answer generator**. The README's index-builder argument `--model_path` refers to E5. In `perf_baseline.py`, `--retriever_path` refers to E5 and `--model_path` refers to Qwen. These similarly named arguments belong to different commands and have different roles.

The optimization goal is to find whether a smaller or per-question k can preserve answer quality while reducing prompt/generation cost. Reducing k may have only a small effect on FAISS search time because the index still searches the corpus. Adaptive top-k is not implemented yet.

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
| 1 | A smaller or adaptive k may reduce prompt and generation cost while preserving answer quality. | The current baseline uses fixed top-5; no k comparison or adaptive policy has been measured yet. | Run the fixed-k comparison below on a representative sample. Then design and test an adaptive policy against fixed top-5. |
| 2 | Generation may spend unnecessary time and memory producing token scores when only answer text is requested. | `HFCausalLMGenerator.generate()` always requests `output_scores=True`, stacks per-token vocabulary logits, applies softmax, and gathers generated-token scores. This is a code-path observation, **not yet a measured bottleneck**. | Compare the current path with a path that skips score collection when neither `return_scores` nor `return_dict` is requested. Use identical prompts, model, device, warmups, and repetitions; compare latency, peak memory, and text equality. Keep score-returning API behavior covered by tests. |
| 3 | Mac retrieval may be able to use MPS instead of CPU. | MPS is available in the local environment, but `get_device()` selects CUDA when available and otherwise CPU. Thus the historical E5 run used CPU. | Benchmark CPU and MPS separately with the same E5 checkpoint and query set. Verify numerical compatibility and synchronized timings; do not assume MPS is faster for this workload. |

## Measurement Caveats

- The historical full baseline was not completed because the local Qwen generator checkpoint was missing. A later smoke-test attempt also stopped before inference because the example placeholder model path was passed literally. Therefore there are still no measured prompt-to-answer, generation, or end-to-end comparison results.
- The fixed-top-k runner synchronizes CUDA/MPS only when the selected device is available. The pipeline exposes retrieval, prompt, and generation timings; evaluation is kept outside the timed inference loop.
- The pipeline prints stage timings, but evaluation is intentionally excluded from the timed loop. Report inference and evaluation costs separately if the meeting question concerns total experiment runtime.
- The current run used only five questions. It is too small to draw strong conclusions about full-dataset throughput or statistical variance.
- No adaptive retrieval policy or measured source-code optimization has been completed yet; the fixed-k runner is an experiment harness.

## Fixed Top-k Comparison

The baseline runner accepts multiple fixed retrieval values of `k` and records each setting independently. It currently uses the local Hugging Face generator configuration; OpenAI API generation is not wired into this benchmark. From `examples/quick_start`, run with actual local model directories or valid Hugging Face model IDs:

```bash
PYTHONPATH=../.. conda run -n flashrag-clean python perf_baseline.py \
  --model_path /path/to/qwen2.5-0.5b-instruct \
  --retriever_path /path/to/e5-base-v2 \
  --k-values 1 3 5 10 \
  --sample-num 100 \
  --warmups 2 \
  --repeats 5
```

By default, each run writes timestamped artifacts into `docs/`: one summary CSV containing per-k quality, token, stage-latency, and end-to-end measurements, plus one JSON per k containing metrics and per-question predictions, retrieved documents, and input-token counts for every measured repeat. Use `--output-dir` to select a different destination. The script flushes the summary after each k so completed settings remain recorded if a later setting fails.

The default sample count is five for a quick smoke test. Use a larger representative sample for conclusions, and compare each k against fixed top-5 before selecting an adaptive policy. The tokenizer configured for `input_tokens` is the GPT-4 tokenizer, so treat that figure as a consistent relative measure unless it matches the generator's tokenizer.

After each k completes, the script writes a timestamped summary CSV row and a JSON file in `docs/`. The CSV contains answer metrics, average input tokens, mean documents per query, stage latency, end-to-end latency, and throughput. The per-k JSON contains metrics and per-question predictions and retrieved documents for each measured repeat. These artifacts are generated by a run and should be reviewed before deciding on an adaptive rule.

## Reproducing the Full Baseline

Run from `examples/quick_start`. `--model_path` must identify the Qwen generator; `--retriever_path` must identify the E5 model used to build the index. The index and corpus must be the matching pair configured in `perf_baseline.py` (`indexes/e5_Flat.index` and `indexes/general_knowledge.jsonl`). For example, the E5 README index-builder path belongs in `--retriever_path`, not the benchmark's `--model_path`:

```bash
cd examples/quick_start
PYTHONPATH=../.. conda run -n flashrag-clean python perf_baseline.py \
  --model_path /path/to/qwen2.5-0.5b-instruct \
  --retriever_path /path/to/e5-base-v2
```

The script defaults to one warmup and three measured repeats over five questions. Use `--warmups` and `--repeats` to increase them for a formal run. For a CPU-versus-MPS or PC-versus-Mac comparison, keep model checkpoints, prompts, generation parameters, sample order, precision, and timing/synchronization policy identical.

## Current Status and Next Steps

- Completed: added retrieval/prompt/generation timing instrumentation and a configurable fixed-k comparison runner; results are checkpointed under `docs/` after each k.
- Not completed: a valid-model smoke test, the fixed `k=1,3,5,10` measurements, and an adaptive top-k implementation.
- Next: provide real Qwen and E5 model paths (or model IDs), run a small smoke test, then run a larger sample with multiple repeats. Review the CSV/JSON results and choose a quality target relative to fixed top-5 before implementing adaptive k.
- Success criterion: reduce average retrieved documents and end-to-end latency/input tokens while keeping answer quality within an agreed tolerance of fixed top-5.

## Questions for the Project Meeting

1. Should the next milestone optimize single-query latency, throughput at larger batch sizes, or end-to-end experiment runtime?
2. Is changing the generator to skip score collection on text-only calls acceptable, provided score-returning APIs retain their behavior?
3. Should Mac MPS support be a project goal, or should the benchmark remain CPU/CUDA-focused?
4. Which PC baseline configuration and measurements should be used for a fair comparison?
5. What representative dataset size and latency/quality trade-off should define success?
