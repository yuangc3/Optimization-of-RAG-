# <div align="center">⚡FlashRAG: A Python Toolkit for Efficient RAG Research<div>

## CS598APE Final Project: Optimizing the Adaptive Top-K RAG Pipeline

**Authors:** Yidong Zhang and Yuang Cai

**Report:** [PDF (submission version)](docs/performance_profilling_report.pdf) · [Markdown source with full experiment history](docs/performance_profilling_report.md)

This repository is a fork of [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG). We optimized a retrieval-augmented generation (RAG) pipeline: E5 retriever → FAISS → Qwen2.5-0.5B generator. The main contribution is **Adaptive Top-K retrieval**: choose, for each query, how many retrieved documents to pass to the LLM. Four supporting optimizations make generation itself cheaper.

| ID | Optimization | Where it lives | How to switch it on | Recorded result (one machine; see report) |
| --- | --- | --- | --- | --- |
| **1** | **Adaptive Top-K selection** (primary): score-gap and similarity-threshold policies | `flashrag/retriever/adaptive_topk.py`, `flashrag/pipeline/pipeline.py` | `SequentialPipeline(..., retrieval_policy=...)`; benchmark flag `--adaptive` | Final 500-question study: threshold policy uses 5.6% fewer documents and 4.7% fewer input tokens than fixed k=5, with EM/F1 ≥ fixed-5. The end-to-end latency gain is **not** significant. |
| 2 | Skip unneeded generation-score collection | `HFCausalLMGenerator.generate()` in `flashrag/generator/generator.py` | On by default; `--collect-scores` restores the old behavior | −10.58 MiB peak GPU allocation on 100 prompts; no latency gain shown |
| 3 | Keep the CUDA allocator cache (no `empty_cache()` per batch) | `flashrag/generator/generator.py` | `--retain-cuda-cache` | Device allocations 339 → 2; latency gain inconclusive |
| 4 | Batched generation (+ continuation-only decoding) | `flashrag/generator/generator.py` | `--generator-batch-size N` (`--decode-generated-only`) | Batch 1 → 8: 2.76× generation throughput on 100 prompts |
| 5 | Length-aware batching (+ tokenization reuse) | `flashrag/generator/generator.py` | `--sort-by-length` (`--reuse-tokenization`) | −11.46% generation time on 500 prompts, but +56.68% peak memory |

The speedups come from different baselines and workloads, so **do not multiply them**. The final Adaptive Top-K study (Section 6) runs every method under one identical configuration (batch 8, no length sorting), so the Adaptive-K results are not mixed up with Optimizations 2–5. We report all results, including the negative ones.

**Quickest way to check our work (about 5 minutes):**

1. Do the setup in Section 2.
2. Run the unit tests (Section 4). They need no GPU and no models.
3. Optionally, run the short GPU demos in Section 5. Each takes under one minute.

---

### 1. What is where

| Purpose | Location |
| --- | --- |
| Adaptive Top-K policies (Opt 1) | [flashrag/retriever/adaptive_topk.py](flashrag/retriever/adaptive_topk.py) |
| Pipeline integration (Opt 1) | [flashrag/pipeline/pipeline.py](flashrag/pipeline/pipeline.py) (`SequentialPipeline`, argument `retrieval_policy`) |
| Generator optimizations (Opts 2–5) | [flashrag/generator/generator.py](flashrag/generator/generator.py) |
| End-to-end RAG benchmark: fixed-k vs adaptive, all generator flags | [examples/quick_start/perf_baseline.py](examples/quick_start/perf_baseline.py) |
| Generation-only A/B profiler for Opts 2–5 (replays frozen prompts) | [scripts/profile_generation.py](scripts/profile_generation.py) |
| Final Adaptive Top-K study (calibrate → freeze → evaluate) | [scripts/run_final_topk_study.py](scripts/run_final_topk_study.py) |
| Paired quality and latency analysis of a benchmark folder | [scripts/summarize_adaptive_benchmark.py](scripts/summarize_adaptive_benchmark.py) |
| Unit tests (56) | [tests/](tests/) |
| Input questions (NQ) | [datasets/](datasets/), described in Section 3 |
| Retrieval corpus and FAISS index | [examples/quick_start/indexes/](examples/quick_start/indexes/) |
| Recorded results of every experiment | [docs/experiments/](docs/experiments/), one folder per experiment (listed in Section 5) |
| Environment record | [docs/environment-lock.txt](docs/environment-lock.txt), [docs/environment_check.json](docs/environment_check.json), [docs/final_resource_manifest.json](docs/final_resource_manifest.json) |

---

### 2. Setup

**Tested on:**

- Windows 11 (build 26200), AMD Ryzen 7 7800X3D, ~63 GiB RAM
- NVIDIA RTX 4070 SUPER (12 GiB VRAM)
- Python 3.11.9, PyTorch 2.11.0+cu128, CPU FAISS

**What needs a GPU:**

- **Unit tests (Section 4) run on CPU only.**
- All benchmarks (Sections 5 and 6) need a CUDA GPU.
- We did not validate Linux or macOS.

All commands below are PowerShell, run from the repository root.

```powershell
# 2.1 Environment
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-baseline.txt

# 2.2 Generator: Qwen2.5-0.5B-Instruct (pinned revision) -> models/Qwen2.5-0.5B-Instruct  (~1 GB)
.\.venv\Scripts\python.exe scripts/download_qwen.py

# 2.3 Retriever: E5-base-v2 (pinned revision) -> Hugging Face cache  (~440 MB)
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('intfloat/e5-base-v2', revision='f52bf8ec8c7124536f0efb74aca902b2995e5bcd', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']))"
```

**Environment variables for every benchmark command** (set them once per PowerShell window):

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:HF_HUB_OFFLINE = '1'          # remove these two lines if models are not downloaded yet
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_DATASETS_CACHE = Join-Path (Get-Location).Path 'models/datasets-cache'
$env:TIKTOKEN_CACHE_DIR = Join-Path (Get-Location).Path 'models/tiktoken-cache'
$E5 = Join-Path $env:USERPROFILE '.cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd'
```

If you stored the models somewhere else, set `$E5` to the E5 folder. Pass a different Qwen folder with `--model_path` (benchmark) or `--model-path` (profiler and study).

---

### 3. Input data

Everything except the model weights is **already in this repository**. Each file is pinned by SHA-256, and the scripts check the checksums.

| File | Content | Source |
| --- | --- | --- |
| `datasets/flashrag_official/nq/test.jsonl` | Official FlashRAG NQ test split, 3,610 questions (488 KB) | Hugging Face [`RUC-NLPIR/FlashRAG_datasets`](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) @ `bcafb8dd07d453be3cbeeeb3f78be1841bddf92c`, file `nq/test.jsonl` |
| `examples/quick_start/dataset/nq/test.jsonl` | 17 questions used for early debugging | Upstream FlashRAG |
| `datasets/nq_500_heldout/nq/test.jsonl` | 500 questions used by the Opt 1–5 experiments | `scripts/prepare_nq_benchmark.py`, seed 2024; [manifest](docs/nq_500_manifest.json) |
| `datasets/nq_final_study/nq/dev.jsonl` / `test.jsonl` | 200 calibration + 500 evaluation questions for the final study | Built by `run_final_topk_study.py`, seed 598; [manifest](docs/experiments/final_topk/split_manifest.json) |
| `examples/quick_start/indexes/general_knowledge.jsonl` + `e5_Flat.index` | Retrieval corpus (6.7 MB) + matching E5 FAISS Flat index (46 MB) | Upstream FlashRAG quick-start; checksums in [resource_audit.json](docs/experiments/final_topk/resource_audit.json) |

**Final-study splits.**

- The splits exclude all 517 questions used in earlier experiments.
- They are disjoint from each other.
- The script refuses to overwrite them if their content changes.

**Re-downloading the official NQ file** (this also rebuilds the 500-question set):

```powershell
.\.venv\Scripts\python.exe scripts/prepare_nq_benchmark.py --revision bcafb8dd07d453be3cbeeeb3f78be1841bddf92c --sample-num 500 --seed 2024
```

---

### 4. Unit tests (CPU only, no downloads, < 1 minute)

Run the whole suite:

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected ending: `Ran 56 tests ... OK`.

Run the tests for one optimization at a time:

```powershell
# Opt 1  Adaptive Top-K policies + pipeline integration (17 tests) and final-study protocol (3 tests)
.\.venv\Scripts\python.exe -m unittest tests.test_adaptive_topk tests.test_final_topk_study -v
# Opt 2  text-only path skips score work but gives identical text; score API still works
.\.venv\Scripts\python.exe -m unittest tests.test_generator_score_path -k text_only -k return_scores -v
# Opt 3  retaining the allocator cache skips empty_cache() and keeps text identical
.\.venv\Scripts\python.exe -m unittest tests.test_generator_score_path -k allocator -v
# Opt 4  batched outputs + continuation-only decoding keep ids/scores/logits aligned
.\.venv\Scripts\python.exe -m unittest tests.test_generator_score_path -k return_dict -k generated_only -v
# Opt 5  length sorting restores the original order; tokenization reuse is exact
.\.venv\Scripts\python.exe -m unittest tests.test_generator_score_path -k length_sort -k reused -v
```

---

### 5. The optimizations: what each does and how to run it

The benchmarks below use two tools.

- **`perf_baseline.py`** runs the **whole RAG pipeline**: retrieval, document selection, prompting, generation and evaluation. It compares fixed-k with adaptive selection and accepts every generator flag.
- **`profile_generation.py`** isolates **generation only**. It replays the exact prompts saved by an earlier `perf_baseline.py` run, so the inputs are identical across variants. It then runs variants A/B with interleaved repetitions and checks every output text against the saved answers.

Each optimization below has:

- **Quick demo:** a small run that checks the code works (under 1 minute on the tested GPU). It does not reproduce the reported numbers.
- **Full reproduction:** the exact command behind the reported result.
- **Recorded evidence:** the folder holding our original output.

Write every rerun to a **new** `--output-dir`. Never overwrite `docs/experiments/*`, because those folders are the evidence for the report.

#### Optimization 1: Adaptive Top-K selection (primary contribution)

**Idea.** Fixed top-k always sends k documents to the LLM. Our policies look only at the retriever similarity scores of one query and cut the ranked list early when the scores show a clear boundary. Fewer documents give shorter prompts and less generation work. The policies:

- **Score-gap:** sort the scores s1 ≥ s2 ≥ …, then cut after the largest adjacent drop if that drop is at least `gap_threshold`; otherwise keep `fallback_k` documents.
- **Similarity-threshold:** keep the documents with score ≥ `score_threshold` (at least `min_k` and at most `max_k`). If no score qualifies, keep `fallback_k`, so a weak query never gets *less* context.

Both policies run in O(k) per query. They never see the question text or the gold answers. They record the chosen k and the reason in the output.

**Use it in code:**

```python
from flashrag.retriever.adaptive_topk import SimilarityThresholdPolicy, ScoreGapPolicy
from flashrag.pipeline import SequentialPipeline

policy = SimilarityThresholdPolicy(score_threshold=0.7174408435821533, min_k=2, max_k=5, fallback_k=5)
# or: policy = ScoreGapPolicy(gap_threshold=0.03, min_k=2, max_k=5, fallback_k=5)
pipeline = SequentialPipeline(config, retrieval_policy=policy)   # retrieval_policy=None -> normal fixed top-k
```

**CPU demo, no models needed:**

```powershell
.\.venv\Scripts\python.exe -c "from flashrag.retriever.adaptive_topk import SimilarityThresholdPolicy as P; print(P(score_threshold=0.7174408435821533, min_k=2, max_k=5, fallback_k=5).select([0.80, 0.76, 0.73, 0.70, 0.69]))"
# -> {'selected_k': 3, ..., 'reason': 'similarity_threshold', ...}  three scores pass the threshold, so 3 docs are kept
```

**Quick GPU demo (~15 s):** fixed k=5 vs adaptive score-gap on 20 questions, full RAG pipeline:

```powershell
.\.venv\Scripts\python.exe examples/quick_start/perf_baseline.py `
  --model_path models/Qwen2.5-0.5B-Instruct --retriever_path $E5 `
  --data-dir datasets/nq_500_heldout --k-values 5 --adaptive --adaptive-gap 0.01 `
  --sample-num 20 --warmups 1 --warmup-sample-num 4 --repeats 1 --generator-batch-size 8 `
  --output-dir results/demo_opt1
.\.venv\Scripts\python.exe scripts/summarize_adaptive_benchmark.py results/demo_opt1
```

The console prints one `[summary]` line per setting: mean documents, EM/F1, average input tokens, median seconds and queries/s. The summarizer then prints the paired adaptive-vs-fixed quality comparison.

**Full reproduction:**

- The **final, held-out evaluation** of both policies is in Section 6. Use that one for grading.
- The earlier exploratory 500-question score-gap run (`min_k=1, max_k=10, gap=0.01`) is reproduced with:

```powershell
.\.venv\Scripts\python.exe examples/quick_start/perf_baseline.py `
  --model_path models/Qwen2.5-0.5B-Instruct --retriever_path $E5 `
  --data-dir datasets/nq_500_heldout --k-values 5 --adaptive --adaptive-gap 0.01 `
  --sample-num 500 --warmups 1 --warmup-sample-num 5 --repeats 3 `
  --output-dir results/repro_opt1_nq500
```

**Recorded evidence:**

- [final_topk/](docs/experiments/final_topk/): final study
- [nq500_gap001/](docs/experiments/nq500_gap001/): exploratory 500-question run
- [adaptive_gap_v1/](docs/experiments/adaptive_gap_v1/) and [adaptive_gap_sensitivity_001/](docs/experiments/adaptive_gap_sensitivity_001/): 17-question pilots

**Result.**

- The exploratory policy cut documents by 29.9% but lost EM (3.0% → 2.2%).
- That loss motivated the guarded policies (`min_k=2`, `max_k=5`, fallback) and a fresh, disjoint evaluation set (Section 6).

#### Optimization 2: Skip unneeded generation-score collection

**Idea.** The original generator always collected per-token output scores and post-processed them, even when the caller only wanted text. The text-only path now skips this work. Score-returning calls still collect the scores. These are LLM token scores, not retrieval similarities.

**Switch:** on by default. Restore the old behavior with `collect_scores=True`, the benchmark flag `--collect-scores`, or the profiler variant `score_collection_on`.

#### Optimization 3: Retain the CUDA allocator cache

**Idea.** The original code called `torch.cuda.empty_cache()` after every generation batch, which forced PyTorch to return memory to the driver and allocate it again for the next batch. Retaining the cache avoids those device allocations.

**Switch:** the benchmark flag `--retain-cuda-cache`, or the profiler variant `retain_allocator_cache`.

**Quick demo for Opts 2 and 3 together (~45 s, generation only, 16 prompts × 3 repeats):**

```powershell
.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_gap001/adaptive_topk_adaptive_gap_20261113_151722_140059.json `
  --sample-num 16 --repeats 3 --skip-profile `
  --variants baseline retain_allocator_cache score_collection_on `
  --output-dir results/demo_opt2_opt3
```

How to read the output:

- `baseline` already includes Opt 2 (no score collection). `score_collection_on` is the old behavior, and `retain_allocator_cache` adds Opt 3.
- Compare `peak_allocated_mib` (Opt 2) and `device_allocations` (Opt 3) between variants in the `[study]` lines or in `comparison.json`.
- `text_mismatches_against_saved` must be 0.

**Full reproduction (100 prompts, order reversed on odd rounds, includes the PyTorch profiler trace):**

```powershell
.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_gap001/adaptive_topk_adaptive_gap_20261113_151722_140059.json `
  --sample-num 100 --repeats 3 --reverse-odd-rounds `
  --output-dir results/repro_opt2_opt3
```

**Recorded evidence:** [generation_optimization/](docs/experiments/generation_optimization/). CPU thread screening for Opt 3 is in [host_overhead_pilot/](docs/experiments/host_overhead_pilot/).

**Results:**

- Opt 2 lowered peak allocation by 10.58 MiB, with no latency gain shown.
- Opt 3 cut device allocations from 339 to 2. Its 2.07% time change is within run-to-run noise.
- All 100 output texts matched in every run.

#### Optimization 4: Batched generation

**Idea.** Profiling showed that generation dominates RAG time and that batch size 1 leaves the GPU mostly idle, waiting on Python and kernel-launch overhead. Generating 8 (or 32) prompts per call amortizes that overhead. Continuation-only decoding (`--decode-generated-only`) decodes only the new tokens instead of the whole prompt plus answer.

**Switch:** `--generator-batch-size N` (default 1), optionally `--decode-generated-only`. Profiler variants: `batch4`, `batch8`, `sorted_batch32`, `sorted_decode_new`.

#### Optimization 5: Length-aware batching

**Idea.** In a batch, every prompt is padded to the longest one. Sorting prompts by token length before batching puts similar lengths together, which removes most of the padding work. Outputs are restored to the original order. `--reuse-tokenization` reuses the token IDs computed for sorting instead of tokenizing twice.

**Switch:** `--sort-by-length`, optionally `--reuse-tokenization`. It needs batch size > 1. Profiler variants: `batch8_length_sorted`, `sorted_reuse_tokens`.

**Quick demo for Opts 4 and 5 together (~50 s, 32 prompts × 3 repeats):**

```powershell
.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_gap001/adaptive_topk_adaptive_gap_20261113_151722_140059.json `
  --sample-num 32 --repeats 3 --skip-profile `
  --variants baseline batch8 batch8_length_sorted `
  --output-dir results/demo_opt4_opt5
```

On the tested GPU, the final `[done]` line showed batch 1 ≈ 7.3 s, batch 8 ≈ 3.0 s, and batch 8 + length sorting ≈ 2.4 s.

**Full reproduction:**

```powershell
# Opt 4  batch size 1 / 4 / 8 on 100 frozen prompts
.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_gap001/adaptive_topk_adaptive_gap_20261113_151722_140059.json `
  --sample-num 100 --repeats 3 --skip-profile --variants baseline batch4 batch8 `
  --output-dir results/repro_opt4_batching

# Opt 4  full RAG pipeline on 500 questions at batch 8
.\.venv\Scripts\python.exe examples/quick_start/perf_baseline.py `
  --model_path models/Qwen2.5-0.5B-Instruct --retriever_path $E5 `
  --data-dir datasets/nq_500_heldout --k-values 5 --adaptive --adaptive-gap 0.01 `
  --sample-num 500 --warmups 1 --warmup-sample-num 5 --repeats 3 --generator-batch-size 8 `
  --output-dir results/repro_opt4_nq500_batch8

# Opt 5  batch 8 vs batch 8 + length sorting on 500 frozen prompts (with padding/GPU diagnostic)
.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_batch8/adaptive_topk_adaptive_gap_20261113_160953_930358.json `
  --sample-num 500 --repeats 4 --warmup-sample-num 8 --skip-profile --padding-study `
  --variants batch8 batch8_length_sorted --output-dir results/repro_opt5_length_sorting

# Opts 4 + 5 follow-ups: token reuse, continuation decoding, batch 32 (500 prompts, CPU/CUDA traces)
.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_batch8/adaptive_topk_adaptive_gap_20261113_160953_930358.json `
  --sample-num 500 --repeats 3 --warmup-sample-num 8 --skip-profile --execution-study --cpu-threads 8 `
  --variants batch8_length_sorted sorted_reuse_tokens sorted_decode_new sorted_batch32 sorted_batch32_combined `
  --output-dir results/repro_opt45_followups
```

**Recorded evidence:**

| Folder | Content |
| --- | --- |
| [generation_batching/](docs/experiments/generation_batching/) | Batch 1/4/8 timings and memory (Opt 4) |
| [nq500_batch8/](docs/experiments/nq500_batch8/) | Full RAG pipeline at batch 8 (Opt 4) |
| [cpu_gpu_runtime/](docs/experiments/cpu_gpu_runtime/) | CPU/GPU traces (Opt 4) |
| [length_bucketing/](docs/experiments/length_bucketing/) | Length sorting (Opt 5) |
| [nq500_length_sorted/](docs/experiments/nq500_length_sorted/) | Full RAG pipeline with length sorting (Opt 5) |
| [host_overhead_pilot/](docs/experiments/host_overhead_pilot/), [host_overhead_500/](docs/experiments/host_overhead_500/), [nq500_host_optimized/](docs/experiments/nq500_host_optimized/) | Follow-ups: token reuse, continuation decoding, batch 32 |

**Results.**

- **Opt 4:**
  - Batch 1 → 8 gave 2.76× generation throughput.
  - With length sorting already on, batch 8 → 32 gave 2.71×, and 2.84× together with token reuse and continuation decoding.
  - Batching changes some output texts (floating-point batch effects), and fixed-5 EM dropped at batch 32.
- **Opt 5:**
  - Generation time fell 11.46% and padded slots fell 53.36%.
  - Peak memory rose 56.68%, so length sorting stays opt-in.

---

### 6. Final Adaptive Top-K study (the main result)

This study evaluates Optimization 1 under a **frozen, held-out protocol**, and it is the result to grade.

- All methods share one generator configuration: batch 8, no length sorting, greedy decoding with at most 32 new tokens.
- The only thing that differs between methods is the number of documents.

```powershell
.\.venv\Scripts\python.exe scripts/run_final_topk_study.py --stage all --output-dir results/repro_final_topk
```

What the script does:

1. **Builds the splits.** It verifies the NQ checksum and creates the 200 development and 500 evaluation questions described in Section 3.
2. **Develop (`--stage develop`).**
   - Audits the corpus/index pair.
   - Grid-searches 81 score-only policy configurations on the development questions.
   - For each family, picks the one that uses the fewest documents without losing any development answer hit compared with fixed k=5.
   - Freezes the choices in `frozen_policies.json`.
   - No evaluation question is used in this stage.
3. **Evaluate (`--stage evaluate`).**
   - Runs fixed k=1, 3, 5 and 10, score-gap, and similarity-threshold on the 500 evaluation questions.
   - Gives each method an 8-question warm-up, then 3 timed repetitions with rotated method order.
   - Compares EM/F1/accuracy against fixed-5 with paired bootstrap 95% intervals.

**Runtime:**

- About 43–59 s per method per repetition on the tested GPU.
- The 18 timed runs take about 15 minutes, plus development, warm-ups and model loading.
- Completed measurements are checkpointed, so rerun the same command to resume.

**Main output:** `evaluation_summary.json`. For each method it gives median time, queries/s, time change vs fixed-5, EM/F1/accuracy, mean k, input tokens, paired confidence intervals and per-stage times.

**Our recorded result:** [docs/experiments/final_topk/evaluation_summary.json](docs/experiments/final_topk/evaluation_summary.json)

| Method | Mean docs | Input tokens | Median time change vs fixed-5 | EM | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed k=1 | 1 | 138 | 8.0% faster | 2.8% | 7.88% |
| fixed k=3 | 3 | 309 | 4.1% faster | 4.0% | 8.66% |
| fixed k=5 (reference) | 5 | 474 | — | 2.6% | 7.66% |
| fixed k=10 | 10 | 785 | 24.8% slower | 2.2% | 7.10% |
| score-gap (frozen) | 4.93 | 469 | 1.3% faster | 2.6% | 7.73% |
| similarity-threshold (frozen) | 4.72 | 452 | 0.2% slower | 2.8% | 7.77% |

**Interpretation:**

- Both adaptive policies kept EM/F1 at or above fixed-5 while using fewer documents.
- On this small corpus, the calibration stayed conservative (mean k close to 5). The resulting time changes are within run-to-run noise, so **no robust latency gain is claimed**.
- Fixed k=3 is a strong simple baseline on this workload.
- Section 4 of the report discusses this in full.

---

### 7. Other entry points (optional)

- **Rebuild the PDF from the Markdown report:** `.\.venv\Scripts\python.exe scripts/export_project_report.py`
- **Build the submission archives:** `.\.venv\Scripts\python.exe scripts/package_submission.py`. This writes `dist/cs598ape_final_artifact.zip` and `dist/cs598ape_resources.zip` with a SHA-256 inventory. Extract both into the same folder. Model weights are never included. A `git clone` of this repository already contains the same files.
- **Windows CPU profiling:**
  - `scripts/profile_generation.py --windows-cpu-study ...` collects process/thread CPU counters and Python profiles.
  - `scripts/capture_windows_cpu.ps1` is an administrator-only WPR capture. Native ETW capture was blocked by a Windows privilege error and is not claimed.
- **Valgrind:** this project adds no C/C++ code, so Valgrind does not apply to our Python code.

### 8. Known limitations

- All results come from one Windows + RTX 4070 SUPER machine, a 0.5B generator and a small quick-start corpus. Absolute times will differ elsewhere.
- Calibrated thresholds are specific to E5-base-v2 and this corpus/index.
- Absolute answer quality is low for every method (EM 2.2–4.0%) because the generator is small. The study compares methods with each other.
- The development criterion "answer hit" is a lexical proxy (does a gold-answer string appear in a retrieved document), not a human relevance judgment.

---

The upstream FlashRAG documentation follows below.

---

\[ English | [中文](README_zh.md) \]
<div align="center">
<a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
<a href="https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://deepwiki.com/RUC-NLPIR/FlashRAG"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="DeepWiki Document" height="20"/></a>
<a href="https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

<h4 align="center">

<p>
<a href="#wrench-installation">Installation</a> |
<a href="#sparkles-features">Features</a> |
<a href="#rocket-quick-start">Quick-Start</a> |
<a href="#gear-components"> Components</a> |
<a href="#art-flashrag-ui"> FlashRAG-UI</a> |
<a href="#robot-supporting-methods"> Supporting Methods</a> |
<a href="#notebook-supporting-datasets--document-corpus"> Supporting Datasets</a> |
<a href="#raised_hands-additional-faqs"> FAQs</a>
</p>

</h4>


FlashRAG is a Python toolkit for the reproduction and development of Retrieval Augmented Generation (RAG) research. Our toolkit includes 36 pre-processed benchmark RAG datasets and **23 state-of-the-art RAG algorithms**, including **7 reasoning-based methods** that combine reasoning ability with retrieval.

<p align="center">
<img src="asset/framework.jpg">
</p>

With FlashRAG and provided resources, you can effortlessly reproduce existing SOTA works in the RAG domain or implement your custom RAG processes and components. Besides, we provide an easy-to-use UI:

https://github.com/user-attachments/assets/8ca00873-5df2-48a7-b853-89e7b18bc6e9

<p>
<a href="https://trendshift.io/repositories/10454" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10454" alt="RUC-NLPIR%2FFlashRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

## :link: Navigation
- [Features](#sparkles-features)
- [Roadmap](#mag_right-roadmap)
- [Changelog](#page_with_curl-changelog)
- [Installation](#wrench-installation)
- [Quick Start](#rocket-quick-start)
- [Components](#gear-components)
- [FlashRAG-UI](#art-flashrag-ui)
- [Supporting Methods](#robot-supporting-methods)
- [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
- [Additional FAQs](#raised_hands-additional-faqs)
- [License](#bookmark-license)
- [Citation](#star2-citation)

## :sparkles: Features

- **Extensive and Customizable Framework**: Includes essential components for RAG scenarios such as retrievers, rerankers, generators, and compressors, allowing for flexible assembly of complex pipelines.

- **Comprehensive Benchmark Datasets**: A collection of 36 pre-processed RAG benchmark datasets to test and validate RAG models' performances.

- **Pre-implemented Advanced RAG Algorithms**: Features **23 advancing RAG algorithms** with reported results, based on our framework. Easily reproducing results under different settings.

- **🚀 Reasoning-based Methods**: **NEW!** We now support **7 reasoning-based methods** that combine reasoning ability with retrieval, achieving superior performance on complex multi-hop tasks.

- **Efficient Preprocessing Stage**: Simplifies the RAG workflow preparation by providing various scripts like corpus processing for retrieval, retrieval index building, and pre-retrieval of documents.

- **Optimized Execution**: The library's efficiency is enhanced with tools like vLLM, FastChat for LLM inference acceleration, and Faiss for vector index management.

- **Easy to Use UI** : We have developed a very easy to use UI to easily and quickly configure and experience the RAG baselines we have implemented, as well as run evaluation scripts on a visual interface.

## :mag_right: Roadmap

FlashRAG is still under development and there are many issues and room for improvement. We will continue to update. And we also sincerely welcome contributions on this open-source toolkit.

- [x] Support OpenAI models
- [x] Provdide instructions for each component
- [x] Integrate sentence Transformers
- [x] Support multimodal RAG
- [x] Support reasoning-based methods
- [ ] Inlcude more RAG approaches
- [ ] Enhance code adaptability and readability
- [ ] Add support for api-based retriever (vllm server)

## :page_with_curl: Changelog
[25/11/06] 🎯 NEW Retriever! We have integrated a Web Search Engine-based Retriever, which seamlessly integrates with existing methods and can be enabled quickly with just a Serper API key! This enhancement significantly expands retrieval coverage and real-time capability, supporting dynamic information access and external knowledge augmentation. Experience a more flexible and powerful retrieval workflow now! 

[25/08/06] 🎯 **NEW!** We have added support for **Reasoning Pipeline**, which is a new paradigm that combines reasoning ability and retrieval, representing work that includes [R1-Searcher](https://github.com/SsmallSong/R1-Searcher), [Search-R1](https://github.com/PeterGriffinJin/Search-R1),.... We evaluate the performance of the pipeline on various RAG benchmarks, it can achieve F1 scores close to 60 on multi hop inference datasets such as HotpotQA. See it in [**result table**](#robot-supporting-methods).

[25/03/21] 🚀 **Major Update!** We have expanded our toolkit to support **23 state-of-the-art RAG algorithms**, including **7 reasoning-based methods** that significantly improve performance on complex reasoning tasks. This represents a major milestone in our toolkit's evolution!

[25/02/24] 🔥🔥🔥 We have added support for **multimodal RAG**, including [**MLLMs like Llava, Qwen, InternVL**](https://ruc-nlpir.github.io/FlashRAG/#/zh-cn/component/generator?id=%e5%a4%9a%e6%a8%a1%e6%80%81%e7%94%9f%e6%88%90%e5%99%a8), and various [**multimodal retrievers with Clip architecture**](https://ruc-nlpir.github.io/FlashRAG/#/zh-cn/component/retriever?id=%e5%a4%9a%e6%a8%a1%e6%80%81%e6%a3%80%e7%b4%a2%e5%99%a8). More information can be found in our new version of arxiv article and our documentation. Try it!

[25/01/21] Our technical paper [FlashRAG: A Python Toolkit for Efficient RAG Research](https://arxiv.org/abs/2405.13576) is honored to have been accepted to the Resource Track of the 2025 **ACM Web Conference (WWW 2025)**. Please Check it out!

[25/01/12] Introduce <strong>FlashRAG-UI</strong>, an easy to use interface. You can easily and quickly configure and experience the supported RAG methods and evaluate them on the benchmarks.

[25/01/11] We have added support for a new method [<u>RQRAG</u>](https://arxiv.org/abs/2404.00610) method, see it in [**reproduce_experiment**](docs/original_docs/reproduce_experiment.md).

[25/01/07] We have currently support the aggregation of multiple retrievers, see it in [**multi retriever usage**](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/multi_retriever_usage.md).

[25/01/07] We have integrated a very flexible and lightweight corpus chunking library [**Chunkie**](https://github.com/chonkie-ai/chonkie?tab=readme-ov-file#usage), which supports various custom chunking methods (tokens, sentences, semantic, etc.). Use it in [<u>chunking doc corpus</u>](docs/original_docs/chunk-doc-corpus.md).

[24/10/21] We have released a version based on the Paddle framework that supports Chinese hardware platforms. Please refer to [FlashRAG Paddle](https://github.com/RUC-NLPIR/FlashRAG-Paddle) for details.

[24/10/13] A new in-domain dataset and corpus - [DomainRAG](https://arxiv.org/pdf/2406.05654) have been added to the dataset. The dataset is based on the internal enrollment data of Renmin University of China, covering seven types of tasks, which can be used for conducting domain-specific RAG testing.

[24/09/24] We have released a version based on the MindSpore framework that supports Chinese hardware platforms. Please refer to [FlashRAG MindSpore](https://github.com/RUC-NLPIR/FlashRAG-MindSpore) for details.

<details>
<summary>Show more</summary>

[24/09/18] Due to the complexity and limitations of installing Pyserini in certain environments, we have introduced a lightweight `BM25s` package as an alternative (faster and easier to use). The retriever based on Pyserini will be deprecated in future versions. To use retriever with `bm25s`, just set `bm25_backend` to `bm25s` in config.

[24/09/09] We add support for a new method [<u>Adaptive-RAG</u>](https://aclanthology.org/2024.naacl-long.389.pdf), which can automatically select the RAG process to execute based on the type of query. See it result in [<u>result table</u>](#robot-supporting-methods).

[24/08/02] We add support for a new method [<u>Spring</u>](https://arxiv.org/abs/2405.19670), significantly improve the performance of LLM by adding only a few token embeddings. See it result in [<u>result table</u>](#robot-supporting-methods).

[24/07/17] Due to some unknown issues with HuggingFace, our original dataset link has been invalid. We have updated it. Please check the [new link](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/) if you encounter any problems.

[24/07/06] We add support for a new method: [<u>Trace</u>](https://arxiv.org/abs/2406.11460), which refine text by constructing a knowledge graph. See it [<u>results</u>](#robot-supporting-methods) and [<u>details</u>](./docs/original_docs/baseline_details.md).

[24/06/19] We add support for a new method: [<u>IRCoT</u>](https://arxiv.org/abs/2212.10509), and update the [<u>result table</u>](#robot-supporting-methods).

[24/06/15] We provide a [<u>demo</u>](./examples/quick_start/demo_en.py) to perform the RAG process using our toolkit.

[24/06/11] We have integrated `sentence transformers` in the retriever module. Now it's easier to use the retriever without setting pooling methods.

[24/06/05] We have provided detailed document for reproducing existing methods (see [how to reproduce](./docs/original_docs/reproduce_experiment.md), [baseline details](./docs/original_docs/baseline_details.md)), and [<u>configurations settings</u>](./docs/original_docs/configuration.md).

[24/06/02] We have provided an introduction of FlashRAG for beginners, see [<u>an introduction to flashrag</u>](./docs/original_docs/introduction_for_beginners_en.md) ([<u>中文版</u>](./docs/original_docs/introduction_for_beginners_zh.md) [<u>한국어</u>](./docs/original_docs/introduction_for_beginners_kr.md)).

[24/05/31] We supported Openai-series models as generator.

</details>

## :wrench: Installation
![PyPI - Version](https://img.shields.io/pypi/v/flashrag-dev) 
![PyPI - Downloads](https://img.shields.io/pypi/dw/flashrag-dev) 
![PyPI - Downloads](https://img.shields.io/pypi/dm/flashrag-dev)

To get started with FlashRAG, you can simply install it with pip:

```base
pip install flashrag-dev --pre
```

Or you can clone it from Github and install (requires Python 3.10+):

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

If you want to use vllm, sentence-transformers or pyserini, you can install the optional dependencies:

```bash
# Install all extra dependencies
pip install flashrag-dev[full]

# Install vllm for faster speed
pip install vllm>=0.4.1

# Install sentence-transformers
pip install sentence-transformers

# Install pyserini for bm25
pip install pyserini
```

Due to the incompatibility when installing `faiss` using `pip`, it is necessary to use the following conda command for installation.

```bash
# CPU-only version
conda install -c pytorch faiss-cpu=1.8.0

# GPU(+CPU) version
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

Note: It is impossible to install the latest version of `faiss` on certain systems.

From the official Faiss repository ([source](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)):

> - The CPU-only faiss-cpu conda package is currently available on Linux (x86_64 and arm64), OSX (arm64 only), and Windows (x86_64)
> - faiss-gpu, containing both CPU and GPU indices, is available on Linux (x86_64 only) for CUDA 11.4 and 12.1

## :rocket: Quick Start

### Corpus Construction
To build an index, you first need to save your corpus as a `jsonl` file with each line representing a document.

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

If you want to use Wikipedia as your corpus, you can refer to our documentation [Processing Wikipedia](./docs/original_docs/process-wiki.md) to convert it into an indexable format.

### Index Construction

You can use the following code to build your own index.

* For **dense retrieval methods**, especially popular embedding models, we use `faiss` to build the index.

* For **sparse retrieval methods (BM25)**, we use `Pyserini` or `bm25s` to build the corpus into a Lucene inverted index. The built index contains the original documents.

#### For Dense Retrieval Methods

Modify the parameters in the following code to your own.

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method e5 \
  --model_path /model/e5-base-v2/ \
  --corpus_path indexes/sample_corpus.jsonl \
  --save_dir indexes/ \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method mean \
  --faiss_type Flat 
```

* ```--pooling_method```: If this parameter is not specified, we will automatically select it based on the model name and model file. However, since different embedding models use different pooling methods, **we may not have fully implemented them**. To ensure accuracy, you can **specify the pooling method corresponding to the retrieval model you are using** (`mean`, `pooler`, or `cls`).

* ```---instruction```: Some embedding models require additional instructions to be concatenated to the query before encoding, which can be specified here. Currently, we will automatically fill in the instructions for **E5** and **BGE** models, while other models need to be supplemented manually.

If the retrieval model supports the `sentence transformers` library, you can use the following code to build the index (**without considering the pooling method**).

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method e5 \
  --model_path /model/e5-base-v2/ \
  --corpus_path indexes/sample_corpus.jsonl \
  --save_dir indexes/ \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method mean \
  --sentence_transformer \
  --faiss_type Flat 
```

#### For Sparse Retrieval Methods (BM25)

If building a bm25 index, there is no need to specify `model_path`.

##### Building Index with BM25s

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/ 
```

##### Building Index with Pyserini

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend pyserini \
  --save_dir indexes/ 
```

### For Sparse Neural Retrieval Methods (SPLADE)

##### Install Seismic Index:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh # Install Rust for compiling
pip install pyseismic-lsr # Install Seismic
```

##### Then build the index with Seismic:
```bash
python -m flashrag.retriever.index_builder \ # builder
        --retrieval_method splade \ # Model name to trigger seismic index (splade only available)
        --model_path retriever/splade-v3 \ # Local path or repository path are both supported.
        --corpus_embedded_path data/ms_marco/ms_marco_embedded_corpus.jsonl \  # Use cached embedded corpus if corpus is already available in seismic expected format
        --corpus_path data/ms_marco/ms_marco_corpus.jsonl \ # Corpus path in format {id, contents} jsonl file to be embedded if not already built
        --save_dir indexes/ \ # save index directory
        --use_fp16 \ # tell to use fp16 for splade model
        --max_length 512 \ # max tokens for each document
        --batch_size 4 \ # batch size for splade model (4-5 seems the best size for Tesla T4 16GB)
        --n_postings 1000 \ # seismic number of posting lists
        --centroid_fraction 0.2 \ # seismic centroids
        --min_cluster_size 2 \ # seismic min cluster
        --summary_energy 0.4 \ # seismic energy
        --batched_indexing 10000000 # seismic batch
        --nknn 32 # Optional parameter. Tell to seismic to use also knn graph. if not present seismic will work without knn graph
```
### Using the ready-made pipeline

You can use the pipeline class we have already built (as shown in [<u>pipelines</u>](#pipelines)) to implement the RAG process inside. In this case, you just need to configure the config and load the corresponding pipeline.

Firstly, load the entire process's config, which records various hyperparameters required in the RAG process. You can input yaml files as parameters or directly as variables.

Please note that **variables as input take precedence over files**.

```python
from flashrag.config import Config

# hybrid load configs
config_dict = {'data_dir': 'dataset/'}
my_config = Config(
    config_file_path = 'my_config.yaml',
    config_dict = config_dict
```

We provide comprehensive guidance on how to set configurations, you can see our [<u>configuration guidance</u>](./docs/original_docs/configuration.md).
You can also refer to the [<u>basic yaml file</u>](./flashrag/config/basic_config.yaml) we provide to set your own parameters.

Next, load the corresponding dataset and initialize the pipeline. The components in the pipeline will be automatically loaded.

```python
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from flashrag.config import Config

config_dict = {'data_dir': 'dataset/'}
my_config = Config(
    config_file_path = 'my_config.yaml',
    config_dict = config_dict
)
all_split = get_dataset(my_config)
test_data = all_split['test']

pipeline = SequentialPipeline(my_config)
```

You can specify your own input prompt using `PromptTemplete`:

```python
prompt_templete = PromptTemplate(
    config,
    system_prompt = "Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
    user_prompt = "Question: {question}\nAnswer:"
)
pipeline = SequentialPipeline(
  my_config,
  prompt_template = prompt_templete
)
```

Finally, execute `pipeline.run` to obtain the final result.

```python
output_dataset = pipeline.run(test_data, do_eval=True)
```

The `output_dataset` contains the intermediate results and metric scores for each item in the input dataset.
Meanwhile, the dataset with intermediate results and the overall evaluation score will also be saved as a file (if `save_intermediate_data` and `save_metric_score` are specified).

### Build your own pipeline!

Sometimes you may need to implement more complex RAG process, and you can build your own pipeline to implement it.
You just need to inherit `BasicPipeline`, initialize the components you need, and complete the `run` function.

```python
from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever, get_generator

class ToyPipeline(BasicPipeline):
  def __init__(self, config, prompt_templete=None):
    # Load your own components
    pass

  def run(self, dataset, do_eval=True):
    # Complete your own process logic

    # get attribute in dataset using `.`
    input_query = dataset.question
    ...
    # use `update_output` to save intermeidate data
    dataset.update_output("pred",pred_answer_list)
    dataset = self.evaluate(dataset, do_eval=do_eval)
    return dataset
```

Please first understand the input and output forms of the components you need to use from our [<u>documentation</u>](./docs/original_docs/basic_usage.md).

### Just use components

If you already have your own code and only want to use our components to embed the original code, you can refer to the [<u>basic introduction of the components</u>](./docs/original_docs/basic_usage.md) to obtain the input and output formats of each component.

## :gear: Components

In FlashRAG, we have built a series of common RAG components, including retrievers, generators, refiners, and more. Based on these components, we have assembled several pipelines to implement the RAG workflow, while also providing the flexibility to combine these components in custom arrangements to create your own pipeline.

#### RAG-Components

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Module</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1">Judger</td>
      <td>SKR Judger</td>
      <td>Judging whether to retrieve using <a href="https://aclanthology.org/2023.findings-emnlp.691.pdf">SKR</a> method</td>
    </tr>
    <tr>
      <td rowspan="4">Retriever</td>
      <td>Dense Retriever</td>
      <td>Bi-encoder models such as dpr, bge, e5, using faiss for search</td>
    </tr>
    <tr>
      <td>BM25 Retriever</td>
      <td>Sparse retrieval method based on Lucene</td>
    </tr>
    <tr>
      <td>Bi-Encoder Reranker</td>
      <td>Calculate matching score using bi-Encoder</td>
    </tr>
    <tr>
      <td>Cross-Encoder Reranker</td>
      <td>Calculate matching score using cross-encoder</td>
    </tr>
    <tr>
      <td rowspan="5">Refiner</td>
      <td>Extractive Refiner</td>
      <td>Refine input by extracting important context</td>
    </tr>
    <tr>
      <td>Abstractive Refiner</td>
      <td>Refine input through seq2seq model</td>
    </tr>
    <tr>
      <td>LLMLingua Refiner</td>
      <td><a href="https://aclanthology.org/2023.emnlp-main.825/">LLMLingua-series</a> prompt compressor</td>
    </tr>
    <tr>
      <td>SelectiveContext Refiner</td>
      <td><a href="https://arxiv.org/abs/2310.06201">Selective-Context</a> prompt compressor</td>
    </tr>
    <tr>
      <td> KG Refiner </td>
      <td>Use <a hred='https://arxiv.org/abs/2406.11460'>Trace method to construct a knowledge graph</td>
    <tr>
      <td rowspan="4">Generator</td>
      <td>Encoder-Decoder Generator</td>
      <td>Encoder-Decoder model, supporting <a href="https://arxiv.org/abs/2007.01282">Fusion-in-Decoder (FiD)</a></td>
    </tr>
    <tr>
      <td>Decoder-only Generator</td>
      <td>Native transformers implementation</td>
    </tr>
    <tr>
      <td>FastChat Generator</td>
      <td>Accelerate with <a href="https://github.com/lm-sys/FastChat">FastChat</a></td>
    </tr>
    <tr>
      <td>vllm Generator</td>
      <td>Accelerate with <a href="https://github.com/vllm-project/vllm">vllm</a></td>
    </tr>
  </tbody>
</table>

#### Pipelines

Referring to a [<u>survey on retrieval-augmented generation</u>](https://arxiv.org/abs/2312.10997), we categorized RAG methods into four types based on their inference paths.

- **Sequential**: Sequential execuation of RAG process, like Query-(pre-retrieval)-retriever-(post-retrieval)-generator
- **Conditional**: Implements different paths for different types of input queries
- **Branching** : Executes multiple paths in parallel, merging the responses from each path
- **Loop**: Iteratively performs retrieval and generation

In each category, we have implemented corresponding common pipelines. Some pipelines have corresponding work papers.

<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Module</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="1">Sequential</td>
            <td>Sequential Pipeline</td>
            <td>Linear execution of query, supporting refiner, reranker</td>
        </tr>
        <tr>
            <td rowspan="1">Conditional</td>
            <td>Conditional Pipeline</td>
            <td>With a judger module, distinct execution paths for various query types</td>
        </tr>
        <tr>
            <td rowspan="2">Branching</td>
            <td>REPLUG Pipeline</td>
            <td>Generate answer by integrating probabilities in multiple generation paths</td>
        </tr>
          <td>SuRe Pipeline</td>
          <td>Ranking and merging generated results based on each document</td>
        </tr>
        <tr>
            <td rowspan="6">Loop</td>
            <td>Iterative Pipeline</td>
            <td>Alternating retrieval and generation</td>
        </tr>
        <tr>
            <td>Self-Ask Pipeline</td>
            <td>Decompose complex problems into subproblems using <a href="https://arxiv.org/abs/2210.03350">self-ask</a> </td>
        </tr>
        <tr>
            <td>Self-RAG Pipeline</td>
            <td>Adaptive retrieval, critique, and generation</td>
        </tr>
        <tr>
            <td>FLARE Pipeline</td>
            <td>Dynamic retrieval during the generation process</td>
        </tr>
        <tr>
            <td>IRCoT Pipeline</td>
            <td>Integrate retrieval process with CoT</td>
        </tr>
        <tr>
            <td>Reasoning Pipeline</td>
            <td>Reasoning with retrieval</td>
        </tr>
    </tbody>
</table>

## :art: FlashRAG-UI
<p>With <strong>FlashRAG-UI</strong>, you can easily and quickly configure and experience the supported RAG methods through our meticulously designed visual interface, and evaluate these methods on benchmarks, making complex research work more efficient!</p>

### :star2: Features
- **One-Click Configuration Loading**
  - You can load parameters and configuration files for various RAG methods through simple clicks, selections, and inputs.</li>
  - Supports preview interface for intuitive parameter settings.</li>
  - Provides save functionality to easily store configurations for future use.</li>
- **Quick Method Experience**
  - Quickly load corpora and index files to explore the characteristics and application scenarios of various RAG methods.</li>
  - Supports loading and switching different components and hyperparameters, seamlessly connecting different RAG Pipelines to quickly experience their performance and differences!</li>
- **Efficient Benchmark Reproduction**
  - Easily reproduce the built-in baseline methods and carefully collected benchmarks on FlashRAG-UI.</li>
  - Use cutting-edge research tools directly without complex settings, providing a smooth experience for your research work!</li>
  
<details>
<summary>Show more</summary>
<table align="center">
  <tr>
    <td align="center">
      <img src="./asset/demo_en1.jpg" alt="Image 1" width="505"/>
    </td>
    <td align="center">
      <img src="./asset/demo_en2.jpg" alt="Image 2" width="505"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="./asset/demo_en4.png" alt="Image 3" width="500"/>
    </td>
    <td align="center">
      <img src="./asset/demo_en3.jpg" alt="Image 4" width="500"/>
    </td>
  </tr>
</table>
</details>

#### Experience our meticulously designed FlashRAG-UI—both user-friendly and visually appealing:
```bash
cd webui
python interface.py
```

## :robot: Supporting Methods

We have implemented **23 works** with a consistent setting of:

- **Generator:** LLAMA3-8B-instruct with input length of 2048
- **Retriever:** e5-base-v2 as embedding model, retrieve 5 docs per query
- **Prompt:** A consistent default prompt, template can be found in the [<u>method details</u>](./docs/original_docs/baseline_details.md).

For open-source methods, we implemented their processes using our framework. For methods where the author did not provide source code, we will try our best to follow the methods in the original paper for implementation.

For necessary settings and hyperparameters specific to some methods, we have documented them in the **specific settings** column. For more details, please consult our [<u>reproduce guidance</u>](./docs/original_docs/reproduce_experiment.md) and [<u>method details</u>](./docs/original_docs/baseline_details.md).

It's important to note that, to ensure consistency, we have utilized a uniform setting. However, this setting may differ from the original setting of the method, leading to variations in results compared to the original outcomes.

| Method                                                                                    | Type        | NQ (EM) | TriviaQA (EM) | Hotpotqa (F1) | 2Wiki (F1) | PopQA (F1) | WebQA(EM) | Specific setting                                |
| ----------------------------------------------------------------------------------------- | ----------- | ------- | ------------- | ------------- | ---------- | ---------- | --------- | ----------------------------------------------- |
| Naive Generation                                                                          | Sequential  | 22.6    | 55.7          | 28.4          | 33.9       | 21.7       | 18.8      |                                                 |
| Standard RAG                                                                              | Sequential  | 35.1    | 58.9          | 35.3          | 21.0       | 36.7       | 15.7      |                                                 |
| [AAR-contriever-kilt](https://aclanthology.org/2023.acl-long.136.pdf)                     | Sequential  | 30.1    | 56.8          | 33.4          | 19.8       | 36.1       | 16.1      |                                                 |
| [LongLLMLingua](https://arxiv.org/abs/2310.06839)                                         | Sequential  | 32.2    | 59.2          | 37.5          | 25.0       | 38.7       | 17.5      | Compress Ratio=0.5                              |
| [RECOMP-abstractive](https://arxiv.org/pdf/2310.04408)                                    | Sequential  | 33.1    | 56.4          | 37.5          | 32.4       | 39.9       | 20.2      |                                                 |
| [Selective-Context](https://arxiv.org/abs/2310.06201)                                     | Sequential  | 30.5    | 55.6          | 34.4          | 18.5       | 33.5       | 17.3      | Compress Ratio=0.5                              |
| [Trace](https://arxiv.org/abs/2406.11460)                                                 | Sequential  | 30.7    | 50.2          | 34.0          | 15.5       | 37.4       | 19.9      |                                                 |
| [Spring](https://arxiv.org/abs/2405.19670)                                                | Sequential  | 37.9    | 64.6          | 42.6          | 37.3       | 54.8       | 27.7      | Use Llama2-7B-chat with trained embedding table |
| [SuRe](https://arxiv.org/abs/2404.13081)                                                  | Branching   | 37.1    | 53.2          | 33.4          | 20.6       | 48.1       | 24.2      | Use provided prompt                             |
| [REPLUG](https://arxiv.org/abs/2301.12652)                                                | Branching   | 28.9    | 57.7          | 31.2          | 21.1       | 27.8       | 20.2      |                                                 |
| [SKR](https://aclanthology.org/2023.findings-emnlp.691.pdf)                               | Conditional | 33.2    | 56.0          | 32.4          | 23.4       | 31.7       | 17.0      | Use infernece-time training data                |
| [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389.pdf)                          | Conditional | 35.1    | 56.6          | 39.1          | 28.4       | 40.4       | 16.0      |                                                 |
| [Ret-Robust](https://arxiv.org/abs/2310.01558)                                            | Loop        | 42.9    | 68.2          | 35.8          | 43.4       | 57.2       | 33.7      | Use LLAMA2-13B with trained lora                |
| [Self-RAG](https://arxiv.org/abs/2310.11511)                                              | Loop        | 36.4    | 38.2          | 29.6          | 25.1       | 32.7       | 21.9      | Use trained selfrag-llama2-7B                   |
| [FLARE](https://arxiv.org/abs/2305.06983)                                                 | Loop        | 22.5    | 55.8          | 28.0          | 33.9       | 20.7       | 20.2      |                                                 |
| [Iter-Retgen](https://arxiv.org/abs/2305.15294), [ITRG](https://arxiv.org/abs/2310.05149) | Loop        | 36.8    | 60.1          | 38.3          | 21.6       | 37.9       | 18.2      |                                                 |
| [IRCoT](https://aclanthology.org/2023.acl-long.557.pdf)                                   | Loop        | 33.3    | 56.9          | 41.5          | 32.4       | 45.6       | 20.7      |                                                 |
| [RQRAG](https://arxiv.org/abs/2404.00610)                                   | Loop        | 32.6    | 52.5          | 33.5          | 35.8       | 46.4       | 26.2      |  Use trained rqrag-llama2-7B                                               | 

#### 🚀 Reasoning-based Methods (NEW!)

We now support **7 reasoning-based methods** that combine reasoning ability with retrieval, achieving superior performance on complex multi-hop tasks:

| Method                                                                                    | Type        | NQ (EM) | TriviaQA (EM) | PopQA (EM) | Hotpotqa (F1) | 2Wiki (F1) |  Musique (F1) | Bamboogle (F1) | Specific setting                             |
| ----------------------------------------------------------------------------------------- | ----------- | ------- | ------- | ------------- | ------------- | ---------- | ---------- | --------- | ----------------------------------------------- |
| [Search-R1](https://arxiv.org/abs/2503.09516) | Reasoning | 45.2 | 62.2 | 49.2 | 54.5 | 42.6 | 29.2 |  59.9 | SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo |
| [R1-Searcher](https://arxiv.org/pdf/2503.05592) | Reasoning | 36.9 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 | Qwen-2.5-7B-base-RAG-RL |
| [O2-Searcher](https://arxiv.org/pdf/2505.16582) | Reasoning | 41.4 | 51.4 | 46.8 | 43.4 | 48.6 | 19.0 | 47.6 | O2-Searcher-Qwen2.5-3B-GRPO |
| [AutoRefine](https://www.arxiv.org/pdf/2505.11277) | Reasoning | 43.8 | 59.8 | 32.4 | 54.0 | 50.3 | 23.6 | 46.6 | AutoRefine-Qwen2.5-3B-Base |
| [ReaRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 26.3 | 51.8 | 24.6 | 42.9 | 41.6 | 21.2 | 41.9 | ReaRAG-9B |
| [CoRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 40.9 | 63.1 | 36.0 | 56.6 | 60.7 | 31.9 | 54.1 | CoRAG-Llama3.1-8B-MultihopQA |
| [SimpleDeepSearcher](https://arxiv.org/pdf/2505.16834) | Reasoning | 36.1 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 | Qwen-7B-SimpleDeepSearcher |

## :notebook: Supporting Datasets & Document Corpus

### Datasets

We have collected and processed 36 datasets widely used in RAG research, pre-processing them to ensure a consistent format for ease of use. For certain datasets (such as Wiki-asp), we have adapted them to fit the requirements of RAG tasks according to the methods commonly used within the community. All datasets are available at [<u>Huggingface datasets</u>](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

For each dataset, we save each split as a `jsonl` file, and each line is a dict as follows:

```python
{
  'id': str,
  'question': str,
  'golden_answers': List[str],
  'golden_doc_ids': List[str],  # optional, for document-level retrieval metrics
  'metadata': dict
}
```

When `golden_doc_ids` is available, FlashRAG can evaluate retrieved source documents with `retrieval_doc_recall`, `retrieval_doc_precision`, `retrieval_doc_f1`, and `retrieval_doc_map`.

Below is the list of datasets along with the corresponding sample sizes:

| Task                      | Dataset Name    | Knowledge Source | # Train   | # Dev   | # Test |
| ------------------------- | --------------- | ---------------- | --------- | ------- | ------ |
| QA                        | NQ              | wiki             | 79,168    | 8,757   | 3,610  |
| QA                        | TriviaQA        | wiki & web       | 78,785    | 8,837   | 11,313 |
| QA                        | PopQA           | wiki             | /         | /       | 14,267 |
| QA                        | SQuAD           | wiki             | 87,599    | 10,570  | /      |
| QA                        | MSMARCO-QA      | web              | 808,731   | 101,093 | /      |
| QA                        | NarrativeQA     | books and story  | 32,747    | 3,461   | 10,557 |
| QA                        | WikiQA          | wiki             | 20,360    | 2,733   | 6,165  |
| QA                        | WebQuestions    | Google Freebase  | 3,778     | /       | 2,032  |
| QA                        | AmbigQA         | wiki             | 10,036    | 2,002   | /      |
| QA                        | SIQA            | -                | 33,410    | 1,954   | /      |
| QA                        | CommonSenseQA   | -                | 9,741     | 1,221   | /      |
| QA                        | BoolQ           | wiki             | 9,427     | 3,270   | /      |
| QA                        | PIQA            | -                | 16,113    | 1,838   | /      |
| QA                        | Fermi           | wiki             | 8,000     | 1,000   | 1,000  |
| multi-hop QA              | HotpotQA        | wiki             | 90,447    | 7,405   | /      |
| multi-hop QA              | 2WikiMultiHopQA | wiki             | 15,000    | 12,576  | /      |
| multi-hop QA              | Musique         | wiki             | 19,938    | 2,417   | /      |
| multi-hop QA              | Bamboogle       | wiki             | /         | /       | 125    |
| multi-hop QA              | StrategyQA      | wiki             | 2290      | /       | /      |
| Long-form QA              | ASQA            | wiki             | 4,353     | 948     | /      |
| Long-form QA              | ELI5            | Reddit           | 272,634   | 1,507   | /      |
| Long-form QA              | WikiPassageQA   | wiki             | 3,332     | 417     | 416    |
| Open-Domain Summarization | WikiASP         | wiki             | 300,636   | 37,046  | 37,368 |
| multiple-choice           | MMLU            | -                | 99,842    | 1,531   | 14,042 |
| multiple-choice           | TruthfulQA      | wiki             | /         | 817     | /      |
| multiple-choice           | HellaSWAG       | ActivityNet      | 39,905    | 10,042  | /      |
| multiple-choice           | ARC             | -                | 3,370     | 869     | 3,548  |
| multiple-choice           | OpenBookQA      | -                | 4,957     | 500     | 500    |
| multiple-choice           | QuaRTz          | -                | 2696      | 384     | 784    |
| Fact Verification         | FEVER           | wiki             | 104,966   | 10,444  | /      |
| Dialog Generation         | WOW             | wiki             | 63,734    | 3,054   | /      |
| Entity Linking            | AIDA CoNll-yago | Freebase & wiki  | 18,395    | 4,784   | /      |
| Entity Linking            | WNED            | Wiki             | /         | 8,995   | /      |
| Slot Filling              | T-REx           | DBPedia          | 2,284,168 | 5,000   | /      |
| Slot Filling              | Zero-shot RE    | wiki             | 147,909   | 3,724   | /      |
| In-domain QA              | DomainRAG       | Web pages of RUC | /         | /       | 485    |

### Document Corpus

Our toolkit supports jsonl format for retrieval document collections, with the following structure:

```jsonl
{"id":"0", "contents": "..."}
{"id":"1", "contents": "..."}
```

The `contents` key is essential for building the index. For documents that include both text and title, we recommend setting the value of `contents` to `{title}\n{text}`. The corpus file can also contain other keys to record additional characteristics of the documents.

In the academic research, Wikipedia and MS MARCO are the most commonly used retrieval document collections. For Wikipedia, we provide a [<u>comprehensive script</u>](./docs/original_docs/process-wiki.md) to process any Wikipedia dump into a clean corpus. Additionally, various processed versions of the Wikipedia corpus are available in many works, and we have listed some reference links.

For MS MARCO, it is already processed upon release and can be directly downloaded from its [<u>hosting link</u>](https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus) on Hugging Face.

### Index

To facilitate easier replication of the experiments, we now provide a preprocessed index available in the ModelScope dataset page: [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

The index was created using the e5-base-v2 retriever on our uploaded wiki18_100w dataset, which is consistent with the index used in our experiments.

## :lollipop: Awesome Work using FlashRAG

- [R1-Searcher](https://github.com/SsmallSong/R1-Searcher), a method that incentivizes the search capability in LLMs via reinforcement learning
- [ReSearch](https://github.com/Agent-RL/ReSearch), a method that learns to reason with search for LLMs via reinforcement learning
- [AutoCoA](https://github.com/ADaM-BJTU/AutoCoA), a method that internalizes chain-of-action generation into reasoning models

## :raised_hands: Additional FAQs

- [How should I set different experimental parameters?](./docs/original_docs/configuration.md)
- [How to build my own corpus, such as a specific segmented Wikipedia?](./docs/original_docs/process-wiki.md)
- [How to index my own corpus?](./docs/original_docs/building-index.md)
- [How to reproduce supporting methods?](./docs/original_docs/reproduce_experiment.md)
- [How can I debug common RAG failure modes when using FlashRAG?](./docs/rag_failure_modes_and_debug_checklist.md)

## :bookmark: License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).

## :star2: Citation

Please kindly cite our paper if helps your research:

```BibTex
@inproceedings{FlashRAG,
  author       = {Jiajie Jin and
                  Yutao Zhu and
                  Zhicheng Dou and
                  Guanting Dong and
                  Xinyu Yang and
                  Chenghao Zhang and
                  Tong Zhao and
                  Zhao Yang and
                  Ji{-}Rong Wen},
  editor       = {Guodong Long and
                  Michale Blumestein and
                  Yi Chang and
                  Liane Lewin{-}Eytan and
                  Zi Helen Huang and
                  Elad Yom{-}Tov},
  title        = {FlashRAG: {A} Modular Toolkit for Efficient Retrieval-Augmented Generation
                  Research},
  booktitle    = {Companion Proceedings of the {ACM} on Web Conference 2025, {WWW} 2025,
                  Sydney, NSW, Australia, 28 April 2025 - 2 May 2025},
  pages        = {737--740},
  publisher    = {{ACM}},
  year         = {2025},
  url          = {https://doi.org/10.1145/3701716.3715313},
  doi          = {10.1145/3701716.3715313}
}
```


## Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=RUC-NLPIR/FlashRAG&type=Date)](https://star-history.dera.page/#RUC-NLPIR/FlashRAG&Date)
