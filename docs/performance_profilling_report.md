# Lightweight Adaptive Top-K Retrieval in FlashRAG: Evaluation and Performance Report

Updated: November 13, 2026 (America/Chicago).

This is the single project report. It is organized by five fixed optimization IDs so each change, experiment, negative result, and trade-off can be traced consistently. The objective is to reduce RAG execution cost while preserving answer quality. Raw evidence remains under `docs/experiments/`.

<!-- SUBMISSION_START -->
# Lightweight Adaptive Top-K Retrieval in FlashRAG

Yidong Zhang and Yuang Cai | CS598APE final project | November 13, 2026

## Abstract

We implement and evaluate training-free Adaptive Top-K retrieval using similarity thresholds and adjacent score gaps in FlashRAG. The goal is to reduce context-processing cost while maintaining answer quality. Early aggressive pruning reduced context but lost correct answers. We therefore introduce minimum-k and maximum-k guards, a weak-similarity fallback, and development-only calibration that preserves every observed fixed-5 answer-string hit. We exclude all earlier questions, calibrate on 200 development questions, freeze two policies, and evaluate six methods on 500 new questions with three timing repetitions each. The final threshold policy reduces documents by 5.64% and input tokens by 4.67% versus fixed-5, with observed EM changing from 2.6% to 2.8% (one additional exact match). A stable latency improvement is not demonstrated, and fixed-3 remains a stronger EM/F1 operating point. The report separates Adaptive-K findings from supporting generation-system speedups and reports negative results. The experiments use a limited 15,000-document general-knowledge corpus; answer-hit measures are lexical proxies, and full-Wikipedia benchmark performance is not claimed.

## 1. Problem, scope, and related work

A fixed retrieval budget can provide unnecessary or insufficient context for individual questions. We study whether scores already produced by a dense retriever provide a cheap signal for selecting a document prefix. The primary contribution is a modular implementation and reproducible cost/quality evaluation of two lightweight selection rules; no additional selector model is trained and inference makes no LLM-feedback calls.

RAG combines external retrieval with generation [1]. Self-RAG learns retrieval and reflection behavior [2]; Adaptive-RAG learns to select retrieval strategies based on question complexity [3]; DynamicRAG learns dynamic reranking using generation-quality feedback [4]. Our scope is narrower: deterministic, score-only prefix selection within the FlashRAG toolkit [5]. We do not reproduce these systems, train comparable models, claim algorithmic novelty for standard threshold/gap heuristics, or claim state-of-the-art results.

The implementation first retrieves a bounded candidate pool and then chooses the number of documents sent to the generator. It does not terminate the FAISS distance scan early. Consequently, context reduction and candidate-search acceleration are separate hypotheses. A successful system-throughput optimization does not establish that the adaptive selection rule is better.

## 2. Implementation and optimization decisions

### Optimization 1: Lightweight Adaptive Top-K selection (primary)

For score-gap selection, let ranked similarity scores be s1 >= s2 >= ... >= sM. Compute adjacent gaps g_i = s_i - s_(i+1), consider boundaries between min_k and M-1, and cut after the largest gap if it reaches a calibrated threshold. Otherwise retain fallback_k. The frozen final policy uses min_k=2, max_k=5, fallback_k=5, and gap threshold=0.03. Ties choose the earliest allowed boundary. Both score directions are supported and tested, although performance experiments use inner-product similarities.

The similarity-threshold rule counts the ranked candidates meeting an absolute threshold, clamps the result to the minimum/maximum budget, and uses fallback_k if no score qualifies. The weak-query fallback avoids interpreting low similarity as evidence that very little context is sufficient. Its frozen threshold is 0.717440844, with min_k=2, max_k=5, and fallback_k=5. Scores are not calibrated correctness probabilities; this numeric threshold is specific to the E5/index configuration.

Both rules validate finite scores, ranking direction, and k bounds. Empty retrieval yields k=0; short result lists retain available documents. Runtime selection receives scores only, with no access to gold answers. Selection is O(M) per query with small bounded M. Both policies plug into SequentialPipeline through the same retrieval_policy interface and record the selected k and reason. The final max_k=5 cap avoids expanding beyond the fixed-5 context budget; fixed-10 remains a separate reference.

The earlier score-gap policy (min_k=1, max_k=10, gap=0.01) selected k=1 for 139 of 500 reused questions. It lost four fixed-5 exact matches with no gains; three losses also removed answer-string coverage. That diagnosis motivated guarded policies and new disjoint data. It is historical evidence, not a held-out result for the revised rules.

### Optimizations 2-5: supporting execution work

| ID | Change | Measured result and limitation |
| --- | --- | --- |
| Optimization 2 | Skip unneeded token-score collection | 10.58 MiB lower peak allocation on 100 prompts; no demonstrated latency benefit. These are generation scores, not retrieval similarities. |
| Optimization 3 | Retain CUDA allocator cache; CPU-thread screening | Device allocations 339 to 2, but latency benefit inconclusive. Intra-op thread counts 1/2/4 did not improve the 100-prompt pilot. |
| Optimization 4 | Batched generation and continuation-only decoding | Earlier same-session batch 1 to 8: 2.76x throughput on 100 prompts. Later sorted batch 8 to 32: 2.71x on 500 prompts; combined path 2.84x, with more GPU memory and changed texts. Fixed-5 EM regressed at batch 32. |
| Optimization 5 | Length-aware batching and token reuse | Paired 500-prompt generation time fell 11.46%, while peak allocation rose 56.68%. Token reuse reduced local preparation overhead; an additional robust total-time gain was not established. |

These experiments use different baselines and workloads, so their speedups must not be multiplied or attributed to Adaptive-K. All settings in the final core study below use the same conservative batch-8 configuration; batch 32, length sorting, token reuse, and continuation-only decoding are not enabled there.

## 3. Experimental protocol

### Data separation and calibration

The source is the pinned FlashRAG NQ JSONL, revision `bcafb8dd07d453be3cbeeeb3f78be1841bddf92c`, containing 3,610 questions. We exclude the prior 517 unique questions, sample using seed 598, and allocate 200 development plus 500 evaluation questions with no normalized-question or ID overlap. These are project-specific splits drawn from the official test JSONL, not the official NQ development split. IDs and SHA-256 values are preserved in split_manifest.json.

Calibration considers 81 configurations across the two policy families. The criterion is zero lost individual development answer-string hits relative to fixed-5, then minimum mean k; ties prefer a larger minimum budget and canonical configuration order. It uses development retrieval results and gold aliases only. No LLM answer or final-test answer is used to select the policies. The chosen rules are frozen before development generation and final evaluation. Calibration retains 24/200 development hits; mean k becomes 4.945 for score-gap and 4.770 for threshold.

The frozen primary answer-quality check compares observed EM and F1 against fixed-5; all other metrics and methods are reported regardless of outcome. No policy is retuned or selected using the final results. Maintaining development lexical coverage does not guarantee answer quality.

### Resources and hardware

| Component | Recorded configuration |
| --- | --- |
| Host | Windows build 26200; AMD Ryzen 7 7800X3D, 8 cores / 16 logical CPUs; about 63.15 GiB usable RAM |
| GPU | NVIDIA RTX 4070 SUPER, 12 GiB VRAM; CUDA PyTorch 2.11.0+cu128 |
| Software | Python 3.11.9; Transformers 5.7.0; FAISS CPU 1.15.1 |
| Retriever | E5-base-v2, float32; 128-token query limit; CPU inner-product Flat index |
| Generator | Local Qwen2.5-0.5B-Instruct, bfloat16; greedy generation; maximum 32 output tokens; 1,024-token input limit |
| Common execution | Batch 8; original query order; eight CPU intra-op/inter-op threads; score collection off; historical per-batch CUDA cache clearing |
| Corpus/index | 15,000 local general-knowledge documents and 15,000 vectors, dimension 768 |

Corpus/index counts match. Sixteen deterministically sampled passages were re-encoded, and corresponding reconstructed vectors agree to cosine greater than 0.99. This verifies sampled alignment, not every vector. Only 74/200 development questions have an answer-string hit anywhere in the corpus. The corpus is not the NQ Wikipedia collection, which limits retrieval coverage and explains why these measurements must not be compared directly with published NQ leaderboard scores.

### Timing and metrics

Each method receives an eight-question warmup outside timing. All six methods run for three measured repetitions, with method order rotated and reversed according to a fixed schedule. CUDA synchronization encloses the complete inference interval. Search, selection, prompt construction, generation, process CPU time, and peak allocated/reserved GPU memory are recorded. Evaluation and token counting run outside inference timing. No profiler or forward hooks are active in final timed runs. Memory figures include both loaded neural models; host peak RSS and minimum hardware requirements were not measured.

Answer quality uses repository-normalized EM, token F1, and answer-containing accuracy. Retrieval answer hit tests whether a nonempty normalized gold alias occurs at token boundaries in any selected document. This is a lexical proxy; gold document relevance labels are unavailable. Context tokens use the actual generator tokenizer after its 1,024-token truncation. Padded input/output slots are separately recorded and are not useful-token counts. Throughput is total questions divided by median batch completion time, not inverse single-request latency.

There are 9,000 final generations but only 500 independent final questions. Paired quality intervals use 2,000 question bootstrap resamples, seed 598. They describe uncertainty on these questions, not timing variance or formal equivalence; sparse binary outcomes and degenerate zero-change intervals require caution.

## 4. Final Adaptive Top-K results

### Quality and selected context

| Method | Mean k | Answer hit (%) | EM (%) | F1 (%) | Accuracy (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed k=1 | 1.000 | 5.20 | 2.80 | 7.878 | 9.20 |
| Fixed k=3 | 3.000 | 9.00 | 4.00 | 8.662 | 9.40 |
| Fixed k=5 | 5.000 | 10.20 | 2.60 | 7.661 | 9.40 |
| Fixed k=10 | 10.000 | 13.80 | 2.20 | 7.101 | 10.60 |
| Adaptive score-gap | 4.934 | 10.20 | 2.60 | 7.729 | 9.20 |
| Adaptive threshold | 4.718 | 10.00 | 2.80 | 7.773 | 9.60 |

### Execution cost

| Method | Input tokens/query | Median time (s) | Throughput (q/s) | Peak allocated (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Fixed k=1 | 138.280 | 43.2736 | 11.554 | 1644.28 |
| Fixed k=3 | 308.764 | 45.1138 | 11.083 | 1807.08 |
| Fixed k=5 | 474.266 | 47.0550 | 10.626 | 1848.76 |
| Fixed k=10 | 785.050 | 58.7443 | 8.511 | 2071.94 |
| Adaptive score-gap | 468.782 | 46.4558 | 10.763 | 1852.94 |
| Adaptive threshold | 452.110 | 47.1255 | 10.610 | 1850.60 |

| Method | Repetition 1 / 2 / 3 time (s) | Median CPU (s) |
| --- | --- | ---: |
| Fixed k=1 | 43.9032 / 42.9656 / 43.2736 | 39.3594 |
| Fixed k=3 | 45.0859 / 46.2251 / 45.1138 | 42.8750 |
| Fixed k=5 | 46.7541 / 48.3374 / 47.0550 | 44.5625 |
| Fixed k=10 | 58.9840 / 58.6191 / 58.7443 | 55.2812 |
| Adaptive score-gap | 46.4299 / 48.4614 / 46.4558 | 44.7656 |
| Adaptive threshold | 47.1938 / 47.1255 / 46.5411 | 44.5781 |

Stage medians are computed independently and need not sum to the median total:

| Method | Retrieval (s) | Selection (s) | Prompt (s) | Generation (s) |
| --- | ---: | ---: | ---: | ---: |
| Fixed k=1 | 0.181072 | 0.000000 | 0.321620 | 42.774261 |
| Fixed k=3 | 0.212908 | 0.000000 | 0.653869 | 44.289423 |
| Fixed k=5 | 0.236085 | 0.000000 | 0.928039 | 45.861117 |
| Fixed k=10 | 0.291331 | 0.000000 | 1.665465 | 56.796666 |
| Adaptive score-gap | 0.240140 | 0.002437 | 0.908750 | 45.282998 |
| Adaptive threshold | 0.227514 | 0.002303 | 0.891770 | 45.994089 |

### Paired interpretation and adoption

**Adaptive score-gap:** mean documents fall 1.32% and truncated input tokens fall 1.16% versus fixed-5. Observed EM and F1 meet the frozen quality criterion. All three answer metrics are not all preserved; inspect the accuracy/quality differences above. The measured median time change is +1.27% reduction. Run ranges overlap, so a robust end-to-end latency benefit is not established.

**Adaptive threshold:** mean documents fall 5.64% and truncated input tokens fall 4.67% versus fixed-5. Observed EM and F1 meet the frozen quality criterion. All three answer metrics are at least the fixed-5 values. The measured median time change is -0.15% reduction. Run ranges overlap, so a robust end-to-end latency benefit is not established.

For Adaptive threshold, lexical retrieval answer hit drops from 10.20% to 10.00% on final questions. The zero-loss calibration constraint held on development data and did not guarantee zero loss on unseen data.

| Policy vs fixed-5 | EM gains / losses | F1 change (pp) | Paired F1 95% interval (pp) |
| --- | ---: | ---: | --- |
| Adaptive score-gap | 0 / 0 | +0.0686 | [-0.0057, +0.2002] |
| Adaptive threshold | 1 / 0 | +0.1123 | [-0.3117, +0.5873] |

Fixed k=3 must be retained in the interpretation: its observed EM is 4.00% and F1 is 8.662%, compared with fixed-5 EM 2.60% and F1 7.661%. Reducing context relative to fixed-5 does not establish that an adaptive method dominates every fixed-k operating point. The tables report the complete trade-off rather than choosing only a favorable comparator.

The contribution is a working score-only adaptive implementation, guarded development calibration, and a fully reported evaluation. Candidate-search acceleration and general quality-preserving latency improvements are not presumed from context reduction. Existing defaults remain unchanged, and calibrated policies are explicit experiment configurations rather than universal replacements for fixed-k.

## 5. CPU/GPU characterization and remaining bottlenecks

Earlier profiling established that generation dominates full RAG time; score selection accounts for milliseconds per 500 questions. Controlled CPU/CUDA traces showed many small launches and host-side generation activity. A fixed-eight-output-token diagnostic reduced kernel count from 39,116 at sorted batch 8 to 10,115 at batch 32, but kernel duration increased from 467.403 to 813.181 ms because padding increased. A lower wall time does not imply less GPU computation. Synchronization call counts also fell while host synchronization intervals increased. These separate diagnostic captures support launch/batching hypotheses, not exact attribution of unprofiled runtime or SM occupancy.

Windows counters on a separate 500-question workload measured median process CPU 38.97 s at batch 8 and 15.02 s at batch 32, concentrated in the main thread. A separate 32-prompt cProfile run with GetThreadTimes identified linear/attention entry points, generation-loop completion checks, and module dispatch. CPU time at a CUDA entry point includes host/driver behavior, not GPU kernel execution; busy waiting can consume CPU. Cumulative function times overlap. No RMSNorm/RoPE fusion, OpenMP C++ implementation, or -O3 compilation experiment was performed.

WPR native CPU-stack capture was attempted normally and through tool escalation, but Windows rejected the profiling privilege with error 0xc5585011. A tested administrator-only capture entry point is supplied; actual WPR ETL capture and native wait-cause analysis remain incomplete. This restriction is stated rather than substituting Python profiling for a successful WPR capture.

## 6. Testing, artifact layout, and reproduction

The complete local discovery suite passed **56 tests**, with no skipped or failed tests. It covers score direction, threshold inclusivity, bounds, ties, empty/short inputs, nonfinite and unsorted scores, policy configuration, pipeline integration, FAISS result handling/cache behavior, document metrics, dataset handling, generator scores/text/dictionaries, batching order and padding, CPU trace interval accounting, WPR session ownership/cleanup, development/evaluation separation, and calibration safeguards. Small real Hugging Face generation tests run on CPU without downloading weights. The initial two warning-test errors were caused by unittest scanning lazy optional vision imports; the test now captures and checks the actual retrieval warning without importing unrelated vision modules. The initial failure log and final passing log are retained.

Larger-program validation checks all six methods, all three final repetitions, all 500 IDs in original order, selected-document counts and adaptive prefixes against fixed-5, frozen hashes, and repeated prediction stability. Prediction disagreement counts are {"fixed_1": 0, "fixed_10": 0, "fixed_3": 0, "fixed_5": 0, "score_gap": 0, "similarity_threshold": 0}. Development generation adds 1,200 calls across six methods. This is 10,200 new generation evaluations on 700 distinct questions; it is not a claim of 10,200 independent examples.

The code archive was also extracted into a fresh temporary directory and the same 56 tests passed without downloaded model weights or the large resource archive. This check used the existing virtual environment's dependencies; it is not a claim of a clean operating-system installation. Archive integrity and per-file checksums are recorded separately.

No C/C++ source was authored for this project. **Valgrind: not applicable to authored Python code; not run.** PyTorch/FAISS contain third-party native implementations, for which no Valgrind-clean claim is made. Optional UI, training, multimodal, vLLM, multi-GPU, and non-Windows final workloads are outside tested scope.

| Artifact path | Purpose |
| --- | --- |
| flashrag/retriever/adaptive_topk.py | Both documented score-only policies and configuration factory |
| flashrag/pipeline/pipeline.py | Generic integration, score/decision recording, stage timing |
| flashrag/generator/generator.py | Supporting opt-in generation optimizations |
| scripts/run_final_topk_study.py | Split preparation, resource audit, calibration, freezing, rotated final experiments, summaries |
| tests/test_*.py | Complete local regression suite |
| docs/experiments/final_topk/ | All calibration, manifests, frozen configurations, 18 final measurements, paired summaries and test/verification logs |
| docs/performance_profilling_report.md | Single English source, including submission narrative and all historical optimization details |
| docs/performance_profilling_report.pdf | Exported submission narrative |
| scripts/export_project_report.py | Deterministic PDF layout/export from the source submission block |
| README.md | Installation, hardware/resource requirements, exact commands and upstream documentation |

Install Python 3.11 and the CUDA build of torch 2.11.0, then requirements-baseline.txt. Model weights are excluded from the archive; download the pinned Qwen model using scripts/download_qwen.py and provide the recorded E5-base-v2 snapshot. The matching corpus/index and exact split files accompany the artifact. The tested 12-GiB GPU is a reproducible hardware configuration, not a measured minimum. A compatible CPU PyTorch environment is sufficient for unit tests, while the final benchmark harness requires CUDA.

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_CACHE = Join-Path (Get-Location).Path 'models/datasets-cache'
$env:TIKTOKEN_CACHE_DIR = Join-Path (Get-Location).Path 'models/tiktoken-cache'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/run_final_topk_study.py --stage all --output-dir docs/experiments/reproduction
.\.venv\Scripts\python.exe scripts/export_project_report.py
```

Use a new output directory for independent reproduction and the same resources/settings for a checkpointed resume. Resource and code hashes in metadata.json identify the executed version. Dependencies and complete setup instructions appear in README.md and docs/environment-lock.txt.

## 7. Limitations and project completion status

Implementation, integration, testing, development calibration, fresh-question fixed/adaptive comparisons, profiling evidence, documentation, and PDF export are delivered. Outcomes are empirical: an implementation can be complete while its efficiency hypothesis remains partly unsupported. The experiments do not establish universal latency improvement, semantic quality equivalence, or faster FAISS distance scans. Low corpus answer coverage, lexical relevance proxies, input truncation, natural variable output lengths, one small model/GPU, three repetitions, other desktop applications, unlocked clocks, and lack of independent deployment traffic limit generalization. CPU peak memory and minimum hardware were not measured. WPR native stacks remain blocked by administrator privileges.

The final policies are specific to the recorded retriever and corpus; absolute thresholds need recalibration after resource changes. A new corpus or policy tuned after inspecting final results requires another unused evaluation set. Future work should prioritize better corpus coverage, token-budget-aware selection calibrated on development data, and comparison with strong fixed-k frontiers before additional model-internal kernel optimization.

## 8. Conclusion

We deliver a modular, tested implementation of similarity-threshold and score-gap Adaptive Top-K with guarded selection and reproducible development/final data separation. The final tables quantify context, quality, latency, throughput, and memory together. Supporting system experiments demonstrate substantial generation-throughput improvements, while the Adaptive-K results must be interpreted according to their own smaller context/quality trade-offs. The report retains unfavorable outcomes and explicitly identifies unachieved or unmeasured objectives.

## References

[1] P. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS, 2020. [Paper](https://papers.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).

[2] A. Asai et al. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR, 2024. [Paper](https://arxiv.org/abs/2310.11511).

[3] S. Jeong et al. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. NAACL, 2024. [Paper](https://aclanthology.org/2024.naacl-long.389/).

[4] J. Sun et al. DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation. arXiv:2505.07233, 2025. [Paper](https://arxiv.org/abs/2505.07233).

[5] RUC-NLPIR. FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research. [Toolkit](https://github.com/RUC-NLPIR/FlashRAG); [paper](https://arxiv.org/abs/2405.13576); [dataset](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).
<!-- SUBMISSION_END -->

## Project objective and alignment with the proposal

The primary research objective is a lightweight, training-free Adaptive Top-K policy driven by retrieval similarity scores and score gaps, integrated into FlashRAG. Success must be evaluated as a trade-off between retrieval/context-processing cost and retrieval/answer quality. No additional learned selector or LLM-feedback reranker is required by the proposed method.

Optimization 1 is the core method contribution. Optimizations 2-5 and CPU/GPU profiling are supporting execution work: their measured batching, memory, tokenization, and decoding gains must not be attributed to Adaptive-K. The five optimization IDs are retained for traceability, but they do not represent five independent Top-K selection algorithms. No RMSNorm/RoPE fusion or new model-kernel optimization has been implemented.

| Proposal commitment | Verified local status | Outstanding evidence or work |
| --- | --- | --- |
| Fixed Top-K baselines with several k values | Completed fixed k=1/3/5/10 on 500 new questions, three repetitions each | Broader corpus/model/hardware generalization remains untested |
| Score-gap Adaptive-K | Implemented in `flashrag/retriever/adaptive_topk.py`, integrated into `SequentialPipeline`, and boundary-tested | A selection rule with a demonstrated favorable cost/quality trade-off |
| Similarity-threshold Adaptive-K | Implemented, boundary-tested, development-calibrated, and evaluated on 500 new questions | Thresholds remain resource-specific; see the final submission results above |
| Larger benchmark evaluation | 500 NQ questions evaluated with repeated timing and answer metrics | Corpus/index coverage checks and explicit retrieval-quality measurements; this uses a 15,000-document general-knowledge corpus, not full-Wikipedia NQ |
| Preserve answer quality while reducing context overhead | Final threshold rule reduces tokens with all three observed answer metrics at least fixed-5 | Lexical retrieval hit falls by one question; quality equivalence and latency improvement are not established |
| Reduce retrieval overhead | Final guarded policies request five candidates before pruning; earlier policy requested ten | Early termination of the Flat-index scan and reduced distance-computation cost are not established |
| Testing and final artifacts | 56-test suite passed; final experiment verified; consolidated source and PDF supplied | WPR native-stack capture still requires administrator execution |

The current design implements adaptive context selection over a fixed candidate pool. This is within the proposed score-based selection direction, but does not by itself achieve every stated efficiency goal. In particular, reducing the number of passages sent to the generator is distinct from reducing candidate-search work.

The final core experiment above now establishes corpus/candidate lexical answer coverage; compares fixed k=1/3/5/10, similarity-threshold and score-gap policies under identical generator/batching settings; calibrates parameters on separate development data; and evaluates frozen rules on 500 unused questions. Report candidate count, selected k distribution, retrieval-quality proxy definitions, context tokens, selection/search/prompt/generation times, throughput, memory, and EM/F1/accuracy. Any quality tolerance must be declared before choosing the winning rule. Reused test-set diagnostics below are not a fresh held-out evaluation.

The recent general-purpose generation optimizations remain useful supporting results. The final project priority was Adaptive Top-K behavior and its cost/quality evidence, now reported above; model-internal kernel fusion was not performed.

## Optimization overview

| ID | Direction | Main measured outcome | Status |
| --- | --- | --- | --- |
| [Optimization 1](#optimization-1-score-gap-adaptive-k-retrieval) | Score-gap adaptive-k retrieval | Documents -29.92%, input tokens -25.67%; batch-size-1 latency +4.79%, EM 3.0% to 2.2% | Context reduction; speed/quality objective not achieved |
| [Optimization 2](#optimization-2-skip-unneeded-generation-score-collection) | Skip unneeded generation scores | Peak allocation -10.58 MiB on 100 prompts; no demonstrated latency benefit | Memory benefit |
| [Optimization 3](#optimization-3-retain-the-cuda-allocator-cache) | Retain CUDA allocator cache | Device allocations 339 to 2; small 2.07% median time change with overlapping run ranges | Latency benefit inconclusive |
| [Optimization 4](#optimization-4-batched-qwen-generation) | Batch generation and continuation decoding | Earlier batch 1 to 8: 2.76x on 100 prompts; later sorted batch 8 to 32: 2.71x on 500 prompts (2.84x combined) | Clear offline throughput benefit; memory/text changes and fixed-top-5 quality regression at batch 32 |
| [Optimization 5](#optimization-5-length-aware-batching) | Group similar input lengths at batch 8 | Paired 500-prompt generation time -11.46%, padded slots -53.36%; peak allocation +56.68% | Speed/memory trade-off; opt-in |

The five directions have different outcomes and use different baselines. Speedup ratios must not be multiplied. Setup, bug fixes, profiling, and larger-dataset testing are shared supporting work rather than additional optimization IDs. CPU intra-op thread screening is recorded as a follow-up under Optimization 3. Native compiler optimization remains unexecuted.

Every optimization uses the same structure: motivation, implementation, experiment design, results, correctness/limitations, and artifacts/reproduction. Shared environment, tests, historical records, and commands follow the five main sections.

## Optimization 1: Score-Gap Adaptive-K Retrieval

### Motivation

A fixed document count can provide unnecessary context for some questions. This direction tests whether a score-based per-question cutoff can reduce context and runtime while preserving answers.

### Implementation

Implementation: `flashrag/retriever/adaptive_topk.py`, class `ScoreGapPolicy`.

```text
Question -> E5 encoding -> FAISS top-10 candidates and scores
         -> per-question prefix selection -> prompt -> one Qwen generation
```

The default policy uses min_k=1, max_k=10, fallback_k=5, and gap_threshold=0.05. Experiments also used gap_threshold=0.01.

1. Compute adjacent score drops in ranked candidates.
2. Find the largest drop within the permitted prefix boundaries.
3. If the drop reaches the threshold, retain the documents before that boundary.
4. Otherwise retain fallback_k, capped by the number of available candidates.

For example, scores `[0.95, 0.94, 0.70, 0.69]` select k=2. Equal largest gaps select the earlier boundary. Insufficient candidates are retained as available; empty retrieval produces k=0. The implementation validates k bounds, finite thresholds/scores, score direction, and ranking order. It supports both higher-is-better similarity and lower-is-better distance scores.

The rule uses retrieval scores only, without gold answers or multiple trial generations. A score drop is a heuristic, not a calibrated probability that enough evidence has been retrieved. Candidate search still requests a fixed upper limit; the adaptive step changes the context sent to generation.

`SequentialPipeline` accepts an optional `retrieval_policy`. Without it, the existing fixed-k call remains available. Adaptive runs record candidate scores, selected_k, decision reasons, and a separate selection_s timer, included in end-to-end latency.

```python
from flashrag.pipeline import SequentialPipeline
from flashrag.retriever.adaptive_topk import ScoreGapPolicy

policy = ScoreGapPolicy(max_k=10, fallback_k=5, gap_threshold=0.01)
pipeline = SequentialPipeline(config, retrieval_policy=policy)
result = pipeline.run(test_data)
```

The 0.01 value is an experimental setting, not a calibrated general recommendation.

### Experiment design

The policy was first explored on 17 local NQ questions, then fixed at gap=0.01 for 500 new questions. The larger test excluded the exploratory questions. All retrieval-policy comparisons below use generator batch size 1; the effects of batching and length grouping belong to Optimizations 4 and 5.

### Results

| Metric: 500 questions, three measured repetitions per setting | Fixed top-5 | Adaptive-k, gap=0.01 |
| --- | ---: | ---: |
| Mean retained documents | 5.000 | 3.504 |
| Mean Qwen input tokens | 488.094 | 362.796 |
| Mean generated tokens, including EOS | 12.014 | 12.258 |
| Exact match (EM) | 3.0% (15/500) | 2.2% (11/500) |
| F1 | 6.4776% | 5.7725% |
| Accuracy | 6.8% | 6.4% |
| Median inference time for the entire 500-question batch | 154.922 s | 162.346 s |
| Throughput | 3.2274 queries/s | 3.0798 queries/s |

Documents decreased 29.92% and Qwen input tokens decreased 25.67%, but latency increased 4.79%.

#### Exploration: fixed k and default gap=0.05

Each setting used one warmup and three measured repetitions. Quality values in the following two tables are on a 0-1 scale. Latency is the median for the entire 17-question batch.

| Setting | Mean documents | Mean Qwen input tokens | EM | F1 | Accuracy | Batch time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed k=1 | 1 | 140.88 | 0.058824 | 0.083473 | 0.117647 | 7.1935 s |
| Fixed k=5 | 5 | 462.12 | 0 | 0.074383 | 0 | 4.1531 s |
| Fixed k=10 | 10 | 751.35 | 0 | 0.014314 | 0.058824 | 8.0531 s |
| Adaptive, gap=0.05 | 5 | 462.12 | 0 | 0.074383 | 0 | 4.4376 s |

The default threshold triggered no cuts: all 17 questions fell back to five documents. Selection time was approximately 0.132 ms per batch. The difference in generation time cannot be attributed solely to that selection overhead.

#### Exploration: gap=0.01

After observing the conservative default behavior, gap=0.01 was explored on the same questions. This was parameter exploration, not an independent test. Fixed top-5 was measured again in the same run.

| Setting | Mean documents | Mean Qwen input tokens | EM | F1 | Accuracy | Batch time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed k=5 | 5 | 462.12 | 0 | 0.074383 | 0 | 4.1121 s |
| Adaptive, gap=0.01 | 3 | 303.94 | 0.058824 | 0.083473 | 0.117647 | 5.0654 s |

Seven questions selected k=1, two selected k=2, and eight selected k=5. Six fell back; two selected a score-gap boundary at k=5. Documents fell 40%, input tokens fell 34.23%, and latency increased 23.18%. Selection took approximately 0.126 ms per batch. The default threshold was not changed to 0.01.

#### 500-question protocol and paired analysis

##### Dataset and protocol

Official source: [FlashRAG NQ test data](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/nq), revision `bcafb8dd07d453be3cbeeeb3f78be1841bddf92c`.

The source contains 3,610 questions. Seventeen previously used questions were excluded by normalized question text, leaving 3,593 unique eligible questions. `random.Random(2024).sample` selected 500. The manifest records source/sample SHA256 checksums and selected IDs.

- Full source: `datasets/flashrag_official/nq/test.jsonl`.
- Selected sample: `datasets/nq_500_heldout/nq/test.jsonl`.
- Provenance: `docs/nq_500_manifest.json`.
- Policy fixed before this evaluation: min_k=1, max_k=10, fallback_k=5, gap=0.01.

Models, corpus/index, prompts, precision, decoding, and batch size stayed the same. Each method used one five-question warmup and three full 500-question measured repetitions. Fixed top-5 ran before adaptive. There were 3,000 measured generations, but only 500 independent questions. No model training or threshold adjustment used this new sample.

##### Latency and k distribution

| Repetition | Fixed top-5 | Adaptive-k |
| --- | ---: | ---: |
| 1 | 146.4567 s | 162.3462 s |
| 2 | 154.9217 s | 161.4162 s |
| 3 | 156.4516 s | 164.9863 s |

| Selected k | Questions |
| --- | ---: |
| 1 | 139 |
| 2 | 52 |
| 3 | 19 |
| 4 | 7 |
| 5 | 277 |
| 6 | 3 |
| 7 | 3 |

There were 231 score-gap decisions and 269 fallbacks. The 277 selections of k=5 include 269 fallbacks and eight score-gap cuts.

For every repetition, 66 fixed-top-5 answers and 69 adaptive answers reached the 32-token generation limit. Mean generated length rose from 12.014 to 12.258 tokens. Output length can affect latency, but this does not fully explain the observed difference; fixed execution order, device state, and measurement variation are also limitations.

The evaluator's mean input-token metric was 503.354 for fixed and 380.264 for adaptive. These differ from the actual Qwen counts in the Optimization 1 results table and must not be mixed.

##### Paired answer-quality analysis

- EM: zero new exact matches, four lost exact matches, and 496 unchanged questions.
- F1: 20 questions improved, 24 worsened, and 456 stayed equal.
- Accuracy: five improved, seven worsened, and 488 stayed equal.
- Predictions were identical across all three repetitions of each method.

Paired bootstrap used 2,000 question resamples and seed=2024, averaging repetitions per question before resampling. It estimates question-sampling uncertainty, not latency uncertainty.

The adaptive-minus-fixed F1 difference was -0.7052 percentage points, with an approximate 95% interval of [-1.7362, +0.1614] percentage points. The interval includes zero.

The EM difference was -0.8 percentage points. Its percentile-bootstrap interval was approximately [-1.605, -0.200] percentage points, but only four questions were discordant. Percentile bootstrap can be unreliable with such sparse binary differences. The exact two-sided McNemar p-value was 0.125; the interval alone is not evidence of significant degradation, and a nonsignificant test does not establish equivalence. The accuracy McNemar p-value was approximately 0.7744.

Lost EM cases were test_3053, test_984, test_526, and test_1149. For test_526, asking about the publication date of Harry Potter and the Philosopher's Stone, fixed top-5 answered 1997 and adaptive k=1 answered 2005. The dataset scored the former as correct. Passage-level inspection is still needed to separate evidence removal from changes in generation behavior.

#### Proposal-focused diagnosis: context truncation and answer losses

A post-hoc check of the original 500-question, batch-size-1 experiment compares each fixed-5 prediction with the score-gap prediction. Fixed-5 has 15 exact matches and Adaptive-K has 11: zero EM gains and four EM losses. The selected-k distribution is {1: 139, 2: 52, 3: 19, 4: 7, 5: 277, 6: 3, 7: 3}; six queries select more than five documents. Thus mean context reduction can coexist with larger contexts for some queries and loss of relevant passages for others.

| Query ID | Adaptive k | Fixed-5 EM | Adaptive EM | Answer-string coverage in fixed-5 context | Coverage after adaptive selection |
| --- | ---: | ---: | ---: | --- | --- |
| test_3053 | 1 | 1 | 0 | True | False |
| test_984 | 3 | 1 | 0 | False | False |
| test_526 | 1 | 1 | 0 | True | False |
| test_1149 | 1 | 1 | 0 | True | False |

Coverage here means that at least one nonempty normalized gold-answer string is a substring of normalized retrieved document contents, using the repository's answer-normalization function. It is a lexical proxy, not a semantic evidence or sufficiency test. Three of the four EM-loss questions selected k=1 and lost this lexical coverage. The fourth selected k=3 and lacked coverage in both contexts. This supports investigating over-aggressive truncation and corpus coverage, but does not prove that every error is caused by missing evidence or that increasing min_k will fix the result.

This diagnosis uses already evaluated questions. A more conservative minimum k, threshold calibration, or fallback rule informed by it must be selected on development data and validated on new questions. No revised policy or quality improvement is claimed here. Evidence: `docs/experiments/nq500_gap001/topk_scope_diagnostic.json`, derived from the existing fixed/adaptive records.

### Correctness and limitations

Eleven adaptive policy/pipeline tests passed. The score-gap heuristic does not guarantee sufficient evidence. Corpus/NQ mismatch, low answer quality, and reused exploratory data limit conclusions. The 500-question result is a negative speed/quality result despite reduced context; it remains part of the report.

### Artifacts and reproduction

Code: `flashrag/retriever/adaptive_topk.py` and `flashrag/pipeline/pipeline.py`. Use `--adaptive --adaptive-gap 0.01`. Evidence: `docs/experiments/adaptive_gap_v1/`, `adaptive_gap_sensitivity_001/`, and `nq500_gap001/`. Exact commands and provenance appear in the shared Reproduction section.

## Optimization 2: Skip Unneeded Generation Score Collection

### Motivation

Text-only answers do not need stored per-step vocabulary scores. Collecting and post-processing those scores can consume additional tensor memory and computation.

### Implementation

Previously, text-only generation still requested all token scores and performed stacking, softmax, gathering, and CPU transfers.

`HFCausalLMGenerator.generate()` now skips score collection for normal text-only calls. Calls requesting return_scores or return_dict preserve their outputs. collect_scores=True forces the previous path for A/B measurements. The language model still computes logits needed for generation.

Functional tests passed. The independent real-Qwen score-on/off experiment is now reported in Optimization 4 and the shared ablation record: score-off reduced peak allocated memory, but no latency benefit was established. The initial fixed/adaptive comparisons both used score-off and could not isolate Optimization 2's effect.

### Experiment design

Replay the same 100 frozen adaptive prompts at batch size 1. Compare score collection on/off, keeping cache clearing, greedy decoding, model, precision, and token budget fixed. Three interleaved repetitions were timed without profiler hooks. Memory counters were reset after warmup. Shared ablation methodology and the original joint result table are retained later in this report.

### Results

| Path | Three times, seconds | Median | Peak allocated | Peak reserved |
| --- | --- | ---: | ---: | ---: |
| Score collection on | 33.3534 / 32.4962 / 31.8852 | 32.4962 s | 1,114.23 MiB | 1,138 MiB |
| Score collection off | 33.0056 / 32.4863 / 32.9346 | 32.9346 s | 1,103.65 MiB | 1,118 MiB |

Peak allocation decreased approximately 10.58 MiB. The score-off median was not faster, and run ranges overlapped. No latency speedup is established.

### Correctness and limitations

All 100 texts matched the saved answers in every repetition. Score-returning APIs remain supported and are covered by focused tests. Measurements are generator-only peaks on this model/workload, not full-RAG memory or a general speed claim. The text-only path skips scores by default; score-returning requests still collect them.

### Artifacts and reproduction

Code: `HFCausalLMGenerator.generate()` in `flashrag/generator/generator.py`. `collect_scores=True` or benchmark `--collect-scores` reproduces score collection. Evidence: `docs/experiments/generation_optimization/`. The shared ablation command reproduces this experiment together with Optimization 3.

## Optimization 3: Retain the CUDA Allocator Cache

### Motivation

The historical generation loop clears the allocator cache before every batch. This direction tests whether avoiding repeated allocation/free activity improves execution.

### Implementation

Added `clear_cuda_cache` to `HFCausalLMGenerator.generate()`. False skips per-batch empty_cache calls; True retains historical behavior. The benchmark exposes `--retain-cuda-cache`. The historical True default was preserved because the experiment did not establish a robust latency benefit.

### Experiment design

Use the same 100 prompts, batch size 1, score-off, and three interleaved repetitions as Optimization 2. Change only allocator-cache retention. Initial clearing and warmup occur outside formal timing; the measured path includes its normal per-batch policy.

### Results

| Path | Three times, seconds | Median | Peak allocated | Peak reserved | Device allocations / frees |
| --- | --- | ---: | ---: | ---: | ---: |
| Clear each batch | 33.0056 / 32.4863 / 32.9346 | 32.9346 s | 1,103.65 MiB | 1,118 MiB | 339 / 344 |
| Retain cache | 33.3201 / 31.8276 / 32.2539 | 32.2539 s | 1,103.65 MiB | 1,124 MiB | 2 / 0 |

Device allocation activity decreased substantially. The observed median was approximately 2.07% lower, but run ranges overlapped. Reserved memory increased 6 MiB and peak allocated memory was unchanged. This is an inconclusive latency optimization, not a demonstrated major speedup.

#### CPU runtime follow-up: intra-op thread screening (negative result)

The prior traces motivated a separate host-runtime experiment. This is a supporting follow-up within Optimization 3; it does not change the allocator result above. The native PyTorch build reports OpenMP 2019 and MKL 2025.3. `torch.set_num_threads()` controls CPU intra-op parallelism; it does not set the number of CUDA kernels or inference requests. See the [PyTorch 2.11 API documentation](https://docs.pytorch.org/docs/2.11/generated/torch.set_num_threads.html).

Protocol: the first 100 frozen adaptive prompts, sorted batch 8, greedy decoding up to 32 tokens, score collection off, and the original per-batch allocator clearing. Seven variants ran for three repetitions, with variant order rotated each round. Each variant received an eight-prompt warmup outside synchronized timing. The reference uses eight intra-op threads; inter-op threads remain eight. One-factor variants change only the named setting; the combined row changes three factors and cannot attribute a gain to one component.

| Variant | Three generation times (s) | Median (s) | Median process CPU (s) | Time change vs reference |
| --- | --- | ---: | ---: | ---: |
| Reference: sorted batch 8, 8 threads | 7.4024 / 7.1014 / 7.2134 | 7.2134 | 7.1719 | +0.00% |
| Reuse tokenization, 8 threads | 7.4892 / 7.5926 / 7.1982 | 7.4892 | 7.4531 | +3.82% |
| Decode generated tokens only, 8 threads | 8.8077 / 7.5156 / 7.1751 | 7.5156 | 7.5000 | +4.19% |
| Intra-op threads = 1 | 10.0060 / 7.3849 / 7.4006 | 7.4006 | 7.3594 | +2.59% |
| Intra-op threads = 2 | 8.6012 / 7.3058 / 7.3351 | 7.3351 | 7.2969 | +1.69% |
| Intra-op threads = 4 | 8.0108 / 7.2262 / 7.7055 | 7.7055 | 7.6875 | +6.82% |
| Reuse + generated-only decode + 1 thread | 7.7608 / 7.2102 / 7.2965 | 7.2965 | 7.2188 | +1.15% |

All variants produced exactly the same 100 texts as the sorted reference in all three repetitions, with the same quality, 1,321.41 MiB peak allocation, and batch shapes. No variant established a latency improvement: thread counts 1/2/4 were slower by 2.59%/1.69%/6.82% in median time. These results support retaining eight threads for the later experiment rather than adopting a smaller thread count.

This is within-process screening: intra-op thread settings change before each warmup, after initial model execution. The official API recommends setting the count before eager/JIT/autograd execution. Any proposed thread-setting promotion therefore needs a fresh-process confirmation with the count set before model loading. No such confirmation was needed to reject a speedup claim here. Only the 100-question prefix was used for thread screening; no 500-question thread-tuning gain is claimed. Other applications were present on the desktop GPU, and short-run timing noise remains a limitation.

Evidence: `docs/experiments/host_overhead_pilot/workload.json`, `measurements.json`, `study.json`, and `comparison.json`. The tokenizer and decoding rows are screening results for the follow-ups under Optimizations 5 and 4, respectively.

### Correctness and limitations

Every measured text matched the reference. Focused tests verify skipped cache-clear calls and unchanged text. Reduced allocation counts do not imply lower total execution time. Runtime and memory results apply to this workload; the setting remains optional.

### Artifacts and reproduction

Code: `flashrag/generator/generator.py`; benchmark flag: `--retain-cuda-cache`. Evidence and all per-run counters: `docs/experiments/generation_optimization/`. See the shared allocator/score ablation record for the original combined table and documentation sources.

## Optimization 4: Batched Qwen Generation

### Motivation

Generation dominated the pipeline, while later CPU/CUDA traces showed substantial host execution and many small kernels. This direction increases the number of questions processed together to improve offline throughput.

### Implementation

Exposed the existing generator batching support through `--generator-batch-size`, recorded the setting in metadata, and extended batch-aware token-log accounting. Batch sizes 1, 4, and 8 were tested. Default batch size remains 1; tested batching changes are explicit. This is a configuration-level optimization, not a newly invented batching algorithm. A later follow-up also tests sorted batch 32 and an opt-in `decode_generated_only` path. The latter copies only continuation IDs to the CPU once per batch and calls `batch_decode`, avoiding two full/prompt decode calls per question. Benchmark flag: `--decode-generated-only`. The legacy decoder remains the default because tokenizer behavior across the prompt/continuation boundary is not universally equivalent.

### Experiment design

Three linked experiments provide evidence: a same-session 100-prompt batch-size ablation, a 500-question full-pipeline regression, and a CPU/GPU follow-up with separate normal-generation timing and fixed-output traces. The common frozen-prompt and timing methodology is specified in Shared experimental methodology.

The host-overhead follow-up replays all 500 saved adaptive prompts, with the same model, bfloat16 precision, greedy maximum-32-token decoding, stable ascending length order, and eight CPU intra-op/inter-op threads. Five variants (sorted batch 8 reference, token reuse only, continuation decoding only, sorted batch 32 only, and batch 32 plus both code changes) run for three rotated-order repetitions. Each receives an eight-prompt warmup and initial allocator clear outside timing; per-batch clearing remains enabled. CUDA synchronization encloses wall timing; process CPU time and peak allocated/reserved CUDA memory are recorded. The source JSON hash, prompt hash, IDs, predictions, phase counters, batch shapes, and processing order are saved. No profiler or forward hooks are active in these formal trials. The separate diagnostic uses the first 32 questions and forces exactly eight output tokens each (256 logical output tokens), so changed EOS behavior cannot explain its kernel counts. The 500 questions are reused performance-regression data, not a new independent quality evaluation.

### Results

#### Initial pipeline bottleneck evidence

The following are independently calculated stage medians for the full 500-question batch. They need not sum exactly to the median end-to-end time.

| Stage | Fixed top-5 | Adaptive-k |
| --- | ---: | ---: |
| E5 encoding and FAISS retrieval | 0.273156 s | 0.362836 s |
| Adaptive selection | 0 | 0.002231 s |
| Prompt construction | 0.917951 s | 0.724209 s |
| Generation | 153.702394 s | 161.312360 s |
| End-to-end inference | 154.921740 s | 162.346170 s |

**Generation dominates runtime.** Optimizing a few milliseconds of score-gap selection alone is unlikely to materially reduce a batch taking approximately 160 seconds.

The generation timer includes tokenization, device transfer, model generation, and output handling. At this initial characterization stage they were not separated. Optimization 4 and the shared ablation record adds diagnostic prefill/decode profiling and generator-only peak-memory measurements; sustained GPU utilization remains unmeasured.

Candidate hypotheses, not confirmed bottlenecks:

- Per-batch `torch.cuda.empty_cache()` may introduce allocator or synchronization overhead.
- Batch size=1 may limit throughput.
- Tokenization, host-to-device transfer, prefill, and decode may contribute differently.
- The real latency and memory effect of Optimization 2 needs a separate A/B test.

#### Initial three-prompt diagnostic

The separate three-prompt diagnostic produced the following host-side named scopes. CUDA kernel totals are taken from the corresponding host-side profiler entries. Rows are nested and must not be added together.

| Diagnostic scope | Calls | Host scope time | Associated CUDA kernel time |
| --- | ---: | ---: | ---: |
| Tokenization | 3 | 3.996 ms | 0 |
| empty_cache | 3 | 1.887 ms | 0 |
| Entire model.generate | 3 | 1,896.664 ms | 207.022 ms |
| Prefill forward | 3 | 191.999 ms | 68.100 ms |
| Decode forwards | 30 | 1,648.290 ms | 137.149 ms |
| Text decoding | 6 | 1.318 ms | 0.007 ms |

CUDA events around forward calls accumulated 200.301 ms for prefill and 1,663.456 ms for decode. Event intervals can include host-launch gaps on the stream; they are not pure kernel-compute time. Profiler and event instrumentation also add overhead. The large difference between these intervals and kernel totals cannot be used as a direct production GPU-utilization measurement.

The trace nevertheless identifies repeated autoregressive decode forwards as the dominant forward-related host scope for these three inputs, while tokenization and output-text handling are much smaller. Matrix multiplication was the largest reported operator-level CUDA contribution: aten::mm self CUDA time was 88.777 ms; aten::copy_ was 23.358 ms and aten::bmm was 14.138 ms. These are diagnostic observations on three prompts, not estimates of the entire 500-question population.

Raw profiler tables also contain device-only annotation entries with duplicated scope names and percentages above 100%. Those annotation rows are not summed or interpreted as utilization. The report uses the host-associated rows and separately labeled CUDA-event intervals.

#### Controlled batch-size comparison

Batch size was the only configuration factor changed. All variants continued to clear the cache per batch and used score-off. The same 100 prompts and greedy output limit were used. Increasing batch size inherently reduces the number of model.generate and per-batch cache-clear calls: 100, 25, and 13 generation calls at batch sizes 1, 4, and 8 respectively.

| Batch size | Three times, seconds | Median | Relative speedup | Peak allocated | Peak reserved | Changed texts out of 100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 27.4766 / 27.3530 / 26.8687 | 27.3530 s | 1.00x | 1,103.65 MiB | 1,118 MiB | 0 |
| 4 | 13.5317 / 13.6385 / 12.9325 | 13.5317 s | 2.02x | 1,169.01 MiB | 1,236 MiB | 14 |
| 8 | 9.1281 / 9.9183 / 10.1169 | 9.9183 s | 2.76x | 1,386.20 MiB | 1,482 MiB | 16 |

Text-mismatch counts were stable across repetitions. Batch size 8 reduced the measured median generation time by approximately 63.74% relative to its same-session baseline, at an additional 282.54 MiB of peak allocated memory. It was selected for the full regression because it had the best observed throughput and fit this machine's memory budget.

Batching amortizes repeated host scheduling and permits parallel processing of several questions. This interpretation is consistent with the decode-heavy trace and reduced generation-call count. The experiment did not separately isolate every mechanism or measure sustained GPU utilization. Left padding and waiting for the longest answer in a batch can also add computation, so fewer logical input tokens do not alone predict batching speed.

The allocator-study baseline was 32.9346 s, whereas this session's baseline was 27.3530 s. This between-session difference demonstrates why ablations must use their own same-session baseline. Cross-session ratios are labeled separately below.

#### Full 500-question regression

Both fixed top-5 and adaptive retrieval were rerun on all 500 frozen questions, each with three repetitions and the original five-question warmup. Models, corpus/index, thresholds, maximum output length, and score-off/cache-clear settings were unchanged. Only generator batch size changed from 1 to 8. This was regression on a reused workload, not a newly held-out quality test.

| Setting | Batch size | Three end-to-end times, seconds | Median | Throughput | EM | F1 | Accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Fixed top-5, earlier run | 1 | 146.4567 / 154.9217 / 156.4516 | 154.9217 s | 3.2274 q/s | 3.0% | 6.4776% | 6.8% |
| Fixed top-5, optimized | 8 | 49.6722 / 50.3371 / 51.4921 | 50.3371 s | 9.9330 q/s | 3.2% | 6.8349% | 7.0% |
| Adaptive, earlier run | 1 | 162.3462 / 161.4162 / 164.9863 | 162.3462 s | 3.0798 q/s | 2.2% | 5.7725% | 6.4% |
| Adaptive, optimized | 8 | 51.9629 / 50.4393 / 50.6425 | 50.6425 s | 9.8731 q/s | 2.2% | 5.8568% | 6.4% |

Observed end-to-end ratios against the earlier batch-size-1 runs were **3.08x** for fixed top-5 and **3.21x** for adaptive, corresponding to reductions of **67.51%** and **68.81%**. These full-run ratios compare separate sessions. The same-session controlled 2.76x result is stronger evidence for the causal batching effect; no exact causal interpretation or confidence interval is claimed for the larger cross-session ratios.

Optimized stage medians:

| Stage, 500 questions | Fixed top-5, batch 8 | Adaptive, batch 8 |
| --- | ---: | ---: |
| Retrieval | 0.240031 s | 0.290913 s |
| Selection | 0 | 0.002207 s |
| Prompt construction | 0.927406 s | 0.697832 s |
| Generation | 49.168841 s | 49.654308 s |
| End-to-end | 50.337055 s | 50.642504 s |

The Qwen logical input means remained exactly 488.094 and 362.796 tokens for fixed/adaptive. Retained documents, question order, and per-question k were identical to the earlier runs. All three optimized repetitions produced identical predictions within each method.

#### CPU/GPU running-time follow-up

##### 13.1 Purpose and measurement definitions

The previous end-to-end results established a throughput improvement but did not, by themselves, establish where the CPU and GPU spent their time. This follow-up measures those quantities separately and tests a fixed-output-length workload.

The machine is running native Windows. This study used PyTorch CPU/CUDA profiling and the process CPU clock, **not Linux perf**. Changing to a Linux environment would introduce another experimental variable.

Three distinct quantities are reported:

1. **Wall time:** elapsed application time, including work, waiting, transfers, and synchronization.
2. **Process CPU time:** time.process_time(), implemented here by Windows GetProcessTimes(), accumulating user/kernel CPU execution across process threads. It is not the CPU-total duration of a profiler scope.
3. **CUDA kernel time:** durations of actual kernel events captured in the CUDA trace. Summed kernel durations and the union of their execution intervals are reported separately to avoid double-counting overlap.

CPU execution can overlap GPU execution. CPU seconds and kernel milliseconds cannot simply be added to explain wall time, or subtracted to obtain a reliable waiting-time decomposition. GPU kernel coverage below is process-specific traced activity; it is not device-wide utilization or SM occupancy.

##### 13.2 Workloads and controls

**Unprofiled formal runs:** the same frozen 100 adaptive prompts from Optimization 4 and the shared ablation record; normal greedy generation up to 32 output tokens; batch sizes 1 and 8; four repetitions per configuration. Each path was warmed with the same first eight prompts. Order alternated 1/8, 8/1, 1/8, 8/1. CUDA was synchronized before and after timing. No profiler or forward hooks ran inside these measurements.

Process CPU time was recorded over the same synchronized interval as wall time. The CPU clock reported a 100 ns nominal resolution, with monotonic and non-adjustable properties. The machine exposed 16 logical CPUs. CPU-seconds divided by wall-seconds is therefore an average number of logical-core equivalents used by this process, not a whole-machine CPU percentage.

**Separate diagnostic traces:** eight identical prompts were selected across the input-length distribution, with lengths 76, 114, 165, 242, 309, 412, 611, and 977 tokens. Both configurations forced exactly eight generated tokens per question using min_new_tokens=max_new_tokens=8. Each trace therefore represents the same eight questions and 64 logical output tokens.

The diagnostic control used a maximum of eight outputs rather than the normal 32-token cap to limit trace size. It is a performance diagnostic and does not evaluate normal answer quality. Batch size 1 processed 2,906 logical input-token slots; padding the eight prompts together required 7,816 input-token slots at batch size 8. The contexts and questions were unchanged, although the padded work differs.

Profiling used CPU and CUDA activities with shapes, memory profiling, and stacks disabled. Named host scopes labeled prefill, decode, tokenization, and text decoding. Extra per-forward CUDA events from the earlier diagnostic were disabled. Traces were captured once per batch size after warmup and compressed after analysis. These instrumented timings are not substitutes for the formal unprofiled timings.

##### 13.3 Formal CPU time and wall time on 100 normal prompts

| Repetition | Batch 1 wall | Batch 1 process CPU | Batch 8 wall | Batch 8 process CPU |
| --- | ---: | ---: | ---: | ---: |
| 1 | 27.7557 s | 25.140625 s | 9.9734 s | 8.687500 s |
| 2 | 28.0746 s | 25.531250 s | 9.2236 s | 8.750000 s |
| 3 | 26.9178 s | 24.718750 s | 9.7448 s | 9.109375 s |
| 4 | 27.1191 s | 25.515625 s | 9.3518 s | 8.625000 s |
| Median | **27.4374 s** | **25.328125 s** | **9.5483 s** | **8.718750 s** |

The observed wall-time ratio was **2.87x**. Process CPU time decreased by **65.58%**. Median per-run CPU-core equivalents were 0.914 and 0.929 respectively: the shorter batch-8 run still used substantial host CPU execution rather than eliminating it.

Batch-size-1 outputs matched every saved answer. Batch size 8 changed 16/100 texts in each repetition, consistent with the prior study. These measurements do not establish text equivalence; the full-workload quality checks remain those in Optimization 4 and the shared ablation record.

##### 13.4 Actual CUDA execution in the fixed-eight-token traces

The following table is for **eight questions**, not the 100-question formal workload above. It must not be combined numerically with the 100-question CPU times.

| Trace measurement | Batch 1 | Batch 8 |
| --- | ---: | ---: |
| Instrumented host window | 2,451.344 ms | 386.262 ms |
| Process CPU during instrumented generation | 2.406250 s | 0.406250 s |
| Actual kernel events | 90,096 | 9,779 |
| Sum of CUDA kernel durations | 446.861 ms | 194.193 ms |
| Union of kernel execution intervals | 446.862 ms | 194.194 ms |
| Process kernel coverage of instrumented host window | 18.23% | 50.28% |
| Union including traced memcpy and memset activity | 447.727 ms | 194.415 ms |
| Recorded kernel-launch API events | 90,096 | 9,779 |
| Union of host intervals inside launch APIs | 756.046 ms | 77.470 ms |
| Synchronization API events | 201 | 41 |
| Union of host synchronization API intervals | 73.655 ms | 113.859 ms |

Tiny sum-versus-union discrepancies arise from exported timestamp precision. Union accounting excludes nested GPU annotation scopes, which otherwise would double-count execution ranges.

Only approximately 18% of the instrumented batch-1 window was covered by this process's recorded kernels, compared with approximately 50% at batch size 8. Uncovered time includes scheduling gaps, host work, instrumentation, and potentially activity not represented by these event categories. It is not automatically global GPU idle time, and these percentages are not production GPU-utilization estimates.

Recorded launch API time fell substantially along with kernel count. However, fewer synchronization calls did **not** mean less synchronization API wall time: the total host interval inside those calls increased in batch 8. API wall durations can include waiting and should not be interpreted as CPU execution clocks. This prevents attributing the speedup simply to "less synchronization."

##### 13.5 Prefill/decode attribution through launch correlations

GPU execution is asynchronous. A kernel may finish after the CPU's prefill scope has already returned. Kernels were therefore assigned to prefill/decode by matching CUDA launch correlation IDs and the host launch scope, not by checking whether the GPU timestamp fell inside a CPU scope.

| Kernel attribution | Batch 1 count | Batch 1 summed GPU time | Batch 8 count | Batch 8 summed GPU time |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 12,216 | 171.314 ms | 1,162 | 143.374 ms |
| Decode | 75,096 | 270.848 ms | 8,254 | 50.049 ms |
| Outside forward scopes or unmatched | 2,784 | 4.699 ms | 363 | 0.770 ms |
| Total | 90,096 | 446.861 ms | 9,779 | 194.193 ms |

Batch 1 issued eight prefill forwards and 56 decode forwards; batch 8 issued one prefill and seven decode forwards. Both generated 64 logical tokens. Decode-related GPU time fell from approximately 271 to 50 ms in this control, while the prefill reduction was much smaller.

The kernel mix also changed. Batch 1's largest kernel entry was a BF16 matrix-vector kernel, with 4,096 calls and 147.500 ms of cumulative execution. Batch 8's largest entry was an Ampere-compatible BF16 GEMM kernel, with 72 calls and 49.824 ms; a fused attention kernel contributed 192 calls and 23.564 ms. These observed implementation differences are consistent with different execution efficiency at larger matrix shapes. They do not isolate the contribution of every kernel or prove that all gains come from CPU scheduling.

##### 13.6 Transfers and synchronization

| Recorded memcpy activity in the eight-question control | Batch 1 | Batch 8 |
| --- | ---: | ---: |
| All memcpy event count | 720 | 90 |
| Union of memcpy execution | 0.787 ms | 0.174 ms |
| Host-to-device count / bytes | 40 / 46,752 | 4 / 62,560 |
| Host-to-device execution union | 0.0231 ms | 0.0046 ms |
| Device-to-device count / bytes | 520 / 136,182,128 | 51 / 119,242,946 |
| Device-to-device execution union | 0.6193 ms | 0.1327 ms |

Most recorded bytes were device-to-device, not PCIe input traffic. The measured memcpy-engine intervals were small relative to the trace windows, so these traces do not support bulk memcpy execution as the dominant bottleneck. This does not measure all transfer-related host preparation or copies implemented as kernels. Tokenization host scopes were 9.131 ms at batch 1 and 7.894 ms at batch 8, also small in this diagnostic.

The model was already resident on the GPU; checkpoint loading and initial model transfer were excluded. No conclusions about cold-start transfer performance follow from these measurements.

##### 13.7 Supported bottleneck interpretation

The results support a **host-driven, many-small-kernel autoregressive execution limitation at batch size 1**, with additional GPU efficiency differences between matrix shapes:

- Formal runs consumed substantial process CPU time, which dropped together with wall time under batching.
- The fixed-output trace removed answer-length differences and still showed a large reduction in launch/kernel count and decode kernel time.
- Traced kernel coverage increased, although it is not a global utilization measurement.
- Bulk memcpy-engine time was small; allocator clearing and tokenizer scope time were also small in the diagnostic.

This is stronger evidence than the earlier wall-time-only argument. It is still not a complete native CPU call-stack study or proof that the whole application is purely CPU-bound. Profiler overhead can exaggerate launch gaps, the GPU trace covers only eight selected prompts, and CPU execution overlaps GPU work. The workload contains both host dispatch costs and real GPU computation.

Potential future work includes reducing per-step launch overhead or improving operator fusion, with new correctness and timing checks. These follow-up optimizations were not implemented during this measurement task.

##### 13.8 Validation, artifacts, and reproduction

The existing test file now also tests trace interval union, exclusion of nested annotations, transfer accounting, and asynchronous launch-correlation attribution. All **six focused tests** passed. A test fixture initially omitted the bisect dependency used by the extracted trace parser; the fixture was corrected and the tests rerun successfully. Actual trace analysis used the complete module and was unaffected.

The raw eight-question traces were checked for nonempty CUDA kernel capture and matching phase totals. Normal timing produced four complete runs of each batch size; input IDs/hash remain in workload.json. No additional Markdown report was created.

Artifacts are under `docs/experiments/cpu_gpu_runtime/`:

| File | Purpose |
| --- | --- |
| `workload.json` | Frozen 100-question IDs, input lengths, source and prompt hashes |
| `cpu_gpu_measurements.json` | All eight unprofiled CPU/wall measurements |
| `cpu_gpu_study.json` | Formal summaries, clock metadata, and complete diagnostic analysis |
| `cpu_gpu_trace_analysis.json` | Kernel, API, transfer, coverage, and phase measurements |
| `batch1_fixed8_trace.json.gz`, `batch8_fixed8_trace.json.gz` | Original compressed CPU/CUDA timelines |
| `run.log` | Execution and token diagnostics |

Using the environment variables in the Reproduction section:

```powershell
.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_gap001/adaptive_topk_adaptive_gap_20261113_151722_140059.json `
  --sample-num 100 --repeats 4 --cpu-gpu-study `
  --output-dir docs/experiments/cpu_gpu_runtime
```

Use a fresh directory for a new experiment. Full-device utilization, native CPU stack sampling, and an exactly-32-output-token control remain unmeasured; none is inferred from these results.

#### Follow-up: larger batches and continuation-only batch decoding

| Variant | Three generation times (s) | Median (s) | Median process CPU (s) | Time reduction vs batch 8 | Peak allocated / reserved (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Sorted batch 8 reference | 44.3638 / 42.7101 / 43.9068 | 43.9068 | 40.5469 | 0.00% | 2,241.96 / 2,342 |
| Batch 8 + token reuse only | 42.1952 / 39.1183 / 44.0015 | 42.1952 | 39.4219 | 3.90% | 2,241.96 / 2,342 |
| Batch 8 + continuation decode only | 38.6025 / 41.7647 / 45.6430 | 41.7647 | 38.2812 | 4.88% | 2,241.96 / 2,342 |
| Sorted batch 32 only | 15.2402 / 16.2190 / 17.0250 | 16.2190 | 15.5000 | 63.06% | 2,642.69 / 2,844 |
| Batch 32 + reuse + continuation decode | 15.4662 / 15.4786 / 16.1207 | 15.4786 | 15.0625 | 64.75% | 2,642.69 / 2,844 |

Sorted batch 32 alone reduced median generation time by **63.06% (2.71x throughput)**. The combined path reduced it by **64.75% (2.84x)** against the same-session sorted-batch-8 reference. These are offline generation throughput measurements, not per-request latency. Peak allocated memory increased 400.73 MiB (17.87%).

Continuation-only decoding reduced its median measured host wall interval from 0.135477 to 0.012468 s (90.80%) at batch 8, preserving all 500 texts. Its overall generation median was 4.88% lower, but run ranges overlapped (reference 42.71-44.36 s; changed decoder 38.60-45.64 s), and the 100-prompt pilot was slower. The roughly 0.123 s component saving alone cannot explain the roughly 2.14 s difference between whole-generation medians. The supported component result is less decoding overhead; a robust 4.88% end-to-end latency claim is not established.

The combined path and batch 32 alone had identical predictions in every repeat. Their median difference was 0.7404 s, but timing ranges overlapped. The two code changes reduce measured host overhead; this experiment does not isolate a statistically robust extra throughput improvement at batch 32.

| Batch-shape or quality measure | Sorted batch 8 | Sorted batch 32, with or without both code changes |
| --- | ---: | ---: |
| Logical input tokens | 181,398 | 181,398 |
| Padded input slots | 184,748 | 195,380 |
| Generation batches | 63 | 16 |
| Generated slots including EOS padding | 13,168 | 16,000 |
| EM | 2.2% (11/500) | 2.2% (11/500) |
| F1 (fraction) | 0.0585364338836 | 0.0582642861386 |
| Answer-containing accuracy | 6.6% (33/500) | 6.6% (33/500) |
| Texts differing from sorted batch 8 | 0 | 77/500 |

Batch 32 increased padded input slots by 5.75%, and output-slot totals are not useful-token counts. All variants were text-stable across their own three repetitions. All per-question EM outcomes were preserved at batch 32; accuracy had one gain and one loss. F1 improved on seven questions and worsened on six: mean change -0.02721 percentage points, paired-bootstrap 95% interval [-0.22515, +0.14716] percentage points (2,000 question resamples, seed 2024). This is a small observed F1 decline with uncertainty, not proof of quality equivalence. No retrieval threshold was retuned.

#### Follow-up CPU/CUDA diagnostic: fewer launches, more GPU work

Separate profiled control: the same 32 questions, 13,270 logical input tokens, exactly eight output tokens each, and all sorting/tokenization/decoding included in the host window. The profiler is absent from formal timing. The diagnostic is one capture per path, not a repeated 500-question latency result.

| Diagnostic measure | Sorted batch 8 | Sorted batch 32 | Batch 32 + both code changes |
| --- | ---: | ---: | ---: |
| Instrumented host window (ms) | 1682.877 | 1011.712 | 987.940 |
| Actual CUDA kernels | 39116 | 10115 | 10116 |
| Kernel duration sum (ms) | 467.403 | 813.181 | 810.697 |
| Kernel interval union (ms) | 467.404 | 813.181 | 810.697 |
| Process kernel coverage (%) | 27.774 | 80.377 | 82.059 |
| Kernel-launch API host interval union (ms) | 406.156 | 136.628 | 133.872 |
| Synchronization API calls | 161 | 89 | 26 |
| Synchronization API host interval union (ms) | 177.000 | 578.697 | 588.771 |
| Copy interval union (ms) | 0.695 | 0.831 | 0.692 |
| Prefill CUDA kernel sum (ms) | 299.684 | 673.195 | 674.243 |
| Decode CUDA kernel sum (ms) | 164.634 | 138.355 | 134.853 |

Batch 32 lowered kernel count by approximately 74.14% and reduced kernel-launch host intervals, while raising traced kernel coverage. However, this 32-question control padded 17,048 input slots at batch 8 versus 31,264 at batch 32. Prefill GPU kernel time increased from 299.684 to 673.195 ms, and total GPU kernel duration increased from 467.403 to 813.181 ms. Fewer batches and launches can improve total host elapsed time despite greater GPU work; the data do not support saying batch 32 reduced GPU computation or improved SM occupancy. The GPU clock/thermal contribution was not isolated.

Synchronization counts fell from 161 to 89 to 26, but synchronization host intervals rose from 177.000 to 578.697 to 588.771 ms. A smaller synchronization count is therefore not a reduction in synchronization time. Process kernel coverage is not device-wide utilization, and overlapping host/device intervals must not be added. Host text-decode annotations fell from 64 calls at either legacy-decoder batch size to one call for combined batch 32. Copy intervals remained below 1 ms in this diagnostic.

Evidence: `docs/experiments/host_overhead_500/` contains all formal runs, workload/source hashes, predictions, comparisons, question-level quality deltas, and three compressed fixed-output traces. The current source hash is recorded in `study.json`.

#### Follow-up full-RAG integration and adoption decision

A one-pass integration check applies sorted batch 32, token reuse, continuation-only decoding, and eight CPU intra-op threads to both fixed top-5 and adaptive retrieval. It uses the same 500 questions, one eight-question warmup per setting, and the original cache-clear policy. This checks complete execution and quality; it is not a repeated, paired end-to-end speed benchmark.

| Setting | Full 500-question time (s) | Generation (s) | EM | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed top-5, combined batch 32 | 18.863107 | 17.628032 | 2.8% | 0.0631126681702 | 6.6% |
| Adaptive-k, combined batch 32 | 17.473780 | 16.457615 | 2.2% | 0.0582642861386 | 6.6% |

Both settings retain all 500 IDs in original order and exactly the same retrieved documents and selected k as the previous sorted-batch-8 pipeline. Adaptive texts exactly match the combined frozen-prompt replay. Relative to the prior batch-8 pipeline, 77 adaptive texts and 89 fixed-top-5 texts changed.

The fixed-top-5 result reveals an adoption cost: EM falls from 3.2% (16/500) to 2.8% (14/500), with zero gains and two losses. Fixed F1 falls from 0.0672358666112 to 0.0631126681702, a change of -0.41232 percentage points (eight question gains, eleven losses; paired-bootstrap 95% interval [-1.07467, +0.07965] percentage points). Accuracy falls from 6.8% to 6.6%. This comparison changes batch size and both optional code paths; it does not isolate the cause of each fixed-setting text change. The isolated code-path equivalence checks above apply to the adaptive replay, not every possible prompt or tokenizer.

**Adoption decision:** retain the existing defaults. Batch 32 and the two code changes remain explicit benchmark options. Batch 32 has a demonstrated offline generation-throughput benefit on this machine, with additional peak GPU allocation and changed answers; it is not a universal quality-preserving replacement for batch 8. In particular, the fixed-top-5 EM regression prevents claiming unchanged quality for both retrieval settings. Latency speedups from different baselines or sessions must not be multiplied.

Integration evidence: `docs/experiments/nq500_host_optimized/`, run ID `20261113_191116_267336`; `comparison.json`, `integration_validation.json`, and `quality_vs_sorted_batch8.json` retain setting-level and paired quality checks. Earlier 500-pipeline timings come from separate sessions and do not provide the causal speedup estimate; use the same-session replay above for that claim.

#### Windows CPU follow-up: thread accounting and Python CPU profiles

The WPR CPU profile is available and includes sampled stacks, context switches, and ready-thread events. Both the normal and tool-escalated start attempts failed with error `0xc5585011`: failure to enable the policy to profile system performance. The current token is not an administrator. No WPR ETL was captured, and no native system-wide call-stack or wait-cause results are claimed. WPA was not found on PATH or in the common toolkit directories checked. These are OS/tool availability limits, not an automatic approval-review rejection.

Completed fallback: the same 500 frozen adaptive prompts, sorted batches 8 and 32, eight intra-op/inter-op threads, three alternating-order repetitions, and eight-prompt warmup before each run. CUDA synchronization encloses generation timing. Windows process and calling-thread CPU clocks, per-thread snapshots, and context-switch counters are recorded. No function profiler runs during these counter measurements.

| Path | Three wall times (s) | Median process CPU (s) | Median main-thread CPU (s) | Three context-switch deltas |
| --- | --- | ---: | ---: | --- |
| sorted_batch8 | 42.4243 / 41.6575 / 39.1422 | 38.968750 | 38.906250 | 7376 / 15168 / 7294 |
| sorted_batch32 | 16.4017 / 16.0527 / 16.6082 | 15.015625 | 14.984375 | 2972 / 2563 / 5034 |

Active process CPU time is concentrated in the calling thread. This is consistent with host-side generation/dispatch overhead, but includes native driver activity and possible busy waiting; it does not prove that Python computations alone limit latency. Process CPU time divided by elapsed time is a core-equivalent measure, not machine-wide utilization.

On this Windows psutil implementation, the field named voluntary contains the OS context-switch count and the involuntary field is zero. The rows therefore report the exposed total rather than attributing switches to voluntary/involuntary causes. Some threads exited between snapshots in three runs; cumulative counters and per-thread coverage can be affected by thread lifetimes. No scheduling-wait duration or reason is inferred from these counts.

Separate cProfile diagnostics use only the first 32 prompts and the custom `time.thread_time` timer (`GetThreadTimes`), with no native C++ stack expansion. These instrumented profiles are excluded from benchmark comparisons. Their short duration and observed approximately 15.625 ms accounting granularity limit small-function attribution. Prefix batch shapes differ from the complete 500-prompt sorted collection.

| Diagnostic | Batch 8 | Batch 32 |
| --- | ---: | ---: |
| Instrumented wall time (s) | 3.039332 | 1.580511 |
| Process CPU time (s) | 3.031250 | 1.515625 |
| Attributed main-thread CPU (s) | 3.031250 | 1.515625 |

The profiles identify linear/attention native entry points, the generation-loop `_has_unfinished_sequences` function, and PyTorch module dispatch among visible CPU consumers. Native CUDA entry-point CPU time does not measure GPU execution, and a synchronization-related Python frame can include driver busy waiting. Inclusive/cumulative function times overlap and must not be summed.

Top exclusive main-thread CPU entries for `sorted_batch8`:

| Function | Calls | Attributed self CPU (s) |
| --- | ---: | ---: |
| `<built-in method torch._C._nn.linear>` | 16731 | 0.437500 |
| `<built-in method torch._C._nn.scaled_dot_product_attention>` | 2376 | 0.296875 |
| `_has_unfinished_sequences` | 103 | 0.281250 |
| `forward` | 4851 | 0.218750 |
| `<built-in method torch.rsqrt>` | 4851 | 0.187500 |
| `<method 'to' of 'torch._C.TensorBase' objects>` | 10502 | 0.156250 |
| `apply_rotary_pos_emb` | 2376 | 0.156250 |
| `_call_impl` | 31482 | 0.125000 |

Top exclusive main-thread CPU entries for `sorted_batch32`:

| Function | Calls | Attributed self CPU (s) |
| --- | ---: | ---: |
| `_has_unfinished_sequences` | 33 | 0.562500 |
| `<built-in method torch._C._nn.linear>` | 5408 | 0.203125 |
| `_call_impl` | 10176 | 0.078125 |
| `<built-in method torch._C._nn.scaled_dot_product_attention>` | 768 | 0.062500 |
| `__getattr__` | 22627 | 0.046875 |
| `<method 'encode_batch' of 'tokenizers.Tokenizer' objects>` | 33 | 0.046875 |
| `forward` | 1568 | 0.046875 |
| `rotate_half` | 1536 | 0.046875 |

Evidence: `docs/experiments/windows_cpu_sampling/availability.json`, `counters/cpu_measurements.json`, `counters/windows_cpu_study.json`, and both `counters/*_thread_cpu.prof` files. Thirteen focused tests passed, including four new WPR ownership/cleanup tests. They check that a non-administrator is rejected before work, an existing recording is untouched, a failed start never triggers stop, and an owned recording is stopped if generation fails.

The administrator capture entry point is `scripts/capture_windows_cpu.ps1`. It prepares warmed batch-8 and batch-32 captures under `docs/experiments/windows_cpu_sampling/etw/` and records PID plus generation-window UTC timestamps for filtering in WPA. The entry point is implemented and cleanup behavior is unit-tested; actual ETW collection and native stack analysis remain pending administrator execution. It does not change security policies or install tools.

To rerun the completed counter/profile study without administrator rights:

```powershell
.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_batch8/adaptive_topk_adaptive_gap_20261113_160953_930358.json `
  --sample-num 500 --repeats 3 --warmup-sample-num 8 --cpu-threads 8 `
  --windows-cpu-study --output-dir docs/experiments/windows_cpu_sampling/counters
```

Use a new output directory for reruns. WPR usage follows the [Microsoft WPR command reference](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/wpr-command-line-options). The custom profiler timer follows the [Python 3.11 profiler documentation](https://docs.python.org/3.11/library/profile.html).

### Correctness and limitations

Batching was not text-equivalent: 87/500 fixed-top-5 answers and 84/500 adaptive answers changed relative to batch size 1. Batch shape and floating-point execution can affect generation; the exact cause of each changed text was not isolated. The default benchmark batch size remains 1, with batch size 8 explicitly selected for the reported offline throughput optimization.

For fixed top-5, one additional question became an exact match and no exact matches were lost; exact paired McNemar p=1.0. For adaptive, all per-question EM outcomes stayed the same, retaining 11 exact matches. Adaptive F1 improved on nine questions and worsened on six; its mean change was +0.0844 percentage points with a paired-bootstrap 95% interval approximately [-0.0744, +0.2615] percentage points. Accuracy had one gain and one loss.

In the earlier batch-1-to-8 regression, the observed quality metrics did not decline, but that experiment does not establish semantic equivalence or a general quality improvement. The later batch-32 integration check below does show a fixed-top-5 quality regression. There is no new evidence that the score-gap retrieval policy itself is superior: optimized fixed top-5 still has higher EM than optimized adaptive retrieval on this workload.

For batches larger than one, token diagnostics report padded batch widths. The analysis script now validates 63 generation batches per 500-question repetition and labels input/output slot totals as including padding. Output slots are not treated as useful generated-token counts. Generator-only peak-memory observations are not reported as full-RAG peak memory.

A supportable result is: **on this GPU and 500-question offline workload, profiling motivated a batching configuration that substantially improved measured RAG throughput while retaining the observed adaptive EM.** The controlled replay measured a 2.76x generation speedup; separate full-run measurements observed adaptive batch time fall from 162.35 to 50.64 seconds.

The contribution is supported by frozen inputs, one-factor ablations, all repetition times, memory measurements, and paired answer checks. Negative allocator/score latency results are retained rather than omitted.

Important threats to validity are one machine/model, only three timing repetitions, session-to-session variation, profiler overhead on three diagnostic prompts, corpus/NQ mismatch, reused quality data, changed answer texts, and the distinction between offline throughput and individual-request response time. The optimization does not demonstrate faster FAISS retrieval, better adaptive-k selection, or the same gains for larger models or online traffic.

Optimization 4 (CPU/GPU follow-up) subsequently adds process CPU clocks, CUDA kernel/transfer/synchronization trace analysis, and a fixed-eight-output-token control. An exactly-32-token control, sustained device-wide utilization, full-RAG peak memory including E5, and broader independent quality evaluation remain future work. The fixed-eight control is not reported as an executed fixed-32 experiment.

### Artifacts and reproduction

Primary evidence: `docs/experiments/generation_batching/`, `nq500_batch8/`, and `cpu_gpu_runtime/`. Commands, workload hashes, trace accounting, and original-versus-optimized summaries are retained in the shared Reproduction section and the CPU/GPU follow-up above. Batching improves offline throughput; it does not prove reduced single-request latency or better adaptive-k selection.

## Optimization 5: Length-Aware Batching

### Motivation

At batch size 8, mixed input lengths created substantial padding. This direction tests whether grouping similar lengths reduces execution time while retaining the existing batch size and retrieval policy.

### Implementation

After batch size 8 improved throughput, input padding became a plausible optimization target. Different-length prompts in a batch are padded to its longest member. The hypothesis was that grouping similar-length prompts would reduce prefill work and total generation time.

`HFCausalLMGenerator.generate()` now accepts **sort_by_length=False**. When enabled, it measures each prompt's tokenized length using the generator's truncation limit, performs a stable ascending length sort, and forms the normal batches. After generation, it restores response texts, token scores, token IDs, and logits to original input order. Stop-criteria construction uses the reordered prompts. The default remains False.

The implementation computes lengths inside the generate call. Length calculation, sorting, normal per-batch tokenization, generation, and output-order restoration are included in the measured elapsed time. Lengths are not supplied from an earlier experiment to hide their cost. Additional per-call diagnostics record processing order, logical input tokens, padded input slots, padded output slots, batch count, and sorting time. Logical input counts are calculated from CPU attention masks before device transfer.

The benchmark exposes **--sort-by-length** alongside --generator-batch-size. Result JSONs store generation_profiles_by_repeat. No k-selection rule, corpus, model, precision, output limit, or score/cache setting was changed for this experiment.

The follow-up `reuse_tokenization=True` encodes the full prompt list once without padding, uses those exact truncated IDs to compute the stable length order, and pads the saved IDs within each batch. It retains all tokenizer fields, including attention masks; batch membership, logical input, truncation, and padding shapes are unchanged. Benchmark flag: `--reuse-tokenization`. CPU storage for the encoded prompt collection is required and host-memory peaks were not measured. The default remains False.

### Experiment design

The main comparison replayed all 500 adaptive prompts from the existing batch-size-8 result, in the frozen question order. Each configuration ran four times in one process:

- Original-order batch 8.
- Length-sorted batch 8, restoring outputs to original order.

Order alternated original/sorted, sorted/original, original/sorted, sorted/original. Each measured path used an outside-timing allocator clear and eight warmup prompts; CUDA was synchronized before and after timing. The harness's initial five-prompt warmup was also excluded. Formal runs did not contain profiler hooks.

Peak-memory counters were reset after warmup. Quality evaluation and output comparisons occurred after timing. Both paths performed the same diagnostic bookkeeping. Every call generated 500 answers in 63 batches, with a final partial batch of four. The source JSON checksum, prompt hash, question IDs, and lengths are saved in workload.json.

This is a generation-only replay: retrieval and prompt construction are excluded. A separate full-pipeline integration run is reported below. All data are reused regression data; no new held-out quality evaluation is claimed.

### Results

#### Paired timing and CPU measurements

| Repetition | Original-order batch 8 | Length-sorted batch 8 |
| --- | ---: | ---: |
| 1 | 46.9279 s | 42.1930 s |
| 2 | 47.5893 s | 41.3263 s |
| 3 | 47.5046 s | 42.4104 s |
| 4 | 49.8225 s | 42.0012 s |
| Median | **47.5470 s** | **42.0971 s** |
| Median process CPU time | 44.335938 s | 38.562500 s |

The observed median generation-time reduction was **11.46%**. All four within-round original/sorted comparisons favored sorting. Four repetitions describe this run's variation but do not justify a broad performance-confidence claim.

Length calculation/sorting took 0.2992, 0.2687, 0.2929, and 0.2683 seconds, with a median of **0.2808 s**, already included above. Sorting did not obtain its speedup by omitting preprocessing cost.

#### Padding and memory trade-off

| Metric, 500 prompts | Original-order batch 8 | Length-sorted batch 8 |
| --- | ---: | ---: |
| Logical input tokens | 181,398 | 181,398 |
| Padded input-token slots | 396,112 | 184,748 |
| Padding-only input slots | 214,714 | 3,350 |
| Generated slots, including EOS padding | 14,224 | 13,168 |
| Generation batches | 63 | 63 |
| Peak allocated GPU memory | 1,427.62 MiB | 2,236.85 MiB |
| Peak reserved GPU memory | 1,514 MiB | 2,340 MiB |

Padded input slots decreased **53.36%**, with the logical input content unchanged. Padded output slots also decreased, but these are not useful generated-token counts. Batch membership affects when the longest sequence finishes, and some answer texts changed; the full timing benefit cannot be attributed to input padding alone.

The memory result contradicts a simple assumption that fewer padded slots always lower peak memory. Peak allocated memory increased by **809.24 MiB**, or **56.68%**; peak reserved memory increased by 826 MiB. These are generator-only peaks, identical within each configuration across its four repetitions. Full-RAG peaks including E5 were not measured.

The precise source of the higher allocation peak has not been isolated. Grouping long inputs together and shape-dependent temporary/workspace behavior are possible factors, not confirmed explanations. The implementation remains opt-in rather than replacing the default ordering.

#### Answer quality and output restoration

| Quality on the same 500 prompts | Original order | Length sorted |
| --- | ---: | ---: |
| EM | 2.2% (11/500) | 2.2% (11/500) |
| F1 | 5.856844% | 5.853643% |
| Accuracy | 6.4% | 6.6% |

Every per-question EM outcome stayed the same: no exact-match gains or losses. F1 improved on eight questions and worsened on seven. Its mean difference was -0.003201 percentage points, with a paired-bootstrap 95% interval approximately [-0.1816, +0.1657] percentage points. Accuracy had two gains and one loss. Bootstrap uses 2,000 question resamples and seed=2024; it is not a timing interval or proof of quality equivalence.

Sorting changed 79/500 answer texts. All four repeats of a configuration produced identical predictions. The original-order run reproduced the saved batch-size-8 answers exactly. The sorted full-pipeline adaptive predictions matched all sorted replay predictions after restoration to original question order.

The processing permutation was checked against a stable sort of all 500 original token lengths. Question IDs, original per-question k, and retrieved documents were unchanged. Thus the measured quality difference is not an accidental mismatch between reordered predictions and gold answers.

#### Fixed-output GPU diagnostic

To examine input grouping separately from variable answer lengths, a separate trace used the first 32 frozen prompts. Both configurations used batch size 8 and forced exactly eight output tokens per question: four batches and 256 logical output tokens each. The same CPU/CUDA trace-analysis method and launch correlations from Optimization 4 (CPU/GPU follow-up) were used, without shape, stack, memory, or extra CUDA-event profiling.

| Diagnostic metric, 32 prompts | Original order | Length sorted |
| --- | ---: | ---: |
| Logical input tokens | 13,270 | 13,270 |
| Padded input slots | 27,504 | 17,048 |
| Output slots | 256 | 256 |
| Instrumented host window | 1,527.108 ms | 1,559.084 ms |
| CUDA kernel count | 39,116 | 39,116 |
| Summed GPU kernel time | 684.426 ms | 463.409 ms |
| Prefill kernel count | 4,552 | 4,648 |
| Prefill kernel time | 490.723 ms | 297.501 ms |
| Decode kernel count | 33,112 | 33,016 |
| Decode kernel time | 190.616 ms | 162.832 ms |

With output length fixed, prefill GPU time decreased approximately 39.4%. This supports a real reduction in GPU work from the changed grouping. Total kernel count was unchanged; the mechanism differs from increasing batch size, which substantially reduced launch counts.

The instrumented host window did not improve in this one-shot diagnostic. It has only 32 prompts, an artificial output length, profiler overhead, and a fixed trace order. It is not the formal 500-prompt latency experiment and is retained here rather than omitted. The result is consistent with host dispatch remaining important even when kernel computation becomes shorter.

#### Full-pipeline integration check

The full 500-question FlashRAG runner was executed with batch size 8 and --sort-by-length for fixed top-5 and adaptive retrieval. Each setting used one five-question warmup and **one** measured repetition. This checks integration and answer alignment; it is not a new multi-repeat end-to-end speed estimate.

| Setting | Single end-to-end time | Generation time | EM | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed top-5, length sorted | 46.9032 s | 45.7146 s | 3.2% | 6.7236% | 6.8% |
| Adaptive, length sorted | 44.0882 s | 43.0949 s | 2.2% | 5.8536% | 6.6% |

The adaptive predictions matched the sorted replay, and all retrieved documents and selected-k values matched the earlier unsorted adaptive run. The 11.46% speed claim comes from the paired four-repeat generation experiment, not from comparing this single full run against a previous session.

#### Follow-up: reuse the tokenization used for sorting

The one-factor 500-prompt trial uses the same sorted batch-8 shapes and predicts exactly the same 500 answers in all three repeats. Logical tokens (181,398), padded slots (184,748), generated slots (13,168), original-order restoration, EM/F1/accuracy, and peak GPU allocation are unchanged. Only CPU preparation of model inputs changes.

| Host wall component, median across three repeats | Reference (s) | Reused token IDs (s) |
| --- | ---: | ---: |
| Length preparation and sorting, including one-time encoding when reused | 0.300888 | 0.286199 |
| Batch tokenization or padding | 0.413493 | 0.068752 |
| Sum of those component medians | 0.714381 | 0.354951 |
| Whole generation, including preparation | 43.906815 | 42.195228 |

Batch input preparation fell 83.37%, and the sum of the two component medians fell about 50.31%. This sum is descriptive; independently calculated medians need not sum to a run's median. The observed total-generation median was 3.90% lower, but the ranges overlapped (42.71-44.36 versus 39.12-44.00 s), and the 100-prompt pilot was slower. Roughly 0.36 s of measured preparation savings does not independently explain a 1.71 s whole-generation difference. The supported optimization is removal of duplicate tokenization work; a robust 3.90% end-to-end speedup is not established.

The batch-32 combined trial also preserves the batch-32-only outputs and shapes. Its host-phase medians are 0.277362 s for preparation/sorting, 0.064399 s for padding, and 0.007283 s for text decoding. No GPU-memory reduction is attributed to token reuse; host peak memory was not measured. The setting remains opt-in. Full timing/CPU/memory/quality tables and reproduction commands are shared with the Optimization 4 follow-up in `docs/experiments/host_overhead_500/`.

### Correctness and limitations

Length-aware batching provided a repeatable **11.46% generation-time improvement on this frozen offline workload**, including its own length-calculation and sorting costs. EM outcomes were preserved on these questions, and the fixed-output diagnostic supported lower prefill GPU work.

It is a speed/memory trade-off: measured peak allocation rose 56.68%, some texts changed, and general quality equivalence has not been established. It requires access to a collection of prompts before generation and should not be presented as reduced individual-request latency. The corpus mismatch, reused questions, one hardware/model configuration, and limited timing repetitions remain limitations. This work optimizes execution, not the accuracy of the adaptive-k selection rule.

Seven focused tests passed, including original-order restoration for texts, scores, IDs, and logits. Retrieved documents, k values, prompt content, and question order were validated independently of generation processing order. The setting remains opt-in.

### Artifacts and reproduction

The existing generator test file now verifies length-based grouping, original-order restoration for text/scores/IDs/logits, variable-length input masks, padding counts, and empty input. All **seven focused tests** passed. Compilation and whitespace checks passed. No additional Markdown report was created.

Formal replay and GPU diagnostic command, using the the Reproduction section environment variables:

```powershell
.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_batch8/adaptive_topk_adaptive_gap_20261113_160953_930358.json `
  --sample-num 500 --repeats 4 --warmup-sample-num 8 --skip-profile --padding-study `
  --variants batch8 batch8_length_sorted --output-dir docs/experiments/length_bucketing
```

Full-pipeline integration used the the Reproduction section runner command with --generator-batch-size 8 --sort-by-length --repeats 1 and output directory docs/experiments/nq500_length_sorted. The run ID was `20261113_165554_182515`.

| Evidence | Location |
| --- | --- |
| All paired runs, CPU time, peaks, predictions, and permutations | `docs/experiments/length_bucketing/study.json` and `measurements.json` |
| Derived speed/padding/memory/quality comparison | `docs/experiments/length_bucketing/comparison.json` |
| Per-question original/sorted answer comparison | `docs/experiments/length_bucketing/question_comparison.csv` |
| Fixed-output trace analysis and compressed traces | `padding_trace_analysis.json`, `original_fixed8_trace.json.gz`, and `sorted_fixed8_trace.json.gz` in that directory |
| Complete pipeline check and generation-profile records | `docs/experiments/nq500_length_sorted/` |

## Shared Experimental Setup and Methodology

### Historical checkpoint status before the final core study

This table describes work verified in this workspace. It does not establish the completion status of every task assigned to either team member. Planned dates are not evidence of actual completion dates.

| Planned stage | Completed or verified | Remaining |
| --- | --- | --- |
| FlashRAG setup and fixed-k baseline | Windows environment; Qwen/E5/FAISS pipeline; k=1/5/10 measurements on 17 questions | Measured k=3 comparison; a better-matched quality benchmark |
| Unit and initial larger-program testing | Score API, dataset isolation, and adaptive retrieval tests | Resolve optional-dependency errors in the full repository suite |
| Performance characterization | Stage timings, throughput, tokens, answer metrics; later generator-only peak memory and diagnostic prefill/decode analysis | Full-RAG peak memory and reliable retrieval-quality measurements |
| Adaptive implementation | Score-gap policy, pipeline integration, and boundary tests | Similarity-threshold branch was not verified during this work |
| Larger-program evaluation | 500 new official NQ questions; three repetitions per method | Additional evaluation with a suitable corpus |
| Optimization and final experiments | Controlled profiling, allocator/score ablations, batch-size tests, and 500-question batch-8 regression | Broader quality validation and remaining measurements in Optimization 4 (interpretation and limitations) |
| Final report and cleanup | Project documentation consolidated into this report | Update this report after the final optimization experiments |

### Environment and resource provenance

| Component | Configuration |
| --- | --- |
| Operating system | Windows |
| GPU | NVIDIA GeForce RTX 4070 SUPER; approximately 12 GB; driver 610.88 |
| Isolated environment | `.venv/`, Python 3.11.9; system site-packages disabled |
| PyTorch | 2.11.0+cu128; CUDA runtime 12.8 |
| Transformers / FAISS | 5.7.0 / faiss-cpu 1.15.1 |
| Generator | Qwen/Qwen2.5-0.5B-Instruct on cuda:0 |
| Qwen directory | `models/Qwen2.5-0.5B-Instruct` |
| Qwen revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Qwen download | Eight files; weights contain 988,097,824 bytes |
| Retriever | intfloat/e5-base-v2, available in the local Hugging Face cache |
| Corpus | `examples/quick_start/indexes/general_knowledge.jsonl`; 15,000 documents |
| Index | `examples/quick_start/indexes/e5_Flat.index`; 15,000 vectors, 768 dimensions |
| Device placement | Neural-network inference uses CUDA; FAISS index search uses CPU |
| Original baseline generation settings | Greedy decoding; do_sample=False; maximum 32 generated tokens; batch size=1; later experiments state their batch sizes separately |

The corpus is a general-knowledge collection and is not guaranteed to cover NQ. Retrieved passages were sometimes related to the topic without answering the question. At that early checkpoint corpus coverage and vector/document correspondence were unverified; the final resource audit above adds development lexical coverage and a sixteen-passage alignment check. This experiment is not a standard full-Wikipedia NQ reproduction.

#### Setup history and corrections

Initial checks used the default Python 3.14 CPU environment and inspected only the empty top-level resource directories. Later checks found the corpus, index, and 17-question sample under `examples/quick_start/`, and FAISS in an existing Python 3.11 environment. The benchmark paths were corrected and a separate CUDA environment was created. The resource files did not need to be recovered from the Mac.

CUDA matrix computation, FlashRAG imports, FAISS loading/search, and dependency consistency were verified. Installing dependencies required controlled network access because the tool environment initially disabled pip indexes and had an unavailable restricted proxy. Global user Python, proxy, and pip configuration were not replaced.

### Benchmark reliability work

The runner is `examples/quick_start/perf_baseline.py`. It supports configurable data paths, fixed/adaptive settings, environment metadata, and per-question artifacts.

- Dataset copying originally failed because dynamic attribute lookup intercepted `__deepcopy__`. The runner now deep-copies plain records and constructs an independent Dataset for each repetition.
- `--warmup-sample-num` permits a short warmup without reducing the measured sample.
- Actual Qwen tokenizer counts are recorded separately from the evaluator's GPT-4 tokenizer metric, outside inference timing and with the generator's input-length truncation.
- Results are checkpointed after each setting. Run IDs include microseconds. Missing/empty splits and insufficient sample counts are reported.

### Research questions behind execution profiling

This stage addresses three questions:

- **RQ1:** Which part of the dominant generation stage consumes time?
- **RQ2:** Do allocator-cache management or unnecessary generation scores explain the slowdown?
- **RQ3:** Can generation batching improve offline RAG throughput without reducing the measured answer metrics?

The optimization target is end-to-end offline RAG execution. The score-gap threshold and retrieval algorithm were not changed during these experiments. Batching uses the generator's existing batch support; the implementation contribution is exposing reproducible benchmark controls and measuring their effect, not inventing a new batching algorithm.

### Common frozen-prompt ablation methodology

The diagnostic workload used the first 100 questions from the previously frozen, randomly sampled 500-question adaptive result. They were selected by position, not by timing, correctness, or outcome. Prompts were reconstructed from the saved questions and retained documents using the original template. Reconstructed Qwen input lengths matched the saved counts, and the batch-size-1 baseline reproduced every saved answer.

The workload manifest contains question IDs, input lengths, a source-result SHA256 checksum, and a hash of the reconstructed prompts. Model and hardware remained Qwen2.5-0.5B-Instruct, BF16, RTX 4070 SUPER, PyTorch 2.11.0+cu128, and Transformers 5.7.0. Maximum input length was 1,024 tokens and maximum generated length was 32. Decoding remained greedy.

**Diagnostic run:** three representative prompts with 76, 271, and 977 input tokens were selected by minimum, median, and maximum length within the 100 prompts. CPU/CUDA profiling, memory tracking, named scopes, and CUDA forward events were enabled only for this run. Profiling followed a five-prompt warmup. The three question IDs were test_1284, test_3422, and test_1942.

**Formal timing runs:** profiling and forward hooks were removed. Each variant began with an outside-timing cache clear and two warmup prompts under its own configuration. CUDA was synchronized before and after each measured 100-prompt generation call. Peak-memory counters were reset after warmup. Each variant ran three times; output comparison and memory-counter collection occurred outside the timing interval.

The allocator/score study used the interleaved orders baseline/cache/score, baseline/score/cache, and score/baseline/cache. The batching study used a balanced cyclic order: batch1/batch4/batch8, batch4/batch8/batch1, and batch8/batch1/batch4. Formal timing excludes model loading and retrieval. Three timing repetitions are insufficient for a reliable performance confidence interval; all observations and medians are reported instead.

Peak allocated memory measures live tensor allocations managed by PyTorch; peak reserved memory includes allocator-held blocks. These generator-only measurements include Qwen but not a loaded E5 model. They are not total GPU process memory or full-RAG peak memory.

### Original joint allocator/score ablation record

All variants used batch size 1 and the same 100 prompts. Baseline clears the CUDA allocator cache between batches and does not collect generation scores. Only the named switch changes in each ablation.

| Variant | Three measured times, seconds | Median | Peak allocated | Peak reserved | Device allocations / frees per run |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | 33.0056 / 32.4863 / 32.9346 | 32.9346 s | 1,103.65 MiB | 1,118 MiB | 339 / 344 |
| Retain allocator cache | 33.3201 / 31.8276 / 32.2539 | 32.2539 s | 1,103.65 MiB | 1,124 MiB | 2 / 0 |
| Score collection on | 33.3534 / 32.4962 / 31.8852 | 32.4962 s | 1,114.23 MiB | 1,138 MiB | 372 / 377 |

Memory/counter values shown are per-variant medians; they were identical across its three runs. Every variant reproduced all 100 saved texts in every repetition.

Retaining the allocator cache reduced device allocation activity substantially but improved the observed median by only about 2.07%, with overlapping run ranges. It increased reserved memory by 6 MiB and did not reduce peak allocated memory. This evidence does not establish allocator clearing as the dominant latency bottleneck; the historical default was retained.

Skipping score collection reduced peak allocated memory by approximately 10.58 MiB relative to score-on. There was no demonstrated latency benefit: the score-on median was slightly lower in this small study, and run ranges overlapped. Optimization 2 can be reported as avoiding unnecessary score storage, but not as a measured latency improvement here.

The distinction between allocated and reserved memory follows the [PyTorch 2.11 empty_cache documentation](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.empty_cache.html). Profiling uses the [PyTorch 2.11 profiler API](https://docs.pytorch.org/docs/2.11/profiler.html).

### Historical Mac retrieval and Windows smoke tests

#### Mac CPU retrieval

Source: the previous report updated October 24, 2026; not rerun on this machine.

| Measurement | Historical result |
| --- | ---: |
| Retriever initialization | 3.342 s |
| First five-question retrieval batch | 2.278 s |
| Warm batches after two warmups | 0.092 / 0.084 / 0.090 / 0.084 / 0.087 s |
| Warm median | 0.087 s per five-question batch |
| Amortized cost | Approximately 17.5 ms/query |
| Retrieval-only throughput | Approximately 57 queries/s |

Reported environment: Python 3.10.22, PyTorch 2.13.0, Transformers 5.19.0; E5 used CPU. Corpus size was 15,000 documents, approximately 6.4 MB, with an approximately 44 MB index. Qwen generation was not measured then.

These numbers do not establish a PC-versus-Mac speedup. Amortized batch time is not independent single-request latency. The original report described older 3-warmup/10-repeat defaults; the current runner defaults are one warmup and three measured repeats.

#### Windows five-question smoke test

Fixed k=5; one warmup and one measured repetition:

| Metric | Result |
| --- | ---: |
| Retrieval | 0.013400 s |
| Prompt construction | 0.009423 s |
| Generation | 1.610247 s |
| End-to-end batch time | 1.633225 s |
| Throughput | Approximately 3.061 queries/s |
| EM / F1 / accuracy | 0 / 0.012903 / 0 |
| Mean evaluator input tokens | 449 |

This verified the real-model pipeline. It was too small and had too few repetitions for a performance claim. An earlier attempt failed before inference during dataset copying and left a header-only summary CSV.

### Validation and remaining limitations

| Check | Result |
| --- | --- |
| Score API, allocator-cache control, real HF model, trace analysis, and length-sort restoration | Seven focused tests passed after the length-bucketing work |
| Dataset isolation and short warmup subset | One test passed with multiple isolation/subset assertions |
| Adaptive policy and pipeline | Eleven tests passed |
| Isolated environment, real CUDA operation, FAISS load/search | Passed |
| Dependency consistency | pip check found no broken requirements |
| 500-question artifacts | Checksums, all six ID sequences/counts, selected-k alignment, decision recomputation, and CSV row count verified |
| Full repository suite | A previous 31-test run had two cache-test subtest errors caused by a lazy Transformers visual import requiring optional torchvision; full-suite success is not claimed |

The 500-question PowerShell log contains NativeCommandError formatting from redirected native stderr progress output, and the shell returned nonzero. Python reached `[done]`, wrote all results, and had no inference traceback or evaluator error. Artifact validation confirmed completion.

Optimization 4 and the shared ablation record subsequently adds generator-only peak memory, diagnostic prefill/decode measurements, and a real-Qwen Optimization 2 ablation. Full-RAG peak memory and standard retrieval recall/precision remain unmeasured. The general-knowledge corpus mismatch and the fixed order of full RAG settings limit interpretation.

- Added clear_cuda_cache control to HFCausalLMGenerator.generate(), preserving the historical True default.
- Added --generator-batch-size and --retain-cuda-cache benchmark controls and recorded their values in metadata. The initial tested throughput configuration was batch size 8; the later follow-up also measures batch 32, retaining the historical cache-clear behavior.
- Added one executable study harness, `scripts/profile_generation.py`, for frozen-prompt replay, separate diagnostic profiling, interleaved ablations, memory counters, and output comparison. No additional Markdown reports were created.
- Extended the existing generator tests to verify that disabling cache clearing does not invoke empty_cache or change text. All five focused tests passed at that stage; the later expanded generator suite contains seven passing tests.
- Updated batch-aware token-log analysis to avoid treating padded output widths as per-question token counts.
- Verified all 500 IDs, all six optimized repetitions, identical retrieved documents and k, score-gap decision recomputation, stable predictions, and the compressed profiler trace.

The raw PowerShell logs contain the same native-stderr progress formatting described in the shared Validation and remaining limitations section; complete study/result artifacts and explicit consistency checks confirmed completion.

## Reproduction and Evidence Index

Run from the repository root. The isolated interpreter is `.venv/Scripts/python.exe`. Activation is optional; direct interpreter invocation also works if PowerShell blocks Activate.ps1.

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_CACHE = Join-Path (Get-Location).Path 'models/datasets-cache'
$env:TIKTOKEN_CACHE_DIR = Join-Path (Get-Location).Path 'models/tiktoken-cache'
$taskE5 = Join-Path $env:USERPROFILE '.cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd'
.\.venv\Scripts\python.exe examples/quick_start/perf_baseline.py `
  --model_path models/Qwen2.5-0.5B-Instruct --retriever_path $taskE5 `
  --data-dir datasets/nq_500_heldout --k-values 5 --adaptive --adaptive-gap 0.01 `
  --sample-num 500 --warmups 1 --warmup-sample-num 5 --repeats 3 `
  --output-dir docs/experiments/nq500_gap001
```

Use a new output directory for a new run: the analysis script expects exactly one result per setting in its directory.

Environment reconstruction and resource download require network access:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-baseline.txt
.\.venv\Scripts\python.exe scripts/check_rag_env.py
.\.venv\Scripts\python.exe scripts/download_qwen.py
.\.venv\Scripts\python.exe scripts/prepare_nq_benchmark.py --sample-num 500 --seed 2024 --revision bcafb8dd07d453be3cbeeeb3f78be1841bddf92c
```

PyTorch installation follows the [official installation instructions](https://pytorch.org/get-started/locally/). This environment covers the text baseline; optional UI, vLLM, training, and multimodal backends were not installed.

Analysis and focused tests:

```powershell
.\.venv\Scripts\python.exe scripts/summarize_adaptive_benchmark.py docs/experiments/nq500_gap001
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_adaptive_topk.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_generator_score_path.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_perf_dataset_clone.py -v
```

The README index-builder model_path argument identifies E5; the benchmark model_path identifies Qwen, and retriever_path identifies E5. Corpus and index must match. First-time tiktoken cache population can require network access even when the model Hub is offline.

Paths below are relative to the repository root. Supporting artifacts are retained for verification; this file explains their purpose and conclusions.

| Evidence | Location / run ID |
| --- | --- |
| Five-question smoke test | `docs/experiments/qwen_smoke/`; successful run `20261113_145929`; `145835` contains only a failed attempt's CSV header |
| Seventeen-question default comparison | `docs/experiments/adaptive_gap_v1/`; `20261113_150718_093378` |
| Seventeen-question sensitivity run | `docs/experiments/adaptive_gap_sensitivity_001/`; `20261113_150923_175397` |
| Complete 500-question run | `docs/experiments/nq500_gap001/`; `20261113_151722_140059` |
| Paired analysis and question-level differences | `comparison.json` and `question_comparison.csv` in the 500-question directory |
| Raw 500-question records | Summary CSV and fixed/adaptive JSON files in that directory |
| Execution evidence | `run.log` and `analysis_console.txt` in that directory |
| Data revision, checksums, and selected IDs | `docs/nq_500_manifest.json` |
| Environment and model provenance | `docs/environment_check.json`, `docs/environment-lock.txt`, `docs/qwen_download.json` |

The earlier root report, separate adaptive/environment guides, obsolete profiling report, and mistakenly created `doc/` report have been consolidated. Their current methods, results, setup history, corrections, and remaining work are presented above in English. Raw experiment files and executable/test code are preserved. Existing framework documentation and README files remain available for using FlashRAG.

Future project performance findings should be added to this file rather than to additional report Markdown files.

Use the same environment variables as the Reproduction section. Initial allocator/score study:

```powershell
.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_gap001/adaptive_topk_adaptive_gap_20261113_151722_140059.json `
  --sample-num 100 --repeats 3 --reverse-odd-rounds `
  --output-dir docs/experiments/generation_optimization
```

Batching study:

```powershell
.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_gap001/adaptive_topk_adaptive_gap_20261113_151722_140059.json `
  --sample-num 100 --repeats 3 --skip-profile --variants baseline batch4 batch8 `
  --output-dir docs/experiments/generation_batching
```

Full regression uses the the Reproduction section command with **--generator-batch-size 8** and output directory **docs/experiments/nq500_batch8**. The run ID was `20261113_160953_930358`. Use fresh directories for new experiments so earlier evidence remains intact.

| Evidence | Location |
| --- | --- |
| Frozen workload, three-prompt profile, allocator/score runs | `docs/experiments/generation_optimization/` |
| Complete diagnostic trace | `profile_trace.json.gz` in that directory; decompress to inspect the Chrome trace |
| Operator summary, per-run rows, full study metadata | `profile_operators.txt`, `measurements.json`, `study.json` in that directory |
| Batch-size 1/4/8 timings and memory | `docs/experiments/generation_batching/study.json` |
| Optimized 500-question results and within-run analysis | `docs/experiments/nq500_batch8/` |
| Original-versus-optimized quality, time ratios, and validation | `docs/experiments/nq500_batch8/optimization_summary.json` |

### Reproduce the host-overhead follow-ups (Optimizations 3, 4, and 5)

These commands use the shared environment variables above. Write reruns to a new output directory. The pilot uses the 100-question prefix only; the formal replay uses all 500 questions and performs separate CPU/CUDA diagnostics after formal timing.

```powershell
.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_batch8/adaptive_topk_adaptive_gap_20261113_160953_930358.json `
  --sample-num 100 --repeats 3 --warmup-sample-num 8 --skip-profile --cpu-threads 8 `
  --variants batch8_length_sorted sorted_reuse_tokens sorted_decode_new sorted_threads1 sorted_threads2 sorted_threads4 sorted_combined `
  --output-dir docs/experiments/host_overhead_pilot

.\.venv\Scripts\python.exe scripts/profile_generation.py `
  --source-json docs/experiments/nq500_batch8/adaptive_topk_adaptive_gap_20261113_160953_930358.json `
  --sample-num 500 --repeats 3 --warmup-sample-num 8 --skip-profile --execution-study --cpu-threads 8 `
  --variants batch8_length_sorted sorted_reuse_tokens sorted_decode_new sorted_batch32 sorted_batch32_combined `
  --output-dir docs/experiments/host_overhead_500

.\.venv\Scripts\python.exe scripts/summarize_adaptive_benchmark.py docs/experiments/host_overhead_500 --replay

.\.venv\Scripts\python.exe examples/quick_start/perf_baseline.py `
  --model_path models/Qwen2.5-0.5B-Instruct --retriever_path $taskE5 `
  --data-dir datasets/nq_500_heldout --k-values 5 --adaptive --adaptive-gap 0.01 `
  --sample-num 500 --warmups 1 --warmup-sample-num 8 --repeats 1 `
  --generator-batch-size 32 --sort-by-length --reuse-tokenization --decode-generated-only --cpu-threads 8 `
  --output-dir docs/experiments/nq500_host_optimized
```

| Follow-up evidence | Location |
| --- | --- |
| All 21 pilot measurements, including negative CPU-thread trials | `docs/experiments/host_overhead_pilot/measurements.json` and `comparison.json` |
| All 15 formal 500-prompt measurements, predictions, CPU clocks and memory | `docs/experiments/host_overhead_500/measurements.json`, `study.json`, and `comparison.json` |
| Frozen IDs, prompt hash and source checksum | `docs/experiments/host_overhead_500/workload.json` |
| Paired quality differences and bootstrap intervals | `docs/experiments/host_overhead_500/quality_comparison.json` |
| Fixed-eight-output CPU/CUDA analysis and three compressed traces | `docs/experiments/host_overhead_500/execution_trace_analysis.json` and `*_fixed8_trace.json.gz` |
| Complete fixed/adaptive pipeline records and retrieval/quality validation | `docs/experiments/nq500_host_optimized/` |
| Nine passing focused generator/trace tests | `docs/experiments/host_overhead_500/generator_tests.txt` |

The expanded nine-test suite checks original text/score APIs, cache control, return dictionaries, real Hugging Face generation, length-order restoration, continuation decoding, and reuse with a real fast tokenizer (truncation, padding, attention masks, output IDs/logits/scores, and empty inputs). All nine tests passed after the changes. Compilation and whitespace checks passed. Earlier references to five or seven tests describe the suites at those earlier experiment stages. No additional Markdown report was created.

## Planned Follow-Ups Within the Five Directions

The following are analysis proposals, not additional completed optimization IDs or measured successes:

| Existing optimization | Planned follow-up | Current status |
| --- | --- | --- |
| Optimization 1 | Calibrate the selection rule on development data; check corpus coverage; examine FAISS OpenMP thread counts separately | Not executed |
| Optimization 2 | Continue measuring score-storage cost when changing model or output length; preserve score-returning API behavior | No new experiment executed |
| Optimization 3 | Fresh-process CPU-thread confirmation only if a candidate improves; isolate allocator effects at larger batch sizes | 100-prompt intra-op screening completed; no observed thread-tuning speedup |
| Optimization 4 | Isolate the fixed-top-5 quality regression; validate compiler support before testing decode fusion or torch.compile/CUDA graphs | Batch 32 and continuation decoding measured; compiler/fusion experiments not executed |
| Optimization 5 | Attribute the higher allocation peak; measure host-memory cost and confirm token-reuse timing on an independent workload | Token reuse implemented and measured on 500 prompts; local preparation overhead reduced, whole-generation gain inconclusive |

OpenMP affects supported native CPU work; it does not automatically parallelize Python dispatch or CUDA kernels. A GCC/Clang -O3 experiment requires an identified native source component and a working compiler toolchain; setting a flag on the Python process does not rebuild installed PyTorch/FAISS binaries. The intra-op thread screening under Optimization 3 did not establish an OpenMP-thread-tuning speedup. No native -O3 experiment or speedup is claimed.

Future updates should retain these five IDs. Add new trials under the matching optimization, record negative results and costs, and introduce a new optimization ID only when a genuinely separate direction is implemented and evaluated.

<details>
<summary>Historical experiment plan before the executed studies</summary>

This plan preceded the experiments in Optimization 4 and the shared ablation record. Profiling, allocator/score ablations, batching tests, and a full 500-question regression have now been executed. A separate forced-32-token experiment and isolated host-to-device timing remain unperformed. The descriptions below document the planned controls; Optimization 4 and the shared ablation record specifies what was actually measured.

#### Test A: fixed-prompt generation profiling

Freeze approximately 100 prompts and their IDs/lengths from the existing workload. Use them for performance diagnosis without retuning the retrieval policy. Keep model, precision, device, input content, and generation settings fixed.

| Measurement | Method | Question |
| --- | --- | --- |
| Tokenization, CPU-to-GPU transfer, output decoding | Separate synchronized timing where appropriate | Is avoidable host-side work significant? |
| Prefill and per-token decode | A separate `torch.profiler` CPU/CUDA run | Which model phase dominates? |
| Actual output length and EOS/limit behavior | Per-question token records | Did different answers change the work performed? |
| Peak GPU allocation and reservation | Reset/read CUDA peak-memory statistics | What is the memory cost of each path? |

Use normal greedy output to characterize real behavior. Add a separate controlled experiment forcing exactly 32 generated tokens to isolate output-length effects. That controlled experiment must not be presented as normal answer-quality evaluation. Profiler overhead must be excluded from formal latency measurements.

#### Test B: one-change-at-a-time A/B experiments

Choose changes based on Test A:

1. Compare per-batch empty_cache with a path that omits it; measure latency and memory.
2. Compare score collection on/off to quantify Optimization 2.
3. If profiling indicates insufficient batching, compare batch sizes 1/4/8 for throughput, memory, and output differences.

These are proposed experiments, not measured successes. Use the same prompts, warmups, CUDA synchronization, at least three repetitions, and alternating before/after order. Check greedy-text consistency for score/memory-management changes and answer quality for batch-size changes.

#### Test C: full RAG regression

Apply a demonstrated generator optimization equally to fixed and adaptive methods, then rerun the same 500 questions. Compare end-to-end latency, throughput, memory, tokens, and EM/F1/accuracy.

Reusing these questions for performance regression must be labeled as reuse. If retrieval thresholds are tuned further, use development data and a new final evaluation sample. Quality tolerance has not yet been agreed. Shorter output limits or changed data/corpora cannot be reported as equivalent-workload implementation speedups.

</details>
