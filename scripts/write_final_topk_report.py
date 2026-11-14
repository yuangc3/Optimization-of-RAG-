"""Validate final-study artifacts and refresh the single English report source.

Numerical tables are generated from saved measurements, never hand-entered.
The submission block is exported by export_project_report.py; all earlier
optimization histories remain below it in the same Markdown file.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/experiments/final_topk"
REPORT = ROOT / "docs/performance_profilling_report.md"


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    rows, splits, metadata = read("evaluation_measurements.json"), read("split_manifest.json"), read("metadata.json")
    frozen = read("frozen_policies.json")
    expected = {"fixed_1", "fixed_3", "fixed_5", "fixed_10", "score_gap", "similarity_threshold"}
    assert len(rows) == 18 and {r["setting"] for r in rows} == expected
    assert sha(EVIDENCE / "frozen_policies.json") == metadata["policy_sha256"]
    assert sha(EVIDENCE / "split_manifest.json") == frozen["manifest_sha256"]
    assert sha(EVIDENCE / "calibration.json") == frozen["calibration_sha256"]
    assert sha(ROOT / "flashrag/retriever/adaptive_topk.py") == metadata["policy_source_sha256"]
    assert sha(ROOT / "flashrag/generator/generator.py") == metadata["generator_source_sha256"]
    assert sha(ROOT / "scripts/run_final_topk_study.py") == metadata["runner_sha256"]
    audit = read("resource_audit.json")
    for key, path in (("corpus_sha256", ROOT / "examples/quick_start/indexes/general_knowledge.jsonl"),
                      ("index_sha256", ROOT / "examples/quick_start/indexes/e5_Flat.index")):
        assert sha(path) == metadata[key] == audit[key]
    dev, test = splits["splits"]["dev"], splits["splits"]["test"]
    assert not set(dev["ids"]) & set(test["ids"])
    for split in (dev, test):
        assert sha(ROOT / split["file"]) == split["sha256"]
    refs = {r["repeat"]: r for r in rows if r["setting"] == "fixed_5"}
    stability = {}
    for name in expected:
        group = [r for r in rows if r["setting"] == name]
        assert sorted(r["repeat"] for r in group) == [1, 2, 3]
        for row in group:
            assert [s["id"] for s in row["samples"]] == test["ids"]
            if name in ("score_gap", "similarity_threshold"):
                for sample, reference in zip(row["samples"], refs[row["repeat"]]["samples"]):
                    k = sample["selected_k"]
                    assert 2 <= k <= 5
                    assert sample["retrieved_ids"] == reference["retrieved_ids"][:k]
                    assert sample["decision"]["selected_k"] == k
        stability[name] = sum(a["prediction"] != b["prediction"] for row in group[1:]
                              for a, b in zip(group[0]["samples"], row["samples"]))
    result = {"status": "passed", "settings": 6, "evaluation_questions": 500, "repetitions": 3,
              "evaluation_generations": 9000, "development_questions": 200, "development_generations": 1200,
              "all_ids_in_original_order": True, "adaptive_documents_match_fixed5_prefix": True,
              "frozen_policy_and_resource_hash_checks": True, "prediction_disagreements": stability}
    (EVIDENCE / "final_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    validation = verify()
    final, dev, audit, frozen, split = (read(n) for n in ("evaluation_summary.json", "development_summary.json", "resource_audit.json", "frozen_policies.json", "split_manifest.json"))
    names = ["fixed_1", "fixed_3", "fixed_5", "fixed_10", "score_gap", "similarity_threshold"]
    labels = {"fixed_1": "Fixed k=1", "fixed_3": "Fixed k=3", "fixed_5": "Fixed k=5", "fixed_10": "Fixed k=10", "score_gap": "Adaptive score-gap", "similarity_threshold": "Adaptive threshold"}
    baseline = final["fixed_5"]
    quality = ["| Method | Mean k | Answer hit (%) | EM (%) | F1 (%) | Accuracy (%) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    cost = ["| Method | Input tokens/query | Median time (s) | Throughput (q/s) | Peak allocated (MiB) |", "| --- | ---: | ---: | ---: | ---: |"]
    repeat = ["| Method | Repetition 1 / 2 / 3 time (s) | Median CPU (s) |", "| --- | --- | ---: |"]
    stages = ["| Method | Retrieval (s) | Selection (s) | Prompt (s) | Generation (s) |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in names:
        r = final[name]
        quality.append(f"| {labels[name]} | {r['mean_k']:.3f} | {r['retrieval_answer_hit']*100:.2f} | {r['metrics']['em']*100:.2f} | {r['metrics']['f1']*100:.3f} | {r['metrics']['acc']*100:.2f} |")
        cost.append(f"| {labels[name]} | {r['mean_input_tokens']:.3f} | {r['median_seconds']:.4f} | {r['throughput_qps']:.3f} | {r['peak_allocated_mib']:.2f} |")
        repeat.append(f"| {labels[name]} | {' / '.join(format(x,'.4f') for x in r['times'])} | {r['median_process_cpu_seconds']:.4f} |")
        stages.append(f"| {labels[name]} | " + " | ".join(f"{r['median_stages'][k]:.6f}" for k in ("retrieval_s", "selection_s", "prompt_s", "generation_s")) + " |")
    conclusions = []
    paired = ["| Policy vs fixed-5 | EM gains / losses | F1 change (pp) | Paired F1 95% interval (pp) |", "| --- | ---: | ---: | --- |"]
    for name in ("score_gap", "similarity_threshold"):
        r = final[name]
        document_cut = 100 * (1 - r["mean_k"] / 5)
        token_cut = 100 * (1 - r["mean_input_tokens"] / baseline["mean_input_tokens"])
        p = r["paired_vs_fixed5"]
        primary = r["metrics"]["em"] >= baseline["metrics"]["em"] - 1e-12 and r["metrics"]["f1"] >= baseline["metrics"]["f1"] - 1e-12
        all_metrics = primary and r["metrics"]["acc"] >= baseline["metrics"]["acc"] - 1e-12
        overlap = max(min(r["times"]), min(baseline["times"])) <= min(max(r["times"]), max(baseline["times"]))
        conclusions.append(f"**{labels[name]}:** mean documents fall {document_cut:.2f}% and truncated input tokens fall {token_cut:.2f}% versus fixed-5. Observed EM and F1 {'meet' if primary else 'do not meet'} the frozen quality criterion. All three answer metrics {'are at least the fixed-5 values' if all_metrics else 'are not all preserved; inspect the accuracy/quality differences above'}. The measured median time change is {r['time_reduction_vs_fixed5_percent']:+.2f}% reduction. {'Run ranges overlap, so a robust end-to-end latency benefit is not established.' if overlap else 'Run ranges do not overlap, but three repetitions on one desktop do not establish general performance.'}")
        if r["retrieval_answer_hit"] < baseline["retrieval_answer_hit"]:
            conclusions.append(f"For {labels[name]}, lexical retrieval answer hit drops from {baseline['retrieval_answer_hit']*100:.2f}% to {r['retrieval_answer_hit']*100:.2f}% on final questions. The zero-loss calibration constraint held on development data and did not guarantee zero loss on unseen data.")
        ci = p["f1"]["bootstrap_95_ci"]
        paired.append(f"| {labels[name]} | {p['em']['better']} / {p['em']['worse']} | {p['f1']['difference']*100:+.4f} | [{ci[0]*100:+.4f}, {ci[1]*100:+.4f}] |")
    gap = frozen["policies"]["score_gap"]["config"]
    threshold = frozen["policies"]["similarity_threshold"]["config"]
    paragraph = "\n\n".join(conclusions)
    body = f'''<!-- SUBMISSION_START -->
# Lightweight Adaptive Top-K Retrieval in FlashRAG

Yidong Zhang and Yuang Cai | CS598APE final project | November 13, 2026

## Abstract

We implement and evaluate training-free Adaptive Top-K retrieval using similarity thresholds and adjacent score gaps in FlashRAG. The goal is to reduce context-processing cost while maintaining answer quality. Early aggressive pruning reduced context but lost correct answers. We therefore introduce minimum-k and maximum-k guards, a weak-similarity fallback, and development-only calibration that preserves every observed fixed-5 answer-string hit. We exclude all earlier questions, calibrate on 200 development questions, freeze two policies, and evaluate six methods on 500 new questions with three timing repetitions each. The final threshold policy reduces documents by {100*(1-final['similarity_threshold']['mean_k']/5):.2f}% and input tokens by {100*(1-final['similarity_threshold']['mean_input_tokens']/baseline['mean_input_tokens']):.2f}% versus fixed-5, with observed EM changing from {baseline['metrics']['em']*100:.1f}% to {final['similarity_threshold']['metrics']['em']*100:.1f}% (one additional exact match). A stable latency improvement is not demonstrated, and fixed-3 remains a stronger EM/F1 operating point. The report separates Adaptive-K findings from supporting generation-system speedups and reports negative results. The experiments use a limited 15,000-document general-knowledge corpus; answer-hit measures are lexical proxies, and full-Wikipedia benchmark performance is not claimed.

## 1. Problem, scope, and related work

A fixed retrieval budget can provide unnecessary or insufficient context for individual questions. We study whether scores already produced by a dense retriever provide a cheap signal for selecting a document prefix. The primary contribution is a modular implementation and reproducible cost/quality evaluation of two lightweight selection rules; no additional selector model is trained and inference makes no LLM-feedback calls.

RAG combines external retrieval with generation [1]. Self-RAG learns retrieval and reflection behavior [2]; Adaptive-RAG learns to select retrieval strategies based on question complexity [3]; DynamicRAG learns dynamic reranking using generation-quality feedback [4]. Our scope is narrower: deterministic, score-only prefix selection within the FlashRAG toolkit [5]. We do not reproduce these systems, train comparable models, claim algorithmic novelty for standard threshold/gap heuristics, or claim state-of-the-art results.

The implementation first retrieves a bounded candidate pool and then chooses the number of documents sent to the generator. It does not terminate the FAISS distance scan early. Consequently, context reduction and candidate-search acceleration are separate hypotheses. A successful system-throughput optimization does not establish that the adaptive selection rule is better.

## 2. Implementation and optimization decisions

### Optimization 1: Lightweight Adaptive Top-K selection (primary)

For score-gap selection, let ranked similarity scores be s1 >= s2 >= ... >= sM. Compute adjacent gaps g_i = s_i - s_(i+1), consider boundaries between min_k and M-1, and cut after the largest gap if it reaches a calibrated threshold. Otherwise retain fallback_k. The frozen final policy uses min_k={gap['min_k']}, max_k={gap['max_k']}, fallback_k={gap['fallback_k']}, and gap threshold={gap['gap_threshold']}. Ties choose the earliest allowed boundary. Both score directions are supported and tested, although performance experiments use inner-product similarities.

The similarity-threshold rule counts the ranked candidates meeting an absolute threshold, clamps the result to the minimum/maximum budget, and uses fallback_k if no score qualifies. The weak-query fallback avoids interpreting low similarity as evidence that very little context is sufficient. Its frozen threshold is {threshold['score_threshold']:.9f}, with min_k={threshold['min_k']}, max_k={threshold['max_k']}, and fallback_k={threshold['fallback_k']}. Scores are not calibrated correctness probabilities; this numeric threshold is specific to the E5/index configuration.

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

The source is the pinned FlashRAG NQ JSONL, revision `{split['source_revision']}`, containing 3,610 questions. We exclude the prior {split['excluded_question_count']} unique questions, sample using seed 598, and allocate 200 development plus 500 evaluation questions with no normalized-question or ID overlap. These are project-specific splits drawn from the official test JSONL, not the official NQ development split. IDs and SHA-256 values are preserved in split_manifest.json.

Calibration considers 81 configurations across the two policy families. The criterion is zero lost individual development answer-string hits relative to fixed-5, then minimum mean k; ties prefer a larger minimum budget and canonical configuration order. It uses development retrieval results and gold aliases only. No LLM answer or final-test answer is used to select the policies. The chosen rules are frozen before development generation and final evaluation. Calibration retains {dev['fixed_5']['retrieval_answer_hit']*200:.0f}/200 development hits; mean k becomes {dev['score_gap']['mean_k']:.3f} for score-gap and {dev['similarity_threshold']['mean_k']:.3f} for threshold.

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
| Corpus/index | {audit['corpus_documents']:,} local general-knowledge documents and {audit['index_vectors']:,} vectors, dimension {audit['dimension']} |

Corpus/index counts match. Sixteen deterministically sampled passages were re-encoded, and corresponding reconstructed vectors agree to cosine greater than 0.99. This verifies sampled alignment, not every vector. Only {audit['development_corpus_lexical_hits']}/200 development questions have an answer-string hit anywhere in the corpus. The corpus is not the NQ Wikipedia collection, which limits retrieval coverage and explains why these measurements must not be compared directly with published NQ leaderboard scores.

### Timing and metrics

Each method receives an eight-question warmup outside timing. All six methods run for three measured repetitions, with method order rotated and reversed according to a fixed schedule. CUDA synchronization encloses the complete inference interval. Search, selection, prompt construction, generation, process CPU time, and peak allocated/reserved GPU memory are recorded. Evaluation and token counting run outside inference timing. No profiler or forward hooks are active in final timed runs. Memory figures include both loaded neural models; host peak RSS and minimum hardware requirements were not measured.

Answer quality uses repository-normalized EM, token F1, and answer-containing accuracy. Retrieval answer hit tests whether a nonempty normalized gold alias occurs at token boundaries in any selected document. This is a lexical proxy; gold document relevance labels are unavailable. Context tokens use the actual generator tokenizer after its 1,024-token truncation. Padded input/output slots are separately recorded and are not useful-token counts. Throughput is total questions divided by median batch completion time, not inverse single-request latency.

There are 9,000 final generations but only 500 independent final questions. Paired quality intervals use 2,000 question bootstrap resamples, seed 598. They describe uncertainty on these questions, not timing variance or formal equivalence; sparse binary outcomes and degenerate zero-change intervals require caution.

## 4. Final Adaptive Top-K results

### Quality and selected context

{chr(10).join(quality)}

### Execution cost

{chr(10).join(cost)}

{chr(10).join(repeat)}

Stage medians are computed independently and need not sum to the median total:

{chr(10).join(stages)}

### Paired interpretation and adoption

{paragraph}

{chr(10).join(paired)}

Fixed k=3 must be retained in the interpretation: its observed EM is {final['fixed_3']['metrics']['em']*100:.2f}% and F1 is {final['fixed_3']['metrics']['f1']*100:.3f}%, compared with fixed-5 EM {baseline['metrics']['em']*100:.2f}% and F1 {baseline['metrics']['f1']*100:.3f}%. Reducing context relative to fixed-5 does not establish that an adaptive method dominates every fixed-k operating point. The tables report the complete trade-off rather than choosing only a favorable comparator.

The contribution is a working score-only adaptive implementation, guarded development calibration, and a fully reported evaluation. Candidate-search acceleration and general quality-preserving latency improvements are not presumed from context reduction. Existing defaults remain unchanged, and calibrated policies are explicit experiment configurations rather than universal replacements for fixed-k.

## 5. CPU/GPU characterization and remaining bottlenecks

Earlier profiling established that generation dominates full RAG time; score selection accounts for milliseconds per 500 questions. Controlled CPU/CUDA traces showed many small launches and host-side generation activity. A fixed-eight-output-token diagnostic reduced kernel count from 39,116 at sorted batch 8 to 10,115 at batch 32, but kernel duration increased from 467.403 to 813.181 ms because padding increased. A lower wall time does not imply less GPU computation. Synchronization call counts also fell while host synchronization intervals increased. These separate diagnostic captures support launch/batching hypotheses, not exact attribution of unprofiled runtime or SM occupancy.

Windows counters on a separate 500-question workload measured median process CPU 38.97 s at batch 8 and 15.02 s at batch 32, concentrated in the main thread. A separate 32-prompt cProfile run with GetThreadTimes identified linear/attention entry points, generation-loop completion checks, and module dispatch. CPU time at a CUDA entry point includes host/driver behavior, not GPU kernel execution; busy waiting can consume CPU. Cumulative function times overlap. No RMSNorm/RoPE fusion, OpenMP C++ implementation, or -O3 compilation experiment was performed.

WPR native CPU-stack capture was attempted normally and through tool escalation, but Windows rejected the profiling privilege with error 0xc5585011. A tested administrator-only capture entry point is supplied; actual WPR ETL capture and native wait-cause analysis remain incomplete. This restriction is stated rather than substituting Python profiling for a successful WPR capture.

## 6. Testing, artifact layout, and reproduction

The complete local discovery suite passed **56 tests**, with no skipped or failed tests. It covers score direction, threshold inclusivity, bounds, ties, empty/short inputs, nonfinite and unsorted scores, policy configuration, pipeline integration, FAISS result handling/cache behavior, document metrics, dataset handling, generator scores/text/dictionaries, batching order and padding, CPU trace interval accounting, WPR session ownership/cleanup, development/evaluation separation, and calibration safeguards. Small real Hugging Face generation tests run on CPU without downloading weights. The initial two warning-test errors were caused by unittest scanning lazy optional vision imports; the test now captures and checks the actual retrieval warning without importing unrelated vision modules. The initial failure log and final passing log are retained.

Larger-program validation checks all six methods, all three final repetitions, all 500 IDs in original order, selected-document counts and adaptive prefixes against fixed-5, frozen hashes, and repeated prediction stability. Prediction disagreement counts are {json.dumps(validation['prediction_disagreements'], sort_keys=True)}. Development generation adds 1,200 calls across six methods. This is {validation['evaluation_generations'] + validation['development_generations']:,} new generation evaluations on 700 distinct questions; it is not a claim of 10,200 independent examples.

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
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe scripts/run_final_topk_study.py --stage all --output-dir docs/experiments/reproduction
.\\.venv\\Scripts\\python.exe scripts/export_project_report.py
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
'''
    text = REPORT.read_text(encoding="utf-8")
    pattern = r"<!-- SUBMISSION_START -->.*?<!-- SUBMISSION_END -->\s*"
    if "<!-- SUBMISSION_START -->" in text:
        text = re.sub(pattern, lambda _: body + "\n", text, flags=re.S)
    else:
        location = text.index("## Project objective")
        text = text[:location] + body + "\n\n" + text[location:]
    # Preserve historical evidence while correcting the current status summary.
    text = text.replace("| Similarity-threshold Adaptive-K | No independent threshold-policy implementation found in the inspected local policy/runner | Implement or integrate the teammate's branch and test it; an uninspected external branch cannot be declared complete or absent |",
                        "| Similarity-threshold Adaptive-K | Implemented, boundary-tested, development-calibrated, and evaluated on 500 new questions | Thresholds remain resource-specific; see the final submission results above |")
    text = text.replace("The next core experiments should establish corpus/candidate answer coverage; compare fixed k=1/3/5/10, similarity-threshold and score-gap policies under identical generator/batching settings; calibrate parameters on separate development data; and evaluate the frozen rule on an unused evaluation set.",
                        "The final core experiment above now establishes corpus/candidate lexical answer coverage; compares fixed k=1/3/5/10, similarity-threshold and score-gap policies under identical generator/batching settings; calibrates parameters on separate development data; and evaluates frozen rules on 500 unused questions.")
    text = text.replace("The next project priority is Adaptive Top-K behavior and its cost/quality evidence; model-internal kernel fusion is outside the immediate work plan.",
                        "The final project priority was Adaptive Top-K behavior and its cost/quality evidence, now reported above; model-internal kernel fusion was not performed.")
    text = text.replace("| Fixed Top-K baselines with several k values | Small-workload k sweep and repeated 500-question fixed-5 reference | A controlled larger-workload k=1/3/5/10 comparison with identical execution settings |",
                        "| Fixed Top-K baselines with several k values | Completed fixed k=1/3/5/10 on 500 new questions, three repetitions each | Broader corpus/model/hardware generalization remains untested |")
    text = text.replace("| Preserve answer quality while reducing context overhead | Mean document/input-token reduction measured | Current adaptive EM is below fixed-5; quality preservation has not been demonstrated |",
                        "| Preserve answer quality while reducing context overhead | Final threshold rule reduces tokens with all three observed answer metrics at least fixed-5 | Lexical retrieval hit falls by one question; quality equivalence and latency improvement are not established |")
    text = text.replace("| Reduce retrieval overhead | Candidate search and policy timing measured | Current adaptive path always requests max_k=10 candidates before pruning, so reduced FAISS search cost is not established |",
                        "| Reduce retrieval overhead | Final guarded policies request five candidates before pruning; earlier policy requested ten | Early termination of the Flat-index scan and reduced distance-computation cost are not established |")
    text = text.replace("| Testing and final artifacts | Policy/integration tests, generator tests, experiment logs and one consolidated report | Final test inventory and PDF packaging; WPR native-stack capture still requires administrator execution |",
                        "| Testing and final artifacts | 56-test suite passed; final experiment verified; consolidated source and PDF supplied | WPR native-stack capture still requires administrator execution |")
    text = text.replace("### Project-plan status", "### Historical checkpoint status before the final core study")
    text = text.replace("Corpus answer coverage and the exact vector-to-document correspondence have not been systematically verified.",
                        "At that early checkpoint corpus coverage and vector/document correspondence were unverified; the final resource audit above adds development lexical coverage and a sixteen-passage alignment check.")
    assert len(re.findall(r"(?m)^## Optimization [1-5]:", text)) == 5
    assert not re.search(r"[\u4e00-\u9fff]", text)
    REPORT.write_text(text, encoding="utf-8")
    print("[REPORT] Updated the single English report; final verification passed.")


if __name__ == "__main__":
    main()
