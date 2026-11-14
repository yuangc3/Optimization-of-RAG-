"""Exercise the production HF generate method without loading model weights.

AST extraction avoids unrelated FlashRAG optional imports (FAISS, vLLM, etc.).
Real torch tensors test slicing, probabilities, logits and batch padding.
Run: python -m unittest discover -s tests -p test_generator_score_path.py -v
"""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time
from typing import List
import unittest
import json
from collections import Counter
from bisect import bisect_right
import tempfile
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_generate():
    tree = ast.parse((ROOT / "flashrag/generator/generator.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "HFCausalLMGenerator")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "generate")
    utils_tree = ast.parse((ROOT / "flashrag/generator/utils.py").read_text(encoding="utf-8"))
    resolver = next(node for node in utils_tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "resolve_max_tokens")
    namespace = {"List": List, "deepcopy": deepcopy, "time": time,
                 "trange": lambda *a, **kw: range(*a)}
    exec(compile(ast.Module(body=[resolver, method], type_ignores=[]), "generate-under-test", "exec"), namespace)
    return namespace["generate"]


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    pad_token_id = 0

    def __call__(self, prompts, **kwargs):
        return Inputs(input_ids=torch.tensor([[1, 2]] * len(prompts)))

    def decode(self, ids, **kwargs):
        return "".join(str(token) for token in ids.tolist() if token)

    def batch_decode(self, rows, **kwargs):
        return [self.decode(torch.tensor(row), **kwargs) for row in rows]


class Model:
    device = torch.device("cpu")

    def __init__(self):
        self.score_requests = []

    def generate(self, input_ids, output_scores, return_dict_in_generate, **kwargs):
        self.score_requests.append(output_scores)
        continuation = torch.tensor([[3, 4]] * len(input_ids))
        result = SimpleNamespace(sequences=torch.cat([input_ids, continuation], dim=1))
        # No scores attribute on the text-only path: accessing it is an error.
        if output_scores:
            result.scores = tuple(torch.arange(6, dtype=torch.float).repeat(len(input_ids), 1)
                                  for _ in range(2))
        return result


class ScorePathTest(unittest.TestCase):
    def setUp(self):
        self.generator = SimpleNamespace(
            batch_size=2, generation_params={"max_new_tokens": 3},
            model_name="test", max_input_len=32, tokenizer=Tokenizer(), model=Model())
        self.generate = load_generate()

    def test_text_only_skips_score_processing_and_matches_legacy(self):
        with patch.object(torch, "stack", side_effect=AssertionError("unneeded score allocation")):
            optimized = self.generate(self.generator, ["a", "b", "c"])
        self.assertEqual(self.generator.model.score_requests, [False, False])
        legacy = self.generate(self.generator, ["a", "b", "c"], collect_scores=True)
        self.assertEqual(optimized, ["34"] * 3)
        self.assertEqual(optimized, legacy)
        self.assertEqual(self.generator.model.score_requests[-2:], [True, True])

    def test_return_scores_preserves_probabilities(self):
        responses, scores = self.generate(self.generator, "a", return_scores=True)
        expected = torch.arange(6, dtype=torch.float).softmax(-1)[[3, 4]]
        self.assertEqual(responses, ["34"])
        torch.testing.assert_close(torch.tensor(scores), expected.unsqueeze(0))
        self.assertEqual(self.generator.model.score_requests, [True])

    def test_allocator_cache_control_preserves_text_and_skips_cache_clear(self):
        with patch.object(torch.cuda, "empty_cache") as clear:
            retained = self.generate(self.generator, ["a", "b", "c"], clear_cuda_cache=False)
            clear.assert_not_called()
            legacy = self.generate(self.generator, ["a", "b", "c"], clear_cuda_cache=True)
            self.assertEqual(clear.call_count, 2)
        self.assertEqual(retained, legacy)

    def test_generated_only_decode_preserves_scores_and_dictionary_outputs(self):
        original = self.generate(self.generator, ["a", "b", "c"], return_dict=True)
        with patch.object(self.generator.tokenizer, "decode", side_effect=AssertionError("prompt decode")):
            with patch.object(self.generator.tokenizer, "batch_decode", return_value=["34", "34"]) as decode:
                # Two prompts avoid the artificial fixed-length mock batch decoder
                # being mistaken for a real tokenizer on the final short batch.
                fast = self.generate(self.generator, ["a", "b"], return_dict=True, decode_generated_only=True)
                self.assertEqual(decode.call_count, 1)
        self.assertEqual(fast["responses"], original["responses"][:2])
        self.assertEqual(fast["scores"], original["scores"][:2])
        for key in ("generated_token_ids", "generated_token_logits"):
            torch.testing.assert_close(fast[key], original[key][:2])

    def test_reused_real_tokenizer_preserves_truncation_padding_and_return_order(self):
        from tokenizers import Tokenizer as BackendTokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        backend = BackendTokenizer(WordLevel({"[PAD]": 0, "a": 1, "b": 2, "c": 3,
                                             "d": 4, "[UNK]": 5}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        self.generator.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", padding_side="left")
        self.generator.tokenizer.model_input_names = ["input_ids", "attention_mask"]
        self.generator.max_input_len = 4
        prompts = ["a b c d a b", "a", "c d a", "a b"]
        with patch.object(self.generator.model, "generate", wraps=self.generator.model.generate) as model:
            original = self.generate(self.generator, prompts, sort_by_length=True, return_dict=True)
            baseline_inputs = [call.kwargs["input_ids"].clone() for call in model.call_args_list]
        profile = dict(self.generator.last_generation_profile)
        with patch.object(self.generator.model, "generate", wraps=self.generator.model.generate) as model:
            reused = self.generate(self.generator, prompts, sort_by_length=True, return_dict=True,
                                   reuse_tokenization=True, decode_generated_only=True)
            for reference, call in zip(baseline_inputs, model.call_args_list):
                torch.testing.assert_close(reference, call.kwargs["input_ids"])
        self.assertEqual(original["responses"], reused["responses"])
        self.assertEqual(original["scores"], reused["scores"])
        for key in ("generated_token_ids", "generated_token_logits"):
            torch.testing.assert_close(original[key], reused[key])
        for key in ("processing_order", "logical_input_tokens", "padded_input_token_slots", "batches"):
            self.assertEqual(profile[key], self.generator.last_generation_profile[key])
        self.assertEqual(profile["logical_input_tokens"], 10)
        self.assertEqual(self.generate(self.generator, [], reuse_tokenization=True, decode_generated_only=True), [])

    def test_return_dict_preserves_logits_ids_scores_and_padding_across_batches(self):
        result = self.generate(self.generator, ["a", "b", "c"], return_dict=True)
        self.assertEqual(result["responses"], ["34"] * 3)
        torch.testing.assert_close(result["generated_token_ids"], torch.tensor([[3, 4, 0]] * 3))
        self.assertEqual(tuple(result["generated_token_logits"].shape), (3, 3, 6))
        torch.testing.assert_close(result["generated_token_logits"][:, :2],
                                   torch.arange(6, dtype=torch.float).repeat(3, 2, 1))
        self.assertEqual(result["generated_token_logits"][:, 2].count_nonzero().item(), 0)
        self.assertEqual(len(result["scores"]), 3)
        self.assertEqual(self.generator.model.score_requests, [True, True])

    def test_real_huggingface_generation_matches_score_path_without_downloads(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(2024)
        self.generator.model = GPT2LMHeadModel(GPT2Config(
            vocab_size=6, n_positions=16, n_embd=8, n_layer=1, n_head=1,
            bos_token_id=1, eos_token_id=None, pad_token_id=0)).eval()
        self.generator.generation_params.update(do_sample=False)
        fast = self.generate(self.generator, ["a", "b", "c"])
        legacy = self.generate(self.generator, ["a", "b", "c"], collect_scores=True)
        scored_text, scores = self.generate(self.generator, ["a", "b", "c"], return_scores=True)
        self.assertEqual(fast, legacy)
        self.assertEqual(fast, scored_text)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(len(row) == 3 for row in scores))

    def test_length_sort_restores_text_scores_ids_logits_and_reduces_padding(self):
        class VariableTokenizer(Tokenizer):
            def encode(self, prompt, **kwargs):
                return [int(char) for char in prompt][:kwargs.get("max_length", 32)]

            def __call__(self, prompts, **kwargs):
                rows = [self.encode(prompt, **kwargs) for prompt in prompts]
                width = max(map(len, rows))
                ids = torch.tensor([[0] * (width - len(row)) + row for row in rows])
                return Inputs(input_ids=ids, attention_mask=(ids != 0).long())

        class VariableModel(Model):
            def generate(self, input_ids, output_scores, return_dict_in_generate, **kwargs):
                tokens = input_ids[:, -1:]
                result = SimpleNamespace(sequences=torch.cat([input_ids, tokens.repeat(1, 2)], dim=1))
                if output_scores:
                    logits = torch.zeros(len(tokens), 6)
                    for row, token in enumerate(tokens[:, 0]):
                        logits[row, token] = float(token)
                    result.scores = (logits, logits)
                return result

        self.generator.tokenizer = VariableTokenizer()
        self.generator.model = VariableModel()
        prompts = ["55555", "1", "333", "22"]
        original = self.generate(self.generator, prompts, return_dict=True)
        original_stats = dict(self.generator.last_generation_profile)
        sorted_result = self.generate(self.generator, prompts, sort_by_length=True, return_dict=True)
        self.assertEqual(sorted_result["responses"], ["55", "11", "33", "22"])
        self.assertEqual(sorted_result["responses"], original["responses"])
        self.assertEqual(sorted_result["scores"], original["scores"])
        for key in ("generated_token_ids", "generated_token_logits"):
            torch.testing.assert_close(sorted_result[key], original[key])
        stats = self.generator.last_generation_profile
        self.assertEqual(stats["processing_order"], [1, 3, 2, 0])
        self.assertEqual(stats["logical_input_tokens"], 11)
        self.assertEqual(stats["padded_input_token_slots"], 14)
        self.assertEqual(original_stats["padded_input_token_slots"], 16)
        texts, scores = self.generate(self.generator, prompts, sort_by_length=True, return_scores=True)
        self.assertEqual(texts, original["responses"])
        self.assertEqual(scores, original["scores"])
        self.assertEqual(self.generate(self.generator, [], sort_by_length=True), [])


class RuntimeTraceAnalysisTest(unittest.TestCase):
    def test_kernel_union_excludes_nested_annotations_and_overlapping_time(self):
        source = ast.parse((ROOT / "scripts/profile_generation.py").read_text(encoding="utf-8"))
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("interval_union_us", "analyze_trace")]
        namespace = {"json": json, "Counter": Counter, "bisect_right": bisect_right}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "trace-analysis", "exec"), namespace)
        def event(name, category, start, duration, **kwargs):
            return {"ph": "X", "name": name, "cat": category, "ts": start, "dur": duration, **kwargs}
        events = [event("rag.profile_window", "user_annotation", 100, 1000),
                  event("k1", "kernel", 100, 200, args={"correlation": 1}),
                  event("k2", "kernel", 200, 200, args={"correlation": 2}),
                  event("k3", "kernel", 600, 100, args={"correlation": 3}),
                  event("rag.prefill", "user_annotation", 80, 70),
                  event("rag.decode", "user_annotation", 180, 30),
                  event("cudaLaunchKernel", "cuda_runtime", 100, 2, args={"correlation": 1}),
                  event("cudaLaunchKernel", "cuda_runtime", 200, 2, args={"correlation": 2}),
                  event("cudaLaunchKernel", "cuda_runtime", 600, 2, args={"correlation": 3}),
                  event("outside", "kernel", -100, 50),
                  event("rag.decode", "gpu_user_annotation", 100, 900),
                  event("Memcpy HtoD", "gpu_memcpy", 350, 100, args={"bytes": 32}),
                  event("cudaStreamSynchronize", "cuda_runtime", 650, 100)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
            result = namespace["analyze_trace"](path)
        self.assertEqual(result["kernel_count"], 3)
        self.assertAlmostEqual(result["kernel_sum_ms"], 0.5)
        self.assertAlmostEqual(result["kernel_union_ms"], 0.4)
        self.assertAlmostEqual(result["process_kernel_coverage_percent"], 40)
        self.assertAlmostEqual(result["kernel_copy_memset_union_ms"], 0.45)
        self.assertEqual(result["sync_api_count"], 1)
        self.assertEqual(result["copy_bytes"], 32)
        self.assertEqual(result["copy_direction_details"]["Memcpy HtoD"]["bytes"], 32)
        self.assertEqual(result["kernels_by_forward_phase"]["rag.prefill"]["kernel_count"], 1)
        self.assertEqual(result["kernels_by_forward_phase"]["rag.decode"]["kernel_count"], 1)
        self.assertEqual(result["launch_api_count"], 3)


if __name__ == "__main__":
    unittest.main()
