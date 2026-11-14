"""Ensure warmups and measured repeats use independent FlashRAG records."""
import ast
import copy
from pathlib import Path
import unittest

from flashrag.dataset import Dataset


class DatasetCloneTest(unittest.TestCase):
    def test_clone_preserves_records_without_mutating_original_or_other_repeat(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "examples/quick_start/perf_baseline.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "clone_dataset")
        namespace = {"copy": copy}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "clone-under-test", "exec"), namespace)
        original = Dataset(config={"dataset_name": "test"}, data=[{
            "id": "1", "question": "q", "golden_answers": ["a"], "metadata": {"source": [1]}}])
        original.update_output("existing", [{"values": [2]}])
        first = namespace["clone_dataset"](original)
        second = namespace["clone_dataset"](original)
        self.assertEqual(first.question, original.question)
        self.assertEqual(first[0].existing, {"values": [2]})
        first[0].golden_answers.append("changed")
        first[0].metadata["source"].append(3)
        first[0].existing["values"].append(4)
        first.update_output("pred", ["answer"])
        for pristine in (original, second):
            self.assertEqual(pristine.golden_answers, [["a"]])
            self.assertEqual(pristine[0].metadata, {"source": [1]})
            self.assertEqual(pristine[0].existing, {"values": [2]})
            self.assertNotIn("pred", pristine[0].output)
        larger = Dataset(config={"dataset_name": "test"}, data=[
            {"id": str(i), "question": str(i), "golden_answers": [str(i)]} for i in range(5)])
        warmup = namespace["clone_dataset"](larger, sample_num=2)
        self.assertEqual(warmup.id, ["0", "1"])
        self.assertEqual(len(namespace["clone_dataset"](larger)), 5)


if __name__ == "__main__":
    unittest.main()
