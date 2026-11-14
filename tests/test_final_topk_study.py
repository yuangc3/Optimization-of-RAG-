"""Data separation and calibration safeguards, without model downloads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_topk_study", ROOT / "scripts/run_final_topk_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class FinalStudyProtocolTest(unittest.TestCase):
    def test_calibration_retains_every_development_fixed5_lexical_hit(self):
        records = [{"golden_answers": ["last evidence"]}, {"golden_answers": ["first evidence"]}]
        documents = [[{"contents": f"irrelevant {i}"} for i in range(5)] for _ in records]
        documents[0][4]["contents"] = "last evidence"
        documents[1][0]["contents"] = "first evidence"
        scores = [[.95, .94, .93, .92, .91], [.95, .94, .5, .49, .48]]
        result = study.calibrate(documents, scores, records, str.casefold)
        self.assertEqual(result["fixed5_lexical_hits"], 2)
        self.assertEqual(set(result["chosen"]), {"score_gap", "similarity_threshold"})
        for chosen in result["chosen"].values():
            self.assertEqual(chosen["lost_fixed5_hits"], 0)
            self.assertLess(chosen["mean_k"], 5)
            self.assertEqual(chosen["config"]["max_k"], 5)

    def test_lexical_proxy_uses_token_boundaries_and_rejects_empty_aliases(self):
        docs = [{"contents": "the theater"}]
        self.assertFalse(study.lexical_hit(docs, ["he", ""], str.casefold))
        self.assertTrue(study.lexical_hit(docs, ["theater"], str.casefold))

    def test_splits_exclude_prior_questions_and_are_disjoint_and_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "datasets/flashrag_official/nq/test.jsonl"
            official.parent.mkdir(parents=True)
            source = [{"id": str(i), "question": f"Question {i}", "golden_answers": [str(i)]} for i in range(10)]
            official.write_text("".join(json.dumps(r) + "\n" for r in source), encoding="utf-8")
            for path, rows in ((root / "examples/quick_start/dataset/nq/test.jsonl", source[:1]),
                               (root / "datasets/nq_500_heldout/nq/test.jsonl", source[1:2])):
                path.parent.mkdir(parents=True)
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            (root / "docs").mkdir()
            study.save(root / "docs/nq_500_manifest.json", {"source_sha256": study.digest(official), "source_repo": "test", "source_revision": "pinned"})
            with patch.object(study, "ROOT", root):
                dev, test, manifest = study.prepare_splits(root / "docs/result", 2, 3)
                self.assertFalse({r["id"] for r in dev} & {r["id"] for r in test})
                self.assertFalse({r["id"] for r in dev + test} & {"0", "1"})
                self.assertEqual(study.prepare_splits(root / "docs/result", 2, 3)[2], manifest)
                with self.assertRaises(ValueError):
                    study.prepare_splits(root / "docs/result", 2, 3, seed=1)


if __name__ == "__main__":
    unittest.main()
