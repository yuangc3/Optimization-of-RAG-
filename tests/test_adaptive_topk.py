import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from flashrag.dataset import Dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.retriever.adaptive_topk import ScoreGapPolicy, SimilarityThresholdPolicy, policy_from_config


class ScoreGapPolicyTest(unittest.TestCase):
    def test_largest_significant_gap_selects_document_prefix(self):
        decision = ScoreGapPolicy().select([0.95, 0.94, 0.70, 0.69, 0.68, 0.67])
        self.assertEqual(decision["selected_k"], 2)
        self.assertEqual(decision["reason"], "score_gap")
        self.assertAlmostEqual(decision["largest_gap"], 0.24)

    def test_flat_or_tied_scores_use_fallback(self):
        for scores in ([0.8] * 10, [0.8 - i * 0.001 for i in range(10)]):
            with self.subTest(scores=scores):
                self.assertEqual(ScoreGapPolicy().select(scores)["selected_k"], 5)

    def test_empty_short_and_singleton_results_are_bounded(self):
        for scores, expected in (([], 0), ([0.9], 1), ([0.9, 0.89], 2)):
            self.assertEqual(ScoreGapPolicy().select(scores)["selected_k"], expected)

    def test_minimum_maximum_and_tie_breaking(self):
        policy = ScoreGapPolicy(min_k=2, max_k=4, fallback_k=3, gap_threshold=0.1)
        decision = policy.select([4, 2, 1, 0, -100])
        self.assertEqual(decision["candidate_count"], 4)
        self.assertEqual(decision["selected_k"], 2)

    def test_significant_gap_can_select_more_than_five(self):
        scores = [0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.60, 0.59, 0.58]
        self.assertEqual(ScoreGapPolicy().select(scores)["selected_k"], 7)

    def test_lower_is_better_distances(self):
        result = ScoreGapPolicy(higher_is_better=False).select([0.1, 0.12, 0.5, 0.51])
        self.assertEqual(result["selected_k"], 2)

    def test_invalid_configuration_scores_and_score_order_are_rejected(self):
        for kwargs in ({"min_k": 0}, {"max_k": 2}, {"fallback_k": 0},
                       {"gap_threshold": 0}, {"gap_threshold": math.nan},
                       {"gap_threshold": math.inf}, {"min_k": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ScoreGapPolicy(**kwargs)
        for scores in ([0.5, 0.6], [math.nan], [math.inf]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                ScoreGapPolicy().select(scores)


class SimilarityThresholdPolicyTest(unittest.TestCase):
    def test_threshold_boundary_and_minimum_are_inclusive(self):
        policy = SimilarityThresholdPolicy(score_threshold=0.8)
        self.assertEqual(policy.select([.95, .9, .8, .79, .7])["selected_k"], 3)
        self.assertEqual(policy.select([.9, .7, .6, .5, .4])["selected_k"], 2)

    def test_weak_query_falls_back_and_scores_are_not_probabilities(self):
        p = SimilarityThresholdPolicy(score_threshold=10)
        self.assertEqual(p.select([9, 8, 7, 6, 5])["selected_k"], 5)
        self.assertEqual(p.select([9, 8, 7, 6, 5])["reason"], "fallback_low_similarity")

    def test_lower_distance_direction_and_short_results(self):
        p = SimilarityThresholdPolicy(score_threshold=.3, higher_is_better=False)
        self.assertEqual(p.select([.1, .2, .3, .4, .5])["selected_k"], 3)
        for values in ([], [.1], [.1, .2]):
            self.assertEqual(p.select(values)["selected_k"], len(values))

    def test_invalid_values_and_unsorted_scores_fail(self):
        for kwargs in ({"score_threshold": math.nan}, {"score_threshold": math.inf},
                       {"score_threshold": True}, {"min_k": 0}, {"max_k": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SimilarityThresholdPolicy(**kwargs)
        for scores in ([.7, .8], [math.nan], [math.inf]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                SimilarityThresholdPolicy().select(scores)

    def test_policy_configuration_is_round_trippable_and_unknown_type_fails(self):
        from dataclasses import asdict
        for kind, policy in (("score_gap", ScoreGapPolicy()), ("similarity_threshold", SimilarityThresholdPolicy())):
            self.assertEqual(policy_from_config(dict(type=kind, **asdict(policy))), policy)
        with self.assertRaises(ValueError):
            policy_from_config({"type": "unknown"})


class AdaptivePipelineTest(unittest.TestCase):
    def make_pipeline(self, policy, results):
        pipeline = SequentialPipeline.__new__(SequentialPipeline)
        pipeline.device = "cpu"
        pipeline.rewriter = None
        pipeline.retrieval_policy = policy
        pipeline.refiner = None
        pipeline.use_fid = False
        pipeline.save_retrieval_cache = False
        pipeline.retriever = SimpleNamespace(batch_search=Mock(return_value=results))
        pipeline.prompt_template = SimpleNamespace(get_string=lambda question, retrieval_result:
            question + ":" + ",".join(str(doc["id"]) for doc in retrieval_result))
        pipeline.generator = SimpleNamespace(model=SimpleNamespace(device="cpu"),
                                              generate=Mock(side_effect=lambda prompts: ["answer"] * len(prompts)))
        return pipeline

    def dataset(self):
        return Dataset(config={"dataset_name": "test"}, data=[
            {"id": "a", "question": "A", "golden_answers": ["gold-a"]},
            {"id": "b", "question": "B", "golden_answers": ["gold-b"]},
        ])

    def test_per_query_k_changes_prompts_and_records_decisions(self):
        docs = [[{"id": i, "contents": str(i)} for i in range(10)] for _ in range(2)]
        scores = [[0.95, 0.94] + [0.7] * 8, [0.8] * 10]
        pipeline = self.make_pipeline(ScoreGapPolicy(), (docs, scores))
        dataset = pipeline.run(self.dataset(), do_eval=False)
        pipeline.retriever.batch_search.assert_called_once_with(["A", "B"], num=10, return_score=True)
        self.assertEqual(dataset.selected_k, [2, 5])
        self.assertEqual(dataset.prompt, ["A:0,1", "B:0,1,2,3,4"])
        self.assertEqual(dataset.retrieval_scores, scores)
        self.assertEqual(dataset[0].adaptive_decision["reason"], "score_gap")
        self.assertIn("selection_s", pipeline.last_profile)
        pipeline.generator.generate.assert_called_once_with(dataset.prompt)

    def test_fixed_path_keeps_retriever_api_and_documents(self):
        docs = [[{"id": 0, "contents": "first"}], []]
        pipeline = self.make_pipeline(None, docs)
        dataset = pipeline.run(self.dataset(), do_eval=False)
        pipeline.retriever.batch_search.assert_called_once_with(["A", "B"])
        self.assertEqual(dataset.retrieval_result, docs)
        self.assertEqual(pipeline.last_profile["selection_s"], 0)
        self.assertNotIn("adaptive_decision", dataset[0].output)

    def test_malformed_retriever_outputs_fail_before_generation(self):
        for results in (([[], []], [[]]), ([[{"id": 1}], []], [[], []])):
            pipeline = self.make_pipeline(ScoreGapPolicy(), results)
            with self.assertRaises(ValueError):
                pipeline.run(self.dataset(), do_eval=False)
            pipeline.generator.generate.assert_not_called()

    def test_empty_retrieval_is_passed_as_an_empty_context(self):
        pipeline = self.make_pipeline(ScoreGapPolicy(), ([[], []], [[], []]))
        dataset = pipeline.run(self.dataset(), do_eval=False)
        self.assertEqual(dataset.selected_k, [0, 0])
        self.assertEqual(dataset.prompt, ["A:", "B:"])

    def test_threshold_policy_uses_existing_pipeline_and_records_scores(self):
        docs = [[{"id": i, "contents": str(i)} for i in range(5)] for _ in range(2)]
        policy = SimilarityThresholdPolicy(score_threshold=.8)
        pipeline = self.make_pipeline(policy, (docs, [[.95, .9, .85, .7, .6], [.7, .6, .5, .4, .3]]))
        data = pipeline.run(self.dataset(), do_eval=False)
        self.assertEqual(data.selected_k, [3, 5])
        pipeline.retriever.batch_search.assert_called_once_with(["A", "B"], num=5, return_score=True)


if __name__ == "__main__":
    unittest.main()
