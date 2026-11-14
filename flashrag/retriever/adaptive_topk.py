"""Score-only selection of a variable document prefix for each query.

Scores must already be ranked. A gap is a heuristic signal, not a calibrated
probability of answer correctness. Tune thresholds on development data only.
"""
from dataclasses import dataclass
import math


def _validate_bounds(min_k, max_k, fallback_k):
    for name, value in (("min_k", min_k), ("max_k", max_k), ("fallback_k", fallback_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not min_k <= fallback_k <= max_k:
        raise ValueError("require min_k <= fallback_k <= max_k")


def _ranked_scores(scores, max_k, higher_is_better):
    values = [float(score) for score in scores[:max_k]]
    if not all(math.isfinite(score) for score in values):
        raise ValueError("retrieval scores must be finite")
    direction = 1 if higher_is_better else -1
    gaps = [direction * (left - right) for left, right in zip(values, values[1:])]
    if any(gap < 0 for gap in gaps):
        raise ValueError("scores are not ranked in the configured score direction")
    return values, gaps


@dataclass(frozen=True)
class ScoreGapPolicy:
    min_k: int = 1
    max_k: int = 10
    fallback_k: int = 5
    gap_threshold: float = 0.05
    higher_is_better: bool = True

    def __post_init__(self):
        _validate_bounds(self.min_k, self.max_k, self.fallback_k)
        if not math.isfinite(self.gap_threshold) or self.gap_threshold <= 0:
            raise ValueError("gap_threshold must be finite and positive")

    def select(self, scores):
        """Return a decision using scores only, never questions or gold answers.

        Cut after the largest adjacent gap above the threshold within the allowed
        prefix lengths. Ties choose the earliest boundary. If no clear gap exists,
        retain fallback_k, capped by available documents. Fewer than min_k results
        are kept as-is; empty results yield k=0.
        """
        values, gaps = _ranked_scores(scores, self.max_k, self.higher_is_better)

        count = len(values)
        selected_k = min(self.fallback_k, count)
        reason = "fallback_no_clear_gap"
        largest_gap = None
        if count <= self.min_k:
            selected_k = count
            reason = "insufficient_candidates" if count else "empty_retrieval"
        else:
            boundary = max(range(self.min_k, count), key=lambda k: gaps[k - 1])
            largest_gap = gaps[boundary - 1]
            if largest_gap >= self.gap_threshold:
                selected_k = boundary
                reason = "score_gap"

        return {
            "selected_k": selected_k,
            "candidate_count": count,
            "reason": reason,
            "largest_gap": largest_gap,
            "gap_threshold": self.gap_threshold,
        }


@dataclass(frozen=True)
class SimilarityThresholdPolicy:
    """Keep a ranked prefix meeting an absolute similarity/distance threshold.

    The threshold is calibrated on development data, not interpreted as a
    probability. When no score qualifies, use fallback_k rather than assuming a
    weakly retrieved query needs little evidence. min_k guards aggressive cuts.
    """

    score_threshold: float = 0.8
    min_k: int = 2
    max_k: int = 5
    fallback_k: int = 5
    higher_is_better: bool = True

    def __post_init__(self):
        _validate_bounds(self.min_k, self.max_k, self.fallback_k)
        if isinstance(self.score_threshold, bool) or not math.isfinite(self.score_threshold):
            raise ValueError("score_threshold must be finite")

    def select(self, scores):
        values, _ = _ranked_scores(scores, self.max_k, self.higher_is_better)
        qualifies = (sum(s >= self.score_threshold for s in values) if self.higher_is_better
                     else sum(s <= self.score_threshold for s in values))
        if len(values) <= self.min_k:
            k, reason = len(values), "insufficient_candidates" if values else "empty_retrieval"
        elif not qualifies:
            k, reason = min(self.fallback_k, len(values)), "fallback_low_similarity"
        else:
            k, reason = max(self.min_k, qualifies), "similarity_threshold"
        return {"selected_k": k, "candidate_count": len(values), "qualifying_count": qualifies,
                "reason": reason, "score_threshold": self.score_threshold}


def policy_from_config(config):
    """Construct a score-only policy from a frozen, JSON-serializable setting."""
    values = dict(config)
    kind = values.pop("type")
    classes = {"score_gap": ScoreGapPolicy, "similarity_threshold": SimilarityThresholdPolicy}
    if kind not in classes:
        raise ValueError(f"Unknown policy type: {kind}")
    return classes[kind](**values)
