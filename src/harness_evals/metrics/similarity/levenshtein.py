"""Levenshtein metric — normalized edit distance similarity."""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Wagner-Fischer algorithm. Pure Python, no external deps."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class LevenshteinMetric(BaseMetric):
    """Normalized edit distance similarity between output and expected.

    Score = 1.0 - (edit_distance / max(len(output), len(expected))).
    Returns 1.0 when both strings are identical (including both empty).
    """

    def __init__(self, threshold: float = 0.8, **kwargs: object) -> None:
        super().__init__(name="levenshtein", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No expected answer provided to compare against (expected is None)",
            )

        output_str = str(eval_case.output)
        expected_str = str(eval_case.expected)

        max_len = max(len(output_str), len(expected_str))
        if max_len == 0:
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason="Output and expected answer are both empty strings",
            )

        dist = _levenshtein_distance(output_str, expected_str)
        value = 1.0 - (dist / max_len)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"Output differs by {dist} characters out of {max_len} from the expected answer (edit distance {dist}/{max_len})",
            metadata={"edit_distance": dist, "max_length": max_len},
        )
