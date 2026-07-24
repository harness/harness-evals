"""ROUGE metric — recall-oriented n-gram overlap for summarization evaluation."""

from __future__ import annotations

from collections import Counter

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score

_VALID_VARIANTS = {"rouge-1", "rouge-2", "rouge-l"}


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n(ref_tokens: list[str], hyp_tokens: list[str], n: int) -> dict[str, float]:
    """Compute ROUGE-N precision, recall, and f-measure."""
    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    ref_ngrams = _get_ngrams(ref_tokens, n)
    hyp_ngrams = _get_ngrams(hyp_tokens, n)

    if not ref_ngrams or not hyp_ngrams:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    overlap = sum((ref_ngrams & hyp_ngrams).values())
    precision = overlap / sum(hyp_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())

    if precision + recall == 0:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    fmeasure = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of the longest common subsequence via dynamic programming."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _rouge_l(ref_tokens: list[str], hyp_tokens: list[str]) -> dict[str, float]:
    """Compute ROUGE-L precision, recall, and f-measure using LCS."""
    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    fmeasure = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


class ROUGEMetric(BaseMetric):
    """ROUGE score between output and expected text.

    Pure Python implementation using whitespace tokenization.
    Score is in [0.0, 1.0] where 1.0 indicates perfect overlap.
    """

    def __init__(self, threshold: float = 0.5, variant: str = "rouge-l", **kwargs: object) -> None:
        if variant not in _VALID_VARIANTS:
            raise ValueError(f"Invalid variant '{variant}'. Must be one of: {sorted(_VALID_VARIANTS)}")
        super().__init__(name="rouge", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.variant = variant

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No expected answer provided to compute ROUGE score against (expected is None)",
            )

        ref_tokens = str(eval_case.expected).split()
        hyp_tokens = str(eval_case.output).split()

        if not ref_tokens:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Expected answer is empty, so ROUGE score cannot be computed",
            )

        if self.variant == "rouge-l":
            scores = _rouge_l(ref_tokens, hyp_tokens)
        else:
            n = int(self.variant[-1])
            scores = _rouge_n(ref_tokens, hyp_tokens, n)

        value = scores["fmeasure"]

        return Score(
            name=self.name,
            value=float(value),
            threshold=self.threshold,
            reason=f"Output has {value * 100:.0f}% overlap with expected answer ({self.variant} F1 = {value:.4f})",
            metadata={
                "variant": self.variant,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "fmeasure": scores["fmeasure"],
            },
        )
