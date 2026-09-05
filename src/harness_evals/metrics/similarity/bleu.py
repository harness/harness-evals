"""BLEU metric — n-gram precision score for text generation quality."""

from __future__ import annotations

import math
from collections import Counter

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _bleu_pure_python(reference_tokens: list[str], hypothesis_tokens: list[str], max_n: int = 4) -> float:
    """Compute sentence BLEU with method-1 smoothing and uniform weights."""
    if not hypothesis_tokens or not reference_tokens:
        return 0.0

    precisions: list[float] = []

    for n in range(1, max_n + 1):
        ref_ngrams = _get_ngrams(reference_tokens, n)
        hyp_ngrams = _get_ngrams(hypothesis_tokens, n)

        clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
        # NLTK's sentence_bleu returns zero before smoothing when there is no
        # unigram overlap at all.
        if n == 1 and clipped == 0:
            return 0.0
        total = max(1, sum(hyp_ngrams.values()))
        # Match NLTK's method-1 smoothing: add epsilon only when an n-gram
        # precision numerator is zero.
        precisions.append(clipped / total if clipped else 0.1 / total)

    log_avg = sum(math.log(p) for p in precisions) / len(precisions)

    bp = 1.0
    if len(hypothesis_tokens) < len(reference_tokens):
        bp = math.exp(1 - len(reference_tokens) / len(hypothesis_tokens))

    return bp * math.exp(log_avg)


class BLEUMetric(BaseMetric):
    """BLEU score between output and expected text.

    Uses a dependency-free sentence BLEU implementation with method-1
    smoothing. Score is in [0.0, 1.0], where 1.0 is perfect n-gram overlap.
    """

    def __init__(self, threshold: float = 0.5, max_n: int = 4, **kwargs: object) -> None:
        super().__init__(name="bleu", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.max_n = max_n

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No expected answer provided to compute BLEU score against (expected is None)",
            )

        ref_tokens = str(eval_case.expected).split()
        hyp_tokens = str(eval_case.output).split()

        if not ref_tokens:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Expected answer is empty, so BLEU score cannot be computed",
            )

        value = _bleu_pure_python(ref_tokens, hyp_tokens, self.max_n)

        return Score(
            name=self.name,
            value=float(value),
            threshold=self.threshold,
            reason=f"Output has {value * 100:.0f}% n-gram overlap with expected answer (BLEU-{self.max_n} = {value:.4f})",
            metadata={"max_n": self.max_n},
        )
