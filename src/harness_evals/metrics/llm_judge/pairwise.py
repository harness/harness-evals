"""Pairwise metric — LLM compares output against expected (A/B comparison)."""

from __future__ import annotations

import asyncio
import json
from collections import Counter

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM


def _to_str(val: str | dict | list | None) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return json.dumps(val, ensure_ascii=False)

_PROMPT_TEMPLATE = """You are an expert evaluator. Compare the two responses below and judge which is better.

**Evaluation criteria**: {criteria}

**Input / Task**: {input}

**Response A (candidate)**: {response_a}

**Response B (reference)**: {response_b}

First, reason step-by-step about the quality of each response according to the criteria.
Then decide which response is better, or if they are tied.

Respond with JSON:
{{"reasoning": "your chain-of-thought reasoning", "winner": "A" or "B" or "tie", "score": <float between 0.0 and 1.0>}}

Where score reflects how good Response A is:
- 1.0 = A is clearly better
- 0.5 = tie / equivalent
- 0.0 = B is clearly better
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "winner", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "winner": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class PairwiseMetric(BaseMetric):
    """LLM-judged A/B comparison between output and expected.

    The LLM evaluates both responses against configurable criteria and
    returns a score reflecting how good the output (A) is relative to
    the expected (B).
    """

    def __init__(
        self,
        llm: BaseLLM,
        criteria: str = "Overall quality, accuracy, and helpfulness",
        threshold: float = 0.5,
        mitigate_position_bias: bool = True,
        num_votes: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(name="pairwise", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.llm = llm
        self.criteria = criteria
        self.mitigate_position_bias = mitigate_position_bias
        if num_votes < 1:
            raise ValueError(f"num_votes must be >= 1, got {num_votes}")
        self.num_votes = num_votes

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def _single_judge(self, input_text: str, response_a: str, response_b: str) -> dict:
        """Run a single judge call with the given ordering."""
        prompt = _PROMPT_TEMPLATE.format(
            criteria=self.criteria,
            input=input_text,
            response_a=response_a,
            response_b=response_b,
        )
        return await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)

    async def a_measure(self, eval_case: EvalCase) -> Score:
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="expected is required for pairwise comparison",
            )

        input_text = _to_str(eval_case.input)
        output = _to_str(eval_case.output)
        expected = _to_str(eval_case.expected)

        if self.mitigate_position_bias:
            scores_ab, scores_ba = await asyncio.gather(
                self._collect_votes(input_text, output, expected),
                self._collect_votes(input_text, expected, output),
            )
            # scores_ba has output as B, so score represents how good expected (A) is
            # Flip it: 1 - score_ba = how good output is when placed as B
            flipped_ba = [1.0 - s for s in scores_ba]

            avg_ab = sum(scores_ab) / len(scores_ab)
            avg_ba = sum(flipped_ba) / len(flipped_ba)
            value = (avg_ab + avg_ba) / 2.0
            position_bias_delta = abs(avg_ab - avg_ba)

            winners = self._compute_winners(scores_ab, flipped_ba)
            reasoning = f"Position-debiased result from {self.num_votes * 2} judgements"

            metadata: dict = {
                "winner": winners["final"],
                "position_bias_delta": round(position_bias_delta, 4),
            }
            if self.num_votes > 1:
                metadata["vote_counts"] = winners["counts"]

        else:
            scores = await self._collect_votes(input_text, output, expected)
            value = sum(scores) / len(scores)
            reasoning = self._get_reasoning_from_votes(scores)

            metadata = {"winner": self._winner_from_score(value)}
            if self.num_votes > 1:
                metadata["vote_counts"] = dict(Counter(
                    self._winner_from_score(s) for s in scores
                ))

        value = max(0.0, min(1.0, value))
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
            metadata=metadata,
        )

    async def _collect_votes(self, input_text: str, response_a: str, response_b: str) -> list[float]:
        """Collect num_votes judge scores for a given ordering."""
        if self.num_votes == 1:
            result = await self._single_judge(input_text, response_a, response_b)
            score = float(result.get("score", 0.0))
            return [max(0.0, min(1.0, score))]

        tasks = [self._single_judge(input_text, response_a, response_b) for _ in range(self.num_votes)]
        results = await asyncio.gather(*tasks)
        return [max(0.0, min(1.0, float(r.get("score", 0.0)))) for r in results]

    @staticmethod
    def _winner_from_score(score: float) -> str:
        if score > 0.55:
            return "A"
        elif score < 0.45:
            return "B"
        return "tie"

    @staticmethod
    def _compute_winners(scores_ab: list[float], flipped_ba: list[float]) -> dict:
        all_scores = scores_ab + flipped_ba
        vote_labels = [PairwiseMetric._winner_from_score(s) for s in all_scores]
        counts = dict(Counter(vote_labels))
        sorted_by_count = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_by_count) >= 2 and sorted_by_count[0][1] == sorted_by_count[1][1]:
            final = "tie"
        else:
            final = sorted_by_count[0][0]
        return {"final": final, "counts": counts}

    @staticmethod
    def _get_reasoning_from_votes(scores: list[float]) -> str:
        if len(scores) == 1:
            return f"Single judge score: {scores[0]:.2f}"
        return f"Majority vote over {len(scores)} judges, mean score: {sum(scores)/len(scores):.2f}"
