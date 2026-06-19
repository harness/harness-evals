"""Tests for evaluate_dataset_pair() runner helper."""

import pytest

from harness_evals import Golden, evaluate_dataset_pair
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.llm_judge.pairwise import PairwiseMetric


class MockLLM(BaseLLM):
    def __init__(self, json_response: dict):
        self._json_response = json_response

    async def generate(self, prompt: str, **kwargs) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return self._json_response


@pytest.mark.unit
class TestEvaluateDatasetPair:
    async def test_basic_pair_evaluation(self):
        llm = MockLLM({"reasoning": "A is better", "winner": "A", "score": 0.8})
        metric = PairwiseMetric(llm=llm, mitigate_position_bias=False)

        goldens = [
            Golden(input="What is 2+2?"),
            Golden(input="What is the capital of France?"),
        ]

        async def model_a(g: Golden) -> str:
            return f"model_a: {g.input}"

        async def model_b(g: Golden) -> str:
            return f"model_b: {g.input}"

        summary = await evaluate_dataset_pair(
            goldens=goldens,
            candidate_a_fn=model_a,
            candidate_b_fn=model_b,
            metric=metric,
        )

        assert summary.total_cases == 2
        assert "pairwise" in summary.by_metric
        assert summary.by_metric["pairwise"].mean == 0.8
        assert summary.by_metric["pairwise"].pass_rate == 1.0

    async def test_pair_with_concurrency(self):
        llm = MockLLM({"reasoning": "tie", "winner": "tie", "score": 0.5})
        metric = PairwiseMetric(llm=llm, mitigate_position_bias=False)

        goldens = [Golden(input=f"q{i}") for i in range(5)]

        async def model_a(g: Golden) -> str:
            return "a"

        async def model_b(g: Golden) -> str:
            return "b"

        summary = await evaluate_dataset_pair(
            goldens=goldens,
            candidate_a_fn=model_a,
            candidate_b_fn=model_b,
            metric=metric,
            concurrency=2,
        )

        assert summary.total_cases == 5
        assert summary.by_metric["pairwise"].mean == 0.5

    async def test_invalid_concurrency(self):
        llm = MockLLM({"reasoning": "ok", "winner": "A", "score": 0.5})
        metric = PairwiseMetric(llm=llm)

        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            await evaluate_dataset_pair(
                goldens=[Golden(input="q")],
                candidate_a_fn=lambda g: "a",  # type: ignore
                candidate_b_fn=lambda g: "b",  # type: ignore
                metric=metric,
                concurrency=0,
            )

    async def test_win_rates(self):
        """B wins all comparisons — pass rate should be 0."""
        llm = MockLLM({"reasoning": "B wins", "winner": "B", "score": 0.2})
        metric = PairwiseMetric(llm=llm, threshold=0.5, mitigate_position_bias=False)

        goldens = [Golden(input=f"q{i}") for i in range(3)]

        async def model_a(g: Golden) -> str:
            return "weak"

        async def model_b(g: Golden) -> str:
            return "strong"

        summary = await evaluate_dataset_pair(
            goldens=goldens,
            candidate_a_fn=model_a,
            candidate_b_fn=model_b,
            metric=metric,
        )

        assert summary.by_metric["pairwise"].pass_rate == 0.0
        assert abs(summary.by_metric["pairwise"].mean - 0.2) < 1e-9
