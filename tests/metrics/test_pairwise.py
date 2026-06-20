"""Tests for PairwiseMetric with mocked LLM."""

import pytest

from harness_evals import EvalCase
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.llm_judge.pairwise import PairwiseMetric


class MockLLM(BaseLLM):
    def __init__(self, json_response: dict | list[dict]):
        self._responses = json_response if isinstance(json_response, list) else [json_response]
        self._call_count = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        resp = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return resp

    @property
    def call_count(self) -> int:
        return self._call_count


@pytest.mark.unit
class TestPairwiseMetric:
    async def test_a_wins(self):
        llm = MockLLM({"reasoning": "A is better", "winner": "A", "score": 0.9})
        metric = PairwiseMetric(llm=llm, threshold=0.5, mitigate_position_bias=False)
        ec = EvalCase(input="q", output="good answer", expected="reference")
        score = await metric.a_measure(ec)
        assert score.value == 0.9
        assert score.passed
        assert score.metadata["winner"] == "A"

    async def test_b_wins(self):
        llm = MockLLM({"reasoning": "B is better", "winner": "B", "score": 0.1})
        metric = PairwiseMetric(llm=llm, threshold=0.5, mitigate_position_bias=False)
        ec = EvalCase(input="q", output="bad", expected="good")
        score = await metric.a_measure(ec)
        assert score.value == 0.1
        assert not score.passed

    async def test_tie(self):
        llm = MockLLM({"reasoning": "Both are equal", "winner": "tie", "score": 0.5})
        metric = PairwiseMetric(llm=llm, threshold=0.5, mitigate_position_bias=False)
        ec = EvalCase(input="q", output="answer", expected="answer")
        score = await metric.a_measure(ec)
        assert score.value == 0.5
        assert score.passed

    async def test_expected_none(self):
        llm = MockLLM({"reasoning": "n/a", "winner": "n/a", "score": 0.0})
        metric = PairwiseMetric(llm=llm)
        ec = EvalCase(input="q", output="answer")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "expected is required" in score.reason

    async def test_clamps_score(self):
        llm = MockLLM({"reasoning": "test", "winner": "A", "score": 1.5})
        metric = PairwiseMetric(llm=llm, mitigate_position_bias=False)
        ec = EvalCase(input="q", output="a", expected="b")
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    def test_sync_measure(self):
        llm = MockLLM({"reasoning": "ok", "winner": "A", "score": 0.7})
        metric = PairwiseMetric(llm=llm, threshold=0.5, mitigate_position_bias=False)
        ec = EvalCase(input="q", output="a", expected="b")
        score = metric.measure(ec)
        assert score.value == 0.7
        assert score.passed


@pytest.mark.unit
class TestPositionBiasMitigation:
    async def test_bias_mitigation_averages_both_orderings(self):
        """When bias mitigation is on, both orderings are evaluated."""
        llm = MockLLM({"reasoning": "test", "winner": "A", "score": 0.8})
        metric = PairwiseMetric(llm=llm, mitigate_position_bias=True)
        ec = EvalCase(input="q", output="a", expected="b")
        score = await metric.a_measure(ec)
        # With constant 0.8 score: AB=0.8, BA flipped=1-0.8=0.2, avg=0.5
        assert abs(score.value - 0.5) < 0.01
        assert "position_bias_delta" in score.metadata
        assert llm.call_count == 2

    async def test_no_bias_when_scores_agree(self):
        """When both orderings give 0.5, there's no position bias."""
        llm = MockLLM({"reasoning": "tie", "winner": "tie", "score": 0.5})
        metric = PairwiseMetric(llm=llm, mitigate_position_bias=True)
        ec = EvalCase(input="q", output="a", expected="b")
        score = await metric.a_measure(ec)
        assert abs(score.value - 0.5) < 0.01
        assert score.metadata["position_bias_delta"] == 0.0

    async def test_bias_delta_detects_divergence(self):
        """position_bias_delta captures how much orderings disagree."""
        responses = [
            {"reasoning": "A first", "winner": "A", "score": 0.9},
            {"reasoning": "A first again", "winner": "A", "score": 0.9},
        ]
        llm = MockLLM(responses)
        metric = PairwiseMetric(llm=llm, mitigate_position_bias=True)
        ec = EvalCase(input="q", output="a", expected="b")
        score = await metric.a_measure(ec)
        # AB=0.9, BA flipped=1-0.9=0.1, delta=|0.9-0.1|=0.8
        assert score.metadata["position_bias_delta"] == 0.8

    async def test_bias_mitigation_disabled(self):
        """When disabled, only one ordering is evaluated."""
        llm = MockLLM({"reasoning": "test", "winner": "A", "score": 0.8})
        metric = PairwiseMetric(llm=llm, mitigate_position_bias=False)
        ec = EvalCase(input="q", output="a", expected="b")
        score = await metric.a_measure(ec)
        assert score.value == 0.8
        assert "position_bias_delta" not in score.metadata
        assert llm.call_count == 1


@pytest.mark.unit
class TestNumVotes:
    async def test_single_vote_no_vote_counts(self):
        llm = MockLLM({"reasoning": "ok", "winner": "A", "score": 0.7})
        metric = PairwiseMetric(llm=llm, num_votes=1, mitigate_position_bias=False)
        ec = EvalCase(input="q", output="a", expected="b")
        score = await metric.a_measure(ec)
        assert "vote_counts" not in score.metadata

    async def test_multiple_votes_majority(self):
        responses = [
            {"reasoning": "A wins", "winner": "A", "score": 0.8},
            {"reasoning": "A wins", "winner": "A", "score": 0.9},
            {"reasoning": "B wins", "winner": "B", "score": 0.3},
        ]
        llm = MockLLM(responses)
        metric = PairwiseMetric(llm=llm, num_votes=3, mitigate_position_bias=False)
        ec = EvalCase(input="q", output="a", expected="b")
        score = await metric.a_measure(ec)
        assert "vote_counts" in score.metadata
        assert score.metadata["vote_counts"]["A"] == 2
        # Mean of 0.8, 0.9, 0.3 ≈ 0.667
        assert abs(score.value - 0.667) < 0.01

    async def test_num_votes_with_bias_mitigation(self):
        llm = MockLLM({"reasoning": "ok", "winner": "A", "score": 0.7})
        metric = PairwiseMetric(llm=llm, num_votes=3, mitigate_position_bias=True)
        ec = EvalCase(input="q", output="a", expected="b")
        score = await metric.a_measure(ec)
        # 3 votes per ordering * 2 orderings = 6 calls
        assert llm.call_count == 6
        assert "vote_counts" in score.metadata
        assert "position_bias_delta" in score.metadata

    def test_invalid_num_votes(self):
        llm = MockLLM({"reasoning": "ok", "winner": "A", "score": 0.5})
        with pytest.raises(ValueError, match="num_votes must be >= 1"):
            PairwiseMetric(llm=llm, num_votes=0)
