"""Tests for BaseBenchmark and BenchmarkResult."""

from __future__ import annotations

import pytest

from harness_evals.benchmarks.base import BaseBenchmark, BenchmarkResult
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink


class SimpleBenchmark(BaseBenchmark):
    """Minimal benchmark for testing the base class."""

    def __init__(self) -> None:
        super().__init__(name="simple_test")
        self._dataset = [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"},
        ]

    async def load_dataset(self, *, offline: bool = False) -> list[dict]:
        return self._dataset

    def format_prompt(self, item: dict, *, shots: int = 0) -> str:
        return f"Q: {item['question']}\nA:"

    def score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        if response.strip() == item["answer"]:
            return 1.0, None
        return 0.0, f"Expected {item['answer']}, got {response.strip()}"


class CollectingSink(BaseSink):
    """Sink that collects write calls for verification."""

    def __init__(self) -> None:
        self.writes: list[tuple[list[Score], EvalCase]] = []
        self.finalized = False
        self.shut_down = False

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        self.writes.append((scores, eval_case))

    def finalize(self) -> None:
        self.finalized = True

    def shutdown(self) -> None:
        self.shut_down = True


@pytest.mark.unit
class TestBenchmarkResult:
    def test_basic_construction(self):
        result = BenchmarkResult(
            name="test",
            accuracy=0.75,
            num_correct=3,
            num_total=4,
        )
        assert result.accuracy == 0.75
        assert result.pass_at_1 == 0.75
        assert result.num_correct == 3
        assert result.num_total == 4

    def test_to_dict(self):
        score = Score(name="test", value=1.0, threshold=0.5)
        result = BenchmarkResult(
            name="test",
            accuracy=1.0,
            num_correct=1,
            num_total=1,
            scores=[score],
            metadata={"key": "value"},
        )
        d = result.to_dict()
        assert d["name"] == "test"
        assert d["accuracy"] == 1.0
        assert d["metadata"] == {"key": "value"}
        assert len(d["scores"]) == 1


@pytest.mark.unit
class TestBaseBenchmark:
    async def test_run_all_correct(self, mock_llm_factory):
        llm = mock_llm_factory(responses=["2", "4", "6"])
        benchmark = SimpleBenchmark()
        result = await benchmark.run(llm)

        assert result.accuracy == 1.0
        assert result.num_correct == 3
        assert result.num_total == 3
        assert len(result.scores) == 3
        assert all(s.passed for s in result.scores)

    async def test_run_partial_correct(self, mock_llm_factory):
        llm = mock_llm_factory(responses=["2", "5", "6"])
        benchmark = SimpleBenchmark()
        result = await benchmark.run(llm)

        assert result.accuracy == pytest.approx(2 / 3)
        assert result.num_correct == 2
        assert result.num_total == 3

    async def test_run_with_limit(self, mock_llm_factory):
        llm = mock_llm_factory(responses=["2", "4"])
        benchmark = SimpleBenchmark()
        result = await benchmark.run(llm, limit=2)

        assert result.num_total == 2
        assert llm.call_count == 2

    async def test_run_with_sinks(self, mock_llm_factory):
        llm = mock_llm_factory(responses=["2", "4", "6"])
        sink = CollectingSink()
        benchmark = SimpleBenchmark()
        await benchmark.run(llm, sinks=[sink])

        assert len(sink.writes) == 3
        assert sink.finalized
        assert sink.shut_down

    async def test_run_handles_errors(self, mock_llm_factory):
        """Model errors should produce zero scores, not crash the run."""

        class FailingLLM(mock_llm_factory().__class__):
            async def generate(self, prompt: str, **kwargs: object) -> str:
                raise RuntimeError("API error")

        llm = FailingLLM()
        benchmark = SimpleBenchmark()
        result = await benchmark.run(llm)

        assert result.num_total == 3
        assert result.num_correct == 0
        assert all(s.value == 0.0 for s in result.scores)

    async def test_concurrency_zero_raises(self, mock_llm_factory):
        """concurrency=0 should raise ValueError, not hang."""
        llm = mock_llm_factory(responses=["2"])
        benchmark = SimpleBenchmark()
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            await benchmark.run(llm, concurrency=0)

    async def test_default_shots_used(self, mock_llm_factory):
        """When shots=None (default), benchmark's default_shots should be used."""

        class FiveShotBenchmark(SimpleBenchmark):
            def __init__(self) -> None:
                super().__init__()
                self.default_shots = 5
                self.received_shots: list[int] = []

            def format_prompt(self, item: dict, *, shots: int = 0) -> str:
                self.received_shots.append(shots)
                return f"Q: {item['question']}\nA:"

        llm = mock_llm_factory(responses=["2", "4", "6"])
        benchmark = FiveShotBenchmark()
        await benchmark.run(llm)
        assert all(s == 5 for s in benchmark.received_shots)

    async def test_explicit_shots_overrides_default(self, mock_llm_factory):
        """Passing shots=2 should override the benchmark's default."""

        class FiveShotBenchmark(SimpleBenchmark):
            def __init__(self) -> None:
                super().__init__()
                self.default_shots = 5
                self.received_shots: list[int] = []

            def format_prompt(self, item: dict, *, shots: int = 0) -> str:
                self.received_shots.append(shots)
                return f"Q: {item['question']}\nA:"

        llm = mock_llm_factory(responses=["2", "4", "6"])
        benchmark = FiveShotBenchmark()
        await benchmark.run(llm, shots=2)
        assert all(s == 2 for s in benchmark.received_shots)

    async def test_metrics_in_result(self, mock_llm_factory):
        """BenchmarkResult.metrics should contain accuracy."""
        llm = mock_llm_factory(responses=["2", "4", "6"])
        benchmark = SimpleBenchmark()
        result = await benchmark.run(llm)
        assert "accuracy" in result.metrics
        assert result.metrics["accuracy"] == 1.0
