"""Tests for agent metrics."""

import pytest

from harness_evals import EvalCase
from harness_evals.metrics.agent.task_completion import TaskCompletionMetric
from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric
from tests.conftest import MockLLM

# ---------------------------------------------------------------------------
# ToolCorrectnessMetric — exact mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolCorrectnessExact:
    def test_perfect_match(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={
                "tools_called": ["search", "summarize", "respond"],
                "expected_tools": ["search", "summarize", "respond"],
            },
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.passed
        assert score.value == 1.0

    def test_partial_match(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={
                "tools_called": ["search", "wrong_tool", "respond"],
                "expected_tools": ["search", "summarize", "respond"],
            },
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert abs(score.value - 2 / 3) < 0.01

    def test_no_match(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={
                "tools_called": ["a", "b"],
                "expected_tools": ["x", "y"],
            },
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.value == 0.0

    def test_different_lengths(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={
                "tools_called": ["search"],
                "expected_tools": ["search", "summarize", "respond"],
            },
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert abs(score.value - 1 / 3) < 0.01

    def test_extra_tools_penalized(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={
                "tools_called": ["search", "summarize", "respond", "extra"],
                "expected_tools": ["search", "summarize", "respond"],
            },
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert abs(score.value - 3 / 4) < 0.01

    def test_missing_expected_tools(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={"tools_called": ["search"]},
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.value == 0.0
        assert "expected_tools" in score.reason

    def test_missing_tools_called(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={"expected_tools": ["search"]},
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.value == 0.0
        assert "tools_called" in score.reason

    def test_no_metadata(self):
        ec = EvalCase(input="task", output="result")
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.value == 0.0

    def test_empty_expected_empty_called(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={"tools_called": [], "expected_tools": []},
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.value == 1.0

    def test_empty_expected_nonempty_called(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={"tools_called": ["search"], "expected_tools": []},
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.value == 0.0


# ---------------------------------------------------------------------------
# ToolCorrectnessMetric — subset mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolCorrectnessSubset:
    def test_all_expected_present(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={
                "tools_called": ["search", "summarize", "respond", "log"],
                "expected_tools": ["search", "respond"],
            },
        )
        score = ToolCorrectnessMetric(mode="subset").measure(ec)
        assert score.passed
        assert score.value == 1.0

    def test_partial_subset(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={
                "tools_called": ["search", "log"],
                "expected_tools": ["search", "summarize", "respond"],
            },
        )
        score = ToolCorrectnessMetric(mode="subset").measure(ec)
        assert abs(score.value - 1 / 3) < 0.01
        assert "summarize" in score.metadata["missing"]
        assert "respond" in score.metadata["missing"]

    def test_none_present(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={
                "tools_called": ["a", "b"],
                "expected_tools": ["x", "y"],
            },
        )
        score = ToolCorrectnessMetric(mode="subset").measure(ec)
        assert score.value == 0.0

    def test_duplicate_expected_tools(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={
                "tools_called": ["search"],
                "expected_tools": ["search", "search"],
            },
        )
        score = ToolCorrectnessMetric(mode="subset").measure(ec)
        assert score.value == 0.5
        assert score.metadata["missing"] == ["search"]

    def test_duplicate_expected_and_called(self):
        ec = EvalCase(
            input="task",
            output="result",
            metadata={
                "tools_called": ["search", "search", "respond"],
                "expected_tools": ["search", "search"],
            },
        )
        score = ToolCorrectnessMetric(mode="subset").measure(ec)
        assert score.value == 1.0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            ToolCorrectnessMetric(mode="invalid")


# ---------------------------------------------------------------------------
# TaskCompletionMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskCompletionMetric:
    @pytest.mark.asyncio
    async def test_fully_completed(self):
        llm = MockLLM(default={"reasoning": "Task fully completed", "score": 1.0})
        metric = TaskCompletionMetric(llm=llm, threshold=0.7)
        ec = EvalCase(
            input="Write a hello world program",
            output='print("Hello, world!")',
        )
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_partially_completed(self):
        llm = MockLLM(default={"reasoning": "Missing error handling", "score": 0.6})
        metric = TaskCompletionMetric(llm=llm, threshold=0.7)
        ec = EvalCase(
            input="Write a REST API with error handling",
            output="def get(): return data",
        )
        score = await metric.a_measure(ec)
        assert not score.passed
        assert abs(score.value - 0.6) < 0.01

    @pytest.mark.asyncio
    async def test_not_attempted(self):
        llm = MockLLM(default={"reasoning": "Agent refused the task", "score": 0.0})
        metric = TaskCompletionMetric(llm=llm, threshold=0.7)
        ec = EvalCase(
            input="Write a deployment script",
            output="I cannot help with that.",
        )
        score = await metric.a_measure(ec)
        assert not score.passed
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_with_expected_output(self):
        llm = MockLLM(default={"reasoning": "Matches expected", "score": 0.95})
        metric = TaskCompletionMetric(llm=llm, threshold=0.7)
        ec = EvalCase(
            input="What is 2+2?",
            output="4",
            expected="4",
        )
        score = await metric.a_measure(ec)
        assert score.passed

    @pytest.mark.asyncio
    async def test_score_clamped(self):
        llm = MockLLM(default={"reasoning": "edge", "score": 1.5})
        metric = TaskCompletionMetric(llm=llm)
        ec = EvalCase(input="task", output="result")
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "Done", "score": 0.9})
        metric = TaskCompletionMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="Write a test", output="def test_foo(): pass")
        score = metric.measure(ec)
        assert score.passed
        assert score.value == 0.9
