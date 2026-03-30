"""Tests for agent metrics."""

import pytest

from harness_evals import EvalCase
from harness_evals.core.types import Message, ToolCall
from harness_evals.metrics.agent.argument_correctness import ArgumentCorrectnessMetric
from harness_evals.metrics.agent.plan_adherence import PlanAdherenceMetric
from harness_evals.metrics.agent.plan_quality import PlanQualityMetric
from harness_evals.metrics.agent.step_efficiency import StepEfficiencyMetric
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
            tool_calls=[ToolCall(name="search"), ToolCall(name="summarize"), ToolCall(name="respond")],
            expected_tools=["search", "summarize", "respond"],
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.passed
        assert score.value == 1.0

    def test_partial_match(self):
        ec = EvalCase(
            input="task",
            output="result",
            tool_calls=[ToolCall(name="search"), ToolCall(name="wrong_tool"), ToolCall(name="respond")],
            expected_tools=["search", "summarize", "respond"],
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert abs(score.value - 2 / 3) < 0.01

    def test_no_match(self):
        ec = EvalCase(
            input="task",
            output="result",
            tool_calls=[ToolCall(name="a"), ToolCall(name="b")],
            expected_tools=["x", "y"],
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.value == 0.0

    def test_different_lengths(self):
        ec = EvalCase(
            input="task",
            output="result",
            tool_calls=[ToolCall(name="search")],
            expected_tools=["search", "summarize", "respond"],
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert abs(score.value - 1 / 3) < 0.01

    def test_extra_tools_penalized(self):
        ec = EvalCase(
            input="task",
            output="result",
            tool_calls=[
                ToolCall(name="search"),
                ToolCall(name="summarize"),
                ToolCall(name="respond"),
                ToolCall(name="extra"),
            ],
            expected_tools=["search", "summarize", "respond"],
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert abs(score.value - 3 / 4) < 0.01

    def test_missing_expected_tools(self):
        ec = EvalCase(
            input="task",
            output="result",
            tool_calls=[ToolCall(name="search")],
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.value == 0.0
        assert "expected_tools" in score.reason

    def test_missing_tool_calls(self):
        ec = EvalCase(
            input="task",
            output="result",
            expected_tools=["search"],
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.value == 0.0
        assert "tool_calls" in score.reason

    def test_no_fields(self):
        ec = EvalCase(input="task", output="result")
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.value == 0.0

    def test_empty_expected_empty_called(self):
        ec = EvalCase(
            input="task",
            output="result",
            tool_calls=[],
            expected_tools=[],
        )
        score = ToolCorrectnessMetric(mode="exact").measure(ec)
        assert score.value == 1.0

    def test_empty_expected_nonempty_called(self):
        ec = EvalCase(
            input="task",
            output="result",
            tool_calls=[ToolCall(name="search")],
            expected_tools=[],
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
            tool_calls=[
                ToolCall(name="search"),
                ToolCall(name="summarize"),
                ToolCall(name="respond"),
                ToolCall(name="log"),
            ],
            expected_tools=["search", "respond"],
        )
        score = ToolCorrectnessMetric(mode="subset").measure(ec)
        assert score.passed
        assert score.value == 1.0

    def test_partial_subset(self):
        ec = EvalCase(
            input="task",
            output="result",
            tool_calls=[ToolCall(name="search"), ToolCall(name="log")],
            expected_tools=["search", "summarize", "respond"],
        )
        score = ToolCorrectnessMetric(mode="subset").measure(ec)
        assert abs(score.value - 1 / 3) < 0.01
        assert "summarize" in score.metadata["missing"]
        assert "respond" in score.metadata["missing"]

    def test_none_present(self):
        ec = EvalCase(
            input="task",
            output="result",
            tool_calls=[ToolCall(name="a"), ToolCall(name="b")],
            expected_tools=["x", "y"],
        )
        score = ToolCorrectnessMetric(mode="subset").measure(ec)
        assert score.value == 0.0

    def test_duplicate_expected_tools(self):
        ec = EvalCase(
            input="task",
            output="result",
            tool_calls=[ToolCall(name="search")],
            expected_tools=["search", "search"],
        )
        score = ToolCorrectnessMetric(mode="subset").measure(ec)
        assert score.value == 0.5
        assert score.metadata["missing"] == ["search"]

    def test_duplicate_expected_and_called(self):
        ec = EvalCase(
            input="task",
            output="result",
            tool_calls=[ToolCall(name="search"), ToolCall(name="search"), ToolCall(name="respond")],
            expected_tools=["search", "search"],
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


# ---------------------------------------------------------------------------
# ArgumentCorrectnessMetric
# ---------------------------------------------------------------------------


SAMPLE_TOOL_CALLS = [
    ToolCall(name="search", input={"query": "weather Paris"}),
    ToolCall(name="summarize", input={"text": "It is sunny in Paris"}),
]


@pytest.mark.unit
class TestArgumentCorrectnessMetric:
    async def test_all_correct(self):
        llm = MockLLM(default={"reasoning": "All arguments correct", "score": 0.95})
        metric = ArgumentCorrectnessMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="What is the weather in Paris?", output="Sunny", tool_calls=SAMPLE_TOOL_CALLS)
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 0.95
        assert score.metadata["n_tool_calls"] == 2

    async def test_partially_correct(self):
        llm = MockLLM(default={"reasoning": "Wrong query param", "score": 0.4})
        metric = ArgumentCorrectnessMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="task", output="result", tool_calls=SAMPLE_TOOL_CALLS)
        score = await metric.a_measure(ec)
        assert not score.passed
        assert abs(score.value - 0.4) < 0.01

    async def test_no_tool_calls(self):
        llm = MockLLM(default={"reasoning": "n/a", "score": 1.0})
        metric = ArgumentCorrectnessMetric(llm=llm)
        ec = EvalCase(input="task", output="result")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "tool_calls" in score.reason.lower()

    async def test_score_clamped_above(self):
        llm = MockLLM(default={"reasoning": "edge", "score": 1.5})
        metric = ArgumentCorrectnessMetric(llm=llm)
        ec = EvalCase(input="task", output="result", tool_calls=SAMPLE_TOOL_CALLS)
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    async def test_score_clamped_below(self):
        llm = MockLLM(default={"reasoning": "edge", "score": -0.5})
        metric = ArgumentCorrectnessMetric(llm=llm)
        ec = EvalCase(input="task", output="result", tool_calls=SAMPLE_TOOL_CALLS)
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.85})
        metric = ArgumentCorrectnessMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="task", output="result", tool_calls=SAMPLE_TOOL_CALLS)
        score = metric.measure(ec)
        assert score.passed
        assert score.value == 0.85


# ---------------------------------------------------------------------------
# PlanQualityMetric
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES = [
    Message(role="user", content="Book a flight to Paris and a hotel"),
    Message(role="assistant", content="I'll search for flights first, then hotels."),
    Message(role="assistant", content="Found flight AA123. Now searching hotels."),
    Message(role="assistant", content="Booked hotel Le Marais. All done."),
]


@pytest.mark.unit
class TestPlanQualityMetric:
    async def test_good_plan(self):
        llm = MockLLM(default={"reasoning": "Complete and logical plan", "score": 0.9})
        metric = PlanQualityMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="Book travel", output="done", messages=SAMPLE_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 0.9
        assert score.metadata["n_messages"] == 4

    async def test_weak_plan(self):
        llm = MockLLM(default={"reasoning": "Missing steps", "score": 0.25})
        metric = PlanQualityMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="task", output="result", messages=SAMPLE_MESSAGES)
        score = await metric.a_measure(ec)
        assert not score.passed

    async def test_no_messages(self):
        llm = MockLLM(default={"reasoning": "n/a", "score": 1.0})
        metric = PlanQualityMetric(llm=llm)
        ec = EvalCase(input="task", output="result")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "messages" in score.reason.lower()

    async def test_score_clamped(self):
        llm = MockLLM(default={"reasoning": "edge", "score": 1.5})
        metric = PlanQualityMetric(llm=llm)
        ec = EvalCase(input="task", output="result", messages=SAMPLE_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.8})
        metric = PlanQualityMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="task", output="result", messages=SAMPLE_MESSAGES)
        score = metric.measure(ec)
        assert score.passed


# ---------------------------------------------------------------------------
# PlanAdherenceMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlanAdherenceMetric:
    async def test_perfect_adherence(self):
        llm = MockLLM(default={"reasoning": "All steps followed", "score": 1.0})
        metric = PlanAdherenceMetric(llm=llm, threshold=0.7)
        ec = EvalCase(
            input="Book travel",
            output="done",
            messages=SAMPLE_MESSAGES,
            tool_calls=SAMPLE_TOOL_CALLS,
        )
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 1.0
        assert score.metadata["n_messages"] == 4
        assert score.metadata["n_tool_calls"] == 2

    async def test_partial_adherence(self):
        llm = MockLLM(default={"reasoning": "Skipped hotel step", "score": 0.5})
        metric = PlanAdherenceMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="task", output="result", messages=SAMPLE_MESSAGES, tool_calls=SAMPLE_TOOL_CALLS)
        score = await metric.a_measure(ec)
        assert not score.passed

    async def test_no_messages(self):
        llm = MockLLM(default={"reasoning": "n/a", "score": 1.0})
        metric = PlanAdherenceMetric(llm=llm)
        ec = EvalCase(input="task", output="result")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "messages" in score.reason.lower()

    async def test_no_tool_calls(self):
        llm = MockLLM(default={"reasoning": "No execution actions", "score": 0.0})
        metric = PlanAdherenceMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="task", output="result", messages=SAMPLE_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.8})
        metric = PlanAdherenceMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="task", output="result", messages=SAMPLE_MESSAGES, tool_calls=SAMPLE_TOOL_CALLS)
        score = metric.measure(ec)
        assert score.passed


# ---------------------------------------------------------------------------
# StepEfficiencyMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStepEfficiencyMetric:
    async def test_perfectly_efficient(self):
        llm = MockLLM(default={"reasoning": "All steps necessary", "score": 1.0})
        metric = StepEfficiencyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="task", output="result", messages=SAMPLE_MESSAGES, tool_calls=SAMPLE_TOOL_CALLS)
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 1.0

    async def test_inefficient(self):
        llm = MockLLM(default={"reasoning": "Redundant retrieval", "score": 0.25})
        metric = StepEfficiencyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="task", output="result", messages=SAMPLE_MESSAGES, tool_calls=SAMPLE_TOOL_CALLS)
        score = await metric.a_measure(ec)
        assert not score.passed
        assert score.value == 0.25

    async def test_no_messages_or_tool_calls(self):
        llm = MockLLM(default={"reasoning": "n/a", "score": 1.0})
        metric = StepEfficiencyMetric(llm=llm)
        ec = EvalCase(input="task", output="result")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "messages" in score.reason.lower() or "tool_calls" in score.reason.lower()

    async def test_only_tool_calls(self):
        llm = MockLLM(default={"reasoning": "Efficient tool use", "score": 0.9})
        metric = StepEfficiencyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="task", output="result", tool_calls=SAMPLE_TOOL_CALLS)
        score = await metric.a_measure(ec)
        assert score.passed

    async def test_score_clamped_above(self):
        llm = MockLLM(default={"reasoning": "edge", "score": 1.5})
        metric = StepEfficiencyMetric(llm=llm)
        ec = EvalCase(input="task", output="result", messages=SAMPLE_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    async def test_score_clamped_below(self):
        llm = MockLLM(default={"reasoning": "edge", "score": -0.3})
        metric = StepEfficiencyMetric(llm=llm)
        ec = EvalCase(input="task", output="result", messages=SAMPLE_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.8})
        metric = StepEfficiencyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="task", output="result", messages=SAMPLE_MESSAGES, tool_calls=SAMPLE_TOOL_CALLS)
        score = metric.measure(ec)
        assert score.passed
