"""Tests for MCP metrics: ToolSelectionAccuracy, MCPTraceCompleteness."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import ToolCall
from harness_evals.metrics.mcp.tool_selection import ToolSelectionAccuracyMetric
from harness_evals.metrics.mcp.trace_completeness import MCPTraceCompletenessMetric

# ---------------------------------------------------------------------------
# ToolSelectionAccuracyMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolSelectionAccuracy:
    def test_all_match(self):
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[ToolCall(name="search"), ToolCall(name="read")],
            expected_tools=["search", "read"],
        )
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 1.0

    def test_partial_match(self):
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[ToolCall(name="search"), ToolCall(name="write")],
            expected_tools=["search", "read"],
        )
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == pytest.approx(1 / 3)

    def test_no_match(self):
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[ToolCall(name="delete")],
            expected_tools=["search", "read"],
        )
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 0.0

    def test_extra_tools(self):
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[ToolCall(name="search"), ToolCall(name="read"), ToolCall(name="write")],
            expected_tools=["search", "read"],
        )
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == pytest.approx(2 / 3)

    def test_missing_tool_calls(self):
        ec = EvalCase(input="q", output="a", expected_tools=["search"])
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 0.0
        assert "tool_calls" in score.reason

    def test_missing_expected_tools(self):
        ec = EvalCase(input="q", output="a", tool_calls=[ToolCall(name="search")])
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 0.0
        assert "expected_tools" in score.reason

    def test_empty_both(self):
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[],
            expected_tools=[],
        )
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 1.0

    def test_duplicate_tools(self):
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[ToolCall(name="search"), ToolCall(name="search")],
            expected_tools=["search", "search"],
        )
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 1.0

    def test_no_fields(self):
        ec = EvalCase(input="q", output="a")
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 0.0


# ---------------------------------------------------------------------------
# MCPTraceCompletenessMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMCPTraceCompleteness:
    def test_full_trace(self):
        expected = [
            ToolCall(name="search", input={"q": "foo"}),
            ToolCall(name="read", input={"path": "/a"}),
        ]
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[
                ToolCall(name="search", input={"q": "foo"}),
                ToolCall(name="read", input={"path": "/a"}),
            ],
        )
        score = MCPTraceCompletenessMetric(expected_trace=expected).measure(ec)
        assert score.value == 1.0

    def test_partial_trace(self):
        expected = [
            ToolCall(name="search", input={"q": "foo"}),
            ToolCall(name="read", input={"path": "/a"}),
        ]
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[ToolCall(name="search", input={"q": "foo"})],
        )
        score = MCPTraceCompletenessMetric(expected_trace=expected).measure(ec)
        assert score.value == pytest.approx(0.5)

    def test_missing_operations(self):
        expected = [
            ToolCall(name="search", input={"q": "foo"}),
            ToolCall(name="read", input={"path": "/a"}),
        ]
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[ToolCall(name="write", input={"data": "x"})],
        )
        score = MCPTraceCompletenessMetric(expected_trace=expected).measure(ec)
        assert score.value == 0.0

    def test_extra_operations(self):
        expected = [
            ToolCall(name="search", input={"q": "foo"}),
            ToolCall(name="read", input={"path": "/a"}),
        ]
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[
                ToolCall(name="search", input={"q": "foo"}),
                ToolCall(name="read", input={"path": "/a"}),
                ToolCall(name="write", input={"data": "x"}),
            ],
        )
        score = MCPTraceCompletenessMetric(expected_trace=expected).measure(ec)
        assert score.value == 1.0

    def test_empty_expected(self):
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[ToolCall(name="search", input={"q": "foo"})],
        )
        score = MCPTraceCompletenessMetric(expected_trace=[]).measure(ec)
        assert score.value == 1.0

    def test_missing_tool_calls(self):
        expected = [ToolCall(name="search", input={"q": "foo"})]
        ec = EvalCase(input="q", output="a")
        score = MCPTraceCompletenessMetric(expected_trace=expected).measure(ec)
        assert score.value == 0.0
        assert "tool_calls" in score.reason

    def test_input_mismatch(self):
        """Same tool name but different input should not match."""
        expected = [ToolCall(name="search", input={"q": "foo"})]
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[ToolCall(name="search", input={"q": "bar"})],
        )
        score = MCPTraceCompletenessMetric(expected_trace=expected).measure(ec)
        assert score.value == 0.0

    def test_metadata_fields(self):
        expected = [
            ToolCall(name="search", input={"q": "foo"}),
            ToolCall(name="read", input={"path": "/a"}),
        ]
        ec = EvalCase(
            input="q",
            output="a",
            tool_calls=[ToolCall(name="search", input={"q": "foo"})],
        )
        score = MCPTraceCompletenessMetric(expected_trace=expected).measure(ec)
        assert score.metadata["found"] == 1
        assert score.metadata["total_expected"] == 2
        assert len(score.metadata["missing"]) == 1
