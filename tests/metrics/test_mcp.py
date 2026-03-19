"""Tests for MCP metrics: ToolSelectionAccuracy, MCPTraceCompleteness."""

import pytest

from harness_evals.core.eval_case import EvalCase
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
            metadata={
                "mcp_trace": [{"tool": "search"}, {"tool": "read"}],
                "expected_tools": ["search", "read"],
            },
        )
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 1.0

    def test_partial_match(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={
                "mcp_trace": [{"tool": "search"}, {"tool": "write"}],
                "expected_tools": ["search", "read"],
            },
        )
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == pytest.approx(1 / 3)

    def test_no_match(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={
                "mcp_trace": [{"tool": "delete"}],
                "expected_tools": ["search", "read"],
            },
        )
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 0.0

    def test_extra_tools(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={
                "mcp_trace": [{"tool": "search"}, {"tool": "read"}, {"tool": "write"}],
                "expected_tools": ["search", "read"],
            },
        )
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == pytest.approx(2 / 3)

    def test_missing_mcp_trace(self):
        ec = EvalCase(input="q", output="a", metadata={"expected_tools": ["search"]})
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 0.0
        assert "mcp_trace" in score.reason

    def test_missing_expected_tools(self):
        ec = EvalCase(input="q", output="a", metadata={"mcp_trace": [{"tool": "search"}]})
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 0.0
        assert "expected_tools" in score.reason

    def test_empty_both(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"mcp_trace": [], "expected_tools": []},
        )
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 1.0

    def test_duplicate_tools(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={
                "mcp_trace": [{"tool": "search"}, {"tool": "search"}],
                "expected_tools": ["search", "search"],
            },
        )
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 1.0

    def test_no_metadata(self):
        ec = EvalCase(input="q", output="a")
        score = ToolSelectionAccuracyMetric().measure(ec)
        assert score.value == 0.0


# ---------------------------------------------------------------------------
# MCPTraceCompletenessMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMCPTraceCompleteness:
    def test_full_trace(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={
                "mcp_trace": [
                    {"tool": "search", "input": {"q": "foo"}},
                    {"tool": "read", "input": {"path": "/a"}},
                ],
                "expected_trace": [
                    {"tool": "search", "input": {"q": "foo"}},
                    {"tool": "read", "input": {"path": "/a"}},
                ],
            },
        )
        score = MCPTraceCompletenessMetric().measure(ec)
        assert score.value == 1.0

    def test_partial_trace(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={
                "mcp_trace": [{"tool": "search", "input": {"q": "foo"}}],
                "expected_trace": [
                    {"tool": "search", "input": {"q": "foo"}},
                    {"tool": "read", "input": {"path": "/a"}},
                ],
            },
        )
        score = MCPTraceCompletenessMetric().measure(ec)
        assert score.value == pytest.approx(0.5)

    def test_missing_operations(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={
                "mcp_trace": [{"tool": "write", "input": {"data": "x"}}],
                "expected_trace": [
                    {"tool": "search", "input": {"q": "foo"}},
                    {"tool": "read", "input": {"path": "/a"}},
                ],
            },
        )
        score = MCPTraceCompletenessMetric().measure(ec)
        assert score.value == 0.0

    def test_extra_operations(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={
                "mcp_trace": [
                    {"tool": "search", "input": {"q": "foo"}},
                    {"tool": "read", "input": {"path": "/a"}},
                    {"tool": "write", "input": {"data": "x"}},
                ],
                "expected_trace": [
                    {"tool": "search", "input": {"q": "foo"}},
                    {"tool": "read", "input": {"path": "/a"}},
                ],
            },
        )
        score = MCPTraceCompletenessMetric().measure(ec)
        assert score.value == 1.0

    def test_empty_expected(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={
                "mcp_trace": [{"tool": "search", "input": {"q": "foo"}}],
                "expected_trace": [],
            },
        )
        score = MCPTraceCompletenessMetric().measure(ec)
        assert score.value == 1.0

    def test_missing_metadata(self):
        ec = EvalCase(input="q", output="a")
        score = MCPTraceCompletenessMetric().measure(ec)
        assert score.value == 0.0
        assert "mcp_trace" in score.reason

    def test_input_mismatch(self):
        """Same tool name but different input should not match."""
        ec = EvalCase(
            input="q",
            output="a",
            metadata={
                "mcp_trace": [{"tool": "search", "input": {"q": "bar"}}],
                "expected_trace": [{"tool": "search", "input": {"q": "foo"}}],
            },
        )
        score = MCPTraceCompletenessMetric().measure(ec)
        assert score.value == 0.0

    def test_metadata_fields(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={
                "mcp_trace": [{"tool": "search", "input": {"q": "foo"}}],
                "expected_trace": [
                    {"tool": "search", "input": {"q": "foo"}},
                    {"tool": "read", "input": {"path": "/a"}},
                ],
            },
        )
        score = MCPTraceCompletenessMetric().measure(ec)
        assert score.metadata["found"] == 1
        assert score.metadata["total_expected"] == 2
        assert len(score.metadata["missing"]) == 1
