"""Tests for TrajectoryConsistencyMetric."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import ToolCall
from harness_evals.metrics.reliability.trajectory_consistency import (
    TrajectoryConsistencyMetric,
    _cosine_similarity,
    _lcs_length,
    _normalized_lcs,
)


def _run_with_tools(names: list[str]) -> EvalCase:
    """Helper: create an EvalCase with tool_calls from a list of names."""
    return EvalCase(
        input="task",
        output="result",
        tool_calls=[ToolCall(name=n, input=None, output=None) for n in names],
    )


@pytest.mark.unit
class TestCosineHelper:
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ({}, {}, 1.0),
            ({"x": 1}, {}, 0.0),
            ({}, {"x": 1}, 0.0),
            ({"x": 1}, {"x": 1}, 1.0),
            ({"x": 1, "y": 0}, {"x": 0, "y": 1}, 0.0),
        ],
        ids=["both_empty", "a_only", "b_only", "identical", "orthogonal"],
    )
    def test_known_values(self, a, b, expected):
        from collections import Counter

        assert _cosine_similarity(Counter(a), Counter(b)) == pytest.approx(expected, abs=1e-9)


@pytest.mark.unit
class TestLCSHelper:
    @pytest.mark.parametrize(
        "a, b, expected_len",
        [
            ([], [], 0),
            (["x"], [], 0),
            ([], ["x"], 0),
            (["a", "b", "c"], ["a", "b", "c"], 3),
            (["a", "b", "c"], ["a", "c"], 2),
            (["a", "b", "c"], ["x", "y", "z"], 0),
        ],
        ids=["both_empty", "a_only", "b_only", "identical", "subsequence", "disjoint"],
    )
    def test_lcs_length(self, a, b, expected_len):
        assert _lcs_length(a, b) == expected_len

    def test_normalized_lcs_empty(self):
        assert _normalized_lcs([], []) == 1.0

    def test_normalized_lcs_identical(self):
        assert _normalized_lcs(["a", "b"], ["a", "b"]) == 1.0

    def test_normalized_lcs_partial(self):
        assert _normalized_lcs(["a", "b", "c"], ["a", "c"]) == pytest.approx(2 / 3)


@pytest.mark.unit
class TestTrajectoryConsistencyDistributional:
    def test_identical_trajectories(self):
        runs = [_run_with_tools(["search", "read", "write"]) for _ in range(3)]
        ec = EvalCase(input="task", output="result", runs=runs)
        score = TrajectoryConsistencyMetric(mode="distributional").measure(ec)
        assert score.value == pytest.approx(1.0)
        assert score.passed
        assert "distributional trajectory similarity" in score.reason

    def test_completely_different_trajectories(self):
        runs = [
            _run_with_tools(["search"]),
            _run_with_tools(["delete"]),
        ]
        ec = EvalCase(input="task", output="result", runs=runs)
        score = TrajectoryConsistencyMetric(mode="distributional").measure(ec)
        assert score.value == pytest.approx(0.0)

    def test_partially_overlapping(self):
        runs = [
            _run_with_tools(["search", "read", "write"]),
            _run_with_tools(["search", "read", "delete"]),
            _run_with_tools(["search", "write", "delete"]),
        ]
        ec = EvalCase(input="task", output="result", runs=runs)
        score = TrajectoryConsistencyMetric(mode="distributional").measure(ec)
        assert 0.0 < score.value < 1.0

    def test_metadata_contains_mode_and_k(self):
        runs = [_run_with_tools(["a"]) for _ in range(4)]
        ec = EvalCase(input="task", output="result", runs=runs)
        score = TrajectoryConsistencyMetric(mode="distributional").measure(ec)
        assert score.metadata["mode"] == "distributional"
        assert score.metadata["k"] == 4
        assert score.metadata["n_pairs"] == 6


@pytest.mark.unit
class TestTrajectoryConsistencySequential:
    def test_identical_trajectories(self):
        runs = [_run_with_tools(["a", "b", "c"]) for _ in range(3)]
        ec = EvalCase(input="task", output="result", runs=runs)
        score = TrajectoryConsistencyMetric(mode="sequential").measure(ec)
        assert score.value == pytest.approx(1.0)
        assert "sequential trajectory similarity" in score.reason

    def test_completely_different_order(self):
        runs = [
            _run_with_tools(["a", "b", "c"]),
            _run_with_tools(["c", "b", "a"]),
        ]
        ec = EvalCase(input="task", output="result", runs=runs)
        score = TrajectoryConsistencyMetric(mode="sequential").measure(ec)
        assert score.value < 1.0

    def test_metadata_contains_mode(self):
        runs = [_run_with_tools(["a"]) for _ in range(2)]
        ec = EvalCase(input="task", output="result", runs=runs)
        score = TrajectoryConsistencyMetric(mode="sequential").measure(ec)
        assert score.metadata["mode"] == "sequential"


@pytest.mark.unit
class TestTrajectoryConsistencyEdgeCases:
    def test_no_runs(self):
        ec = EvalCase(input="task", output="result")
        score = TrajectoryConsistencyMetric().measure(ec)
        assert not score.passed
        assert "No runs" in score.reason

    def test_single_run(self):
        runs = [_run_with_tools(["search"])]
        ec = EvalCase(input="task", output="result", runs=runs)
        score = TrajectoryConsistencyMetric().measure(ec)
        assert score.value == 0.0
        assert "at least 2" in score.reason

    def test_runs_without_tool_calls(self):
        runs = [
            EvalCase(input="task", output="result"),
            EvalCase(input="task", output="result"),
        ]
        ec = EvalCase(input="task", output="result", runs=runs)
        score = TrajectoryConsistencyMetric().measure(ec)
        assert score.value == 0.0
        assert "at least 2" in score.reason

    def test_one_run_with_tools_one_without(self):
        runs = [
            _run_with_tools(["search"]),
            EvalCase(input="task", output="result"),
        ]
        ec = EvalCase(input="task", output="result", runs=runs)
        score = TrajectoryConsistencyMetric().measure(ec)
        assert score.value == 0.0

    def test_max_trajectory_length_truncates(self):
        runs = [
            _run_with_tools(["a", "b", "c", "d", "e"]),
            _run_with_tools(["x", "y", "c", "d", "e"]),
        ]
        ec = EvalCase(input="task", output="result", runs=runs)
        score_full = TrajectoryConsistencyMetric(mode="sequential").measure(ec)
        score_truncated = TrajectoryConsistencyMetric(mode="sequential", max_trajectory_length=3).measure(ec)
        assert score_truncated.value > score_full.value

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            TrajectoryConsistencyMetric(mode="invalid")

    def test_invalid_max_trajectory_length_raises(self):
        with pytest.raises(ValueError, match="max_trajectory_length"):
            TrajectoryConsistencyMetric(max_trajectory_length=0)

    def test_threshold_controls_pass_fail(self):
        runs = [
            _run_with_tools(["a", "b"]),
            _run_with_tools(["a", "c"]),
        ]
        ec = EvalCase(input="task", output="result", runs=runs)
        score_low = TrajectoryConsistencyMetric(threshold=0.1).measure(ec)
        score_high = TrajectoryConsistencyMetric(threshold=0.99).measure(ec)
        assert score_low.passed
        assert not score_high.passed
