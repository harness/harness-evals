"""Tests for TrajectoryConsistencyMetric (distributional + sequential modes)."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.reliability.trajectory_consistency import (
    TrajectoryConsistencyMetric,
)


def _run_with_traj(trajectory: list[str]) -> EvalCase:
    return EvalCase(input="q", output="a", metadata={"trajectory": trajectory})


# ---------------------------------------------------------------------------
# Distributional mode (cosine similarity of action histograms)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrajectoryDistributional:
    def test_identical_trajectories(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["search", "read", "write"]),
                _run_with_traj(["search", "read", "write"]),
                _run_with_traj(["search", "read", "write"]),
            ],
        )
        metric = TrajectoryConsistencyMetric(mode="distributional")
        score = metric.measure(ec)
        assert score.value == pytest.approx(1.0)
        assert score.metadata["mode"] == "distributional"
        assert score.metadata["n_pairs"] == 3

    def test_completely_different(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["a"]),
                _run_with_traj(["b"]),
            ],
        )
        metric = TrajectoryConsistencyMetric(mode="distributional")
        score = metric.measure(ec)
        assert score.value == pytest.approx(0.0)

    def test_partial_overlap(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["search", "read"]),
                _run_with_traj(["search", "write"]),
            ],
        )
        metric = TrajectoryConsistencyMetric(mode="distributional")
        score = metric.measure(ec)
        assert 0.0 < score.value < 1.0

    def test_single_run(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[_run_with_traj(["search"])],
        )
        metric = TrajectoryConsistencyMetric(mode="distributional")
        score = metric.measure(ec)
        assert score.value == 0.0
        assert "at least 2" in score.reason


# ---------------------------------------------------------------------------
# Sequential mode (normalized LCS)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrajectorySequential:
    def test_identical_sequences(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["a", "b", "c"]),
                _run_with_traj(["a", "b", "c"]),
            ],
        )
        metric = TrajectoryConsistencyMetric(mode="sequential")
        score = metric.measure(ec)
        assert score.value == pytest.approx(1.0)

    def test_reversed_sequence(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["a", "b", "c"]),
                _run_with_traj(["c", "b", "a"]),
            ],
        )
        metric = TrajectoryConsistencyMetric(mode="sequential")
        score = metric.measure(ec)
        assert score.value == pytest.approx(1 / 3)

    def test_partial_subsequence(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["a", "b", "c", "d"]),
                _run_with_traj(["a", "c"]),
            ],
        )
        metric = TrajectoryConsistencyMetric(mode="sequential")
        score = metric.measure(ec)
        assert score.value == pytest.approx(2 / 4)

    def test_different_lengths(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["a", "b"]),
                _run_with_traj(["a", "b", "c", "d", "e"]),
            ],
        )
        metric = TrajectoryConsistencyMetric(mode="sequential")
        score = metric.measure(ec)
        assert score.value == pytest.approx(2 / 5)

    def test_completely_different(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["a", "b"]),
                _run_with_traj(["c", "d"]),
            ],
        )
        metric = TrajectoryConsistencyMetric(mode="sequential")
        score = metric.measure(ec)
        assert score.value == 0.0


# ---------------------------------------------------------------------------
# Mode validation & edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrajectoryEdgeCases:
    def test_modes_produce_different_scores(self):
        """Distributional and sequential should give different results for reordered actions."""
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["a", "b", "c"]),
                _run_with_traj(["c", "b", "a"]),
            ],
        )
        dist = TrajectoryConsistencyMetric(mode="distributional").measure(ec)
        seq = TrajectoryConsistencyMetric(mode="sequential").measure(ec)
        assert dist.value == pytest.approx(1.0)
        assert seq.value == pytest.approx(1 / 3)
        assert dist.value != seq.value

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            TrajectoryConsistencyMetric(mode="invalid")

    def test_missing_trajectory_metadata(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                EvalCase(input="q", output="a", metadata={}),
                EvalCase(input="q", output="a", metadata={}),
            ],
        )
        metric = TrajectoryConsistencyMetric(mode="distributional")
        score = metric.measure(ec)
        assert score.value == 0.0

    def test_no_runs(self):
        ec = EvalCase(input="q", output="a")
        metric = TrajectoryConsistencyMetric(mode="distributional")
        score = metric.measure(ec)
        assert score.value == 0.0

    def test_empty_trajectories(self):
        """Two runs that both took zero actions are perfectly consistent."""
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj([]),
                _run_with_traj([]),
            ],
        )
        dist = TrajectoryConsistencyMetric(mode="distributional").measure(ec)
        seq = TrajectoryConsistencyMetric(mode="sequential").measure(ec)
        assert dist.value == 1.0
        assert seq.value == 1.0

    def test_one_empty_one_nonempty(self):
        """One run with actions and one without are maximally inconsistent."""
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj([]),
                _run_with_traj(["a", "b"]),
            ],
        )
        dist = TrajectoryConsistencyMetric(mode="distributional").measure(ec)
        seq = TrajectoryConsistencyMetric(mode="sequential").measure(ec)
        assert dist.value == 0.0
        assert seq.value == 0.0

    def test_threshold_applied(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["a"]),
                _run_with_traj(["b"]),
            ],
        )
        metric = TrajectoryConsistencyMetric(mode="distributional", threshold=0.5)
        score = metric.measure(ec)
        assert not score.passed

    def test_max_trajectory_length_truncates(self):
        """Only the last N actions are compared when max_trajectory_length is set."""
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["x", "y", "a", "b", "c"]),
                _run_with_traj(["z", "w", "a", "b", "c"]),
            ],
        )
        full = TrajectoryConsistencyMetric(mode="sequential").measure(ec)
        capped = TrajectoryConsistencyMetric(mode="sequential", max_trajectory_length=3).measure(ec)
        assert capped.value == pytest.approx(1.0)
        assert capped.value > full.value

    def test_max_trajectory_length_distributional(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["x", "y", "a"]),
                _run_with_traj(["z", "w", "a"]),
            ],
        )
        capped = TrajectoryConsistencyMetric(mode="distributional", max_trajectory_length=1).measure(ec)
        assert capped.value == pytest.approx(1.0)

    def test_max_trajectory_length_no_op_when_short(self):
        ec = EvalCase(
            input="q",
            output="a",
            runs=[
                _run_with_traj(["a", "b"]),
                _run_with_traj(["a", "b"]),
            ],
        )
        score = TrajectoryConsistencyMetric(mode="sequential", max_trajectory_length=100).measure(ec)
        assert score.value == pytest.approx(1.0)

    def test_max_trajectory_length_validation(self):
        with pytest.raises(ValueError, match="max_trajectory_length"):
            TrajectoryConsistencyMetric(max_trajectory_length=0)
