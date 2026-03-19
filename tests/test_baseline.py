"""Tests for baseline store and comparison system."""

import json

import pytest

from harness_evals.baseline.compare import BaselineResult, MetricDelta, compare_to_baseline
from harness_evals.baseline.json_store import JsonBaselineStore
from harness_evals.core.score import Score


def _score(name: str, value: float, threshold: float = 0.8) -> Score:
    """Shorthand for creating test scores."""
    return Score(name=name, value=value, threshold=threshold)


# ---------------------------------------------------------------------------
# JsonBaselineStore — round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJsonBaselineStoreRoundTrip:
    def test_save_and_load(self, tmp_path):
        store = JsonBaselineStore(baseline_dir=str(tmp_path / "baselines"))
        scores = {
            "exact_match": [_score("exact_match", 1.0), _score("exact_match", 0.5)],
            "latency": [_score("latency", 0.9, threshold=0.5)],
        }
        store.save("run-001", scores)
        loaded = store.load("run-001")

        assert set(loaded.keys()) == {"exact_match", "latency"}
        assert len(loaded["exact_match"]) == 2
        assert loaded["exact_match"][0].value == 1.0
        assert loaded["exact_match"][1].value == 0.5
        assert loaded["latency"][0].threshold == 0.5

    def test_preserves_reason_and_metadata(self, tmp_path):
        store = JsonBaselineStore(baseline_dir=str(tmp_path / "baselines"))
        scores = {
            "metric_a": [
                Score(name="metric_a", value=0.8, threshold=0.7, reason="some reason", metadata={"k": 42}),
            ],
        }
        store.save("run-002", scores)
        loaded = store.load("run-002")

        s = loaded["metric_a"][0]
        assert s.reason == "some reason"
        assert s.metadata == {"k": 42}

    def test_preserves_created_at(self, tmp_path):
        store = JsonBaselineStore(baseline_dir=str(tmp_path / "baselines"))
        original = _score("m", 0.5)
        store.save("run-003", {"m": [original]})
        loaded = store.load("run-003")

        assert loaded["m"][0].created_at == original.created_at

    def test_passed_recomputed_on_load(self, tmp_path):
        store = JsonBaselineStore(baseline_dir=str(tmp_path / "baselines"))
        scores = {
            "m": [_score("m", 0.9, threshold=0.8), _score("m", 0.3, threshold=0.8)],
        }
        store.save("run-004", scores)
        loaded = store.load("run-004")

        assert loaded["m"][0].passed is True
        assert loaded["m"][1].passed is False


# ---------------------------------------------------------------------------
# JsonBaselineStore — latest tracking
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJsonBaselineStoreLatest:
    def test_load_latest(self, tmp_path):
        store = JsonBaselineStore(baseline_dir=str(tmp_path / "baselines"))
        store.save("run-a", {"m": [_score("m", 0.5)]})
        store.save("run-b", {"m": [_score("m", 0.9)]})

        latest = store.load()
        assert latest["m"][0].value == 0.9

    def test_latest_pointer_updated(self, tmp_path):
        store = JsonBaselineStore(baseline_dir=str(tmp_path / "baselines"))
        store.save("run-a", {"m": [_score("m", 0.5)]})
        store.save("run-b", {"m": [_score("m", 0.9)]})

        latest_path = tmp_path / "baselines" / "latest.json"
        data = json.loads(latest_path.read_text())
        assert data["run_id"] == "run-b"

    def test_load_latest_when_none_saved(self, tmp_path):
        store = JsonBaselineStore(baseline_dir=str(tmp_path / "baselines"))
        with pytest.raises(FileNotFoundError, match="No baselines saved"):
            store.load()

    def test_load_nonexistent_run_id(self, tmp_path):
        store = JsonBaselineStore(baseline_dir=str(tmp_path / "baselines"))
        with pytest.raises(FileNotFoundError, match="not found"):
            store.load("no-such-run")


# ---------------------------------------------------------------------------
# JsonBaselineStore — list_runs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJsonBaselineStoreListRuns:
    def test_empty_store(self, tmp_path):
        store = JsonBaselineStore(baseline_dir=str(tmp_path / "baselines"))
        assert store.list_runs() == []

    def test_lists_all_runs(self, tmp_path):
        store = JsonBaselineStore(baseline_dir=str(tmp_path / "baselines"))
        store.save("alpha", {"m": [_score("m", 0.5)]})
        store.save("beta", {"m": [_score("m", 0.6)]})
        store.save("gamma", {"m": [_score("m", 0.7)]})

        runs = store.list_runs()
        assert set(runs) == {"alpha", "beta", "gamma"}
        assert len(runs) == 3


# ---------------------------------------------------------------------------
# compare_to_baseline — regression detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompareToBaseline:
    def test_no_change(self):
        baseline = {"m": [_score("m", 0.8)]}
        current = {"m": [_score("m", 0.8)]}
        result = compare_to_baseline(current, baseline, tolerance=0.05)

        assert not result.has_regressions
        assert len(result.unchanged) == 1
        assert len(result.regressions) == 0
        assert len(result.improvements) == 0

    def test_regression_detected(self):
        baseline = {"m": [_score("m", 0.8)]}
        current = {"m": [_score("m", 0.6)]}
        result = compare_to_baseline(current, baseline, tolerance=0.05)

        assert result.has_regressions
        assert len(result.regressions) == 1
        assert result.regressions[0].metric == "m"
        assert result.regressions[0].delta == pytest.approx(-0.2)

    def test_improvement_detected(self):
        baseline = {"m": [_score("m", 0.6)]}
        current = {"m": [_score("m", 0.9)]}
        result = compare_to_baseline(current, baseline, tolerance=0.05)

        assert not result.has_regressions
        assert len(result.improvements) == 1
        assert result.improvements[0].delta == pytest.approx(0.3)

    def test_tolerance_boundary_exactly_at(self):
        """Delta == tolerance => unchanged (not a regression)."""
        baseline = {"m": [_score("m", 0.8)]}
        current = {"m": [_score("m", 0.75)]}
        result = compare_to_baseline(current, baseline, tolerance=0.05)

        assert not result.has_regressions
        assert len(result.unchanged) == 1

    def test_tolerance_boundary_just_beyond(self):
        """Delta just past tolerance => regression."""
        baseline = {"m": [_score("m", 0.8)]}
        current = {"m": [_score("m", 0.749)]}
        result = compare_to_baseline(current, baseline, tolerance=0.05)

        assert result.has_regressions
        assert len(result.regressions) == 1

    def test_tolerance_boundary_just_within(self):
        """Delta just inside tolerance => unchanged."""
        baseline = {"m": [_score("m", 0.8)]}
        current = {"m": [_score("m", 0.751)]}
        result = compare_to_baseline(current, baseline, tolerance=0.05)

        assert not result.has_regressions
        assert len(result.unchanged) == 1

    def test_multiple_metrics(self):
        baseline = {
            "exact_match": [_score("exact_match", 0.9)],
            "latency": [_score("latency", 0.8)],
            "contains": [_score("contains", 0.7)],
        }
        current = {
            "exact_match": [_score("exact_match", 0.96)],  # improvement (+0.06 > 0.05)
            "latency": [_score("latency", 0.5)],  # regression
            "contains": [_score("contains", 0.7)],  # unchanged
        }
        result = compare_to_baseline(current, baseline, tolerance=0.05)

        assert result.has_regressions
        assert len(result.regressions) == 1
        assert result.regressions[0].metric == "latency"
        assert len(result.improvements) == 1
        assert result.improvements[0].metric == "exact_match"
        assert len(result.unchanged) == 1

    def test_averages_multiple_scores(self):
        """When a metric has multiple per-case scores, average them."""
        baseline = {"m": [_score("m", 1.0), _score("m", 0.6)]}  # avg 0.8
        current = {"m": [_score("m", 0.5), _score("m", 0.5)]}  # avg 0.5
        result = compare_to_baseline(current, baseline, tolerance=0.05)

        assert result.has_regressions
        assert result.regressions[0].delta == pytest.approx(-0.3)

    def test_metric_only_in_current_ignored(self):
        """New metrics not in baseline are not regressions."""
        baseline = {"m": [_score("m", 0.8)]}
        current = {"m": [_score("m", 0.8)], "new_metric": [_score("new_metric", 0.5)]}
        result = compare_to_baseline(current, baseline, tolerance=0.05)

        assert not result.has_regressions
        assert len(result.unchanged) == 1

    def test_metric_only_in_baseline_ignored(self):
        """Removed metrics from baseline are not regressions."""
        baseline = {"m": [_score("m", 0.8)], "old_metric": [_score("old_metric", 0.9)]}
        current = {"m": [_score("m", 0.8)]}
        result = compare_to_baseline(current, baseline, tolerance=0.05)

        assert not result.has_regressions
        assert len(result.unchanged) == 1

    def test_empty_inputs(self):
        result = compare_to_baseline({}, {}, tolerance=0.05)
        assert not result.has_regressions
        assert result.summary() == "No metrics to compare"

    def test_zero_tolerance(self):
        """With tolerance=0, any change is detected."""
        baseline = {"m": [_score("m", 0.8)]}
        current = {"m": [_score("m", 0.799)]}
        result = compare_to_baseline(current, baseline, tolerance=0.0)

        assert result.has_regressions


# ---------------------------------------------------------------------------
# BaselineResult
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBaselineResult:
    def test_summary_with_all_categories(self):
        result = BaselineResult(
            tolerance=0.05,
            regressions=[MetricDelta("a", 0.8, 0.5, -0.3)],
            improvements=[MetricDelta("b", 0.5, 0.9, 0.4)],
            unchanged=[MetricDelta("c", 0.7, 0.7, 0.0)],
        )
        s = result.summary()
        assert "Regressions (1)" in s
        assert "Improvements (1)" in s
        assert "Unchanged (1)" in s

    def test_has_regressions_false_when_empty(self):
        result = BaselineResult(tolerance=0.05)
        assert not result.has_regressions


# ---------------------------------------------------------------------------
# End-to-end: store + compare
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEndToEnd:
    def test_save_compare_detect_regression(self, tmp_path):
        store = JsonBaselineStore(baseline_dir=str(tmp_path / "baselines"))

        baseline_scores = {
            "exact_match": [_score("exact_match", 0.9), _score("exact_match", 1.0)],
            "latency": [_score("latency", 0.8)],
        }
        store.save("baseline-run", baseline_scores)

        current_scores = {
            "exact_match": [_score("exact_match", 0.8), _score("exact_match", 0.9)],
            "latency": [_score("latency", 0.85)],
        }

        baseline = store.load("baseline-run")
        result = compare_to_baseline(current_scores, baseline, tolerance=0.05)

        assert result.has_regressions
        assert result.regressions[0].metric == "exact_match"
        assert not any(r.metric == "latency" for r in result.regressions)
