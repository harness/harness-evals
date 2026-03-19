"""Tests for Batch 5 reliability metrics: BrierScore, PromptRobustness, EnvironmentRobustness."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.reliability.brier_score import BrierScoreMetric
from harness_evals.metrics.reliability.environment_robustness import (
    EnvironmentRobustnessMetric,
)
from harness_evals.metrics.reliability.prompt_robustness import (
    PromptRobustnessMetric,
)

# ---------------------------------------------------------------------------
# BrierScoreMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBrierScore:
    def test_perfect_calibration(self):
        """Confidence 1.0 for successes, 0.0 for failures => score 1.0."""
        cases = [
            EvalCase(input="q", output="a", confidence=1.0),
            EvalCase(input="q", output="a", confidence=1.0),
            EvalCase(input="q", output="a", confidence=0.0),
        ]
        outcomes = [True, True, False]
        score = BrierScoreMetric().measure_dataset(cases, outcomes)
        assert score.value == 1.0

    def test_worst_case(self):
        """Confidence 1.0 for failures, 0.0 for successes => score 0.0."""
        cases = [
            EvalCase(input="q", output="a", confidence=1.0),
            EvalCase(input="q", output="a", confidence=0.0),
        ]
        outcomes = [False, True]
        score = BrierScoreMetric().measure_dataset(cases, outcomes)
        assert score.value == 0.0

    def test_constant_half_confidence(self):
        """Confidence 0.5 for everything => MSE = 0.25, score = 0.75."""
        cases = [
            EvalCase(input="q", output="a", confidence=0.5),
            EvalCase(input="q", output="a", confidence=0.5),
            EvalCase(input="q", output="a", confidence=0.5),
            EvalCase(input="q", output="a", confidence=0.5),
        ]
        outcomes = [True, False, True, False]
        score = BrierScoreMetric().measure_dataset(cases, outcomes)
        assert score.value == pytest.approx(0.75)

    def test_single_case_returns_zero(self):
        """Single eval case is insufficient — use measure_dataset instead."""
        ec = EvalCase(input="q", output="a", confidence=0.9)
        score = BrierScoreMetric().measure(ec)
        assert score.value == 0.0
        assert "measure_dataset" in score.reason

    def test_too_few_cases(self):
        cases = [EvalCase(input="q", output="a", confidence=0.8)]
        score = BrierScoreMetric().measure_dataset(cases, [True])
        assert score.value == 0.0
        assert "at least 2" in score.reason

    def test_missing_confidence_skipped(self):
        """Cases without confidence are excluded; remaining cases scored."""
        cases = [
            EvalCase(input="q", output="a", confidence=1.0),
            EvalCase(input="q", output="a"),  # no confidence
            EvalCase(input="q", output="a", confidence=0.0),
        ]
        outcomes = [True, True, False]
        score = BrierScoreMetric().measure_dataset(cases, outcomes)
        assert score.value == 1.0
        assert score.metadata["n_cases"] == 2

    def test_length_mismatch(self):
        cases = [EvalCase(input="q", output="a", confidence=0.5)]
        score = BrierScoreMetric().measure_dataset(cases, [True, False])
        assert score.value == 0.0
        assert "same length" in score.reason

    def test_threshold_applied(self):
        cases = [
            EvalCase(input="q", output="a", confidence=0.9),
            EvalCase(input="q", output="a", confidence=0.9),
        ]
        outcomes = [True, True]
        # MSE = (0.1)^2 = 0.01, score = 0.99
        score = BrierScoreMetric(threshold=0.95).measure_dataset(cases, outcomes)
        assert score.value == pytest.approx(0.99)
        assert score.passed

    def test_metadata_contains_mse(self):
        cases = [
            EvalCase(input="q", output="a", confidence=0.8),
            EvalCase(input="q", output="a", confidence=0.2),
        ]
        outcomes = [True, False]
        score = BrierScoreMetric().measure_dataset(cases, outcomes)
        assert "mse" in score.metadata
        assert score.metadata["mse"] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# PromptRobustnessMetric — per-case
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptRobustnessPerCase:
    def test_all_perturbed_pass(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"nominal_passed": True, "perturbed_results": [True, True, True, True, True]},
        )
        score = PromptRobustnessMetric().measure(ec)
        assert score.value == 1.0
        assert score.passed

    def test_all_perturbed_fail(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"nominal_passed": True, "perturbed_results": [False, False, False]},
        )
        score = PromptRobustnessMetric().measure(ec)
        assert score.value == 0.0
        assert not score.passed

    def test_partial_perturbed(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"nominal_passed": True, "perturbed_results": [True, True, False, False]},
        )
        score = PromptRobustnessMetric().measure(ec)
        assert score.value == pytest.approx(0.5)

    def test_nominal_failed(self):
        """When nominal fails, robustness is 1.0 — can't attribute degradation."""
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"nominal_passed": False, "perturbed_results": [False, False]},
        )
        score = PromptRobustnessMetric().measure(ec)
        assert score.value == 1.0

    def test_missing_metadata(self):
        ec = EvalCase(input="q", output="a")
        score = PromptRobustnessMetric().measure(ec)
        assert score.value == 0.0
        assert "nominal_passed" in score.reason

    def test_empty_perturbed_results(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"nominal_passed": True, "perturbed_results": []},
        )
        score = PromptRobustnessMetric().measure(ec)
        assert score.value == 0.0
        assert "empty" in score.reason


# ---------------------------------------------------------------------------
# PromptRobustnessMetric — dataset-level
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptRobustnessDataset:
    def test_perfect_robustness(self):
        """All perturbed runs match nominal — ratio = 1.0."""
        nominal = [True, True, True, False, False]
        perturbed = [
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [False, False, False],
            [False, False, False],
        ]
        score = PromptRobustnessMetric().measure_robustness(nominal, perturbed)
        assert score.value == 1.0

    def test_degradation(self):
        """Perturbed accuracy lower than nominal."""
        nominal = [True, True, True, True]  # Acc_0 = 1.0
        perturbed = [
            [True, True],
            [True, False],
            [False, False],
            [True, True],
        ]
        # Acc_perturbed = 5/8 = 0.625, ratio = 0.625/1.0 = 0.625
        score = PromptRobustnessMetric().measure_robustness(nominal, perturbed)
        assert score.value == pytest.approx(0.625)

    def test_nominal_zero_accuracy(self):
        """Nominal accuracy = 0 => score 1.0 (can't measure degradation)."""
        nominal = [False, False, False]
        perturbed = [[False], [False], [True]]
        score = PromptRobustnessMetric().measure_robustness(nominal, perturbed)
        assert score.value == 1.0

    def test_perturbed_better_than_nominal_clamped(self):
        """Perturbation accidentally improves accuracy — clamp to 1.0."""
        nominal = [True, False, False]  # Acc_0 = 1/3
        perturbed = [
            [True, True],
            [True, True],
            [True, True],
        ]
        # Acc_perturbed = 6/6 = 1.0, ratio = 1.0 / (1/3) = 3.0, clamped to 1.0
        score = PromptRobustnessMetric().measure_robustness(nominal, perturbed)
        assert score.value == 1.0

    def test_length_mismatch(self):
        score = PromptRobustnessMetric().measure_robustness([True], [[True], [False]])
        assert score.value == 0.0
        assert "same length" in score.reason

    def test_empty_dataset(self):
        score = PromptRobustnessMetric().measure_robustness([], [])
        assert score.value == 0.0

    def test_metadata_fields(self):
        nominal = [True, True]
        perturbed = [[True, False], [True, True]]
        score = PromptRobustnessMetric().measure_robustness(nominal, perturbed)
        assert "acc_nominal" in score.metadata
        assert "acc_perturbed" in score.metadata
        assert score.metadata["n_tasks"] == 2


# ---------------------------------------------------------------------------
# EnvironmentRobustnessMetric — per-case
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnvironmentRobustnessPerCase:
    def test_all_perturbed_pass(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"nominal_passed": True, "perturbed_results": [True, True, True]},
        )
        score = EnvironmentRobustnessMetric().measure(ec)
        assert score.value == 1.0

    def test_all_perturbed_fail(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"nominal_passed": True, "perturbed_results": [False, False]},
        )
        score = EnvironmentRobustnessMetric().measure(ec)
        assert score.value == 0.0

    def test_nominal_failed(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"nominal_passed": False, "perturbed_results": [False, True]},
        )
        score = EnvironmentRobustnessMetric().measure(ec)
        assert score.value == 1.0

    def test_missing_metadata(self):
        ec = EvalCase(input="q", output="a")
        score = EnvironmentRobustnessMetric().measure(ec)
        assert score.value == 0.0


# ---------------------------------------------------------------------------
# EnvironmentRobustnessMetric — dataset-level
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnvironmentRobustnessDataset:
    def test_perfect_robustness(self):
        nominal = [True, True, False]
        perturbed = [[True, True], [True, True], [False, False]]
        score = EnvironmentRobustnessMetric().measure_robustness(nominal, perturbed)
        assert score.value == 1.0

    def test_degradation(self):
        nominal = [True, True]  # Acc_0 = 1.0
        perturbed = [[True, False], [False, False]]
        # Acc_perturbed = 1/4 = 0.25
        score = EnvironmentRobustnessMetric().measure_robustness(nominal, perturbed)
        assert score.value == pytest.approx(0.25)

    def test_has_correct_name(self):
        score = EnvironmentRobustnessMetric().measure(
            EvalCase(
                input="q",
                output="a",
                metadata={"nominal_passed": True, "perturbed_results": [True]},
            )
        )
        assert score.name == "environment_robustness"


# ---------------------------------------------------------------------------
# Integration: PromptRephrase + PromptRobustnessMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptRephraseIntegration:
    """End-to-end: generate rephrasings with mocked LLM, run robustness metric."""

    async def test_rephrase_then_robustness(self, mock_llm):
        from harness_evals.perturbations.rephrase import PromptRephrase

        llm = mock_llm(
            responses=[
                {"rephrasings": ["variant 1", "variant 2", "variant 3"]},
            ]
        )
        rephrase = PromptRephrase(llm=llm)
        variants = await rephrase.perturb("original prompt", n=3)

        assert len(variants) == 3
        assert all(v != "original prompt" for v in variants)

        # Simulate: nominal passed, 2/3 perturbed also passed
        ec = EvalCase(
            input="original prompt",
            output="correct answer",
            metadata={
                "nominal_passed": True,
                "perturbed_results": [True, True, False],
            },
        )
        score = PromptRobustnessMetric(threshold=0.5).measure(ec)
        assert score.value == pytest.approx(2.0 / 3.0)
        assert score.passed
