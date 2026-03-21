"""Robustness evaluation example — measure prompt and environment robustness.

Demonstrates how to use robustness metrics at both per-case and dataset
levels, and how to save/compare results using the baseline store.

Usage:
    python examples/robustness_eval.py
"""

from __future__ import annotations

from harness_evals import EvalCase
from harness_evals.metrics.reliability import EnvironmentRobustnessMetric, PromptRobustnessMetric

# ---------------------------------------------------------------------------
# 1. Per-case robustness: one task with perturbation results in metadata
# ---------------------------------------------------------------------------

print("=== Per-case Prompt Robustness ===\n")

case = EvalCase(
    input="What is the capital of France?",
    output="Paris",
    expected="Paris",
    metadata={
        "nominal_passed": True,
        "perturbed_results": [True, True, True, False, True],
    },
)

metric = PromptRobustnessMetric(threshold=0.8)
score = metric.measure(case)
print(f"Score: {score.value:.2f}  (threshold={score.threshold})")
print(f"Passed: {score.passed}")
print(f"Reason: {score.reason}")
print(f"Metadata: {score.metadata}")

# ---------------------------------------------------------------------------
# 2. Per-case environment robustness
# ---------------------------------------------------------------------------

print("\n=== Per-case Environment Robustness ===\n")

env_case = EvalCase(
    input={"query": "list users", "format": "json"},
    output={"users": ["alice", "bob"]},
    metadata={
        "nominal_passed": True,
        "perturbed_results": [True, True, False],
    },
)

env_metric = EnvironmentRobustnessMetric(threshold=0.7)
env_score = env_metric.measure(env_case)
print(f"Score: {env_score.value:.2f}  (threshold={env_score.threshold})")
print(f"Passed: {env_score.passed}")
print(f"Reason: {env_score.reason}")

# ---------------------------------------------------------------------------
# 3. Dataset-level robustness: aggregate across multiple tasks
# ---------------------------------------------------------------------------

print("\n=== Dataset-level Prompt Robustness ===\n")

nominal_passed = [True, True, True, False, True]
perturbed_passed = [
    [True, True, True],  # task 1: 3/3 perturbed pass
    [True, True, False],  # task 2: 2/3
    [True, False, False],  # task 3: 1/3
    [False, False, False],  # task 4: nominal failed, all perturbed fail
    [True, True, True],  # task 5: 3/3
]

dataset_metric = PromptRobustnessMetric(threshold=0.7)
dataset_score = dataset_metric.measure_robustness(nominal_passed, perturbed_passed)
print(f"Score: {dataset_score.value:.4f}  (threshold={dataset_score.threshold})")
print(f"Passed: {dataset_score.passed}")
print(f"Reason: {dataset_score.reason}")
print(f"Metadata: {dataset_score.metadata}")
