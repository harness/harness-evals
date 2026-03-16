"""Minimal working example of harness-evals.

Run: python examples/basic_eval.py
"""

from harness_evals import TestCase, assert_test, evaluate
from harness_evals.metrics import ContainsMetric, ExactMatchMetric, LatencyMetric
from harness_evals.sinks import StdoutSink

# 1. Simple exact match
tc = TestCase(
    input="What is the capital of France?",
    actual_output="Paris",
    expected_output="Paris",
    metadata={"latency_ms": 450},
)

scores = evaluate(
    tc,
    metrics=[
        ExactMatchMetric(),
        ContainsMetric(),
        LatencyMetric(max_ms=2000, threshold=0.5),
    ],
    sinks=[StdoutSink()],
)

print(f"\nAll passed: {all(s.success for s in scores)}")

# 2. assert_test raises on failure
print("\n--- assert_test example ---")
try:
    assert_test(
        TestCase(input="q", actual_output="wrong", expected_output="right"),
        metrics=[ExactMatchMetric()],
    )
except AssertionError as e:
    print(f"Caught expected error: {e}")
