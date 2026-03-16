"""Minimal working example of harness-evals.

Run: python examples/basic_eval.py
"""

from harness_evals import EvalCase, Golden, assert_test, evaluate
from harness_evals.metrics import ContainsMetric, ExactMatchMetric, LatencyMetric
from harness_evals.sinks import StdoutSink

# 1. Simple exact match with typed operational fields
ec = EvalCase(
    input="What is the capital of France?",
    output="Paris",
    expected="Paris",
    latency_ms=450,
)

scores = evaluate(
    ec,
    metrics=[
        ExactMatchMetric(),
        ContainsMetric(),
        LatencyMetric(max_ms=2000, threshold=0.5),
    ],
    sinks=[StdoutSink()],
)

print(f"\nAll passed: {all(s.passed for s in scores)}")

# 2. assert_test raises on failure
print("\n--- assert_test example ---")
try:
    assert_test(
        EvalCase(input="q", output="wrong", expected="right"),
        metrics=[ExactMatchMetric()],
    )
except AssertionError as e:
    print(f"Caught expected error: {e}")

# 3. EvalCase.from_golden — bridge authored data to evaluation
print("\n--- from_golden example ---")
golden = Golden(input="What is 2+2?", expected="4")
ec = EvalCase.from_golden(golden, output="4", latency_ms=120)
scores = evaluate(ec, metrics=[ExactMatchMetric()], sinks=[StdoutSink()])
