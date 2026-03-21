"""CI baseline regression check — evaluate, compare to baseline, exit non-zero on regression.

Usage:
    # First run: save a baseline
    python examples/ci_baseline_regression.py --save baseline-v1

    # Subsequent runs: compare against the latest baseline
    python examples/ci_baseline_regression.py

    # Compare with a custom tolerance
    python examples/ci_baseline_regression.py --tolerance 0.10
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from harness_evals import EvalCase, Score, evaluate
from harness_evals.baseline import JsonBaselineStore, compare_to_baseline
from harness_evals.metrics import ContainsMetric, ExactMatchMetric, LatencyMetric

EVAL_CASES = [
    EvalCase(input="Capital of France?", output="Paris", expected="Paris", latency_ms=200),
    EvalCase(input="Capital of Germany?", output="Berlin", expected="Berlin", latency_ms=350),
    EvalCase(input="Capital of Japan?", output="Tokyo", expected="Tokyo", latency_ms=500),
]

METRICS = [
    ExactMatchMetric(),
    ContainsMetric(),
    LatencyMetric(max_ms=1000, threshold=0.5),
]


def run_evaluation() -> dict[str, list[Score]]:
    """Run metrics on all cases and group scores by metric name."""
    grouped: dict[str, list[Score]] = defaultdict(list)
    for case in EVAL_CASES:
        scores = evaluate(case, METRICS)
        for score in scores:
            grouped[score.name].append(score)
    return dict(grouped)


def main() -> None:
    parser = argparse.ArgumentParser(description="CI baseline regression check")
    parser.add_argument("--save", metavar="RUN_ID", help="Save current results as a baseline with this run ID")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Regression tolerance (default: 0.05)")
    parser.add_argument("--baseline-dir", default=".harness-evals/baselines", help="Baseline storage directory")
    args = parser.parse_args()

    store = JsonBaselineStore(baseline_dir=args.baseline_dir)
    current = run_evaluation()

    print(f"Evaluated {len(EVAL_CASES)} cases across {len(METRICS)} metrics")
    for name, scores in current.items():
        avg = sum(s.value for s in scores) / len(scores)
        print(f"  {name}: avg={avg:.3f}")

    if args.save:
        store.save(args.save, current)
        print(f"\nBaseline saved as '{args.save}'")
        return

    try:
        baseline = store.load()
    except FileNotFoundError:
        print("\nNo baseline found. Run with --save <run-id> first.")
        sys.exit(0)

    result = compare_to_baseline(current, baseline, tolerance=args.tolerance)
    print(f"\n{result.summary()}")

    if result.has_regressions:
        print("\nREGRESSION DETECTED — failing CI")
        sys.exit(1)
    else:
        print("\nNo regressions detected")


if __name__ == "__main__":
    main()
