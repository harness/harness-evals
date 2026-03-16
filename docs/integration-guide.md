# Integration Guide

## pytest

The simplest integration. Use `assert_test()` in any test file:

```python
from harness_evals import TestCase, assert_test
from harness_evals.metrics import ExactMatchMetric, JsonDiffMetric, LatencyMetric


def test_agent_response():
    tc = TestCase(
        input="Create a K8s deployment for nginx",
        actual_output=agent.run("Create a K8s deployment for nginx"),
        expected_output={"apiVersion": "apps/v1", "kind": "Deployment"},
        metadata={"latency_ms": measure_latency()},
    )
    assert_test(tc, metrics=[
        JsonDiffMetric(threshold=0.9),
        LatencyMetric(max_ms=5000, threshold=0.5),
    ])
```

`assert_test()` raises `AssertionError` with details on which metrics failed. pytest picks this up natively — no plugins needed.

### JUnit XML Output (Phase 3)

```bash
pytest test_agent.py --junitxml=eval-results.xml
```

Since `assert_test()` raises standard `AssertionError`, pytest's built-in `--junitxml` flag works out of the box. For programmatic JUnit output (outside pytest), use `JUnitSink`:

```python
from harness_evals import evaluate
from harness_evals.sinks import JUnitSink

scores = evaluate(tc, metrics=[...], sinks=[JUnitSink("eval-results.xml")])
```

### Parametrized Tests

Run the same metrics across a dataset:

```python
import json
import pytest

def load_cases():
    with open("datasets/regression.jsonl") as f:
        return [json.loads(line) for line in f]

@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c["input"][:50])
def test_regression(case):
    tc = TestCase(**case)
    assert_test(tc, metrics=[ExactMatchMetric(), LatencyMetric(max_ms=3000)])
```

## GitHub Actions

```yaml
name: Eval
on: [push, pull_request]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install harness-evals
      - run: pytest tests/evals/ --junitxml=eval-results.xml
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: eval-results
          path: eval-results.xml
```

## Harness CI

In a Harness CI pipeline, add a **Run** step:

```yaml
- step:
    type: Run
    name: Run Evals
    identifier: run_evals
    spec:
      shell: Sh
      command: |
        pip install harness-evals
        pytest tests/evals/ --junitxml=eval-results.xml
      reports:
        type: JUnit
        spec:
          paths:
            - eval-results.xml
```

The `reports.type: JUnit` block tells Harness CI to parse the XML and display results in the Tests tab.

### Quality Gate with Baseline Comparison (Phase 3)

```yaml
- step:
    type: Run
    name: Eval Quality Gate
    identifier: eval_quality_gate
    spec:
      shell: Sh
      command: |
        pip install harness-evals
        python scripts/eval_gate.py
```

Where `scripts/eval_gate.py`:

```python
import sys
from harness_evals import evaluate_dataset, load_dataset
from harness_evals.metrics import JsonDiffMetric, LatencyMetric
from harness_evals.sinks import JUnitSink
from harness_evals.baseline import JsonBaselineStore, compare_to_baseline

dataset = load_dataset("datasets/regression.jsonl")
scores = evaluate_dataset(dataset, metrics=[
    JsonDiffMetric(threshold=0.85),
    LatencyMetric(max_ms=5000, threshold=0.5),
], sinks=[JUnitSink("eval-results.xml")])

store = JsonBaselineStore()
result = compare_to_baseline(scores, store.load())

if result.regressions:
    for r in result.regressions:
        print(f"REGRESSION: {r.metric_name} {r.baseline_score:.2f} -> {r.current_score:.2f}")
    sys.exit(1)

store.save(run_id=os.environ.get("HARNESS_BUILD_ID", "local"), scores=scores)
```

## GitLab CI

```yaml
eval:
  image: python:3.12
  script:
    - pip install harness-evals
    - pytest tests/evals/ --junitxml=eval-results.xml
  artifacts:
    reports:
      junit: eval-results.xml
```

## Programmatic Usage (No Test Framework)

If you're not using pytest, use `evaluate()` directly:

```python
from harness_evals import TestCase, evaluate
from harness_evals.metrics import ExactMatchMetric
from harness_evals.sinks import StdoutSink, JsonSink

tc = TestCase(input="What is 2+2?", actual_output="4", expected_output="4")
scores = evaluate(tc, metrics=[ExactMatchMetric()], sinks=[
    StdoutSink(),
    JsonSink("results/scores.jsonl"),
])

# Check programmatically
if not all(s.success for s in scores):
    print("Some metrics failed!")
```

## Environment-Based Thresholds

Use environment variables or config files to vary thresholds:

```python
import os

threshold = float(os.environ.get("EVAL_THRESHOLD", "0.85"))

scores = evaluate(tc, metrics=[
    JsonDiffMetric(threshold=threshold),
])
```

Or load from a YAML config:

```yaml
# eval-config.yaml
metrics:
  json_diff:
    threshold: 0.85
  latency:
    max_ms: 5000
    threshold: 0.5
```

## Output Sinks

| Sink | Phase | Format | Use Case |
|------|-------|--------|----------|
| `StdoutSink` | 1 | Human-readable text | Local development |
| `JsonSink` | 1 | JSONL (one object per evaluation) | Programmatic processing |
| `JUnitSink` | 3 | JUnit XML | CI/CD integration |
| `CsvSink` | 3 | CSV (one row per metric) | Spreadsheet analysis |
