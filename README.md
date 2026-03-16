# harness-evals

Open-source AI evaluation framework for LLM agents, prompts, and pipelines.

**Core principle**: Every metric produces a normalized `Score` (0.0–1.0). Pass/fail is determined by a configurable threshold. No magic, no hidden state.

## Quick Start

```bash
pip install harness-evals
```

```python
from harness_evals import TestCase, evaluate
from harness_evals.metrics import ExactMatchMetric, LatencyMetric

tc = TestCase(
    input="What is 2+2?",
    actual_output="4",
    expected_output="4",
    metadata={"latency_ms": 320},
)

scores = evaluate(tc, metrics=[
    ExactMatchMetric(),
    LatencyMetric(max_ms=2000, threshold=0.5),
])

for s in scores:
    print(f"{'PASS' if s.success else 'FAIL'} {s.name}: {s.value:.2f}")
```

## Use with pytest

```python
from harness_evals import TestCase, assert_test
from harness_evals.metrics import ExactMatchMetric, JsonDiffMetric

def test_pipeline_output():
    tc = TestCase(
        input="Create a deployment",
        actual_output={"apiVersion": "apps/v1", "kind": "Deployment"},
        expected_output={"apiVersion": "apps/v1", "kind": "Deployment"},
    )
    assert_test(tc, metrics=[JsonDiffMetric(threshold=0.9)])
```

`assert_test()` raises `AssertionError` on failure — works natively with pytest, unittest, and CI pipelines.

## Metrics (Phase 1 — no LLM key needed)

| Category | Metrics |
|----------|---------|
| **Deterministic** | ExactMatch, Contains, Regex, NumericDiff |
| **Structural** | JsonDiff, SchemaValidation |
| **Operational** | Latency, TokenCost, CostEfficiency, RetryCount |
| **Reliability** | OutcomeConsistency, ResourceConsistency |

See [PLAN.md](PLAN.md) for the full 6-phase roadmap with ~37 metrics including LLM-judged, RAG, safety, agent, and conversation metrics.

## Output Sinks

```python
from harness_evals.sinks import StdoutSink, JsonSink

scores = evaluate(tc, metrics=[...], sinks=[
    StdoutSink(),
    JsonSink("results/scores.jsonl"),
])
```

## Multi-Run Reliability

```python
from harness_evals import TestCase
from harness_evals.metrics import OutcomeConsistencyMetric

runs = [TestCase(input="task", actual_output=agent.run("task")) for _ in range(5)]
tc = TestCase(input="task", actual_output=runs[0].actual_output, runs=runs)

scores = evaluate(tc, metrics=[OutcomeConsistencyMetric(threshold=0.8)])
```

## Development

```bash
pip install -e ".[all,dev]"
make check   # lint + test
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
