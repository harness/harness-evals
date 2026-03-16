# harness-evals

Open-source AI evaluation framework for LLM agents, prompts, and structured outputs.

Every metric produces a normalized `Score` (0.0–1.0). Pass/fail is determined by a configurable threshold. No magic, no hidden state.

## Install

```bash
pip install harness-evals
```

## Core Concepts

**`TestCase`** — what you're evaluating: an input, what the agent produced, and optionally what it should have produced.

**`BaseMetric`** — a scoring function. Takes a `TestCase`, returns a `Score`. Each metric is a single class with a `measure()` method.

**`Score`** — the result: a value between 0.0 and 1.0, a threshold, and a pass/fail boolean.

**`evaluate()`** — runs multiple metrics on a test case. Never raises — returns all scores including failures.

**`assert_test()`** — same as `evaluate()`, but raises `AssertionError` if any metric fails. Drop it into pytest.

## Usage

### Evaluate a response

```python
from harness_evals import TestCase, evaluate
from harness_evals.metrics import ExactMatchMetric, LatencyMetric

tc = TestCase(
    input="What is the capital of France?",
    actual_output="Paris",
    expected_output="Paris",
    metadata={"latency_ms": 320},
)

scores = evaluate(tc, metrics=[
    ExactMatchMetric(),
    LatencyMetric(max_ms=2000, threshold=0.5),
])

for s in scores:
    print(f"{'PASS' if s.success else 'FAIL'} {s.name}: {s.value:.2f}")
# PASS exact_match: 1.00
# PASS latency: 0.84
```

### Evaluate structured output (JSON/YAML)

```python
from harness_evals import TestCase, assert_test
from harness_evals.metrics import JsonDiffMetric, SchemaValidationMetric

tc = TestCase(
    input="Create a K8s deployment for nginx",
    actual_output={"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "nginx"}},
    expected_output={"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "nginx"}},
)

assert_test(tc, metrics=[
    JsonDiffMetric(threshold=0.9),        # structural similarity via DeepDiff
    SchemaValidationMetric(),              # validates against a JSON Schema
])
```

### Use with pytest

```python
def test_agent_accuracy():
    tc = TestCase(
        input="What is 2+2?",
        actual_output=agent.run("What is 2+2?"),
        expected_output="4",
    )
    assert_test(tc, metrics=[ExactMatchMetric()])
```

`assert_test()` raises `AssertionError` on failure — works natively with pytest, unittest, and any CI system.

### Measure reliability across multiple runs

```python
from harness_evals.metrics import OutcomeConsistencyMetric

runs = [
    TestCase(input="task", actual_output=agent.run("task"))
    for _ in range(5)
]

tc = TestCase(input="task", actual_output=runs[0].actual_output, runs=runs)
scores = evaluate(tc, metrics=[OutcomeConsistencyMetric(threshold=0.8)])
# Measures: do repeated runs produce the same output?
```

### Write results to a file

```python
from harness_evals.sinks import StdoutSink, JsonSink

scores = evaluate(tc, metrics=[...], sinks=[
    StdoutSink(),                          # human-readable to terminal
    JsonSink("results/scores.jsonl"),      # machine-readable JSONL
])
```

## Available Metrics

| Category | Metrics | What They Measure |
|----------|---------|------------------|
| **Deterministic** | ExactMatch, Contains, Regex, NumericDiff | Exact comparison against expected output |
| **Structural** | JsonDiff, SchemaValidation | Structural similarity and schema conformance for JSON/YAML |
| **Operational** | Latency, TokenCost, CostEfficiency, RetryCount | Performance and cost from metadata |
| **Reliability** | OutcomeConsistency, ResourceConsistency | Consistency across repeated runs |

## TestCase Fields

```python
TestCase(
    input="the prompt or task",                    # required
    actual_output="what the agent produced",       # required
    expected_output="ground truth",                # optional (not needed for LLM-judged metrics)
    context=["retrieved doc 1", "retrieved doc 2"],# optional (for RAG metrics)
    metadata={"latency_ms": 320, "token_usage": 150},  # optional (for operational metrics)
    tags={"env": "ci", "model": "gpt-4o"},         # optional (for filtering)
    runs=[...],                                    # optional (for reliability metrics)
)
```

## Extending

### Custom metric

```python
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.test_case import TestCase

class MyMetric(BaseMetric):
    def __init__(self, threshold: float = 0.8):
        super().__init__(name="my_metric", threshold=threshold)

    def measure(self, test_case: TestCase) -> Score:
        value = compute_something(test_case)  # return 0.0–1.0
        return Score(name=self.name, value=value, threshold=self.threshold,
                     success=value >= self.threshold)
```

### Custom sink

```python
from harness_evals.core.sink import BaseSink

class MySink(BaseSink):
    def write(self, scores, test_case):
        for s in scores:
            send_to_my_system(s.name, s.value, s.success)
```

## Documentation

- [Architecture](docs/architecture.md) — system design, data flow, extension points
- [Metrics Guide](docs/metrics-guide.md) — how to write a new metric, templates for every category
- [Integration Guide](docs/integration-guide.md) — pytest, GitHub Actions, Harness CI, GitLab CI
- [Contributing](docs/CONTRIBUTING.md) — development workflow, code style, PR process
- [Architecture Decision Records](docs/adr/) — why we made key design choices
- [Changelog](CHANGELOG.md) — version history

## Development

```bash
git clone git@github.com:sunilgattupalle/harness-evals.git
cd harness-evals
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
ruff check src/ tests/   # lint
pytest tests/ -v         # test
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
