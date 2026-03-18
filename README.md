# harness-evals

Open-source AI evaluation framework for LLM agents, prompts, and structured outputs.

Every metric produces a normalized `Score` (0.0–1.0). Pass/fail is determined by a configurable threshold. No magic, no hidden state.

## Install

```bash
pip install harness-evals
```

## Core Concepts

**`Golden`** — what you author: an input, the expected output, and optional context. Lives in your dataset files.

**`EvalCase`** — what metrics receive: a Golden enriched with the agent's actual output and runtime metadata (latency, tokens, cost).

**`BaseMetric`** — a scoring function. Takes an `EvalCase`, returns a `Score`. Each metric is a single class with a `measure()` method. Specialized base classes: `ReliabilityMetric` for multi-run metrics, `SafetyMetric` for safety metrics (reported separately, never averaged).

**`Score`** — the result: a value between 0.0 and 1.0, a threshold, and an auto-computed `passed` boolean.

**`evaluate()`** — runs multiple metrics on an eval case. Never raises — returns all scores including failures.

**`assert_test()`** — same as `evaluate()`, but raises `AssertionError` if any metric fails. Drop it into pytest.

### Data Flow

```
Golden (authored) + agent output → EvalCase (evaluated) → Score (result)
```

## Usage

### Evaluate a response

```python
from harness_evals import EvalCase, evaluate
from harness_evals.metrics import ExactMatchMetric, LatencyMetric

ec = EvalCase(
    input="What is the capital of France?",
    output="Paris",
    expected="Paris",
    latency_ms=320,
)

scores = evaluate(ec, metrics=[
    ExactMatchMetric(),
    LatencyMetric(max_ms=2000, threshold=0.5),
])

for s in scores:
    print(f"{'PASS' if s.passed else 'FAIL'} {s.name}: {s.value:.2f}")
# PASS exact_match: 1.00
# PASS latency: 0.84
```

### Evaluate structured output (JSON/YAML)

```python
from harness_evals import EvalCase, assert_test
from harness_evals.metrics import JsonDiffMetric, SchemaValidationMetric

expected = {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "nginx"}}
schema = {"type": "object", "required": ["apiVersion", "kind"], "properties": {
    "apiVersion": {"type": "string"}, "kind": {"type": "string"}
}}

ec = EvalCase(
    input="Create a K8s deployment for nginx",
    output={"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "nginx"}},
    expected=expected,
)

assert_test(ec, metrics=[
    JsonDiffMetric(threshold=0.9),
    SchemaValidationMetric(schema=schema),
])
```

### Use with pytest

```python
def test_agent_accuracy():
    ec = EvalCase(
        input="What is 2+2?",
        output=agent.run("What is 2+2?"),
        expected="4",
    )
    assert_test(ec, metrics=[ExactMatchMetric()])
```

`assert_test()` raises `AssertionError` on failure — works natively with pytest, unittest, and any CI system.

### Evaluate a dataset with an agent

```python
import asyncio
from harness_evals import Golden, EvalCase, evaluate_dataset
from harness_evals.metrics import ExactMatchMetric

goldens = [
    Golden(input="What is 2+2?", expected="4"),
    Golden(input="Capital of France?", expected="Paris"),
]

async def run_agent(golden: Golden) -> EvalCase:
    result = await agent.arun(golden.input)
    return EvalCase.from_golden(golden, output=result)

results = asyncio.run(evaluate_dataset(goldens, run_agent, metrics=[ExactMatchMetric()]))
```

### Measure reliability across multiple runs

```python
from harness_evals import EvalCase, evaluate
from harness_evals.metrics import OutcomeConsistencyMetric

runs = [
    EvalCase(input="task", output=agent.run("task"))
    for _ in range(5)
]

ec = EvalCase(input="task", output=runs[0].output, runs=runs)
scores = evaluate(ec, metrics=[OutcomeConsistencyMetric(threshold=0.8)])
```

### Write results to a file

```python
from harness_evals.sinks import StdoutSink, JsonSink

scores = evaluate(ec, metrics=[...], sinks=[
    StdoutSink(),
    JsonSink("results/scores.jsonl"),
])
```

## Available Metrics

| Category | Metrics | What They Measure |
|----------|---------|------------------|
| **Deterministic** | ExactMatch, Contains, Regex, NumericDiff | Exact comparison against expected output |
| **Structural** | JsonDiff, SchemaValidation | Structural similarity and schema conformance for JSON/YAML |
| **Operational** | Latency, TokenCost, CostEfficiency, RetryCount | Performance and cost from typed fields |
| **Reliability** | OutcomeConsistency, ResourceConsistency | Consistency across repeated runs |
| **Predictability** | Calibration, Discrimination | Expected calibration error and AUC-ROC over confidence scores |
| **LLM-Judged** | GEval, RubricJudge | LLM scores output against criteria or rubric (requires `[llm]`) |
| **RAG** | Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall | Retrieval-augmented generation quality (requires `[llm]`) |
| **Safety** | PIIMetric, ToxicityMetric, PromptInjectionMetric, HallucinationMetric | PII leaks, toxic content, prompt injection, hallucination (reported separately, never averaged) |
| **Agent** | ToolCorrectnessMetric, TaskCompletionMetric | Tool call sequence correctness and LLM-judged task completion |

## EvalCase Fields

```python
EvalCase(
    input="the prompt or task",                    # required
    output="what the agent produced",              # required
    expected="ground truth",                       # optional (not needed for LLM-judged metrics)
    context=["retrieved doc 1", "retrieved doc 2"],# optional (for RAG metrics)
    latency_ms=320,                                # optional (typed, for LatencyMetric)
    token_count=150,                               # optional (typed, for TokenCostMetric)
    cost_usd=0.003,                                # optional (typed, for CostEfficiencyMetric)
    retry_count=0,                                 # optional (typed, for RetryCountMetric)
    confidence=0.95,                               # optional (typed)
    tags={"env": "ci", "model": "gpt-4o"},         # optional (for filtering)
    metadata={"custom_key": "value"},              # optional (extensible)
    runs=[...],                                    # optional (for reliability metrics)
)
```

## Extending

### Custom metric

```python
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.eval_case import EvalCase

class MyMetric(BaseMetric):
    def __init__(self, threshold: float = 0.8):
        super().__init__(name="my_metric", threshold=threshold)

    def measure(self, eval_case: EvalCase) -> Score:
        value = compute_something(eval_case)  # return 0.0–1.0
        return Score(name=self.name, value=value, threshold=self.threshold)
```

### Custom sink

```python
from harness_evals.core.sink import BaseSink

class MySink(BaseSink):
    def write(self, scores, eval_case):
        for s in scores:
            send_to_my_system(s.name, s.value, s.passed)
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

## References

- Rabanser, Kapoor, Kirgis, Liu, Utpala, Narayanan. ["Towards a Science of AI Agent Reliability"](https://arxiv.org/abs/2602.16666v2). Princeton, 2026. — Defines the 12 reliability metrics across 4 dimensions (consistency, robustness, predictability, safety) that inform this framework's reliability metric design.
- [DeepEval](https://github.com/confident-ai/deepeval) — LLM evaluation framework. Influenced the `measure()` / `a_measure()` metric interface pattern.
- [RAGAS](https://docs.ragas.io) — RAG evaluation toolkit. Informed the RAG metric decomposition (faithfulness, context precision/recall).
- [promptfoo](https://www.promptfoo.dev/docs/intro/) — CLI-first eval tool. Validated the CI/CD-native approach (JUnit output, exit codes, baseline regression).

## License

Apache 2.0 — see [LICENSE](LICENSE).
