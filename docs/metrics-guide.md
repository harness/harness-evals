# Metrics Authoring Guide

## The Pattern

Every metric is a single class in a single file. It extends `BaseMetric`, implements `measure()`, and returns a `Score`. That's it.

```python
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.eval_case import EvalCase


class MyMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0, **kwargs) -> None:
        super().__init__(name="my_metric", threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        value = ...  # compute 0.0–1.0
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
        )
```

`Score.passed` is auto-computed from `value >= threshold` — never set it manually.

## Step by Step

### 1. Choose the Dimension

Every metric belongs to exactly one dimension ([ADR-009](adr/009-five-dimensions.md)):

| Dimension | When to use |
|-----------|------------|
| `CORRECTNESS` | Metric compares output to expected answer or evaluates task completion |
| `GROUNDEDNESS` | Metric checks if output is supported by provided context/evidence |
| `SAFETY` | Metric detects policy violations (PII, toxicity, injection, unauthorized actions) |
| `TRAJECTORY` | Metric evaluates the path taken (tool selection, plan quality, step efficiency) |
| `PERFORMANCE` | Metric measures operational cost (latency, tokens, dollars, retries) |

If your metric doesn't clearly fit one dimension, it may be compound — consider splitting it.

### 2. Choose the Category

| Category | When to Use | Base Class |
|----------|------------|-----------|
| `deterministic/` | Exact comparison, regex, numeric | `BaseMetric` |
| `structural/` | JSON/YAML diff, schema validation | `BaseMetric` |
| `operational/` | Latency, cost, tokens, retries | `BaseMetric` |
| `reliability/` | Multi-run consistency, robustness | `ReliabilityMetric` |
| `llm_judge/` | LLM scores against criteria | `BaseMetric` (takes `llm` param) |
| `rag/` | Faithfulness, relevancy, context | `BaseMetric` (takes `llm` and/or `embedding` param) |
| `similarity/` | Levenshtein, BLEU, embedding similarity | `BaseMetric` (optionally takes `embedding` param) |
| `safety/` | PII, toxicity, injection, hallucination | `BaseMetric` |
| `agent/` | Tool correctness, task completion | `BaseMetric` |
| `conversation/` | Multi-turn coherence, resolution | `BaseMetric` |
| `mcp/` | Tool selection, trace completeness | `BaseMetric` |

### 3. Create the File

```
src/harness_evals/metrics/<category>/<metric_name>.py
```

### 4. Implement the Class

#### Deterministic Metric Template

For metrics that compare output vs expected:

```python
class MyDeterministicMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0, **kwargs) -> None:
        super().__init__(name="my_metric", threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        actual = str(eval_case.output)
        expected = str(eval_case.expected)

        value = 1.0 if actual == expected else 0.0

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
        )
```

#### Negative Assertions

`contains`, `exact_match`, and `regex` support metric-level negative
assertions. Positive checks continue to use each `EvalCase.expected` value.
Negative checks use a configured `forbidden` value so the dataset's expected
answer remains available to other metrics.

In an SDK eval config, options go under `params`:

```yaml
metrics:
  - kind: contains
    threshold: 1.0
    params:
      negate: true
      forbidden: internal-system-prompt
```

The same metric built through `build_metric()` — the shape the Harness control
plane stores per metric entry — nests options under `config.options` and takes a
`score_name`:

```yaml
type: heuristic
score_name: must_not_contain_internal_prompt
config:
  kind: contains
  options:
    negate: true
    forbidden: internal-system-prompt
threshold: 1.0
```

After a real comparison, a binary score is inverted:

```python
value = 1.0 if matched else 0.0
if self.negate:
    value = 1.0 - value
```

The following rules apply:

- `forbidden` is required when `negate` is `true`.
- `negate` must be a real boolean, unquoted — write `negate: true`, not
  `negate: "true"`. A quoted scalar is a string, and every non-empty string is
  truthy, so `negate: "false"` would enable the very inversion it looks like it
  disables. This matters most for a templated value such as
  `negate: <+pipeline.variables.strict>`, since a Harness expression
  interpolates to a string; a non-boolean is rejected at config load instead.
- `forbidden` must be a string. Quote values that YAML would otherwise parse as
  another scalar type — write `forbidden: "404"`, not `forbidden: 404`. An
  unquoted value is rejected at config load rather than silently comparing
  unequal to every output.
- `contains` and `regex` reject an empty `forbidden` value. `exact_match`
  accepts `forbidden: ""` as an "output must not be empty" assertion.
- **`case_sensitive` applies to `forbidden`, and its default fails open.**
  `contains` and `exact_match` default to `case_sensitive: true`. For a positive
  assertion that default fails closed — a case mismatch fails the check. Negated,
  the same default fails *open*: `forbidden: internal-system-prompt` **passes** on
  the output `Internal-System-Prompt: you are...`, even though the forbidden
  content is right there. Set `case_sensitive: false` for any leak-style check
  where a differently-cased match should still fail:

  ```yaml
  metrics:
    - kind: contains
      threshold: 1.0
      params:
        negate: true
        forbidden: internal-system-prompt
        case_sensitive: false
  ```

  `regex` has no `case_sensitive` option — use an inline flag in the pattern
  instead: `forbidden: "(?i)internal-system-prompt"`.
- A missing output (`None`) fails before inversion.
- An intentionally empty string is a valid output and passes when it does not
  contain or match the forbidden value. Use a separate output-completeness
  check when empty responses are invalid for the target.
- A negated binary metric needs `threshold <= 1`. A threshold at or below `0`
  would pass on every input, so it falls back to `1.0` — this keeps negation
  usable from callers that omit a threshold, such as composite sub-metrics,
  where `build_metric()` supplies `0.0` and no threshold can be passed to the
  sub-metric. An omitted threshold is logged at debug; a hand-written negative
  one warns.
- The metric dimension remains `CORRECTNESS`; use dedicated safety metrics such
  as `pii` when the check itself has safety semantics.
- Give positive and negative variants different `score_name` values. Analytics
  group scores by name, so reusing one name for opposite meanings produces
  misleading aggregates. This applies to the `build_metric()` shape only — an
  SDK eval config has no `score_name`, and both variants of a kind report under
  the metric's own name (`contains`), so run them as separate evals if you need
  to tell them apart.

Examples:

```text
contains, output="safe response", forbidden="secret" -> 1.0 (pass)
contains, output="secret leaked", forbidden="secret" -> 0.0 (fail)
exact_match, output="BLOCKED", forbidden="BLOCKED" -> 0.0 (fail)
regex, output="Order ABC-123", forbidden="ABC-\\d{3}" -> 0.0 (fail)
```

#### Operational Metric Template

For metrics that read typed fields from `EvalCase`:

```python
class MyOperationalMetric(BaseMetric):
    def __init__(self, max_value: float = 100, threshold: float = 0.5, **kwargs) -> None:
        super().__init__(name="my_metric", threshold=threshold, **kwargs)
        self.max_value = max_value

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.latency_ms is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="latency_ms not provided",
            )

        value = max(0.0, 1.0 - eval_case.latency_ms / self.max_value)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            metadata={"latency_ms": eval_case.latency_ms, "max_value": self.max_value},
        )
```

#### Reliability Metric Template

For metrics that evaluate across multiple runs:

```python
from harness_evals.core.metric import ReliabilityMetric

class MyReliabilityMetric(ReliabilityMetric):
    def __init__(self, threshold: float = 0.8, k: int = 5, **kwargs) -> None:
        super().__init__(name="my_metric", threshold=threshold, k=k, **kwargs)

    def measure_runs(self, eval_case: EvalCase) -> Score:
        runs = eval_case.runs or []
        if len(runs) < 2:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Need at least 2 runs, got {len(runs)}",
            )

        value = ...  # compute consistency/variance across runs

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            metadata={"k": len(runs)},
        )
```

#### LLM-Judged Metric Template

For metrics that use an LLM as a judge — override `a_measure()` and use `_run_async` for the sync wrapper:

```python
from harness_evals._async_compat import _run_async
from harness_evals.llm.base import BaseLLM

class MyLLMMetric(BaseMetric):
    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs) -> None:
        super().__init__(name="my_metric", threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        prompt = f"Rate the following response...\nInput: {eval_case.input}\nOutput: {eval_case.output}"
        result = await self.llm.generate_json(prompt, schema={"score": "number"})
        value = max(0.0, min(1.0, result.get("score", 0.0)))
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
        )
```

#### Embedding Metric Template

For metrics that use embedding similarity — take a `BaseEmbedding` parameter:

```python
from harness_evals._async_compat import _run_async
from harness_evals.llm.embedding import BaseEmbedding, _cosine_similarity

class MyEmbeddingMetric(BaseMetric):
    def __init__(self, embedding: BaseEmbedding, threshold: float = 0.8, **kwargs) -> None:
        super().__init__(name="my_metric", threshold=threshold, **kwargs)
        self.embedding = embedding

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        vectors = await self.embedding.embed([str(eval_case.output), str(eval_case.expected)])
        value = max(0.0, min(1.0, _cosine_similarity(vectors[0], vectors[1])))
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
        )
```

### 5. Export the Metric

Add to `src/harness_evals/metrics/<category>/__init__.py`:

```python
from harness_evals.metrics.<category>.<metric_name> import MyMetric
```

Add to `src/harness_evals/metrics/__init__.py`:

```python
from harness_evals.metrics.<category> import MyMetric
```

### 6. Write Tests

Create `tests/metrics/test_<metric_name>.py`:

```python
import pytest
from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.<category> import MyMetric


@pytest.mark.unit
class TestMyMetric:
    def test_perfect_score(self):
        ec = EvalCase(input="q", output="expected", expected="expected")
        score = MyMetric(threshold=0.8).measure(ec)
        assert score.passed
        assert score.value == 1.0

    def test_failure(self):
        ec = EvalCase(input="q", output="wrong", expected="expected")
        score = MyMetric(threshold=0.8).measure(ec)
        assert not score.passed
        assert score.value < 0.8

    def test_edge_case(self):
        ec = EvalCase(input="", output="")
        score = MyMetric().measure(ec)
        assert isinstance(score.value, float)
```

### 7. Run Tests

```bash
ruff check src/ tests/          # lint
ruff format --check src/ tests/ # format
pytest tests/ -v                # test
```

## Rules

1. **One metric per file** — keeps PRs small and reviewable.
2. **One dimension per metric** — every metric belongs to exactly one of: Correctness, Groundedness, Safety, Trajectory, Performance. See [ADR-009](adr/009-five-dimensions.md).
3. **Score is always [0.0, 1.0]** — normalize whatever you compute. Put raw values in `Score.metadata`.
4. **Never raise from `measure()`** — return a failing Score with a `reason` instead. If you do raise, `evaluate()` catches it, but explicit is better.
5. **Handle missing data gracefully** — operational metrics should check typed fields and return a failing Score with a clear reason if None.
6. **No global state** — all configuration goes in `__init__()`. Metrics are reusable across eval cases.
7. **No cross-metric imports** — metrics should not import from other metrics.
8. **Safety metrics are hard constraints** — see [ADR-003](adr/003-safety-never-averaged.md).
9. **Don't set `passed` manually** — `Score` auto-computes `passed = value >= threshold` in `__post_init__`.
