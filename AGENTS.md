# AGENTS.md - harness-evals

## Project Overview

**harness-evals** is an open-source AI evaluation framework for LLM agents, prompts, and structured outputs. It provides a `pip install`-able scoring engine with ~37 metrics across deterministic, structural, operational, reliability, RAG, safety, agent, and conversational categories.

**Core principle**: An eval always produces a `Score`. Every metric is a single class with a `measure()` method.

**Data flow**: `Golden` (authored) + agent output -> `EvalCase` (evaluated) -> `Score` (result)

**Language**: Python 3.10+
**License**: Apache 2.0
**Package name**: `harness-evals`
**Import name**: `harness_evals`

## Build System

```bash
pip install -e "."              # Core only (Phase 1 metrics, no LLM key needed)
pip install -e ".[llm]"         # + OpenAI, Anthropic for LLM-judged metrics
pip install -e ".[dev]"         # + pytest, ruff, pre-commit
pip install -e ".[all,dev]"     # Everything
```

**Build tool**: setuptools via `pyproject.toml`
**No compiled extensions** — pure Python.

## Testing

```bash
pytest tests/ -v                              # All tests
pytest tests/ -v -m unit                      # Unit tests only
pytest tests/metrics/ -v                      # Specific directory
pytest tests/test_core.py -v                  # Specific file
pytest tests/test_core.py::test_evaluate -v   # Specific function
pytest tests/ --cov=harness_evals --cov-report=html  # With coverage
```

- Mark tests: `@pytest.mark.unit`, `@pytest.mark.integration`
- Test data: `tests/data/`

**ALWAYS run `pytest tests/ -v` before committing.**

## Linting & Formatting

```bash
ruff check src/ tests/           # Lint check
ruff format --check src/ tests/  # Format check
ruff format src/ tests/          # Auto-format
ruff check --fix src/ tests/     # Auto-fix lint issues
```

Ruff handles both formatting and linting (replaces black + flake8 + isort).

## Git Workflow

- **Branch naming**: `feat/short-description` or `fix/short-description`
- **Commit format**: `type: description` where type is `feat`, `fix`, `chore`, `refactor`, `test`, `docs`
- **Default branch**: `main`

## DOs

- Use type hints on all function signatures
- Follow existing patterns — look at any metric file as a template
- Use `@dataclass` for structured data (Golden, EvalCase, Score)
- Keep metrics as single-file, single-class modules
- Write a test file for every new metric
- Use async/await for I/O (LLM calls, HTTP) — override `a_measure()` for async metrics
- Run `ruff check` and `pytest` before committing

## DON'Ts

- Never force push to main
- Never commit secrets or `.env` files
- Never add heavy dependencies (torch, transformers) to core — use optional extras
- Never modify `Golden`, `EvalCase`, or `Score` fields without updating PLAN.md
- Never average safety scores into an overall score — report them separately
- Don't use `print()` — use the sink system for output

## Project Structure

```
harness-evals/
├── pyproject.toml                   # Package config, dependencies, tool settings
├── README.md                        # User-facing documentation
├── AGENTS.md                        # This file
├── PLAN.md                          # Full vision spec with all phases
├── LICENSE                          # Apache 2.0
├── .gitignore
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
│
├── src/harness_evals/
│   ├── __init__.py                  # Public API: Golden, EvalCase, Score, evaluate, assert_test, etc.
│   ├── py.typed                     # PEP 561 marker
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── golden.py                # Golden dataclass (authored data)
│   │   ├── eval_case.py             # EvalCase dataclass (what metrics receive)
│   │   ├── score.py                 # Score dataclass (passed auto-computed)
│   │   ├── metric.py                # BaseMetric, ReliabilityMetric ABCs
│   │   ├── sink.py                  # BaseSink ABC
│   │   └── runner.py                # evaluate(), assert_test(), evaluate_cases(), evaluate_dataset()
│   │
│   ├── metrics/
│   │   ├── __init__.py              # Re-exports all metrics
│   │   ├── deterministic/           # ExactMatch, Contains, Regex, NumericDiff
│   │   ├── structural/              # JsonDiff, SchemaValidation
│   │   ├── operational/             # Latency, TokenCost, CostEfficiency, RetryCount
│   │   └── reliability/             # OutcomeConsistency, ResourceConsistency
│   │
│   └── sinks/
│       ├── __init__.py
│       ├── stdout.py                # StdoutSink
│       └── json_sink.py             # JsonSink
│
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_core.py                 # Golden, EvalCase, Score, evaluate, assert_test, etc.
│   └── metrics/                     # One test file per metric category
│
└── examples/
    └── basic_eval.py                # Minimal working example
```

## How to Add a New Metric

This is the most common task an AI agent will do. Follow these steps:

1. **Pick the category** — deterministic, structural, operational, reliability, etc.
2. **Create the file** — `src/harness_evals/metrics/<category>/<metric_name>.py`
3. **Implement the class** — extend `BaseMetric` (or `ReliabilityMetric` for multi-run):

```python
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.eval_case import EvalCase


class MyMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0, **kwargs):
        super().__init__(name="my_metric", threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        value = ...  # compute 0.0–1.0
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
        )
```

4. **Export it** — add to `src/harness_evals/metrics/<category>/__init__.py` and `src/harness_evals/metrics/__init__.py`
5. **Write the test** — `tests/metrics/test_<metric_name>.py`:

```python
import pytest
from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.<category>.<metric_name> import MyMetric


@pytest.mark.unit
def test_my_metric_perfect():
    ec = EvalCase(input="x", output="y", expected="y")
    score = MyMetric(threshold=0.8).measure(ec)
    assert score.passed
    assert score.value == 1.0


@pytest.mark.unit
def test_my_metric_failure():
    ec = EvalCase(input="x", output="wrong", expected="y")
    score = MyMetric(threshold=0.8).measure(ec)
    assert not score.passed
```

6. **Run tests** — `pytest tests/ -v`

## Core Types Reference

### Golden (authored data)

```python
@dataclass
class Golden:
    input: str | dict | list
    expected: str | dict | list | None = None
    context: list[str] | None = None
    metadata: dict[str, Any] | None = None
    tags: dict[str, str] | None = None
```

### EvalCase (what metrics receive)

```python
@dataclass
class EvalCase:
    input: str | dict | list
    output: str | dict | list
    expected: str | dict | list | None = None
    context: list[str] | None = None
    latency_ms: float | None = None         # typed operational fields
    token_count: int | None = None
    cost_usd: float | None = None
    retry_count: int | None = None
    confidence: float | None = None
    tags: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None  # extensible for custom keys
    runs: list["EvalCase"] | None = None    # K runs for reliability metrics
```

### Score

```python
@dataclass
class Score:
    name: str
    value: float           # 0.0 to 1.0
    threshold: float       # pass/fail threshold
    passed: bool           # auto-computed: value >= threshold (not in constructor)
    reason: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime   # auto-set to UTC now
```

### BaseMetric

```python
class BaseMetric(ABC):
    name: str
    threshold: float

    @abstractmethod
    def measure(self, eval_case: EvalCase) -> Score: ...

    async def a_measure(self, eval_case: EvalCase) -> Score:
        """Async variant. Override for I/O-bound metrics. Default calls measure()."""
        return self.measure(eval_case)
```

### ReliabilityMetric (for multi-run)

```python
class ReliabilityMetric(BaseMetric):
    k: int  # number of runs expected

    @abstractmethod
    def measure_runs(self, eval_case: EvalCase) -> Score:
        """Evaluate across eval_case.runs. Called by measure()."""

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.runs:
            return self.measure_runs(eval_case)
        return Score(name=self.name, value=0.0, threshold=self.threshold,
                     reason="No runs provided")
```

## Phased Implementation

See `PLAN.md` for the full vision with 6 phases and ~37 metrics. Phase 1 (this skeleton) covers core framework + 12 metrics. Each subsequent phase adds metrics, capabilities, and directory structure as described in `PLAN.md`.

## Dependencies

**Core (Phase 1)**: `deepdiff>=7.0`, `jsonschema>=4.0` — two dependencies total.
**LLM (Phase 2+)**: `openai>=1.0`, `anthropic>=0.30` — optional.
**Dev**: `pytest>=8.0`, `ruff>=0.4`, `pytest-cov`, `pytest-asyncio`, `pre-commit`.
