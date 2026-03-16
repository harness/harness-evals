# AGENTS.md - harness-evals

## Project Overview

**harness-evals** is an open-source AI evaluation framework for LLM agents, prompts, and pipelines. It provides a `pip install`-able scoring engine with ~37 metrics across deterministic, structural, operational, reliability, RAG, safety, agent, and conversational categories.

**Core principle**: An eval always produces a `Score`. Every metric is a single class with a `measure()` method.

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
make test                       # Run all tests
make test-unit                  # Unit tests only
pytest tests/ -v                # Direct pytest
pytest tests/metrics/ -v        # Test a specific directory
pytest tests/test_core.py -v    # Test a specific file
pytest tests/test_core.py::test_evaluate -v  # Test a specific function
```

- Mark tests: `@pytest.mark.unit`, `@pytest.mark.integration`
- Test data: `tests/data/`
- Coverage: `pytest tests/ --cov=harness_evals --cov-report=html`

**ALWAYS run `make test` before committing.**

## Linting & Formatting

```bash
make format                     # Auto-format with ruff
make lint                       # Lint check with ruff
make check                      # lint + test
```

Ruff handles both formatting and linting (replaces black + flake8 + isort).

## Git Workflow

- **Branch naming**: `feat/short-description` or `fix/short-description`
- **Commit format**: `type: description` where type is `feat`, `fix`, `chore`, `refactor`, `test`, `docs`
- **Default branch**: `main`

## DOs

- Use type hints on all function signatures
- Follow existing patterns — look at any metric file as a template
- Use `@dataclass` for structured data (TestCase, Score)
- Keep metrics as single-file, single-class modules
- Write a test file for every new metric
- Use async/await for I/O (LLM calls, HTTP)
- Run `make check` before committing

## DON'Ts

- Never force push to main
- Never commit secrets or `.env` files
- Never add heavy dependencies (torch, transformers) to core — use optional extras
- Never modify `TestCase` or `Score` fields without updating PLAN.md
- Never average safety scores into an overall score — report them separately
- Don't use `print()` — use the sink system for output

## Project Structure

```
harness-evals/
├── pyproject.toml                   # Package config, dependencies, tool settings
├── Makefile                         # format, lint, test, check
├── README.md                        # User-facing documentation
├── AGENTS.md                        # This file
├── PLAN.md                          # Full vision spec with all phases
├── LICENSE                          # Apache 2.0
├── .gitignore
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
│
├── src/harness_evals/
│   ├── __init__.py                  # Public API: TestCase, Score, evaluate, assert_test
│   ├── py.typed                     # PEP 561 marker
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── test_case.py             # TestCase dataclass
│   │   ├── score.py                 # Score dataclass
│   │   ├── metric.py                # BaseMetric, ReliabilityMetric ABCs
│   │   ├── sink.py                  # BaseSink ABC
│   │   └── runner.py                # evaluate(), evaluate_dataset(), assert_test()
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
│   ├── test_core.py                 # TestCase, Score, evaluate, assert_test
│   └── metrics/                     # One test file per metric
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
from harness_evals.core.test_case import TestCase


class MyMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0, **kwargs):
        super().__init__(name="my_metric", threshold=threshold, **kwargs)

    def measure(self, test_case: TestCase) -> Score:
        # Compute value between 0.0 and 1.0
        value = ...
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            success=value >= self.threshold,
        )
```

4. **Export it** — add to `src/harness_evals/metrics/<category>/__init__.py` and `src/harness_evals/metrics/__init__.py`
5. **Write the test** — `tests/metrics/test_<metric_name>.py`:

```python
import pytest
from harness_evals.core.test_case import TestCase
from harness_evals.metrics.<category>.<metric_name> import MyMetric


@pytest.mark.unit
def test_my_metric_perfect():
    tc = TestCase(input="x", actual_output="y", expected_output="y")
    score = MyMetric(threshold=0.8).measure(tc)
    assert score.success
    assert score.value == 1.0


@pytest.mark.unit
def test_my_metric_failure():
    tc = TestCase(input="x", actual_output="wrong", expected_output="y")
    score = MyMetric(threshold=0.8).measure(tc)
    assert not score.success
```

6. **Run tests** — `make test`

## Core Types Reference

### TestCase

```python
@dataclass
class TestCase:
    input: str
    actual_output: str | dict | list
    expected_output: str | dict | list | None = None
    context: list[str] | None = None
    metadata: dict[str, Any] | None = None  # latency_ms, token_usage, cost_usd, confidence
    tags: dict[str, str] | None = None      # env, model, version
    runs: list["TestCase"] | None = None    # K runs for reliability metrics
```

### Score

```python
@dataclass
class Score:
    name: str
    value: float           # 0.0 to 1.0
    threshold: float       # pass/fail threshold
    success: bool          # value >= threshold
    reason: str | None = None
    metadata: dict[str, Any] | None = None
```

### BaseMetric

```python
class BaseMetric(ABC):
    name: str
    threshold: float

    @abstractmethod
    def measure(self, test_case: TestCase) -> Score: ...
```

### ReliabilityMetric (for multi-run)

```python
class ReliabilityMetric(BaseMetric):
    k: int  # number of runs expected

    @abstractmethod
    def measure_runs(self, test_case: TestCase) -> Score:
        """Evaluate across test_case.runs. Called by measure()."""

    def measure(self, test_case: TestCase) -> Score:
        if test_case.runs:
            return self.measure_runs(test_case)
        return Score(name=self.name, value=0.0, threshold=self.threshold,
                     success=False, reason="No runs provided")
```

## Phased Implementation

See `PLAN.md` for the full vision with 6 phases and ~37 metrics. Phase 1 (this skeleton) covers core framework + 14 metrics. Each subsequent phase adds metrics, capabilities, and directory structure as described in `PLAN.md`.

## Dependencies

**Core (Phase 1)**: `deepdiff>=7.0`, `jsonschema>=4.0` — two dependencies total.
**LLM (Phase 2+)**: `openai>=1.0`, `anthropic>=0.30` — optional.
**Dev**: `pytest>=8.0`, `ruff>=0.4`, `pytest-cov`, `pre-commit`.
