# Contributing to harness-evals

## Quick Start

```bash
git clone git@github.com:sunilgattupalle/harness-evals.git
cd harness-evals
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pre-commit install
ruff check src/ tests/          # lint
ruff format --check src/ tests/ # format
pytest tests/ -v                # 61 tests should pass
```

## Development Workflow

1. Create a branch: `git checkout -b feat/my-feature`
2. Make changes, following existing patterns
3. Run `ruff check src/ tests/`, `ruff format --check src/ tests/`, and `pytest tests/ -v`
4. Commit with conventional format: `feat: add faithfulness metric`
5. Push and open a PR

## Commit Convention

```
type: short description

Optional body explaining why.
```

Types: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`

## Adding a New Metric

This is the most common contribution. See [docs/metrics-guide.md](metrics-guide.md) for the full guide. Quick version:

1. Decide the category: `deterministic`, `structural`, `operational`, `reliability`, `llm_judge`, `rag`, `safety`, `agent`, `conversation`, `mcp`
2. Create `src/harness_evals/metrics/<category>/<metric_name>.py`
3. Extend `BaseMetric` (or `ReliabilityMetric` for multi-run metrics)
4. Implement `measure(self, eval_case: EvalCase) -> Score`
5. Export from `<category>/__init__.py` and `metrics/__init__.py`
6. Create `tests/metrics/test_<metric_name>.py`
7. Run `ruff check src/ tests/`, `ruff format --check src/ tests/`, and `pytest tests/ -v`

A single metric is a single-file PR. Look at `exact_match.py` as a template.

## Adding a New Sink

1. Create `src/harness_evals/sinks/<sink_name>.py`
2. Extend `BaseSink` and implement `write(self, scores, eval_case)`
3. Export from `sinks/__init__.py`
4. Add tests

## Code Style

- **Formatter/linter**: ruff (configured in pyproject.toml)
- **Type hints**: Required on all function signatures
- **Line length**: 120 characters
- **Imports**: Sorted by ruff (stdlib, third-party, first-party)
- **Docstrings**: Required on classes. One-line for simple functions.

## Testing

```bash
pytest tests/ -v                              # All tests
pytest tests/ -v -m unit                      # Unit tests only
pytest tests/metrics/test_exact_match.py -v   # Specific file
```

- Mark tests with `@pytest.mark.unit` or `@pytest.mark.integration`
- Test data goes in `tests/data/`
- Use fixtures from `tests/conftest.py` for common EvalCase patterns
- Every metric needs at least: one pass test, one fail test, one edge case test

## What NOT to Do

- Don't add heavy ML dependencies (torch, transformers) to core
- Don't modify `Golden`, `EvalCase`, or `Score` fields without updating PLAN.md
- Don't average safety scores into overall scores
- Don't use `print()` for output — use the sink system
- Don't skip pre-commit hooks

## Project Map

| File | What It Is |
|------|-----------|
| `PLAN.md` | Full 6-phase vision with specs for all ~37 metrics |
| `AGENTS.md` | Quick reference for AI agents (types, patterns, commands) |
| `docs/architecture.md` | System design, data flow, extension points |
| `docs/metrics-guide.md` | Detailed metric authoring guide |
| `docs/integration-guide.md` | pytest, CI/CD, Harness CI integration |
| `docs/adr/` | Architecture Decision Records |

## Questions?

Open a GitHub issue with the `question` label.
