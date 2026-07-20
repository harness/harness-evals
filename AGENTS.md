# AGENTS.md - harness-evals

## Project Overview

**harness-evals** is an open-source AI evaluation framework for LLM agents, prompts, and structured outputs. It provides a `pip install`-able scoring engine with 70+ metrics across deterministic, structural, operational, reliability, predictability, MCP, similarity, LLM-judged, RAG, safety, agent, conversation, and security categories.

**Core principle**: An eval always produces a `Score`. Every metric is a single class with a `measure()` method.

**Data flow**: `Golden` (authored) + agent output -> `EvalCase` (evaluated) -> `Score` (result)

**Language**: Python 3.10+
**License**: Apache 2.0
**Package name**: `harness-evals`
**Import name**: `harness_evals`

## Build System

```bash
pip install -e "."                # Core only (deterministic metrics, no LLM key needed)
pip install -e ".[llm]"           # + OpenAI, Anthropic for LLM-judged metrics
pip install -e ".[otlp]"          # + OTLP metrics & traces export
pip install -e ".[langfuse]"      # + Langfuse source/sink
pip install -e ".[similarity]"    # + BLEU metric (nltk)
pip install -e ".[harness]"       # + Harness AI Service LLM provider
pip install -e ".[benchmarks]"    # + Academic benchmarks (MMLU, GSM8K, HumanEval, etc.)
pip install -e ".[all]"           # Everything
pip install -e ".[all,dev]"       # Everything + dev tools
```

**Build tool**: Poetry via `pyproject.toml` (backend: `poetry-core`)
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

## Publishing

The package is published automatically via the Harness CI pipeline (`.harness/publish.yaml`) when a version change is detected on `main`.

**How it works**: The pipeline compares `version` in `pyproject.toml` at `HEAD` vs `HEAD~1`. If the version changed, it builds and publishes to `harness-pip-internal`.

**You MUST bump the version in `pyproject.toml`** whenever your changes should be released. If you don't bump the version, the package will NOT be published — even if code changes are merged.

```bash
# In pyproject.toml, update:
version = "X.Y.Z"  # Bump this to trigger a publish
```

Follow semver: patch for fixes, minor for new metrics/features, major for breaking changes.

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
├── pyproject.toml                   # Package config (Poetry), dependencies, tool settings
├── README.md                        # User-facing documentation
├── AGENTS.md                        # This file
├── PLAN.md                          # Full vision spec
├── CHANGELOG.md                     # Version history
├── LICENSE                          # Apache 2.0
├── .harness/publish.yaml            # CI pipeline for publishing
│
├── src/harness_evals/
│   ├── __init__.py                  # Public API re-exports
│   ├── py.typed                     # PEP 561 marker
│   ├── cli.py                       # CLI entry point (harness-evals command)
│   ├── eval.py                      # run_eval() one-liner
│   ├── catalog.py                   # Metric catalog/registry
│   ├── plugins.py                   # Plugin registration (decorators + entry points)
│   ├── refs.py                      # ResourceRef URI system
│   ├── errors.py                    # Custom exceptions
│   ├── summary.py                   # Score summarization & aggregation
│   ├── http_utils.py               # Shared HTTP utilities
│   │
│   ├── core/
│   │   ├── golden.py                # Golden dataclass (authored data)
│   │   ├── eval_case.py             # EvalCase dataclass (what metrics receive)
│   │   ├── score.py                 # Score dataclass (passed auto-computed)
│   │   ├── metric.py                # BaseMetric, ReliabilityMetric, SafetyMetric ABCs
│   │   ├── types.py                 # Message, ToolCall dataclasses
│   │   ├── sink.py                  # BaseSink ABC
│   │   └── runner.py                # evaluate(), assert_test(), evaluate_cases(), evaluate_dataset()
│   │
│   ├── metrics/
│   │   ├── deterministic/           # ExactMatch, Contains, Regex, NumericDiff, ListContains, Webhook
│   │   ├── structural/              # JsonDiff, SchemaValidation, StructuralSimilarity
│   │   ├── operational/             # Latency, TokenCost, CostEfficiency, RetryCount, TurnLatency, TurnTokenCost
│   │   ├── reliability/             # OutcomeConsistency, ResourceConsistency, TrajectoryConsistency,
│   │   │                            # PromptRobustness, EnvironmentRobustness, FaultRobustness,
│   │   │                            # BrierScore, Calibration, Discrimination
│   │   ├── similarity/              # Levenshtein, BLEU, EmbeddingSimilarity
│   │   ├── llm_judge/              # GEval, RubricJudge, Pairwise, DAG, PromptAlignment, Summarization
│   │   ├── rag/                     # Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
│   │   │                            # AnswerCorrectness, AnswerSimilarity, ContextEntityRecall,
│   │   │                            # ContextRelevancy, Conversational (turn-level RAG)
│   │   ├── safety/                  # PII, Toxicity, PromptInjection, Hallucination, Bias, Compliance,
│   │   │                            # HarmSeverity, HarmfulAdvice, MisuseDetection, RoleViolation
│   │   ├── agent/                   # ToolCorrectness, ToolArgumentMatch, TaskCompletion,
│   │   │                            # ArgumentCorrectness, PlanQuality, PlanAdherence, StepEfficiency
│   │   ├── conversation/            # Coherence, Resolution, Completeness, TurnEfficiency, TurnRelevancy,
│   │   │                            # KnowledgeRetention, RoleAdherence, TopicAdherence, GoalAccuracy,
│   │   │                            # ToolUse, ConversationalGEval
│   │   ├── mcp/                     # ToolSelectionAccuracy, MCPTraceCompleteness
│   │   ├── security/               # VulnerabilityCorrectness, SecurityCompleteness, CodeSafety,
│   │   │                            # CodeQuality, ExplanationQuality, RootCauseAnalysis, Actionability
│   │   └── composite/              # CompositeMetric (combine metrics with operators)
│   │
│   ├── benchmarks/                  # Academic benchmark evaluation suites
│   │   ├── __init__.py              # Public exports (all benchmark classes)
│   │   ├── base.py                  # BaseBenchmark ABC, BenchmarkResult
│   │   ├── dataset_cache.py         # HuggingFace dataset fetching + local caching
│   │   ├── sandbox.py               # Process-isolated Python code execution (subprocess)
│   │   ├── _answer_utils.py         # Shared answer extraction (choice, number, F1)
│   │   ├── mmlu.py                  # MMLU (57-subject knowledge)
│   │   ├── gsm8k.py                # GSM8K (math word problems)
│   │   ├── humaneval.py            # HumanEval (code generation, sandboxed)
│   │   ├── truthfulqa.py           # TruthfulQA (LLM-judged truthfulness)
│   │   ├── arc.py                  # ARC Easy + Challenge (science questions)
│   │   ├── hellaswag.py            # HellaSwag (commonsense reasoning)
│   │   ├── winogrande.py           # WinoGrande (pronoun resolution)
│   │   ├── boolq.py                # BoolQ (boolean reading comprehension)
│   │   ├── drop.py                 # DROP (numerical reasoning, F1 + EM)
│   │   └── bbh.py                  # BBH (23 hard reasoning tasks)
│   │
│   ├── llm/                         # LLM provider abstraction
│   │   ├── base.py                  # BaseLLM ABC
│   │   ├── openai.py               # OpenAILLM
│   │   ├── anthropic.py            # AnthropicLLM
│   │   ├── harness_ai.py           # HarnessAILLM
│   │   ├── embedding.py            # Embedding base
│   │   └── openai_embedding.py     # OpenAI embeddings
│   │
│   ├── targets/                     # System-under-test adapters
│   │   ├── base.py                  # BaseTarget ABC
│   │   ├── prompt.py               # PromptTarget (template + LLM)
│   │   ├── http.py                 # HttpTarget (POST to endpoint)
│   │   ├── streaming_http.py       # StreamingHttpTarget (SSE)
│   │   ├── auth.py                 # Auth configs (Bearer, ApiKey, Basic)
│   │   ├── templating.py           # Request template rendering
│   │   └── trajectory.py           # Trajectory capture
│   │
│   ├── datasets/                    # Dataset I/O and sources
│   │   ├── io.py                    # load_dataset(), save_dataset()
│   │   ├── base.py                  # BaseDatasetSource ABC
│   │   ├── local.py                # LocalDatasetSource
│   │   ├── http.py                 # HttpDatasetSource
│   │   └── langfuse.py            # LangfuseDatasetSource
│   │
│   ├── prompts/                     # Prompt template system
│   │   ├── template.py             # PromptTemplate ({{var}} placeholders)
│   │   ├── base.py                 # BasePromptSource ABC
│   │   ├── local.py                # LocalPromptSource
│   │   ├── http.py                 # HttpPromptSource
│   │   └── langfuse.py            # LangfusePromptSource
│   │
│   ├── importers/                   # Production trace → EvalCase
│   │   ├── base.py                  # BaseEvalCaseSource, BaseEvalConfigSource ABCs
│   │   ├── langfuse.py             # LangfuseEvalCaseSource
│   │   └── otel.py                 # OTELEvalCaseSource
│   │
│   ├── config/                      # YAML eval config system
│   │   ├── schema.py               # EvalConfig dataclass
│   │   └── runner.py               # run_config(), load_config()
│   │
│   ├── conversation/                # Multi-turn conversation evaluation
│   │   ├── golden.py               # ConversationGolden, ConversationMode
│   │   ├── simulator.py            # ConversationSimulator
│   │   ├── graph.py                # SimulationGraph, ScriptedNode, LLMNode, BranchNode, StopNode
│   │   └── runner.py               # evaluate_conversation(), evaluate_conversations()
│   │
│   ├── baseline/                    # Score regression detection
│   │   ├── store.py                # BaselineStore ABC
│   │   ├── json_store.py           # JsonBaselineStore
│   │   └── compare.py             # compare_to_baseline()
│   │
│   ├── optimizer/                   # Prompt optimization loop
│   │   └── optimizer.py            # PromptOptimizer
│   │
│   ├── synthesizer/                 # Dataset generation from documents
│   │   ├── base.py                 # Synthesizer
│   │   ├── conversation.py         # ConversationSynthesizer, ScriptedConversationSynthesizer
│   │   ├── extraction.py           # Extraction-style synthesis
│   │   ├── qa.py                   # QA-style synthesis
│   │   ├── structured.py          # Structured output synthesis
│   │   └── summarization.py       # Summarization synthesis
│   │
│   ├── input_generator/             # Input variation generation
│   │   ├── base.py                 # InputGenerator
│   │   ├── rephrase.py            # Rephrasings
│   │   ├── adversarial.py         # Adversarial rewrites
│   │   ├── complexity_ladder.py   # Difficulty scaling
│   │   └── use_case.py            # Use-case variations
│   │
│   ├── perturbations/               # Input perturbation for robustness metrics
│   │   ├── base.py                 # BasePerturbation
│   │   ├── rephrase.py            # Rephrase perturbation
│   │   ├── typo.py                # Typo injection
│   │   ├── json_reorder.py        # JSON key reordering
│   │   └── schema_variation.py    # Schema variations
│   │
│   ├── testing/                     # Fault injection for robustness testing
│   │   └── fault_injector.py      # FaultInjector, Fault
│   │
│   ├── reporting/                   # HTML report generation
│   │   ├── html_reporter.py       # HtmlReporter
│   │   └── html_sink.py           # HtmlSink
│   │
│   ├── sinks/                       # Output destinations
│   │   ├── stdout.py              # StdoutSink
│   │   ├── json_sink.py           # JsonSink
│   │   ├── csv_sink.py            # CsvSink
│   │   ├── junit_sink.py          # JUnitSink
│   │   ├── langfuse_sink.py       # LangfuseSink
│   │   └── otlp_sink.py           # OtlpSink
│   │
│   └── sources/                     # Deprecated import path shims
│       ├── langfuse.py            # → importers.langfuse (DeprecationWarning)
│       └── otel.py                # → importers.otel (DeprecationWarning)
│
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_core.py                 # Golden, EvalCase, Score, evaluate, assert_test, etc.
│   ├── metrics/                     # One test file per metric category
│   └── benchmarks/                  # Tests for academic benchmarks (mocked, no HF calls)
│
└── examples/
    └── integrations/                # Framework integration examples
```

## How to Add a New Metric

This is the most common task an AI agent will do. Follow these steps:

1. **Pick the category** — deterministic, structural, operational, reliability, similarity, llm_judge, rag, safety, agent, conversation, mcp, security, or composite.
2. **Create the file** — `src/harness_evals/metrics/<category>/<metric_name>.py`
3. **Implement the class** — extend `BaseMetric` (or `ReliabilityMetric` for multi-run, `SafetyMetric` for safety):

```python
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.core.eval_case import EvalCase


class MyMetric(BaseMetric):
    dimension = Dimension.CORRECTNESS  # or GROUNDEDNESS, SAFETY, TRAJECTORY, PERFORMANCE

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
    expected_tools: list[str] | None = None
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
    messages: list[Message] | None = None
    tool_calls: list[ToolCall] | None = None
    expected_tools: list[str] | None = None
    expected_tool_calls: list[ToolCall] | None = None
    latency_ms: float | None = None
    token_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    retry_count: int | None = None
    confidence: float | None = None
    tags: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
    runs: list["EvalCase"] | None = None
```

`input_tokens`/`output_tokens` are the target's token split. `PromptTarget` populates them automatically from whatever the underlying `BaseLLM.generate()` records (via `collect_token_usage()`); `HttpTarget`/`StreamingHttpTarget` populate them from the response body via `input_tokens_path`/`output_tokens_path` JSONPath config. For `StreamingHttpTarget`, these paths resolve against the *whole* SSE stream, not only the answer payload — usage typically arrives in a separate trailing event (e.g. `model_usage`/`done`), so any numeric field the answer payload lacks is recovered by scanning the other events (latest match wins). A configured path that resolves to a non-numeric or out-of-range value is dropped with a logged warning rather than silently ignored. `None` means "no usage was observed" — never coerce to `0`.

### Token usage capture (`llm/usage.py`, `core/runner.py`)

Token accounting is per-`asyncio`-task, not global: `collect_token_usage()` is a context manager that scopes a `ContextVar`-backed collector to the current task, and LLM providers (`OpenAILLM`, `AnthropicLLM`, `HarnessAILLM`) call `record_token_usage(input_tokens=..., output_tokens=...)` inside `generate()`/`generate_json()`. This keeps concurrent judge calls that share one `BaseLLM` instance (e.g. under `asyncio.gather` in `a_evaluate`) from cross-contaminating each other's counts.

- `a_evaluate()` wraps each metric's `a_measure()` call in `collect_token_usage()` and, if any usage was recorded, writes it to `score.metadata["input_tokens"]`/`["output_tokens"]` (via `setdefault`, so a metric that sets these explicitly wins).
- `HarnessAILLM._record_gateway_usage()` is defensive by design: it runs inline in a live judge call, so a malformed `/chat` response must never raise — it's wrapped in `contextlib.suppress(Exception)` plus best-effort int coercion, and silently no-ops on any unrecognized response shape.

### Message (conversation turn)

```python
@dataclass
class Message:
    role: str
    content: str
    tool_calls: list[ToolCall] | None = None
```

### ToolCall (tool/function invocation)

```python
@dataclass
class ToolCall:
    name: str
    input: dict[str, Any] | None = None
    output: str | dict | None = None
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
    dimension: Dimension   # CORRECTNESS, GROUNDEDNESS, SAFETY, TRAJECTORY, PERFORMANCE

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

### SafetyMetric (for safety — never averaged)

```python
class SafetyMetric(BaseMetric):
    dimension = Dimension.SAFETY
```

## Dependencies

**Core**: `deepdiff>=7.0`, `jsonschema>=4.0`, `jsonpath-ng>=1.6`, `pyyaml>=6.0`
**LLM**: `openai>=1.40`, `anthropic>=0.30` — optional `[llm]`
**OTLP**: `opentelemetry-sdk>=1.20`, `opentelemetry-exporter-otlp-proto-grpc>=1.20`, `opentelemetry-exporter-otlp-proto-http>=1.20` — optional `[otlp]`
**Similarity**: `nltk>=3.9.4` — optional `[similarity]`
**Harness**: `httpx>=0.27`, `pyjwt>=2.13.0` — optional `[harness]`
**Benchmarks**: `httpx>=0.27` — optional `[benchmarks]` (datasets fetched from HuggingFace Hub)
**Langfuse**: `langfuse>=2.0` — optional `[langfuse]`
**Dev**: `pytest>=8.0`, `ruff>=0.15`, `pytest-cov`, `pytest-asyncio`, `pre-commit`, `build`
