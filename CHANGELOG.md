# Changelog

All notable changes to harness-evals will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.14.0]

### Added

- **ROUGEMetric**: new similarity metric supporting ROUGE-1, ROUGE-2, and
  ROUGE-L variants for summarization evaluation. Pure Python implementation
  using whitespace tokenization with no external dependencies. Returns
  F-measure as primary score with precision/recall in metadata.

## [0.12.2]

### Added

- **Per-item progress callback on `evaluate_dataset`**: new optional `on_result`
  parameter (typed `OnResult`, exported from the top-level package) invoked once
  per dataset item as it finishes with `(index, total, eval_case, scores)`
  (`index` 0-based, completion order). Lets consumers surface per-item
  progress/logging at whatever level and format they choose without the library
  picking one — e.g. the eval-runner service restoring the per-item execution
  log line that used to be emitted by its in-house engine. The callback is
  observation-only: exceptions it raises are caught and logged and never abort
  the run. Fires on both the single-turn and conversation evaluation paths. The
  existing per-item `DEBUG` line in the runner is unchanged.

## [0.12.1]

### Added

- **Import-time logging auto-config**: `harness_evals` now honors the
  `HARNESS_EVALS_LOG_LEVEL` env var at import time via the new
  `logging_config.init_from_env()` (called from `__init__`). Library consumers
  that don't go through the CLI (e.g. the eval-runner service, which calls
  `run_config`/`evaluate_dataset` directly) can enable framework logs — including
  the per-case `[i/total] input=… output=… metrics=[…] target_error=…` debug
  line — simply by setting `HARNESS_EVALS_LOG_LEVEL=debug`, with no code change
  on their side. When the env var is **unset**, logging is left untouched: the
  host application's configuration (and Python's `WARNING`+ `lastResort`
  behavior) is unchanged, so the framework never silences its own warnings. An
  invalid env value is ignored rather than raising during import.
  `configure_logging` is now exported from the top-level package.

## [0.11.5]

### Added

- **AWS Bedrock LLM clients (#47)**: new `harness_evals.llm.bedrock` module with two deferred-import
  clients for judge/eval use. `BedrockAnthropicLLM` runs Claude via `anthropic.AsyncAnthropicBedrock`;
  `BedrockOpenAILLM` runs OpenAI-compatible models (e.g. gpt-oss) via Bedrock's OpenAI-compatible
  endpoint (`https://bedrock-runtime.<region>.amazonaws.com/openai/v1`). Both authenticate with a
  Bedrock **API key (bearer)** — `api_key=` or `AWS_BEARER_TOKEN_BEDROCK` — and fail fast with a clear
  error if it is missing (no silent fallback to `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`); region via
  `aws_region` / `AWS_REGION`. `BedrockOpenAILLM.generate_json` appends the schema to the prompt and
  robustly extracts the JSON object (strips `<reasoning>` wrappers and markdown fences, balanced-brace
  fallback) since Bedrock's OpenAI models don't reliably enforce `json_schema`. Both clients subclass
  their direct-API counterparts, so `OpenAILLM` / `AnthropicLLM` are unchanged. Requires the `llm` extra.

### Changed

- `anthropic` dependency floor raised `>=0.30` → `>=0.60` (needed for `AsyncAnthropicBedrock` bearer auth).

## [0.11.4]

### Added

- **Dimension radar chart (ADR-009, closes #23)**: `HtmlReporter` now renders a self-contained SVG
  radar chart of `ScoreSummary.by_dimension` (one axis per dimension, radius = mean score) with no
  external dependencies, suitable for email/PR/CI artifacts. When the report has multiple variants,
  each variant is drawn as its own polygon on shared axes (in its variant color) rather than averaged
  into one shape; a single-variant report shows one polygon with a per-dimension legend. The safety
  axis is drawn in red (violation counts annotated on the axis for a single variant, or per variant
  in the legend), and scores with no declared dimension (the `"unknown"` bucket) are omitted from the
  axes and noted in a footnote. The viewBox is fitted to the labels so none are clipped.
- **StdoutSink**: the per-dimension summary block now shows a visual progress bar per dimension, and
  moves the `"unknown"` bucket to a footnote instead of listing it as an axis.

## [0.11.1]

### Added

- **Request templating (`{{input}}` / `{{input.foo}}`)** — `HttpTarget` and
  `StreamingHttpTarget` resolve `{{...}}` placeholders per request against the
  golden, in both `body_template` values **and header values**: `{{input}}`
  (whole value, native type preserved in the body), `{{input.field}}` /
  `{{input.items.0}}` (dotted paths with list indices), and `{{metadata.key}}`.
  In the body a whole-string placeholder keeps the value's type; an embedded one
  is string-interpolated. Header values are always string-interpolated (a header
  can't carry a dict), and header names are never templated — e.g.
  `headers={"Authorization": "Bearer {{input.token}}"}`. Unresolved placeholders
  raise rather than silently sending null. This is separate from `${VAR}` env
  interpolation (config-load time).
- **Best-effort trajectory for all targets** — `EvalCase.messages` is now always
  populated. `PromptTarget` records the `[user, assistant]` exchange; `HttpTarget`
  and `StreamingHttpTarget` synthesize a trace from the input, any captured
  `tool_calls`, and the output when the agent does not report its own trajectory
  via `messages_path` (which stays authoritative). Extracted `messages`/`tool_calls`
  are coerced into `Message`/`ToolCall` objects so agent/trajectory metrics receive
  attribute-accessible instances rather than raw dicts. A reported trajectory that
  fails to coerce is left empty (and logged) rather than silently replaced by a
  fabricated trace, so instrumentation failures surface instead of grading clean.
  `ToolCall.from_dict` now honors `arguments`/`result` field aliases so extracted
  tool calls don't lose input/output.
- **SSE trajectory reconstruction** — `StreamingHttpTarget` rebuilds an ordered,
  interleaved trajectory (assistant text / tool calls / tool results) from the
  event stream in emission order, using the existing `output_path` /
  `tool_calls_path`. Streams with no assemblable structure fall back to the plain
  envelope; raw events are still captured to `sse_events`.

### Changed

- **BREAKING (pre-1.0): removed `input_path` from `HttpTarget` / `StreamingHttpTarget`.**
  Single-value injection at a path is replaced by `{{input}}` placeholders in
  `body_template`. Migration: `body_template={"prompt": null}, input_path="$.prompt"`
  becomes `body_template={"prompt": "{{input}}"}`. Omitting `body_template` still
  wraps the input as `{"input": <golden.input>}`.
- **`evaluate_dataset()` no longer requires `simulator_llm` for SCRIPTED/REPLAY
  conversations** — the LLM is now only required when at least one
  `ConversationGolden` uses SIMULATE or GRAPH mode. `ConversationSimulator` accepts
  `simulator_llm=None` and raises a clear error at point of use if an LLM-dependent
  code path is reached without one.

## [0.11.0]

### Added

- **Turn-level (conversational) RAG metrics**: `TurnFaithfulnessMetric`,
  `TurnContextualPrecisionMetric`, `TurnContextualRecallMetric`, and
  `TurnContextualRelevancyMetric`. Each scores every assistant turn against *that turn's own*
  retrieved context, aggregates to a conversation-level mean, and exposes the per-turn breakdown in
  `Score.metadata["turn_scores"]`. Registered in the catalog as `turn_faithfulness`,
  `turn_contextual_precision`, `turn_contextual_recall`, and `turn_contextual_relevancy`.
- `Message.retrieval_context: list[str] | None` and `Message.expected: str | None` — set on the
  assistant message of a turn to carry per-turn retrieval and gold answers through
  `evaluate_conversation()`.

### Changed

- Turn-level RAG metrics treat an assistant turn missing required inputs (retrieval context, or a
  required query/expected) as a **localized failure** scored `0.0`, so a per-turn retriever failure
  drags the conversation score down instead of vanishing from the average. Pass `allow_skips=True`
  to exclude such turns instead; skipped turns are still surfaced in `turn_scores` with
  `"skipped": True` and counted in `metadata["n_skipped_turns"]`.

## [0.10.0]

### Removed

- **BREAKING (pre-1.0): `ScoreSummary.overall_pass_rate`** has been removed. It blended safety scores
  into an aggregate pass rate, contradicting ADR-003 (safety is a hard constraint, never averaged). Use
  `quality_pass_rate` (non-safety scores) together with `safety_pass_rate` / `safety_violations`.
  Migration: `s/overall_pass_rate/quality_pass_rate/`. `StdoutSink` now prints "Quality pass rate".

### Added

- **Dimension-level aggregation (ADR-009)**: `summarize()` now populates `ScoreSummary.by_dimension`
  (a `DimensionSummary` per dimension: `mean`, `pass_rate`, `metric_count`, `is_safety`). Scores
  without a declared dimension are bucketed under `"unknown"`.
- **Safety separation (ADR-003)**: `ScoreSummary` exposes `quality_pass_rate`, `safety_pass_rate`,
  and `safety_violations`; safety is reported separately and excluded from the quality pass rate.
- **StdoutSink**: prints a per-dimension breakdown block and a distinct safety line.
- **OtlpSink**: emits `eval.summary.dimension.<dim>.{mean,pass_rate,count}` and
  `eval.summary.dimension.safety.violations` on the root eval-run span. Per-score gauges and events
  always carry `eval.dimension` (falling back to `"unknown"`), matching the summary aggregation so
  dashboards grouped by dimension never silently drop undeclared metrics.

## [0.9.4]

### Added

- **`StreamingHttpTarget`** (`type: streaming_http`) — a generic, vendor-neutral
  target that POSTs to a Server-Sent Events (`text/event-stream`) endpoint and
  maps the stream to an `EvalCase`. Parses named SSE events, optionally captures
  a configured subset into `metadata["sse_events"]` (`capture_events`), and
  selects a final output via `output_event` + `output_path`. When `output_event`
  is unset, output is auto-selected by scanning backward for the last JSON `data`
  payload from which `output_path` resolves — trailing envelope/telemetry events
  (e.g. `model_usage`, `done`, `stream_metadata`) are skipped instead of being
  mistaken for the answer; if nothing resolves it falls back to concatenated text
  (token streams) or empty + a warning for structured streams. Non-streaming responses fall
  back to buffered JSON/text parsing, matching `HttpTarget`. Async path uses
  `httpx.AsyncClient.stream`; a sync `urllib` fallback is used outside an
  `async with` context.
- Recursive `${VAR}` environment interpolation for target params in eval configs,
  including nested values in `headers` and `body_template`.
- `examples/streaming-http.*` — generic SSE example (config, goldens, custom
  `sse_trajectory` observability metric, README).

## [0.9.2]

### Fixed

- **Discrimination metric**: AUC-ROC now handles tied confidence scores by advancing the ROC
  curve one tied block at a time. Previously, cases sharing a confidence value produced
  order-dependent results (e.g. all-equal confidence could score anywhere from 0.0 to 1.0
  instead of the correct 0.5).
- **RAG faithfulness & context precision**: clamp the score to `≤1.0` so a malformed judge
  response returning more verdicts than there were claims/chunks no longer raises and aborts
  the evaluation run.
- **Bias metric**: fails closed when the classifier returns no classifications for extracted
  opinions (returns `0.0` instead of a silent `1.0` pass), and normalizes against opinions
  actually classified rather than opinions extracted.
- **CLI `--update-baseline`**: no longer overwrites the baseline when the run failed its gates,
  preventing a regressed run from silently poisoning the stored baseline.
- **JUnit sink**: strips XML-1.0-illegal control characters from names, inputs, and reasons so
  output is no longer rejected as malformed by CI parsers (Jenkins, GitHub Actions, GitLab).
- **HTTP target**: fails fast on `4xx` client errors instead of exhausting retries with
  exponential backoff; `5xx` and network/timeout errors are still retried. Applies to both the
  async and sync paths.
- **Operational metrics** (`Latency`, `CostEfficiency`, `TokenCost`, `RetryCount`,
  `TurnLatency`, `TurnTokenCost`): validate that their configured bound is positive, raising a
  clear `ValueError` instead of `ZeroDivisionError` at measure time.
- **Judge metrics**: non-numeric judge output (e.g. `{"score": "high"}` or `null`) now degrades
  to the metric's safe default via a shared coercion helper instead of raising and aborting the
  run. Covers toxicity, harmful-advice, misuse, prompt-injection, role-violation, hallucination,
  harm-severity, pairwise, and rubric metrics.
- **Security composite**: stores a copy of the weight mapping in result metadata so callers can
  no longer mutate the module-global default weights in place.
- **Langfuse dataset source**: guards against a missing `page.meta` during pagination, falling
  back to a short-page stop condition instead of raising `AttributeError` mid-fetch.
- **HTTP dataset source**: honors the `Content-Type` charset when decoding responses instead of
  hard-coding UTF-8.
- **Langfuse importer**: parses OpenAI-style `function.arguments` JSON strings into a dict so
  `ToolCall.input` is a consistent type for downstream tool-argument metrics.
- **Runner**: single-turn dataset sink writes are now emitted in input (golden) order while
  preserving concurrency, so appended sink rows line up with the dataset.

## [0.9.1]

### Changed

- README brought back in sync with the shipped API. Added the previously undocumented metrics
  (Webhook, StructuralSimilarity, TurnLatency, TurnTokenCost, DAG, PromptAlignment, Summarization,
  and the expanded safety detectors: Bias, Compliance, HarmSeverity, HarmfulAdvice, MisuseDetection,
  RoleViolation) to the Available Metrics table; documented `SimulationGraph` (and the `GRAPH`
  conversation mode), `PromptOptimizer`, and the `Synthesizer`/`InputGenerator` dataset generators;
  linked the framework integration examples; and fixed the stale git clone URL.

## [0.7.3]

### Added

- **`scoring_duration_ms` on `Score`** — optional field tracking how long metric scoring took
  (in milliseconds). `JUnitSink` now emits `time` attributes on `<testcase>` and `<testsuite>`
  elements, so CI test tabs (Harness, Jenkins, etc.) show actual durations instead of 0s.

## [0.5.0] - 2026-04-17

### Added

- `ToolArgumentMatchMetric` — deterministic comparison of tool-call arguments against authored expectations. Companion to `ToolCorrectnessMetric` (names) and the LLM-judged `ArgumentCorrectnessMetric`. Supports `pair=exact|subset`, `arg_match=exact|subset`, `ignore_keys`, and `wildcard_value`. Registered in the catalog as `"tool_argument_match"`.
- `Golden.expected_tool_calls: list[ToolCall] | None` and `EvalCase.expected_tool_calls: list[ToolCall] | None` — optional, defaults to `None`. Lets dataset authors carry expected tool-call arguments alongside `expected_tools`. `Golden.from_dict`, `EvalCase.from_dict`, and `EvalCase.from_golden` handle (de)serialization and propagation.
- ADR-010: Why `ToolArgumentMatchMetric` is a separate metric (not an enrichment of `ToolCorrectness`).

### Changed

- README and PLAN.md updated with the new metric, the canonical `ToolCorrectness` + `ToolArgumentMatch` pairing snippet, and the data-model note.

### Notes

- Fully backward-compatible: no existing field changes its type or default; existing JSONL datasets continue to load unchanged.

## [0.2.0] - 2026-03-16

### Added

- `Golden` dataclass — authored evaluation data (input, expected, context)
- `EvalCase` dataclass — replaces `TestCase`, adds typed operational fields (`latency_ms`, `token_count`, `cost_usd`, `retry_count`, `confidence`)
- `EvalCase.from_golden()` — factory to create EvalCase from Golden + agent output
- `EvalCase.from_dict()` / `Golden.from_dict()` — with backward-compat aliases (`actual_output` -> `output`, `expected_output` -> `expected`, `token_usage` -> `token_count`)
- `Score.passed` — auto-computed from `value >= threshold` in `__post_init__`
- `Score.created_at` — UTC timestamp set at creation
- `Score.to_dict()`, `Golden.to_dict()`, `EvalCase.to_dict()` — serialization methods
- `evaluate_cases()` — sync batch evaluation of pre-captured eval cases
- `evaluate_dataset()` — async evaluation: runs agent on goldens, then scores
- `BaseMetric.a_measure()` — async variant, defaults to calling sync `measure()`
- `ReliabilityMetric.a_measure_runs()` — async variant for multi-run metrics
- ADR-007: Why Golden and EvalCase are separate types

### Changed

- **BREAKING**: `TestCase` removed, replaced by `Golden` + `EvalCase`
- **BREAKING**: `actual_output` renamed to `output`
- **BREAKING**: `expected_output` renamed to `expected`
- **BREAKING**: `Score.success` renamed to `Score.passed` (auto-computed, not in constructor)
- Operational metrics read typed fields (`eval_case.latency_ms`) instead of `metadata` dict
- `ResourceConsistencyMetric` default `resource_key` changed from `"token_usage"` to `"token_count"`
- ADR-006 updated to reflect sync `measure()` + async `a_measure()` pattern

### Removed

- `TestCase` dataclass (use `Golden` + `EvalCase` instead)
- `Score.success` constructor parameter (use auto-computed `Score.passed`)

### Migration

Replace `TestCase` usage:

```python
# Before
tc = TestCase(input="q", actual_output="a", expected_output="e",
              metadata={"latency_ms": 100})
score = metric.measure(tc)
if score.success: ...

# After
ec = EvalCase(input="q", output="a", expected="e", latency_ms=100)
score = metric.measure(ec)
if score.passed: ...
```

Or use `from_dict()` with old field names (backward compatible):

```python
ec = EvalCase.from_dict({"input": "q", "actual_output": "a", "expected_output": "e"})
```

## [0.1.0] - 2026-03-16

### Added

- Core types: `TestCase`, `Score`, `BaseMetric`, `ReliabilityMetric`, `BaseSink`
- Runner functions: `evaluate()` (non-raising) and `assert_test()` (raises on failure)
- Deterministic metrics: `ExactMatchMetric`, `ContainsMetric`, `RegexMetric`, `NumericDiffMetric`
- Structural metrics: `JsonDiffMetric` (DeepDiff-backed), `SchemaValidationMetric` (jsonschema-backed)
- Operational metrics: `LatencyMetric`, `TokenCostMetric`, `CostEfficiencyMetric`, `RetryCountMetric`
- Reliability metrics: `OutcomeConsistencyMetric`, `ResourceConsistencyMetric`
- Output sinks: `StdoutSink`, `JsonSink`
- Project infrastructure: pyproject.toml, .gitignore, .pre-commit-config.yaml
- CI: GitHub Actions workflow for Python 3.10/3.11/3.12
- Documentation: README.md, AGENTS.md, PLAN.md (full 6-phase vision)
- 42 passing tests covering all metrics and core functions
- Example: `examples/basic_eval.py`

### Planned

- Phase 5: Synthesizer, perturbation generators
- Phase 6: Harness AI Evals integration
