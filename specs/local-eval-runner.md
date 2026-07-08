# Design Spec: Local Eval Runner & Source Adapters

**Status:** Draft
**Date:** 2026-06-16
**Replaces:** `local-eval-runner.md` (original draft)

---

## Motivation

`harness-evals` today is a scoring engine: you hand-build `EvalCase`s (or hydrate
them from production traces), wire metrics in Python, and call `evaluate()`. There
is no way to:

1. Declare an eval in a config file and run it locally with one command.
2. Pull a dataset or prompt **by ID and version** from any registry.

This blocks the local-first developer loop that tools like Braintrust offer:

```
edit prompt → harness-evals run my-eval.yaml → diff scores against baseline
```

This spec adds four additive layers — **source adapters**, **targets**, **eval
importers**, and a **config/CLI runner** — plus a code-first `run_eval()` one-liner.
None of it touches `core/`; the existing `evaluate_dataset()` remains the execution
engine.

---

## Design Principles

1. **Core stays frozen.** `Golden`, `EvalCase`, `Score`, `BaseMetric`, `BaseSink`,
   `evaluate_dataset` are unchanged. No vendor references enter `core/`.

2. **Layered, not forked.** Vendor-specific adapters (Langfuse, Harness, etc.) live
   at the leaf level behind optional extras. The framework ABCs and registries are
   vendor-neutral. Harness is a reference implementation, not a privileged citizen.

3. **Local-first.** The zero-*external-account* path (`local://` dataset,
   `PromptTarget` + any `BaseLLM`, `harness-evals run`) works with no external
   platform accounts. Deterministic metrics (e.g. `exact_match`, `json_diff`) need
   no extras at all. LLM-judged metrics (e.g. `geval`) require `[llm]` but no
   external *account* beyond an API key.

4. **Fail loud at resolve-time.** A `ResourceRef` that references an uninstalled
   adapter raises `MissingAdapterError("<source>", extra="harness-evals[langfuse]")`
   at config-load time — never a cryptic `ImportError` at execution time.

---

## 1. Architecture

```
harness_evals/
├── core/                  ← FROZEN — zero new imports
├── llm/                   ← BaseLLM + openai, anthropic, harness_ai (behind [harness])
├── refs.py                ← NEW — ResourceRef + resolve()
├── plugins.py             ← NEW — four-family registries + entry-point loader
├── datasets/              ← EXTEND datasets.py → package
│   ├── base.py            —   BaseDatasetSource ABC + register_dataset_source
│   ├── local.py           —   LocalDatasetSource  (wraps load_dataset)
│   ├── http.py            —   HttpDatasetSource
│   ├── langfuse.py        —   LangfuseDatasetSource  ([langfuse] extra)
│   └── harness.py         —   HarnessDatasetSource   ([harness] extra)
├── prompts/               ← NEW
│   ├── base.py            —   BasePromptSource ABC + PromptTemplate + register_prompt_source
│   ├── local.py           —   LocalPromptSource
│   ├── http.py            —   HttpPromptSource
│   ├── langfuse.py        —   LangfusePromptSource   ([langfuse] extra)
│   └── harness.py         —   HarnessPromptSource    ([harness] extra)
├── targets/               ← NEW
│   ├── base.py            —   BaseTarget ABC
│   ├── prompt.py          —   PromptTarget
│   ├── http.py            —   HttpTarget
│   └── auth.py            —   AuthConfig variants
├── importers/             ← NEW (generalises sources/)
│   ├── base.py            —   BaseEvalCaseSource ABC + BaseEvalConfigSource ABC
│   ├── langfuse.py        —   LangfuseEvalCaseSource  ([langfuse] extra)
│   ├── otel.py            —   OTELEvalCaseSource      ([otlp] extra)
│   └── harness.py         —   HarnessEvalConfigSource ([harness] extra)
├── config/                ← NEW
│   ├── schema.py          —   EvalConfig dataclass + validation
│   └── runner.py          —   run_config(cfg) → list[list[Score]]
├── cli.py                 ← NEW — harness-evals run|import|list-metrics|discover
└── eval.py                ← NEW — run_eval() one-liner
```

**Dependency rule (unchanged from existing architecture.md):**

```
cli.py, eval.py
    └── config/
        └── datasets/, prompts/, targets/, importers/
            └── core/        ← leaf, imports nothing new
```

Vendor code (`langfuse`, `harness_ai`, etc.) only ever appears inside leaf adapter
files — never inside ABCs, registries, `refs.py`, `plugins.py`, or `config/`.

---

## 2. Data Flow

```
EvalConfig (YAML or run_eval() kwargs)
   │
   ├── dataset ref  ──► DatasetSource.fetch()  ──► list[Golden]
   ├── prompt ref   ──► PromptSource.fetch()   ──► PromptTemplate ─┐
   ├── target spec  ──────────────────────────► BaseTarget  ◄──────┘
   └── metric kinds ──► catalog() lookup ──────► list[BaseMetric]
                                  │
       evaluate_dataset(goldens, target.ainvoke, metrics, sinks)  ← EXISTING ENGINE
                                  │
                             list[list[Score]]  ──► sinks, baseline gate
```

The runner is **wiring only**. The for-each-golden loop is the existing
`evaluate_dataset()`; `target.ainvoke` is the `agent_fn` it already expects.

---

## 3. `ResourceRef` and `resolve()`

Both the URI shorthand and the typed-block YAML forms normalise to one handle.

```python
@dataclass(frozen=True)
class ResourceRef:
    source: str                       # "local" | "http" | "langfuse" | "harness" | <plugin>
    id: str
    version: str | None = None        # None = "latest" (adapter-defined semantics)
    extra: dict[str, Any] = field(default_factory=dict)

def resolve(spec: str | dict) -> ResourceRef:
    """Normalise either syntax to a ResourceRef.

    URI form:    "langfuse://datasets/support-goldens@3"
                  scheme=source, host+path=id, @suffix=version
    Typed form:  {"source": "langfuse", "id": "support-goldens", "version": 3}
                  unknown keys land in .extra
    Bare string: "./goldens.jsonl"  →  ResourceRef(source="local", id="./goldens.jsonl")
    """
```

- **URI form:** `<source>://<id>@<version>`. Path after `://` is `id`; `@version`
  is optional; `?key=val` populates `extra`.
- **Typed form:** `source` and `id` required; `version` optional; unknown keys → `extra`.
- `version` normalised to `str | None` (so `3` and `"3"` are equal).
- A bare string with no `://` is a local file path shorthand.

### `MissingAdapterError`

`resolve()` looks up the source name in the appropriate family registry. If the
adapter class is not registered (because its optional extra is not installed), it
raises immediately:

```python
class MissingAdapterError(Exception):
    def __init__(self, source: str, family: str, install_hint: str) -> None: ...
    # "Adapter 'langfuse' for family 'dataset_sources' is not installed.
    #  Install it with: pip install harness-evals[langfuse]"
```

This fires at config-load time — before any network call or golden is fetched.

---

## 4. Plugin Registry (`plugins.py`)

Seven families, each with its own dict and decorator. One entry-point group per
family for auto-discovery of installed packages.

```python
# module-level registries
_DATASET_SOURCES:     dict[str, type[BaseDatasetSource]]    = {}
_PROMPT_SOURCES:      dict[str, type[BasePromptSource]]     = {}
_EVAL_CASE_SOURCES:   dict[str, type[BaseEvalCaseSource]]   = {}
_EVAL_CONFIG_SOURCES: dict[str, type[BaseEvalConfigSource]] = {}
_TARGETS:             dict[str, type[BaseTarget]]           = {}
_METRICS:             dict[str, type[BaseMetric]]           = {}
_BASELINE_STORES:     dict[str, type[BaselineStore]]        = {}
_SINKS:               dict[str, type[BaseSink]]             = {}

def register_dataset_source(name: str): ...
def register_prompt_source(name: str): ...
def register_eval_case_source(name: str): ...
def register_eval_config_source(name: str): ...
def register_target(name: str): ...
def register_metric(kind: str): ...
def register_baseline_store(name: str): ...
def register_sink(name: str): ...
```

`register_metric` makes third-party metrics discoverable via `catalog()` and
referenceable by `kind:` in YAML — the same mechanism used for built-in metrics.

`register_target` makes custom target types declarable in YAML `target.type:`.
Example: `target: {type: braintrust, ...}` resolves to `BrainTrustTarget`.

`register_baseline_store` allows platform-backed baseline stores (e.g. S3, Langfuse).
v1 ships `JsonBaselineStore` as the only built-in; the registration path is open.

Entry-point groups (one per family):

```toml
[project.entry-points."harness_evals.dataset_sources"]
acme = "acme_evals.adapters:AcmeDatasetSource"

[project.entry-points."harness_evals.prompt_sources"]
acme = "acme_evals.adapters:AcmePromptSource"

[project.entry-points."harness_evals.eval_case_sources"]
acme = "acme_evals.adapters:AcmeEvalCaseSource"

[project.entry-points."harness_evals.eval_config_sources"]
acme = "acme_evals.adapters:AcmeEvalConfigSource"

[project.entry-points."harness_evals.targets"]
acme = "acme_evals.targets:AcmeTarget"

[project.entry-points."harness_evals.metrics"]
acme_faithfulness = "acme_evals.metrics:AcmeFaithfulnessMetric"

[project.entry-points."harness_evals.baseline_stores"]
s3 = "acme_evals.stores:S3BaselineStore"

[project.entry-points."harness_evals.sinks"]
acme = "acme_evals.sinks:AcmeScoreSink"
```

`load_plugins(modules: list[str])` imports explicit modules listed in the YAML
`plugins:` key, triggering their `@register_*` decorators.

`catalog()` is extended to return a merged view: built-in metrics + any metrics
registered via `@register_metric` or plugin entry-point discovery.

### Invariant: families never mix

A vendor implements separate classes for each family it supports. `langfuse`
registers `LangfuseDatasetSource`, `LangfusePromptSource`, and
`LangfuseEvalCaseSource` as three distinct classes in three distinct registries.
Resolution dispatches on `(family, source_name)` — no collision.

---

## 5. Adapter Registry Table

The table is the canonical answer to "what ships by default vs what requires an
extra."

| Source | Dataset | Prompt | EvalCase | EvalConfig | Target | Metric | Baseline | Sink | Extra |
|--------|:-------:|:------:|:--------:|:----------:|:------:|:------:|:--------:|:----:|-------|
| `local` | ✅ | ✅ | — | — | — | — | — | — | none |
| `http`  | ✅ | ✅ | — | — | ✅ | — | — | — | none |
| `prompt` | — | — | — | — | ✅ | — | — | — | none |
| `langfuse` | ✅ | ✅ | ✅ | — | — | — | — | ✅ | `[langfuse]` |
| `otel`  | — | — | ✅ | — | — | — | — | — | `[otlp]` |
| `harness` | ✅ | ✅ | — | ✅ | — | — | — | — | `[harness]` |
| `json`  | — | — | — | — | — | — | ✅ | ✅ | none |
| `stdout` | — | — | — | — | — | — | — | ✅ | none |
| `csv`   | — | — | — | — | — | — | — | ✅ | none |
| `junit` | — | — | — | — | — | — | — | ✅ | none |
| `otlp`  | — | — | — | — | — | — | — | ✅ | `[otlp]` |

Built-in metrics (all `kind:` strings in `catalog()`) are pre-registered. Third
parties add a row by publishing a package with the relevant entry-point groups.
`Metric` and `Baseline` columns are omitted from rows where the adapter adds no
built-in implementations — third parties fill them via `@register_metric` /
`@register_baseline_store`.

---

## 6. Source Adapter ABCs

### 6.1 Invariant: the families never mix

| | `BaseDatasetSource` | `BasePromptSource` |
|---|---|---|
| **Produces** | `list[Golden]` | a single `PromptTemplate` |
| **Cardinality** | N rows | 1 |
| **Role** | the test set | the artifact under test |

```python
class BaseDatasetSource(ABC):
    name: str
    @abstractmethod
    async def fetch(self, ref: ResourceRef) -> list[Golden]: ...
    async def close(self) -> None: ...   # optional resource cleanup

class BasePromptSource(ABC):
    name: str
    @abstractmethod
    async def fetch(self, ref: ResourceRef) -> "PromptTemplate": ...
    async def close(self) -> None: ...
```

Both ABCs expose `async def fetch()` for consistency with network-backed adapters.
`LocalDatasetSource` and `LocalPromptSource` wrap synchronous file I/O inside async
methods — no threading needed for file reads at eval scale.

### 6.2 `PromptTemplate`

```python
@dataclass
class PromptTemplate:
    template: str                          # text with {{var}} placeholders
    input_variables: list[str] = field(default_factory=lambda: ["input"])
    model_hint: dict | None = None         # e.g. {"provider": "openai", "name": "gpt-4o"}
    version: str | None = None
    metadata: dict = field(default_factory=dict)

    def render(self, **kwargs) -> str: ...
```

**Placeholder syntax: `{{var}}` only.** Not `str.format` / `{var}`, not Jinja.
Rationale: prompts routinely contain JSON examples and code with literal `{` braces;
`{{var}}` is unambiguous, matches what Langfuse and Harness registries emit natively,
and keeps templates as data (no logic, no conditionals).

- `render()` replaces every `{{name}}` with `str(kwargs[name])`.
- A placeholder with no matching var raises immediately with the missing name.
- `input_variables` is validated against placeholders found at load time.
- Literal `{{` in template body: escape as `\{\{`, passes through as `{{`.

### 6.3 HTTP payload auto-detection

`HttpDatasetSource` detects format in this order:
1. Explicit `ref.extra["format"]` (`"jsonl"` or `"json"`) if present.
2. `Content-Type` header.
3. Content sniff: body trimmed starts with `[` → JSON array, otherwise JSONL.

A malformed row is skipped with a warning, consistent with `load_dataset()`.

---

## 7. Eval Importers (`importers/`)

Two ABCs capturing two different directions.

| | `BaseEvalCaseSource` | `BaseEvalConfigSource` |
|---|---|---|
| **Platform has** | records already produced (traces, runs) | an eval *definition* |
| **Produces** | `list[EvalCase]` (ready to **score**) | an `EvalConfig` (ready to **run**) |
| **You then** | `evaluate_cases(cases, metrics, sinks)` | `run_config(cfg)` |
| **Direction** | import *outputs* | import *the definition* |

```python
class BaseEvalCaseSource(ABC):
    name: str
    @abstractmethod
    async def fetch(self, ref: ResourceRef) -> list[EvalCase]: ...
    async def close(self) -> None: ...

class BaseEvalConfigSource(ABC):
    name: str
    @abstractmethod
    async def fetch(self, ref: ResourceRef) -> "EvalConfig": ...
    async def close(self) -> None: ...
```

### Migrating existing `sources/`

`LangfuseSource` → `LangfuseEvalCaseSource(BaseEvalCaseSource)`.
`OTELSource` → `OTELEvalCaseSource(BaseEvalCaseSource)`.
Original import paths (`harness_evals.sources.langfuse`, `harness_evals.sources.otel`)
kept as aliases for backwards compatibility. `from_trace` / `from_spans` helpers
remain; `fetch(ref)` is the new uniform entry point.

### `BaseEvalConfigSource` — translation contract

Translation is best-effort and explicit about gaps. A platform metric with no
catalog equivalent raises `UnmappedMetricError("<platform_metric>", suggestions=[...])`
listing the closest catalog kinds — never silently dropped.

---

## 8. Targets — the System Under Test

```python
class BaseTarget(ABC):
    @abstractmethod
    async def ainvoke(self, golden: Golden) -> EvalCase: ...
    async def close(self) -> None: ...    # resource cleanup (HTTP sessions, token caches)

    async def __aenter__(self): return self
    async def __aexit__(self, *_): await self.close()
```

`ainvoke` is exactly the `agent_fn` signature `evaluate_dataset()` already takes —
a `BaseTarget` is a drop-in with no changes to the engine.

### 8.1 `PromptTarget`

Framework renders a `PromptTemplate` and calls a `BaseLLM` directly. Grades a
*prompt + model pair* in isolation.

```python
@dataclass
class PromptTarget(BaseTarget):
    prompt: PromptTemplate              # pre-resolved; callers use build_target() to resolve refs
    model: BaseLLM

    async def ainvoke(self, golden: Golden) -> EvalCase:
        # Golden.input may be str | dict | list; normalise to str for template rendering
        input_str = golden.input if isinstance(golden.input, str) else json.dumps(golden.input)
        rendered = self.prompt.render(input=input_str, **(golden.metadata or {}))
        t0 = perf_counter()
        out = await self.model.generate(rendered)
        ms = (perf_counter() - t0) * 1000
        return EvalCase.from_golden(golden, output=out, latency_ms=ms)
```

**`Golden.input` serialisation contract:** `str` inputs are passed through as-is.
`dict` and `list` inputs are serialised with `json.dumps` before template rendering.
If a caller needs custom serialisation (e.g. a multimodal schema), they should
pre-process `golden.input` or subclass `PromptTarget`.

`build_target()` in `config/runner.py` resolves a ref string to a `PromptTemplate`
via the prompt source registry before constructing `PromptTarget`.

### 8.2 `HttpTarget`

POSTs to a deployed agent. The agent's internals are opaque — grades the *shipped
system* end-to-end.

```python
@dataclass
class HttpTarget(BaseTarget):
    # connection
    url: str
    method: str = "POST"
    auth: AuthConfig = field(default_factory=NoAuth)
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 60.0
    verify_tls: bool = True

    # request shaping
    input_path: str = "$.input"
    body_template: dict | None = None

    # resilience
    retries: int = 2
    backoff_s: float = 0.5

    # response → EvalCase mapping
    output_path: str = "$.output"          # required
    tool_calls_path: str | None = None
    context_path: str | None = None
    messages_path: str | None = None
    token_count_path: str | None = None
    cost_usd_path: str | None = None
    retry_count_path: str | None = None
    confidence_path: str | None = None
    latency_ms_path: str | None = None     # None = measure wall-clock
```

All JSONPath extraction uses `jsonpath-ng` (existing core dependency).

**Transport failure:** after `retries` attempts, raise `TargetInvocationError`.
`evaluate_dataset()` catches target exceptions so the run continues and marks the
row with `metadata["target_error"]` on its failed scores.

**Non-JSON response:** if `output_path == "$"` and `Content-Type` is `text/*`, the
raw body becomes `output`.

### 8.3 `AuthConfig` — v1 scope

Auth is a first-class config object, not smuggled through `headers`. All
secret-bearing fields accept `${VAR}` interpolation resolved at config-load time.

```python
class AuthConfig(ABC):
    @abstractmethod
    def apply(self, request: RequestParts) -> RequestParts: ...

@dataclass
class NoAuth(AuthConfig): ...
@dataclass
class BearerAuth(AuthConfig):   token: str
@dataclass
class ApiKeyAuth(AuthConfig):   key: str; header: str = "X-API-Key"; in_: str = "header"
@dataclass
class BasicAuth(AuthConfig):    username: str; password: str
```

**Deferred to v2:** `OAuthClientCredentials`, `MtlsAuth`. The `AuthConfig` ABC is
the extension point — adding them later requires no changes to `HttpTarget`.

---

## 9. Config Schema & Runner

### YAML format

```yaml
name: nightly-regression

dataset: ./goldens.jsonl                    # bare path → local source
# dataset: langfuse://datasets/support-goldens@3   # URI shorthand
# dataset: {source: langfuse, id: support-goldens, version: 3}  # typed block

target:
  type: prompt
  prompt: ./prompts/support-bot.txt         # local file
  model: {provider: openai, name: gpt-4o}
# --- OR ---
# target:
#   type: http
#   url: http://localhost:8080/run
#   output_path: $.answer

metrics:
  - exact_match
  - {kind: geval, threshold: 0.7, params: {criteria: "Correct and helpful?"}}

sinks: [stdout, {type: json, path: results.jsonl}]

baseline: {store: json, path: .evals/baseline.json}

plugins: [acme_evals.adapters]    # optional explicit module imports
```

### `EvalConfig` dataclass

```python
@dataclass
class EvalConfig:
    name: str
    dataset: ResourceRef
    target: TargetSpec
    metrics: list[MetricSpec]
    sinks: list[SinkSpec] = field(default_factory=lambda: [SinkSpec("stdout")])
    baseline: BaselineSpec | None = None
    plugins: list[str] = field(default_factory=list)
```

### `run_config()`

```python
def run_config(cfg: EvalConfig) -> list[list[Score]]:
    """Synchronous entry point. Wraps async execution via _run_async()."""
    load_plugins(cfg.plugins)
    return _run_async(_run_config_async(cfg))

async def _run_config_async(cfg: EvalConfig) -> list[list[Score]]:
    goldens  = await dataset_source(cfg.dataset.source).fetch(cfg.dataset)
    target   = await build_target(cfg.target)   # resolves prompt ref if PromptTarget
    metrics  = [build_metric(m) for m in cfg.metrics]
    sinks    = [build_sink(s) for s in cfg.sinks]
    scores   = await evaluate_dataset(goldens, target.ainvoke, metrics=metrics, sinks=sinks)
    if cfg.baseline:
        gate_against_baseline(scores, cfg.baseline)
    return scores
```

`run_config()` is synchronous (usable from scripts and CLI). `_run_async()` from
`_async_compat.py` handles active event loops — works inside Jupyter notebooks and
`pytest-asyncio` with `asyncio_mode = "auto"`.

`build_metric` resolves `kind` via the merged `catalog()` (built-ins + plugin
registrations), applies `threshold` and `params`. Unknown kind →
`UnknownMetricError` listing valid kinds.

### `gate_against_baseline()` contract

```python
def gate_against_baseline(
    scores: list[list[Score]],
    spec: BaselineSpec,
) -> None:
    """Compare current scores against stored baseline. Raises BaselineRegressionError
    if any metric regresses beyond spec.tolerance (default 0.05).

    BaselineSpec fields:
        store:     str   — baseline store type key (default "json")
        path:      str   — store-specific location (file path, S3 URI, etc.)
        tolerance: float — allowed regression before raising (default 0.05)
        run_id:    str | None — baseline run to compare against (None = latest)
    """
```

`BaselineRegressionError` lists each regressed metric with baseline vs. current
value and delta — the same output as `BaselineResult.summary()` from
`baseline/compare.py`. The CLI `run` command catches this and exits non-zero.
`--update-baseline` saves the current run as the new baseline instead of gating.

---

## 10. CLI

```
harness-evals run <config.yaml> [--baseline] [--fail-under 0.8] [--update-baseline]
harness-evals import <eval-config-ref> [-o out.eval.yaml]
harness-evals list-metrics
harness-evals discover [path] [--glob "**/*.eval.yaml"]
```

- `run` exits non-zero if any metric fails or score falls under `--fail-under` — CI gate.
- `discover` default globs: `**/*.eval.yaml` and `**/eval_*.py`. `.py` files are
  imported; module-level `run_eval()` calls self-execute. Respects a
  `HARNESS_EVALS_IGNORE` gitignore-style file.
- `import` writes a translated `EvalConfig` to YAML — auditable and editable before
  running.
- Registered via `[tool.poetry.scripts]` as `harness-evals`.

---

## 11. Code-First `run_eval()` One-Liner

```python
from harness_evals import run_eval
from harness_evals.metrics import ExactMatchMetric, GEvalMetric
from harness_evals.targets import PromptTarget
from harness_evals.llm.openai import OpenAILLM

run_eval(
    "support-bot",
    data="./goldens.jsonl",           # ref string, ResourceRef, or list[Golden]
    target=PromptTarget(
        prompt="./prompts/support-bot.txt",
        model=OpenAILLM("gpt-4o"),
    ),
    metrics=[ExactMatchMetric(), GEvalMetric(criteria="Correct and helpful?")],
)
```

`run_eval()` and `run_config()` both funnel into `evaluate_dataset()`. `data` accepts a
ref string, a `ResourceRef`, or a literal `list[Golden]`. `target` accepts any
`BaseTarget` or a plain callable `(Golden) -> EvalCase` (sync or async).

---

## 12. Harness-Specific Adapters (Reference Implementation)

The following adapters ship in the `harness-evals` repo behind the `[harness]`
optional extra. They are reference implementations of the vendor-neutral ABCs
above — no different in status from `LangfuseDatasetSource` or `OTELEvalCaseSource`.

**Requires:** `pip install harness-evals[harness]`
**Env vars:** `HARNESS_AI_SERVICE_URL`, `HARNESS_AI_SERVICE_SECRET`

| Class | Family | What it does |
|-------|--------|-------------|
| `HarnessDatasetSource` | `BaseDatasetSource` | Fetches a dataset by ID + version from Harness registry |
| `HarnessPromptSource` | `BasePromptSource` | Fetches a prompt by ID + version from Harness registry |
| `HarnessEvalConfigSource` | `BaseEvalConfigSource` | Translates a Harness eval definition → `EvalConfig` |
| `HarnessAILLM` | `BaseLLM` | Routes LLM calls through Harness AI gateway (already exists in `llm/harness_ai.py`) |

`HarnessEvalConfigSource.fetch()` translates best-effort. Unmapped metrics raise
`UnmappedMetricError` listing catalog alternatives.

### Harness URI examples

```yaml
dataset: harness://datasets/support-goldens@3
target:
  type: prompt
  prompt: harness://prompts/support-bot@5
  model: {provider: openai, name: gpt-4o}
```

These work identically to `langfuse://` or `./local-file.jsonl` — the `harness`
source name is just a registry key, resolved through the same `resolve()` path.

---

## 13. Local-First Developer Loop (Primary Use Case)

The zero-external-account path requires no platform accounts. Deterministic-only
evals need no extras at all. LLM-judged metrics (`geval`, `rubric_judge`, etc.)
require `pip install harness-evals[llm]` and an API key but no external platform:

```bash
# 1. Author golden dataset
cat > goldens.jsonl << 'EOF'
{"input": "What is your return policy?", "expected": "30-day returns on all items."}
EOF

# 2. Write eval config
cat > my-eval.yaml << 'EOF'
name: support-bot
dataset: ./goldens.jsonl
target:
  type: prompt
  prompt: ./prompts/support-bot.txt
  model: {provider: openai, name: gpt-4o}
metrics:
  - exact_match
  - {kind: geval, threshold: 0.7, params: {criteria: "Correct and helpful?"}}
sinks: [stdout]
baseline: {store: json, path: .evals/baseline.json}
EOF

# 3. Run
harness-evals run my-eval.yaml

# 4. Edit prompt, re-run, diff against baseline
harness-evals run my-eval.yaml --baseline
```

The same loop works with `HttpTarget` pointing at `http://localhost:8080` for
local agent testing — no Harness account required.

---

## 14. Build Order

Each step is an independently shippable PR.

1. **`refs.py`** — `ResourceRef` + `resolve()` + `MissingAdapterError` + tests.
2. **`plugins.py`** — four-family registries + `@register_*` decorators + entry-point
   loader + `load_plugins()`. Extend `catalog()` with plugin hook.
3. **`datasets/`** — migrate `datasets.py` → `datasets/` package **atomically in
   this same PR** (not a later PR). `datasets/__init__.py` must re-export
   `load_dataset`, `save_dataset`, and `Dataset` so existing `from harness_evals.datasets
   import load_dataset` imports continue to work without change. Add
   `BaseDatasetSource`, `LocalDatasetSource` (wraps `load_dataset`),
   `HttpDatasetSource`.
4. **`prompts/`** — `BasePromptSource` + `PromptTemplate` + `LocalPromptSource`,
   `HttpPromptSource`.
5. **`targets/`** — `BaseTarget`, `AuthConfig` (NoAuth + BearerAuth + ApiKeyAuth +
   BasicAuth), `PromptTarget`, `HttpTarget`.
6. **`importers/`** — `BaseEvalCaseSource` + `BaseEvalConfigSource`; reframe
   `sources/` as `BaseEvalCaseSource` impls (`LangfuseEvalCaseSource`,
   `OTELEvalCaseSource`) with aliases; `HarnessEvalConfigSource` behind `[harness]`.
7. **`config/`** — `EvalConfig` schema + `run_config()`; wire to `evaluate_dataset`.
8. **`cli.py`** — `run` / `import` / `list-metrics` / `discover` + poetry script.
9. **`run_eval()`** — one-liner + docs + example `*.eval.yaml` files.
10. **Vendor dataset/prompt adapters** — `LangfuseDatasetSource`,
    `LangfusePromptSource`, `HarnessDatasetSource`, `HarnessPromptSource`.

Steps 1–5 have zero vendor dependencies and can be reviewed/merged independently.
Steps 1–7 deliver the full local-first loop. Steps 8–10 add discoverability and
vendor integrations.

---

## 15. Resolved Design Decisions

| Decision | Resolution | Rationale |
|----------|------------|-----------|
| Vendor isolation | Layered: ABCs are neutral, vendor impls behind optional extras | Core stays clean; Harness is a reference implementation, not a privileged citizen |
| `asyncio.run()` vs `_run_async()` | Use `_run_async()` from `_async_compat.py` | Works in Jupyter and `pytest-asyncio` with `asyncio_mode = "auto"` |
| Prompt placeholder syntax | `{{var}}` only, regex substitution | Prompts contain literal `{` braces (JSON, code); `str.format` would silently mangle them |
| HTTP auth scope (v1) | NoAuth + BearerAuth + ApiKeyAuth + BasicAuth | OAuth/mTLS deferred; `AuthConfig` ABC is the extension point |
| Fail-fast on missing adapter | `MissingAdapterError` at resolve-time | Cryptic `ImportError` at execution time wastes a long eval run |
| HTTP format detection | Auto-detect (Content-Type → content sniff) with `ref.extra["format"]` override | Zero-config for the common case; deterministic escape hatch for misconfigured servers |
| `EvalConfigSource` as full ABC | Full ABC with all four families | Deferring to a single class now means refactoring when a second platform needs it |
| `Dimension` enum | Closed (5 values); external adapters must map into it | Adding a dimension requires an ADR; `CORRECTNESS` is the safe default for external metrics |
| Sink registration | Added to plugin registry | Most natural third-party extension point (Datadog, custom dashboards, etc.) |
| Adapter lifecycle | `async close()` + context-manager on all ABCs | `HttpTarget` (OAuth refresh), source clients (SDK sessions) need deterministic cleanup |
| Target registration | `_TARGETS` registry + `register_target` | Third-party targets (e.g. BrainTrustTarget) must be declarable by name in YAML |
| Metric registration | `_METRICS` registry + `register_metric` | Third-party metrics must be discoverable via `catalog()` and referenceable by `kind:` |
| BaselineStore registration | `_BASELINE_STORES` registry + `register_baseline_store` | Platform-backed baselines (S3, Langfuse) need a registration path; v1 ships JSON only |
| `datasets.py` migration | Atomic with step 3 PR; `__init__.py` re-exports `load_dataset` | Module-to-package rename breaks imports without explicit re-exports |
| `Golden.input` serialisation | Non-string inputs `json.dumps`-serialised before template rendering | `Golden.input: str \| dict \| list`; `{{var}}` rendering requires a string |
| `gate_against_baseline()` contract | Raises `BaselineRegressionError` listing regressed metrics; CLI exits non-zero | CI integration requires a defined failure contract, not just a comparison report |
| "Zero-dependency" scope | Zero *external account* dependency; `[llm]` extra needed for LLM-judged metrics | Deterministic metrics need no extras; LLM-judged metrics need SDK but no platform |

---

## Deferred

- `OAuthClientCredentials`, `MtlsAuth` auth variants (v2 `HttpTarget`)
- HuggingFace dataset source
- Multimodal targets
- Python-ref target (`mypkg.agent:run`) and shell-command target
- Per-row prompt override (prompt-injection-robustness case; `metadata` field allows it, no dedicated support in v1)
- `LangfuseEvalConfigSource` (Langfuse has dataset + prompt management but not eval definitions in the same sense)
