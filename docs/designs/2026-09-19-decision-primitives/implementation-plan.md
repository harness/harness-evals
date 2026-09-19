# Decision primitives (TypeSafe) — harness-evals implementation plan

**Status:** Draft
**Date:** 2026-09-19
**Depends on:** [ADR-011](../../adr/011-decision-primitives.md)

---

## Scope and ownership

`harness-evals` owns a vendor-neutral `decision` provider abstraction plus three
primitive metrics (`ChoiceMetric`, `ScoreMetric`, `NoulMetric`) and a batching
extension to `CompositeMetric`. TypeSafe is the reference implementation of
`BaseDecisionProvider`, gated behind an optional `[decision]` extra — core
stays dependency-free, same as every other LLM-backed category.

This plan does **not** touch `Golden`, `EvalCase`, or `Score` field
definitions (additive `metadata` usage only), and does not touch any existing
metric outside `metrics/composite/`.

## Deliverables

- `BaseDecisionProvider` + typed question/answer dataclasses
  (`src/harness_evals/decision/`).
- `TypeSafeDecisionProvider` (`typesafe_sdk`-backed), extra `[decision]`.
- `ChoiceMetric`, `ScoreMetric`, `NoulMetric` — catalog kinds `choice`,
  `score_decision`, `noul` — dimension `CORRECTNESS` by default, overridable.
- `CompositeMetric` async `"decision"` operator with per-`state_field`
  batching of multiple decision sub-checks into one API call.
- Docs: `README.md`, `docs/metrics-guide.md`, `AGENTS.md` project-structure
  tree, `CHANGELOG.md`, `pyproject.toml` version bump.

## Progress

| PR | Status | Notes |
|---|---|---|
| DP-1 | Not started | Provider abstraction + TypeSafe provider, no metrics yet |
| DP-2 | Not started | Three primitive metrics, catalog/factory, release |
| DP-3 | Not started | `CompositeMetric` async batching (independent, can land after DP-2) |

---

### DP-1 — `BaseDecisionProvider` and `TypeSafeDecisionProvider`

**Goal:** lock the provider contract independently of any metric, so DP-2 can
be tested against a fake provider with canned responses.

Add:

- `src/harness_evals/decision/__init__.py`
- `src/harness_evals/decision/types.py`

  ```python
  @dataclass
  class ChoiceQuestion:
      instructions: str | dict | list
      criteria: dict[str, str | dict | list | None]

  @dataclass
  class ScoreQuestion:
      instructions: str | dict | list
      criteria: list[str | dict]          # 2-10 ordered levels

  @dataclass
  class NoulQuestion:
      instructions: str | dict | list
      criteria: dict[str, str] | None = None   # optional {"true": ..., "false": ...}

  Question = ChoiceQuestion | ScoreQuestion | NoulQuestion

  @dataclass
  class ChoiceAnswer:
      choice: str
      confidence: float
      probabilities: dict[str, float]

  @dataclass
  class ScoreAnswer:
      score: float                        # probability-weighted, may be non-integer
      confidence: float
      legend: dict[int, str]
      probabilities: dict[int, float]

  @dataclass
  class NoulAnswer:
      noul: float                         # 0-1, probability of "yes"

  Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer

  @dataclass
  class DecisionResponse:
      model: str
      answers: dict[str, Answer]
      input_tokens: int | None = None
      output_tokens: int | None = None
  ```

- `src/harness_evals/decision/base.py`

  ```python
  class BaseDecisionProvider(ABC):
      @abstractmethod
      async def a_ask(
          self, state: str | dict | list, questions: dict[str, Question]
      ) -> DecisionResponse: ...

      def ask(self, state, questions) -> DecisionResponse:
          return _run_async(self.a_ask(state, questions))
  ```

- `src/harness_evals/decision/typesafe.py`

  ```python
  class TypeSafeDecisionProvider(BaseDecisionProvider):
      """Requires ``pip install harness-evals[decision]``.

      API key resolution: constructor ``api_key`` > ``TYPESAFE_API_KEY`` env var.
      """

      def __init__(self, model: str = "jev-latest", api_key: str | None = None) -> None: ...
      async def a_ask(self, state, questions) -> DecisionResponse: ...
  ```

  Internals: lazy-import `typesafe_sdk` inside `__init__` (raise
  `ImportError` with the `pip install harness-evals[decision]` hint if
  missing — mirrors `OpenAILLM`/`AnthropicLLM`), translate our
  `Question`/`Answer` dataclasses to/from `typesafe_sdk.Choice/Score/Noul`
  and the SDK's response objects. Report `input_tokens`/`output_tokens` via
  the existing `llm.usage.record_token_usage()` collector so decision calls
  show up in the same cost/usage accounting as judge calls — reuse, don't
  duplicate, `TokenUsage`.

Tests (`tests/decision/test_typesafe_provider.py`):

- Request/response translation for each of the three question types
  (unit test against a stubbed `typesafe_sdk` client — no live network call).
- Missing `TYPESAFE_API_KEY` and missing `typesafe-sdk` package both raise
  clear, actionable errors.
- Mixed-type multi-question request in one call round-trips correctly keyed
  by question id.
- `record_token_usage()` is called with the response's `usage` fields.

**Manual check (uses the user's live TypeSafe API key):**

```bash
export TYPESAFE_API_KEY=...
python -c "
from harness_evals.decision.typesafe import TypeSafeDecisionProvider
from harness_evals.decision.types import NoulQuestion, ChoiceQuestion, ScoreQuestion

p = TypeSafeDecisionProvider()
r = p.ask(
    state='I have asked three times now. Can I please just talk to a real person?',
    questions={
        'is_escalation': NoulQuestion(instructions='Is the customer asking for a human agent?'),
        'department': ChoiceQuestion(instructions='Which team should handle this?', criteria={'returns': None, 'billing': None}),
        'urgency': ScoreQuestion(instructions='How urgent is this?', criteria=['low', 'medium', 'high']),
    },
)
print(r.answers['is_escalation'].noul, r.answers['department'].choice, r.answers['urgency'].score)
"
```

Confirm the three answers come back typed and non-empty, and that
`r.input_tokens`/`r.output_tokens` are populated from the real API response.

**Merge gate:** no metric-layer changes; `pytest tests/decision/ -v` green;
`ruff check`/`format --check` clean on the new package.

---

### DP-2 — `ChoiceMetric`, `ScoreMetric`, `NoulMetric`

**Goal:** expose the three primitives as standalone, catalog-registered
metrics per `AGENTS.md`'s "one metric = one file" convention.

Add/update:

- `src/harness_evals/metrics/decision/__init__.py`
- `src/harness_evals/metrics/decision/choice.py`
- `src/harness_evals/metrics/decision/score.py`
- `src/harness_evals/metrics/decision/noul.py`
- `src/harness_evals/metrics/__init__.py`
- `src/harness_evals/catalog.py`
- `tests/metrics/decision/test_choice.py`
- `tests/metrics/decision/test_score.py`
- `tests/metrics/decision/test_noul.py`
- `tests/metrics/test_factory.py` (construction via `build_metric`)
- `pyproject.toml` — `decision = ["typesafe-sdk"]` extra, add to `all`
- `README.md`, `docs/metrics-guide.md`, `AGENTS.md` project tree
- `CHANGELOG.md`

Common constructor shape across all three:

```python
ChoiceMetric(
    provider: BaseDecisionProvider,
    instructions: str | dict | list,
    criteria: dict[str, str | dict | list | None],
    state_field: str = "output",     # JSONPath via utils/path.extract_path
    threshold: float = 1.0,
    dimension: Dimension = Dimension.CORRECTNESS,
)
```

(`ScoreMetric`/`NoulMetric` take the primitive-appropriate `criteria` shape
per DP-1's `Question` types.)

`measure()`/`a_measure()` contract (implements the value table from
ADR-011):

- Resolve `state` via `extract_path(eval_case.to_dict(), state_field)`;
  missing state → `Score(value=0.0, reason="Missing state field '<path>'")`.
- Call `provider.a_ask(state, {self.name: question})`, single question per
  call (batching multiple primitives is `CompositeMetric`'s job — DP-3, not
  this metric).
- `eval_case.expected is not None` → correctness mode (match/closeness
  table in ADR-011); `eval_case.expected is None` → confidence-passthrough
  mode.
- `Score.metadata` always carries the raw typed answer: `{"choice": ...,
  "probabilities": {...}}` / `{"score": ..., "legend": {...}, "probabilities":
  {...}}` / `{"noul": ...}`, plus `"confidence"` where the primitive provides
  one, plus `"mode": "correctness" | "confidence"`.
- Sync `measure()` calls `_run_async(self.a_measure(eval_case))` (same
  pattern as `ReliabilityMetric`/other async-first metrics) — do **not**
  reimplement a sync HTTP path.

Catalog/factory work:

- Export `ChoiceMetric`, `ScoreMetric`, `NoulMetric` from
  `metrics/decision/__init__.py` and `metrics/__init__.py`.
- Register catalog kinds `choice`, `score_decision`, `noul` in
  `_build_registry()`. **Confirm `score_decision` (not `score`)** — the
  catalog already reserves bare names for existing concepts; verify no
  collision before merge (grep `catalog.py` for `"score"`).
- `requires_llm` doesn't apply cleanly here (provider isn't a `BaseLLM`) —
  add a `requires_decision_provider=True` catalog flag, or reuse
  `requires_llm` if the factory only checks "needs an external key" — decide
  during implementation by reading `metrics/factory.py`'s `allow_code_loading`
  guard and matching its intent rather than introducing a parallel flag if
  one isn't needed.
- `build_metric()` must accept a `provider` kwarg the same way it accepts
  `llm` for judge metrics.

Tests use a fake `BaseDecisionProvider` returning canned `DecisionResponse`s
(no live network calls in the test suite):

- Correctness mode: exact match → `1.0`; mismatch → `0.0` (Choice/Noul);
  partial credit at various normalized distances (Score).
- Confidence mode (`expected=None`): `value` equals the primitive's own
  confidence/`noul`; metadata contains full probability distribution.
- Missing `state_field` → `0.0` with a clear reason, not an exception.
- `Score.value` never leaves `[0, 1]` even with an out-of-range or malformed
  provider response (defensive clamp, matching `Score.clamped` usage
  elsewhere).
- Factory/catalog construction round-trip.

**Automated checks**

```bash
pytest tests/metrics/decision/ -v
pytest tests/metrics/test_factory.py -v
ruff check src/harness_evals/metrics/decision src/harness_evals/decision tests/metrics/decision tests/decision
ruff format --check src/harness_evals/metrics/decision src/harness_evals/decision tests/metrics/decision tests/decision
pytest tests/ -v
```

**Manual check (live key)**

```python
from harness_evals.core.eval_case import EvalCase
from harness_evals.decision.typesafe import TypeSafeDecisionProvider
from harness_evals.metrics.decision.noul import NoulMetric

provider = TypeSafeDecisionProvider()
metric = NoulMetric(
    provider=provider,
    instructions="Is the customer asking for a human agent?",
    threshold=0.5,
)
ec = EvalCase(input="...", output="I have asked three times now. Can I please just talk to a real person?")
score = metric.measure(ec)
print(score.value, score.metadata)
```

Confirm `score.value` is the live `noul` probability and `score.passed`
matches the 0.5 threshold.

**Merge gate:** version bump in `pyproject.toml` (minor — new built-in
metrics) with matching `CHANGELOG.md` entry per `AGENTS.md`; `[decision]`
extra installs cleanly in a fresh venv; catalog discovers all three kinds.

---

### DP-3 — `CompositeMetric` async batching (independent, non-blocking)

**Goal:** let multiple decision sub-checks against the same `state_field`
collapse into one `a_ask()` call, per ADR-011's batching section, without
touching existing sync-only composite behavior.

Update:

- `src/harness_evals/metrics/composite/operators.py` — add
  `ASYNC_OPERATORS: dict[str, AsyncOperatorFn]` alongside `OPERATORS`, with a
  `"decision"` entry. An async operator's `config` names `provider` (a
  `BaseDecisionProvider` instance, passed through by the caller — not
  serialized), `question_type` (`choice|score|noul`), `instructions`,
  `criteria`, `state_field`, and — when scoring against a golden —
  `expected_field`. Reuses `extract_path` exactly like the sync operators.
- `src/harness_evals/metrics/composite/composite.py`:
  - Extract the shared weighted-sum/`effective_weights` fold out of
    `measure()` into a private `_combine(details, sub_scores) -> float`
    helper so both `measure()` and the new `a_measure()` call one
    implementation — no duplicated arithmetic.
  - Add `async def a_measure(self, eval_case) -> Score`: partitions
    `sub_scores` by whether `check.type` is in `OPERATORS` (run synchronously,
    unchanged) or `ASYNC_OPERATORS` (grouped by `state_field`, one
    `provider.a_ask()` per distinct resolved state value with all its
    questions keyed by sub-check `name`), then folds both result sets through
    `_combine`.
  - `measure()` (sync) stays byte-for-byte behavior-compatible for composites
    with **no** decision sub-checks; when decision sub-checks are present, it
    delegates to `_run_async(self.a_measure(eval_case))`.
- `tests/metrics/composite/test_composite_decision.py` (new)

Tests:

- Two decision sub-checks sharing one `state_field` produce exactly one fake-
  provider call with two questions (assert call count + keys).
- Two decision sub-checks with *different* `state_field`s produce two calls.
- Mixed sync + decision sub-checks combine correctly (weighted sum matches
  hand computed expectation).
- Existing all-sync `CompositeMetric` test suite (`tests/metrics/composite/`)
  passes unchanged — regression guard that `_combine` extraction didn't
  alter arithmetic.
- `skip_when_missing` still works for a decision sub-check whose
  `state_field` resolves to `None` (skip before ever calling the provider).

**Automated checks**

```bash
pytest tests/metrics/composite/ -v
ruff check src/harness_evals/metrics/composite tests/metrics/composite
ruff format --check src/harness_evals/metrics/composite tests/metrics/composite
pytest tests/ -v
```

**Manual check (live key)**

Reproduce the docs' resume-scoring composite example — one `state`, four
`Score` questions (`python_depth`, `team_leadership`, `system_design`,
`generalist`) as decision sub-checks with weights matching the
`ic_score`/`em_score` example — verify one API call fires (log/print
`provider.a_ask` call count) and the final weighted score matches manual
arithmetic on the returned raw scores.

**Dependency:** none on DP-1/DP-2 being released first for *development*,
but ships after DP-2 so the `"decision"` operator config can reference the
same `Question`/`Answer` types.

---

## Milestone mapping

| Milestone | Required PRs | Exit condition |
|---|---|---|
| M0 — provider contract frozen | DP-1 | `BaseDecisionProvider`/`TypeSafeDecisionProvider` tested against fakes and once against the live API |
| M1 — releasable metrics | DP-2 | Published package exposes `choice`/`score_decision`/`noul` catalog kinds, factory can construct them |
| M2 — batched composites | DP-3 | One `CompositeMetric` with N decision sub-checks on one state fires one API call |

## Open questions to resolve before DP-2 merges

- Final catalog key for the Score primitive metric (`score_decision` is a
  placeholder — pick whatever avoids collision after reading
  `catalog.py`).
- Whether `requires_llm`-style factory gating needs a new flag or can be
  generalized to "requires external credentials" so decision metrics don't
  bolt on a parallel special case.
- Whether `TypeSafeDecisionProvider` should also expose a `models()`/model
  metadata pass-through (the SDK's `ModelCard`/`Models` interfaces) — out of
  scope for v1 unless a concrete need shows up.

## Downstream handoff

Provide consumers with:

- Published package version containing `decision/` and `metrics/decision/`.
- Import paths for `BaseDecisionProvider`, `TypeSafeDecisionProvider`,
  `ChoiceMetric`/`ScoreMetric`/`NoulMetric`.
- Exact `Score.metadata` schema per primitive (correctness vs. confidence
  mode).
- Fake-provider fixtures mirroring the ones in `tests/decision/` and
  `tests/metrics/decision/` for downstream contract tests.
