# Decision primitives (TypeSafe) — harness-evals implementation plan

**Status:** Revised after architecture review (see ADR-011 "Revisions from review")
**Date:** 2026-09-19
**Depends on:** [ADR-011](../../adr/011-decision-primitives.md)

---

## Scope and ownership

`harness-evals` owns a vendor-neutral `decision` provider abstraction plus
three primitive metrics (`ChoiceMetric`, `ScoreMetric`, `NoulMetric`) and a
batching sibling to `CompositeMetric` (`DecisionCompositeMetric`). TypeSafe
is the reference implementation of `BaseDecisionProvider`, gated behind an
optional `[decision]` extra — core stays dependency-free, same as every
other LLM-backed category.

This plan does **not** touch `Golden`, `EvalCase`, or `Score` field
definitions (additive `metadata` usage only), and does not modify
`composite.py`/`operators.py`'s existing public contract.

## Deliverables

- `BaseDecisionProvider` + typed question/answer dataclasses
  (`src/harness_evals/decision/`).
- `TypeSafeDecisionProvider` (`typesafe_sdk`-backed), extra `[decision]`.
- `ChoiceMetric`, `ScoreMetric`, `NoulMetric` — catalog kinds
  `decision_choice`, `decision_score`, `decision_noul` — dimension
  `CORRECTNESS` by default, overridable; explicit `mode` kwarg.
- `DecisionCompositeMetric` — batches decision sub-checks sharing a
  `state_field` into one API call; shares fold arithmetic with
  `CompositeMetric` via an extracted `_combine()` helper.
- An efficacy harness comparing decision primitives against equivalent
  LLM-judge metrics on the same labeled cases (accuracy, calibration,
  latency, cost) — this is the actual point of doing this integration, not
  an afterthought, so it gets its own phase and isn't skipped if time is
  short.
- Docs: `README.md`, `docs/metrics-guide.md`, `AGENTS.md` project-structure
  tree, `CHANGELOG.md`, `pyproject.toml` version bump.

## Progress

| PR | Status | Notes |
|---|---|---|
| DP-1 | Done | Provider abstraction + TypeSafe provider, tests green |
| DP-2 | Done | Three primitive metrics, catalog/factory wired, tests/docs/CHANGELOG/version bump done. Live manual-check step (real `TYPESAFE_API_KEY`) still outstanding. |
| DP-3 | Not started | `DecisionCompositeMetric` (independent, can land after DP-2) |
| DP-4 | Not started | Efficacy comparison vs. LLM-judge (needs live `TYPESAFE_API_KEY`) |

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
      criteria: list[str | dict]          # 2-10 ordered levels, index 0 = low end

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
      model: str                          # concrete model TypeSafe used, not the requested alias
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

  Transport/API errors from `a_ask()` **propagate as exceptions** — a
  provider implementation must not catch and convert them into a fabricated
  answer or a zero score. This is load-bearing for DP-2's error-handling
  contract (see below) and is the fix for review concern #8.

- `src/harness_evals/decision/typesafe.py`

  ```python
  class TypeSafeDecisionProvider(BaseDecisionProvider):
      """Requires ``pip install harness-evals[decision]``.

      API key resolution: constructor ``api_key`` > ``TYPESAFE_API_KEY`` env var.
      """

      def __init__(
          self,
          model: str = "jev-latest",
          api_key: str | None = None,
          timeout: float | None = None,
      ) -> None: ...
      async def a_ask(self, state, questions) -> DecisionResponse: ...
  ```

  Internals: lazy-import `typesafe_sdk` inside `__init__` (raise
  `ImportError` with the `pip install harness-evals[decision]` hint if
  missing — mirrors `OpenAILLM`/`AnthropicLLM`), translate our
  `Question`/`Answer` dataclasses to/from `typesafe_sdk.Choice/Score/Noul`
  and the SDK's response objects. Report `input_tokens`/`output_tokens` via
  the existing `llm.usage.record_token_usage()` collector so decision calls
  show up in the same cost/usage accounting as judge calls — reuse, don't
  duplicate, `TokenUsage`. Do **not** wrap the SDK call in a broad
  `try/except` — let `typesafe_sdk`'s own `TypeSafeAPIError`/connection
  errors surface. Rely on the SDK's documented retry policy for 429/529;
  do not add a second retry loop on top of it.

Tests (`tests/decision/test_typesafe_provider.py`):

- Request/response translation for each of the three question types
  (unit test against a stubbed `typesafe_sdk` client — no live network call).
- Missing `TYPESAFE_API_KEY` and missing `typesafe-sdk` package both raise
  clear, actionable errors.
- Mixed-type multi-question request in one call round-trips correctly keyed
  by question id.
- `record_token_usage()` is called with the response's `usage` fields, and
  `DecisionResponse.input_tokens`/`output_tokens` are populated independent
  of whether a `collect_token_usage()` block is active (review concern #7 —
  don't rely solely on the contextvar path).
- A stubbed SDK error (e.g. `TypeSafeAPIError`) propagates out of `a_ask()`
  unchanged — it is not caught and converted into a `DecisionResponse`.

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
print(r.model, r.answers['is_escalation'].noul, r.answers['department'].choice, r.answers['urgency'].score)
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
- `pyproject.toml` — `decision = ["typesafe-sdk>=<lower>,<upper>"]` extra
  (pinned — review concern #10), add to `all`
- `README.md`, `docs/metrics-guide.md`, `AGENTS.md` project tree
- `CHANGELOG.md`

Common constructor shape:

```python
NoulMetric(
    provider: BaseDecisionProvider,
    instructions: str | dict | list,
    criteria: dict[str, str] | None = None,
    state_field: str = "output",           # JSONPath via utils/path.extract_path
    mode: Literal["correctness", "confidence"] | None = None,  # None = infer from expected
    invert: bool = False,                  # Noul only
    threshold: float = 1.0,
    dimension: Dimension = Dimension.CORRECTNESS,
)
```

(`ChoiceMetric`/`ScoreMetric` drop `invert`, take the primitive-appropriate
`criteria` shape per DP-1's `Question` types, and `ScoreMetric.expected` must
be a numeric level, not a legend string — see ADR-011's normalization
section.)

`measure()`/`a_measure()` contract (implements the revised value table from
ADR-011):

- Resolve `state` via `extract_path(eval_case.to_dict(), state_field)`;
  missing state → `Score(value=0.0, reason="Missing state field '<path>'")`
  — this is a data-shape problem, still fine to score as `0.0`/skip, unlike
  a provider transport failure.
- Resolve effective mode: explicit `mode` wins; else infer
  `"correctness"` if `eval_case.expected is not None` else `"confidence"`.
  `mode="correctness"` with `eval_case.expected is None` is an error, not a
  silent fallback (review concern #3) — raise `ValueError` at `measure()`
  time (not construction time, since the same metric instance is reused
  across cases with and without `expected` in a mixed dataset — wait, no:
  the mode contract says an author who forces `mode="correctness"` intends
  *every* case in the run to carry `expected`; treat a violating case as a
  hard `ValueError` surfaced through `evaluate()`'s existing per-case
  exception capture (ADR-004), not a special-cased `Score`).
- Call `provider.a_ask(state, {self.name: question})`, single question per
  call (batching multiple primitives is `DecisionCompositeMetric`'s job —
  DP-3, not this metric). Provider exceptions propagate — do not catch them
  here (review concern #8).
- Apply the value table from ADR-011, including Noul's `expected` type
  validation and `invert`, and Score's `level_norm`.
- `Score.metadata` always carries: the raw typed answer (`choice`/`score`/
  `noul`, `probabilities`, `legend` where applicable), `"confidence"` where
  the primitive provides one, `"mode"`, `"model"` (the concrete
  `DecisionResponse.model`, review concern #6), and `"input_tokens"`/
  `"output_tokens"` (review concern #7 — always set here, independent of the
  contextvar collector).
- Sync `measure()` calls `_run_async(self.a_measure(eval_case))` (same
  pattern as `BaseDecisionProvider.ask()`) — do **not** reimplement a sync
  HTTP path.

Catalog/factory work:

- Export `ChoiceMetric`, `ScoreMetric`, `NoulMetric` from
  `metrics/decision/__init__.py` and `metrics/__init__.py`.
- Register catalog kinds `decision_choice`, `decision_score`,
  `decision_noul` in `_build_registry()` (namespaced per ADR-011's
  consequences section — fixes review concern #11).
- Add `requires_provider: bool` to `CatalogEntry`
  (`src/harness_evals/catalog.py`), computed via
  `_requires_param(cls, "provider")` next to the existing
  `requires_llm`/`requires_embedding` computation. This is not optional
  polish — without it every decision metric is misfiled into the factory's
  zero-dependency ("heuristic") bucket and `build_metric()` raises
  `TypeError` at config-load time for any YAML-driven caller (review concern
  #2). Add a `tests/test_catalog.py` (or extend the existing catalog test)
  assertion that all three decision kinds report `requires_provider=True`
  and are excluded from the heuristic registry in `factory.py`.
- `build_metric()` (`metrics/factory.py`) must accept a `provider` kwarg
  the same way it accepts `llm` for judge metrics, and dispatch decision
  kinds to a path that requires one.

Tests use a fake `BaseDecisionProvider` returning canned `DecisionResponse`s
(no live network calls in the test suite):

- Correctness mode: exact match → `1.0`; mismatch → `0.0` (Choice/Noul);
  partial credit at various `level_norm` distances (Score), including a
  case where the raw response `score` is non-integer (e.g. `1.3`).
- Confidence mode (`expected=None`, or `mode="confidence"` explicit):
  `value` equals the primitive's own confidence/`noul` (and its `invert`ed
  form for Noul); metadata contains full probability distribution, `model`,
  and token counts.
- `mode="correctness"` with `expected=None` raises `ValueError` and that
  exception surfaces through `evaluate()`'s per-case capture rather than
  being swallowed.
- Noul `expected` validation: `True`/`False`/`0`/`1`/`"true"`/`"false"`
  (case-insensitive) accepted; anything else (e.g. `"yes"`, `2`) raises a
  clear validation error at construction time.
- A fake provider that raises propagates the exception out of
  `measure()`/`a_measure()` — asserted explicitly, this is the regression
  test for review concern #8.
- Missing `state_field` → `0.0` with a clear reason, not an exception.
- `Score.value` never leaves `[0, 1]` even with an out-of-range provider
  response (defensive clamp on the *provider response*, not silently
  swallowing bad data as `0.0`).
- Factory/catalog construction round-trip, including the `requires_provider`
  assertion above.

**Automated checks**

```bash
pytest tests/metrics/decision/ -v
pytest tests/metrics/test_factory.py tests/test_catalog.py -v
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

Confirm `score.value` is the live `noul` probability, `score.passed` matches
the 0.5 threshold, and `score.metadata["model"]` is populated.

**Merge gate:** version bump in `pyproject.toml` (minor — new built-in
metrics) with matching `CHANGELOG.md` entry per `AGENTS.md`; `[decision]`
extra installs cleanly in a fresh venv; catalog discovers all three kinds
with `requires_provider=True`.

---

### DP-3 — `DecisionCompositeMetric` (independent, non-blocking)

**Goal:** let multiple decision sub-checks against the same `state_field`
collapse into one `a_ask()` call, per ADR-011's revised batching section,
without touching `composite.py`'s existing sync-only public contract.

Add/update:

- `src/harness_evals/metrics/composite/_combine.py` (new) — extract the
  weighted-sum/`effective_weights` fold currently inlined in
  `CompositeMetric.measure()` into a standalone
  `fold_sub_scores(sub_scores_config, results) -> tuple[float, dict]`
  function. Update `composite.py` to call it — **zero behavior change**,
  this is a pure refactor guarded by the existing `tests/metrics/composite/`
  suite passing unchanged.
- `src/harness_evals/metrics/composite/decision_composite.py` (new) —
  `DecisionCompositeMetric(provider: BaseDecisionProvider, sub_scores: list[dict], threshold: float = 0.85, **kwargs)`.
  Each `sub_scores[i]` has `name`, `weight`, `state_field`, a `question`
  (one of DP-1's `Question` dataclasses), optional `expected_field` (path
  into the eval case for the correctness-mode comparison), `skip_when_missing`.
  `a_measure()`:
  1. Skip sub-checks whose resolved `state_field` is `None` (per
     `skip_when_missing`) before any network call.
  2. Group remaining sub-checks by resolved `state_field` value; one
     `provider.a_ask(state, {name: question, ...})` per group.
  3. A group's `a_ask()` failure marks every sub-check *in that group*
     `status="error"` (excluded from `active_weight_sum`) — does not raise
     out of `a_measure()` and does not zero unrelated groups' sub-checks.
     This is different from DP-2's standalone metrics, where a provider
     exception propagates: `DecisionCompositeMetric` is explicitly a
     multi-check aggregator, so partial failure is the expected shape (same
     as `composite.py`'s existing per-sub `try/except`), whereas a
     standalone `NoulMetric` has nothing to aggregate around a failure.
  4. Apply each sub-check's value using DP-2's per-primitive formulas, then
     fold through the shared `fold_sub_scores()` from `_combine.py`.
  `measure()` (sync) = `_run_async(self.a_measure(eval_case))`.
- `tests/metrics/composite/test_decision_composite.py` (new)
- `tests/metrics/composite/test_composite.py` — add one test asserting the
  `_combine.py` refactor produced byte-identical scores for a handful of
  existing fixtures.

Tests:

- Two decision sub-checks sharing one `state_field` produce exactly one
  fake-provider call with two questions (assert call count + keys).
- Two decision sub-checks with *different* `state_field`s produce two calls.
- Mixed weights combine correctly (weighted sum matches hand-computed
  expectation) — reuse a fixture with known arithmetic from
  `tests/metrics/composite/test_composite.py` translated to decision
  sub-checks.
- One group's provider call raising marks only that group's sub-checks
  `status="error"`; a sibling group's successful sub-checks still
  contribute to the weighted sum.
- `skip_when_missing` skips before calling the provider (assert zero calls
  for an all-skipped case).

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

**Dependency:** ships after DP-2 so `decision_composite.py` can reference
the same `Question`/`Answer` types and per-primitive value formulas.

---

### DP-4 — Efficacy comparison vs. LLM-judge metrics

**Goal:** the actual point of this integration — not "does it run," but
"is a calibrated typed decision measurably better than an LLM-judge doing
the same job," on axes that matter for an eval framework: label accuracy
against a golden, calibration (does confidence track correctness), latency,
and cost. Requires a live `TYPESAFE_API_KEY` and a live judge-model key
(this repo already has `ANTHROPIC_API_KEY` available; `AnthropicLLM` +
`GEvalMetric`/`RubricJudgeMetric` is the comparison baseline).

Add:

- `examples/decision_vs_llm_judge/` — not shipped as a metric or a test
  under `src/`, this is a standalone comparison harness (README precedent:
  `examples/integrations/`).
  - `dataset.py` — a small labeled dataset (20-50 hand-labeled cases) for
    one task with an unambiguous ground truth. Recommended: **bug-report
    triage** (mirrors the TypeSafe docs' own worked examples almost
    exactly, so the primitive's prompt/criteria are apples-to-apples with
    published expectations): three labeled dimensions per case —
    `department` (Choice: returns/shipping/billing), `severity` (Score:
    3-level rubric), `is_escalation` (Noul: yes/no). Each case: real-ish
    support-ticket text + a hand-assigned golden label per dimension.
  - `run_comparison.py` — for each case, dimension, and each of the two
    approaches:
    - **Decision primitive**: `ChoiceMetric`/`ScoreMetric`/`NoulMetric`
      against `TypeSafeDecisionProvider`, `mode="correctness"`.
    - **LLM judge**: an equivalent `GEvalMetric`/custom prompt against
      `AnthropicLLM` asked to classify/rate the same dimension and
      self-report a confidence — same task, same input, deliberately
      comparable criteria text.
    Records per (case, dimension, approach): predicted value, correct
    (bool), reported confidence/probability, latency (`time.perf_counter`
    around the call), and token usage/cost (`Score.metadata`).
  - `report.py` — aggregates into a comparison table per dimension:
    accuracy, Brier score (reusing the pure scoring logic already in
    `metrics/reliability/brier_score.py` if it's usable standalone —
    otherwise reimplement `mean((confidence - correct)**2)` inline; do not
    import a judged metric for this, it's arithmetic on already-collected
    predictions), mean latency, mean cost, and — for Noul/Choice — a
    reliability diagram bucket table (predicted-confidence bucket vs.
    observed accuracy in that bucket) since "calibrated" is the headline
    claim being tested, not just "accurate."
- `docs/designs/2026-09-19-decision-primitives/efficacy-results.md` — the
  actual numbers once DP-4 runs, plus a one-paragraph verdict. This file
  doesn't exist yet because the run hasn't happened; write it from the real
  `report.py` output, not from expected/hoped-for numbers.

What this deliberately does **not** do:

- No claim of "decision primitives are always better." The comparison is
  scoped to one task family (short-text triage/classification/rating) where
  TypeSafe's docs claim strength; a different task shape (e.g. long-context
  summarization judging) is out of scope for this pass and should be called
  out as such in `efficacy-results.md` rather than generalized from.
- No production wiring based on the result. This phase produces evidence;
  whether to recommend decision primitives as the *default* for
  classification-shaped metrics in `metrics-guide.md` is a follow-up
  decision made after reading the numbers, not before.

**Automated checks:** none — this is an experiment harness, not shipped
library code. `ruff check examples/decision_vs_llm_judge/` for lint hygiene
only.

**Manual check:** run `python examples/decision_vs_llm_judge/run_comparison.py`
with both `TYPESAFE_API_KEY` and `ANTHROPIC_API_KEY` set, confirm
`report.py`'s table populates for both approaches on all three dimensions,
and read the numbers before writing any conclusion.

**Dependency:** DP-1 and DP-2 (uses `TypeSafeDecisionProvider` and the three
metrics directly). Independent of DP-3.

---

## Milestone mapping

| Milestone | Required PRs | Exit condition |
|---|---|---|
| M0 — provider contract frozen | DP-1 | `BaseDecisionProvider`/`TypeSafeDecisionProvider` tested against fakes and once against the live API |
| M1 — releasable metrics | DP-2 | Published package exposes `decision_choice`/`decision_score`/`decision_noul` catalog kinds with `requires_provider=True`, factory can construct them |
| M2 — batched composites | DP-3 | One `DecisionCompositeMetric` with N decision sub-checks on one state fires one API call, partial-failure semantics verified |
| M3 — evidence | DP-4 | `efficacy-results.md` published with real accuracy/calibration/latency/cost numbers on at least one task family |

## Open questions to resolve before DP-2 merges

- Exact `typesafe-sdk` version pin (needs checking against what's actually
  published/available once implementation starts).
- Whether `TypeSafeDecisionProvider` should also expose a `models()`/model
  metadata pass-through (the SDK's `ModelCard`/`Models` interfaces) — out of
  scope for v1 unless a concrete need shows up.

## Downstream handoff

Provide consumers with:

- Published package version containing `decision/` and `metrics/decision/`.
- Import paths for `BaseDecisionProvider`, `TypeSafeDecisionProvider`,
  `ChoiceMetric`/`ScoreMetric`/`NoulMetric`, `DecisionCompositeMetric`.
- Exact `Score.metadata` schema per primitive (correctness vs. confidence
  mode, including `model`/token fields).
- Fake-provider fixtures mirroring the ones in `tests/decision/` and
  `tests/metrics/decision/` for downstream contract tests.
- `efficacy-results.md` findings, so downstream consumers know which task
  shapes this was actually validated against before reaching for it as a
  default.
