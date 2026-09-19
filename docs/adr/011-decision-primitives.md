# ADR-011: Decision primitives (Choice/Score/Noul) as a new metric category, not LLM-judge metrics

## Status

Accepted (revised after architecture review — see "Revisions from review" below)

## Context

Every judged metric `harness-evals` ships today (`llm_judge/`, most of `safety/`,
parts of `rag/`) follows the same shape: prompt a general-purpose chat LLM
(`BaseLLM.generate_json()`), ask it to reason in free text, and parse a JSON
object with a `reasoning` string and a `score`/`verdict` field out of that
text. The score's reliability rides entirely on prompt engineering; there is
no calibration guarantee, and "confidence" doesn't exist as a first-class
signal — at best a metric fabricates one by asking the judge to self-report
it, which the docs for TypeSafe explicitly call out as not meaningfully
calibrated.

TypeSafe (`docs.typesafe.ai`) ships a different kind of model — "System One"
— that answers one of three fixed question shapes and returns a **typed,
calibrated** answer, not prose:

- **Choice** — pick one of N named categories. Returns the top choice, a
  full probability distribution over all categories, and a `confidence`
  (0-1) computed from the shape of that distribution.
- **Score** — rate against an ordered rubric of 2-10 levels. Returns a
  probability-weighted mean across levels (so it can fall *between* levels),
  the level `legend`, `probabilities` per level, and `confidence`.
- **Noul** — a calibrated yes/no. Returns a single `noul` value in [0, 1]:
  the model's estimated probability the answer is "yes." No separate
  confidence field — 0.5 *is* maximal uncertainty here, not "medium."

All three are served over one endpoint (`POST
https://api.typesafe.ai/v1/systemone`), and multiple questions of any mix of
these three types can be sent in a single request against one `state` blob
— the docs' "ask independent questions together" pattern — with the API
evaluating them in parallel server-side rather than requiring N round trips.

The corollary pattern ("Composite scoring," `patterns/composite-scoring`) is
explicit that these primitives are meant to be combined *deterministically in
your own code* — normalize each raw score to [0, 1], weight them, sum them —
rather than asking one model to produce one holistic number. That is close
to, but not identical to, what `CompositeMetric`
(`metrics/composite/composite.py`) already does for this SDK: weighted
sub-scores with deterministic combination, driven by an `OPERATORS` registry
of pure `(eval_case_dict, config) -> float` functions.

The question: what do we call this, and where does it live?

## Decision

Add a new metric category, **`decision`**, distinct from `llm_judge`. The
defining trait of a decision metric is not "an LLM produced this score" (that
describes half the categories in this repo already) — it is that the
**underlying model returns a typed value plus a calibrated probability**, and
produces no free-text reasoning to parse. This is a different contract, not
a stylistic variant of LLM-judging, so it gets its own category rather than
living under `llm_judge/`.

### Provider abstraction: not `BaseLLM`

`BaseLLM.generate()` / `generate_json()` model "send text prompt, get back
text/JSON" — a shape that requires round-tripping through a prompt template
and can't express "N typed questions against one state, answered together."
Decision providers get a sibling abstraction, `BaseDecisionProvider`, in a
new `src/harness_evals/decision/` package (parallel to `llm/`):

```
src/harness_evals/decision/
├── __init__.py
├── base.py       # BaseDecisionProvider ABC
├── types.py      # ChoiceQuestion, ScoreQuestion, NoulQuestion,
│                 # ChoiceAnswer, ScoreAnswer, NoulAnswer, DecisionResponse
└── typesafe.py   # TypeSafeDecisionProvider (behind extras = ["typesafe"])
```

```python
class BaseDecisionProvider(ABC):
    @abstractmethod
    async def a_ask(self, state: str | dict | list, questions: dict[str, Question]) -> DecisionResponse: ...

    def ask(self, state, questions) -> DecisionResponse:
        return _run_async(self.a_ask(state, questions))
```

One provider call can answer several questions of mixed type against one
`state` — this is the hook that lets `CompositeMetric` batch (see "Batching"
below). `TypeSafeDecisionProvider` wraps `typesafe_sdk.AsyncTypeSafeClient`
(`pip install typesafe-sdk`), resolves the API key from a constructor arg or
the `TYPESAFE_API_KEY` env var (mirrors `OpenAILLM`'s `api_key` resolution
order), and defaults `model="jev-latest"`.

### Metrics: one primitive, one file, per `AGENTS.md`

```
src/harness_evals/metrics/decision/
├── __init__.py
├── choice.py   # ChoiceMetric
├── score.py    # ScoreMetric
└── noul.py     # NoulMetric
```

Each metric:

- Takes a `BaseDecisionProvider`, `instructions`, `criteria` (matching the
  primitive's request shape), and a `state_field` (default: `eval_case.output`,
  can point at `input`/`context`/a dict field via the existing
  `utils/path.extract_path` used by `composite/operators.py`).
- **With `eval_case.expected` set** — behaves like any other correctness
  metric: `value` = match/closeness against `expected`, `threshold` gates
  pass/fail as usual. This is the "does the model classify/rate this the way
  the golden says it should" use case (evaluating a classifier, a triage
  step, a rubric-graded response).
- **With `eval_case.expected` unset** — there is nothing to be "correct"
  against; the metric reports the raw decision. `value` is the primitive's
  own calibrated signal (`Noul` → `noul` directly; `Choice`/`Score` →
  `confidence`), and the full typed answer (`choice`/`score`/`noul`,
  `probabilities`, `legend`) goes into `Score.metadata`. This is the
  "audit/monitor a production routing decision" use case — you are scoring
  the decision's *confidence*, not its correctness, because there's no golden
  answer to compare to.
- Dimension defaults to `Dimension.CORRECTNESS` but is a constructor kwarg
  (matches the existing `build_metric()` registry-dimension-override pattern
  from PR #90) — a Noul checking for PII, for instance, is a `SAFETY` call at
  the site that wires it up.
- `mode: Literal["correctness", "confidence"] | None = None` is an explicit
  constructor kwarg, not inferred silently per-row. `None` (default) infers
  from `eval_case.expected is not None` at measure time, but authored
  `mode="correctness"` against a case with no `expected` is a hard error
  (`Score(value=0.0, reason=...)`, not a silent fallback to confidence mode) —
  a dataset with partial goldens must not flip the meaning of `Score.value`
  row-to-row under one metric name, which would corrupt summary averaging
  and `baseline/compare.py` regression detection.

Per-primitive value semantics. `expected` for `Noul` must be `bool`, `0`/`1`,
or the literal strings `"true"`/`"false"` (case-insensitive) — anything else
is a validation error at construction time, per `AGENTS.md`'s alias rule
("preserve ambiguous values as validation errors"); `bool("false")` is
truthy in Python, so this cannot be a bare `bool(expected)` cast. `NoulMetric`
also takes `invert: bool = False`, for cases like "does this contain PII"
where a *high* `noul` is the failure mode, not the success mode — without it,
confidence-mode value is indistinguishable from "definitely present" vs.
"definitely absent."

| Primitive | With `expected` (`mode="correctness"`) | Without `expected` (`mode="confidence"`) |
|---|---|---|
| `Noul` | `1.0` if `(noul >= 0.5) == expected_bool` else `0.0` | `value = noul if not invert else 1.0 - noul` |
| `Choice` | `1.0` if `choice == expected` else `0.0` | `value = confidence` |
| `Score` | `1.0 - abs(level_norm(score) - level_norm(expected))` | `value = confidence` (raw `score` in metadata) |

`Score` normalization: `criteria` is an ordered list, levels are indexed
`0..len(criteria)-1` (matching the docs' own indexing — level 0 is always the
low end). `level_norm(x) = x / (len(criteria) - 1)`, applied identically to
the response's `score` (which is a probability-weighted mean and can be
non-integer, e.g. `1.3`) and to `expected`. `expected` must be a numeric
level (float in `[0, len(criteria)-1]`) — not a legend string — because the
weighted-mean response has no exact string to compare against; document this
requirement on `ScoreMetric` and reject non-numeric `expected` as a
validation error rather than attempting fuzzy legend matching. Both
`level_norm` outputs are independently guaranteed in `[0, 1]`, so
`1.0 - abs(a - b)` is guaranteed in `[0, 1]` — no `Score.clamped` needed here,
but the provider response is still defensively clamped before use in case a
future TypeSafe API version returns a level outside the declared range.

### Batching: a `DecisionCompositeMetric` sibling, not a dual-mode `CompositeMetric`

Firing one HTTP call per primitive metric (three `ChoiceMetric`/`ScoreMetric`
instances against the same `eval_case.output` → three round trips) throws
away the exact efficiency TypeSafe's API is designed around ("ask independent
questions together"). The original draft of this ADR proposed solving this
by adding an async operator path directly to `CompositeMetric`. Review
caught two problems with that: (1) `composite.py`'s whole appeal is that it
is pure, synchronous, and fully config-serializable — the YAML-driven
`config/runner.py` path is the *primary* way `CompositeMetric` gets
constructed, and a decision operator needs a live `BaseDecisionProvider`
object injected into its config, which a YAML file cannot express, so the
batching feature would be unreachable from the main entry point; (2) one
failed batched `a_ask()` call would, under the original design, zero out
every sub-check in that group, whereas today each sync sub-check fails
independently (`composite.py`'s per-`sub` `try`/`except`).

Instead: a new sibling class, `DecisionCompositeMetric`, in
`metrics/composite/decision_composite.py`, constructed directly in Python
with an explicit `provider: BaseDecisionProvider` and a list of decision
sub-checks (same `name`/`weight`/`skip_when_missing` shape as
`CompositeMetric.sub_scores`, but `check` is one of the `Question` types
from `decision/types.py` plus a `state_field`, grouped by shared
`state_field` into one `a_ask()` per distinct resolved state). It reuses
`CompositeMetric`'s weighted-sum/`effective_weights` arithmetic via a shared
module-level `_combine(details, sub_scores) -> float` helper extracted out
of `composite.py`, so the two classes cannot silently drift in how they fold
weights — but it does not touch `composite.py`'s public sync `measure()`
contract or its `OPERATORS` registry at all. A batched call that fails
marks every sub-check *in that batch* `status="error"` (excluded from
`active_weight_sum`, same as today's per-sub-check error handling), not the
whole metric.

`DecisionCompositeMetric` implements `a_measure()` as primary, `measure()`
via `_run_async(self.a_measure(eval_case))` — same sync-wrapper pattern as
`BaseLLM.generate_sync()` and `BaseDecisionProvider.ask()`.

Cross-metric batching (e.g. two separate standalone `ChoiceMetric` instances
in the same `evaluate()` call happening to share a state) is explicitly
**out of scope** for v1. `DecisionCompositeMetric` is the batching primitive;
anyone who wants combined questions in one call reaches for it, same as
`CompositeMetric` today for sync operators.

### What we are *not* doing

- **Not** adding decision metrics under `llm_judge/`. The calibration
  contract is the whole point; folding this into "LLM judged with a
  different prompt shape" would bury it.
- **Not** adding a `confidence` field to the `Score` dataclass. `Score`'s
  fields are frozen per `AGENTS.md` ("never modify Golden, EvalCase, or Score
  fields without updating PLAN.md") and `metadata: dict[str, Any] | None`
  already exists for exactly this — every LLM-judge metric already stashes
  extra signal there (see `GEval`'s raw rubric score). `probability`/
  `confidence`/`probabilities`/`legend` all go in `Score.metadata`.
- **Not** using `BaseLLM` as the provider interface, for the reasons above.
- **Not** exposing `Noul` as a `SafetyMetric` subclass. It's a general
  calibrated-boolean primitive; a caller building a PII/toxicity check on top
  of it should compose `NoulMetric` behind their own `SafetyMetric` wrapper
  if they want the "never averaged" guarantee (ADR-003), same as any other
  building-block metric.

## Rationale

1. **The contract, not the vendor, defines the category.** "Decision" names
   the shape (typed value + calibrated probability, no reasoning text) so
   the category doesn't become "TypeSafe-specific." A second System-One-style
   provider (or a self-hosted classifier exposing the same three primitives)
   plugs in as another `BaseDecisionProvider`, not a new metric category.
2. **Reuse `CompositeMetric` instead of building a second combination
   engine.** It already implements exactly the weighted-sum/normalize
   pattern TypeSafe's own "composite scoring" doc describes. Building a
   parallel `DecisionComposite` would duplicate `effective_weights`/
   `skip_when_missing` logic that already exists and is tested.
3. **`expected`-present vs. `expected`-absent is a real fork, not a hack.**
   Existing metrics assume a golden always exists. Decision primitives are
   equally useful as production monitors with no golden (routing audits,
   escalation gating) — the docs' own "confidence-gated routing" pattern has
   no notion of "expected." Making this explicit in the value semantics table
   avoids a metric that silently returns `0.0`/garbage when `expected` is
   `None`.

## Trade-offs

- **New optional dependency surface.** `typesafe-sdk` becomes a new extra
  (`pip install harness-evals[decision]`), mirroring `[llm]`. Core stays
  dependency-free. Pinned `>=0.1,<1` (adjust to the SDK's actual first stable
  range at implementation time) — every other optional dependency in
  `pyproject.toml` carries a lower bound; this one should too.
- **`DecisionCompositeMetric` duplicates `CompositeMetric`'s shape.** Two
  classes instead of one dual-mode class. Mitigated by extracting shared
  fold arithmetic into one `_combine()` helper both call, so the two cannot
  drift on weighting semantics; accepted because it keeps `composite.py`
  fully synchronous, pure, and YAML-constructible, which review flagged as
  the more important property to preserve.
- **v1 batching is scoped to one `DecisionCompositeMetric` call.** Users who
  want cross-metric batching must model their questions as one composite
  rather than several standalone primitive metrics. Documented as a known
  limitation, not silently degraded.
- **Provider transport failures are skips, not zero scores.** A network
  error, timeout, or non-2xx response from TypeSafe returns `Score(value=0.0,
  metadata={"error": ...})` **is explicitly wrong and must not ship**; the
  correct behavior is to let the exception propagate (consistent with how
  judge-metric HTTP failures behave today — `evaluate()` catches and records
  exceptions per ADR-004, it does not let a metric quietly launder an infra
  failure into a passing-or-failing judgment). Decision metrics must not
  catch provider exceptions internally.
- **Floating model tag (`jev-latest`) affects reproducibility.** Two runs of
  the same eval on different days can silently use different model versions.
  `DecisionResponse.model` (the concrete model TypeSafe actually used) is
  always recorded in `Score.metadata["model"]` so this is at least visible,
  and `README`/`metrics-guide.md` call out pinning to a specific model
  version for any eval used as a CI gate or baseline.
- **No timeout/retry/concurrency policy specified yet.** `a_evaluate()`
  gathers across all metrics × cases with no global concurrency cap today;
  decision metrics add a real external rate limit (429/529 per the TypeSafe
  API docs) that heuristic/deterministic metrics never had to think about.
  `TypeSafeDecisionProvider` takes an explicit `timeout` and lets the SDK's
  own retry policy handle 429/529 backoff (per `sdk/python/api/retries.md`)
  — harness-evals does not reimplement retry logic, but documents what the
  SDK already covers so callers don't double-wrap it.

## Consequences

- New package `src/harness_evals/decision/` (provider abstraction) and new
  metric category `src/harness_evals/metrics/decision/` (`ChoiceMetric`,
  `ScoreMetric`, `NoulMetric`), registered in `metrics/__init__.py` and the
  catalog as `decision_choice`, `decision_score`, `decision_noul` — namespaced
  under `decision_` rather than bare `choice`/`score`/`noul` to read
  unambiguously in a catalog listing next to unrelated `score`-shaped
  concepts (the `Score` dataclass, any future deterministic "score" metric)
  and to group visually by category the way `turn_*` and `context_*` already
  do.
- `CatalogEntry` gains a `requires_provider: bool` field (alongside the
  existing `requires_llm`/`requires_embedding`), computed the same way via
  `_requires_param(cls, "provider")` in `catalog.py`. Without this, all three
  decision metrics land in the factory's default "heuristic" bucket (which
  assumes zero-dependency construction) and `build_metric()` would try to
  instantiate them with no provider, raising `TypeError` at config-load time
  for every user of the YAML config path — this must be fixed in the same PR
  that registers the metrics, not deferred.
- New sibling `metrics/composite/decision_composite.py`
  (`DecisionCompositeMetric`) plus an extracted `_combine()` helper shared
  with `composite.py`. `composite.py`/`operators.py` themselves are
  unchanged.
- `pyproject.toml` gains a `decision = ["typesafe-sdk"]` extra (pinned), added
  to `all`.
- No changes to `Golden`, `EvalCase`, `Score`, or any existing metric.

## Revisions from review

An architecture review (pre-implementation gate, conducted before any code
landed) raised eleven concerns against the original draft of this ADR. This
revision addresses all of them:

1. Score-normalization formula was ambiguous about level indexing and could
   produce values outside `[0, 1]` → fixed above with explicit `level_norm`.
2. Catalog auto-derivation would silently misfile decision metrics as
   zero-dependency → fixed with `CatalogEntry.requires_provider`.
3. Correctness/confidence mode was inferred from data presence, corrupting
   cross-row comparability → fixed with an explicit `mode` kwarg that errors
   rather than silently switches.
4. `CompositeMetric` async dual-mode was the wrong shape (unreachable from
   YAML config, group failure blast radius) → replaced with the
   `DecisionCompositeMetric` sibling design above.
5. Noul `bool(expected)` mishandles falsy-looking strings, and there was no
   inversion knob for "high probability = failure" checks → fixed with
   explicit `expected` type validation and `invert`.
6. Floating `model="jev-latest"` default breaks reproducibility → addressed
   by always recording the concrete `model` in `Score.metadata` and
   documenting pinning; the default itself is kept for ergonomics.
7. Token usage recorded inside `a_ask()` may not reach `a_evaluate()`'s
   contextvar-based collector when reached via the sync `measure()` →
   `_run_async` path (thread-pool dispatch does not propagate contextvars)
   → addressed by also stashing `input_tokens`/`output_tokens` directly in
   `Score.metadata` as a redundant, always-reliable path, independent of
   whether the contextvar propagates.
8. Provider transport errors were folding into `value=0.0` → fixed above:
   decision metrics let transport exceptions propagate rather than judging
   with them.
9. No stated timeout/retry/concurrency policy → addressed above (explicit
   `timeout`, defer to SDK retry policy, documented rather than
   reimplemented).
10. The `typesafe-sdk` extra was unpinned, unlike every other optional
    dependency → fixed with an explicit version range.
11. `score_decision`/`noul` catalog names were awkward/ambiguous → fixed with
    the `decision_choice`/`decision_score`/`decision_noul` namespace above.
