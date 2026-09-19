# ADR-011: Decision primitives (Choice/Score/Noul) as a new metric category, not LLM-judge metrics

## Status

Proposed

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

Per-primitive value semantics:

| Primitive | With `expected` | Without `expected` |
|---|---|---|
| `Noul` | `1.0` if `(noul >= 0.5) == bool(expected)` else `0.0` | `value = noul` |
| `Choice` | `1.0` if `choice == expected` else `0.0` | `value = confidence` |
| `Score` | `1.0 - abs(normalized_score - normalized_expected)` | `value = confidence` (raw `score` in metadata) |

`Score` normalization follows the docs' own guidance: divide the raw
level-weighted score by `len(criteria) - 1` to land in [0, 1], consistent
with `Score.value`'s existing [0, 1] contract (ADR-002) and with how the
Typesafe docs themselves normalize before combining.

### Batching: reuse `CompositeMetric`, don't reinvent it

Firing one HTTP call per primitive metric (three `ChoiceMetric`/`ScoreMetric`
instances against the same `eval_case.output` → three round trips) throws
away the exact efficiency TypeSafe's API is designed around ("ask independent
questions together"). Rather than build a new batching/coalescing layer,
extend `CompositeMetric` — which already does deterministic weighted
combination of named sub-checks — with a new operator kind that groups
decision sub-checks sharing a `state_field` into one `a_ask()` call:

- `operators.py` gains an **async** operator registry (`ASYNC_OPERATORS`)
  alongside the existing sync `OPERATORS`; a `"decision"` check type lives
  there, since it requires network I/O.
- `CompositeMetric` gains `a_measure()` (today it only implements sync
  `measure()`). It partitions `sub_scores` into sync operators (run as
  today) and decision operators (grouped by `state_field`, one `a_ask()` per
  distinct state, questions fanned out by `name`), then folds both sets into
  the same weighted-sum/`effective_weights` logic already in
  `composite.py`.
- Calling `CompositeMetric.measure()` (sync) with decision sub-checks present
  still works via `_run_async`, same as `BaseLLM.generate_sync()`.

This is additive to `composite.py`/`operators.py` — existing sub-checks are
untouched — and it's the only piece of this ADR that changes an existing
file rather than adding new ones.

Cross-metric batching (e.g. two separate standalone `ChoiceMetric` instances
in the same `evaluate()` call happening to share a state) is explicitly
**out of scope** for v1. `CompositeMetric` is the batching primitive; anyone
who wants combined questions in one call reaches for it, same as today.

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
  dependency-free.
- **`CompositeMetric` gains an async code path.** Slightly more surface in a
  file that was previously pure/sync-only, but it's additive and gated behind
  the presence of a `"decision"` check type — existing sync-only composites
  see no behavior change.
- **v1 batching is scoped to one `CompositeMetric` call.** Users who want
  cross-metric batching must model their questions as one composite rather
  than several standalone primitive metrics. Documented as a known
  limitation, not silently degraded.

## Consequences

- New package `src/harness_evals/decision/` (provider abstraction) and new
  metric category `src/harness_evals/metrics/decision/` (`ChoiceMetric`,
  `ScoreMetric`, `NoulMetric`), registered in `metrics/__init__.py` and the
  catalog as `choice`, `score_decision` (avoids catalog-name collision with
  `Score` the dataclass and any future `score`-named deterministic metric —
  confirm final catalog key during implementation), `noul`.
- `metrics/composite/operators.py` and `metrics/composite/composite.py` gain
  an async `"decision"` operator and `CompositeMetric.a_measure()`.
- `pyproject.toml` gains a `decision = ["typesafe-sdk"]` extra, added to
  `all`.
- No changes to `Golden`, `EvalCase`, `Score`, or any existing metric.
