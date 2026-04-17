# ADR-010: Why ToolArgumentMatchMetric is a separate metric (not an enrichment of ToolCorrectness)

## Status

Accepted

## Context

`harness-evals` ships two related metrics for evaluating an agent's tool use:

- `ToolCorrectnessMetric` — deterministic, name-only. Compares `eval_case.tool_calls[*].name`
  against `eval_case.expected_tools: list[str]`. Two modes: `exact` (order- and
  count-sensitive) and `subset` (multiset, order-independent).
- `ArgumentCorrectnessMetric` — LLM-judged. Reads `eval_case.tool_calls` (including
  `input` payloads) and asks an LLM to score whether the arguments are correct
  and relevant.

There was no deterministic way to assert that an agent called the right tool
*with the right arguments*. Users wanting this had only the LLM-judged option,
which is non-deterministic, costs money/time per eval, and gives a fuzzy score
that's hard to use as a CI gate.

`Golden.expected_tools: list[str] | None` carries names only, so the data
model also could not express argument expectations.

Three implementation shapes were considered:

1. **Enrich `ToolCorrectnessMetric`** — overload its inputs/scoring to optionally
   compare arguments when authors provide them.
2. **Add a unified `ToolCallMatchMetric`** — a single new metric covering both
   selection and arguments, intended to *replace* `ToolCorrectness` when
   argument expectations exist.
3. **Add an args-only sibling `ToolArgumentMatchMetric`** — separate metric,
   composed with `ToolCorrectness` when both checks are wanted.

## Decision

Add a new args-only `ToolArgumentMatchMetric` in
`src/harness_evals/metrics/agent/tool_argument_match.py`, plus an additive
optional field on `Golden` and `EvalCase`:

```python
expected_tool_calls: list[ToolCall] | None = None
```

`ToolCorrectnessMetric` is unchanged. A composing `ToolCallMatchMetric` wrapper
is deferred to a follow-up.

The new metric's v1 surface:

```python
ToolArgumentMatchMetric(
    pair: str = "exact",            # "exact" | "subset"  (mirrors ToolCorrectness)
    arg_match: str = "exact",       # "exact" | "subset"
    ignore_keys: set[str] | None = None,
    wildcard_value: object = "*",
    threshold: float = 1.0,
)
```

## Rationale

1. **Single-responsibility per `AGENTS.md`.** The project conventions explicitly
   say "Keep metrics as single-file, single-class modules." A separate sibling
   mirrors the existing `ToolCorrectness` (names) / `ArgumentCorrectness`
   (args, LLM) split, completing a clean 2x2:

   |               | Names                   | Arguments                       |
   | ------------- | ----------------------- | ------------------------------- |
   | Deterministic | `ToolCorrectnessMetric` | **`ToolArgumentMatchMetric`**   |
   | LLM-judged    | —                       | `ArgumentCorrectnessMetric`     |

2. **Score interpretability.** `ToolCorrectness`'s contract today is crisp:
   "fraction of expected tool *name* positions matched." Fusing names and
   arguments into one metric forces an arbitrary weighting (50/50? args only
   when name matches? per-key partial credit?) and a `value` that no longer
   has a single intuitive meaning. Two metrics keep two simple contracts.

3. **No breaking change to `Golden`.** Adding `expected_tool_calls` is purely
   additive (defaults to `None`). The alternative — redefining
   `expected_tools` as `list[str | ToolCall | dict]` — would mean union-typing
   pain, dataset migration, and rewriting examples and integration guides. All
   existing JSONL datasets continue to load via the `from_dict` aliases on
   `EvalCase` (see ADR-007).

4. **Composability and separate thresholds.** Users typically want
   `threshold=1.0` for tool selection (a wrong tool is a hard failure) but a
   looser threshold for arguments (some args are noisy IDs/timestamps).
   Separate metrics expose two independent pass/fail signals in the score
   summary; a fused metric collapses them.

5. **Configuration containment.** Argument comparison has many reasonable
   strategies: exact dict equality, subset of expected keys, ignore-keys list,
   wildcard values, and (in future) JSON-Schema validation, numeric tolerance,
   case-insensitive string compare, regex on string values. Putting these on
   `ToolCorrectness` would more than double its surface area; in a sibling
   metric they sit naturally as the metric's primary concern.

## Trade-offs

- **Two metrics to wire when you want both checks.** Mitigation: README shows
  the canonical pairing snippet, and a composing `ToolCallMatchMetric` wrapper
  is planned as a follow-up so users with simple needs can use one metric.
- **One more optional field on `Golden`/`EvalCase`.** Mitigation: additive,
  default `None`, fully backward-compatible. Per `AGENTS.md`'s "Never modify
  Golden/EvalCase/Score fields without updating PLAN.md," `PLAN.md` is updated
  with the new field and the new metric row.
- **Lean v1 deliberately omits some strategies.** v1 supports
  `pair = exact|subset`, `arg_match = exact|subset`, `ignore_keys`, and
  `wildcard_value`. JSON-Schema mode, numeric tolerance, and per-key partial
  credit are listed as future work. This keeps the first cut small enough to
  review and avoids API churn.
- **Per-pair arg score is binary in v1.** A pair either matches or it doesn't.
  Per-key partial credit is more informative but introduces another weighting
  decision; deferred.

## Consequences

- New metric `ToolArgumentMatchMetric` is exported from
  `harness_evals.metrics.agent` and registered in the catalog as
  `"tool_argument_match"`.
- `Golden` and `EvalCase` gain an optional
  `expected_tool_calls: list[ToolCall] | None` field. `Golden.from_dict`,
  `EvalCase.from_dict`, and `EvalCase.from_golden` handle deserialization
  and propagation respectively.
- `ToolCorrectnessMetric` is unchanged — no risk to existing evals.
- The path is opened for a future `ToolCallMatchMetric` composing wrapper, an
  `arg_match="schema"` mode, and a `numeric_tolerance` knob, without further
  changes to the data model.
