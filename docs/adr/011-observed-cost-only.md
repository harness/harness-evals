# ADR-011: Cost is observed and caller-supplied, never estimated or bundled

## Status

Accepted

## Context

Eval runs spend real money — a target invocation plus one LLM call per judge metric, times every
dataset item. Users asked for a budget: stop the run before it spends more than $N.

A budget needs a cost figure, and there were three ways to get one:

1. **Estimate before the call** from a token-count heuristic and a price table.
2. **Bundle a public price catalog** (e.g. the `genai-prices` package) and price reported usage
   against it offline.
3. **Price reported usage against a rate card the caller supplies.**

Option 1 is unusable for a guardrail. The failure the feature exists to prevent is precisely the
case an estimate gets wrong — one item that costs 100x the others. A guardrail built on
predictions stops runs that were fine and admits the run that empties the budget.

Option 2 was implemented first and abandoned. Public list prices are not what Harness customers
pay. Enterprise accounts have negotiated rates, committed-use discounts, regional pricing, and
Bedrock provisioned-throughput arrangements. A bundled catalog is confidently wrong for exactly
the accounts that care most about cost, and it goes stale on the package's release cadence rather
than the customer's contract.

There is also a boundary constraint. harness-evals is open source and must not depend on Harness
services, tenancy, or credentials (see the CLI/SDK independence rule). Whatever the SDK does with
prices, it cannot go fetch them from a Harness API.

## Decision

**`harness_evals.cost` prices only usage a provider actually reported, against a rate card the
caller supplies. It has no price data of its own and makes no network calls.**

This scope is deliberately narrow to `harness_evals.cost`. `harness_evals.llm.cost` is a
separate, pre-existing module that estimates judge-call spend for informational reporting via
`litellm.completion_cost()` when available — it predates this ADR, estimates rather than
observes, and is not affected by this decision. The two modules intentionally take opposite
positions: `llm.cost` gives a best-effort estimate for judge telemetry when nothing more accurate
exists; `harness_evals.cost` refuses to estimate and reports unknown instead, because it exists to
back a spend guardrail rather than a telemetry line. Do not blend them, and do not read this ADR
as governing `llm.cost`.

Three consequences follow:

### 1. Observed-only

`ObservedUsage` holds token counts that came back from a real call. There is no code path that
prices a call that has not happened. Cost is therefore always retrospective, and a budget built
on it is inherently a *soft* limit — you learn what a batch cost after it ran. We accept bounded
overshoot in exchange for a number that is never wrong.

### 2. Caller-supplied rate card

`PricingSnapshot` is a serializable set of rate rows and aliases with an effective interval. The
SDK validates, resolves, and prices; it never sources. In Harness the snapshot comes from CCM via
the AI Evals control plane. Anyone else can supply their own. The SDK stays free of service
coupling, and customers get their real negotiated rates.

Snapshots are immutable and round-trip through plain dicts so a driver can fetch once per
lifecycle and distribute JSON to workers — no credentials or live clients cross that boundary.

### 3. Fail open, and say so

An unpriceable component yields `complete=False` with the gap named in `unknown_components`. It
never raises, and it never silently reports zero.

Two invariants protect this:

- **Unknown is not zero.** A missing price means "we don't know," so it cannot be added to a
  subtotal and cannot count against a budget. Only *known* spend enforces a limit. Losing
  pricing must not stop evaluations — evaluation is the product, cost accounting is a guardrail
  on it.
- **Never manufacture a complete zero.** `usd` stays `None` unless at least one usage component
  actually priced. Reporting a confident `$0.00` for a call nobody could price is worse than
  reporting nothing, because a caller cannot distinguish it from a genuinely free call.

Known partial subtotals are still returned. One unpriceable judge does not erase the components
that resolved.

### Strict dimension matching

An unobserved dimension matches **only** NULL rate rows. Never a wildcard, never "any region."

This is the least obvious part of the design and the easiest to break. A wildcard fallback would
make unobserved-region requests silently select whichever regional row sorted first — a specific,
plausible, wrong price. Matching NULL-only means the answer is either the region-agnostic rate or
an honest unknown.

Ambiguity is handled the same way: equally-ranked rows that disagree on price make the component
unknown rather than picking one.

## Rationale

The through-line is that a wrong cost number is worse than a missing one. A missing number is
visible — `complete=False`, components named, someone investigates. A wrong number is invisible
and silently authorizes or blocks spending.

Every rejected alternative fails this test. Estimation is wrong in the tail. A public catalog is
wrong for negotiated accounts. Wildcard dimensions are wrong per-region. A default-zero fee is
wrong in a way that looks like success.

## Trade-offs

- **Soft limits only.** Overshoot is bounded by one admitted concurrency batch. A hard cap would
  require serializing items, which defeats concurrency and slows every run to protect the rare
  one. Callers wanting a stricter bound can reduce concurrency.
- **The caller must plumb a snapshot.** More integration work than importing a catalog, and a
  caller without one gets no cost accounting at all (`UnavailablePricingProvider` keeps that path
  degrading rather than crashing).
- **Incompleteness is common early on.** Any model missing from the rate card is unknown. This is
  intentionally visible so rate cards get fixed, rather than papered over with list prices.
- **Decimal is required for money.** Slower than float and a second numeric convention alongside
  float `Score` values, accepted because monetary rounding drift across thousands of items is not
  acceptable.

## Consequences

- `harness_evals.cost` ships with no price data and no new runtime dependencies.
- Callers own snapshot acquisition, caching, and refresh.
- Cost guardrails are documented as best-effort with bounded overshoot, not hard caps.
- New usage types or dimensions extend `ResolvedRateRow` and its matching rank; unobserved
  dimensions must keep matching NULL-only.
- Adding a fallback price source to `harness_evals.cost` would reverse this ADR and needs a new
  one; it does not constrain `harness_evals.llm.cost`, whose estimation predates and sits outside
  this decision.

## See also

- [Cost Guide](../cost-guide.md) — usage, resolution order, dimensions
