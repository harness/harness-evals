# Cost Guide

`harness_evals.cost` prices **observed** token usage. It answers one question: given usage a
provider actually reported, and a rate card, what did that call cost?

It deliberately does not estimate, predict, or budget. There are no network calls, no bundled
price catalog, and no fallback to public list prices.

```python
from datetime import UTC, datetime
from harness_evals.cost import ObservedUsage, PricingSnapshot, ResolvedRateCardProvider

snapshot = PricingSnapshot.from_dict(payload)  # payload fetched by the caller
provider = ResolvedRateCardProvider(snapshot)

observation = provider.price_observed_usage(
    "anthropic",
    "claude-sonnet-4",
    ObservedUsage(input_tokens=1200, output_tokens=300),
    occurred_at=datetime.now(UTC),
)

if observation.complete:
    charge(observation.usd)          # exact Decimal
else:
    record_gap(observation.usd, observation.unknown_components)
```

## Who supplies the rate card

The SDK never fetches prices. A caller passes in a `PricingSnapshot` — a serializable set of
rate rows and model aliases with an effective interval.

In Harness deployments the snapshot comes from the AI Evals control plane, which reads Harness
CCM rate cards. Any other caller can build a snapshot from its own source; the SDK only cares
about the shape. This keeps the SDK free of service, tenancy, and credential concerns.

Snapshots round-trip through `to_dict()` / `from_dict()`, so a driver process can fetch once and
hand plain JSON to workers. Both directions are strict — unknown keys are rejected rather than
silently dropped, so a producer/consumer version skew fails loudly.

## The two-step resolution

**1. Alias → canonical model.** Providers are called by many names: a bare model ID, an Azure
deployment name, a Bedrock inference-profile ARN. `ModelAliasRow` maps those onto a canonical
model. Match kinds:

| Match kind | Behavior |
|------------|----------|
| `EXACT`, `ARN`, `PATH`, `DEPLOYMENT` | Case-insensitive equality |
| `PREFIX` | Requested model starts with the alias |
| `CONTAINS` | Alias appears anywhere in the requested model |

When several aliases match, the winner is chosen by `scope_priority`, then `alias_tier`, then
`match_specificity`. Account-specific rows beat rows scoped to `__GLOBAL__`, so a negotiated
mapping overrides the shared default.

**2. Canonical model → rate row.** `ResolvedRateRow` carries a unit price per usage type
(`Input`, `Output`, `CacheRead`, `CacheWrite`), a `unit_block_size`, a `min_charge_units`, an
optional tier range, and an optional `base_request_fee`. Only rows whose effective interval
contains `occurred_at` are eligible.

Provider spellings are normalized first, so `bedrock`, `aws.bedrock`, `vertex`,
`gcp.vertex_ai`, and Azure OpenAI spellings land on canonical CCM providers (`aws`, `gcp`,
`azure`). `normalize_provider()` is exported if you need the same mapping; it passes
unrecognized spellings through unchanged rather than coercing them into a wrong identity.
Resolution is data-driven: a provider is priceable when the snapshot contains matching alias
and rate rows, including Azure deployment-name aliases.

## Dimensions

Rate rows can be qualified by `region`, `sub_provider_id`, `commitment_type`,
`context_window_bucket`, and `service_tier`. The matching rule is strict and worth
internalizing:

> A dimension you did not observe matches **only** rate rows where that dimension is NULL.

So if you don't know the region, you get the region-agnostic price or nothing — never an
arbitrary region's price. Pass only what you actually observed:

```python
provider.price_observed_usage(
    "aws",
    "arn:aws:bedrock:us-east-1:...:inference-profile/...",
    usage,
    dimensions={"region": "us-east-1", "sub_provider_id": "bedrock"},
)
```

Concrete dimensions attached to a matched alias are merged in automatically, so a CCM alias that
already knows its region will select the regional rate without the caller repeating it. If an
alias and an explicit caller dimension disagree, the observation fails incomplete with
`Dimensions` listed — a conflict is never resolved by guessing.

Bedrock LLM clients (`BedrockAnthropicLLM`, `BedrockOpenAILLM`) expose the region they resolved
as `.aws_region` for exactly this purpose.

## Incompleteness is a first-class result

`CostObservation.complete` is `False` whenever any part of the calculation could not be resolved,
with `unknown_components` naming the gaps (`Identity`, `Dimensions`, `Input`, `Output`,
`CacheRead`, `CacheWrite`, `BaseRequestFee`, plus `PricingProvider` from
`UnavailablePricingProvider`). Omitted cache token counts default to `0` (no cache activity);
pass `None` only when cache usage was observed as unknown.

Two rules matter:

**Known partial subtotals survive.** If input tokens priced but output tokens did not, `usd`
holds the input cost and `complete` is `False`. Callers get everything that could be determined
plus an honest statement of what could not.

**A complete zero is never manufactured.** `usd` stays `None` unless at least one usage
component actually priced. This is why `price_observed_usage` accumulates a `base_request_fee`
only from a confirmed, unambiguous USD winner — otherwise an ambiguous row with a default zero
fee could produce a confident-looking `$0.00` for a call that was never priced at all.

Ambiguity is treated the same as absence. If two equally-ranked rows disagree on price, the
component is unknown rather than arbitrarily resolved. Rows agreeing on price but differing only
in currency case (`USD` / `usd`) are treated as equivalent, not ambiguous.

When no snapshot is available, `UnavailablePricingProvider` satisfies the same protocol and
returns incomplete for everything. Callers keep one code path and degrade instead of crashing.

## Money is always Decimal

Prices parse from strings or `Decimal`, never `float`, and all arithmetic is `Decimal`. Producers
should serialize money as fixed decimal strings so nothing is lost in transport. `Score` and
metric values remain floats; this applies only to money.

## Extending

`PricingProvider` is a `Protocol`, so any object with a matching `price_observed_usage` works —
useful for tests and for callers with a different rate source:

```python
class FlatRateProvider:
    def price_observed_usage(self, provider, model, usage, *, occurred_at=None, dimensions=None):
        ...
```

## See also

- [ADR-011](adr/011-observed-cost-only.md) — why observed-only, and why fail-open
- [Architecture](architecture.md) — where cost sits relative to metrics and sinks
