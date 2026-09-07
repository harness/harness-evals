from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from harness_evals.cost.pricing import (
    GLOBAL_SCOPE,
    CostObservation,
    ModelAliasRow,
    ObservedUsage,
    PricingSnapshot,
    RateDimension,
    ResolvedRateCardProvider,
    ResolvedRateRow,
    normalize_provider,
    price_observed_usage,
)

NOW = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)
GLOBAL = GLOBAL_SCOPE
DEFAULT_USAGE = ObservedUsage(1000, 1000, 0, 0)


def alias(
    *,
    alias_id: str = "alias-1",
    account_id: str = GLOBAL,
    provider: str | None = "openai",
    value: str = "gpt-4o",
    model: str = "gpt-4o-2024-08-06",
    match_kind: str = "EXACT",
    scope_priority: int = 0,
    alias_tier: int = 0,
    match_specificity: int | None = None,
    active: bool = True,
    effective_from: datetime = NOW - timedelta(days=1),
    effective_to: datetime | None = NOW + timedelta(days=1),
    sync_id: str = "sync-alias",
    dimensions: tuple[RateDimension, ...] = (),
) -> ModelAliasRow:
    return ModelAliasRow(
        alias_id=alias_id,
        account_id=account_id,
        provider=provider,
        alias=value,
        canonical_model=model,
        match_kind=match_kind,
        scope_priority=scope_priority,
        alias_tier=alias_tier,
        match_specificity=len(value) if match_specificity is None else match_specificity,
        active=active,
        effective_from=effective_from,
        effective_to=effective_to,
        sync_id=sync_id,
        dimensions=dimensions,
    )


def rate(
    usage_type: str,
    *,
    rate_id: str | None = None,
    price_id: str | None = None,
    account_id: str = GLOBAL,
    provider: str = "openai",
    model: str = "gpt-4o-2024-08-06",
    unit_price: str = "1",
    currency: str = "USD",
    unit_block_size: int = 1000,
    min_charge_units: int = 0,
    tier_min_units: int = 0,
    tier_max_units: int | None = None,
    dimensions: tuple[RateDimension, ...] = (),
    effective_from: datetime = NOW - timedelta(days=1),
    effective_to: datetime | None = NOW + timedelta(days=1),
    base_request_fee: str = "0",
    sync_id: str = "sync-rate",
) -> ResolvedRateRow:
    slug = usage_type.lower()
    return ResolvedRateRow(
        rate_id=rate_id or f"rate-{slug}",
        price_id=price_id or f"price-{slug}",
        account_id=account_id,
        provider=provider,
        model=model,
        usage_type=usage_type,
        unit_price=Decimal(unit_price),
        currency=currency,
        unit_block_size=unit_block_size,
        min_charge_units=min_charge_units,
        tier_min_units=tier_min_units,
        tier_max_units=tier_max_units,
        dimensions=dimensions,
        effective_from=effective_from,
        effective_to=effective_to,
        base_request_fee=Decimal(base_request_fee),
        sync_id=sync_id,
    )


def snapshot(
    *,
    account_id: str = "acct-1",
    aliases: tuple[ModelAliasRow, ...] | None = None,
    rates: tuple[ResolvedRateRow, ...] | None = None,
) -> PricingSnapshot:
    return PricingSnapshot(
        account_id=account_id,
        interval_start=NOW - timedelta(hours=1),
        interval_end=NOW + timedelta(hours=1),
        alias_type="ccm:model_alias",
        rate_type="ccm:resolved_rate",
        aliases=aliases if aliases is not None else (alias(),),
        rates=rates if rates is not None else (rate("Input"), rate("Output"), rate("CacheRead"), rate("CacheWrite")),
    )


def price(
    card: PricingSnapshot,
    usage: ObservedUsage = DEFAULT_USAGE,
    **kwargs: object,
) -> CostObservation:
    return price_observed_usage(
        "openai",
        "gpt-4o",
        usage,
        occurred_at=NOW,
        pricing_provider=ResolvedRateCardProvider(card),
        **kwargs,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("anthropic", "anthropic"),
        ("openai", "openai"),
        ("bedrock", "aws"),
        ("aws.bedrock", "aws"),
        ("gcp.vertex_ai", "gcp"),
        ("vertex", "gcp"),
        ("google", "gcp"),
        ("unknown", "unknown"),
    ],
)
def test_normalize_provider(raw: str, expected: str) -> None:
    assert normalize_provider(raw) == expected


@pytest.mark.unit
def test_snapshot_rows_round_trip_as_strict_json_safe_payloads() -> None:
    payload = snapshot().to_dict()

    assert payload["aliases"][0]["effective_from"].endswith("Z")
    assert payload["rates"][0]["unit_price"] == "1"
    assert payload["rates"][0]["dimensions"] == []
    assert isinstance(payload["aliases"], list)
    assert PricingSnapshot.from_dict(payload) == snapshot()


@pytest.mark.unit
def test_snapshot_round_trip_accepts_provider_wildcard_alias() -> None:
    card = snapshot(aliases=(alias(provider=None),))
    payload = card.to_dict()

    assert payload["aliases"][0]["provider"] is None
    assert PricingSnapshot.from_dict(payload) == card


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "bad_value", "exception"),
    [
        (("rates", 0, "unit_price"), 1, TypeError),
        (("rates", 0, "currency"), None, TypeError),
        (("rates", 0, "unit_block_size"), 0, ValueError),
        (("rates", 0, "base_request_fee"), "-0.01", ValueError),
        (("rates", 0, "usage_type"), "Bogus", ValueError),
        (("aliases", 0, "match_kind"), "GLOB", ValueError),
        (("aliases", 0, "effective_from"), "not-a-date", ValueError),
    ],
)
def test_snapshot_rejects_malformed_rows(
    path: tuple[object, ...], bad_value: object, exception: type[Exception]
) -> None:
    payload = snapshot().to_dict()
    target: object = payload
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = bad_value  # type: ignore[index]

    with pytest.raises(exception):
        PricingSnapshot.from_dict(payload)


@pytest.mark.unit
def test_snapshot_rejects_missing_and_unknown_fields() -> None:
    missing = snapshot().to_dict()
    del missing["rates"][0]["unit_price"]
    unknown = snapshot().to_dict()
    unknown["extra"] = True

    with pytest.raises(ValueError, match="fields"):
        PricingSnapshot.from_dict(missing)
    with pytest.raises(ValueError, match="fields"):
        PricingSnapshot.from_dict(unknown)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["EXACT", "ARN", "PATH", "DEPLOYMENT"])
def test_equality_alias_kinds_are_case_insensitive(kind: str) -> None:
    card = snapshot(aliases=(alias(value="GPT-4O", match_kind=kind),))
    assert price(card).resolved_model == "gpt-4o-2024-08-06"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kind", "value", "reported"),
    [("PREFIX", "gpt-4", "gpt-4o-us"), ("CONTAINS", "4O-", "gpt-4o-us")],
)
def test_pattern_alias_kinds_use_documented_matching(kind: str, value: str, reported: str) -> None:
    card = snapshot(aliases=(alias(value=value, match_kind=kind),))
    observation = price_observed_usage(
        "openai",
        reported,
        ObservedUsage(1000, 1000, 0, 0),
        occurred_at=NOW,
        pricing_provider=ResolvedRateCardProvider(card),
    )
    assert observation.resolved_model == "gpt-4o-2024-08-06"


@pytest.mark.unit
def test_prefix_alias_matching_is_case_insensitive() -> None:
    card = snapshot(aliases=(alias(value="GPT-4", match_kind="PREFIX"),))
    observation = price_observed_usage(
        "openai",
        "gPt-4O-us",
        DEFAULT_USAGE,
        occurred_at=NOW,
        pricing_provider=ResolvedRateCardProvider(card),
    )
    assert observation.resolved_model == "gpt-4o-2024-08-06"


@pytest.mark.unit
def test_provider_wildcard_alias_resolves_for_normalized_provider() -> None:
    model = "claude-sonnet"
    card = snapshot(
        aliases=(alias(provider=None, model=model),),
        rates=tuple(rate(kind, provider="aws", model=model) for kind in ("Input", "Output", "CacheRead", "CacheWrite")),
    )
    observation = price_observed_usage(
        "bedrock",
        "gpt-4o",
        DEFAULT_USAGE,
        occurred_at=NOW,
        pricing_provider=ResolvedRateCardProvider(card),
    )

    assert observation.complete is True
    assert observation.provider == "aws"
    assert observation.resolved_model == model


@pytest.mark.unit
def test_exact_provider_alias_beats_wildcard_at_equal_rank() -> None:
    aliases = (
        alias(alias_id="wildcard", provider=None, model="wildcard-model"),
        alias(alias_id="exact", provider="openai", model="exact-model"),
    )
    rates = tuple(rate(kind, model="exact-model") for kind in ("Input", "Output", "CacheRead", "CacheWrite"))
    observation = price(snapshot(aliases=aliases, rates=rates))

    assert observation.complete is True
    assert observation.resolved_model == "exact-model"
    assert observation.provenance is not None
    assert observation.provenance.alias_ids == ("exact",)


@pytest.mark.unit
def test_provider_wildcard_does_not_make_unknown_provider_priceable() -> None:
    card = snapshot(aliases=(alias(provider=None),))
    observation = price_observed_usage(
        "unknown-provider",
        "gpt-4o",
        DEFAULT_USAGE,
        occurred_at=NOW,
        pricing_provider=ResolvedRateCardProvider(card),
    )

    assert observation.complete is False
    assert observation.usd is None
    assert observation.resolved_model is None
    assert observation.unknown_components == ("Identity",)


@pytest.mark.unit
def test_conflicting_wildcard_aliases_retain_ambiguity_provenance() -> None:
    aliases = (
        alias(alias_id="wild-a", provider=None, model="model-a", sync_id="sync-a"),
        alias(alias_id="wild-b", provider=None, model="model-b", sync_id="sync-b"),
    )
    observation = price(snapshot(aliases=aliases))

    assert observation.complete is False
    assert observation.provenance is not None
    assert observation.provenance.alias_ids == ("wild-a", "wild-b")
    assert observation.provenance.sync_ids == ("sync-a", "sync-b")


SERVICE_DIMENSIONS = (RateDimension("region", "us-east-1"), RateDimension("sub_provider_id", "bedrock"))


def service_snapshot(
    *,
    rates: tuple[ResolvedRateRow, ...],
    alias_dimensions: tuple[RateDimension, ...] = SERVICE_DIMENSIONS,
) -> PricingSnapshot:
    aliases = (
        alias(
            alias_id="account-service",
            account_id="acct-1",
            provider="aws",
            model="claude-sonnet",
            dimensions=alias_dimensions,
        ),
    )
    return snapshot(aliases=aliases, rates=rates)


def price_bedrock(card: PricingSnapshot, **kwargs: object) -> CostObservation:
    return price_observed_usage(
        "bedrock",
        "gpt-4o",
        DEFAULT_USAGE,
        occurred_at=NOW,
        pricing_provider=ResolvedRateCardProvider(card),
        **kwargs,
    )


@pytest.mark.unit
def test_snapshot_round_trip_carries_alias_dimensions_as_a_strict_field() -> None:
    card = snapshot(aliases=(alias(dimensions=SERVICE_DIMENSIONS),))
    payload = card.to_dict()

    assert payload["aliases"][0]["dimensions"] == [
        {"name": "region", "value": "us-east-1"},
        {"name": "sub_provider_id", "value": "bedrock"},
    ]
    assert PricingSnapshot.from_dict(payload) == card

    del payload["aliases"][0]["dimensions"]
    with pytest.raises(ValueError, match="fields"):
        PricingSnapshot.from_dict(payload)


@pytest.mark.unit
def test_alias_dimensions_select_service_rates_without_caller_dimensions() -> None:
    rows = tuple(
        rate(
            kind,
            account_id="acct-1",
            provider="aws",
            model="claude-sonnet",
            unit_price="2",
            dimensions=SERVICE_DIMENSIONS,
        )
        for kind in ("Input", "Output", "CacheRead", "CacheWrite")
    )
    observation = price_bedrock(service_snapshot(rates=rows))

    assert observation.complete is True
    assert observation.usd == Decimal("4")
    assert observation.resolved_model == "claude-sonnet"


@pytest.mark.unit
def test_alias_dimensions_fall_back_to_global_all_null_rates() -> None:
    rows = tuple(
        rate(kind, provider="aws", model="claude-sonnet") for kind in ("Input", "Output", "CacheRead", "CacheWrite")
    )
    observation = price_bedrock(service_snapshot(rates=rows))

    assert observation.complete is True
    assert observation.usd == Decimal("2")
    assert observation.provenance is not None
    assert observation.provenance.scopes == ("acct-1", GLOBAL)


@pytest.mark.unit
def test_caller_dimension_conflicting_with_alias_is_incomplete() -> None:
    rows = tuple(
        rate(
            kind,
            account_id="acct-1",
            provider="aws",
            model="claude-sonnet",
            unit_price="2",
            dimensions=SERVICE_DIMENSIONS,
        )
        for kind in ("Input", "Output", "CacheRead", "CacheWrite")
    )
    observation = price_bedrock(service_snapshot(rates=rows), dimensions={"region": "us-west-2"})

    assert observation.complete is False
    assert observation.usd is None
    assert observation.unknown_components == ("Dimensions",)
    assert observation.resolved_model == "claude-sonnet"
    assert observation.provenance is not None
    assert observation.provenance.alias_ids == ("account-service",)
    assert observation.provenance.snapshot_start == NOW - timedelta(hours=1)


@pytest.mark.unit
def test_caller_dimension_agreeing_with_alias_is_deduplicated() -> None:
    rows = tuple(
        rate(
            kind,
            account_id="acct-1",
            provider="aws",
            model="claude-sonnet",
            unit_price="2",
            dimensions=SERVICE_DIMENSIONS,
        )
        for kind in ("Input", "Output", "CacheRead", "CacheWrite")
    )
    observation = price_bedrock(service_snapshot(rates=rows), dimensions={"region": "us-east-1"})

    assert observation.complete is True
    assert observation.usd == Decimal("4")


@pytest.mark.unit
def test_unobserved_service_dimension_still_requires_null_rate_rows() -> None:
    rows = tuple(
        rate(
            kind,
            account_id="acct-1",
            provider="aws",
            model="claude-sonnet",
            unit_price="2",
            dimensions=SERVICE_DIMENSIONS,
        )
        for kind in ("Input", "Output", "CacheRead", "CacheWrite")
    )
    observation = price_bedrock(service_snapshot(rates=rows, alias_dimensions=()))

    assert observation.complete is False
    assert observation.usd is None
    assert observation.unknown_components == ("Input", "Output", "CacheRead", "CacheWrite")


@pytest.mark.unit
def test_alias_filters_inactive_expired_and_incompatible_provider_rows() -> None:
    aliases = (
        alias(alias_id="inactive", model="wrong-1", active=False, scope_priority=99),
        alias(alias_id="expired", model="wrong-2", effective_to=NOW, scope_priority=99),
        alias(alias_id="provider", model="wrong-3", provider="anthropic", scope_priority=99),
        alias(alias_id="winner"),
    )
    assert price(snapshot(aliases=aliases)).resolved_model == "gpt-4o-2024-08-06"


@pytest.mark.unit
def test_alias_winner_uses_scope_priority_then_tier_then_specificity() -> None:
    aliases = (
        alias(alias_id="low-scope", value="gpt-4o", model="wrong", scope_priority=0, alias_tier=99),
        alias(alias_id="low-tier", value="gpt-4o", model="wrong", scope_priority=1, alias_tier=1),
        alias(
            alias_id="long-but-lower-specificity",
            value="gpt-4",
            model="wrong",
            match_kind="PREFIX",
            scope_priority=1,
            alias_tier=2,
            match_specificity=1,
        ),
        alias(
            alias_id="winner",
            value="gpt",
            match_kind="PREFIX",
            scope_priority=1,
            alias_tier=2,
            match_specificity=2,
        ),
    )
    assert price(snapshot(aliases=aliases)).resolved_model == "gpt-4o-2024-08-06"


@pytest.mark.unit
def test_equal_rank_conflicting_aliases_are_ambiguous() -> None:
    card = snapshot(
        aliases=(
            alias(alias_id="a", sync_id="sync-a"),
            alias(alias_id="b", model="other", sync_id="sync-b"),
        )
    )
    observation = price(card)
    assert observation.complete is False
    assert observation.usd is None
    assert observation.resolved_model is None
    assert observation.unknown_components == ("Identity",)
    assert observation.provenance is not None
    assert observation.provenance.account_key == "acct-1"
    assert observation.provenance.alias_type == "ccm:model_alias"
    assert observation.provenance.rate_type == "ccm:resolved_rate"
    assert observation.provenance.alias_ids == ("a", "b")
    assert observation.provenance.sync_ids == ("sync-a", "sync-b")
    assert observation.provenance.snapshot_start == NOW - timedelta(hours=1)
    assert observation.provenance.snapshot_end == NOW + timedelta(hours=1)


@pytest.mark.unit
def test_global_only_alias_and_rates_fall_back_for_account_snapshot() -> None:
    observation = price(snapshot())
    assert observation.complete is True
    assert observation.usd == Decimal("2")
    assert observation.provenance is not None
    assert observation.provenance.account_key == "acct-1"
    assert observation.provenance.scopes == (GLOBAL,)


@pytest.mark.unit
def test_account_alias_and_rates_win_with_global_fallback_per_component() -> None:
    aliases = (
        alias(alias_id="global", model="global-model"),
        alias(alias_id="account", account_id="acct-1", model="account-model"),
    )
    rates = (
        rate("Input", account_id="acct-1", model="account-model", unit_price="2"),
        rate("Output", model="account-model", unit_price="3"),
        rate("CacheRead", model="account-model"),
        rate("CacheWrite", model="account-model"),
    )
    observation = price(snapshot(aliases=aliases, rates=rates))
    assert observation.usd == Decimal("5")
    assert observation.provenance is not None
    assert observation.provenance.scopes == ("acct-1", GLOBAL)


@pytest.mark.unit
def test_effective_intervals_are_half_open_for_aliases_and_rates() -> None:
    aliases = (
        alias(alias_id="old", model="old", effective_from=NOW - timedelta(days=1), effective_to=NOW),
        alias(alias_id="new", model="new", effective_from=NOW, effective_to=None),
    )
    rates = (
        rate("Input", model="new", unit_price="2", effective_from=NOW, effective_to=None),
        rate("Output", model="new", unit_price="3", effective_from=NOW, effective_to=None),
        rate("CacheRead", model="new", effective_from=NOW, effective_to=None),
        rate("CacheWrite", model="new", effective_from=NOW, effective_to=None),
    )
    observation = price(snapshot(aliases=aliases, rates=rates))
    assert observation.resolved_model == "new"
    assert observation.usd == Decimal("5")


@pytest.mark.unit
def test_rate_effective_end_equal_to_occurrence_is_excluded() -> None:
    rows = (
        rate("Input", rate_id="expired", unit_price="99", effective_to=NOW),
        rate("Input", rate_id="active", unit_price="2", effective_from=NOW, effective_to=None),
        rate("Output"),
        rate("CacheRead"),
        rate("CacheWrite"),
    )
    observation = price(snapshot(rates=rows))
    assert observation.usd == Decimal("3")
    assert observation.provenance is not None
    assert "expired" not in observation.provenance.rate_ids


@pytest.mark.unit
def test_dimensions_require_null_for_missing_and_prefer_exact_for_concrete_request() -> None:
    null_dimension = (RateDimension("region", None),)
    exact_dimension = (RateDimension("region", "us-east-1"),)
    rows = (
        rate("Input", rate_id="null", unit_price="1", dimensions=null_dimension),
        rate("Input", rate_id="exact", unit_price="2", dimensions=exact_dimension),
        rate("Output"),
        rate("CacheRead"),
        rate("CacheWrite"),
    )
    card = snapshot(rates=rows)
    assert price(card, dimensions={"region": "us-east-1"}).usd == Decimal("3")
    assert price(card).usd == Decimal("2")


@pytest.mark.unit
def test_equal_precedence_conflicting_rates_are_ambiguous() -> None:
    rows = (
        rate("Input", rate_id="one", unit_price="1"),
        rate("Input", rate_id="two", unit_price="2"),
        rate("Output"),
        rate("CacheRead"),
        rate("CacheWrite"),
    )
    observation = price(snapshot(rates=rows))
    assert observation.usd == Decimal("1")
    assert observation.complete is False
    assert observation.unknown_components == ("Input",)
    assert observation.provenance is not None
    assert observation.provenance.rate_ids == ("one", "two", "rate-output", "rate-cacheread", "rate-cachewrite")


@pytest.mark.unit
def test_conflicting_rates_with_no_other_usage_do_not_manufacture_zero_cost() -> None:
    rows = (
        rate("Input", rate_id="one", unit_price="1"),
        rate("Input", rate_id="two", unit_price="2"),
    )
    observation = price(snapshot(rates=rows), ObservedUsage(1000, None, None, None))
    assert observation.usd is None
    assert observation.complete is False
    assert observation.unknown_components == ("Input", "Output", "CacheRead", "CacheWrite")


@pytest.mark.unit
def test_agreed_nonzero_fee_on_ambiguous_usage_does_not_produce_cost() -> None:
    rows = (
        rate("Input", rate_id="one", unit_price="1", base_request_fee="0.25"),
        rate("Input", rate_id="two", unit_price="2", base_request_fee="0.25"),
    )
    observation = price(snapshot(rates=rows), ObservedUsage(1000, None, None, None))
    assert observation.usd is None
    assert observation.complete is False
    assert observation.unknown_components.count("BaseRequestFee") == 1


@pytest.mark.unit
def test_currency_case_differences_are_equivalent_rather_than_ambiguous() -> None:
    rows = (
        rate("Input", rate_id="upper", currency="USD"),
        rate("Input", rate_id="lower", currency="usd"),
        rate("Output"),
        rate("CacheRead"),
        rate("CacheWrite"),
    )
    observation = price(snapshot(rates=rows))
    assert observation.usd == Decimal("2")
    assert observation.complete is True


@pytest.mark.unit
def test_unpriceable_currency_fee_marks_fee_unknown_once_and_excludes_every_fee() -> None:
    rows = (
        rate("Input", currency="EUR", base_request_fee="0.50"),
        rate("Output", base_request_fee="0.25"),
        rate("CacheRead", base_request_fee="0.25"),
        rate("CacheWrite", base_request_fee="0.25"),
    )
    observation = price(snapshot(rates=rows))
    assert observation.usd == Decimal("1")
    assert observation.complete is False
    assert observation.unknown_components == ("Input", "BaseRequestFee")


@pytest.mark.unit
def test_tier_bounds_and_minimum_charge_select_billable_units() -> None:
    rows = (
        rate("Input", rate_id="low", unit_price="1", min_charge_units=1500, tier_max_units=2000),
        rate("Input", rate_id="high", unit_price="2", min_charge_units=1500, tier_min_units=2000),
        rate("Output"),
        rate("CacheRead"),
        rate("CacheWrite"),
    )
    card = snapshot(rates=rows)
    assert price(card, ObservedUsage(10, 0, 0, 0)).usd == Decimal("1.5")
    assert price(card, ObservedUsage(2000, 0, 0, 0)).usd == Decimal("4")


@pytest.mark.unit
def test_all_usage_types_are_independent_and_zero_is_priceable() -> None:
    rows = (
        rate("Input", unit_price="1"),
        rate("Output", unit_price="2"),
        rate("CacheRead", unit_price="3"),
        rate("CacheWrite", unit_price="4"),
    )
    card = snapshot(rates=rows)
    observation = price(card, ObservedUsage(1000, 1000, 1000, 1000))
    zero = price(card, ObservedUsage(0, 0, 0, 0))
    assert observation.usd == Decimal("10")
    assert observation.complete is True
    assert zero.usd == Decimal("0")
    assert zero.complete is True


@pytest.mark.unit
def test_base_request_fee_is_added_once() -> None:
    rows = tuple(rate(kind, base_request_fee="0.25") for kind in ("Input", "Output", "CacheRead", "CacheWrite"))
    assert price(snapshot(rates=rows)).usd == Decimal("2.25")


@pytest.mark.unit
def test_inconsistent_base_request_fee_is_unknown_without_erasing_usage_subtotal() -> None:
    rows = (
        rate("Input", base_request_fee="0.25"),
        rate("Output", base_request_fee="0.50"),
        rate("CacheRead", base_request_fee="0.25"),
        rate("CacheWrite", base_request_fee="0.25"),
    )
    observation = price(snapshot(rates=rows))
    assert observation.usd == Decimal("2")
    assert observation.complete is False
    assert observation.unknown_components == ("BaseRequestFee",)


@pytest.mark.unit
def test_missing_usage_or_rate_preserves_known_subtotal_and_provenance() -> None:
    card = snapshot(rates=(rate("Input"), rate("Output")))
    observation = price(card, ObservedUsage(1000, None, 50, 0))
    assert observation.usd == Decimal("1")
    assert observation.known_subtotal_usd == Decimal("1")
    assert observation.complete is False
    assert observation.unknown_components == ("Output", "CacheRead", "CacheWrite")
    assert observation.provenance is not None
    assert observation.provenance.rate_ids == ("rate-input",)
    assert observation.provenance.price_ids == ("price-input",)
    assert observation.provenance.sync_ids == ("sync-alias", "sync-rate")
    assert observation.provenance.snapshot_start == NOW - timedelta(hours=1)
    assert type(observation.provenance).from_dict(observation.provenance.to_dict()) == observation.provenance


@pytest.mark.unit
def test_omitted_occurrence_time_uses_snapshot_start_inside_half_open_interval() -> None:
    card = PricingSnapshot(
        account_id="acct-1",
        interval_start=NOW,
        interval_end=NOW + timedelta(hours=1),
        alias_type="ccm:model_alias",
        rate_type="ccm:resolved_rate",
        aliases=(alias(effective_from=NOW, effective_to=NOW + timedelta(hours=1)),),
        rates=tuple(
            rate(kind, effective_from=NOW, effective_to=NOW + timedelta(hours=1))
            for kind in ("Input", "Output", "CacheRead", "CacheWrite")
        ),
    )
    observation = price_observed_usage(
        "openai",
        "gpt-4o",
        DEFAULT_USAGE,
        pricing_provider=ResolvedRateCardProvider(card),
    )
    assert observation.complete is True


@pytest.mark.unit
def test_injected_snapshot_provider_is_used() -> None:
    provider = ResolvedRateCardProvider(snapshot())
    observation = price_observed_usage(
        "openai",
        "gpt-4o",
        DEFAULT_USAGE,
        occurred_at=NOW,
        pricing_provider=provider,
    )
    assert observation.source == "udp-resolved-rate-card"
    assert observation.usd == Decimal("2")


@pytest.mark.unit
def test_legacy_provider_without_dimensions_keyword_is_used_when_dimensions_are_none() -> None:
    class LegacyProvider:
        def price_observed_usage(
            self,
            provider: str | None,
            model: str | None,
            usage: ObservedUsage,
            *,
            occurred_at: datetime | None = None,
        ) -> CostObservation:
            assert (provider, model, usage, occurred_at) == ("openai", "gpt-4o", DEFAULT_USAGE, NOW)
            return CostObservation(Decimal("7"), True, provider, model, model, "legacy", "v1")

    observation = price_observed_usage(
        "openai",
        "gpt-4o",
        DEFAULT_USAGE,
        occurred_at=NOW,
        pricing_provider=LegacyProvider(),
        dimensions=None,
    )
    assert observation.usd == Decimal("7")
    assert observation.source == "legacy"


@pytest.mark.unit
def test_typeerror_raised_inside_legacy_provider_is_not_swallowed() -> None:
    class BrokenLegacyProvider:
        def price_observed_usage(
            self,
            provider: str | None,
            model: str | None,
            usage: ObservedUsage,
            *,
            occurred_at: datetime | None = None,
        ) -> CostObservation:
            raise TypeError("provider bug")

    with pytest.raises(TypeError, match="provider bug"):
        price_observed_usage(
            "openai",
            "gpt-4o",
            DEFAULT_USAGE,
            pricing_provider=BrokenLegacyProvider(),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "model", "aliases", "rates", "unknown"),
    [
        (None, "gpt-4o", None, None, ("Identity",)),
        ("openai", None, None, None, ("Identity",)),
        ("openai", "missing", None, None, ("Identity",)),
        ("openai", "gpt-4o", (), None, ("Identity",)),
        ("openai", "gpt-4o", None, (), ("Input", "Output", "CacheRead", "CacheWrite")),
    ],
)
def test_missing_identity_alias_or_rates_are_incomplete(
    provider: str | None,
    model: str | None,
    aliases: tuple[ModelAliasRow, ...] | None,
    rates: tuple[ResolvedRateRow, ...] | None,
    unknown: tuple[str, ...],
) -> None:
    card = snapshot(
        aliases=(alias(),) if aliases is None else aliases,
        rates=(rate("Input"), rate("Output"), rate("CacheRead"), rate("CacheWrite")) if rates is None else rates,
    )
    observation = price_observed_usage(
        provider,
        model,
        ObservedUsage(1000, 1000, 0, 0),
        occurred_at=NOW,
        pricing_provider=ResolvedRateCardProvider(card),
    )
    assert observation.complete is False
    assert observation.unknown_components == unknown
    assert observation.provenance is not None
    assert observation.provenance.account_key == "acct-1"
    assert observation.provenance.alias_type == "ccm:model_alias"
    assert observation.provenance.rate_type == "ccm:resolved_rate"
    assert observation.provenance.snapshot_start == NOW - timedelta(hours=1)
    assert observation.provenance.snapshot_end == NOW + timedelta(hours=1)


@pytest.mark.unit
def test_unsupported_currency_is_incomplete_without_estimate() -> None:
    rows = (
        rate("Input", currency="EUR"),
        rate("Output"),
        rate("CacheRead"),
        rate("CacheWrite"),
    )
    observation = price(snapshot(rates=rows))
    assert observation.usd == Decimal("1")
    assert observation.complete is False
    assert observation.unknown_components == ("Input",)


@pytest.mark.unit
def test_no_injected_provider_returns_udp_unavailable_without_network_fallback() -> None:
    observation = price_observed_usage("openai", "gpt-4o", ObservedUsage(1, 1))
    source = Path(__file__).parents[2] / "src" / "harness_evals" / "cost" / "pricing.py"
    package = Path(__file__).parents[2] / "src" / "harness_evals" / "cost" / "__init__.py"

    assert observation.usd is None
    assert observation.complete is False
    assert observation.source == "udp-unavailable"
    assert "genai_prices" not in source.read_text()
    assert "genai_prices" not in package.read_text()


@pytest.mark.unit
def test_pricing_has_no_catalog_dependency_or_backend_imports() -> None:
    root = Path(__file__).parents[2]
    source = root / "src" / "harness_evals" / "cost" / "pricing.py"
    pyproject = root / "pyproject.toml"
    lock = root / "poetry.lock"
    tree = ast.parse(source.read_text())
    imported_roots = {
        name.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert "genai-prices" not in pyproject.read_text()
    assert "genai-prices" not in lock.read_text()
    assert imported_roots.isdisjoint({"genai_prices", "httpx", "requests", "urllib"})
    assert "harness_evals.api" not in source.read_text()
    assert "service." not in source.read_text()
