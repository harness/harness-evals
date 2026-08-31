"""Offline pricing from backend-neutral UDP resolved-rate-card snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

GLOBAL_SCOPE = "__GLOBAL__"
UDP_PRICING_SOURCE = "udp-resolved-rate-card"
UDP_UNAVAILABLE_SOURCE = "udp-unavailable"

_USAGE_TYPES = ("Input", "Output", "CacheRead", "CacheWrite")
_ALIAS_KINDS = frozenset({"EXACT", "ARN", "PATH", "DEPLOYMENT", "PREFIX", "CONTAINS"})
_PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "openai": "openai",
    "bedrock": "aws",
    "aws.bedrock": "aws",
    "gcp.vertex_ai": "gcp",
    "vertex": "gcp",
    "google": "gcp",
}


def _strict_fields(payload: Mapping[str, Any], expected: frozenset[str], type_name: str) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{type_name} must be an object")
    actual = frozenset(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{type_name} fields mismatch: missing={missing}, unknown={unknown}")


def _string(value: object, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value:
        raise ValueError(f"{field} must not be empty")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _optional_integer(value: object, field: str, *, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(value, field, minimum=minimum)


def _decimal(value: object, field: str) -> Decimal:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field} must be a decimal string") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field} must be finite")
    return parsed


def _utc_datetime(value: object, field: str, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be an ISO datetime string")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO datetime string") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime_wire(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _active_at(start: datetime, end: datetime | None, occurred_at: datetime) -> bool:
    return _as_utc(start) <= occurred_at and (end is None or occurred_at < _as_utc(end))


@dataclass(frozen=True)
class ObservedUsage:
    """Actual usage reported by a model provider."""

    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


@dataclass(frozen=True)
class RateDimension:
    """One nullable rate-card matching dimension."""

    name: str
    value: str | None

    _FIELDS = frozenset({"name", "value"})

    def __post_init__(self) -> None:
        _string(self.name, "RateDimension.name")
        _string(self.value, "RateDimension.value", optional=True)

    def to_dict(self) -> dict[str, str | None]:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RateDimension:
        _strict_fields(payload, cls._FIELDS, cls.__name__)
        return cls(
            name=_string(payload["name"], "RateDimension.name"),  # type: ignore[arg-type]
            value=_string(payload["value"], "RateDimension.value", optional=True),
        )


@dataclass(frozen=True)
class ModelAliasRow:
    """Serializable UDP model-alias row used before rate lookup."""

    alias_id: str
    account_id: str
    provider: str
    alias: str
    canonical_model: str
    match_kind: str
    scope_priority: int
    alias_tier: int
    match_specificity: int
    active: bool
    effective_from: datetime
    effective_to: datetime | None
    sync_id: str

    _FIELDS = frozenset(
        {
            "alias_id",
            "account_id",
            "provider",
            "alias",
            "canonical_model",
            "match_kind",
            "scope_priority",
            "alias_tier",
            "match_specificity",
            "active",
            "effective_from",
            "effective_to",
            "sync_id",
        }
    )

    def __post_init__(self) -> None:
        for field in ("alias_id", "account_id", "provider", "alias", "canonical_model", "sync_id"):
            _string(getattr(self, field), f"ModelAliasRow.{field}")
        if self.match_kind not in _ALIAS_KINDS:
            raise ValueError(f"unsupported alias match kind: {self.match_kind}")
        _integer(self.scope_priority, "ModelAliasRow.scope_priority")
        _integer(self.alias_tier, "ModelAliasRow.alias_tier")
        _integer(self.match_specificity, "ModelAliasRow.match_specificity")
        if not isinstance(self.active, bool):
            raise TypeError("ModelAliasRow.active must be a boolean")
        if not isinstance(self.effective_from, datetime):
            raise TypeError("ModelAliasRow.effective_from must be a datetime")
        if self.effective_to is not None and not isinstance(self.effective_to, datetime):
            raise TypeError("ModelAliasRow.effective_to must be a datetime or null")
        if self.effective_to is not None and _as_utc(self.effective_to) <= _as_utc(self.effective_from):
            raise ValueError("ModelAliasRow effective interval must be increasing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias_id": self.alias_id,
            "account_id": self.account_id,
            "provider": self.provider,
            "alias": self.alias,
            "canonical_model": self.canonical_model,
            "match_kind": self.match_kind,
            "scope_priority": self.scope_priority,
            "alias_tier": self.alias_tier,
            "match_specificity": self.match_specificity,
            "active": self.active,
            "effective_from": _datetime_wire(self.effective_from),
            "effective_to": _datetime_wire(self.effective_to) if self.effective_to is not None else None,
            "sync_id": self.sync_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelAliasRow:
        _strict_fields(payload, cls._FIELDS, cls.__name__)
        active = payload["active"]
        if not isinstance(active, bool):
            raise TypeError("ModelAliasRow.active must be a boolean")
        match_kind = _string(payload["match_kind"], "ModelAliasRow.match_kind")
        if match_kind not in _ALIAS_KINDS:
            raise ValueError(f"unsupported alias match kind: {match_kind}")
        return cls(
            alias_id=_string(payload["alias_id"], "ModelAliasRow.alias_id"),  # type: ignore[arg-type]
            account_id=_string(payload["account_id"], "ModelAliasRow.account_id"),  # type: ignore[arg-type]
            provider=_string(payload["provider"], "ModelAliasRow.provider"),  # type: ignore[arg-type]
            alias=_string(payload["alias"], "ModelAliasRow.alias"),  # type: ignore[arg-type]
            canonical_model=_string(payload["canonical_model"], "ModelAliasRow.canonical_model"),  # type: ignore[arg-type]
            match_kind=match_kind,
            scope_priority=_integer(payload["scope_priority"], "ModelAliasRow.scope_priority"),
            alias_tier=_integer(payload["alias_tier"], "ModelAliasRow.alias_tier"),
            match_specificity=_integer(payload["match_specificity"], "ModelAliasRow.match_specificity"),
            active=active,
            effective_from=_utc_datetime(payload["effective_from"], "ModelAliasRow.effective_from"),  # type: ignore[arg-type]
            effective_to=_utc_datetime(payload["effective_to"], "ModelAliasRow.effective_to", optional=True),
            sync_id=_string(payload["sync_id"], "ModelAliasRow.sync_id"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ResolvedRateRow:
    """Serializable UDP rate row after backend-side price resolution."""

    rate_id: str
    price_id: str
    account_id: str
    provider: str
    model: str
    usage_type: str
    unit_price: Decimal
    currency: str
    unit_block_size: int
    min_charge_units: int
    tier_min_units: int
    tier_max_units: int | None
    dimensions: tuple[RateDimension, ...]
    effective_from: datetime
    effective_to: datetime | None
    base_request_fee: Decimal
    sync_id: str

    _FIELDS = frozenset(
        {
            "rate_id",
            "price_id",
            "account_id",
            "provider",
            "model",
            "usage_type",
            "unit_price",
            "currency",
            "unit_block_size",
            "min_charge_units",
            "tier_min_units",
            "tier_max_units",
            "dimensions",
            "effective_from",
            "effective_to",
            "base_request_fee",
            "sync_id",
        }
    )

    def __post_init__(self) -> None:
        for field in ("rate_id", "price_id", "account_id", "provider", "model", "currency", "sync_id"):
            _string(getattr(self, field), f"ResolvedRateRow.{field}")
        if self.usage_type not in _USAGE_TYPES:
            raise ValueError(f"unsupported usage type: {self.usage_type}")
        if not isinstance(self.unit_price, Decimal) or not self.unit_price.is_finite() or self.unit_price < 0:
            raise ValueError("ResolvedRateRow.unit_price must be a finite non-negative Decimal")
        if (
            not isinstance(self.base_request_fee, Decimal)
            or not self.base_request_fee.is_finite()
            or self.base_request_fee < 0
        ):
            raise ValueError("ResolvedRateRow.base_request_fee must be a finite non-negative Decimal")
        _integer(self.unit_block_size, "ResolvedRateRow.unit_block_size", minimum=1)
        _integer(self.min_charge_units, "ResolvedRateRow.min_charge_units")
        _integer(self.tier_min_units, "ResolvedRateRow.tier_min_units")
        _optional_integer(self.tier_max_units, "ResolvedRateRow.tier_max_units", minimum=1)
        if self.tier_max_units is not None and self.tier_max_units <= self.tier_min_units:
            raise ValueError("ResolvedRateRow tier interval must be increasing")
        if not isinstance(self.dimensions, tuple) or not all(
            isinstance(item, RateDimension) for item in self.dimensions
        ):
            raise TypeError("ResolvedRateRow.dimensions must be a tuple of RateDimension")
        if len({item.name for item in self.dimensions}) != len(self.dimensions):
            raise ValueError("ResolvedRateRow dimensions must have unique names")
        if not isinstance(self.effective_from, datetime):
            raise TypeError("ResolvedRateRow.effective_from must be a datetime")
        if self.effective_to is not None and not isinstance(self.effective_to, datetime):
            raise TypeError("ResolvedRateRow.effective_to must be a datetime or null")
        if self.effective_to is not None and _as_utc(self.effective_to) <= _as_utc(self.effective_from):
            raise ValueError("ResolvedRateRow effective interval must be increasing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate_id": self.rate_id,
            "price_id": self.price_id,
            "account_id": self.account_id,
            "provider": self.provider,
            "model": self.model,
            "usage_type": self.usage_type,
            "unit_price": str(self.unit_price),
            "currency": self.currency,
            "unit_block_size": self.unit_block_size,
            "min_charge_units": self.min_charge_units,
            "tier_min_units": self.tier_min_units,
            "tier_max_units": self.tier_max_units,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "effective_from": _datetime_wire(self.effective_from),
            "effective_to": _datetime_wire(self.effective_to) if self.effective_to is not None else None,
            "base_request_fee": str(self.base_request_fee),
            "sync_id": self.sync_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResolvedRateRow:
        _strict_fields(payload, cls._FIELDS, cls.__name__)
        dimensions = payload["dimensions"]
        if not isinstance(dimensions, list):
            raise TypeError("ResolvedRateRow.dimensions must be a list")
        usage_type = _string(payload["usage_type"], "ResolvedRateRow.usage_type")
        if usage_type not in _USAGE_TYPES:
            raise ValueError(f"unsupported usage type: {usage_type}")
        return cls(
            rate_id=_string(payload["rate_id"], "ResolvedRateRow.rate_id"),  # type: ignore[arg-type]
            price_id=_string(payload["price_id"], "ResolvedRateRow.price_id"),  # type: ignore[arg-type]
            account_id=_string(payload["account_id"], "ResolvedRateRow.account_id"),  # type: ignore[arg-type]
            provider=_string(payload["provider"], "ResolvedRateRow.provider"),  # type: ignore[arg-type]
            model=_string(payload["model"], "ResolvedRateRow.model"),  # type: ignore[arg-type]
            usage_type=usage_type,
            unit_price=_decimal(payload["unit_price"], "ResolvedRateRow.unit_price"),
            currency=_string(payload["currency"], "ResolvedRateRow.currency"),  # type: ignore[arg-type]
            unit_block_size=_integer(payload["unit_block_size"], "ResolvedRateRow.unit_block_size", minimum=1),
            min_charge_units=_integer(payload["min_charge_units"], "ResolvedRateRow.min_charge_units"),
            tier_min_units=_integer(payload["tier_min_units"], "ResolvedRateRow.tier_min_units"),
            tier_max_units=_optional_integer(payload["tier_max_units"], "ResolvedRateRow.tier_max_units", minimum=1),
            dimensions=tuple(RateDimension.from_dict(item) for item in dimensions),
            effective_from=_utc_datetime(payload["effective_from"], "ResolvedRateRow.effective_from"),  # type: ignore[arg-type]
            effective_to=_utc_datetime(payload["effective_to"], "ResolvedRateRow.effective_to", optional=True),
            base_request_fee=_decimal(payload["base_request_fee"], "ResolvedRateRow.base_request_fee"),
            sync_id=_string(payload["sync_id"], "ResolvedRateRow.sync_id"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PricingSnapshot:
    """A portable point-in-time UDP alias and resolved-rate-card response."""

    account_id: str
    interval_start: datetime
    interval_end: datetime
    alias_type: str
    rate_type: str
    aliases: tuple[ModelAliasRow, ...]
    rates: tuple[ResolvedRateRow, ...]

    _FIELDS = frozenset({"account_id", "interval_start", "interval_end", "alias_type", "rate_type", "aliases", "rates"})

    def __post_init__(self) -> None:
        for field in ("account_id", "alias_type", "rate_type"):
            _string(getattr(self, field), f"PricingSnapshot.{field}")
        if not isinstance(self.interval_start, datetime) or not isinstance(self.interval_end, datetime):
            raise TypeError("PricingSnapshot interval values must be datetimes")
        if _as_utc(self.interval_end) <= _as_utc(self.interval_start):
            raise ValueError("PricingSnapshot interval must be increasing")
        if not isinstance(self.aliases, tuple) or not all(isinstance(item, ModelAliasRow) for item in self.aliases):
            raise TypeError("PricingSnapshot.aliases must be a tuple of ModelAliasRow")
        if not isinstance(self.rates, tuple) or not all(isinstance(item, ResolvedRateRow) for item in self.rates):
            raise TypeError("PricingSnapshot.rates must be a tuple of ResolvedRateRow")

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "interval_start": _datetime_wire(self.interval_start),
            "interval_end": _datetime_wire(self.interval_end),
            "alias_type": self.alias_type,
            "rate_type": self.rate_type,
            "aliases": [item.to_dict() for item in self.aliases],
            "rates": [item.to_dict() for item in self.rates],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PricingSnapshot:
        _strict_fields(payload, cls._FIELDS, cls.__name__)
        aliases = payload["aliases"]
        rates = payload["rates"]
        if not isinstance(aliases, list):
            raise TypeError("PricingSnapshot.aliases must be a list")
        if not isinstance(rates, list):
            raise TypeError("PricingSnapshot.rates must be a list")
        return cls(
            account_id=_string(payload["account_id"], "PricingSnapshot.account_id"),  # type: ignore[arg-type]
            interval_start=_utc_datetime(payload["interval_start"], "PricingSnapshot.interval_start"),  # type: ignore[arg-type]
            interval_end=_utc_datetime(payload["interval_end"], "PricingSnapshot.interval_end"),  # type: ignore[arg-type]
            alias_type=_string(payload["alias_type"], "PricingSnapshot.alias_type"),  # type: ignore[arg-type]
            rate_type=_string(payload["rate_type"], "PricingSnapshot.rate_type"),  # type: ignore[arg-type]
            aliases=tuple(ModelAliasRow.from_dict(item) for item in aliases),
            rates=tuple(ResolvedRateRow.from_dict(item) for item in rates),
        )


@dataclass(frozen=True)
class PricingProvenance:
    """Compact provenance for the rows contributing to an observation."""

    alias_type: str
    rate_type: str
    scopes: tuple[str, ...]
    alias_ids: tuple[str, ...]
    rate_ids: tuple[str, ...]
    price_ids: tuple[str, ...]
    sync_ids: tuple[str, ...]
    snapshot_start: datetime
    snapshot_end: datetime

    _FIELDS = frozenset(
        {
            "alias_type",
            "rate_type",
            "scopes",
            "alias_ids",
            "rate_ids",
            "price_ids",
            "sync_ids",
            "snapshot_start",
            "snapshot_end",
        }
    )

    def __post_init__(self) -> None:
        for field in ("alias_type", "rate_type"):
            _string(getattr(self, field), f"PricingProvenance.{field}")
        for field in ("scopes", "alias_ids", "rate_ids", "price_ids", "sync_ids"):
            value = getattr(self, field)
            if not isinstance(value, tuple) or not all(isinstance(item, str) and item for item in value):
                raise TypeError(f"PricingProvenance.{field} must be a tuple of non-empty strings")
        if not isinstance(self.snapshot_start, datetime) or not isinstance(self.snapshot_end, datetime):
            raise TypeError("PricingProvenance snapshot interval values must be datetimes")
        if _as_utc(self.snapshot_end) <= _as_utc(self.snapshot_start):
            raise ValueError("PricingProvenance snapshot interval must be increasing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias_type": self.alias_type,
            "rate_type": self.rate_type,
            "scopes": list(self.scopes),
            "alias_ids": list(self.alias_ids),
            "rate_ids": list(self.rate_ids),
            "price_ids": list(self.price_ids),
            "sync_ids": list(self.sync_ids),
            "snapshot_start": _datetime_wire(self.snapshot_start),
            "snapshot_end": _datetime_wire(self.snapshot_end),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PricingProvenance:
        _strict_fields(payload, cls._FIELDS, cls.__name__)

        def strings(field: str) -> tuple[str, ...]:
            value = payload[field]
            if not isinstance(value, list):
                raise TypeError(f"PricingProvenance.{field} must be a list")
            return tuple(_string(item, f"PricingProvenance.{field} item") for item in value)  # type: ignore[misc]

        return cls(
            alias_type=_string(payload["alias_type"], "PricingProvenance.alias_type"),  # type: ignore[arg-type]
            rate_type=_string(payload["rate_type"], "PricingProvenance.rate_type"),  # type: ignore[arg-type]
            scopes=strings("scopes"),
            alias_ids=strings("alias_ids"),
            rate_ids=strings("rate_ids"),
            price_ids=strings("price_ids"),
            sync_ids=strings("sync_ids"),
            snapshot_start=_utc_datetime(payload["snapshot_start"], "PricingProvenance.snapshot_start"),  # type: ignore[arg-type]
            snapshot_end=_utc_datetime(payload["snapshot_end"], "PricingProvenance.snapshot_end"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CostObservation:
    """A priced usage observation with an explicitly known Decimal subtotal."""

    usd: Decimal | None
    complete: bool
    provider: str | None
    requested_model: str | None
    resolved_model: str | None
    source: str
    catalog_version: str
    unknown_components: tuple[str, ...] = ()
    provenance: PricingProvenance | None = None

    @property
    def known_subtotal_usd(self) -> Decimal | None:
        return self.usd


class PricingProvider(Protocol):
    """Replaceable boundary for offline price resolution."""

    def price_observed_usage(
        self,
        provider: str | None,
        model: str | None,
        usage: ObservedUsage,
        *,
        occurred_at: datetime | None = None,
        dimensions: Mapping[str, str] | None = None,
    ) -> CostObservation: ...


def normalize_provider(provider: str | None) -> str | None:
    """Normalize known provider spellings while preserving unknown identities."""
    if not isinstance(provider, str) or not provider.strip():
        return None
    normalized = provider.strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def _valid_token_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _alias_matches(row: ModelAliasRow, requested_model: str) -> bool:
    alias_value = row.alias.casefold()
    requested = requested_model.casefold()
    if row.match_kind in {"EXACT", "ARN", "PATH", "DEPLOYMENT"}:
        return requested == alias_value
    if row.match_kind == "PREFIX":
        return requested.startswith(alias_value)
    return alias_value in requested


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


class ResolvedRateCardProvider:
    """Resolve and price usage solely from an injected UDP snapshot."""

    def __init__(self, snapshot: PricingSnapshot):
        self.snapshot = snapshot

    def _resolve_alias(
        self, provider: str, requested_model: str, occurred_at: datetime
    ) -> tuple[str | None, tuple[ModelAliasRow, ...]]:
        candidates = [
            row
            for row in self.snapshot.aliases
            if normalize_provider(row.provider) == provider
            and row.active
            and _active_at(row.effective_from, row.effective_to, occurred_at)
            and _alias_matches(row, requested_model)
            and row.account_id in {self.snapshot.account_id, GLOBAL_SCOPE}
        ]
        account_candidates = [row for row in candidates if row.account_id == self.snapshot.account_id]
        candidates = account_candidates or [row for row in candidates if row.account_id == GLOBAL_SCOPE]
        if not candidates:
            return None, ()
        best_rank = max((row.scope_priority, row.alias_tier, row.match_specificity) for row in candidates)
        winners = tuple(
            row for row in candidates if (row.scope_priority, row.alias_tier, row.match_specificity) == best_rank
        )
        models = {row.canonical_model.casefold() for row in winners}
        if len(models) != 1:
            return None, winners
        return winners[0].canonical_model, winners

    @staticmethod
    def _dimension_rank(
        row: ResolvedRateRow,
        request_dimensions: Mapping[str, str],
        dimension_names: frozenset[str],
    ) -> int | None:
        row_dimensions = {item.name: item.value for item in row.dimensions}
        exact = 0
        for name in dimension_names:
            requested = request_dimensions.get(name)
            row_value = row_dimensions.get(name)
            if requested is None:
                if row_value is not None:
                    return None
            elif row_value is not None:
                if row_value != requested:
                    return None
                exact += 1
        return exact

    def _resolve_rate(
        self,
        provider: str,
        model: str,
        usage_type: str,
        actual_units: int,
        occurred_at: datetime,
        request_dimensions: Mapping[str, str],
    ) -> tuple[ResolvedRateRow | None, tuple[ResolvedRateRow, ...], int | None]:
        identity_rows = [
            row
            for row in self.snapshot.rates
            if normalize_provider(row.provider) == provider
            and row.model.casefold() == model.casefold()
            and row.usage_type == usage_type
            and row.account_id in {self.snapshot.account_id, GLOBAL_SCOPE}
            and _active_at(row.effective_from, row.effective_to, occurred_at)
        ]
        dimension_names = frozenset(request_dimensions).union(
            item.name for row in identity_rows for item in row.dimensions
        )
        ranked: list[tuple[ResolvedRateRow, int, int]] = []
        for row in identity_rows:
            dimension_rank = self._dimension_rank(row, request_dimensions, dimension_names)
            if dimension_rank is None:
                continue
            billable = max(actual_units, row.min_charge_units)
            if row.tier_min_units <= billable and (row.tier_max_units is None or billable < row.tier_max_units):
                ranked.append((row, dimension_rank, billable))
        account_rows = [item for item in ranked if item[0].account_id == self.snapshot.account_id]
        ranked = account_rows or [item for item in ranked if item[0].account_id == GLOBAL_SCOPE]
        if not ranked:
            return None, (), None
        best_dimension_rank = max(item[1] for item in ranked)
        winners_with_units = [item for item in ranked if item[1] == best_dimension_rank]
        signatures = {
            (
                row.unit_price,
                row.currency,
                row.unit_block_size,
                row.base_request_fee,
                billable,
            )
            for row, _, billable in winners_with_units
        }
        winners = tuple(item[0] for item in winners_with_units)
        if len(signatures) != 1:
            return None, winners, None
        winner, _, billable = winners_with_units[0]
        return winner, winners, billable

    def price_observed_usage(
        self,
        provider: str | None,
        model: str | None,
        usage: ObservedUsage,
        *,
        occurred_at: datetime | None = None,
        dimensions: Mapping[str, str] | None = None,
    ) -> CostObservation:
        normalized_provider = normalize_provider(provider)
        requested_model = model.strip() if isinstance(model, str) and model.strip() else None
        occurred = _as_utc(occurred_at or self.snapshot.interval_start)
        request_dimensions = dict(dimensions or {})
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in request_dimensions.items()):
            raise TypeError("pricing dimensions must map strings to strings")

        if normalized_provider is None or requested_model is None:
            return self._identity_incomplete(normalized_provider, requested_model)

        resolved_model, alias_winners = self._resolve_alias(normalized_provider, requested_model, occurred)
        if resolved_model is None:
            return self._identity_incomplete(normalized_provider, requested_model)

        usage_values = (
            ("Input", usage.input_tokens),
            ("Output", usage.output_tokens),
            ("CacheRead", usage.cache_read_tokens),
            ("CacheWrite", usage.cache_write_tokens),
        )
        subtotal = Decimal(0)
        known_component = False
        unknown: list[str] = []
        matched_rates: list[ResolvedRateRow] = []
        for usage_type, token_count in usage_values:
            if not _valid_token_count(token_count):
                unknown.append(usage_type)
                continue
            winner, equivalent_winners, billable = self._resolve_rate(
                normalized_provider,
                resolved_model,
                usage_type,
                token_count,
                occurred,
                request_dimensions,
            )
            matched_rates.extend(equivalent_winners)
            if winner is None or billable is None or winner.currency.upper() != "USD":
                unknown.append(usage_type)
                continue
            subtotal += Decimal(billable) * winner.unit_price / Decimal(winner.unit_block_size)
            known_component = True

        fees = {row.base_request_fee for row in matched_rates if row.currency.upper() == "USD"}
        if len(fees) == 1:
            subtotal += next(iter(fees))
            known_component = True
        elif len(fees) > 1:
            unknown.append("BaseRequestFee")
        if any(row.currency.upper() != "USD" and row.base_request_fee != 0 for row in matched_rates):
            unknown.append("BaseRequestFee")

        provenance = PricingProvenance(
            alias_type=self.snapshot.alias_type,
            rate_type=self.snapshot.rate_type,
            scopes=_unique([row.account_id for row in (*alias_winners, *matched_rates)]),
            alias_ids=_unique([row.alias_id for row in alias_winners]),
            rate_ids=_unique([row.rate_id for row in matched_rates]),
            price_ids=_unique([row.price_id for row in matched_rates]),
            sync_ids=_unique([row.sync_id for row in (*alias_winners, *matched_rates)]),
            snapshot_start=self.snapshot.interval_start,
            snapshot_end=self.snapshot.interval_end,
        )
        return CostObservation(
            usd=subtotal if known_component else None,
            complete=not unknown,
            provider=normalized_provider,
            requested_model=requested_model,
            resolved_model=resolved_model,
            source=UDP_PRICING_SOURCE,
            catalog_version=",".join(provenance.sync_ids),
            unknown_components=tuple(unknown),
            provenance=provenance,
        )

    def _identity_incomplete(self, provider: str | None, requested_model: str | None) -> CostObservation:
        return CostObservation(
            usd=None,
            complete=False,
            provider=provider,
            requested_model=requested_model,
            resolved_model=None,
            source=UDP_PRICING_SOURCE,
            catalog_version="",
            unknown_components=("Identity",),
        )


class UnavailablePricingProvider:
    """Explicit default when no resolved UDP snapshot was injected."""

    def price_observed_usage(
        self,
        provider: str | None,
        model: str | None,
        usage: ObservedUsage,
        *,
        occurred_at: datetime | None = None,
        dimensions: Mapping[str, str] | None = None,
    ) -> CostObservation:
        del usage, occurred_at, dimensions
        return CostObservation(
            usd=None,
            complete=False,
            provider=normalize_provider(provider),
            requested_model=model.strip() if isinstance(model, str) and model.strip() else None,
            resolved_model=None,
            source=UDP_UNAVAILABLE_SOURCE,
            catalog_version="",
            unknown_components=("PricingProvider",),
        )


_DEFAULT_PROVIDER: PricingProvider = UnavailablePricingProvider()


def price_observed_usage(
    provider: str | None,
    model: str | None,
    usage: ObservedUsage,
    *,
    occurred_at: datetime | None = None,
    pricing_provider: PricingProvider | None = None,
    dimensions: Mapping[str, str] | None = None,
) -> CostObservation:
    """Price one observed provider call without performing network I/O."""
    selected = pricing_provider or _DEFAULT_PROVIDER
    if dimensions is None:
        # Preserve compatibility with existing custom providers implementing the
        # pre-dimensions protocol signature.
        return selected.price_observed_usage(provider, model, usage, occurred_at=occurred_at)
    return selected.price_observed_usage(
        provider,
        model,
        usage,
        occurred_at=occurred_at,
        dimensions=dimensions,
    )
