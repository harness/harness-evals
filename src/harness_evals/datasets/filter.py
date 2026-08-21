"""Filter dataset goldens by id or tags."""

from __future__ import annotations

from typing import Any

from harness_evals.errors import HarnessEvalsError

GoldenTagFilter = dict[str, str | list[str]]


def resolve_golden_id(golden: object) -> str | None:
    """Return the stable id for a golden, if one is defined."""
    golden_id = getattr(golden, "id", None)
    if golden_id:
        return str(golden_id)
    metadata = getattr(golden, "metadata", None) or {}
    if isinstance(metadata, dict):
        meta_id = metadata.get("golden_id") or metadata.get("id")
        if meta_id:
            return str(meta_id)
    return None


def parse_golden_ids(value: str | list[str] | None) -> list[str] | None:
    """Parse ``golden_ids`` from YAML or CLI (comma-separated string or list)."""
    if value is None:
        return None
    if isinstance(value, str):
        ids = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        ids = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise HarnessEvalsError(
            f"'golden_ids' must be a comma-separated string or list of strings, got {type(value).__name__}"
        )
    if not ids:
        raise HarnessEvalsError("'golden_ids' must contain at least one id")
    return ids


def filter_goldens_by_ids(goldens: list, golden_ids: list[str]) -> list:
    """Return goldens matching *golden_ids*, preserving the requested order."""
    by_id: dict[str, object] = {}
    for golden in goldens:
        gid = resolve_golden_id(golden)
        if gid is not None:
            by_id[gid] = golden

    missing = [gid for gid in golden_ids if gid not in by_id]
    if missing:
        available = ", ".join(sorted(by_id)) if by_id else "(none with ids in dataset)"
        raise HarnessEvalsError(
            f"golden_ids not found in dataset: {', '.join(missing)}. Available: {available}"
        )

    return [by_id[gid] for gid in golden_ids]


def resolve_golden_tags(golden: object) -> dict[str, str]:
    """Return tag key/value pairs on a golden."""
    tags = getattr(golden, "tags", None)
    if not isinstance(tags, dict):
        return {}
    return {str(key): str(value) for key, value in tags.items()}


def parse_modules(value: str | list[str] | None) -> list[str] | None:
    """Parse ``modules`` from YAML or CLI (comma-separated string or list)."""
    return parse_golden_ids(value)


def parse_golden_tags(value: str | dict[str, Any] | None) -> GoldenTagFilter | None:
    """Parse ``golden_tags`` from YAML or CLI ``key=value`` pairs."""
    if value is None:
        return None
    if isinstance(value, dict):
        parsed: GoldenTagFilter = {}
        for key, raw in value.items():
            tag_key = str(key).strip()
            if not tag_key:
                continue
            if isinstance(raw, list):
                values = [str(part).strip() for part in raw if str(part).strip()]
                if not values:
                    raise HarnessEvalsError(f"'golden_tags.{tag_key}' must contain at least one value")
                parsed[tag_key] = values if len(values) > 1 else values[0]
            elif raw is not None:
                text = str(raw).strip()
                if text:
                    parsed[tag_key] = text
        return parsed or None
    if isinstance(value, str):
        parsed = {}
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise HarnessEvalsError(
                    f"Invalid golden tag filter {part!r}; expected key=value pairs "
                    "(e.g. module=ci,scenario_type=write). Use --modules for module-only filters."
                )
            key, tag_value = part.split("=", 1)
            key = key.strip()
            tag_value = tag_value.strip()
            if not key or not tag_value:
                raise HarnessEvalsError(f"Invalid golden tag filter {part!r}; key and value must be non-empty")
            existing = parsed.get(key)
            if existing is None:
                parsed[key] = tag_value
            elif isinstance(existing, list):
                existing.append(tag_value)
            else:
                parsed[key] = [existing, tag_value]
        if not parsed:
            raise HarnessEvalsError("'golden_tags' must contain at least one key=value pair")
        return parsed
    raise HarnessEvalsError(
        f"'golden_tags' must be a comma-separated key=value string or dict, got {type(value).__name__}"
    )


def merge_golden_tag_filter(
    *,
    modules: list[str] | None,
    golden_tags: GoldenTagFilter | None,
) -> GoldenTagFilter | None:
    """Combine ``modules`` shorthand with explicit ``golden_tags`` (AND across keys)."""
    merged: GoldenTagFilter = {}
    if golden_tags:
        merged.update(golden_tags)
    if modules:
        merged["module"] = modules
    return merged or None


def golden_matches_tag_filter(golden: object, tag_filter: GoldenTagFilter) -> bool:
    """Return True when a golden satisfies every tag constraint."""
    tags = resolve_golden_tags(golden)
    for key, expected in tag_filter.items():
        actual = tags.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def summarize_tag_values(goldens: list) -> str:
    """Summarize tag values present in a dataset for error messages."""
    by_key: dict[str, set[str]] = {}
    for golden in goldens:
        for key, value in resolve_golden_tags(golden).items():
            by_key.setdefault(key, set()).add(value)
    if not by_key:
        return "(no tags on any golden)"
    return "; ".join(f"{key}={sorted(values)}" for key, values in sorted(by_key.items()))


def filter_goldens_by_tags(goldens: list, tag_filter: GoldenTagFilter) -> list:
    """Return goldens whose tags match *tag_filter*, preserving dataset order."""
    matched = [golden for golden in goldens if golden_matches_tag_filter(golden, tag_filter)]
    if not matched:
        raise HarnessEvalsError(
            f"No goldens matched tag filter {tag_filter!r}. Available tag values: {summarize_tag_values(goldens)}"
        )
    return matched
