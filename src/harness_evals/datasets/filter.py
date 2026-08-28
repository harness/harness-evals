"""Filter dataset goldens by id or tags."""

from __future__ import annotations

from typing import Any

from harness_evals.errors import HarnessEvalsError

GoldenTagFilter = dict[str, str | list[str]]


def resolve_golden_id(golden: object) -> str | None:
    """Return the stable id for a golden, if one is defined."""
    golden_id = getattr(golden, "id", None)
    if golden_id is not None and str(golden_id).strip():
        return str(golden_id)
    metadata = getattr(golden, "metadata", None) or {}
    if isinstance(metadata, dict):
        for key in ("golden_id", "id", "langfuse_dataset_item_id"):
            meta_id = metadata.get(key)
            if meta_id is not None and str(meta_id).strip():
                return str(meta_id)
    return None


def parse_golden_ids(value: str | list[str] | None) -> list[str] | None:
    """Parse ``golden_ids`` from YAML or CLI (comma-separated string or list)."""
    return _parse_string_list(value, field_name="golden_ids", empty_item_label="id")


def _parse_string_list(
    value: str | list[str] | None,
    *,
    field_name: str,
    empty_item_label: str = "value",
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        items = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise HarnessEvalsError(
            f"'{field_name}' must be a comma-separated string or list of strings, got {type(value).__name__}"
        )
    if not items:
        raise HarnessEvalsError(f"'{field_name}' must contain at least one {empty_item_label}")
    return list(dict.fromkeys(items))


def filter_goldens_by_ids(goldens: list, golden_ids: list[str]) -> list:
    """Return goldens matching *golden_ids*, preserving the requested order.

    Duplicate ids elsewhere in the dataset are ignored unless a requested id is
    ambiguous (more than one golden shares that id).
    """
    golden_ids = list(dict.fromkeys(golden_ids))
    by_id: dict[str, object] = {}
    duplicates: set[str] = set()
    for golden in goldens:
        gid = resolve_golden_id(golden)
        if gid is None:
            continue
        if gid in by_id:
            duplicates.add(gid)
            continue
        by_id[gid] = golden

    ambiguous = next((gid for gid in golden_ids if gid in duplicates), None)
    if ambiguous is not None:
        raise HarnessEvalsError(f"Duplicate golden id {ambiguous!r} in dataset; ids must be unique to filter by id")

    missing = [gid for gid in golden_ids if gid not in by_id]
    if missing:
        available = ", ".join(sorted(by_id)) if by_id else "(none with ids in dataset)"
        raise HarnessEvalsError(f"golden_ids not found in dataset: {', '.join(missing)}. Available: {available}")

    return [by_id[gid] for gid in golden_ids]


def resolve_golden_tags(golden: object) -> dict[str, str]:
    """Return tag key/value pairs on a golden."""
    resolved: dict[str, str] = {}
    metadata = getattr(golden, "metadata", None)
    if isinstance(metadata, dict):
        metadata_tags = metadata.get("tags")
        if isinstance(metadata_tags, dict):
            resolved.update({str(key): str(value) for key, value in metadata_tags.items()})
        if "module" in metadata and metadata["module"] is not None:
            resolved.setdefault("module", str(metadata["module"]))
    tags = getattr(golden, "tags", None)
    if isinstance(tags, dict):
        resolved.update({str(key): str(value) for key, value in tags.items()})
    return resolved


def parse_modules(value: str | list[str] | None) -> list[str] | None:
    """Parse ``modules`` from YAML or CLI (comma-separated string or list)."""
    return _parse_string_list(value, field_name="modules", empty_item_label="module")


def intersect_string_filters(
    configured: list[str] | None,
    override: list[str],
    *,
    field_name: str,
) -> list[str]:
    """Apply a CLI filter as a further restriction of a configured allowlist."""
    if configured is None:
        return override
    intersection = [value for value in override if value in configured]
    if not intersection:
        raise HarnessEvalsError(
            f"CLI '{field_name}' {override!r} has no values in common with configured {field_name} {configured!r}"
        )
    return intersection


def parse_golden_tags(value: str | dict[str, Any] | None) -> GoldenTagFilter | None:
    """Parse ``golden_tags`` from YAML or CLI ``key=value`` pairs."""
    if value is None:
        return None
    if isinstance(value, dict):
        parsed: GoldenTagFilter = {}
        for key, raw in value.items():
            tag_key = str(key).strip()
            if not tag_key:
                raise HarnessEvalsError("'golden_tags' keys must be non-empty")
            if isinstance(raw, list):
                values = [str(part).strip() for part in raw if str(part).strip()]
                if not values:
                    raise HarnessEvalsError(f"'golden_tags.{tag_key}' must contain at least one value")
                parsed[tag_key] = values if len(values) > 1 else values[0]
            else:
                if raw is None:
                    raise HarnessEvalsError(f"'golden_tags.{tag_key}' must have a non-empty value")
                text = str(raw).strip()
                if not text:
                    raise HarnessEvalsError(f"'golden_tags.{tag_key}' must have a non-empty value")
                parsed[tag_key] = text
        if not parsed:
            raise HarnessEvalsError("'golden_tags' must contain at least one key/value pair")
        return parsed
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


def intersect_golden_tag_filters(
    configured: GoldenTagFilter | None,
    override: GoldenTagFilter,
) -> GoldenTagFilter:
    """Apply CLI tags as additional constraints without widening configured tags."""
    if configured is None:
        return override
    merged = dict(configured)
    for key, override_value in override.items():
        existing = merged.get(key)
        if existing is None:
            merged[key] = override_value
            continue
        existing_values = existing if isinstance(existing, list) else [existing]
        override_values = override_value if isinstance(override_value, list) else [override_value]
        intersection = [value for value in override_values if value in existing_values]
        if not intersection:
            raise HarnessEvalsError(
                f"CLI 'golden_tags' {key}={override_value!r} conflicts with configured golden_tags {key}={existing!r}"
            )
        merged[key] = intersection if len(intersection) > 1 else intersection[0]
    return merged


def merge_golden_tag_filter(
    *,
    modules: list[str] | None,
    golden_tags: GoldenTagFilter | None,
) -> GoldenTagFilter | None:
    """Combine ``modules`` shorthand with explicit ``golden_tags`` (AND across keys).

    ``modules`` and ``golden_tags['module']`` address the same tag key, so they are
    intersected rather than one silently overwriting the other.
    """
    merged: GoldenTagFilter = {}
    if golden_tags:
        merged.update(golden_tags)
    if modules:
        existing = merged.get("module")
        if existing is None:
            merged["module"] = modules
        else:
            existing_values = existing if isinstance(existing, list) else [existing]
            intersection = [value for value in modules if value in existing_values]
            if not intersection:
                raise HarnessEvalsError(
                    f"'modules' {modules!r} and 'golden_tags' module={existing!r} have no values in "
                    "common; both filter the same 'module' tag, so nothing can match."
                )
            merged["module"] = intersection if len(intersection) > 1 else intersection[0]
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
