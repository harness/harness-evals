"""Resource reference parsing for dataset, prompt, target, and config adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlsplit


@dataclass(frozen=True)
class ResourceRef:
    """Normalized reference to an adapter-backed resource."""

    source: str
    id: str
    version: str | None = None
    # ``frozen=True`` is shallow: callers may still mutate entries in this dict.
    extra: dict[str, Any] = field(default_factory=dict, compare=False)


def resolve(spec: str | dict[str, Any]) -> ResourceRef:
    """Normalize a bare path, URI shorthand, or typed dict into a ``ResourceRef``."""

    if isinstance(spec, str):
        return _resolve_string(spec)
    if isinstance(spec, dict):
        return _resolve_dict(spec)
    raise TypeError(f"Resource spec must be a string or dict, got {type(spec).__name__}")


def _resolve_string(spec: str) -> ResourceRef:
    if "://" not in spec:
        return ResourceRef(source="local", id=spec)

    parsed = urlsplit(spec)
    if not parsed.scheme:
        raise ValueError(f"Resource URI {spec!r} is missing a source scheme")

    raw_id = f"{parsed.netloc}{parsed.path}"
    resource_id, version = _split_version(raw_id)
    extra = dict(parse_qsl(parsed.query, keep_blank_values=True))

    return ResourceRef(
        source=parsed.scheme,
        id=resource_id,
        version=str(version) if version is not None else None,
        extra=extra,
    )


def _resolve_dict(spec: dict[str, Any]) -> ResourceRef:
    missing = [key for key in ("source", "id") if key not in spec]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Resource spec is missing required key(s): {joined}")

    source = spec["source"]
    resource_id = spec["id"]
    if not isinstance(source, str):
        raise TypeError("Resource spec 'source' must be a string")
    if not isinstance(resource_id, str):
        raise TypeError("Resource spec 'id' must be a string")

    version = spec.get("version")
    extra = {key: value for key, value in spec.items() if key not in {"source", "id", "version"}}
    return ResourceRef(
        source=source,
        id=resource_id,
        version=str(version) if version is not None else None,
        extra=extra,
    )


def _split_version(raw_id: str) -> tuple[str, str | None]:
    if "@" not in raw_id:
        return raw_id, None
    resource_id, version = raw_id.rsplit("@", maxsplit=1)
    return resource_id, version
