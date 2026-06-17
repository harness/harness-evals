"""Tests for ResourceRef parsing."""

import pytest

from harness_evals.refs import ResourceRef, resolve


@pytest.mark.unit
@pytest.mark.parametrize(
    ("spec", "source", "resource_id", "version", "extra"),
    [
        ("./goldens.jsonl", "local", "./goldens.jsonl", None, {}),
        ("langfuse://datasets/support-goldens@3", "langfuse", "datasets/support-goldens", "3", {}),
        (
            "http://example.test/goldens.jsonl?format=jsonl&token=",
            "http",
            "example.test/goldens.jsonl",
            None,
            {"format": "jsonl", "token": ""},
        ),
        (
            {"source": "langfuse", "id": "support-goldens", "version": 3, "environment": "prod"},
            "langfuse",
            "support-goldens",
            "3",
            {"environment": "prod"},
        ),
    ],
)
def test_resolve_supported_syntaxes(
    spec: str | dict,
    source: str,
    resource_id: str,
    version: str | None,
    extra: dict,
) -> None:
    ref = resolve(spec)

    assert ref.source == source
    assert ref.id == resource_id
    assert ref.version == version
    assert ref.extra == extra


@pytest.mark.unit
def test_resource_ref_version_coercion_makes_int_and_string_versions_equal() -> None:
    int_version = resolve({"source": "langfuse", "id": "support-goldens", "version": 3})
    str_version = resolve({"source": "langfuse", "id": "support-goldens", "version": "3"})

    assert int_version == str_version
    assert int_version.version == "3"


@pytest.mark.unit
def test_uri_version_splits_on_last_at_symbol() -> None:
    ref = resolve("custom://team/path/with@email@example@42?region=us")

    assert ref.source == "custom"
    assert ref.id == "team/path/with@email@example"
    assert ref.version == "42"
    assert ref.extra == {"region": "us"}


@pytest.mark.unit
def test_resource_ref_is_frozen_and_hashable() -> None:
    ref = ResourceRef(source="local", id="./goldens.jsonl", extra={"format": "jsonl"})

    assert hash(ref) == hash(ResourceRef(source="local", id="./goldens.jsonl", extra={"format": "csv"}))
    assert ref == ResourceRef(source="local", id="./goldens.jsonl", extra={"format": "csv"})
    with pytest.raises(AttributeError):
        ref.source = "http"  # type: ignore[misc]


@pytest.mark.unit
def test_resolve_rejects_missing_required_typed_keys() -> None:
    with pytest.raises(ValueError, match="source"):
        resolve({"id": "support-goldens"})

    with pytest.raises(ValueError, match="id"):
        resolve({"source": "langfuse"})


@pytest.mark.unit
def test_resolve_rejects_unsupported_spec_types() -> None:
    with pytest.raises(TypeError, match="string or dict"):
        resolve(["not", "a", "valid", "spec"])  # type: ignore[arg-type]
