"""Tests for LangfuseDatasetSource.

Uses mock objects to avoid requiring the langfuse package at test time.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from harness_evals.refs import ResourceRef


@dataclass
class _FakeDatasetItem:
    id: str = "item-001"
    input: Any = "What is your return policy?"
    expected_output: Any = "30-day returns on all items."
    metadata: dict | None = field(default_factory=lambda: {"category": "support"})
    status: str = "ACTIVE"
    source_trace_id: str | None = None
    dataset_name: str = "test-dataset"


@dataclass
class _FakePaginatedMeta:
    total_pages: int = 1


@dataclass
class _FakePaginatedItems:
    data: list[_FakeDatasetItem] = field(default_factory=list)
    meta: _FakePaginatedMeta = field(default_factory=_FakePaginatedMeta)


@pytest.fixture()
def _langfuse_module():
    """Inject a fake langfuse module so LangfuseDatasetSource can be imported without the real package."""
    fake_mod = ModuleType("langfuse")
    fake_mod.Langfuse = MagicMock
    already = "langfuse" in sys.modules
    old = sys.modules.get("langfuse")
    sys.modules["langfuse"] = fake_mod
    yield
    if already:
        sys.modules["langfuse"] = old
    else:
        del sys.modules["langfuse"]


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_fetch_returns_all_goldens() -> None:
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    items = [
        _FakeDatasetItem(id="item-1", input="q1", expected_output="a1"),
        _FakeDatasetItem(id="item-2", input="q2", expected_output="a2"),
    ]
    client = MagicMock()
    client.api.dataset_items.list.return_value = _FakePaginatedItems(data=items)

    source = LangfuseDatasetSource(client)
    goldens = await source.fetch(ResourceRef(source="langfuse", id="my-dataset"))

    assert len(goldens) == 2
    assert goldens[0].input == "q1"
    assert goldens[0].expected == "a1"
    assert goldens[1].input == "q2"
    assert goldens[1].expected == "a2"


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_fetch_iter_yields_page_by_page() -> None:
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    page1 = _FakePaginatedItems(
        data=[_FakeDatasetItem(id="item-1", input="q1", expected_output="a1")],
        meta=_FakePaginatedMeta(total_pages=2),
    )
    page2 = _FakePaginatedItems(
        data=[_FakeDatasetItem(id="item-2", input="q2", expected_output="a2")],
        meta=_FakePaginatedMeta(total_pages=2),
    )

    client = MagicMock()
    client.api.dataset_items.list.side_effect = [page1, page2]

    source = LangfuseDatasetSource(client)
    ref = ResourceRef(source="langfuse", id="my-dataset", extra={"fetch_items_page_size": "1"})

    goldens = []
    async for golden in source.fetch_iter(ref):
        goldens.append(golden)

    assert len(goldens) == 2
    assert goldens[0].input == "q1"
    assert goldens[1].input == "q2"
    assert client.api.dataset_items.list.call_count == 2


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_archived_items_skipped() -> None:
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    items = [
        _FakeDatasetItem(id="active", input="keep", status="ACTIVE"),
        _FakeDatasetItem(id="archived", input="skip", status="ARCHIVED"),
    ]
    client = MagicMock()
    client.api.dataset_items.list.return_value = _FakePaginatedItems(data=items)

    source = LangfuseDatasetSource(client)
    goldens = await source.fetch(ResourceRef(source="langfuse", id="test"))

    assert len(goldens) == 1
    assert goldens[0].input == "keep"


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_empty_dataset_returns_empty() -> None:
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    client = MagicMock()
    client.api.dataset_items.list.return_value = _FakePaginatedItems(data=[])

    source = LangfuseDatasetSource(client)
    goldens = await source.fetch(ResourceRef(source="langfuse", id="empty"))

    assert goldens == []


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_metadata_includes_provenance() -> None:
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    items = [
        _FakeDatasetItem(
            id="item-123",
            input="q",
            metadata={"custom": "value"},
            source_trace_id="trace-456",
        ),
    ]
    client = MagicMock()
    client.api.dataset_items.list.return_value = _FakePaginatedItems(data=items)

    source = LangfuseDatasetSource(client)
    goldens = await source.fetch(ResourceRef(source="langfuse", id="test"))

    assert goldens[0].metadata["langfuse_dataset_item_id"] == "item-123"
    assert goldens[0].metadata["langfuse_source_trace_id"] == "trace-456"
    assert goldens[0].metadata["custom"] == "value"


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_version_parsed_as_datetime() -> None:
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    client = MagicMock()
    client.api.dataset_items.list.return_value = _FakePaginatedItems(data=[])

    source = LangfuseDatasetSource(client)
    await source.fetch(ResourceRef(source="langfuse", id="test", version="2024-06-01T00:00:00+00:00"))

    call_kwargs = client.api.dataset_items.list.call_args[1]
    assert call_kwargs["version"] is not None
    assert call_kwargs["version"].year == 2024
    assert call_kwargs["version"].month == 6


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_version_non_parseable_fetches_latest_with_warning() -> None:
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    client = MagicMock()
    client.api.dataset_items.list.return_value = _FakePaginatedItems(data=[])

    source = LangfuseDatasetSource(client)
    with pytest.warns(UserWarning, match="not a valid ISO datetime"):
        await source.fetch(ResourceRef(source="langfuse", id="test", version="not-a-date"))

    call_kwargs = client.api.dataset_items.list.call_args[1]
    assert call_kwargs["version"] is None


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_strip_datasets_prefix() -> None:
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    client = MagicMock()
    client.api.dataset_items.list.return_value = _FakePaginatedItems(data=[])

    source = LangfuseDatasetSource(client)
    await source.fetch(ResourceRef(source="langfuse", id="datasets/my-dataset"))

    call_kwargs = client.api.dataset_items.list.call_args[1]
    assert call_kwargs["dataset_name"] == "my-dataset"


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_no_prefix_passes_through() -> None:
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    client = MagicMock()
    client.api.dataset_items.list.return_value = _FakePaginatedItems(data=[])

    source = LangfuseDatasetSource(client)
    await source.fetch(ResourceRef(source="langfuse", id="my-dataset"))

    call_kwargs = client.api.dataset_items.list.call_args[1]
    assert call_kwargs["dataset_name"] == "my-dataset"


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_empty_ref_id_raises() -> None:
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    client = MagicMock()
    source = LangfuseDatasetSource(client)

    with pytest.raises(ValueError, match="dataset name"):
        await source.fetch(ResourceRef(source="langfuse", id=""))


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_close_flushes_client() -> None:
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    client = MagicMock()
    source = LangfuseDatasetSource(client)
    await source.close()

    client.flush.assert_called_once()


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_custom_page_size() -> None:
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    client = MagicMock()
    client.api.dataset_items.list.return_value = _FakePaginatedItems(data=[])

    source = LangfuseDatasetSource(client)
    await source.fetch(ResourceRef(source="langfuse", id="test", extra={"fetch_items_page_size": "25"}))

    call_kwargs = client.api.dataset_items.list.call_args[1]
    assert call_kwargs["limit"] == 25


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_fetch_iter_stops_at_total_pages_boundary() -> None:
    """fetch_iter() stops when page_num reaches total_pages, even if page is full."""
    from harness_evals.datasets.langfuse import LangfuseDatasetSource

    full_page = _FakePaginatedItems(
        data=[_FakeDatasetItem(id=f"item-{i}", input=f"q{i}") for i in range(3)],
        meta=_FakePaginatedMeta(total_pages=2),
    )
    last_page = _FakePaginatedItems(
        data=[_FakeDatasetItem(id="item-last", input="qlast")],
        meta=_FakePaginatedMeta(total_pages=2),
    )

    client = MagicMock()
    client.api.dataset_items.list.side_effect = [full_page, last_page]

    source = LangfuseDatasetSource(client)
    ref = ResourceRef(source="langfuse", id="test", extra={"fetch_items_page_size": "3"})

    goldens = []
    async for golden in source.fetch_iter(ref):
        goldens.append(golden)

    assert len(goldens) == 4
    assert client.api.dataset_items.list.call_count == 2


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_plugin_registry_resolves_langfuse_source() -> None:
    """Verify that the @register_dataset_source decorator is correctly applied."""
    from harness_evals.datasets.langfuse import LangfuseDatasetSource
    from harness_evals.plugins import _REGISTRIES, DATASET_SOURCES, register_dataset_source

    # Re-register since conftest restores registries between tests
    register_dataset_source("langfuse")(LangfuseDatasetSource)
    assert _REGISTRIES[DATASET_SOURCES]["langfuse"] is LangfuseDatasetSource
