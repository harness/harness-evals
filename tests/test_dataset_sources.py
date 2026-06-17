"""Tests for dataset source adapters."""

from __future__ import annotations

import json
from urllib.request import Request

import pytest

from harness_evals import plugins
from harness_evals.datasets import Dataset, HttpDatasetSource, LocalDatasetSource, load_dataset, save_dataset
from harness_evals.refs import ResourceRef


class FakeHTTPResponse:
    def __init__(self, body: str, content_type: str = "") -> None:
        self._body = body.encode("utf-8")
        self.headers = {"Content-Type": content_type}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


@pytest.mark.unit
def test_dataset_package_keeps_existing_public_imports() -> None:
    assert Dataset is not None
    assert callable(load_dataset)
    assert callable(save_dataset)


@pytest.mark.unit
async def test_local_dataset_source_fetches_jsonl(tmp_path) -> None:
    dataset_path = tmp_path / "goldens.jsonl"
    dataset_path.write_text('{"input": "q1", "expected": "a1"}\n{"input": "q2"}\n')

    source = LocalDatasetSource()
    dataset = await source.fetch(ResourceRef(source="local", id=str(dataset_path)))

    assert len(dataset) == 2
    assert dataset[0].input == "q1"
    assert dataset[0].expected == "a1"
    assert dataset[1].input == "q2"


@pytest.mark.unit
async def test_local_dataset_source_infers_json_format(tmp_path) -> None:
    dataset_path = tmp_path / "goldens.json"
    dataset_path.write_text(json.dumps([{"input": "q1"}, {"input": "q2"}]))

    dataset = await LocalDatasetSource().fetch(ResourceRef(source="local", id=str(dataset_path)))

    assert [golden.input for golden in dataset] == ["q1", "q2"]


@pytest.mark.unit
async def test_local_dataset_source_uses_thread_for_file_io(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.datasets import local as dataset_local

    dataset_path = tmp_path / "goldens.jsonl"
    dataset_path.write_text('{"input": "q"}\n')
    calls: list[str] = []
    original_to_thread = dataset_local.asyncio.to_thread

    async def spy_to_thread(func, /, *args, **kwargs):
        calls.append(func.__name__)
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(dataset_local.asyncio, "to_thread", spy_to_thread)

    dataset = await LocalDatasetSource().fetch(ResourceRef(source="local", id=str(dataset_path)))

    assert calls == ["load_dataset"]
    assert dataset[0].input == "q"


@pytest.mark.unit
async def test_http_dataset_source_uses_explicit_format(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.datasets import http as dataset_http

    requests: list[str] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        requests.append(request.full_url)
        assert timeout == 60.0
        return FakeHTTPResponse('{"input": "q1"}\n{"input": "q2"}\n', "application/json")

    monkeypatch.setattr(dataset_http, "urlopen", fake_urlopen)

    dataset = await HttpDatasetSource().fetch(
        ResourceRef(source="http", id="example.test/goldens", extra={"format": "jsonl"})
    )

    assert requests == ["http://example.test/goldens"]
    assert [golden.input for golden in dataset] == ["q1", "q2"]


@pytest.mark.unit
async def test_https_dataset_source_preserves_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.datasets import http as dataset_http

    requests: list[str] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        requests.append(request.full_url)
        return FakeHTTPResponse('[{"input": "secure"}]', "application/json")

    monkeypatch.setattr(dataset_http, "urlopen", fake_urlopen)

    dataset = await HttpDatasetSource().fetch(ResourceRef(source="https", id="example.test/goldens"))

    assert requests == ["https://example.test/goldens"]
    assert dataset[0].input == "secure"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("body", "content_type", "expected"),
    [
        ('[{"input": "json"}]', "application/json", ["json"]),
        ('{"input": "jsonl"}\nnot json\n', "application/x-ndjson", ["jsonl"]),
        ('[{"input": "sniffed"}]', "text/plain", ["sniffed"]),
    ],
)
async def test_http_dataset_source_detects_format(
    body: str,
    content_type: str,
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harness_evals.datasets import http as dataset_http

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse(body, content_type)

    monkeypatch.setattr(dataset_http, "urlopen", fake_urlopen)

    dataset = await HttpDatasetSource().fetch(ResourceRef(source="http", id="example.test/goldens"))

    assert [golden.input for golden in dataset] == expected


@pytest.mark.unit
async def test_http_dataset_source_rejects_top_level_json_object(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.datasets import http as dataset_http

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse('{"input": "single"}', "text/plain")

    monkeypatch.setattr(dataset_http, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="JSON array"):
        await HttpDatasetSource().fetch(ResourceRef(source="http", id="example.test/goldens"))


@pytest.mark.unit
async def test_dataset_source_context_manager() -> None:
    async with LocalDatasetSource() as source:
        assert source.name == "local"


@pytest.mark.unit
def test_dataset_sources_are_registered() -> None:
    assert plugins.dataset_source("local") is LocalDatasetSource
    assert plugins.dataset_source("http") is HttpDatasetSource
    assert plugins.dataset_source("https") is HttpDatasetSource
