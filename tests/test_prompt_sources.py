"""Tests for prompt source adapters."""

from __future__ import annotations

from urllib.request import Request

import pytest

from harness_evals import plugins
from harness_evals.prompts import HttpPromptSource, LocalPromptSource, PromptTemplate
from harness_evals.refs import ResourceRef


class FakeHTTPResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


@pytest.mark.unit
async def test_local_prompt_source_reads_text_file(tmp_path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Answer {{input}} with {{tone}} tone.", encoding="utf-8")

    source = LocalPromptSource()
    prompt = await source.fetch(ResourceRef(source="local", id=str(prompt_path), version="7", extra={"team": "ai"}))

    assert isinstance(prompt, PromptTemplate)
    assert prompt.input_variables == ["input", "tone"]
    assert prompt.version == "7"
    assert prompt.metadata == {"team": "ai"}
    assert prompt.render(input="hello", tone="friendly") == "Answer hello with friendly tone."


@pytest.mark.unit
async def test_local_prompt_source_reports_empty_variables_for_static_prompt(tmp_path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Always answer politely.", encoding="utf-8")

    prompt = await LocalPromptSource().fetch(ResourceRef(source="local", id=str(prompt_path)))

    assert prompt.input_variables == []
    assert prompt.render() == "Always answer politely."


@pytest.mark.unit
async def test_local_prompt_source_uses_thread_for_file_io(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.prompts import local as prompt_local

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Answer {{input}}.", encoding="utf-8")
    calls: list[str] = []
    original_to_thread = prompt_local.asyncio.to_thread

    async def spy_to_thread(func, /, *args, **kwargs):
        calls.append(func.__name__)
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(prompt_local.asyncio, "to_thread", spy_to_thread)

    prompt = await LocalPromptSource().fetch(ResourceRef(source="local", id=str(prompt_path)))

    assert calls == ["_read_prompt"]
    assert prompt.render(input="q") == "Answer q."


@pytest.mark.unit
async def test_http_prompt_source_reads_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.prompts import http as prompt_http

    requests: list[str] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        requests.append(request.full_url)
        assert timeout == 60.0
        return FakeHTTPResponse("Summarize {{input}}.")

    monkeypatch.setattr(prompt_http, "urlopen", fake_urlopen)

    prompt = await HttpPromptSource().fetch(ResourceRef(source="http", id="example.test/prompt@ignored", version="2"))

    assert requests == ["http://example.test/prompt@ignored"]
    assert prompt.version == "2"
    assert prompt.input_variables == ["input"]
    assert prompt.render(input="this") == "Summarize this."


@pytest.mark.unit
async def test_https_prompt_source_preserves_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.prompts import http as prompt_http

    requests: list[str] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        requests.append(request.full_url)
        return FakeHTTPResponse("Summarize {{input}}.")

    monkeypatch.setattr(prompt_http, "urlopen", fake_urlopen)

    prompt = await HttpPromptSource().fetch(ResourceRef(source="https", id="example.test/prompt"))

    assert requests == ["https://example.test/prompt"]
    assert prompt.render(input="secure") == "Summarize secure."


@pytest.mark.unit
async def test_prompt_source_context_manager() -> None:
    async with LocalPromptSource() as source:
        assert source.name == "local"


@pytest.mark.unit
def test_prompt_sources_are_registered() -> None:
    assert plugins.prompt_source("local") is LocalPromptSource
    assert plugins.prompt_source("http") is HttpPromptSource
    assert plugins.prompt_source("https") is HttpPromptSource
