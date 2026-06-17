"""Tests for target adapters."""

from __future__ import annotations

import json
import time
from urllib.error import URLError
from urllib.request import Request

import pytest

from harness_evals import plugins
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.llm.base import BaseLLM
from harness_evals.prompts.template import PromptTemplate
from harness_evals.targets import (
    ApiKeyAuth,
    BaseTarget,
    BasicAuth,
    BearerAuth,
    HttpTarget,
    NoAuth,
    PromptTarget,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class StubLLM(BaseLLM):
    """Returns a fixed string from generate()."""

    def __init__(self, response: str = "model output") -> None:
        self._response = response

    async def generate(self, prompt: str, **kwargs) -> str:
        return self._response

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return {}


class CloseableLLM(StubLLM):
    def __init__(self, response: str = "model output") -> None:
        super().__init__(response)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeHTTPResponse:
    def __init__(self, body: str, content_type: str = "application/json", status: int = 200) -> None:
        self._body = body.encode("utf-8")
        self.headers = {"Content-Type": content_type}
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


# ---------------------------------------------------------------------------
# BaseTarget protocol
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_base_target_context_manager() -> None:
    class DummyTarget(BaseTarget):
        closed = False

        async def ainvoke(self, golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output="ok")

        async def close(self) -> None:
            self.closed = True

    async with DummyTarget() as t:
        result = await t.ainvoke(Golden(input="hi"))
        assert result.output == "ok"
    assert t.closed


# ---------------------------------------------------------------------------
# PromptTarget
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_prompt_target_string_input() -> None:
    prompt = PromptTemplate(template="Answer: {{input}}", input_variables=["input"])
    target = PromptTarget(prompt=prompt, model=StubLLM("42"))

    golden = Golden(input="What is 6*7?", expected="42")
    result = await target.ainvoke(golden)

    assert result.output == "42"
    assert result.expected == "42"
    assert result.input == "What is 6*7?"
    assert result.latency_ms is not None
    assert result.latency_ms >= 0


@pytest.mark.unit
async def test_prompt_target_dict_input_json_serialized() -> None:
    prompt = PromptTemplate(template="Process: {{input}}", input_variables=["input"])
    target = PromptTarget(prompt=prompt, model=StubLLM("done"))

    golden = Golden(input={"query": "hello", "lang": "en"})
    result = await target.ainvoke(golden)

    assert result.output == "done"
    assert result.input == {"query": "hello", "lang": "en"}


@pytest.mark.unit
async def test_prompt_target_metadata_as_template_vars() -> None:
    prompt = PromptTemplate(template="{{input}} in {{tone}} tone", input_variables=["input", "tone"])
    target = PromptTarget(prompt=prompt, model=StubLLM("friendly answer"))

    golden = Golden(input="greet me", metadata={"tone": "friendly"})
    result = await target.ainvoke(golden)

    assert result.output == "friendly answer"


@pytest.mark.unit
async def test_prompt_target_preserves_golden_fields() -> None:
    prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
    target = PromptTarget(prompt=prompt, model=StubLLM("out"))

    golden = Golden(
        input="q",
        expected="a",
        context=["ctx1"],
        tags={"env": "test"},
        metadata={"key": "val"},
    )
    result = await target.ainvoke(golden)

    assert result.expected == "a"
    assert result.context == ["ctx1"]
    assert result.tags == {"env": "test"}
    assert result.metadata == {"key": "val"}


@pytest.mark.unit
async def test_prompt_target_closes_model() -> None:
    prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
    model = CloseableLLM("out")
    target = PromptTarget(prompt=prompt, model=model)

    await target.close()

    assert model.closed


# ---------------------------------------------------------------------------
# HttpTarget
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_http_target_successful_post(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    captured_requests: list[Request] = []

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        captured_requests.append(request)
        return FakeHTTPResponse(json.dumps({"output": "agent response", "tokens": 50}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(url="http://localhost:8080/run", token_count_path="$.tokens")
    golden = Golden(input="hello", expected="world")
    result = await target.ainvoke(golden)

    assert result.output == "agent response"
    assert result.token_count == 50
    assert result.latency_ms is not None
    assert result.latency_ms >= 0
    assert result.expected == "world"

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.full_url == "http://localhost:8080/run"
    assert req.method == "POST"
    body = json.loads(req.data.decode())
    assert body == {"input": "hello"}


@pytest.mark.unit
async def test_http_target_with_body_template(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        return FakeHTTPResponse(json.dumps({"output": "ok"}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(
        url="http://localhost:8080/api",
        body_template={"data": {"query": None, "mode": "eval"}},
        input_path="$.data.query",
    )
    golden = Golden(input="test query")
    result = await target.ainvoke(golden)
    assert result.output == "ok"


@pytest.mark.unit
async def test_http_target_text_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        return FakeHTTPResponse("plain text answer", content_type="text/plain")

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(url="http://localhost:8080/run", output_path="$")
    golden = Golden(input="hi")
    result = await target.ainvoke(golden)
    assert result.output == "plain text answer"


@pytest.mark.unit
async def test_http_target_retries_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    call_count = 0

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise URLError("Connection refused")
        return FakeHTTPResponse(json.dumps({"output": "recovered"}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    target = HttpTarget(url="http://localhost:8080/run", retries=2, backoff_s=0.01)
    golden = Golden(input="test")
    result = await target.ainvoke(golden)

    assert result.output == "recovered"
    assert call_count == 3


@pytest.mark.unit
async def test_http_target_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        raise URLError("Connection refused")

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    target = HttpTarget(url="http://localhost:8080/run", retries=1, backoff_s=0.01)
    golden = Golden(input="test")
    result = await target.ainvoke(golden)

    assert result.output == ""
    assert result.metadata is not None
    assert "http_error" in result.metadata
    assert "Connection refused" in result.metadata["http_error"]


@pytest.mark.unit
async def test_http_target_failure_latency_uses_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    times = iter([0.0, 0.01, 100.0, 100.02])

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        raise URLError("Connection refused")

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)
    monkeypatch.setattr(target_http.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(time, "sleep", lambda _: None)

    target = HttpTarget(url="http://localhost:8080/run", retries=1, backoff_s=0.01)
    result = await target.ainvoke(Golden(input="test"))

    assert result.latency_ms == pytest.approx(20.0)


@pytest.mark.unit
async def test_http_target_extracts_all_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    response_data = {
        "output": "answer",
        "cost": 0.005,
        "confidence_score": 0.95,
        "context_docs": ["doc1", "doc2"],
    }

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        return FakeHTTPResponse(json.dumps(response_data))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(
        url="http://localhost:8080/run",
        cost_usd_path="$.cost",
        confidence_path="$.confidence_score",
        context_path="$.context_docs",
    )
    golden = Golden(input="q")
    result = await target.ainvoke(golden)

    assert result.output == "answer"
    assert result.cost_usd == 0.005
    assert result.confidence == 0.95
    assert result.context == ["doc1", "doc2"]


@pytest.mark.unit
def test_http_target_rejects_bracket_input_path() -> None:
    with pytest.raises(ValueError, match="dot notation"):
        HttpTarget(url="http://localhost:8080/run", input_path="$.data[0].query")


# ---------------------------------------------------------------------------
# AuthConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_auth_is_noop() -> None:
    headers: dict[str, str] = {}
    params: dict[str, str] = {}
    NoAuth().apply(headers, params)
    assert headers == {}
    assert params == {}


@pytest.mark.unit
def test_bearer_auth_adds_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TOKEN", "secret123")
    headers: dict[str, str] = {}
    BearerAuth(token="${MY_TOKEN}").apply(headers, {})
    assert headers["Authorization"] == "Bearer secret123"


@pytest.mark.unit
def test_bearer_auth_literal_token() -> None:
    headers: dict[str, str] = {}
    BearerAuth(token="literal-token").apply(headers, {})
    assert headers["Authorization"] == "Bearer literal-token"


@pytest.mark.unit
def test_api_key_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "key-val")
    headers: dict[str, str] = {}
    ApiKeyAuth(key="${API_KEY}", header="X-Custom-Key").apply(headers, {})
    assert headers["X-Custom-Key"] == "key-val"


@pytest.mark.unit
def test_api_key_auth_query() -> None:
    params: dict[str, str] = {}
    ApiKeyAuth(key="my-key", header="api_key", location="query").apply({}, params)
    assert params["api_key"] == "my-key"


@pytest.mark.unit
def test_api_key_auth_rejects_unknown_location() -> None:
    with pytest.raises(ValueError, match="location"):
        ApiKeyAuth(key="my-key", location="cookie")


@pytest.mark.unit
def test_basic_auth_encodes_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "admin")
    monkeypatch.setenv("PASS", "secret")
    headers: dict[str, str] = {}
    BasicAuth(username="${USER}", password="${PASS}").apply(headers, {})
    assert headers["Authorization"].startswith("Basic ")
    import base64

    decoded = base64.b64decode(headers["Authorization"][6:]).decode()
    assert decoded == "admin:secret"


@pytest.mark.unit
def test_env_var_missing_raises_value_error() -> None:
    headers: dict[str, str] = {}
    with pytest.raises(ValueError, match="NONEXISTENT_VAR_XYZ"):
        BearerAuth(token="${NONEXISTENT_VAR_XYZ}").apply(headers, {})


# ---------------------------------------------------------------------------
# HttpTarget with auth
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_http_target_applies_bearer_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    captured_headers: dict[str, str] = {}

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        captured_headers.update(dict(request.headers))
        return FakeHTTPResponse(json.dumps({"output": "ok"}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(url="http://localhost:8080/run", auth=BearerAuth(token="tok123"))
    golden = Golden(input="q")
    await target.ainvoke(golden)

    assert captured_headers.get("Authorization") == "Bearer tok123"


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_targets_are_registered() -> None:
    assert plugins.target("prompt") is PromptTarget
    assert plugins.target("http") is HttpTarget
