"""Tests for target adapters."""

from __future__ import annotations

import json
import logging
import time
from urllib.error import URLError
from urllib.request import Request

import pytest

from harness_evals import plugins
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.errors import TargetInvocationError
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
        self.prompts: list[str] = []
        self.system_prompts: list[object | None] = []

    async def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        self.system_prompts.append(kwargs.get("system_prompt"))
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
async def test_prompt_target_passes_system_prompt_separately() -> None:
    prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
    model = StubLLM("policy answer")
    target = PromptTarget(prompt=prompt, model=model, system_prompt="You are a policy assistant.")

    result = await target.ainvoke(Golden(input="What is the return policy?"))

    assert result.output == "policy answer"
    assert model.prompts == ["What is the return policy?"]
    assert model.system_prompts == ["You are a policy assistant."]


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
async def test_prompt_target_synthesizes_trajectory() -> None:
    prompt = PromptTemplate(template="Answer: {{input}}", input_variables=["input"])
    target = PromptTarget(prompt=prompt, model=StubLLM("42"))

    golden = Golden(input="What is 6*7?", expected="42")
    result = await target.ainvoke(golden)

    assert result.messages is not None
    assert [(m.role, m.content) for m in result.messages] == [
        ("user", "What is 6*7?"),
        ("assistant", "42"),
    ]


@pytest.mark.unit
async def test_prompt_target_trajectory_json_encodes_dict_input() -> None:
    prompt = PromptTemplate(template="Process: {{input}}", input_variables=["input"])
    target = PromptTarget(prompt=prompt, model=StubLLM("done"))

    golden = Golden(input={"query": "hello", "lang": "en"})
    result = await target.ainvoke(golden)

    assert result.messages is not None
    assert result.messages[0].role == "user"
    assert json.loads(result.messages[0].content) == {"query": "hello", "lang": "en"}


@pytest.mark.unit
async def test_prompt_target_closes_model() -> None:
    prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
    model = CloseableLLM("out")
    target = PromptTarget(prompt=prompt, model=model)

    await target.close()

    assert model.closed


class UsageStubLLM(StubLLM):
    """StubLLM that reports token usage to the active collector."""

    def __init__(self, response: str = "out", in_tok: int = 42, out_tok: int = 7) -> None:
        super().__init__(response)
        self._in, self._out = in_tok, out_tok

    async def generate(self, prompt: str, **kwargs) -> str:
        from harness_evals.llm.usage import record_token_usage

        record_token_usage(input_tokens=self._in, output_tokens=self._out)
        return await super().generate(prompt, **kwargs)


@pytest.mark.unit
async def test_prompt_target_captures_token_split() -> None:
    prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
    target = PromptTarget(prompt=prompt, model=UsageStubLLM(response="42", in_tok=42, out_tok=7))

    result = await target.ainvoke(Golden(input="q"))

    assert result.input_tokens == 42
    assert result.output_tokens == 7


@pytest.mark.unit
async def test_prompt_target_no_usage_leaves_tokens_none() -> None:
    prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
    target = PromptTarget(prompt=prompt, model=StubLLM("42"))

    result = await target.ainvoke(Golden(input="q"))

    assert result.input_tokens is None
    assert result.output_tokens is None


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

    captured_requests: list[Request] = []

    def fake_urlopen_capture(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        captured_requests.append(request)
        return FakeHTTPResponse(json.dumps({"output": "ok"}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen_capture)

    target = HttpTarget(
        url="http://localhost:8080/api",
        body_template={"data": {"query": "{{input}}", "mode": "eval"}},
    )
    golden = Golden(input="test query")
    result = await target.ainvoke(golden)
    assert result.output == "ok"
    sent = json.loads(captured_requests[0].data.decode())
    assert sent == {"data": {"query": "test query", "mode": "eval"}}


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

    with pytest.raises(TargetInvocationError, match="Connection refused"):
        await target.ainvoke(golden)


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

    with pytest.raises(TargetInvocationError) as exc_info:
        await target.ainvoke(Golden(input="test"))

    assert exc_info.value.latency_ms == pytest.approx(20.0)


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
async def test_http_target_extracts_token_split(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        return FakeHTTPResponse(json.dumps({"output": "ok", "in": 120, "out": 34}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(
        url="http://localhost:8080/run",
        input_tokens_path="$.in",
        output_tokens_path="$.out",
    )
    result = await target.ainvoke(Golden(input="hi"))
    assert result.input_tokens == 120
    assert result.output_tokens == 34


@pytest.mark.unit
async def test_http_target_token_split_zero_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    # A genuine 0 must be preserved as 0, not collapsed to None.
    from harness_evals.targets import http as target_http

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        return FakeHTTPResponse(json.dumps({"output": "ok", "in": 0, "out": 0}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(url="http://localhost:8080/run", input_tokens_path="$.in", output_tokens_path="$.out")
    result = await target.ainvoke(Golden(input="hi"))
    assert result.input_tokens == 0
    assert result.output_tokens == 0


@pytest.mark.unit
@pytest.mark.parametrize("bad_value", ["N/A", float("inf"), float("nan")])
async def test_http_target_bad_token_value_dropped_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad_value: object
) -> None:
    # json.loads accepts Infinity/NaN, and endpoints can return non-numeric
    # strings. A configured path resolving to any of these must not raise
    # (int(inf) raises OverflowError) — it is dropped with a visible warning.
    from harness_evals.targets import http as target_http

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        return FakeHTTPResponse(json.dumps({"output": "ok", "in": bad_value}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(url="http://localhost:8080/run", input_tokens_path="$.in")
    with caplog.at_level(logging.WARNING):
        result = await target.ainvoke(Golden(input="hi"))

    assert result.input_tokens is None
    assert any("input_tokens path" in r.message for r in caplog.records)


@pytest.mark.unit
async def test_http_target_reported_messages_take_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    response_data = {
        "output": "answer",
        "trace": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "thinking"},
            {"role": "assistant", "content": "answer"},
        ],
    }

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        return FakeHTTPResponse(json.dumps(response_data))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(url="http://localhost:8080/run", messages_path="$.trace")
    result = await target.ainvoke(Golden(input="q"))

    # A reported trajectory is authoritative — used verbatim (coerced from the
    # extracted dicts into Message objects), not synthesized.
    assert result.messages is not None
    assert len(result.messages) == 3
    assert [m.content for m in result.messages] == ["q", "thinking", "answer"]


@pytest.mark.unit
async def test_http_target_malformed_reported_messages_not_synthesized_over(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from harness_evals.targets import http as target_http

    # messages_path resolves to a value that can't be coerced (list of strings,
    # not message dicts). This is an instrumentation failure — the target must
    # NOT silently fall back to a fabricated [user, assistant] trace over it.
    response_data = {"output": "answer", "trace": ["not", "a", "trajectory"]}

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        return FakeHTTPResponse(json.dumps(response_data))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(url="http://localhost:8080/run", messages_path="$.trace")
    with caplog.at_level(logging.WARNING, logger="harness_evals.targets.trajectory"):
        result = await target.ainvoke(Golden(input="q"))

    # Empty (not a synthesized 2-message exchange) so metrics surface "no messages".
    assert result.messages == []
    assert any("could not be coerced" in rec.message for rec in caplog.records)


@pytest.mark.unit
async def test_http_target_empty_reported_messages_is_not_treated_as_malformed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from harness_evals.targets import http as target_http

    # An endpoint that reports an explicitly-empty trajectory (messages: []) is
    # valid "no turns" — it must stay [] with no coercion warning, not synthesize.
    response_data = {"output": "answer", "trace": []}

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        return FakeHTTPResponse(json.dumps(response_data))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(url="http://localhost:8080/run", messages_path="$.trace")
    with caplog.at_level(logging.WARNING, logger="harness_evals.targets.trajectory"):
        result = await target.ainvoke(Golden(input="q"))

    assert result.messages == []
    assert not any("could not be coerced" in rec.message for rec in caplog.records)


@pytest.mark.unit
async def test_http_target_synthesizes_trajectory_when_not_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        return FakeHTTPResponse(json.dumps({"output": "agent response"}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(url="http://localhost:8080/run")
    result = await target.ainvoke(Golden(input="hello"))

    assert result.messages is not None
    assert [(m.role, m.content) for m in result.messages] == [
        ("user", "hello"),
        ("assistant", "agent response"),
    ]


@pytest.mark.unit
async def test_http_target_synthesized_trajectory_includes_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    response_data = {
        "output": "done",
        "tools": [{"name": "search", "input": {"q": "cats"}, "output": "found"}],
    }

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        return FakeHTTPResponse(json.dumps(response_data))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(url="http://localhost:8080/run", tool_calls_path="$.tools")
    result = await target.ainvoke(Golden(input="find cats"))

    assert result.messages is not None
    # user, assistant(tool_calls), assistant(output)
    assert [m.role for m in result.messages] == ["user", "assistant", "assistant"]
    assert result.messages[1].tool_calls is not None
    assert result.messages[1].tool_calls[0].name == "search"
    assert result.messages[2].content == "done"
    # tool_calls are also coerced into ToolCall objects on the eval case.
    assert result.tool_calls is not None
    assert result.tool_calls[0].name == "search"


@pytest.mark.unit
async def test_http_target_default_body_wraps_input(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    captured: list[Request] = []

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        captured.append(request)
        return FakeHTTPResponse(json.dumps({"output": "ok"}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    # No body_template → the whole golden.input is wrapped as {"input": ...}.
    target = HttpTarget(url="http://localhost:8080/run")
    await target.ainvoke(Golden(input={"question": "hi", "k": 3}))

    sent = json.loads(captured[0].data.decode())
    assert sent == {"input": {"question": "hi", "k": 3}}


@pytest.mark.unit
async def test_http_target_scatters_input_fields_via_templating(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    captured: list[Request] = []

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        captured.append(request)
        return FakeHTTPResponse(json.dumps({"output": "ok"}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(
        url="http://localhost:8080/run",
        body_template={
            "query": "{{input.question}}",
            "top_k": "{{input.k}}",
            "greeting": "Hello {{metadata.user}}",
        },
    )
    await target.ainvoke(Golden(input={"question": "cats?", "k": 5}, metadata={"user": "srikar"}))

    sent = json.loads(captured[0].data.decode())
    # Whole-string placeholder preserves native type (int stays int); embedded
    # placeholder is string-interpolated.
    assert sent == {"query": "cats?", "top_k": 5, "greeting": "Hello srikar"}


@pytest.mark.unit
async def test_http_target_unresolved_placeholder_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        return FakeHTTPResponse(json.dumps({"output": "ok"}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(
        url="http://localhost:8080/run",
        body_template={"query": "{{input.missing}}"},
    )
    with pytest.raises(ValueError, match="did not resolve"):
        await target.ainvoke(Golden(input={"question": "hi"}))


@pytest.mark.unit
async def test_http_target_templates_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import http as target_http

    captured: list[Request] = []

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        captured.append(request)
        return FakeHTTPResponse(json.dumps({"output": "ok"}))

    monkeypatch.setattr(target_http, "urlopen", fake_urlopen)

    target = HttpTarget(
        url="http://localhost:8080/run",
        headers={"Authorization": "Bearer {{input.token}}", "X-Tenant": "{{metadata.tenant}}"},
    )
    await target.ainvoke(Golden(input={"token": "abc123"}, metadata={"tenant": "acme"}))

    # urllib title-cases header names on the Request object.
    assert captured[0].headers["Authorization"] == "Bearer abc123"
    assert captured[0].headers["X-tenant"] == "acme"


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
