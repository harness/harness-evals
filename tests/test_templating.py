"""Unit tests for HTTP request-body templating (`{{input.foo}}` placeholders)."""

from __future__ import annotations

import pytest

from harness_evals.core.golden import Golden
from harness_evals.targets.templating import render_headers, render_request_body


@pytest.mark.unit
def test_none_template_wraps_input() -> None:
    assert render_request_body(None, Golden(input="hi")) == {"input": "hi"}


@pytest.mark.unit
def test_none_template_preserves_structured_input() -> None:
    g = Golden(input={"a": 1, "b": [2, 3]})
    assert render_request_body(None, g) == {"input": {"a": 1, "b": [2, 3]}}


@pytest.mark.unit
def test_whole_placeholder_preserves_native_type() -> None:
    # A string that is exactly one placeholder yields the referenced value's
    # native type — not a stringified copy.
    g = Golden(input={"nested": {"x": 1}})
    body = render_request_body({"payload": "{{input}}"}, g)
    assert body == {"payload": {"nested": {"x": 1}}}
    assert isinstance(body["payload"], dict)


@pytest.mark.unit
def test_whole_placeholder_int_stays_int() -> None:
    g = Golden(input={"k": 7})
    body = render_request_body({"top_k": "{{input.k}}"}, g)
    assert body == {"top_k": 7}
    assert isinstance(body["top_k"], int)


@pytest.mark.unit
def test_embedded_placeholder_is_string_interpolated() -> None:
    g = Golden(input={"name": "Ada"})
    body = render_request_body({"greeting": "Hello {{input.name}}!"}, g)
    assert body == {"greeting": "Hello Ada!"}


@pytest.mark.unit
def test_multiple_embedded_placeholders() -> None:
    g = Golden(input={"a": "x", "b": "y"})
    body = render_request_body({"s": "{{input.a}}-{{input.b}}"}, g)
    assert body == {"s": "x-y"}


@pytest.mark.unit
def test_embedded_non_string_value_is_json_encoded() -> None:
    g = Golden(input={"cfg": {"deep": True}})
    body = render_request_body({"s": "cfg={{input.cfg}}"}, g)
    assert body == {"s": 'cfg={"deep": true}'}


@pytest.mark.unit
def test_metadata_placeholder() -> None:
    g = Golden(input="q", metadata={"user_id": "u1"})
    body = render_request_body({"uid": "{{metadata.user_id}}"}, g)
    assert body == {"uid": "u1"}


@pytest.mark.unit
def test_list_index_path() -> None:
    g = Golden(input={"items": ["first", "second"]})
    body = render_request_body({"pick": "{{input.items.1}}"}, g)
    assert body == {"pick": "second"}


@pytest.mark.unit
def test_nested_dicts_and_lists_are_walked() -> None:
    g = Golden(input={"q": "hi", "n": 2})
    template = {
        "outer": {"inner": "{{input.q}}"},
        "arr": ["{{input.n}}", "literal"],
        "static": 42,
    }
    body = render_request_body(template, g)
    assert body == {"outer": {"inner": "hi"}, "arr": [2, "literal"], "static": 42}


@pytest.mark.unit
def test_non_string_leaves_pass_through() -> None:
    g = Golden(input="q")
    body = render_request_body({"a": 1, "b": True, "c": None, "d": 1.5}, g)
    assert body == {"a": 1, "b": True, "c": None, "d": 1.5}


@pytest.mark.unit
def test_whitespace_inside_placeholder_is_tolerated() -> None:
    g = Golden(input={"x": "ok"})
    assert render_request_body({"v": "{{ input.x }}"}, g) == {"v": "ok"}


@pytest.mark.unit
def test_unknown_root_raises() -> None:
    with pytest.raises(ValueError, match="unknown root"):
        render_request_body({"v": "{{output.x}}"}, Golden(input="q"))


@pytest.mark.unit
def test_missing_field_raises() -> None:
    with pytest.raises(ValueError, match="did not resolve"):
        render_request_body({"v": "{{input.missing}}"}, Golden(input={"present": 1}))


@pytest.mark.unit
def test_missing_field_in_embedded_placeholder_also_raises() -> None:
    with pytest.raises(ValueError, match="did not resolve"):
        render_request_body({"v": "prefix {{input.nope}}"}, Golden(input={"present": 1}))


@pytest.mark.unit
def test_out_of_range_list_index_raises() -> None:
    with pytest.raises(ValueError, match="did not resolve"):
        render_request_body({"v": "{{input.items.9}}"}, Golden(input={"items": ["only"]}))


@pytest.mark.unit
def test_literal_string_without_placeholder_unchanged() -> None:
    g = Golden(input="q")
    assert render_request_body({"mode": "eval", "n": "5"}, g) == {"mode": "eval", "n": "5"}


# ---------------------------------------------------------------------------
# render_headers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_headers_resolves_embedded_placeholder() -> None:
    g = Golden(input={"token": "abc123"})
    headers = render_headers({"Authorization": "Bearer {{input.token}}"}, g)
    assert headers == {"Authorization": "Bearer abc123"}


@pytest.mark.unit
def test_headers_whole_placeholder_is_stringified() -> None:
    # A header value must be a string even when it is exactly one placeholder;
    # unlike the body, it cannot carry a native int/dict.
    g = Golden(input={"sid": 42})
    headers = render_headers({"X-Session": "{{input.sid}}"}, g)
    assert headers == {"X-Session": "42"}
    assert isinstance(headers["X-Session"], str)


@pytest.mark.unit
def test_headers_metadata_placeholder() -> None:
    g = Golden(input="q", metadata={"tenant": "acme"})
    assert render_headers({"X-Tenant": "{{metadata.tenant}}"}, g) == {"X-Tenant": "acme"}


@pytest.mark.unit
def test_headers_without_placeholder_pass_through() -> None:
    g = Golden(input="q")
    headers = {"Content-Type": "application/json", "X-Static": "v1"}
    assert render_headers(headers, g) == headers


@pytest.mark.unit
def test_headers_empty_returns_empty() -> None:
    assert render_headers({}, Golden(input="q")) == {}


@pytest.mark.unit
def test_headers_name_is_never_templated() -> None:
    # Only values are resolved; a placeholder-looking name is left alone.
    g = Golden(input={"x": "v"})
    headers = render_headers({"{{input.x}}": "static"}, g)
    assert headers == {"{{input.x}}": "static"}


@pytest.mark.unit
def test_headers_unresolved_placeholder_raises() -> None:
    with pytest.raises(ValueError, match="did not resolve"):
        render_headers({"X-Id": "{{input.missing}}"}, Golden(input={"present": 1}))


# ---------------------------------------------------------------------------
# env.* placeholders
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_env_placeholder_resolves_from_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_TARGET_API_KEY", "secret123")
    g = Golden(input="q")
    headers = render_headers({"Authorization": "Bearer {{env.EVAL_TARGET_API_KEY}}"}, g)
    assert headers == {"Authorization": "Bearer secret123"}


@pytest.mark.unit
def test_env_placeholder_in_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TOKEN", "tok")
    g = Golden(input={"q": "hi"})
    body = render_request_body({"key": "{{env.MY_TOKEN}}", "query": "{{input.q}}"}, g)
    assert body == {"key": "tok", "query": "hi"}


@pytest.mark.unit
def test_env_placeholder_missing_var_raises() -> None:
    g = Golden(input="q")
    with pytest.raises(ValueError, match="not set"):
        render_headers({"Authorization": "Bearer {{env.NONEXISTENT_VAR_XYZ}}"}, g)


@pytest.mark.unit
def test_env_placeholder_bare_env_raises() -> None:
    g = Golden(input="q")
    with pytest.raises(ValueError, match="requires a variable name"):
        render_request_body({"v": "{{env}}"}, g)
