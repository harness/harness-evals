"""Tests for LangfusePromptSource.

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
class _FakeTextPrompt:
    name: str = "support-bot"
    version: int = 5
    prompt: str = "Answer {{input}} helpfully."
    config: dict[str, Any] = field(default_factory=dict)
    labels: list[str] = field(default_factory=lambda: ["production"])
    tags: list[str] = field(default_factory=lambda: ["support"])


@pytest.fixture()
def _langfuse_module():
    """Inject a fake langfuse module so LangfusePromptSource can be imported without the real package."""
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
async def test_fetch_text_prompt() -> None:
    from harness_evals.prompts.langfuse import LangfusePromptSource

    client = MagicMock()
    client.get_prompt.return_value = _FakeTextPrompt()

    source = LangfusePromptSource(client)
    template = await source.fetch(ResourceRef(source="langfuse", id="support-bot"))

    assert template.template == "Answer {{input}} helpfully."
    assert template.input_variables == ["input"]
    assert template.version == "5"
    assert template.render(input="question") == "Answer question helpfully."


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_version_parsed_as_int() -> None:
    from harness_evals.prompts.langfuse import LangfusePromptSource

    client = MagicMock()
    client.get_prompt.return_value = _FakeTextPrompt(version=3)

    source = LangfusePromptSource(client)
    await source.fetch(ResourceRef(source="langfuse", id="my-prompt", version="3"))

    call_kwargs = client.get_prompt.call_args[1]
    assert call_kwargs["version"] == 3


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_label_from_extra() -> None:
    from harness_evals.prompts.langfuse import LangfusePromptSource

    client = MagicMock()
    client.get_prompt.return_value = _FakeTextPrompt(labels=["staging"])

    source = LangfusePromptSource(client)
    await source.fetch(ResourceRef(source="langfuse", id="my-prompt", extra={"label": "staging"}))

    call_kwargs = client.get_prompt.call_args[1]
    assert call_kwargs["label"] == "staging"
    assert "version" not in call_kwargs


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_no_version_no_label_uses_default() -> None:
    from harness_evals.prompts.langfuse import LangfusePromptSource

    client = MagicMock()
    client.get_prompt.return_value = _FakeTextPrompt()

    source = LangfusePromptSource(client)
    await source.fetch(ResourceRef(source="langfuse", id="my-prompt"))

    call_kwargs = client.get_prompt.call_args[1]
    assert "version" not in call_kwargs
    assert "label" not in call_kwargs
    assert call_kwargs["type"] == "text"


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_model_hint_extracted() -> None:
    from harness_evals.prompts.langfuse import LangfusePromptSource

    client = MagicMock()
    client.get_prompt.return_value = _FakeTextPrompt(
        config={"model": "gpt-4o", "provider": "openai", "temperature": 0.7}
    )

    source = LangfusePromptSource(client)
    template = await source.fetch(ResourceRef(source="langfuse", id="my-prompt"))

    assert template.model_hint is not None
    assert template.model_hint["name"] == "gpt-4o"
    assert template.model_hint["provider"] == "openai"
    assert template.model_hint["params"] == {"temperature": 0.7}


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_model_hint_none_when_no_model_in_config() -> None:
    from harness_evals.prompts.langfuse import LangfusePromptSource

    client = MagicMock()
    client.get_prompt.return_value = _FakeTextPrompt(config={"temperature": 0.5})

    source = LangfusePromptSource(client)
    template = await source.fetch(ResourceRef(source="langfuse", id="my-prompt"))

    assert template.model_hint is None


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_chat_type_raises() -> None:
    from harness_evals.prompts.langfuse import LangfusePromptSource

    client = MagicMock()
    source = LangfusePromptSource(client)

    with pytest.raises(ValueError, match="type='text'"):
        await source.fetch(ResourceRef(source="langfuse", id="chat-prompt", extra={"type": "chat"}))


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_strip_prompts_prefix() -> None:
    from harness_evals.prompts.langfuse import LangfusePromptSource

    client = MagicMock()
    client.get_prompt.return_value = _FakeTextPrompt()

    source = LangfusePromptSource(client)
    await source.fetch(ResourceRef(source="langfuse", id="prompts/support-bot"))

    call_args = client.get_prompt.call_args[0]
    assert call_args[0] == "support-bot"


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_metadata_has_langfuse_fields() -> None:
    from harness_evals.prompts.langfuse import LangfusePromptSource

    client = MagicMock()
    client.get_prompt.return_value = _FakeTextPrompt(name="bot", labels=["production", "staging"], tags=["v2"])

    source = LangfusePromptSource(client)
    template = await source.fetch(ResourceRef(source="langfuse", id="bot"))

    assert template.metadata["langfuse_prompt_name"] == "bot"
    assert template.metadata["langfuse_prompt_labels"] == ["production", "staging"]
    assert template.metadata["langfuse_prompt_tags"] == ["v2"]


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_empty_ref_id_raises() -> None:
    from harness_evals.prompts.langfuse import LangfusePromptSource

    client = MagicMock()
    source = LangfusePromptSource(client)

    with pytest.raises(ValueError, match="prompt name"):
        await source.fetch(ResourceRef(source="langfuse", id=""))


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_close_flushes_client() -> None:
    from harness_evals.prompts.langfuse import LangfusePromptSource

    client = MagicMock()
    source = LangfusePromptSource(client)
    await source.close()

    client.flush.assert_called_once()


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_input_variables_inferred_from_template() -> None:
    from harness_evals.prompts.langfuse import LangfusePromptSource

    client = MagicMock()
    client.get_prompt.return_value = _FakeTextPrompt(prompt="Hello {{name}}, your question is: {{query}}")

    source = LangfusePromptSource(client)
    template = await source.fetch(ResourceRef(source="langfuse", id="greet"))

    assert template.input_variables == ["name", "query"]


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
async def test_plugin_registry_resolves_langfuse_source() -> None:
    """Verify that the @register_prompt_source decorator is correctly applied."""
    from harness_evals.plugins import _REGISTRIES, PROMPT_SOURCES, register_prompt_source
    from harness_evals.prompts.langfuse import LangfusePromptSource

    # Re-register since conftest restores registries between tests
    register_prompt_source("langfuse")(LangfusePromptSource)
    assert _REGISTRIES[PROMPT_SOURCES]["langfuse"] is LangfusePromptSource
