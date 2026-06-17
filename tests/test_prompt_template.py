"""Tests for prompt template rendering."""

from __future__ import annotations

import pytest

from harness_evals.prompts import PromptTemplate, extract_template_variables


@pytest.mark.unit
def test_render_replaces_single_placeholder() -> None:
    template = PromptTemplate("Answer this: {{input}}")

    assert template.render(input="What is your return policy?") == "Answer this: What is your return policy?"


@pytest.mark.unit
def test_render_replaces_multiple_placeholders() -> None:
    template = PromptTemplate(
        "Use {{tone}} tone for {{input}}.",
        input_variables=["tone", "input"],
    )

    assert template.render(input="hello", tone="friendly") == "Use friendly tone for hello."


@pytest.mark.unit
def test_render_raises_for_missing_placeholder_value() -> None:
    template = PromptTemplate("Answer: {{input}}")

    with pytest.raises(KeyError, match="input"):
        template.render()


@pytest.mark.unit
def test_escaped_open_braces_are_literal() -> None:
    template = PromptTemplate(
        r"Show \{\{input}} literally, then render {{input}}.",
        input_variables=["input"],
    )

    assert template.render(input="value") == "Show {{input}} literally, then render value."


@pytest.mark.unit
def test_input_variables_must_include_template_placeholders() -> None:
    with pytest.raises(ValueError, match="name"):
        PromptTemplate("Hello {{name}}")


@pytest.mark.unit
@pytest.mark.parametrize("placeholder", ["{{user.name}}", "{{items[0]}}"])
def test_unsupported_placeholder_syntax_raises_clear_error(placeholder: str) -> None:
    with pytest.raises(ValueError, match="flat"):
        PromptTemplate(f"Unsupported {placeholder}", input_variables=[])


@pytest.mark.unit
def test_extract_template_variables_preserves_first_seen_order() -> None:
    assert extract_template_variables("{{b}} {{a}} {{b}}") == ["b", "a"]
