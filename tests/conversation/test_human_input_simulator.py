"""Tests for generic HumanInputSimulator."""

import pytest

from harness_evals.conversation import ConversationGolden, HumanInputSimulator
from harness_evals.conversation.human_input import PendingHumanInput
from harness_evals.llm.base import BaseLLM


class ScriptedLLM(BaseLLM):
    async def generate(self, prompt: str, **kwargs) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return {"response": {"choice": "blue", "confirmed": True}}


@pytest.mark.unit
async def test_scripted_human_input_response():
    golden = ConversationGolden(
        scenario="Pick a color",
        expected_outcome="User picks blue",
        metadata={
            "elicitation_script": [
                {
                    "trigger": "choice_prompt",
                    "human_input": {"choice": "blue"},
                }
            ]
        },
    )
    pending = PendingHumanInput(type="choice_prompt", payload={"question": "Which color?"})

    result = await HumanInputSimulator().respond(pending, golden, [])

    assert result == {"choice": "blue"}


@pytest.mark.unit
async def test_llm_fallback_returns_generic_response_object():
    golden = ConversationGolden(
        scenario="Confirm deployment",
        expected_outcome="Deployment confirmed",
        elicitation_hints={"intents": {"confirm": "yes"}},
    )
    pending = PendingHumanInput(type="confirmation", payload={"message": "Proceed?"})

    result = await HumanInputSimulator(llm=ScriptedLLM()).respond(pending, golden, [])

    assert result == {"choice": "blue", "confirmed": True}


@pytest.mark.unit
async def test_pending_human_input_reads_legacy_metadata_shape():
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_free_text",
            "review_id": "ask-1",
            "payload": {"content": {"question": "Name?"}},
        }
    )

    assert pending.type == "elicitation_free_text"
    assert pending.correlation_id == "ask-1"
    assert pending.payload["content"]["question"] == "Name?"


@pytest.mark.unit
def test_resolve_intent_skips_structured_only_for_plain_text():
    from harness_evals.conversation.human_input import resolve_intent

    golden = ConversationGolden(
        scenario="SCS components",
        expected_outcome="Listed",
        elicitation_hints={
            "intents": {"component_type": "SBOM Components"},
            "matchers": [
                {
                    "intent": "component_type",
                    "structured_only": True,
                    "question_contains": ["component names"],
                }
            ],
        },
    )
    question = "No SBOM components in testevals; I searched component names across repos."

    assert resolve_intent(question, golden, plain_text=True) is None
    assert resolve_intent(question, golden, plain_text=False) == "component_type"


@pytest.mark.unit
def test_resolve_intent_question_contains_all():
    from harness_evals.conversation.human_input import resolve_intent

    golden = ConversationGolden(
        scenario="SCS components",
        expected_outcome="Listed",
        elicitation_hints={
            "intents": {"component_type": "SBOM Components"},
            "matchers": [
                {
                    "intent": "component_type",
                    "question_contains_all": ["what", "type", "component"],
                }
            ],
        },
    )

    assert resolve_intent("What type of component should I search for?", golden) == "component_type"
    assert resolve_intent("Here are the component names found", golden) is None
