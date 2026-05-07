"""Tests for ConversationSynthesizer."""

import pytest

from harness_evals.conversation.golden import ConversationGolden
from harness_evals.core.types import Message
from harness_evals.synthesizer.conversation import (
    ConversationSynthesizer,
    ScriptedConversationSynthesizer,
)
from tests.conftest import MockLLM

SCENARIO_RESPONSE = {
    "scenarios": [
        {
            "scenario": "User wants to know refund policy",
            "expected_outcome": "Agent explains 30-day refund window and process",
            "user_persona": "Frustrated customer who bought last week",
        },
        {
            "scenario": "User needs help with password reset",
            "expected_outcome": "Agent guides user through reset steps successfully",
            "user_persona": "Non-technical user",
        },
    ]
}

SCRIPTED_RESPONSE = {
    "conversations": [
        {
            "scenario": "User asks about shipping time",
            "expected_outcome": "User gets a clear shipping ETA",
            "turns": [
                {"role": "user", "content": "How long does shipping take?"},
                {"role": "assistant", "content": "Standard shipping takes 3-5 business days."},
                {"role": "user", "content": "What about express?"},
                {"role": "assistant", "content": "Express shipping takes 1-2 business days."},
            ],
        }
    ]
}


@pytest.mark.unit
class TestConversationSynthesizer:
    async def test_generates_scenario_goldens(self):
        llm = MockLLM(default=SCENARIO_RESPONSE)
        synth = ConversationSynthesizer(llm=llm)
        goldens = await synth.generate(["Customer support documentation."], n=2)
        assert len(goldens) == 2
        assert all(isinstance(g, ConversationGolden) for g in goldens)
        assert goldens[0].scenario == "User wants to know refund policy"
        assert goldens[0].expected_outcome == "Agent explains 30-day refund window and process"
        assert goldens[0].user_persona == "Frustrated customer who bought last week"
        assert goldens[0].turns is None  # scenario mode has no turns

    async def test_respects_n(self):
        llm = MockLLM(default=SCENARIO_RESPONSE)
        synth = ConversationSynthesizer(llm=llm)
        goldens = await synth.generate(["doc"], n=1)
        assert len(goldens) <= 1

    async def test_empty_document_returns_empty(self):
        llm = MockLLM(default={"scenarios": []})
        synth = ConversationSynthesizer(llm=llm)
        goldens = await synth.generate(["   "], n=5)
        assert goldens == []

    async def test_deduplicates_by_scenario(self):
        duplicate_response = {
            "scenarios": [
                {"scenario": "same", "expected_outcome": "o1", "user_persona": None},
                {"scenario": "same", "expected_outcome": "o2", "user_persona": None},
            ]
        }
        llm = MockLLM(default=duplicate_response)
        synth = ConversationSynthesizer(llm=llm)
        goldens = await synth.generate(["doc"], n=5)
        assert len(goldens) == 1


@pytest.mark.unit
class TestScriptedConversationSynthesizer:
    async def test_generates_scripted_goldens(self):
        llm = MockLLM(default=SCRIPTED_RESPONSE)
        synth = ScriptedConversationSynthesizer(llm=llm)
        goldens = await synth.generate(["E-commerce help docs."], n=1)
        assert len(goldens) == 1
        g = goldens[0]
        assert isinstance(g, ConversationGolden)
        assert g.scenario == "User asks about shipping time"
        assert g.turns is not None
        assert len(g.turns) == 4
        assert all(isinstance(t, Message) for t in g.turns)
        assert g.turns[0].role == "user"
        assert g.turns[1].role == "assistant"

    async def test_skips_invalid_turns(self):
        response = {
            "conversations": [
                {
                    "scenario": "test",
                    "expected_outcome": "done",
                    "turns": [
                        {"role": "user", "content": "hi"},
                        # missing role — should be skipped gracefully
                        {"content": "oops"},
                        {"role": "assistant", "content": "hello"},
                    ],
                }
            ]
        }
        llm = MockLLM(default=response)
        synth = ScriptedConversationSynthesizer(llm=llm)
        goldens = await synth.generate(["doc"], n=1)
        assert len(goldens) == 1
        # Invalid turn skipped, 2 valid turns remain
        assert len(goldens[0].turns) == 2
