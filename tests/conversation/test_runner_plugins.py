"""Tests for conversation runner plugin wiring."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from harness_evals.conversation import ConversationGolden
from harness_evals.conversation.runner import evaluate_conversation
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM
from harness_evals.plugins import _restore, _snapshot, register_plain_text_followup_resolver


class StopLLM(BaseLLM):
    async def generate(self, prompt: str, **kwargs) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return {"achieved": True, "reasoning": "done"}


@pytest.mark.unit
async def test_evaluate_conversation_passes_explicit_plain_text_followup_resolvers(monkeypatch):
    snapshot = _snapshot()
    try:
        _restore(snapshot)

        @register_plain_text_followup_resolver
        def _test_resolver(golden, match_scope, *, used_intents):  # type: ignore[no-untyped-def]
            return None

        captured: dict = {}

        class CapturingSimulator:
            def __init__(self, simulator_llm, **kwargs):
                captured.update(kwargs)
                self._sim = MagicMock()
                self._sim.simulate = AsyncMock(return_value=MagicMock(input="x", output="", messages=[], metadata={}))

            async def simulate(self, golden, agent_fn):
                return await self._sim.simulate(golden, agent_fn)

        monkeypatch.setattr("harness_evals.conversation.runner.ConversationSimulator", CapturingSimulator)
        monkeypatch.setattr(
            "harness_evals.conversation.runner.a_evaluate",
            AsyncMock(return_value=[]),
        )

        golden = ConversationGolden(
            scenario="test",
            expected_outcome="done",
            max_turns=1,
            initial_prompt="hi",
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="ok")

        await evaluate_conversation(
            golden,
            agent_fn,
            [],
            simulator_llm=StopLLM(),
            plain_text_followup_resolvers=[_test_resolver],
        )

        resolvers = captured.get("plain_text_followup_resolvers") or ()
        assert _test_resolver in resolvers
    finally:
        _restore(snapshot)


@pytest.mark.unit
async def test_evaluate_conversation_does_not_use_global_resolvers(monkeypatch):
    snapshot = _snapshot()
    try:

        @register_plain_text_followup_resolver
        def _global_resolver(golden, match_scope, *, used_intents):  # type: ignore[no-untyped-def]
            return None

        captured: dict = {}

        class CapturingSimulator:
            def __init__(self, simulator_llm, **kwargs):
                captured.update(kwargs)

            async def simulate(self, golden, agent_fn):
                return MagicMock(input="x", output="", messages=[], metadata={})

        monkeypatch.setattr("harness_evals.conversation.runner.ConversationSimulator", CapturingSimulator)
        monkeypatch.setattr(
            "harness_evals.conversation.runner.a_evaluate",
            AsyncMock(return_value=[]),
        )

        golden = ConversationGolden(
            scenario="test",
            expected_outcome="done",
            max_turns=1,
            initial_prompt="hi",
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="ok")

        await evaluate_conversation(golden, agent_fn, [], simulator_llm=StopLLM())

        assert captured["plain_text_followup_resolvers"] is None
        assert _global_resolver not in (captured["plain_text_followup_resolvers"] or ())
    finally:
        _restore(snapshot)
