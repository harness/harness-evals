"""ConversationSimulator — drives multi-turn conversations between a simulated user and an agent."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from harness_evals._async_compat import _run_async
from harness_evals.conversation.golden import ConversationGolden, ConversationMode
from harness_evals.conversation.graph import (
    BranchNode,
    LLMNode,
    ScriptedNode,
    SimulationGraph,
    StopNode,
)
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM

_logger = logging.getLogger(__name__)

_USER_PROMPT = """You are simulating a user in a conversation. Your goal is to achieve the following scenario:

**Scenario**: {scenario}

{persona_section}
{context_section}

**Conversation so far**:
{history}

Generate the next user message. Be natural and concise. Stay focused on achieving the scenario goal.
Respond with ONLY the user message text, nothing else."""

_STOP_PROMPT = """You are evaluating whether a conversation has achieved its expected outcome.

**Expected outcome**: {expected_outcome}

**Conversation**:
{history}

Has the expected outcome been fully achieved? Respond with JSON:
{{"achieved": true/false, "reasoning": "brief explanation"}}"""

_STOP_SCHEMA = {
    "type": "object",
    "required": ["achieved", "reasoning"],
    "properties": {
        "achieved": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
}


class ConversationSimulator:
    """Drives a multi-turn conversation between a simulated user and agent-under-test."""

    def __init__(
        self, simulator_llm: BaseLLM | None = None, *, max_concurrent: int = 5, graph: SimulationGraph | None = None
    ) -> None:
        self.simulator_llm = simulator_llm
        self.max_concurrent = max_concurrent
        self.graph = graph

    def _require_llm(self) -> BaseLLM:
        if self.simulator_llm is None:
            raise ValueError(
                "simulator_llm is required for SIMULATE/GRAPH conversation modes. "
                "Pass simulator_llm=<BaseLLM instance> when constructing the simulator "
                "or calling evaluate_dataset()."
            )
        return self.simulator_llm

    def simulate_sync(
        self,
        golden: ConversationGolden,
        agent_fn: Callable[[list[Message]], Awaitable[Message]],
    ) -> EvalCase:
        return _run_async(self.simulate(golden, agent_fn))

    async def simulate(
        self,
        golden: ConversationGolden,
        agent_fn: Callable[[list[Message]], Awaitable[Message]],
    ) -> EvalCase:
        """Run one conversation and return the resulting EvalCase."""
        if golden.mode == ConversationMode.REPLAY:
            return self._replay(golden)
        if golden.mode == ConversationMode.SCRIPTED:
            return await self._scripted(golden, agent_fn)
        if golden.mode == ConversationMode.GRAPH:
            return await self._graph_simulate(golden, agent_fn)

        history: list[Message] = []

        for _ in range(golden.max_turns):
            user_text = await self._generate_user_message(golden, history)
            history.append(Message(role="user", content=user_text))

            assistant_msg = await agent_fn(list(history))
            history.append(assistant_msg)

            if await self._should_stop(golden, history):
                break

        return self._build_eval_case(golden, history)

    async def simulate_batch(
        self,
        goldens: list[ConversationGolden],
        agent_fn: Callable[[list[Message]], Awaitable[Message]],
    ) -> list[EvalCase]:
        """Simulate multiple conversations concurrently."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _run(g: ConversationGolden) -> EvalCase:
            async with semaphore:
                return await self.simulate(g, agent_fn)

        return list(await asyncio.gather(*[_run(g) for g in goldens]))

    async def _generate_user_message(self, golden: ConversationGolden, history: list[Message]) -> str:
        persona_section = f"**Your persona**: {golden.user_persona}" if golden.user_persona else ""
        context_section = f"**Background context**: {'; '.join(golden.context)}" if golden.context else ""
        history_text = (
            "\n".join(f"[{m.role}]: {m.content or ''}" for m in history)
            if history
            else "(conversation has not started yet)"
        )

        prompt = _USER_PROMPT.format(
            scenario=golden.scenario,
            persona_section=persona_section,
            context_section=context_section,
            history=history_text,
        )
        return await self._require_llm().generate(prompt)

    async def _should_stop(self, golden: ConversationGolden, history: list[Message]) -> bool:
        history_text = "\n".join(f"[{m.role}]: {m.content or ''}" for m in history)
        prompt = _STOP_PROMPT.format(
            expected_outcome=golden.expected_outcome,
            history=history_text,
        )
        result = await self._require_llm().generate_json(prompt, _STOP_SCHEMA)
        return bool(result.get("achieved", False))

    async def _scripted(
        self,
        golden: ConversationGolden,
        agent_fn: Callable[[list[Message]], Awaitable[Message]],
    ) -> EvalCase:
        """Run agent against pre-scripted user turns from a dataset."""
        assert golden.turns is not None
        history: list[Message] = []
        for turn in golden.turns:
            if turn.role == "user":
                history.append(turn)
                assistant_msg = await agent_fn(list(history))
                history.append(assistant_msg)
        return self._build_eval_case(golden, history)

    def _replay(self, golden: ConversationGolden) -> EvalCase:
        """Replay pre-scripted turns without simulation."""
        assert golden.turns is not None
        return self._build_eval_case(golden, golden.turns)

    async def _graph_simulate(
        self,
        golden: ConversationGolden,
        agent_fn: Callable[[list[Message]], Awaitable[Message]],
    ) -> EvalCase:
        """Run conversation driven by a SimulationGraph."""
        graph = self.graph
        if graph is None and golden.graph_config is not None:
            graph = SimulationGraph.from_dict(golden.graph_config)
        if graph is None:
            raise ValueError("GRAPH mode requires a SimulationGraph (via simulator or golden.graph_config)")

        history: list[Message] = []
        current = graph.start
        turns_used = 0
        prev_node: str | None = None
        repeat_count = 0

        while turns_used < golden.max_turns:
            node = graph.nodes[current]

            if isinstance(node, StopNode):
                break

            if isinstance(node, BranchNode):
                last_response = next((m for m in reversed(history) if m.role == "assistant"), None)
                if last_response is None:
                    raise RuntimeError("BranchNode reached with no prior agent response")
                next_id = graph.resolve_next(current, last_response)
                if next_id is None:
                    break
                current = next_id
                continue

            if current == prev_node:
                repeat_count += 1
                if repeat_count == 3 and isinstance(node, ScriptedNode):
                    _logger.warning(
                        "ScriptedNode '%s' has repeated %d times (no edge matched); conversation may be stuck",
                        current,
                        repeat_count,
                    )
            else:
                repeat_count = 0
            prev_node = current

            if isinstance(node, ScriptedNode):
                user_text = node.message
            elif isinstance(node, LLMNode):
                user_text = await self._generate_user_message_for_goal(node.goal, golden, history)
            else:
                raise ValueError(f"unexpected node type: {type(node)}")

            history.append(Message(role="user", content=user_text))
            assistant_msg = await agent_fn(list(history))
            history.append(assistant_msg)
            turns_used += 1

            next_id = graph.resolve_next(current, assistant_msg)
            if next_id is not None:
                current = next_id

        return self._build_eval_case(golden, history)

    async def _generate_user_message_for_goal(
        self, goal: str, golden: ConversationGolden, history: list[Message]
    ) -> str:
        """Generate a user message for an LLMNode using goal + golden scenario as context."""
        persona_section = f"**Your persona**: {golden.user_persona}" if golden.user_persona else ""
        context_parts = []
        if golden.scenario:
            context_parts.append(f"Overall scenario: {golden.scenario}")
        if golden.context:
            context_parts.extend(golden.context)
        context_section = f"**Background context**: {'; '.join(context_parts)}"
        history_text = (
            "\n".join(f"[{m.role}]: {m.content or ''}" for m in history)
            if history
            else "(conversation has not started yet)"
        )

        prompt = _USER_PROMPT.format(
            scenario=goal,
            persona_section=persona_section,
            context_section=context_section,
            history=history_text,
        )
        return await self._require_llm().generate(prompt)

    def _build_eval_case(self, golden: ConversationGolden, history: list[Message]) -> EvalCase:
        last_assistant = ""
        for msg in reversed(history):
            if msg.role == "assistant":
                last_assistant = msg.content or ""
                break

        return EvalCase(
            input=golden.scenario,
            output=last_assistant,
            messages=history,
            metadata={
                "scenario": golden.scenario,
                "expected_outcome": golden.expected_outcome,
                "n_turns": len(history),
                **(golden.metadata or {}),
            },
            tags=golden.tags,
        )
