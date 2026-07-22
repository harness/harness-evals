"""ConversationSimulator — drives multi-turn conversations between a simulated user and an agent."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict

from harness_evals._async_compat import _run_async
from harness_evals.conversation.golden import ConversationGolden, ConversationMode
from harness_evals.conversation.graph import (
    BranchNode,
    LLMNode,
    ScriptedNode,
    SimulationGraph,
    StopNode,
)
from harness_evals.conversation.human_input import HumanInputSimulator, PendingHumanInput
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM
from harness_evals.logging_config import compact_json

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
        self,
        simulator_llm: BaseLLM | None = None,
        *,
        max_concurrent: int = 5,
        graph: SimulationGraph | None = None,
        human_input_simulator: HumanInputSimulator | None = None,
        elicitation_simulator: HumanInputSimulator | None = None,
    ) -> None:
        self.simulator_llm = simulator_llm
        self.max_concurrent = max_concurrent
        self.graph = graph
        self.human_input_simulator = human_input_simulator or elicitation_simulator
        self.elicitation_simulator = self.human_input_simulator

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
        from harness_evals.conversation.context import (
            conversation_key_for_golden,
            reset_conversation_key,
            set_conversation_key,
        )

        key_token = set_conversation_key(conversation_key_for_golden(golden))
        try:
            return await self._simulate(golden, agent_fn)
        finally:
            reset_conversation_key(key_token)

    async def _simulate(
        self,
        golden: ConversationGolden,
        agent_fn: Callable[[list[Message]], Awaitable[Message]],
    ) -> EvalCase:
        self._reset_adapter_intent_misses()
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

            assistant_msg = await self._call_agent(agent_fn, history)
            assistant_msg = await self._resolve_elicitations(golden, agent_fn, history, assistant_msg)
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
        if not history and golden.initial_prompt:
            return golden.initial_prompt

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
                assistant_msg = await self._call_agent(agent_fn, history)
                assistant_msg = await self._resolve_elicitations(golden, agent_fn, history, assistant_msg)
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
            assistant_msg = await self._call_agent(agent_fn, history)
            assistant_msg = await self._resolve_elicitations(golden, agent_fn, history, assistant_msg)
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
            if msg.role == "assistant" and msg.content:
                last_assistant = msg.content
                break
        if not last_assistant:
            for msg in reversed(history):
                if msg.role == "assistant":
                    last_assistant = msg.content or ""
                    break

        metadata = {
            "scenario": golden.scenario,
            "expected_outcome": golden.expected_outcome,
            "n_turns": len(history),
            **(golden.metadata or {}),
        }
        sse_events = _sse_events_from_history(history)
        if sse_events:
            metadata["sse_events"] = sse_events
            metadata["sse_event_names"] = sorted(sse_events)
        last_assistant_meta = next(
            (msg.metadata for msg in reversed(history) if msg.role == "assistant" and msg.metadata),
            None,
        )
        if isinstance(last_assistant_meta, dict):
            for key in ("elicitation_rounds", "elicitation_error"):
                if key in last_assistant_meta:
                    metadata[key] = last_assistant_meta[key]

        intent_misses = self._adapter_intent_misses()
        if intent_misses:
            metadata["elicitation_intent_misses"] = intent_misses

        return EvalCase(
            input=golden.scenario,
            output=last_assistant,
            messages=history,
            metadata=metadata,
            tags=golden.tags,
        )

    async def _resolve_elicitations(
        self,
        golden: ConversationGolden,
        agent_fn: Callable,
        history: list[Message],
        assistant_msg: Message,
    ) -> Message:
        if self.human_input_simulator is None:
            return assistant_msg

        accumulated_sse: dict[str, list] = {}
        _merge_sse_events(accumulated_sse, assistant_msg.metadata)

        rounds = 0
        while rounds < golden.max_elicitation_rounds:
            pending = _pending_human_input(assistant_msg)
            if pending is None:
                _logger.debug(
                    "Elicitation complete after %d round(s); last_content=%r events=%s",
                    rounds,
                    assistant_msg.content,
                    sorted((assistant_msg.metadata or {}).get("sse_events", {})),
                )
                return _attach_sse_events(assistant_msg, accumulated_sse)

            _logger.debug(
                "Elicitation round %d/%d: pending=%s correlation_id=%s",
                rounds + 1,
                golden.max_elicitation_rounds,
                pending.type,
                pending.correlation_id,
            )
            human_input = await self.human_input_simulator.respond(pending, golden, history)
            _logger.debug(
                "Elicitation round %d/%d human_input: type=%s correlation_id=%s response=%s",
                rounds + 1,
                golden.max_elicitation_rounds,
                pending.type,
                pending.correlation_id,
                compact_json(human_input),
            )
            assistant_msg = await self._call_agent(agent_fn, history, human_input=human_input)
            _merge_sse_events(accumulated_sse, assistant_msg.metadata)
            rounds += 1

        _logger.warning(
            "Elicitation stopped: max_elicitation_rounds=%d exceeded; events=%s",
            golden.max_elicitation_rounds,
            sorted(accumulated_sse),
        )

        metadata = dict(assistant_msg.metadata or {})
        metadata["elicitation_error"] = "max_elicitation_rounds_exceeded"
        metadata["elicitation_rounds"] = rounds
        assistant_msg.metadata = metadata
        return _attach_sse_events(assistant_msg, accumulated_sse)

    async def _call_agent(
        self,
        agent_fn: Callable,
        history: list[Message],
        *,
        human_input: dict | None = None,
        system_event: dict | None = None,
    ) -> Message:
        continuation = human_input if human_input is not None else system_event
        if continuation is None:
            return await agent_fn(list(history))

        if _accepts_human_input(agent_fn):
            return await agent_fn(list(history), human_input=continuation)

        if _accepts_system_event(agent_fn):
            return await agent_fn(list(history), system_event=continuation)

        raise TypeError(
            "agent_fn must accept a 'human_input' or 'system_event' keyword argument "
            "when human-input simulation is enabled"
        )

    def _reset_adapter_intent_misses(self) -> None:
        simulator = self.human_input_simulator
        if simulator is None or simulator.adapter is None:
            return
        reset = getattr(simulator.adapter, "reset_intent_misses", None)
        if callable(reset):
            reset()

    def _adapter_intent_misses(self) -> list[dict] | None:
        simulator = self.human_input_simulator
        if simulator is None or simulator.adapter is None:
            return None
        misses = getattr(simulator.adapter, "intent_misses", None)
        if not misses:
            return None
        return [asdict(miss) for miss in misses]


def _merge_sse_events(accumulator: dict[str, list], metadata: dict | None) -> dict[str, list]:
    if not metadata:
        return accumulator
    events = metadata.get("sse_events")
    if not isinstance(events, dict):
        return accumulator
    for name, payloads in events.items():
        if isinstance(payloads, list):
            accumulator.setdefault(name, []).extend(payloads)
    return accumulator


def _attach_sse_events(message: Message, sse_events: dict[str, list]) -> Message:
    if not sse_events:
        return message
    metadata = dict(message.metadata or {})
    metadata["sse_events"] = sse_events
    message.metadata = metadata
    return message


def _sse_events_from_history(history: list[Message]) -> dict[str, list]:
    accumulated: dict[str, list] = {}
    for msg in history:
        if msg.role == "assistant":
            _merge_sse_events(accumulated, msg.metadata)
    return accumulated


def _pending_human_input(message: Message) -> PendingHumanInput | None:
    metadata = message.metadata or {}
    raw = metadata.get("pending_human_input") or metadata.get("pending_elicitation")
    if not isinstance(raw, dict):
        return None
    return PendingHumanInput.from_metadata(raw)


def _accepts_human_input(agent_fn: Callable) -> bool:
    try:
        sig = inspect.signature(agent_fn)
    except (TypeError, ValueError):
        return True
    return "human_input" in sig.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in sig.parameters.values()
    )


def _accepts_system_event(agent_fn: Callable) -> bool:
    try:
        sig = inspect.signature(agent_fn)
    except (TypeError, ValueError):
        return True
    return "system_event" in sig.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in sig.parameters.values()
    )
