"""ConversationSimulator — drives multi-turn conversations between a simulated user and an agent."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

from harness_evals._async_compat import _run_async
from harness_evals.conversation.golden import ConversationGolden, ConversationMode
from harness_evals.conversation.graph import (
    BranchNode,
    LLMNode,
    ScriptedNode,
    SimulationGraph,
    StopNode,
)
from harness_evals.conversation.human_input import (
    HumanInputSimulator,
    PendingHumanInput,
    intents,
    resolve_intent,
)
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import Message, ToolCall
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
                await self._resolve_elicitations(golden, agent_fn, history, assistant_msg)
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
            await self._resolve_elicitations(golden, agent_fn, history, assistant_msg)
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
        sse_timeline = _sse_timeline_from_history(history)
        if sse_timeline:
            metadata["sse_timeline"] = sse_timeline
        last_assistant_meta = next(
            (msg.metadata for msg in reversed(history) if msg.role == "assistant" and msg.metadata),
            None,
        )
        if isinstance(last_assistant_meta, dict):
            for key in ("elicitation_rounds", "elicitation_error", "elicitation_trace"):
                if key in last_assistant_meta:
                    metadata[key] = last_assistant_meta[key]

        intent_misses = self._adapter_intent_misses()
        if intent_misses:
            metadata["elicitation_intent_misses"] = intent_misses

        expanded_messages = _history_with_chronological_tool_events(history)
        # The expanded trace needs a per-turn ``sse_timeline``; fall back to the
        # merged event map when only that was captured.
        tool_calls = _tool_calls_from_messages(expanded_messages) or _tool_calls_from_sse_events(sse_events)

        return EvalCase(
            input=golden.scenario,
            output=last_assistant,
            messages=expanded_messages,
            tool_calls=tool_calls or None,
            expected_tool_calls=golden.expected_tool_calls,
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
            if not history or history[-1] is not assistant_msg:
                history.append(assistant_msg)
            return assistant_msg

        accumulated_sse: dict[str, list] = {}
        accumulated_timeline: list[dict[str, Any]] = []
        _merge_sse_events(accumulated_sse, assistant_msg.metadata)
        _merge_sse_timeline(accumulated_timeline, assistant_msg.metadata)
        if not history or history[-1] is not assistant_msg:
            history.append(assistant_msg)

        trace: list[dict[str, Any]] = []
        used_plain_text_intents: set[str] = set()
        rounds = 0
        while rounds < golden.max_elicitation_rounds:
            pending = _pending_human_input(assistant_msg)
            if pending is None:
                followup = _plain_text_followup(
                    golden,
                    assistant_msg,
                    used_intents=used_plain_text_intents,
                )
                if followup is not None:
                    for intent_key in followup.get("intents") or [followup["intent"]]:
                        used_plain_text_intents.add(str(intent_key))
                    simulated_user = Message(
                        role="user",
                        content=followup["answer"],
                        metadata={
                            "simulated": True,
                            "plain_text_followup": True,
                            "intent": followup["intent"],
                        },
                    )
                    history.append(simulated_user)
                    trace.append(
                        {
                            "round": rounds + 1,
                            "kind": "plain_text_user_reply",
                            "intent": followup["intent"],
                            "content": followup["answer"],
                            "trigger": (assistant_msg.content or "")[:240],
                        }
                    )
                    assistant_msg = await self._call_agent(agent_fn, history)
                    _merge_sse_events(accumulated_sse, assistant_msg.metadata)
                    _merge_sse_timeline(accumulated_timeline, assistant_msg.metadata)
                    history.append(assistant_msg)
                    rounds += 1
                    continue

                if rounds == 0 and golden.elicitation_hints:
                    trace.append(
                        {
                            "round": 0,
                            "kind": "no_elicitation_detected",
                            "reason": (
                                "Agent completed without structured elicitation events "
                                "(elicitation_form/yaml/select/multi_select/free_text); "
                                "simulator did not send a response."
                            ),
                            "assistant_preview": (assistant_msg.content or "")[:240],
                        }
                    )
                incomplete = _incomplete_after_elicitation(assistant_msg, golden, accumulated_sse, rounds)
                if incomplete:
                    metadata = dict(assistant_msg.metadata or {})
                    metadata["elicitation_error"] = incomplete
                    assistant_msg.metadata = metadata
                    trace.append(
                        {
                            "round": rounds,
                            "kind": "incomplete_after_elicitation",
                            "reason": incomplete,
                            "assistant_preview": (assistant_msg.content or "")[:240],
                        }
                    )
                    _logger.warning(
                        "Elicitation incomplete after %d round(s): %s; events=%s",
                        rounds,
                        incomplete,
                        sorted(accumulated_sse),
                    )
                else:
                    _logger.debug(
                        "Elicitation complete after %d round(s); last_content=%r events=%s",
                        rounds,
                        assistant_msg.content,
                        sorted((assistant_msg.metadata or {}).get("sse_events", {})),
                    )
                return _finalize_elicitation(assistant_msg, accumulated_sse, accumulated_timeline, trace, rounds)

            _logger.debug(
                "Elicitation round %d/%d: pending=%s correlation_id=%s",
                rounds + 1,
                golden.max_elicitation_rounds,
                pending.type,
                pending.correlation_id,
            )
            human_input = await self.human_input_simulator.respond(pending, golden, history)
            simulated_user = _simulated_user_message(pending, human_input)
            history.append(simulated_user)
            trace.append(
                {
                    "round": rounds + 1,
                    "kind": "structured_elicitation_reply",
                    "pending_type": pending.type,
                    "correlation_id": pending.correlation_id,
                    "simulated_user_content": simulated_user.content,
                }
            )
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
            _merge_sse_timeline(accumulated_timeline, assistant_msg.metadata)
            history.append(assistant_msg)
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
        trace.append(
            {
                "round": rounds,
                "kind": "max_rounds_exceeded",
                "reason": f"Stopped after {rounds} elicitation round(s).",
            }
        )
        return _finalize_elicitation(assistant_msg, accumulated_sse, accumulated_timeline, trace, rounds)

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


def _finalize_elicitation(
    assistant_msg: Message,
    accumulated_sse: dict[str, list],
    accumulated_timeline: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    rounds: int,
) -> Message:
    assistant_msg = _attach_sse_events(assistant_msg, accumulated_sse)
    metadata = dict(assistant_msg.metadata or {})
    if accumulated_timeline:
        metadata["sse_timeline"] = accumulated_timeline
    metadata["elicitation_rounds"] = rounds
    if trace:
        metadata["elicitation_trace"] = trace
    assistant_msg.metadata = metadata
    return assistant_msg


def _simulated_user_message(pending: PendingHumanInput, human_input: dict) -> Message:
    return Message(
        role="user",
        content=_human_input_preview(pending, human_input),
        metadata={
            "simulated": True,
            "elicitation_type": pending.type,
            "correlation_id": pending.correlation_id,
        },
    )


def _human_input_preview(pending: PendingHumanInput, human_input: dict) -> str:
    result = human_input.get("result") if isinstance(human_input.get("result"), dict) else {}
    if pending.type == "elicitation_yaml":
        action_id = str(result.get("action_id") or "accept")
        entity_type = human_input.get("entity_type") or result.get("entity_type") or "entity"
        return f"[Simulated YAML review: {action_id} {entity_type}]"
    if result.get("free_text"):
        return str(result["free_text"])
    form_values = result.get("form_values")
    if isinstance(form_values, dict) and form_values:
        return ", ".join(f"{label}={value}" for label, value in form_values.items())
    selections = result.get("selections")
    if isinstance(selections, list) and selections:
        return ", ".join(str(item) for item in selections)
    selection = result.get("selection") or result.get("selected_value")
    if selection:
        return str(selection)
    return f"[Simulated {pending.type} response]"


def _incomplete_after_elicitation(
    assistant_msg: Message,
    golden: ConversationGolden,
    accumulated_sse: dict[str, list],
    rounds: int,
) -> str | None:
    """Detect wizard runs that stop with empty content before expected tools fire."""
    if rounds <= 0:
        return None
    if (assistant_msg.content or "").strip():
        return None

    expected_tools = _expected_tool_names_from_golden(golden)
    if expected_tools and not _sse_contains_any_tool(accumulated_sse, expected_tools):
        return (
            "incomplete_empty_after_elicitation: expected tool(s) "
            f"{expected_tools} never appeared after {rounds} elicitation round(s)"
        )
    if not expected_tools:
        return (
            "incomplete_empty_after_elicitation: assistant content empty after "
            f"{rounds} elicitation round(s) with no completion message"
        )
    return None


def _expected_tool_names_from_golden(golden: ConversationGolden) -> list[str]:
    if golden.expected_tool_calls:
        return [call.name for call in golden.expected_tool_calls]
    checks = (golden.metadata or {}).get("sse_checks") or []
    names: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("event") != "assistant_tool_request":
            continue
        for match in check.get("match") or []:
            if not isinstance(match, dict):
                continue
            path = str(match.get("path") or "")
            contains = match.get("contains")
            if "name" in path and isinstance(contains, str) and contains:
                names.append(contains)
    return names


def _sse_contains_any_tool(accumulated_sse: dict[str, list], tool_names: list[str]) -> bool:
    payloads = accumulated_sse.get("assistant_tool_request") or []
    needles = [name.lower() for name in tool_names]
    for payload in payloads:
        text = json.dumps(payload, ensure_ascii=False).lower() if not isinstance(payload, str) else payload.lower()
        if any(needle in text for needle in needles):
            return True
    return False


def _plain_text_followup(
    golden: ConversationGolden,
    assistant_msg: Message,
    *,
    used_intents: set[str],
) -> dict[str, Any] | None:
    """Answer plain-text assistant questions using golden elicitation_hints."""
    content = assistant_msg.content or ""
    if not golden.elicitation_hints or not content.strip():
        return None
    if not _looks_like_user_question(content):
        return None

    lowered = content.lower()
    if (
        "cost category" in lowered
        and sum(1 for token in ("name", "bucket", "structure", "organize") if token in lowered) >= 2
    ):
        composite = _composite_cost_category_answer(golden, used_intents)
        if composite is not None:
            return composite

    intent = resolve_intent(content, golden)
    if intent and intent not in used_intents:
        answer = intents(golden).get(intent, "")
        if answer:
            return {"intent": intent, "answer": answer, "intents": [intent]}

    return None


def _composite_cost_category_answer(
    golden: ConversationGolden,
    used_intents: set[str],
) -> dict[str, Any] | None:
    parts: list[str] = []
    consumed: list[str] = []
    templates = [
        ("category_name", "Name it {value}."),
        ("name_template", "Use the template name {value}."),
        ("bucketing_criteria", "Organize costs by {value}."),
        ("filter_type", "Use filter type {value}."),
    ]
    intent_values = intents(golden)
    for key, template in templates:
        if key in used_intents:
            continue
        value = intent_values.get(key, "")
        if value:
            parts.append(template.format(value=value))
            consumed.append(key)
    if not parts:
        return None
    return {
        "intent": "composite",
        "answer": " ".join(parts),
        "intents": consumed,
    }


def _looks_like_user_question(content: str) -> bool:
    text = content.strip()
    if not text:
        return False
    if "?" in text:
        return True
    lowered = text.lower()
    return lowered.startswith(
        (
            "what ",
            "how ",
            "which ",
            "please ",
            "would you ",
            "can you ",
            "could you ",
        )
    )


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


def _merge_sse_timeline(accumulator: list[dict[str, Any]], metadata: dict | None) -> list[dict[str, Any]]:
    if not metadata:
        return accumulator
    timeline = metadata.get("sse_timeline")
    if isinstance(timeline, list):
        for entry in timeline:
            if isinstance(entry, dict) and entry.get("event") is not None:
                accumulator.append(entry)
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


def _sse_timeline_from_history(history: list[Message]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for msg in history:
        if msg.role == "assistant":
            _merge_sse_timeline(timeline, msg.metadata)
    return timeline


def _history_without_message_sse_events(history: list[Message]) -> list[Message]:
    """Drop per-message ``sse_events`` once aggregated onto ``EvalCase.metadata``.

    Metrics read the canonical top-level ``metadata["sse_events"]``; keeping the
    same payloads on assistant messages too makes JSON traces unreadably large.
    """
    cleaned: list[Message] = []
    for msg in history:
        if not msg.metadata:
            cleaned.append(msg)
            continue
        drop_keys = {"sse_events", "sse_timeline"}
        if msg.metadata.get("simulated"):
            drop_keys.add("system_event")
        if not any(key in msg.metadata for key in drop_keys):
            cleaned.append(msg)
            continue
        metadata = {key: value for key, value in msg.metadata.items() if key not in drop_keys}
        cleaned.append(
            Message(
                role=msg.role,
                content=msg.content,
                tool_calls=msg.tool_calls,
                latency_ms=msg.latency_ms,
                token_count=msg.token_count,
                cost_usd=msg.cost_usd,
                metadata=metadata or None,
                retrieval_context=msg.retrieval_context,
                expected=msg.expected,
            )
        )
    return cleaned


def _history_with_chronological_tool_events(history: list[Message]) -> list[Message]:
    """Expand captured tool request/result events into the visible message trace.

    Each assistant turn carries its own ordered ``sse_timeline``. Preserve the
    conversation turn order while inserting tool-call and tool-result messages
    at their observed positions, then strip the raw per-message SSE payloads.
    The complete raw timeline remains available on ``EvalCase.metadata``.
    """
    expanded: list[Message] = []
    for message in history:
        cleaned = _history_without_message_sse_events([message])[0]
        timeline = (message.metadata or {}).get("sse_timeline")
        if message.role != "assistant" or not isinstance(timeline, list):
            expanded.append(cleaned)
            continue

        assistant_event_indexes = [
            index
            for index, entry in enumerate(timeline)
            if isinstance(entry, dict) and entry.get("event") == "assistant_message"
        ]
        assistant_insert_index = assistant_event_indexes[-1] if assistant_event_indexes else None
        inserted_assistant = False

        for index, entry in enumerate(timeline):
            if not isinstance(entry, dict):
                continue
            event_name = entry.get("event")
            if index == assistant_insert_index:
                expanded.append(cleaned)
                inserted_assistant = True
            if event_name == "assistant_tool_request":
                tool_calls = _tool_calls_from_sse_payload(entry.get("payload"), result=False)
                if tool_calls:
                    expanded.append(
                        Message(
                            role="assistant",
                            tool_calls=tool_calls,
                            metadata={"sse_event": event_name},
                        )
                    )
            elif event_name == "assistant_tool_result":
                for tool_call in _tool_calls_from_sse_payload(entry.get("payload"), result=True):
                    expanded.append(
                        Message(
                            role="tool",
                            content=_tool_result_content(tool_call.output),
                            tool_calls=[tool_call],
                            metadata={"sse_event": event_name},
                        )
                    )

        if not inserted_assistant:
            expanded.append(cleaned)
    return expanded


def _tool_calls_from_sse_events(sse_events: dict[str, list]) -> list[ToolCall]:
    """Collect Harness MCP tool requests straight from the captured event map."""
    calls: list[ToolCall] = []
    for payload in sse_events.get("assistant_tool_request", []):
        calls.extend(
            call for call in _tool_calls_from_sse_payload(payload, result=False) if _is_harness_mcp_tool_name(call.name)
        )
    return calls


def _tool_calls_from_messages(messages: list[Message]) -> list[ToolCall]:
    """Collect Harness MCP tool requests from the expanded message trace in order."""
    calls: list[ToolCall] = []
    for msg in messages:
        if msg.role != "assistant" or not msg.tool_calls:
            continue
        if (msg.metadata or {}).get("sse_event") != "assistant_tool_request":
            continue
        for tool_call in msg.tool_calls:
            normalized_name = _normalize_tool_name(tool_call.name)
            if not _is_harness_mcp_tool_name(normalized_name):
                continue
            calls.append(
                ToolCall(
                    name=normalized_name,
                    input=tool_call.input,
                    output=tool_call.output,
                )
            )
    return calls


def _is_harness_mcp_tool_name(name: str) -> bool:
    """True for Harness MCP tools; excludes agent utilities like Skill, Grep, Read."""
    if not name or name in _NON_HARNESS_AGENT_TOOLS:
        return False
    lowered = name.lower()
    return name.startswith("harness_") or name.startswith("validate_") or "hql" in lowered


_NON_HARNESS_AGENT_TOOLS = frozenset(
    {
        "AskUserQuestion",
        "Read",
        "Write",
        "Grep",
        "Glob",
        "Bash",
        "Task",
        "WebFetch",
        "WebSearch",
        "Skill",
    }
)


def _normalize_tool_name(name: str) -> str:
    """Strip MCP namespace prefixes so goldens can use short Harness tool names."""
    if "__" in name:
        return name.rsplit("__", 1)[-1]
    return name


def _tool_calls_from_sse_payload(payload: object, *, result: bool) -> list[ToolCall]:
    """Normalize Harness ``assistant_tool_*`` payloads into ``ToolCall`` values."""
    raw_entries = payload.get("v") if isinstance(payload, dict) and "v" in payload else payload
    if isinstance(raw_entries, dict):
        raw_entries = [raw_entries]
    if not isinstance(raw_entries, list):
        return []

    tool_calls: list[ToolCall] = []
    for entry in raw_entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        name = str(entry["name"])
        if result:
            output = entry.get("result") if "result" in entry else entry.get("output")
            tool_calls.append(ToolCall(name=_normalize_tool_name(name), output=output))
        else:
            arguments = entry.get("arguments") if "arguments" in entry else entry.get("input")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                    arguments = parsed if isinstance(parsed, dict) else {"value": parsed}
                except json.JSONDecodeError:
                    arguments = {"value": arguments}
            tool_calls.append(
                ToolCall(
                    name=_normalize_tool_name(name),
                    input=arguments if isinstance(arguments, dict) else None,
                )
            )
    return tool_calls


def _tool_result_content(output: object) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


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
