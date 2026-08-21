"""ConversationSimulator — drives multi-turn conversations between a simulated user and an agent."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, replace
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
        turns = golden.turns
        if turns is None and golden.initial_prompt:
            turns = [Message(role="user", content=golden.initial_prompt)]
        if not turns:
            raise ValueError("SCRIPTED mode requires turns or an initial_prompt")
        history: list[Message] = []
        for turn in turns:
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
        if golden.id:
            metadata["golden_id"] = golden.id
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

        expanded_messages = _history_with_chronological_tool_events(
            history, _expected_tool_names(golden.expected_tool_calls)
        )
        tool_calls = _tool_calls_for_eval_case(
            expanded_messages,
            expected_tool_calls=golden.expected_tool_calls,
        )
        # Goldens authored from imported traces may carry wire-prefixed names
        # (``tool:Write``). Normalize both expectation shapes with the same rule
        # applied to observed calls so metrics compare like with like.
        expected_tool_calls = (
            [replace(call, name=_short_tool_name(call.name)) for call in golden.expected_tool_calls]
            if golden.expected_tool_calls is not None
            else None
        )
        expected_tools = [call.name for call in expected_tool_calls] if expected_tool_calls is not None else None

        return EvalCase(
            input=golden.scenario,
            output=last_assistant,
            expected=golden.expected,
            expected_tools=expected_tools,
            expected_tool_calls=expected_tool_calls,
            tool_calls=tool_calls,
            messages=expanded_messages,
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
        _merge_sse_events(accumulated_sse, assistant_msg.metadata)
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
                return _finalize_elicitation(assistant_msg, trace, rounds)

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
        return _finalize_elicitation(assistant_msg, trace, rounds)

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
    trace: list[dict[str, Any]],
    rounds: int,
) -> Message:
    metadata = dict(assistant_msg.metadata or {})
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
        if human_input.get("event_type") == "action_cancelled":
            entity_info = pending.payload.get("entity_info") or {}
            entity_type = pending.payload.get("entity_type") or entity_info.get("entity_type") or "entity"
            return f"[Simulated YAML review: reject {entity_type}]"
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

    question_scope = _plain_text_question_scope(content)
    if not question_scope or not _looks_like_user_question(question_scope):
        return None

    lowered = question_scope.lower()
    if (
        "cost category" in lowered
        and sum(1 for token in ("name", "bucket", "structure", "organize") if token in lowered) >= 2
    ):
        composite = _composite_cost_category_answer(golden, used_intents)
        if composite is not None:
            return composite

    intent = resolve_intent(question_scope, golden, plain_text=True)
    if intent and intent not in used_intents:
        answer = intents(golden).get(intent, "")
        if answer:
            return {"intent": intent, "answer": answer, "intents": [intent]}

    return None


def _plain_text_question_scope(content: str) -> str:
    """Return the assistant text slice to scan for follow-up intent matchers.

    Long reports often mention scope keywords in prose (e.g. "Account level"
    in a table). Matchers should only run against the closing question block.
    """
    text = content.strip()
    if not text:
        return text

    lowered = text.lower()
    markers = (
        "\nwould you like me to",
        "\nwould you like to",
        "\nwhich scope",
        "\nwhat scope",
        "\nplease confirm",
        "\nplease let me know",
        "\nshould i ",
    )
    for marker in markers:
        idx = lowered.rfind(marker)
        if idx >= 0:
            return text[idx:].strip()

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if paragraphs and _looks_like_user_question(paragraphs[-1]):
        return paragraphs[-1]

    return text if _looks_like_user_question(text) else ""


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


def _assistant_message_from_timeline(message: Message, timeline_entry: dict[str, Any]) -> Message:
    """Prefer assistant_message SSE text when the history Message has no content yet."""
    content = (message.content or "").strip()
    if content:
        return message
    payload = timeline_entry.get("payload")
    if isinstance(payload, dict):
        text = payload.get("v")
        if isinstance(text, str) and text.strip():
            return Message(
                role=message.role,
                content=text,
                tool_calls=message.tool_calls,
                latency_ms=message.latency_ms,
                token_count=message.token_count,
                cost_usd=message.cost_usd,
                metadata=message.metadata,
                retrieval_context=message.retrieval_context,
                expected=message.expected,
            )
    return message


def _history_with_chronological_tool_events(
    history: list[Message], expected_names: set[str] | None = None
) -> list[Message]:
    """Expand captured tool request/result events into the visible message trace.

    Each assistant turn carries its own ordered ``sse_timeline``. Preserve the
    conversation turn order while inserting tool-call and tool-result messages
    at their observed positions, then strip the raw per-message SSE payloads.
    The complete raw timeline remains available on ``EvalCase.metadata``.
    """
    expanded: list[Message] = []
    for message in history:
        cleaned = _history_without_message_sse_events([message])[0]
        meta = message.metadata or {}
        timeline = meta.get("sse_timeline")
        if message.role != "assistant":
            expanded.append(cleaned)
            continue

        # A turn that captured tool requests over SSE records them under the
        # wire's ``mcp__`` marker, so its embedded copy is an SSE-shaped record
        # and takes the SSE keep rule. Without this the same live call is
        # filtered one way when it arrives on the timeline and another when it
        # arrives on the message.
        if cleaned.tool_calls and _turn_captured_sse_tool_requests(meta):
            cleaned = replace(cleaned, metadata={**(cleaned.metadata or {}), "tool_call_source": "sse"})

        names = expected_names or set()

        # Parsed with raw names: these calls feed trajectory filtering, which
        # must still see the ``mcp__`` marker. The displayed message below gets
        # the short form, and _tool_calls_for_eval_case normalizes on emit.
        timeline_tool_calls: dict[int, list[ToolCall]] = {}
        for index, entry in enumerate(timeline if isinstance(timeline, list) else []):
            if isinstance(entry, dict) and entry.get("event") == "assistant_tool_request":
                parsed = _tool_calls_from_sse_payload(entry.get("payload"), result=False, normalize_names=False)
                if parsed:
                    timeline_tool_calls[index] = parsed

        # A turn can carry three records of the same calls: embeds, grouped SSE
        # events, and an ordered SSE timeline. Exactly one may reach the expanded
        # trace. Select once across all three records; branch-specific decisions
        # cannot compare an empty or partial timeline with a fuller grouped
        # record.
        #
        # Completeness counts only calls that survive scoring, under the same
        # provenance-specific rule final emission applies. The fuller record
        # wins. Ties prefer the ordered timeline, then grouped SSE events, then
        # embeds: raw stream records are authoritative when contradictory
        # records have equal evidence. The one exception is an all-zero tie with
        # embeds, where preserving local-work calls on the transcript is more
        # useful than replacing them with equally unscorable runtime machinery.
        #
        # This precedence is deliberately independent of the golden's exact
        # sequence. Choosing the source that happens to match expectations would
        # hide a genuine unexpected call in a contradictory trace.
        aggregated = _aggregated_request_messages(meta)
        timeline_score_count = sum(
            _scoring_relevant_call_count(calls, names, keep=_keep_sse_tool_call)
            for calls in timeline_tool_calls.values()
        )
        aggregated_score_count = sum(
            _scoring_relevant_call_count(_with_raw_tool_names(message), names, keep=_keep_sse_tool_call)
            for message in aggregated
        )
        turn_score_count = _scoring_relevant_call_count(
            cleaned.tool_calls or [], names, keep=_keep_rule_for_message(cleaned)
        )
        max_score_count = max(timeline_score_count, aggregated_score_count, turn_score_count)
        preserve_zero_score_embeds = bool(cleaned.tool_calls) and max_score_count == 0
        use_timeline_calls = (
            not preserve_zero_score_embeds and bool(timeline_tool_calls) and timeline_score_count == max_score_count
        )
        use_aggregated_calls = (
            not preserve_zero_score_embeds
            and not use_timeline_calls
            and bool(aggregated)
            and aggregated_score_count == max_score_count
        )
        if cleaned.tool_calls and (use_timeline_calls or use_aggregated_calls):
            cleaned = replace(cleaned, tool_calls=None)

        assistant_event_indexes = [
            index
            for index, entry in enumerate(timeline if isinstance(timeline, list) else [])
            if isinstance(entry, dict) and entry.get("event") == "assistant_message"
        ]
        assistant_insert_index = assistant_event_indexes[-1] if assistant_event_indexes else None
        inserted_assistant = False
        # Buffer this turn so aggregated requests can be placed at the tool-request
        # position (before assistant text), not after the whole turn.
        turn_messages: list[Message] = []
        pending_aggregated = list(aggregated) if use_aggregated_calls else []

        for index, entry in enumerate(timeline if isinstance(timeline, list) else []):
            if not isinstance(entry, dict):
                continue
            event_name = entry.get("event")
            # Aggregated requests belong where timeline requests would have been —
            # at the first tool-request slot, or just before assistant text.
            if pending_aggregated and (event_name == "assistant_tool_request" or index == assistant_insert_index):
                turn_messages.extend(pending_aggregated)
                pending_aggregated = []
            if index == assistant_insert_index:
                turn_messages.append(_assistant_message_from_timeline(cleaned, entry))
                inserted_assistant = True
            if event_name == "assistant_tool_request" and use_timeline_calls:
                tool_calls = timeline_tool_calls.get(index)
                if tool_calls:
                    turn_messages.append(
                        Message(
                            role="assistant",
                            tool_calls=[replace(call, name=_short_tool_name(call.name)) for call in tool_calls],
                            metadata={"sse_event": event_name, "raw_tool_names": [call.name for call in tool_calls]},
                        )
                    )
            elif event_name == "assistant_tool_result" and use_timeline_calls:
                # Keep requests and results together: when arbitration drops the
                # timeline's requests, its results must go with them or the
                # transcript shows a tool result the agent never requested.
                for tool_call in _tool_calls_from_sse_payload(entry.get("payload"), result=True):
                    turn_messages.append(
                        Message(
                            role="tool",
                            content=_tool_result_content(tool_call.output),
                            tool_calls=[tool_call],
                            metadata={"sse_event": event_name},
                        )
                    )

        if not inserted_assistant:
            if pending_aggregated:
                turn_messages.extend(pending_aggregated)
            turn_messages.append(cleaned)
        expanded.extend(turn_messages)
    return expanded


def _turn_captured_sse_tool_requests(metadata: dict[str, Any]) -> bool:
    """Whether this turn actually recorded a tool request over SSE.

    Provenance has to come from a captured request, not from the presence of SSE
    keys. A turn carrying only assistant-text events, an empty ``sse_timeline``,
    or a timeline whose payloads do not parse has said nothing about how its
    tool calls were produced — treating that as SSE applies the closed keep rule
    to a trace-imported record, and a genuinely wrong bare-named tool
    disappears from scoring while still showing on the transcript.
    """
    events = metadata.get("sse_events")
    if isinstance(events, dict) and _raw_tool_calls_from_sse_events(events):
        return True
    timeline = metadata.get("sse_timeline")
    if not isinstance(timeline, list):
        return False
    return any(
        isinstance(entry, dict)
        and entry.get("event") == "assistant_tool_request"
        and _tool_calls_from_sse_payload(entry.get("payload"), result=False, normalize_names=False)
        for entry in timeline
    )


def _aggregated_request_messages(metadata: dict[str, Any]) -> list[Message]:
    """Render one turn's own ``sse_events`` tool requests as trace messages.

    ``sse_events`` is captured per turn and only later flattened across the
    conversation by ``_sse_events_from_history``. Reading it here, while the
    owning turn is still in hand, is what makes source selection a per-record
    decision: the flattened map cannot say which turn a call came from, so any
    choice made from it applies to every turn at once.
    """
    events = metadata.get("sse_events")
    raw = _raw_tool_calls_from_sse_events(events) if isinstance(events, dict) else []
    if not raw:
        return []
    return [
        Message(
            role="assistant",
            tool_calls=[replace(call, name=_short_tool_name(call.name)) for call in raw],
            metadata={"sse_event": "assistant_tool_request", "raw_tool_names": [call.name for call in raw]},
        )
    ]


def _tool_calls_for_eval_case(
    expanded_messages: list[Message],
    expected_tool_calls: list[ToolCall] | None = None,
) -> list[ToolCall] | None:
    """Return observed tool calls, distinguishing missing capture from empty trajectory.

    The expanded trace is the single source. Every capture mode — ``sse_timeline``
    entries, a turn's aggregated ``sse_events``, and plain REPLAY embeds — has
    already been reduced to at most one record per turn by
    ``_history_with_chronological_tool_events``, which is the only place holding
    the turn-to-record association. Reading a conversation-wide aggregate here
    instead cannot express "this turn already reported these calls", so it either
    double-counts turns that captured both ways or drops turns that captured
    neither.

    ``None`` — no assistant behavior was captured at all, so the trajectory is
    unknown (``skip_when_missing`` may apply).
    ``[]`` — assistant behavior was captured and no (scorable) tools were
    requested. This is a real trajectory that metrics must score, not a skip.

    Keep rules differ by record shape because the records do not carry the same
    information (``_keep_rule_for_message`` picks per message):

    * **SSE-shaped**: closed over ``mcp__``. An unrecognized runtime tool
      degrades to "not scored".
    * **REPLAY / trace-imported** embeds: open denylist. A bare
      ``lookup_order`` may be the tool under test, so unexpected non-builtins
      stay visible to metrics.

    An unexpected bare name is therefore penalized on REPLAY and invisible on
    SSE. That divergence is intentional.
    """
    expected_names = _expected_tool_names(expected_tool_calls)
    kept: list[ToolCall] = []
    saw_assistant_turn = False

    for msg in expanded_messages:
        if msg.role != "assistant":
            continue
        saw_assistant_turn = True
        if msg.tool_calls:
            kept.extend(
                _filter_and_normalize_tool_calls(
                    _with_raw_tool_names(msg), expected_names, keep=_keep_rule_for_message(msg)
                )
            )
    if not saw_assistant_turn:
        return None
    return kept


def _keep_rule_for_message(message: Message) -> Callable[[ToolCall, set[str]], bool]:
    """Pick the keep rule matching where a turn's calls came from.

    A message synthesized from an ``sse_timeline`` entry or from a turn's
    aggregated ``sse_events``, and a live turn's own embedded copy of those
    calls, are all SSE-shaped records carrying the ``mcp__`` marker, so they
    take the closed SSE rule; filtering them with the denylist would score an
    unenumerated runtime tool as a wrong tool. Trace-imported calls have no such
    marker and take the denylist rule. See ``_tool_calls_for_eval_case``.
    """
    metadata = message.metadata or {}
    if metadata.get("sse_event") == "assistant_tool_request" or metadata.get("tool_call_source") == "sse":
        return _keep_sse_tool_call
    return _keep_message_embedded_tool_call


def _filter_and_normalize_tool_calls(
    raw_calls: list[ToolCall],
    expected_names: set[str],
    *,
    keep: Callable[[ToolCall, set[str]], bool],
) -> list[ToolCall]:
    """Apply a path's keep rule, then normalize names once on the way out."""
    kept = [call for call in raw_calls if keep(call, expected_names)]
    return [ToolCall(name=_short_tool_name(call.name), input=call.input, output=call.output) for call in kept]


def _keep_sse_tool_call(call: ToolCall, expected_names: set[str]) -> bool:
    """SSE keep rule: MCP-routed calls, plus any name the golden expects.

    Closed by construction — it never has to recognize a runtime builtin, so a
    tool the runtime adds tomorrow cannot turn a passing trajectory into a
    failing one. See ``_tool_calls_for_eval_case`` for why SSE and REPLAY
    differ here.
    """
    return call.name.removeprefix("tool:").startswith("mcp__") or _is_expected_by_name(call, expected_names)


# Agent-runtime builtins that trace import puts on Message.tool_calls but that
# goldens never list in expected_tool_calls (those are Harness tools only). Used by
# the REPLAY keep rule *only* — SSE is closed over the mcp__ prefix and never
# consults this list, so a builtin missing from it cannot fail a live trajectory
# (see _tool_calls_for_eval_case). Best-effort by nature: it cannot enumerate every
# runtime's vocabulary, so treat it as reducing REPLAY false failures, not as
# something correctness depends on. A golden that explicitly expects one of these
# names keeps it (_is_expected_by_name), which is what makes the list safe for
# agents that genuinely expose a tool named Write.
_AGENT_INTERNAL_TOOL_NAMES = frozenset(
    {
        "AskUserQuestion",
        "Bash",
        "BashOutput",
        "Edit",
        "ExitPlanMode",
        "Glob",
        "Grep",
        "KillShell",
        "LS",
        "MultiEdit",
        "NotebookEdit",
        "NotebookRead",
        "Read",
        "Skill",
        "SlashCommand",
        "Task",
        "TodoWrite",
        "WebFetch",
        "WebSearch",
        "Write",
    }
)


def _expected_tool_names(expected_tool_calls: list[ToolCall] | None) -> set[str]:
    """Expected tool names under both the raw and normalized spellings.

    Goldens authored from imported traces may carry wire-prefixed names, so
    matching on the raw name alone would miss.
    """
    names = {call.name for call in (expected_tool_calls or [])}
    return names | {_short_tool_name(name) for name in names}


def _scoring_relevant_call_count(
    calls: list[ToolCall],
    expected_names: set[str],
    *,
    keep: Callable[[ToolCall, set[str]], bool],
) -> int:
    """Count the calls that will still be there once filtering runs.

    ``keep`` must be the rule that will actually filter these calls, or the
    count overstates one side of a timeline-vs-turn comparison.
    """
    return sum(1 for call in calls if keep(call, expected_names))


def _with_raw_tool_names(message: Message) -> list[ToolCall]:
    """Return a turn's tool calls under the raw wire names used for filtering.

    Timeline-derived messages carry short names for display and their raw names
    in ``metadata["raw_tool_names"]``. Filtering has to read the raw form, or an
    ``mcp__server__tool`` whose short name matches an agent builtin is dropped
    on this path while the SSE path keeps it.
    """
    calls = message.tool_calls or []
    raw_names = (message.metadata or {}).get("raw_tool_names")
    if not isinstance(raw_names, list) or len(raw_names) != len(calls):
        return list(calls)
    return [
        replace(call, name=raw_name) if isinstance(raw_name, str) else call
        for call, raw_name in zip(calls, raw_names, strict=True)
    ]


def _is_agent_internal_tool_name(name: str) -> bool:
    """Return True for agent-runtime builtins (under any wire prefix).

    An ``mcp__server__tool`` name is never one of these, however its short form
    reads: MCP-routed calls are the tools under test, and the SSE path keeps
    every one of them. Without this guard a server tool whose short name
    collides with a builtin (``mcp__crm__Task``) would be dropped on REPLAY and
    kept on SSE — the capture-mode divergence this denylist exists to remove.
    """
    bare = name.removeprefix("tool:")
    if bare.startswith("mcp__"):
        return False
    return bare in _AGENT_INTERNAL_TOOL_NAMES


def _is_expected_by_name(call: ToolCall, expected_names: set[str]) -> bool:
    """Return True when the golden expects this call, raw or normalized."""
    return call.name in expected_names or _short_tool_name(call.name) in expected_names


def _keep_message_embedded_tool_call(call: ToolCall, expected_names: set[str]) -> bool:
    """Drop agent-internal builtins unless the golden explicitly expects them."""
    if _is_expected_by_name(call, expected_names):
        return True
    return not _is_agent_internal_tool_name(call.name)


def _tool_calls_from_sse_payload(
    payload: object,
    *,
    result: bool,
    normalize_names: bool = True,
) -> list[ToolCall]:
    """Normalize Harness ``assistant_tool_*`` payloads into ``ToolCall`` values.

    Clear ``normalize_names`` to keep the raw wire name. Callers that filter
    before scoring need it: shortening first discards the ``mcp__`` marker, and
    a server tool whose short name collides with an agent builtin
    (``mcp__crm__Task``) would then be misread as internal and dropped.
    """
    raw_entries = payload.get("v") if isinstance(payload, dict) and "v" in payload else payload
    if isinstance(raw_entries, dict):
        raw_entries = [raw_entries]
    if not isinstance(raw_entries, list):
        return []

    tool_calls: list[ToolCall] = []
    for entry in raw_entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        raw_name = str(entry["name"])
        name = _short_tool_name(raw_name) if normalize_names else raw_name
        if result:
            output = entry.get("result") if "result" in entry else entry.get("output")
            tool_calls.append(ToolCall(name=name, output=output))
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
                    name=name,
                    input=arguments if isinstance(arguments, dict) else None,
                )
            )
    return tool_calls


def _short_tool_name(name: str) -> str:
    """Reduce a raw tool name to the short form goldens author.

    Strips both wire prefixes seen on imported traces: the Langfuse/OTel
    ``tool:`` span prefix (``tool:Read``, see ``importers/otel.py``) and the
    MCP ``mcp__server__`` prefix. This is the single normalization used for
    both filtering and emission — a name that passes a filter under its short
    form must be emitted under that same form, or metrics compare
    ``tool:lookup_order`` against an expected ``lookup_order`` and score 0.0.
    """
    bare = name.removeprefix("tool:")
    return bare.rsplit("__", 1)[-1] if bare.startswith("mcp__") else bare


def _raw_tool_calls_from_sse_events(sse_events: dict[str, list]) -> list[ToolCall]:
    """Flatten assistant_tool_request payloads into chronological ToolCall values.

    Returns calls under their *raw* wire names and unfiltered; the caller
    (``_tool_calls_for_eval_case``) applies the SSE keep rule and name
    normalization. Filtering here would discard the ``mcp__`` marker that rule
    depends on. The full trajectory, including agent builtins, remains on
    ``eval_case.messages`` via ``_history_with_chronological_tool_events``.
    """
    calls: list[ToolCall] = []
    for payload in sse_events.get("assistant_tool_request") or []:
        calls.extend(_tool_calls_from_sse_payload(payload, result=False, normalize_names=False))
    return calls


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
