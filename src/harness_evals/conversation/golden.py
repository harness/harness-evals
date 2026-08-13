"""ConversationGolden — scenario-based golden for multi-turn evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from harness_evals.core.types import Message, ToolCall


class ConversationMode(str, Enum):
    """Mode controlling how multi-turn conversations are executed.

    SIMULATE: LLM generates user turns, agent responds each turn.
    REPLAY: Full transcript provided as-is, no agent call.
    SCRIPTED: User turns from dataset, agent called after each user turn.
    GRAPH: Declarative DAG controls user turn generation with conditional branching.
    """

    SIMULATE = "simulate"
    REPLAY = "replay"
    SCRIPTED = "scripted"
    GRAPH = "graph"


@dataclass
class ConversationGolden:
    """A scenario-based golden for multi-turn evaluation.

    Unlike Golden (single input -> expected output), this describes a
    conversation scenario that can be simulated or replayed.
    """

    scenario: str
    expected_outcome: str
    id: str | None = None
    context: list[str] | None = None
    turns: list[Message] | None = field(default=None)
    max_turns: int = 10
    max_elicitation_rounds: int = 10
    initial_prompt: str | None = None
    user_persona: str | None = None
    elicitation_hints: dict[str, Any] | None = field(default=None)
    # Optional expected agent output (e.g. pipeline YAML for structural_similarity).
    # May contain shell-style ``${var}`` placeholders; those are not eval env vars.
    expected: str | dict | list | None = None
    # Optional expected tool trajectory for tool_argument_match (names as emitted on
    # the wire, e.g. mcp__harness__harness_list or Skill).
    expected_tool_calls: list[ToolCall] | None = field(default=None)
    mode: ConversationMode | None = None
    graph_config: dict | None = field(default=None)
    metadata: dict[str, Any] | None = field(default=None)
    tags: dict[str, str] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.mode is None:
            if self.graph_config is not None:
                self.mode = ConversationMode.GRAPH
            elif self.turns:
                self.mode = ConversationMode.REPLAY
            else:
                self.mode = ConversationMode.SIMULATE

        mode = self.mode
        if mode in (ConversationMode.REPLAY, ConversationMode.SCRIPTED) and not self.turns:
            raise ValueError(f"mode={mode.value!r} requires 'turns' to be provided")

        if (
            mode == ConversationMode.SCRIPTED
            and self.turns is not None
            and not any(t.role == "user" for t in self.turns)
        ):
            raise ValueError("mode='scripted' requires at least one user-role message in 'turns'")

        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")

        if self.max_elicitation_rounds < 1:
            raise ValueError("max_elicitation_rounds must be >= 1")

        if self.elicitation_hints is not None and not isinstance(self.elicitation_hints, dict):
            raise TypeError("elicitation_hints must be a dict when provided")

        sse_checks = (self.metadata or {}).get("sse_checks")
        if sse_checks is not None and (
            not isinstance(sse_checks, list) or not all(isinstance(check, dict) for check in sse_checks)
        ):
            raise TypeError("metadata['sse_checks'] must be a list of dicts when provided")

        if self.expected_tool_calls is not None and (
            not isinstance(self.expected_tool_calls, list)
            or not all(isinstance(call, ToolCall) and call.name for call in self.expected_tool_calls)
        ):
            raise TypeError("expected_tool_calls must be a list of ToolCall values with non-empty names when provided")

    def to_dict(self) -> dict:
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        if self.mode is not None:
            result["mode"] = self.mode.value
        if self.expected_tool_calls is not None:
            result["expected_tool_calls"] = [call.to_dict() for call in self.expected_tool_calls]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ConversationGolden:
        mapped = dict(data)
        if "turns" in mapped and mapped["turns"] is not None:
            mapped["turns"] = [m if isinstance(m, Message) else Message.from_dict(m) for m in mapped["turns"]]
        if "expected_tool_calls" in mapped and mapped["expected_tool_calls"] is not None:
            mapped["expected_tool_calls"] = [
                tc if isinstance(tc, ToolCall) else ToolCall.from_dict(tc)
                for tc in mapped["expected_tool_calls"]
            ]
        if "mode" in mapped and isinstance(mapped["mode"], str):
            mapped["mode"] = ConversationMode(mapped["mode"])
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in mapped.items() if k in known})
