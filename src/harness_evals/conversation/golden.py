"""ConversationGolden — scenario-based golden for multi-turn evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from harness_evals.core.types import Message


@dataclass
class ConversationGolden:
    """A scenario-based golden for multi-turn evaluation.

    Unlike Golden (single input -> expected output), this describes a
    conversation scenario that can be simulated or replayed.
    """

    scenario: str
    expected_outcome: str
    context: list[str] | None = None
    turns: list[Message] | None = field(default=None)
    max_turns: int = 10
    user_persona: str | None = None
    metadata: dict[str, Any] | None = field(default=None)
    tags: dict[str, str] | None = field(default=None)

    def to_dict(self) -> dict:
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ConversationGolden:
        mapped = dict(data)
        if "turns" in mapped and mapped["turns"] is not None:
            mapped["turns"] = [m if isinstance(m, Message) else Message.from_dict(m) for m in mapped["turns"]]
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in mapped.items() if k in known})
