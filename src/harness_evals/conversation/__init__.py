"""Multi-turn conversation evaluation: goldens, simulation, and runners."""

from harness_evals.conversation.golden import ConversationGolden
from harness_evals.conversation.runner import (
    evaluate_conversation,
    evaluate_conversations,
)
from harness_evals.conversation.simulator import ConversationSimulator

__all__ = [
    "ConversationGolden",
    "ConversationSimulator",
    "evaluate_conversation",
    "evaluate_conversations",
    "load_conversation_dataset",
    "save_conversation_dataset",
]


import json
from pathlib import Path


def load_conversation_dataset(path: str | Path) -> list[ConversationGolden]:
    """Load a list of ConversationGolden from a JSONL or JSON file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError("JSON file must contain a list of objects")
    return [ConversationGolden.from_dict(r) for r in records]


def save_conversation_dataset(
    dataset: list[ConversationGolden],
    path: str | Path,
    *,
    format: str = "jsonl",
) -> None:
    """Save a list of ConversationGolden to a JSONL or JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if format == "jsonl":
        lines = [json.dumps(g.to_dict(), ensure_ascii=False) for g in dataset]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        p.write_text(
            json.dumps([g.to_dict() for g in dataset], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
