"""Dataset loading and saving utilities.

A Dataset is just a list[Golden] — no ORM, no versioning.
The loader and saver are the only conveniences.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from harness_evals.core.golden import Golden

Dataset = list[Golden]

logger = logging.getLogger(__name__)


def load_dataset(path: str, format: str = "jsonl") -> Dataset:
    """Load goldens from a JSONL or JSON file.

    Args:
        path: Path to the dataset file.
        format: ``"jsonl"`` (one JSON object per line) or ``"json"`` (JSON array).

    Returns:
        A list of Golden instances. Malformed lines are skipped with a warning.
    """
    text = Path(path).read_text(encoding="utf-8")

    if format == "json":
        raw_items = json.loads(text)
        if not isinstance(raw_items, list):
            raise ValueError(f"Expected a JSON array, got {type(raw_items).__name__}")
        return [Golden.from_dict(item) for item in raw_items]

    if format == "jsonl":
        dataset: Dataset = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                dataset.append(Golden.from_dict(obj))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Skipping malformed line %d: %s", i, e)
        return dataset

    raise ValueError(f"Unsupported format: {format!r}. Use 'jsonl' or 'json'.")


def save_dataset(dataset: Dataset, path: str, format: str = "jsonl") -> None:
    """Write goldens to a JSONL or JSON file.

    Args:
        dataset: List of Golden instances to save.
        path: Output file path.
        format: ``"jsonl"`` or ``"json"``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with p.open("w", encoding="utf-8") as f:
            for golden in dataset:
                f.write(json.dumps(golden.to_dict(), ensure_ascii=False) + "\n")
    elif format == "json":
        with p.open("w", encoding="utf-8") as f:
            json.dump([g.to_dict() for g in dataset], f, ensure_ascii=False, indent=2)
            f.write("\n")
    else:
        raise ValueError(f"Unsupported format: {format!r}. Use 'jsonl' or 'json'.")
