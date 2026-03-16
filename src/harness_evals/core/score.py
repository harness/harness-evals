from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Score:
    """Result of a single metric evaluation.

    value is normalized to [0.0, 1.0]. success = value >= threshold.
    """

    name: str
    value: float
    threshold: float
    success: bool
    reason: str | None = None
    metadata: dict[str, Any] | None = field(default=None)
