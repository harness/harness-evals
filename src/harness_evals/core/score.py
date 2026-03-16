from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Score:
    """Result of a single metric evaluation.

    ``value`` is normalized to [0.0, 1.0]. ``passed`` is auto-computed
    as ``value >= threshold`` — never set it directly.
    """

    name: str
    value: float
    threshold: float
    passed: bool = field(init=False)
    reason: str | None = None
    metadata: dict[str, Any] | None = field(default=None)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.passed = self.value >= self.threshold

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return {k: v for k, v in d.items() if v is not None}
