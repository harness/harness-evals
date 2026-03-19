from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Score:
    """Result of a single metric evaluation.

    ``value`` must be in [0.0, 1.0]. ``passed`` is a read-only property
    computed dynamically as ``value >= threshold``.
    """

    name: str
    value: float
    threshold: float
    reason: str | None = None
    metadata: dict[str, Any] | None = field(default=None)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(
                f"Score.value must be between 0.0 and 1.0, got {self.value} "
                f"(metric={self.name!r}). Fix the metric implementation."
            )

    @property
    def passed(self) -> bool:
        return self.value >= self.threshold

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        d["created_at"] = self.created_at.isoformat()
        return {k: v for k, v in d.items() if v is not None}
