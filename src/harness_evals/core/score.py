from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Score:
    """Result of a single metric evaluation.

    ``value`` must be in [0.0, 1.0]. ``passed`` is auto-computed
    as ``value >= threshold`` — never set it directly.
    """

    name: str
    value: float
    threshold: float
    passed: bool = field(init=False)
    reason: str | None = None
    metadata: dict[str, Any] | None = field(default=None)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    _CLAMP_EPS = 1e-6

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            if -self._CLAMP_EPS <= self.value <= 1.0 + self._CLAMP_EPS:
                self.value = max(0.0, min(1.0, self.value))
            else:
                raise ValueError(
                    f"Score.value must be between 0.0 and 1.0, got {self.value} "
                    f"(metric={self.name!r}). Fix the metric implementation."
                )
        self.passed = self.value >= self.threshold

    @classmethod
    def clamped(cls, *, name: str, value: float, threshold: float, **kwargs: Any) -> Score:
        """Create a Score, clamping ``value`` to [0.0, 1.0] without raising."""
        return cls(name=name, value=max(0.0, min(1.0, value)), threshold=threshold, **kwargs)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return {k: v for k, v in d.items() if v is not None}
