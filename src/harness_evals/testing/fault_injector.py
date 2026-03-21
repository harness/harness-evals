"""FaultInjector — lightweight test harness for fault injection robustness testing."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Fault:
    """A single fault configuration for injection.

    Attributes:
        type: One of ``"timeout"``, ``"malformed_response"``,
              ``"rate_limit"``, ``"empty_response"``.
        probability: Injection probability per call (0.0–1.0).
        config: Type-specific configuration (e.g. custom response body for
                ``malformed_response``).
    """

    type: str
    probability: float
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_types = ("timeout", "malformed_response", "rate_limit", "empty_response")
        if self.type not in valid_types:
            raise ValueError(f"Fault type must be one of {valid_types}, got '{self.type}'")
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability must be between 0.0 and 1.0, got {self.probability}")


class FaultInjector:
    """Wraps an async agent callable with configurable fault injection.

    On each ``run()`` call the injector rolls against each fault's
    probability; the first fault that triggers is injected.  If no fault
    triggers, the real ``agent_fn`` is called.

    Use a fixed ``seed`` for reproducible injection sequences.
    ``history`` records every call for test assertions.
    """

    def __init__(
        self,
        agent_fn: Callable[..., Any],
        faults: list[Fault],
        seed: int | None = None,
    ) -> None:
        self.agent_fn = agent_fn
        self.faults = faults
        self._rng = random.Random(seed)
        self.history: list[dict[str, Any]] = []

    async def run(self, agent_input: Any) -> Any:
        """Execute a single call, possibly injecting a fault.

        ``timeout`` and ``rate_limit`` faults raise (``TimeoutError`` /
        ``RuntimeError``).  ``malformed_response`` and ``empty_response``
        faults return a degraded value.
        """
        for fault in self.faults:
            if self._rng.random() < fault.probability:
                entry: dict[str, Any] = {
                    "input": agent_input,
                    "injected": True,
                    "fault_type": fault.type,
                }
                try:
                    result = self._apply_fault(fault)
                except Exception:
                    self.history.append(entry)
                    raise
                entry["result"] = result
                self.history.append(entry)
                return result

        result = await self.agent_fn(agent_input)
        self.history.append({"input": agent_input, "injected": False, "result": result})
        return result

    @staticmethod
    def _apply_fault(fault: Fault) -> Any:
        if fault.type == "timeout":
            raise TimeoutError("Injected timeout fault")
        if fault.type == "rate_limit":
            raise RuntimeError("rate limited")
        if fault.type == "malformed_response":
            return fault.config.get("response", "{malformed}")
        if fault.type == "empty_response":
            return ""
        raise ValueError(f"Unknown fault type: {fault.type}")
