from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Golden:
    """Authored evaluation data — what lives in your dataset files.

    Contains the input, expected output, and context. No agent output,
    no runtime data. Feed goldens to an agent to produce EvalCases.
    """

    input: str | dict | list
    expected: str | dict | list | None = None
    context: list[str] | None = None
    expected_tools: list[str] | None = None
    metadata: dict[str, Any] | None = field(default=None)
    tags: dict[str, str] | None = field(default=None)

    def meta(self, key: str, default: Any = None) -> Any:
        """Safely retrieve a metadata value without ``(self.metadata or {})`` boilerplate."""
        return (self.metadata or {}).get(key, default)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> Golden:
        """Create a Golden from a dict. Accepts ``expected_output`` as alias for ``expected``."""
        mapped = dict(data)
        if "expected_output" in mapped and "expected" not in mapped:
            mapped["expected"] = mapped.pop("expected_output")

        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in mapped.items() if k in known})
