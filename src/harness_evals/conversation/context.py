"""Per-conversation context for isolating session state across concurrent simulations."""

from __future__ import annotations

import contextvars
from typing import Any

_conversation_key: contextvars.ContextVar[str | None] = contextvars.ContextVar("conversation_key", default=None)
_request_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "request_context", default=None
)


def conversation_key_for_golden(golden: object) -> str:
    """Return a stable key for session isolation within one simulation run."""
    golden_id = getattr(golden, "id", None)
    if golden_id:
        return str(golden_id)
    return f"golden-{id(golden)}"


def set_conversation_key(key: str) -> contextvars.Token[str | None]:
    """Bind the active conversation key for the current async task."""
    return _conversation_key.set(key)


def get_conversation_key() -> str | None:
    """Read the active conversation key for the current async task."""
    return _conversation_key.get()


def reset_conversation_key(token: contextvars.Token[str | None]) -> None:
    """Restore the previous conversation key after a simulation completes."""
    _conversation_key.reset(token)


def set_request_context(context: dict[str, Any] | None) -> contextvars.Token[dict[str, Any] | None]:
    """Bind per-golden request-body ``context`` fields for the current async task."""
    return _request_context.set(context)


def get_request_context() -> dict[str, Any] | None:
    """Read per-golden request-body ``context`` fields for the current async task."""
    return _request_context.get()


def reset_request_context(token: contextvars.Token[dict[str, Any] | None]) -> None:
    """Restore the previous request context after a simulation completes."""
    _request_context.reset(token)
