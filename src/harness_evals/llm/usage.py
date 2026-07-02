"""Token-usage capture for LLM calls.

Judge metrics call an :class:`~harness_evals.llm.base.BaseLLM` under the hood,
and the token counts each provider returns are normally discarded. To surface
them (e.g. for tracing or cost accounting) without threading return values
through every metric, providers report usage to a ``ContextVar``-backed
collector.

Because the collector is a context variable, it is isolated per asyncio task:
concurrent judge calls each see only their own usage, so there is no cross-talk
even when a single ``BaseLLM`` instance is shared across many concurrent
evaluations.

Usage::

    from harness_evals.llm.usage import collect_token_usage

    with collect_token_usage() as usage:
        await llm.generate_json(prompt, schema)
    print(usage.input_tokens, usage.output_tokens)

Outside an active ``collect_token_usage`` block, ``record_token_usage`` is a
no-op, so providers can always call it unconditionally.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class TokenUsage:
    """Token counts accumulated within a ``collect_token_usage`` block.

    Values sum across every LLM call made inside the block. A count stays
    ``None`` if no call reported it, distinguishing "unknown" from "zero".
    """

    input_tokens: int | None = None
    output_tokens: int | None = None

    def add(self, *, input_tokens: int | None = None, output_tokens: int | None = None) -> None:
        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens


_usage_var: ContextVar[TokenUsage | None] = ContextVar("harness_evals_token_usage", default=None)


@contextmanager
def collect_token_usage() -> Iterator[TokenUsage]:
    """Capture token usage from LLM calls made within this block.

    Returns a fresh :class:`TokenUsage` that providers populate via
    :func:`record_token_usage`. Safe to nest and safe across concurrent asyncio
    tasks — each active block sees only usage from calls in its own task.
    """
    usage = TokenUsage()
    token = _usage_var.set(usage)
    try:
        yield usage
    finally:
        _usage_var.reset(token)


def record_token_usage(*, input_tokens: int | None = None, output_tokens: int | None = None) -> None:
    """Report token counts to the active collector, if any. No-op otherwise."""
    usage = _usage_var.get()
    if usage is not None:
        usage.add(input_tokens=input_tokens, output_tokens=output_tokens)
