"""Async compatibility helper for sync measure() calls.

Handles the common pattern where a sync ``measure()`` needs to call
``a_measure()`` — safely detecting whether an event loop is already
running and dispatching accordingly.
"""

from __future__ import annotations

import asyncio
import atexit
import os
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")

_THREAD_POOL = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) + 4))
atexit.register(_THREAD_POOL.shutdown, wait=False)


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from a sync context, even inside a running event loop.

    If no event loop is running, uses ``asyncio.run()``.
    If an event loop is already running (e.g. Jupyter, ``evaluate_dataset()``),
    dispatches the coroutine to a background thread with its own event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    future = _THREAD_POOL.submit(asyncio.run, coro)
    return future.result()
