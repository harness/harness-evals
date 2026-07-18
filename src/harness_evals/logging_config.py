from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Any

from harness_evals.errors import HarnessEvalsError

DEFAULT_LOG_LEVEL = "WARNING"
ENV_VAR = "HARNESS_EVALS_LOG_LEVEL"

_LOGGER_NAME = "harness_evals"
_HANDLER_ATTR = "_harness_evals_handler"
_VALID_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configure_logging(level: str | None = None) -> None:
    """Configure harness-evals framework logging."""

    resolved_level = _resolve_level(level)
    logger = logging.getLogger(_LOGGER_NAME)
    handler = _get_or_create_handler(logger)

    logger.setLevel(resolved_level)
    handler.setLevel(resolved_level)
    logger.propagate = False


def init_from_env() -> None:
    """Auto-configure framework logging at import time.

    Library consumers that don't go through the CLI (e.g. the eval runner
    service, which calls ``run_config``/``evaluate_dataset`` directly) can turn
    on framework logs simply by setting ``HARNESS_EVALS_LOG_LEVEL`` — no code
    change on their side. When the env var is unset, the logger is left
    untouched: the host application's logging config (or Python's ``lastResort``
    for ``WARNING``+ records) still applies, so the library never silences its
    own warnings.

    Safe to call at import: an invalid env value is ignored rather than raising
    and breaking ``import harness_evals``.
    """

    if not os.environ.get(ENV_VAR):
        return
    # An invalid level in the env var must never break import.
    with contextlib.suppress(HarnessEvalsError):
        configure_logging()


def truncate_repr(value: Any, max_len: int = 80) -> str:
    """Return a bounded repr for debug logs."""

    text = repr(value)
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return "." * max_len
    return text[: max_len - 3] + "..."


def dataset_sample_summary(goldens: list[Any], *, max_samples: int = 3, max_len: int = 80) -> str:
    """Return a compact summary of representative golden inputs."""

    samples = [_golden_sample(golden, max_len=max_len) for golden in goldens[:max_samples]]
    suffix = ", ..." if len(goldens) > max_samples else ""
    return ", ".join(samples) + suffix


def _golden_sample(golden: Any, *, max_len: int) -> str:
    turns = getattr(golden, "turns", None)
    if turns is not None:
        return f"turns={len(turns)}"
    if hasattr(golden, "scenario"):
        return truncate_repr(golden.scenario, max_len=max_len)
    return truncate_repr(getattr(golden, "input", golden), max_len=max_len)


def _resolve_level(level: str | None) -> int:
    raw_level = level or os.environ.get(ENV_VAR) or DEFAULT_LOG_LEVEL
    normalized = raw_level.upper()
    try:
        return _VALID_LEVELS[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(_VALID_LEVELS))
        raise HarnessEvalsError(f"Invalid log level {raw_level!r}. Valid levels: {valid}") from exc


def _get_or_create_handler(logger: logging.Logger) -> logging.Handler:
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_ATTR, False):
            if isinstance(handler, logging.StreamHandler):
                handler.stream = sys.stderr
            return handler

    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _HANDLER_ATTR, True)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return handler
