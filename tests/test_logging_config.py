from __future__ import annotations

import logging

import pytest

from harness_evals.errors import HarnessEvalsError
from harness_evals.logging_config import (
    ENV_VAR,
    compact_json,
    configure_logging,
    dataset_sample_summary,
    init_from_env,
    truncate_repr,
)


@pytest.fixture(autouse=True)
def reset_harness_logger(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    logger = logging.getLogger("harness_evals")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    logger.handlers.clear()
    yield
    logger.handlers.clear()
    logger.handlers.extend(original_handlers)
    logger.setLevel(original_level)
    logger.propagate = original_propagate


@pytest.mark.unit
def test_configure_logging_uses_default_warning_level() -> None:
    configure_logging()

    logger = logging.getLogger("harness_evals")
    assert logger.level == logging.WARNING
    assert len(logger.handlers) == 1
    assert logger.propagate is False


@pytest.mark.unit
def test_configure_logging_uses_env_level(monkeypatch) -> None:
    monkeypatch.setenv(ENV_VAR, "debug")

    configure_logging()

    assert logging.getLogger("harness_evals").level == logging.DEBUG


@pytest.mark.unit
def test_configure_logging_explicit_level_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv(ENV_VAR, "error")

    configure_logging("info")

    assert logging.getLogger("harness_evals").level == logging.INFO


@pytest.mark.unit
def test_configure_logging_is_idempotent() -> None:
    configure_logging("warning")
    configure_logging("debug")

    logger = logging.getLogger("harness_evals")
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1


@pytest.mark.unit
def test_configure_logging_rejects_invalid_level() -> None:
    with pytest.raises(HarnessEvalsError, match="Invalid log level"):
        configure_logging("chatty")


@pytest.mark.unit
def test_init_from_env_configures_when_env_set(monkeypatch) -> None:
    monkeypatch.setenv(ENV_VAR, "debug")

    init_from_env()

    logger = logging.getLogger("harness_evals")
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    assert any(getattr(h, "_harness_evals_handler", False) for h in logger.handlers)


@pytest.mark.unit
def test_init_from_env_noop_when_env_unset() -> None:
    logger = logging.getLogger("harness_evals")

    init_from_env()

    # Unset: leave logging untouched so host config / lastResort still applies.
    # The library does not attach a handler or silence its own warnings.
    assert logger.handlers == []
    assert not any(getattr(h, "_harness_evals_handler", False) for h in logger.handlers)


@pytest.mark.unit
def test_init_from_env_is_idempotent_when_env_set(monkeypatch) -> None:
    monkeypatch.setenv(ENV_VAR, "debug")

    init_from_env()
    init_from_env()

    logger = logging.getLogger("harness_evals")
    # configure_logging reuses its sentinel handler rather than stacking.
    assert sum(getattr(h, "_harness_evals_handler", False) for h in logger.handlers) == 1


@pytest.mark.unit
def test_init_from_env_tolerates_invalid_level(monkeypatch) -> None:
    monkeypatch.setenv(ENV_VAR, "chatty")

    # Must not raise at import time.
    init_from_env()

    logger = logging.getLogger("harness_evals")
    # Invalid value is ignored: the logger is not configured.
    assert not any(getattr(h, "_harness_evals_handler", False) for h in logger.handlers)


@pytest.mark.unit
def test_truncate_repr_bounds_output() -> None:
    assert truncate_repr("abcdef", max_len=6) == "'ab..."


@pytest.mark.unit
def test_compact_json_bounds_output() -> None:
    payload = {"resource_type": "pipeline", "identifier": "testk8spipeline"}
    assert compact_json(payload, max_len=40).endswith("...")


@pytest.mark.unit
def test_dataset_sample_summary_truncates_and_limits_samples() -> None:
    class GoldenLike:
        def __init__(self, input):
            self.input = input

    summary = dataset_sample_summary(
        [
            GoldenLike("first prompt"),
            GoldenLike("second prompt"),
            GoldenLike("third prompt"),
            GoldenLike("fourth prompt"),
        ],
        max_samples=2,
        max_len=20,
    )

    assert summary == "'first prompt', 'second prompt', ..."
