"""Plugin registries and lazy entry-point discovery for harness-evals adapters."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from importlib.metadata import EntryPoint, entry_points
from typing import TypeVar

from harness_evals.errors import HarnessEvalsError, MissingAdapterError

T = TypeVar("T", bound=type)

DATASET_SOURCES = "dataset_sources"
PROMPT_SOURCES = "prompt_sources"
EVAL_CASE_SOURCES = "eval_case_sources"
EVAL_CONFIG_SOURCES = "eval_config_sources"
TARGETS = "targets"
METRICS = "metrics"
BASELINE_STORES = "baseline_stores"
SINKS = "sinks"

FAMILIES: tuple[str, ...] = (
    DATASET_SOURCES,
    PROMPT_SOURCES,
    EVAL_CASE_SOURCES,
    EVAL_CONFIG_SOURCES,
    TARGETS,
    METRICS,
    BASELINE_STORES,
    SINKS,
)

_DATASET_SOURCES: dict[str, type] = {}
_PROMPT_SOURCES: dict[str, type] = {}
_EVAL_CASE_SOURCES: dict[str, type] = {}
_EVAL_CONFIG_SOURCES: dict[str, type] = {}
_TARGETS: dict[str, type] = {}
_METRICS: dict[str, type] = {}
_BASELINE_STORES: dict[str, type] = {}
_SINKS: dict[str, type] = {}

_REGISTRIES: dict[str, dict[str, type]] = {
    DATASET_SOURCES: _DATASET_SOURCES,
    PROMPT_SOURCES: _PROMPT_SOURCES,
    EVAL_CASE_SOURCES: _EVAL_CASE_SOURCES,
    EVAL_CONFIG_SOURCES: _EVAL_CONFIG_SOURCES,
    TARGETS: _TARGETS,
    METRICS: _METRICS,
    BASELINE_STORES: _BASELINE_STORES,
    SINKS: _SINKS,
}

_ENTRY_POINTS: dict[str, dict[str, EntryPoint]] = {family: {} for family in FAMILIES}
_ENTRY_POINTS_DISCOVERED = False

INSTALL_HINTS: dict[str, str] = {
    "harness": "harness-evals[harness]",
    "langfuse": "harness-evals[langfuse]",
    "otel": "harness-evals[otlp]",
    "otlp": "harness-evals[otlp]",
}


def register_dataset_source(name: str) -> Callable[[T], T]:
    """Register a dataset source class. Later registrations with the same name win."""

    return _register(DATASET_SOURCES, name)


def register_prompt_source(name: str) -> Callable[[T], T]:
    """Register a prompt source class. Later registrations with the same name win."""

    return _register(PROMPT_SOURCES, name)


def register_eval_case_source(name: str) -> Callable[[T], T]:
    """Register an eval-case source class. Later registrations with the same name win."""

    return _register(EVAL_CASE_SOURCES, name)


def register_eval_config_source(name: str) -> Callable[[T], T]:
    """Register an eval-config source class. Later registrations with the same name win."""

    return _register(EVAL_CONFIG_SOURCES, name)


def register_target(name: str) -> Callable[[T], T]:
    """Register a target class. Later registrations with the same name win."""

    return _register(TARGETS, name)


def register_metric(kind: str) -> Callable[[T], T]:
    """Register a metric class. Later registrations with the same kind win."""

    return _register(METRICS, kind)


def register_baseline_store(name: str) -> Callable[[T], T]:
    """Register a baseline store class. Later registrations with the same name win."""

    return _register(BASELINE_STORES, name)


def register_sink(name: str) -> Callable[[T], T]:
    """Register a sink class. Later registrations with the same name win."""

    return _register(SINKS, name)


def dataset_source(name: str) -> type:
    """Return the registered dataset source class for ``name``."""

    return _lookup(DATASET_SOURCES, name)


def prompt_source(name: str) -> type:
    """Return the registered prompt source class for ``name``."""

    return _lookup(PROMPT_SOURCES, name)


def eval_case_source(name: str) -> type:
    """Return the registered eval-case source class for ``name``."""

    return _lookup(EVAL_CASE_SOURCES, name)


def eval_config_source(name: str) -> type:
    """Return the registered eval-config source class for ``name``."""

    return _lookup(EVAL_CONFIG_SOURCES, name)


def target(name: str) -> type:
    """Return the registered target class for ``name``."""

    return _lookup(TARGETS, name)


def metric(kind: str) -> type:
    """Return the registered metric class for ``kind``."""

    return _lookup(METRICS, kind)


def registered_metrics() -> dict[str, type]:
    """Return all registered metric classes, including lazy entry points."""

    return _registered(METRICS)


def baseline_store(name: str) -> type:
    """Return the registered baseline store class for ``name``."""

    return _lookup(BASELINE_STORES, name)


def sink(name: str) -> type:
    """Return the registered sink class for ``name``."""

    return _lookup(SINKS, name)


def load_plugins(modules: list[str]) -> None:
    """Import explicit plugin modules so their register decorators run."""

    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as err:
            raise HarnessEvalsError(f"Failed to load plugin module {module!r}: {err}") from err


def _register(family: str, name: str) -> Callable[[T], T]:
    if family not in _REGISTRIES:
        raise ValueError(f"Unknown plugin family {family!r}")
    if not name:
        raise ValueError("Plugin registration name must be non-empty")

    def decorator(cls: T) -> T:
        if name in _REGISTRIES[family]:
            warnings.warn(
                f"Plugin {name!r} in family {family!r} is being overwritten by {cls!r}",
                UserWarning,
                stacklevel=2,
            )
        _REGISTRIES[family][name] = cls
        return cls

    return decorator


def _lookup(family: str, name: str) -> type:
    if family not in _REGISTRIES:
        raise ValueError(f"Unknown plugin family {family!r}")

    registry = _REGISTRIES[family]
    if name in registry:
        return registry[name]

    _discover_entry_points()
    entry_point = _ENTRY_POINTS[family].get(name)
    if entry_point is not None:
        adapter_cls = entry_point.load()
        registry[name] = adapter_cls
        return adapter_cls

    raise MissingAdapterError(name, family, _install_hint(name))


def _registered(family: str) -> dict[str, type]:
    if family not in _REGISTRIES:
        raise ValueError(f"Unknown plugin family {family!r}")

    registry = _REGISTRIES[family]
    _discover_entry_points()
    for name, entry_point in _ENTRY_POINTS[family].items():
        if name not in registry:
            registry[name] = entry_point.load()
    return dict(registry)


def _discover_entry_points() -> None:
    global _ENTRY_POINTS_DISCOVERED

    if _ENTRY_POINTS_DISCOVERED:
        return

    for family in FAMILIES:
        group = f"harness_evals.{family}"
        for entry_point in entry_points(group=group):
            _ENTRY_POINTS[family][entry_point.name] = entry_point

    _ENTRY_POINTS_DISCOVERED = True


def _install_hint(name: str) -> str:
    return INSTALL_HINTS.get(name, f"harness-evals[{name}]")


# ------------------------------------------------------------------
# Test helpers — snapshot / restore for isolation
# ------------------------------------------------------------------


def _snapshot() -> dict:
    """Capture the current state of all plugin registries.

    Returns an opaque dict suitable for passing to :func:`_restore`.
    """
    return {
        "registries": {family: registry.copy() for family, registry in _REGISTRIES.items()},
        "entry_points": {family: discovered.copy() for family, discovered in _ENTRY_POINTS.items()},
        "discovered": _ENTRY_POINTS_DISCOVERED,
    }


def _restore(snapshot: dict) -> None:
    """Restore plugin registries from a previous :func:`_snapshot`."""
    global _ENTRY_POINTS_DISCOVERED

    for family, registry in _REGISTRIES.items():
        registry.clear()
        registry.update(snapshot["registries"][family])
    for family, discovered in _ENTRY_POINTS.items():
        discovered.clear()
        discovered.update(snapshot["entry_points"][family])
    _ENTRY_POINTS_DISCOVERED = snapshot["discovered"]
