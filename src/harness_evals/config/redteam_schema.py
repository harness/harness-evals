"""Red-team config dataclasses and YAML loader.

Kept separate from :class:`~harness_evals.config.schema.EvalConfig` because the
two modes require different things. An eval needs a dataset and a metric list;
a red-team run needs neither, because each attack plugin supplies its own
attacks and its own graders. Modelling them as one dataclass would mean making
``dataset`` and ``metrics`` optional for everyone and validating them at
runtime instead of at parse time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from harness_evals.config.schema import ModelSpec, SinkSpec, TargetSpec, _parse_model, _parse_sink, _parse_target
from harness_evals.errors import HarnessEvalsError
from harness_evals.redteam.plugin import Capability, Precondition


@dataclass
class PluginSpec:
    """Parsed representation of one entry in the ``redteam.plugins`` list."""

    id: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RedTeamConfig:
    """Full red-team configuration — the parsed output of a YAML redteam file."""

    name: str
    target: TargetSpec
    packs: list[str] = field(default_factory=list)
    plugins: list[PluginSpec] = field(default_factory=list)
    capabilities: list[Capability] = field(default_factory=list)
    preconditions: list[Precondition] = field(default_factory=list)
    judge_llm: ModelSpec | None = None
    sinks: list[SinkSpec] = field(default_factory=list)
    plugin_modules: list[str] = field(default_factory=list)
    concurrency: int | None = None


_KNOWN_TOP_LEVEL_KEYS = frozenset(
    {
        "name",
        "mode",
        "target",
        "redteam",
        "judge_llm",
        "sinks",
        "plugin_modules",
        "concurrency",
    }
)

_KNOWN_REDTEAM_KEYS = frozenset({"packs", "plugins", "capabilities", "preconditions"})


def config_mode(text: str) -> str:
    """Return the declared ``mode:`` of a config document, defaulting to ``eval``."""
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise HarnessEvalsError("Config must be a YAML mapping")
    mode = raw.get("mode", "eval")
    if mode not in {"eval", "redteam"}:
        raise HarnessEvalsError(f"Unknown mode {mode!r}. Valid modes: eval, redteam")
    return str(mode)


def load_redteam_config(path: str) -> RedTeamConfig:
    """Read a YAML red-team config file and return a validated ``RedTeamConfig``."""
    cfg_path = Path(path).resolve()
    return loads_redteam_config(cfg_path.read_text(encoding="utf-8"))


def loads_redteam_config(text: str) -> RedTeamConfig:
    """Parse a YAML string into a validated ``RedTeamConfig``."""
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise HarnessEvalsError("Red-team config must be a YAML mapping")

    unknown = set(raw) - _KNOWN_TOP_LEVEL_KEYS
    if unknown:
        raise HarnessEvalsError(f"Unknown top-level key(s) in red-team config: {', '.join(sorted(unknown))}")

    mode = raw.get("mode", "redteam")
    if mode != "redteam":
        raise HarnessEvalsError(f"Expected mode 'redteam', got {mode!r}. Use 'harness-evals run' for eval configs.")

    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise HarnessEvalsError("Red-team config requires a non-empty 'name' string")

    if "target" not in raw:
        raise HarnessEvalsError("Red-team config requires a 'target' field")
    target = _parse_target(raw["target"])

    redteam = raw.get("redteam")
    if not isinstance(redteam, dict):
        raise HarnessEvalsError("Red-team config requires a 'redteam' mapping")

    unknown_rt = set(redteam) - _KNOWN_REDTEAM_KEYS
    if unknown_rt:
        raise HarnessEvalsError(f"Unknown key(s) in 'redteam' block: {', '.join(sorted(unknown_rt))}")

    packs = _parse_str_list(redteam.get("packs", []), "redteam.packs")
    plugins = [_parse_plugin(p) for p in redteam.get("plugins", [])]
    if not packs and not plugins:
        raise HarnessEvalsError("Red-team config requires at least one entry in 'redteam.packs' or 'redteam.plugins'")

    capabilities = [
        _parse_enum(c, Capability, "redteam.capabilities")
        for c in _parse_str_list(redteam.get("capabilities", []), "redteam.capabilities")
    ]
    preconditions = [
        _parse_enum(p, Precondition, "redteam.preconditions")
        for p in _parse_str_list(redteam.get("preconditions", []), "redteam.preconditions")
    ]

    judge_llm = _parse_model(raw["judge_llm"]) if raw.get("judge_llm") else None

    raw_sinks = raw.get("sinks", [])
    sinks = [_parse_sink(s) for s in (raw_sinks if isinstance(raw_sinks, list) else [raw_sinks])]

    plugin_modules = _parse_str_list(raw.get("plugin_modules", []), "plugin_modules")

    concurrency = raw.get("concurrency")
    if concurrency is not None and (
        isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1
    ):
        raise HarnessEvalsError("'concurrency' must be an integer >= 1")

    return RedTeamConfig(
        name=name,
        target=target,
        packs=packs,
        plugins=plugins,
        capabilities=capabilities,
        preconditions=preconditions,
        judge_llm=judge_llm,
        sinks=sinks,
        plugin_modules=plugin_modules,
        concurrency=concurrency,
    )


def _parse_plugin(raw: str | dict) -> PluginSpec:
    from harness_evals.redteam.packs import normalize_plugin_id

    if isinstance(raw, str):
        return PluginSpec(id=normalize_plugin_id(raw))
    if not isinstance(raw, dict):
        raise HarnessEvalsError(f"Each plugin must be a string or dict, got {type(raw).__name__}")
    if "id" not in raw:
        raise HarnessEvalsError("Plugin dict requires an 'id' key")
    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise HarnessEvalsError(f"Plugin params must be a dict, got {type(params).__name__}")
    return PluginSpec(id=normalize_plugin_id(str(raw["id"])), params=params)


def _parse_str_list(raw: Any, field_name: str) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise HarnessEvalsError(f"'{field_name}' must be a list of strings")
    return list(raw)


def _parse_enum(value: str, enum_cls: type, field_name: str):
    try:
        return enum_cls(value)
    except ValueError:
        valid = ", ".join(sorted(member.value for member in enum_cls))
        raise HarnessEvalsError(f"Unknown {field_name} value {value!r}. Valid values: {valid}") from None
