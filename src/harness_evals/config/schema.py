"""Eval config dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from harness_evals.errors import HarnessEvalsError
from harness_evals.refs import ResourceRef, resolve


@dataclass
class MetricSpec:
    """Parsed representation of a single metric entry in a YAML config."""

    kind: str
    threshold: float | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SinkSpec:
    """Parsed representation of a single sink entry in a YAML config."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    """LLM provider + model name, resolved to a BaseLLM by the runner."""

    provider: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetSpec:
    """Parsed representation of the ``target:`` block in a YAML config."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineSpec:
    """Parsed representation of the ``baseline:`` block in a YAML config."""

    store: str = "json"
    path: str = ".harness-evals/baselines"
    tolerance: float = 0.05
    run_id: str | None = None


@dataclass
class EvalConfig:
    """Full eval configuration — the parsed output of a YAML eval file."""

    name: str
    dataset: ResourceRef
    target: TargetSpec
    metrics: list[MetricSpec]
    judge_llm: ModelSpec | None = None
    sinks: list[SinkSpec] = field(default_factory=lambda: [SinkSpec("stdout")])
    baseline: BaselineSpec | None = None
    plugins: list[str] = field(default_factory=list)


_KNOWN_TOP_LEVEL_KEYS = frozenset({
    "name", "dataset", "target", "metrics", "judge_llm",
    "sinks", "baseline", "plugins",
})


def load_config(path: str) -> EvalConfig:
    """Read a YAML eval config file and return a validated ``EvalConfig``.

    Relative paths inside the config (e.g. ``dataset: ./goldens.jsonl``)
    are resolved against the config file's parent directory.
    """

    cfg_path = Path(path).resolve()
    text = cfg_path.read_text(encoding="utf-8")
    return loads_config(text, base_dir=cfg_path.parent)


def loads_config(text: str, *, base_dir: Path | None = None) -> EvalConfig:
    """Parse a YAML string into a validated ``EvalConfig``.

    If *base_dir* is provided, relative dataset paths are resolved against it.
    """

    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise HarnessEvalsError("Eval config must be a YAML mapping")

    unknown = set(raw) - _KNOWN_TOP_LEVEL_KEYS
    if unknown:
        raise HarnessEvalsError(f"Unknown top-level key(s) in eval config: {', '.join(sorted(unknown))}")

    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise HarnessEvalsError("Eval config requires a non-empty 'name' string")

    if "dataset" not in raw:
        raise HarnessEvalsError("Eval config requires a 'dataset' field")
    raw_dataset = raw["dataset"]
    if base_dir is not None and isinstance(raw_dataset, str):
        dataset_path = Path(raw_dataset)
        if not dataset_path.is_absolute() and "://" not in raw_dataset:
            raw_dataset = str(base_dir / dataset_path)
    dataset = resolve(raw_dataset)

    if "target" not in raw:
        raise HarnessEvalsError("Eval config requires a 'target' field")
    target = _parse_target(raw["target"])

    raw_metrics = raw.get("metrics")
    if not raw_metrics or not isinstance(raw_metrics, list):
        raise HarnessEvalsError("Eval config requires a non-empty 'metrics' list")
    metrics = [_parse_metric(m) for m in raw_metrics]

    judge_llm = _parse_model(raw["judge_llm"]) if raw.get("judge_llm") else None

    raw_sinks = raw.get("sinks", ["stdout"])
    sinks = [_parse_sink(s) for s in (raw_sinks if isinstance(raw_sinks, list) else [raw_sinks])]

    baseline = _parse_baseline(raw["baseline"]) if raw.get("baseline") else None

    plugins = raw.get("plugins", [])
    if not isinstance(plugins, list):
        raise HarnessEvalsError("'plugins' must be a list of module names")

    return EvalConfig(
        name=name,
        dataset=dataset,
        target=target,
        metrics=metrics,
        judge_llm=judge_llm,
        sinks=sinks,
        baseline=baseline,
        plugins=plugins,
    )


def _parse_metric(raw: str | dict) -> MetricSpec:
    if isinstance(raw, str):
        return MetricSpec(kind=raw)
    if not isinstance(raw, dict):
        raise HarnessEvalsError(f"Each metric must be a string or dict, got {type(raw).__name__}")
    if "kind" not in raw:
        raise HarnessEvalsError("Metric dict requires a 'kind' key")
    kind = raw["kind"]
    threshold = raw.get("threshold")
    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise HarnessEvalsError(f"Metric params must be a dict, got {type(params).__name__}")
    return MetricSpec(kind=kind, threshold=float(threshold) if threshold is not None else None, params=params)


def _parse_sink(raw: str | dict) -> SinkSpec:
    if isinstance(raw, str):
        return SinkSpec(type=raw)
    if not isinstance(raw, dict):
        raise HarnessEvalsError(f"Each sink must be a string or dict, got {type(raw).__name__}")
    if "type" not in raw:
        raise HarnessEvalsError("Sink dict requires a 'type' key")
    sink_type = raw["type"]
    params = {k: v for k, v in raw.items() if k != "type"}
    return SinkSpec(type=sink_type, params=params)


def _parse_target(raw: dict) -> TargetSpec:
    if not isinstance(raw, dict):
        raise HarnessEvalsError(f"'target' must be a dict, got {type(raw).__name__}")
    if "type" not in raw:
        raise HarnessEvalsError("Target dict requires a 'type' key")
    target_type = raw["type"]
    params = {k: v for k, v in raw.items() if k != "type"}
    return TargetSpec(type=target_type, params=params)


def _parse_model(raw: dict) -> ModelSpec:
    if not isinstance(raw, dict):
        raise HarnessEvalsError(f"Model spec must be a dict, got {type(raw).__name__}")
    if "provider" not in raw or "name" not in raw:
        raise HarnessEvalsError("Model spec requires 'provider' and 'name' keys")
    provider = raw["provider"]
    name = raw["name"]
    params = {k: v for k, v in raw.items() if k not in {"provider", "name"}}
    return ModelSpec(provider=provider, name=name, params=params)


def _parse_baseline(raw: dict) -> BaselineSpec:
    if not isinstance(raw, dict):
        raise HarnessEvalsError(f"'baseline' must be a dict, got {type(raw).__name__}")
    return BaselineSpec(
        store=raw.get("store", "json"),
        path=raw.get("path", ".harness-evals/baselines"),
        tolerance=float(raw.get("tolerance", 0.05)),
        run_id=raw.get("run_id"),
    )
