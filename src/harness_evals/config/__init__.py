"""Eval configuration layer — YAML loading, building, and running."""

from harness_evals.config.runner import (
    build_baseline_store,
    build_llm,
    build_metric,
    build_sink,
    build_target,
    gate_against_baseline,
    run_config,
    scores_to_baseline_dict,
)
from harness_evals.config.schema import (
    BaselineSpec,
    EvalConfig,
    MetricSpec,
    ModelSpec,
    SinkSpec,
    TargetSpec,
    load_config,
    loads_config,
)

__all__ = [
    "EvalConfig",
    "MetricSpec",
    "SinkSpec",
    "ModelSpec",
    "TargetSpec",
    "BaselineSpec",
    "load_config",
    "loads_config",
    "run_config",
    "build_llm",
    "build_target",
    "build_metric",
    "build_sink",
    "build_baseline_store",
    "gate_against_baseline",
    "scores_to_baseline_dict",
]
