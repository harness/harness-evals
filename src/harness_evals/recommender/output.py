"""Turn a recommendation dict into a goldens dataset and an ``EvalConfig`` YAML file."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from harness_evals.config.schema import (
    BaselineSpec,
    EvalConfig,
    MetricSpec,
    ModelSpec,
    SinkSpec,
    TargetSpec,
)
from harness_evals.core.golden import Golden
from harness_evals.datasets.io import save_dataset
from harness_evals.refs import ResourceRef

logger = logging.getLogger(__name__)

_DATASET_FILENAME = "recommended.goldens.jsonl"
_CONFIG_FILENAME = "recommended.eval.yaml"

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
}


def default_model(provider: str) -> str:
    """Return the default model name for *provider*."""
    return _DEFAULT_MODELS.get(provider, _DEFAULT_MODELS["anthropic"])


def build_goldens(recommendation: dict) -> list[Golden]:
    """Convert the recommended dataset entries into ``Golden`` objects."""

    goldens: list[Golden] = []
    for case in recommendation.get("recommended_dataset", []):
        metadata = dict(case.get("metadata") or {})
        metric_tested = case.get("metric_tested")
        if metric_tested:
            metadata["metric_tested"] = metric_tested
        goldens.append(
            Golden(
                input=case.get("input", ""),
                expected=case.get("expected", ""),
                context=case.get("context"),
                expected_tools=case.get("expected_tools"),
                metadata=metadata or None,
                tags=case.get("tags") or None,
            )
        )
    return goldens


def build_eval_config(
    recommendation: dict,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    dataset_filename: str = _DATASET_FILENAME,
) -> EvalConfig:
    """Assemble an ``EvalConfig`` from a recommendation using the config specs."""

    model_name = model or default_model(provider)

    metrics = [
        MetricSpec(kind=m["name"], threshold=m.get("threshold")) for m in recommendation.get("recommended_metrics", [])
    ]

    return EvalConfig(
        name="recommended-eval",
        # Relative dataset path so the config is portable alongside the goldens file.
        dataset=ResourceRef(source="local", id=dataset_filename),
        target=TargetSpec(
            type="prompt",
            params={
                "prompt": "./your-prompt.txt",
                "model": {"provider": provider, "name": model_name},
            },
        ),
        metrics=metrics,
        judge_llm=ModelSpec(provider=provider, name=model_name),
        sinks=[
            SinkSpec(type="stdout"),
            SinkSpec(type="json", params={"path": "./results.jsonl"}),
        ],
        baseline=BaselineSpec(store="json", path=".evals/baseline.json"),
    )


def _eval_config_to_dict(cfg: EvalConfig) -> dict:
    """Serialize an ``EvalConfig`` (including judge_llm and baseline) to a YAML-ready dict."""

    d: dict = {"name": cfg.name}

    if cfg.dataset.source == "local":
        d["dataset"] = cfg.dataset.id
    else:
        d["dataset"] = f"{cfg.dataset.source}://{cfg.dataset.id}" + (
            f"@{cfg.dataset.version}" if cfg.dataset.version else ""
        )

    d["target"] = {"type": cfg.target.type, **cfg.target.params}

    if cfg.judge_llm is not None:
        d["judge_llm"] = {
            "provider": cfg.judge_llm.provider,
            "name": cfg.judge_llm.name,
            **cfg.judge_llm.params,
        }

    d["metrics"] = []
    for m in cfg.metrics:
        if not m.params and m.threshold is None:
            d["metrics"].append(m.kind)
        else:
            entry: dict = {"kind": m.kind}
            if m.threshold is not None:
                entry["threshold"] = m.threshold
            if m.params:
                entry["params"] = m.params
            d["metrics"].append(entry)

    if cfg.sinks:
        d["sinks"] = []
        for s in cfg.sinks:
            if not s.params:
                d["sinks"].append(s.type)
            else:
                d["sinks"].append({"type": s.type, **s.params})

    if cfg.baseline is not None:
        d["baseline"] = {"store": cfg.baseline.store, "path": cfg.baseline.path}

    return d


def write_outputs(
    recommendation: dict,
    output_dir: str = ".",
    provider: str = "anthropic",
    model: str | None = None,
) -> tuple[Path, Path]:
    """Write the goldens dataset and the ``EvalConfig`` YAML, returning their paths."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    goldens_path = out / _DATASET_FILENAME
    config_path = out / _CONFIG_FILENAME

    goldens = build_goldens(recommendation)
    save_dataset(goldens, str(goldens_path))

    cfg = build_eval_config(recommendation, provider=provider, model=model)
    config_yaml = yaml.dump(_eval_config_to_dict(cfg), default_flow_style=False, sort_keys=False)
    config_path.write_text(config_yaml, encoding="utf-8")

    logger.info("Wrote %d goldens to %s and config to %s", len(goldens), goldens_path, config_path)
    return config_path, goldens_path


def print_recommendation(recommendation: dict) -> None:
    print("\n=== DIMENSIONS COVERED ===\n")
    for d in recommendation.get("dimensions_covered", []):
        applies = "YES" if d.get("applies") else "no"
        print(f"  {str(d.get('dimension', '')):<15} {applies:<5}  {d.get('rationale', '')}")

    print("\n=== RECOMMENDED METRICS ===\n")
    for m in recommendation.get("recommended_metrics", []):
        print(f"  {str(m.get('name', '')):<35} threshold={m.get('threshold')}  ({m.get('dimension', '')})")
        print(f"    → {m.get('rationale', '')}")

    print("\n=== RECOMMENDED DATASET ===\n")
    for i, case in enumerate(recommendation.get("recommended_dataset", []), 1):
        print(f"  Test Case {i} (tests: {case.get('metric_tested', 'n/a')})")
        inp = str(case.get("input", ""))[:80]
        exp = str(case.get("expected", ""))[:80]
        print(f"    Input:    {inp}")
        print(f"    Expected: {exp}")

    print("\n=== RECOMMENDED ACTIONS ===\n")
    print(f"  {recommendation.get('recommended_actions', '')}")
    print()
