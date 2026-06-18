"""Config runner — build live objects from specs and execute an eval."""

from __future__ import annotations

import inspect
import os
import re
from collections import defaultdict
from typing import Any

from harness_evals._async_compat import _run_async
from harness_evals.baseline.compare import compare_to_baseline
from harness_evals.baseline.json_store import JsonBaselineStore
from harness_evals.baseline.store import BaselineStore
from harness_evals.catalog import _build_registry
from harness_evals.config.schema import BaselineSpec, EvalConfig, MetricSpec, ModelSpec, SinkSpec, TargetSpec
from harness_evals.core.metric import BaseMetric
from harness_evals.core.runner import evaluate_dataset
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.errors import BaselineRegressionError, HarnessEvalsError, UnknownMetricError
from harness_evals.llm.base import BaseLLM
from harness_evals.plugins import (
    baseline_store as lookup_baseline_store,
)
from harness_evals.plugins import (
    dataset_source as lookup_dataset_source,
)
from harness_evals.plugins import (
    load_plugins,
    registered_metrics,
)
from harness_evals.plugins import (
    prompt_source as lookup_prompt_source,
)
from harness_evals.plugins import (
    sink as lookup_sink,
)
from harness_evals.plugins import (
    target as lookup_target,
)
from harness_evals.targets.base import BaseTarget

_LLM_PROVIDERS: dict[str, str] = {
    "openai": "harness_evals.llm.openai.OpenAILLM",
    "anthropic": "harness_evals.llm.anthropic.AnthropicLLM",
    "harness": "harness_evals.llm.harness_ai.HarnessAILLM",
}


def build_llm(spec: ModelSpec) -> BaseLLM:
    """Construct a ``BaseLLM`` from a ``ModelSpec``.

    String values in ``spec.params`` containing ``${VAR}`` references
    are resolved from environment variables, so different LLM instances
    (target vs judge) can use separate API keys::

        target.model:  {provider: openai, name: gpt-4o, api_key: "${TARGET_KEY}"}
        judge_llm:     {provider: openai, name: gpt-4o, api_key: "${JUDGE_KEY}"}
    """

    dotted = _LLM_PROVIDERS.get(spec.provider)
    if dotted is None:
        valid = ", ".join(sorted(_LLM_PROVIDERS))
        raise HarnessEvalsError(f"Unknown LLM provider {spec.provider!r}. Valid providers: {valid}")

    module_path, class_name = dotted.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    resolved_params = _resolve_env_in_params(spec.params)
    return cls(model=spec.name, **resolved_params)


_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env_in_params(params: dict) -> dict:
    """Resolve ``${VAR}`` references in string values of a params dict."""

    resolved = {}
    for key, value in params.items():
        if isinstance(value, str) and "${" in value:
            resolved[key] = _resolve_env_value(value)
        else:
            resolved[key] = value
    return resolved


def _resolve_env_value(value: str) -> str:
    """Replace ``${VAR}`` references with environment variable values."""

    def _replace(match: re.Match[str]) -> str:
        var = match.group(1)
        val = os.environ.get(var)
        if val is None:
            raise ValueError(f"Environment variable ${{{var}}} is not set")
        return val

    return _ENV_VAR_RE.sub(_replace, value)


async def build_target(spec: TargetSpec) -> BaseTarget:
    """Construct a ``BaseTarget`` from a ``TargetSpec``.

    For ``prompt`` type: resolves the prompt ref, builds the LLM,
    returns ``PromptTarget(prompt, model)``.

    For ``http`` type: constructs ``HttpTarget`` with auth.

    Falls back to the plugin registry for unknown types.
    """

    if spec.type == "prompt":
        return await _build_prompt_target(spec.params)
    if spec.type == "http":
        return _build_http_target(spec.params)
    cls = lookup_target(spec.type)
    return cls(**spec.params)


async def _build_prompt_target(params: dict[str, Any]) -> BaseTarget:
    from harness_evals.prompts.template import PromptTemplate
    from harness_evals.refs import resolve
    from harness_evals.targets.prompt import PromptTarget

    raw_prompt = params.get("prompt")
    if raw_prompt is None:
        raise HarnessEvalsError("PromptTarget requires a 'prompt' field")
    raw_model = params.get("model")
    if raw_model is None:
        raise HarnessEvalsError("PromptTarget requires a 'model' field")

    if isinstance(raw_prompt, str):
        ref = resolve(raw_prompt)
        source_cls = lookup_prompt_source(ref.source)
        source = source_cls()
        prompt = await source.fetch(ref)
    elif isinstance(raw_prompt, PromptTemplate):
        prompt = raw_prompt
    else:
        raise HarnessEvalsError(f"PromptTarget 'prompt' must be a ref string or PromptTemplate, got {type(raw_prompt).__name__}")

    if isinstance(raw_model, dict):
        model_spec = ModelSpec(
            provider=raw_model["provider"],
            name=raw_model["name"],
            params={k: v for k, v in raw_model.items() if k not in {"provider", "name"}},
        )
        model = build_llm(model_spec)
    elif isinstance(raw_model, BaseLLM):
        model = raw_model
    else:
        raise HarnessEvalsError(f"PromptTarget 'model' must be a dict or BaseLLM, got {type(raw_model).__name__}")

    return PromptTarget(prompt=prompt, model=model)


def _build_http_target(params: dict[str, Any]) -> BaseTarget:
    from harness_evals.targets.http import HttpTarget

    kwargs = dict(params)
    raw_auth = kwargs.pop("auth", None)
    if raw_auth is not None:
        kwargs["auth"] = _build_auth(raw_auth)
    return HttpTarget(**kwargs)


def _build_auth(raw: dict) -> Any:
    from harness_evals.targets.auth import ApiKeyAuth, BasicAuth, BearerAuth, NoAuth

    if not isinstance(raw, dict):
        raise HarnessEvalsError(f"'auth' must be a dict, got {type(raw).__name__}")

    auth_type = raw.get("type", "none")
    if auth_type == "none":
        return NoAuth()
    if auth_type == "bearer":
        return BearerAuth(token=raw.get("token", ""))
    if auth_type == "api_key":
        return ApiKeyAuth(
            key=raw.get("key", ""),
            header=raw.get("header", "X-API-Key"),
            location=raw.get("location", "header"),
        )
    if auth_type == "basic":
        return BasicAuth(username=raw.get("username", ""), password=raw.get("password", ""))
    raise HarnessEvalsError(f"Unknown auth type {auth_type!r}. Valid types: none, bearer, api_key, basic")


def build_metric(spec: MetricSpec, llm: BaseLLM | None = None) -> BaseMetric:
    """Construct a ``BaseMetric`` from a ``MetricSpec``."""

    registry = _build_registry()
    registry.update(registered_metrics())
    cls = registry.get(spec.kind)
    if cls is None:
        raise UnknownMetricError(spec.kind, sorted(registry))

    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}

    if "llm" in sig.parameters:
        if llm is None:
            raise HarnessEvalsError(
                f"Metric {spec.kind!r} requires an LLM. "
                "Set 'judge_llm' in your eval config or use a prompt target with a model."
            )
        kwargs["llm"] = llm

    if spec.threshold is not None:
        kwargs["threshold"] = spec.threshold

    kwargs.update(spec.params)

    return cls(**kwargs)


def build_sink(spec: SinkSpec) -> BaseSink:
    """Construct a ``BaseSink`` from a ``SinkSpec``."""

    from harness_evals.sinks.csv_sink import CsvSink
    from harness_evals.sinks.json_sink import JsonSink
    from harness_evals.sinks.junit_sink import JUnitSink
    from harness_evals.sinks.stdout import StdoutSink

    builtin_sinks: dict[str, type] = {
        "stdout": StdoutSink,
        "json": JsonSink,
        "csv": CsvSink,
        "junit": JUnitSink,
    }

    cls = builtin_sinks.get(spec.type)
    if cls is not None:
        return cls(**spec.params)

    cls = lookup_sink(spec.type)
    return cls(**spec.params)


def build_baseline_store(spec: BaselineSpec) -> BaselineStore:
    """Construct a ``BaselineStore`` from a ``BaselineSpec``.

    Plugin baseline stores registered via ``@register_baseline_store``
    must accept a ``path`` keyword argument in their constructor. This is
    the store-specific location string from the YAML config (file path,
    S3 URI, etc.).
    """

    if spec.store == "json":
        return JsonBaselineStore(baseline_dir=spec.path)
    cls = lookup_baseline_store(spec.store)
    return cls(path=spec.path)


def scores_to_baseline_dict(scores: list[list[Score]]) -> dict[str, list[Score]]:
    """Pivot ``list[list[Score]]`` into ``dict[metric_name -> list[Score]]``."""

    grouped: dict[str, list[Score]] = defaultdict(list)
    for case_scores in scores:
        for score in case_scores:
            grouped[score.name].append(score)
    return dict(grouped)


def gate_against_baseline(scores: list[list[Score]], spec: BaselineSpec) -> None:
    """Compare current scores against stored baseline.

    Raises ``BaselineRegressionError`` if any metric regresses beyond
    ``spec.tolerance``.
    """

    store = build_baseline_store(spec)
    baseline = store.load(spec.run_id)
    current = scores_to_baseline_dict(scores)
    result = compare_to_baseline(current, baseline, tolerance=spec.tolerance)
    if result.has_regressions:
        raise BaselineRegressionError(result)


def run_config(cfg: EvalConfig) -> list[list[Score]]:
    """Synchronous entry point — load plugins, build objects, run eval.

    Returns the per-golden score lists on success. If ``cfg.baseline`` is
    set and scores regress beyond tolerance, raises
    :class:`~harness_evals.errors.BaselineRegressionError` (a subclass of
    ``HarnessEvalsError``). Callers that need to distinguish "eval ran but
    regressed" from other failures should catch ``BaselineRegressionError``
    specifically.
    """

    load_plugins(cfg.plugins)
    return _run_async(_run_config_async(cfg))


async def _run_config_async(cfg: EvalConfig) -> list[list[Score]]:
    """Wire specs to live objects and execute via ``evaluate_dataset()``."""

    source_cls = lookup_dataset_source(cfg.dataset.source)
    source = source_cls.from_ref(cfg.dataset)
    async with source:
        goldens = await source.fetch(cfg.dataset)

    target = await build_target(cfg.target)

    judge_llm = _resolve_judge_llm(cfg, target)
    metrics = [build_metric(m, llm=judge_llm) for m in cfg.metrics]
    sinks = [build_sink(s) for s in cfg.sinks]

    async with target:
        scores = await evaluate_dataset(goldens, target.ainvoke, metrics=metrics, sinks=sinks)

    if cfg.baseline:
        gate_against_baseline(scores, cfg.baseline)

    return scores


def _resolve_judge_llm(cfg: EvalConfig, target: BaseTarget) -> BaseLLM | None:
    """Determine the LLM for judge metrics.

    Resolution order:
    1. Explicit ``cfg.judge_llm``
    2. If target is ``PromptTarget``, reuse its ``model``
    3. ``None`` — builder will raise if a metric actually needs one
    """

    if cfg.judge_llm is not None:
        return build_llm(cfg.judge_llm)

    from harness_evals.targets.prompt import PromptTarget

    if isinstance(target, PromptTarget):
        return target.model

    return None
