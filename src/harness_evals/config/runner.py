"""Config runner — build live objects from specs and execute an eval."""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from typing import Any

from harness_evals._async_compat import _run_async
from harness_evals.baseline.compare import compare_to_baseline
from harness_evals.baseline.json_store import JsonBaselineStore
from harness_evals.baseline.store import BaselineStore
from harness_evals.catalog import _build_registry
from harness_evals.config.schema import (
    BaselineSpec,
    ConversationSpec,
    EvalConfig,
    MetricSpec,
    ModelSpec,
    SinkSpec,
    TargetSpec,
)
from harness_evals.core.metric import BaseMetric
from harness_evals.core.runner import evaluate_dataset
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.env import resolve_env_in_value
from harness_evals.errors import BaselineRegressionError, HarnessEvalsError, UnknownMetricError
from harness_evals.llm.base import BaseLLM
from harness_evals.logging_config import dataset_sample_summary
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

logger = logging.getLogger(__name__)


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


def _resolve_env_in_params(params: dict) -> dict:
    """Resolve ``${VAR}`` references in params.

    .. note::
        YAML requires ``${VAR}`` references to be quoted (``"${VAR}"``).
        An unquoted ``${VAR}`` is parsed as ``null`` by the YAML spec, which
        means this function will never see the interpolation placeholder.
    """

    return {key: _resolve_env_param(key, value) for key, value in params.items()}


def _resolve_env_param(key: str, value: Any) -> Any:
    if value is None:
        import warnings

        warnings.warn(
            f"Parameter {key!r} resolved to None. If you intended environment variable "
            f'interpolation, ensure the value is quoted in YAML: {key}: "${{{key.upper()}}}"',
            UserWarning,
            stacklevel=2,
        )
    return resolve_env_in_value(value)


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
    if spec.type == "streaming_http":
        return _build_streaming_http_target(spec.params)
    if spec.type == "conversational_streaming_http":
        return _build_conversational_streaming_http_target(spec.params)
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
        raise HarnessEvalsError(
            f"PromptTarget 'prompt' must be a ref string or PromptTemplate, got {type(raw_prompt).__name__}"
        )

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

    system_prompt = params.get("system_prompt") or params.get("system_message")
    return PromptTarget(prompt=prompt, model=model, system_prompt=str(system_prompt) if system_prompt else None)


def _build_http_target(params: dict[str, Any]) -> BaseTarget:
    from harness_evals.targets.http import HttpTarget

    kwargs = _resolve_env_in_params(params)
    raw_auth = kwargs.pop("auth", None)
    if raw_auth is not None:
        kwargs["auth"] = _build_auth(raw_auth)
    return HttpTarget(**kwargs)


def _build_streaming_http_target(params: dict[str, Any]) -> BaseTarget:
    from harness_evals.targets.streaming_http import StreamingHttpTarget

    kwargs = _resolve_env_in_params(params)
    raw_auth = kwargs.pop("auth", None)
    if raw_auth is not None:
        kwargs["auth"] = _build_auth(raw_auth)
    return StreamingHttpTarget(**kwargs)


def _build_conversational_streaming_http_target(params: dict[str, Any]) -> BaseTarget:
    from harness_evals.targets.conversational_streaming_http import ConversationalStreamingHttpTarget

    kwargs = _resolve_env_in_params(params)
    raw_auth = kwargs.pop("auth", None)
    if raw_auth is not None:
        kwargs["auth"] = _build_auth(raw_auth)
    return ConversationalStreamingHttpTarget(**kwargs)


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


def build_metric(
    spec: MetricSpec,
    llm: BaseLLM | None = None,
    *,
    registry: dict[str, type] | None = None,
) -> BaseMetric:
    """Construct a ``BaseMetric`` from a ``MetricSpec``.

    Pass a pre-built ``registry`` to avoid rebuilding it for every metric.
    """

    if registry is None:
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


def run_config(cfg: EvalConfig, *, baseline: BaselineSpec | None = ...) -> list[list[Score]]:  # type: ignore[assignment]
    """Synchronous entry point — load plugins, build objects, run eval.

    Returns the per-golden score lists on success. If a baseline spec is
    active and scores regress beyond tolerance, raises
    :class:`~harness_evals.errors.BaselineRegressionError` (a subclass of
    ``HarnessEvalsError``). Callers that need to distinguish "eval ran but
    regressed" from other failures should catch ``BaselineRegressionError``
    specifically.

    If ``cfg.baseline`` is set, gating happens automatically inside this call.
    Pass ``baseline=None`` to suppress it and handle gating yourself.

    Args:
        baseline: Override baseline spec. Pass ``None`` to disable baseline
            comparison regardless of ``cfg.baseline``. When omitted (default
            sentinel), uses ``cfg.baseline``.
    """

    load_plugins(cfg.plugins)
    effective_baseline = cfg.baseline if baseline is ... else baseline
    return _run_async(_run_config_async(cfg, baseline=effective_baseline))


async def _run_config_async(cfg: EvalConfig, *, baseline: BaselineSpec | None = None) -> list[list[Score]]:
    """Wire specs to live objects and execute via ``evaluate_dataset()``."""

    if cfg.conversation is not None:
        if cfg.dataset.source != "local":
            raise HarnessEvalsError(
                "Conversation evals only support local dataset sources; "
                f"got {cfg.dataset.source!r}. Use a local file path or "
                "dataset: {source: local, id: path/to/goldens.jsonl}."
            )
        from harness_evals.conversation import load_conversation_dataset

        goldens = load_conversation_dataset(cfg.dataset.id)
        _apply_conversation_defaults(goldens, cfg.conversation)
    else:
        source_cls = lookup_dataset_source(cfg.dataset.source)
        source = source_cls.from_ref(cfg.dataset)
        async with source:
            goldens = await source.fetch(cfg.dataset)
    logger.debug(
        "Loaded dataset %s://%s: %d goldens (samples: %s)",
        cfg.dataset.source,
        cfg.dataset.id,
        len(goldens),
        dataset_sample_summary(goldens),
    )

    target = await build_target(cfg.target)

    judge_llm = _resolve_judge_llm(cfg, target)
    metric_registry = _build_registry()
    metric_registry.update(registered_metrics())
    metrics = [build_metric(m, llm=judge_llm, registry=metric_registry) for m in cfg.metrics]
    sinks = [build_sink(s) for s in cfg.sinks]

    async with target:
        if cfg.conversation is not None:
            agent_fn = getattr(target, "agenerate", None)
            if agent_fn is None:
                raise HarnessEvalsError(
                    f"Target {cfg.target.type!r} does not support conversation evals; "
                    "use a target with an agenerate(messages, system_event=None) method."
                )
            simulator_llm = _resolve_simulator_llm(cfg.conversation, judge_llm)
            human_input_simulator = _build_human_input_simulator(cfg.conversation, simulator_llm)

            scores = await evaluate_dataset(
                goldens,
                agent_fn,
                metrics=metrics,
                sinks=sinks,
                simulator_llm=simulator_llm,
                human_input_simulator=human_input_simulator,
            )
        else:
            scores = await evaluate_dataset(goldens, target.ainvoke, metrics=metrics, sinks=sinks)

    if baseline:
        gate_against_baseline(scores, baseline)

    return scores


def _apply_conversation_defaults(goldens: list, spec: ConversationSpec) -> None:
    for golden in goldens:
        if spec.mode is not None:
            from harness_evals.conversation.golden import ConversationMode

            golden.mode = ConversationMode(spec.mode)
        if spec.max_turns is not None:
            golden.max_turns = spec.max_turns
        if spec.max_elicitation_rounds is not None:
            golden.max_elicitation_rounds = spec.max_elicitation_rounds


def _resolve_simulator_llm(spec: ConversationSpec, judge_llm: BaseLLM | None) -> BaseLLM | None:
    if spec.simulator_llm is not None:
        return build_llm(spec.simulator_llm)
    return judge_llm


def _build_human_input_simulator(spec: ConversationSpec, simulator_llm: BaseLLM | None):
    from harness_evals.conversation.human_input import HumanInputSimulator
    from harness_evals.plugins import elicitation_adapter

    adapter = None
    if spec.elicitation_adapter is not None:
        adapter = elicitation_adapter(spec.elicitation_adapter)()
        if hasattr(adapter, "llm"):
            adapter.llm = simulator_llm
    return HumanInputSimulator(simulator_llm, adapter=adapter)


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
