"""Shared metric factory — single code path for building BaseMetric instances.

Used by the eval engine (CLI/SDK) and by server-side scoring services.
Dispatches on metric type: llm, heuristic, code, composite. Server-side
callers must pass ``allow_code_loading=False`` so arbitrary code metrics
are refused.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import importlib.util
import inspect
import logging
import os
from pathlib import Path
from typing import Any

from harness_evals import BaseMetric, Dimension, EvalCase, Score
from harness_evals.catalog import catalog
from harness_evals.llm.openai_embedding import OpenAIEmbedding
from harness_evals.metrics import AnswerCorrectnessMetric, GEvalMetric, RubricJudgeMetric

logger = logging.getLogger("harness_evals.metrics.factory")

try:
    from harness_evals.llm.openai import OpenAILLM
except ImportError:
    OpenAILLM = None  # type: ignore[assignment,misc]

try:
    from harness_evals.llm.anthropic import AnthropicLLM
except ImportError:
    AnthropicLLM = None  # type: ignore[assignment,misc]

try:
    from harness_evals.llm.bedrock import BedrockAnthropicLLM
except ImportError:
    BedrockAnthropicLLM = None  # type: ignore[assignment,misc]

try:
    from harness_evals.llm.bedrock import BedrockOpenAILLM
except ImportError:
    BedrockOpenAILLM = None  # type: ignore[assignment,misc]


def build_metric(
    metric_type: str,
    config: dict[str, Any],
    score_name: str | None = None,
    threshold: float = 0.0,
    suite_path: Path | None = None,
    *,
    entry_config: dict[str, Any] | None = None,
    allow_code_loading: bool = True,
) -> BaseMetric:
    """Build a BaseMetric instance from type and config.

    Parameters
    ----------
    allow_code_loading:
        When ``False`` (server-side/online), refuse to load arbitrary code
        metrics. Only CLI/SDK callers should set this to ``True``.

    Raises ValueError if type is unknown or config is invalid.
    """
    effective_config = merge_metric_config(config, entry_config)

    if metric_type in ("llm", "ai_judge"):
        return _build_llm_metric(effective_config, score_name, threshold)
    elif metric_type == "embedding":
        return _build_embedding_metric(effective_config, score_name, threshold)
    elif metric_type == "heuristic":
        return _build_heuristic_metric(effective_config, score_name, threshold)
    elif metric_type == "code":
        return _build_code_metric(
            effective_config, score_name, threshold, suite_path, allow_code_loading=allow_code_loading
        )
    elif metric_type == "composite":
        return _build_composite_metric(
            effective_config, score_name, threshold, suite_path, allow_code_loading=allow_code_loading
        )
    else:
        raise ValueError(f"Unknown metric type: {metric_type!r}")


def merge_metric_config(
    base_config: dict[str, Any],
    entry_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge base metric config with per-entry overrides.

    Top-level keys are merged shallowly. The nested ``options`` dict is merged
    one level deeper so partial per-entry overrides do not discard base options.
    """
    override = entry_config or {}
    effective = {**base_config, **override}

    base_options = base_config.get("options")
    override_options = override.get("options")
    if isinstance(base_options, dict) or isinstance(override_options, dict):
        effective["options"] = {
            **(base_options if isinstance(base_options, dict) else {}),
            **(override_options if isinstance(override_options, dict) else {}),
        }

    return effective


@functools.cache
def _catalog_registry() -> tuple[dict[str, type[BaseMetric]], dict[str, type[BaseMetric]], dict[str, type[BaseMetric]]]:
    """Auto-derive heuristic, LLM, and embedding registries from the catalog."""
    heuristic: dict[str, type[BaseMetric]] = {}
    llm: dict[str, type[BaseMetric]] = {}
    embedding: dict[str, type[BaseMetric]] = {}

    for entry in catalog():
        if entry.kind == "composite":
            continue
        if entry.requires_llm:
            llm[entry.kind] = entry.metric_class
        elif entry.requires_embedding:
            embedding[entry.kind] = entry.metric_class
        else:
            heuristic[entry.kind] = entry.metric_class

    return heuristic, llm, embedding


def _heuristic_registry() -> dict[str, type[BaseMetric]]:
    return _catalog_registry()[0]


def _llm_metric_registry() -> dict[str, type[BaseMetric]]:
    return _catalog_registry()[1]


def _embedding_registry() -> dict[str, type[BaseMetric]]:
    return _catalog_registry()[2]


def _validate_constructor_options(
    metric_class: type[BaseMetric],
    options: dict[str, Any],
    reserved: set[str] | None = None,
) -> None:
    """Reject misspelled metric options before constructors can swallow them."""
    accepted: set[str] = set()
    for klass in metric_class.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for name, parameter in inspect.signature(init).parameters.items():
            if name != "self" and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                accepted.add(name)

    declared_by_metric: set[str] = set()
    for klass in metric_class.__mro__:
        if klass.__module__ == "harness_evals.core.metric":
            continue
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        declared_by_metric.update(inspect.signature(init).parameters)

    # name/dimension are BaseMetric-only kwargs unless the metric declares them itself
    factory_supplied = {"llm", "embedding", "threshold"} | ({"name", "dimension"} - declared_by_metric)

    conflicts = sorted(set(options) & (factory_supplied | (reserved or set())))
    if conflicts:
        raise TypeError(
            f"{metric_class.__name__} option(s) conflict with factory-supplied arguments: {', '.join(conflicts)}"
        )
    unknown = sorted(set(options) - accepted)
    if unknown:
        raise TypeError(f"{metric_class.__name__} received unknown option(s): {', '.join(unknown)}")


def build_llm_provider(config: dict[str, Any]) -> Any:
    """Instantiate the correct LLM provider from config metadata."""
    metadata = config.get("metadata", {})
    if metadata.get("llm") is not None:
        return metadata["llm"]
    provider = metadata.get("provider", "openai")
    model = metadata.get("model", "gpt-4o-mini")
    api_key = metadata.get("api_key")
    base_url = metadata.get("base_url")
    temperature = float(metadata["temperature"]) if metadata.get("temperature") is not None else None
    max_tokens = int(metadata["max_tokens"]) if metadata.get("max_tokens") is not None else 4096

    # Harness-managed connectors route through the OpenAI-compatible LLM
    # gateway (see harness_evals.llm.harness_ai). Only fires when base_url is
    # absent: callers that already resolved the gateway URL skip this branch.
    if metadata.get("use_llm_gateway") and not base_url:
        gateway_path = metadata.get("llm_gateway_path", "/llm-gw/v1")
        harness_base = os.environ.get("HARNESS_BASE_URL", "").rstrip("/")
        for suffix in ("/ng", "/gateway"):
            if harness_base.endswith(suffix):
                harness_base = harness_base[: -len(suffix)]
                break
        base_url = f"{harness_base}{gateway_path}" if harness_base else None
        api_key = os.environ.get("HARNESS_TOKEN", "")
        if not base_url or not api_key:
            raise ValueError(
                "Harness-managed LLM connector requires HARNESS_BASE_URL and HARNESS_TOKEN env vars "
                "when base_url is not pre-resolved"
            )

    if provider == "anthropic" and metadata.get("bedrock"):
        if BedrockAnthropicLLM is None:
            raise ValueError("Bedrock provider requires: pip install 'harness-evals[llm]'")
        return BedrockAnthropicLLM(
            model=model,
            api_key=api_key,
            aws_region=metadata.get("region"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "anthropic":
        if AnthropicLLM is None:
            raise ValueError("Anthropic provider requires: pip install 'harness-evals[llm]'")
        return AnthropicLLM(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "openai" and metadata.get("bedrock"):
        if BedrockOpenAILLM is None:
            raise ValueError("Bedrock provider requires: pip install 'harness-evals[llm]'")
        return BedrockOpenAILLM(
            model=model,
            api_key=api_key,
            aws_region=metadata.get("region"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if OpenAILLM is None:
        raise ValueError("OpenAI provider requires: pip install 'harness-evals[llm]'")
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAILLM(**kwargs)


def _build_llm_metric(
    config: dict[str, Any],
    score_name: str | None,
    threshold: float,
) -> BaseMetric:
    if OpenAILLM is None and AnthropicLLM is None:
        raise ValueError("LLM metric requires the 'llm' optional dependency.")

    llm = build_llm_provider(config)
    kind = config.get("kind")
    registry = _llm_metric_registry()
    metric_class = registry.get(kind) if kind else None
    options = config.get("options") or {}

    if metric_class is not None:
        if metric_class is AnswerCorrectnessMetric:
            try:
                embedding = OpenAIEmbedding()
            except ImportError as err:
                raise ValueError("AnswerCorrectnessMetric requires: pip install 'harness-evals[llm]'") from err
            _validate_constructor_options(metric_class, options)
            try:
                metric = metric_class(llm=llm, embedding=embedding, threshold=threshold, **options)
            except TypeError as e:
                if "got multiple values for keyword argument" in str(e):
                    import re
                    match = re.search(r"keyword argument '(\w+)'", str(e))
                    key = match.group(1) if match else "unknown"
                    raise TypeError(
                        f"{metric_class.__name__} option(s) conflict with factory-supplied arguments: {key}"
                    ) from None
                raise

        else:
            kind_kwargs: dict[str, Any] = {}
            if "criteria" in config:
                kind_kwargs["criteria"] = config["criteria"]
            if "rubric" in config and isinstance(config["rubric"], dict):
                kind_kwargs["rubric"] = {int(k): v for k, v in config["rubric"].items()}
            if "prompt_instructions" in config:
                kind_kwargs["prompt_instructions"] = config["prompt_instructions"]
            if "allowed_topics" in config:
                kind_kwargs["allowed_topics"] = config["allowed_topics"]
            _validate_constructor_options(metric_class, options, set(kind_kwargs))
            try:
                metric = metric_class(llm=llm, threshold=threshold, **kind_kwargs, **options)
            except TypeError as e:
                if "got multiple values for keyword argument" in str(e):
                    import re
                    match = re.search(r"keyword argument '(\w+)'", str(e))
                    key = match.group(1) if match else "unknown"
                    raise TypeError(
                        f"{metric_class.__name__} option(s) conflict with factory-supplied arguments: {key}"
                    ) from None
                raise

    else:
        rubric = config.get("rubric")
        if isinstance(rubric, dict):
            metric = RubricJudgeMetric(llm, rubric={int(k): v for k, v in rubric.items()}, threshold=threshold)
            metric.name = score_name or kind or "llm_judge"
            return metric
        criteria_text = _extract_criteria(config)
        metric = GEvalMetric(llm=llm, criteria=criteria_text, threshold=threshold)

    metric.name = score_name or kind or "llm_judge"
    return metric


def _build_embedding_metric(
    config: dict[str, Any],
    score_name: str | None,
    threshold: float,
) -> BaseMetric:
    kind = config.get("kind")
    if not kind:
        raise ValueError("Embedding metric config must have 'kind' field")

    registry = _embedding_registry()
    if kind not in registry:
        raise ValueError(f"Unknown embedding kind: {kind!r}. Available: {sorted(registry.keys())}")

    metadata = config.get("metadata", {})
    api_key = metadata.get("api_key")
    embedding_model = metadata.get("embedding_model")

    embed_kwargs: dict[str, Any] = {}
    if api_key:
        embed_kwargs["api_key"] = api_key
    if embedding_model:
        embed_kwargs["model"] = embedding_model
    embedding = OpenAIEmbedding(**embed_kwargs)

    metric_class = registry[kind]
    options = config.get("options") or {}
    _validate_constructor_options(metric_class, options)
    metric = metric_class(embedding=embedding, threshold=threshold, **options)

    metric.name = score_name or kind
    return metric


def _extract_criteria(config: dict[str, Any]) -> str:
    """Extract criteria/rubric text from config for GEval fallback."""
    criteria_list = config.get("criteria", [])
    prompt_config = config.get("prompt", {})
    if isinstance(prompt_config, dict):
        messages = prompt_config.get("messages", [])
        criteria_text = "\n".join(msg.get("content", "") for msg in messages).strip()
    else:
        criteria_text = str(prompt_config) if prompt_config else ""

    if not criteria_text and criteria_list:
        if isinstance(criteria_list, list):
            criteria_text = "; ".join(str(c) for c in criteria_list)
        else:
            criteria_text = str(criteria_list)

    return criteria_text or "Is the response accurate, relevant, and complete?"


def _build_heuristic_metric(
    config: dict[str, Any],
    score_name: str | None,
    threshold: float,
) -> BaseMetric:
    kind = config.get("kind")
    if not kind:
        raise ValueError("Heuristic metric config must have 'kind' field")

    registry = _heuristic_registry()
    if kind not in registry:
        raise ValueError(f"Unknown heuristic kind: {kind!r}. Available: {sorted(registry.keys())}")
    metric_class = registry[kind]
    options = config.get("options") or {}

    _validate_constructor_options(metric_class, options)
    metric = metric_class(threshold=threshold, **options)

    metric.name = score_name or kind
    return metric


def _build_code_metric(
    config: dict[str, Any],
    score_name: str | None,
    threshold: float,
    suite_path: Path | None,
    *,
    allow_code_loading: bool = True,
) -> BaseMetric:
    """Build code metric — supports module:ClassName or relative file path."""
    path = config.get("path")
    if not path:
        raise ValueError("Code metric config must have 'path' field")

    if not allow_code_loading:
        raise ValueError(
            "Code metrics are not allowed in server-side/online execution. "
            "Use the CLI or SDK to run code metrics locally."
        )

    if ":" in path:
        module_name, class_name = path.split(":", 1)
        _ALLOWED_MODULE_PREFIXES = ("harness_evals.",)
        if not any(module_name.startswith(prefix) for prefix in _ALLOWED_MODULE_PREFIXES):
            raise ValueError(
                f"Code metric module '{module_name}' is not from an allowed package. "
                f"Allowed prefixes: {_ALLOWED_MODULE_PREFIXES}"
            )
        module = importlib.import_module(module_name)
        metric_class = getattr(module, class_name)
    else:
        p = Path(path)
        if p.is_absolute():
            full_path = p.resolve()
            if not full_path.exists():
                raise ValueError(f"Code metric file not found: {full_path}")
        else:
            if suite_path is None:
                raise ValueError(
                    f"Code metric path '{path}' is a relative file path but no suite_path was provided to resolve it"
                )
            full_path = (suite_path.parent / path).resolve()
            suite_dir = suite_path.parent.resolve()
            if not str(full_path).startswith(str(suite_dir) + "/") and full_path != suite_dir:
                raise ValueError(f"Code metric path '{path}' resolves outside the suite directory")
        spec = importlib.util.spec_from_file_location("metric_module", full_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load metric from {full_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metric_class = _find_metric_class(module)
        if metric_class is None:
            raise ValueError(f"Metric module {full_path} must contain a BaseMetric subclass")

    if not issubclass(metric_class, BaseMetric):
        raise ValueError(f"{metric_class.__name__} must extend BaseMetric, got {metric_class}")

    extra_config = config.get("config") or {}
    _validate_constructor_options(metric_class, extra_config)
    metric = metric_class(threshold=threshold, **extra_config)

    metric.name = score_name or path
    return metric


def _build_composite_metric(
    config: dict[str, Any],
    score_name: str | None,
    threshold: float,
    suite_path: Path | None,
    *,
    allow_code_loading: bool = True,
) -> BaseMetric:
    sub_configs = config.get("metrics", [])
    aggregation = config.get("aggregation", "average")

    if not sub_configs:
        raise ValueError("Composite metric must have at least one sub-metric")

    sub_metrics: list[BaseMetric] = []
    weights: list[float] = []
    for ref_config in sub_configs:
        ref = ref_config.get("ref")
        if not ref:
            raise ValueError("Composite sub-metric must have 'ref' field")
        raw_weight = ref_config.get("weight", 1.0)
        weight = float(raw_weight) if raw_weight is not None else 1.0

        sub_type = ref_config.get("type")
        sub_config = ref_config.get("config") or {}
        if sub_type:
            sub_metric = build_metric(
                sub_type, sub_config, score_name=ref, suite_path=suite_path, allow_code_loading=allow_code_loading
            )
        else:
            raise ValueError(f"Composite sub-metric '{ref}' must have 'type' field")

        sub_metrics.append(sub_metric)
        weights.append(weight)

    if aggregation == "weighted_average" and sum(weights) == 0:
        raise ValueError("Composite metric weighted_average has zero total weight")

    final_name = score_name or "composite"

    class CompositeMetric(BaseMetric):
        def __init__(self) -> None:
            super().__init__(name=final_name, dimension=Dimension.CORRECTNESS, threshold=threshold)
            self.sub_metrics = sub_metrics
            self.weights = weights
            self.aggregation = aggregation

        def _build_score(
            self,
            sub_scores: list[float],
            sub_weights: list[float],
            sub_details: list[dict[str, Any]],
        ) -> Score | None:
            if not sub_scores:
                return None

            if self.aggregation == "weighted_average":
                total_w = sum(sub_weights)
                if total_w == 0:
                    return None
                final = sum(s * w for s, w in zip(sub_scores, sub_weights, strict=True)) / total_w
            elif self.aggregation == "min":
                final = min(sub_scores)
            elif self.aggregation == "max":
                final = max(sub_scores)
            elif self.aggregation == "all_pass":
                final = 1.0 if all(s >= 1.0 for s in sub_scores) else 0.0
            else:
                final = sum(sub_scores) / len(sub_scores)

            return Score(
                name=self.name,
                value=max(0.0, min(1.0, final)),
                threshold=self.threshold,
                metadata={"sub_scores": sub_details, "aggregation": self.aggregation},
            )

        async def a_measure(self, eval_case: EvalCase) -> Score | None:
            sub_scores: list[float] = []
            sub_weights: list[float] = []
            sub_details: list[dict[str, Any]] = []
            for m, weight in zip(self.sub_metrics, self.weights, strict=True):
                s = await m.a_measure(eval_case)
                if asyncio.iscoroutine(s):
                    s = await s
                if s is None:
                    continue
                sub_scores.append(s.value)
                sub_weights.append(weight)
                sub_details.append({"name": m.name, "value": s.value})

            return self._build_score(sub_scores, sub_weights, sub_details)

        def measure(self, eval_case: EvalCase) -> Score | None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                logger.warning(
                    "Composite metric '%s' is being measured synchronously inside a running event loop; "
                    "async sub-metrics may be skipped.",
                    self.name,
                )
                sub_scores: list[float] = []
                sub_weights: list[float] = []
                sub_details: list[dict[str, Any]] = []
                for m, weight in zip(self.sub_metrics, self.weights, strict=True):
                    s = m.measure(eval_case)
                    if asyncio.iscoroutine(s):
                        s.close()
                        continue
                    if s is None:
                        continue
                    sub_scores.append(s.value)
                    sub_weights.append(weight)
                    sub_details.append({"name": m.name, "value": s.value})
                return self._build_score(sub_scores, sub_weights, sub_details)
            return asyncio.run(self.a_measure(eval_case))

    return CompositeMetric()


def _find_metric_class(module: Any) -> type[BaseMetric] | None:
    fallback = None
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if isinstance(obj, type) and issubclass(obj, BaseMetric) and obj is not BaseMetric:
            if obj.__module__ == module.__name__:
                return obj
            if fallback is None:
                fallback = obj
    return fallback
