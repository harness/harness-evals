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
import types
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from harness_evals import BaseMetric, Dimension, EvalCase, Score
from harness_evals.catalog import catalog
from harness_evals.llm.openai_embedding import OpenAIEmbedding
from harness_evals.metrics import AnswerCorrectnessMetric, GEvalMetric, RubricJudgeMetric

logger = logging.getLogger("harness_evals.metrics.factory")

# Legacy option names supported by earlier metric configuration schemas.
# Translate only values whose semantics remain unambiguous.
_LEGACY_HEURISTIC_OPTIONS = {
    "latency": ("max_value", "max_ms"),
    "token_cost": ("max_value", "max_tokens"),
    "cost_efficiency": ("max_value", "max_cost_usd"),
    "bleu": ("max_ngram", "max_n"),
    "tool_correctness": ("pair", "mode"),
}

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
    effective_config = normalize_metric_config(metric_type, config, entry_config)

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


def normalize_metric_config(
    metric_type: str,
    config: dict[str, Any],
    entry_config: dict[str, Any] | None = None,
    *,
    kind: str | None = None,
) -> dict[str, Any]:
    """Merge metric config and translate compatible legacy heuristic options.

    The result is a new mapping.  In particular, ``token_cost.max_value`` is
    only interpreted as ``max_tokens`` for positive, non-boolean integers:
    historical fractional values represented a monetary cost instead.
    """
    effective_config = merge_metric_config(config, entry_config)
    options = effective_config.get("options")
    kind = kind or effective_config.get("kind")
    if metric_type != "heuristic" or not kind or not isinstance(options, dict):
        return effective_config

    normalized_options = dict(options)
    option_names = _LEGACY_HEURISTIC_OPTIONS.get(kind)
    if option_names is not None:
        legacy_option_name, option_name = option_names
        if legacy_option_name in normalized_options:
            legacy_value = normalized_options[legacy_option_name]
            token_cost_value_is_compatible = kind != "token_cost" or (
                isinstance(legacy_value, int) and not isinstance(legacy_value, bool) and legacy_value > 0
            )
            if token_cost_value_is_compatible:
                normalized_options.pop(legacy_option_name)
                if option_name in normalized_options:
                    logger.warning(
                        "Heuristic metric %r received both %r and %r; ignoring the legacy %r.",
                        kind,
                        legacy_option_name,
                        option_name,
                        legacy_option_name,
                    )
                normalized_options.setdefault(option_name, legacy_value)

    options_schema = heuristic_options_schema(kind)
    if options_schema is not None:
        unsupported_options = sorted(set(normalized_options) - set(options_schema["properties"]))
        if unsupported_options:
            logger.warning(
                "Heuristic metric %r has options not declared by the SDK: %s; "
                "they may be rejected by the metric factory.",
                kind,
                ", ".join(unsupported_options),
            )
    return {**effective_config, "options": normalized_options}


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


def heuristic_options_schema(kind: str) -> dict[str, Any] | None:
    """Return JSON Schema properties for a built-in heuristic metric's options."""
    metric_class = _heuristic_registry().get(kind)
    if metric_class is None:
        return None

    properties: dict[str, Any] = {}
    for klass in metric_class.__mro__:
        if klass.__module__ == "harness_evals.core.metric":
            continue
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            annotations = get_type_hints(init)
        except (NameError, TypeError):
            annotations = {}
        for name, parameter in inspect.signature(init).parameters.items():
            if name in ("self", "threshold") or parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            prop = _infer_json_schema_type(
                parameter.default,
                annotations.get(name, parameter.annotation),
            )
            if prop and parameter.default is not None and parameter.default is not inspect.Parameter.empty:
                prop["default"] = parameter.default
            properties.setdefault(name, prop)
    return {"type": "object", "properties": properties}


def _infer_json_schema_type(value: Any, annotation: Any = inspect.Parameter.empty) -> dict[str, Any]:
    """Infer a JSON Schema property type from a constructor default."""
    annotated = _json_schema_from_annotation(annotation)
    if annotated:
        return {**annotated, "default": None} if value is None else annotated
    if value is inspect.Parameter.empty or value is None:
        return {}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        if value and all(isinstance(item, str) for item in value):
            return {"type": "array", "items": {"type": "string"}}
        return {"type": "array"}
    if isinstance(value, dict):
        return {"type": "object", "additionalProperties": True}
    return {}


def _json_schema_from_annotation(annotation: Any) -> dict[str, Any]:
    """Convert a supported constructor type annotation to a JSON Schema type."""
    origin = get_origin(annotation)
    if origin in (types.UnionType, getattr(types, "UnionType", None)) or str(origin) == "typing.Union":
        schemas = [_json_schema_from_annotation(arg) for arg in get_args(annotation)]
        if not all(schemas):
            return {}
        null_schema = {"type": "null"}
        non_null_schemas = [schema for schema in schemas if schema != null_schema]
        if len(non_null_schemas) == 1 and len(non_null_schemas) != len(schemas):
            schema = dict(non_null_schemas[0])
            schema_type = schema["type"]
            schema["type"] = [*schema_type, "null"] if isinstance(schema_type, list) else [schema_type, "null"]
            return schema
        schema_types = [
            schema_type
            for schema in schemas
            for schema_type in (schema["type"] if isinstance(schema["type"], list) else [schema["type"]])
        ]
        if schema_types:
            return {"type": schema_types[0] if len(schema_types) == 1 else schema_types}
        return {}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is str:
        return {"type": "string"}
    if annotation is list or origin is list:
        args = get_args(annotation)
        return {"type": "array", "items": {"type": "string"}} if args == (str,) else {"type": "array"}
    if annotation is set or origin is set:
        args = get_args(annotation)
        schema = {"type": "array", "uniqueItems": True}
        if args == (str,):
            schema["items"] = {"type": "string"}
        return schema
    if annotation is dict or origin is dict:
        return {"type": "object", "additionalProperties": True}
    if annotation is type(None):
        return {"type": "null"}
    return {}


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
    # None means "not set" — each provider's own constructor default wins. Only pass max_tokens
    # when explicitly configured so e.g. BedrockOpenAILLM's 8192 default applies for gpt-oss judges.
    max_tokens: int | None = int(metadata["max_tokens"]) if metadata.get("max_tokens") is not None else None
    token_kwargs: dict[str, Any] = {"max_tokens": max_tokens} if max_tokens is not None else {}

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
            **token_kwargs,
        )

    if provider == "anthropic":
        if AnthropicLLM is None:
            raise ValueError("Anthropic provider requires: pip install 'harness-evals[llm]'")
        return AnthropicLLM(
            model=model,
            api_key=api_key,
            temperature=temperature,
            **token_kwargs,
        )

    if provider == "openai" and metadata.get("bedrock"):
        if BedrockOpenAILLM is None:
            raise ValueError("Bedrock provider requires: pip install 'harness-evals[llm]'")
        return BedrockOpenAILLM(
            model=model,
            api_key=api_key,
            aws_region=metadata.get("region"),
            temperature=temperature,
            **token_kwargs,
        )

    if OpenAILLM is None:
        raise ValueError("OpenAI provider requires: pip install 'harness-evals[llm]'")
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        **token_kwargs,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAILLM(**kwargs)


def build_embedding_provider(metadata: dict[str, Any]) -> OpenAIEmbedding:
    """Instantiate an OpenAIEmbedding routed to the right backend.

    Routing precedence (mirrors build_llm_provider):
    1. ``embedding_api_key`` in metadata — explicit per-metric override, wins always.
       Optionally paired with ``embedding_base_url`` and ``embedding_model``.
    2. ``use_llm_gateway`` — Harness-managed connector; route through the LLM gateway
       (OpenAI-compatible) using HARNESS_TOKEN + HARNESS_BASE_URL.
    3. ``provider == "openai"`` + ``bedrock`` — same bearer key, Bedrock OpenAI-compat endpoint.
    4. ``provider == "openai"`` direct — use the judge key as-is, no base_url.
    5. ``provider == "anthropic"`` (direct or bedrock) — Anthropic has no embeddings API.
       Raises a clear error directing the user to set ``embedding_api_key`` on the metric config.
    """
    embedding_api_key = metadata.get("embedding_api_key")
    embedding_model = metadata.get("embedding_model") or "text-embedding-3-small"

    # 1. Explicit per-metric embedding credential — always wins.
    # Targets platform.openai.com directly; no base_url needed.
    if embedding_api_key:
        return OpenAIEmbedding(api_key=embedding_api_key, model=embedding_model)

    provider = metadata.get("provider", "openai")

    # 2. Harness-managed (gateway) — resolve base_url from env, use HARNESS_TOKEN.
    if metadata.get("use_llm_gateway"):
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
                "Harness-managed connector requires HARNESS_BASE_URL and HARNESS_TOKEN env vars "
                "to route embeddings through the gateway"
            )
        return OpenAIEmbedding(api_key=api_key, base_url=base_url, model=embedding_model)

    # 3. OpenAI via Bedrock — same bearer, Bedrock OpenAI-compat endpoint.
    if provider == "openai" and metadata.get("bedrock"):
        region = metadata.get("region") or os.environ.get("AWS_REGION") or "us-east-1"
        base_url = f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1"
        api_key = metadata.get("api_key")
        if not api_key:
            raise ValueError(
                "Bedrock OpenAI embedding requires an API key (the Bedrock bearer token). "
                "Ensure the judge connector key is decrypted and available in metric metadata."
            )
        return OpenAIEmbedding(api_key=api_key, base_url=base_url, model=embedding_model)

    # 4. Direct OpenAI — use the judge key.
    if provider == "openai":
        api_key = metadata.get("api_key")
        kwargs = {"model": embedding_model}
        if api_key:
            kwargs["api_key"] = api_key
        try:
            return OpenAIEmbedding(**kwargs)
        except ValueError as err:
            raise ValueError(
                "Embedding-based metric requires an OpenAI API key. "
                "Set it via 'embedding_api_key' in the metric config, or set OPENAI_API_KEY."
            ) from err

    # 5. Anthropic (direct or Bedrock) — no embeddings API exists.
    raise ValueError(
        f"The judge connector uses provider '{provider}', which has no embeddings API. "
        "Embedding-based metrics (answer_correctness, answer_similarity, embedding_similarity) "
        "require an OpenAI-compatible embedding backend. "
        "Set 'embedding_api_key' in the metric config with a valid OpenAI key."
    )


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
            embedding = build_embedding_provider(config.get("metadata", {}))
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
    embedding = build_embedding_provider(metadata)

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
