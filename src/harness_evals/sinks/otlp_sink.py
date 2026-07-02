"""OTLP sink — export eval scores as OpenTelemetry metrics and traces."""

from __future__ import annotations

import json
import threading
import uuid
import warnings
from typing import Any

try:
    from opentelemetry import trace as trace_api
    from opentelemetry.context import Context
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import StatusCode
except ImportError as _err:
    raise ImportError(
        "OtlpSink requires the opentelemetry packages. Install them with: pip install harness-evals[otlp]"
    ) from _err

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.summary import SAFETY_DIMENSION, UNKNOWN_DIMENSION

_MAX_ATTR_LEN = 1000
_VALID_PROTOCOLS = {"grpc", "http"}


def _truncate(value: Any, max_len: int = _MAX_ATTR_LEN) -> str:
    """Convert value to string and truncate to max_len."""
    s = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return s[:max_len] if len(s) > max_len else s


def _http_otlp_endpoints(endpoint: str) -> tuple[str, str]:
    """Return (trace_url, metrics_url) for OTLP HTTP/protobuf exporters.

    OpenTelemetry HTTP exporters use the constructor ``endpoint`` verbatim; they do
    not append ``/v1/traces`` or ``/v1/metrics`` when ``endpoint`` is set explicitly.

    * If ``endpoint`` already ends with ``/v1/traces`` or ``/v1/metrics``, the sibling
      signal URL is derived from the same prefix.
    * Otherwise ``endpoint`` is treated as a base URL and the standard paths are
      appended (after stripping a trailing slash).
    """
    base = endpoint.rstrip("/")
    suffix_traces = "/v1/traces"
    suffix_metrics = "/v1/metrics"
    if base.endswith(suffix_traces):
        trace_url = base
        metrics_url = base[: -len(suffix_traces)] + suffix_metrics
        return trace_url, metrics_url
    if base.endswith(suffix_metrics):
        metrics_url = base
        trace_url = base[: -len(suffix_metrics)] + suffix_traces
        return trace_url, metrics_url
    return f"{base}{suffix_traces}", f"{base}{suffix_metrics}"


def _load_grpc_exporters(endpoint: str, insecure: bool, *, headers: dict[str, str] | None = None) -> tuple[Any, Any]:
    """Import and instantiate gRPC metric + span exporters."""
    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ImportError as exc:
        raise ImportError(
            "gRPC exporters require opentelemetry-exporter-otlp-proto-grpc. "
            "Install with: pip install harness-evals[otlp]"
        ) from exc
    kwargs: dict[str, Any] = {"endpoint": endpoint, "insecure": insecure}
    if headers:
        kwargs["headers"] = list(headers.items())
    return OTLPMetricExporter(**kwargs), OTLPSpanExporter(**kwargs)


def _load_http_exporters(
    trace_endpoint: str,
    metrics_endpoint: str,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[Any, Any]:
    """Import and instantiate HTTP metric + span exporters (distinct OTLP URLs)."""
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as exc:
        raise ImportError(
            "HTTP exporters require opentelemetry-exporter-otlp-proto-http. "
            "Install with: pip install harness-evals[otlp]"
        ) from exc
    m_kwargs: dict[str, Any] = {"endpoint": metrics_endpoint}
    t_kwargs: dict[str, Any] = {"endpoint": trace_endpoint}
    if headers:
        m_kwargs["headers"] = headers
        t_kwargs["headers"] = headers
    return OTLPMetricExporter(**m_kwargs), OTLPSpanExporter(**t_kwargs)


class OtlpSink(BaseSink):
    """Export eval scores as OpenTelemetry metrics and traces to an OTLP endpoint.

    **Metrics**: Each ``Score`` becomes a gauge observation on ``evals.score``.
    ``EvalCase`` runtime fields (latency, tokens, cost) are recorded as histograms.

    **Traces**: A root ``eval-run`` span contains child ``eval-item`` spans
    (one per ``write()`` call), each with ``evals.score`` events per metric.

    Deployment-specific attributes (environment, team, project) are injected by the
    caller via ``resource_attributes`` and ``extra_attributes`` — the sink itself
    uses only generic ``eval.*`` attribute names. Use ``item_span_attributes`` to
    override or add attributes specific to item spans (e.g., ``span.type: eval_item``).

    **Context propagation**: Pass ``parent_context`` to attach the eval-run span
    to an existing trace (e.g., an orchestration span from the calling engine).
    The sink's spans will share the parent's trace ID.

    **Engine-owned item spans**: When the engine creates its own item spans
    (with structured children like target-invoke, scorer), pass ``item_context``
    to ``write()`` so the sink decorates the engine's span with score events
    instead of creating a duplicate child span. When using ``item_context``,
    also pass ``parent_context`` at construction so the sink's root eval-run
    span (which holds the summary) attaches to the engine's trace tree.
    Without ``parent_context``, the summary span becomes an orphaned root —
    still queryable by ``eval.run_id`` but disconnected from the engine's trace.

    **Shared providers**: Pass ``tracer_provider`` and/or ``meter_provider`` to
    reuse an existing export pipeline. The sink will NOT flush or shutdown
    providers it does not own — the caller retains lifecycle control. When
    external providers are supplied, ``endpoint``, ``protocol``, ``insecure``,
    and ``headers`` are ignored for the corresponding signal.

    Call ``finalize()`` after all writes to end the root span and flush data.
    ``shutdown()`` calls ``finalize()`` if it hasn't been called, then releases
    owned resources. The caller must ensure all ``write()`` calls complete before
    calling ``finalize()`` — concurrent write and finalize is not supported.

    The ``insecure`` parameter controls TLS for ``protocol="grpc"`` only.
    For ``protocol="http"``, TLS is determined by the URL scheme (``https://``
    vs ``http://``). For HTTP, ``endpoint`` is the OTLP **base** URL: ``/v1/traces``
    and ``/v1/metrics`` are appended unless the URL already ends with one of those
    paths (the sibling signal URL is derived automatically).

    Requires ``pip install harness-evals[otlp]``.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        service_name: str = "harness-evals",
        insecure: bool = True,
        *,
        protocol: str = "grpc",
        headers: dict[str, str] | None = None,
        run_id: str | None = None,
        model: str | None = None,
        resource_attributes: dict[str, str] | None = None,
        extra_attributes: dict[str, str] | None = None,
        item_span_attributes: dict[str, str] | None = None,
        parent_context: Context | None = None,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
    ) -> None:
        if protocol not in _VALID_PROTOCOLS:
            raise ValueError(f"Unsupported protocol {protocol!r}, must be one of {_VALID_PROTOCOLS}")

        self.endpoint = endpoint
        self.service_name = service_name
        self._run_id = run_id or str(uuid.uuid4())
        self._model = model
        self._extra_attributes = extra_attributes or {}
        self._item_span_attributes = item_span_attributes or {}
        self._parent_context = parent_context
        self._finalized = False

        self._owns_trace_provider = tracer_provider is None
        self._owns_meter_provider = meter_provider is None

        # --- Providers (internal or external) ---
        if (tracer_provider is not None or meter_provider is not None) and endpoint != "http://localhost:4317":
            warnings.warn(
                "endpoint/protocol/insecure/headers are ignored when tracer_provider "
                "or meter_provider is supplied. Configure export on the provider you pass in.",
                stacklevel=2,
            )

        if tracer_provider is not None and meter_provider is not None:
            self._trace_provider = tracer_provider
            self._meter_provider = meter_provider
        else:
            tp, mp = self._create_internal_providers(
                endpoint, service_name, insecure, protocol, headers, resource_attributes,
            )
            self._trace_provider = tracer_provider if tracer_provider is not None else tp
            self._meter_provider = meter_provider if meter_provider is not None else mp

        # --- Metrics ---
        meter = self._meter_provider.get_meter(service_name)

        self._score_gauge = meter.create_gauge(
            name="evals.score",
            description="Evaluation metric score",
            unit="ratio",
        )
        self._latency_hist = meter.create_histogram(
            name="evals.item.latency",
            description="Target invocation latency per eval item",
            unit="ms",
        )
        self._token_hist = meter.create_histogram(
            name="evals.item.tokens",
            description="Token count per eval item",
            unit="tokens",
        )
        self._cost_hist = meter.create_histogram(
            name="evals.item.cost",
            description="Cost per eval item",
            unit="usd",
        )

        # --- Traces ---
        self._tracer = self._trace_provider.get_tracer(service_name)

        # Root span is created lazily on first write() so that its start time
        # aligns with actual evaluation start, not sink construction.
        self._root_span = None

        # --- Summary accumulators (for finalize) ---
        self._lock = threading.Lock()
        self._item_count = 0
        self._items_passed = 0
        self._score_values: dict[str, list[float]] = {}
        # Per-dimension accumulators for the root-span summary (ADR-009).
        self._dimension_values: dict[str, list[float]] = {}
        self._dimension_passed: dict[str, int] = {}

    @staticmethod
    def _create_internal_providers(
        endpoint: str,
        service_name: str,
        insecure: bool,
        protocol: str,
        headers: dict[str, str] | None,
        resource_attributes: dict[str, str] | None,
    ) -> tuple[TracerProvider, MeterProvider]:
        """Build owned TracerProvider + MeterProvider with exporters."""
        res_attrs = {"service.name": service_name, **(resource_attributes or {})}
        resource = Resource.create(res_attrs)

        if protocol == "http":
            if not insecure:
                warnings.warn(
                    "insecure=False has no effect with protocol='http'. "
                    "TLS is controlled by the URL scheme (https:// vs http://).",
                    stacklevel=2,
                )
            trace_url, metrics_url = _http_otlp_endpoints(endpoint)
            metric_exporter, span_exporter = _load_http_exporters(trace_url, metrics_url, headers=headers)
        else:
            metric_exporter, span_exporter = _load_grpc_exporters(endpoint, insecure, headers=headers)

        reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=1000)
        mp = MeterProvider(resource=resource, metric_readers=[reader])

        tp = TracerProvider(resource=resource)
        tp.add_span_processor(BatchSpanProcessor(span_exporter))

        return tp, mp

    def _ensure_root_span(self) -> None:
        """Create the root eval-run span on first write (under lock)."""
        if self._root_span is not None:
            return
        root_attrs: dict[str, Any] = {
            "eval.run_id": self._run_id,
            **self._extra_attributes,
        }
        if self._model:
            root_attrs["model"] = self._model
        self._root_span = self._tracer.start_span(
            "eval-run",
            context=self._parent_context,
            attributes=root_attrs,
        )

    def write(self, scores: list[Score], eval_case: EvalCase, *, item_context: Context | None = None) -> None:
        """Emit scores for a single eval case.

        When ``item_context`` is provided (a Context containing the engine's
        item span), score events are added to that existing span instead of
        creating a new child span. The engine owns the span lifecycle — the
        sink will not end it. Summary accumulators are still updated so
        ``finalize()`` can emit the root span summary.
        """
        if not scores:
            return

        tags = eval_case.tags or {}
        base_attrs = {
            "eval.run_id": self._run_id,
            **{f"tag.{k}": v for k, v in tags.items()},
            **self._extra_attributes,
        }

        # --- Metrics ---
        for score in scores:
            metric_attrs = {
                **base_attrs,
                "eval.metric_name": score.name,
                "eval.threshold": str(score.threshold),
                "eval.passed": str(score.passed),
            }
            meta = score.metadata or {}
            for mk, mv in meta.items():
                if isinstance(mv, (str, int, float, bool)):
                    metric_attrs[f"eval.meta.{mk}"] = mv if isinstance(mv, str) else str(mv)
            if meta.get("dimension"):
                metric_attrs["eval.dimension"] = meta["dimension"]
            if self._model:
                metric_attrs["model"] = self._model
            self._score_gauge.set(score.value, attributes=metric_attrs)

        if eval_case.latency_ms is not None:
            self._latency_hist.record(eval_case.latency_ms, attributes=base_attrs)
        if eval_case.token_count is not None:
            self._token_hist.record(eval_case.token_count, attributes=base_attrs)
        if eval_case.cost_usd is not None:
            self._cost_hist.record(eval_case.cost_usd, attributes=base_attrs)

        # --- Traces ---
        with self._lock:
            self._ensure_root_span()
            self._item_count += 1
            item_index = self._item_count
            all_passed = all(s.passed for s in scores)
            if all_passed:
                self._items_passed += 1
            for s in scores:
                self._score_values.setdefault(s.name, []).append(s.value)
                dimension = (s.metadata or {}).get("dimension") or UNKNOWN_DIMENSION
                self._dimension_values.setdefault(dimension, []).append(s.value)
                self._dimension_passed[dimension] = self._dimension_passed.get(dimension, 0) + int(s.passed)

        # Build score event attributes (used in both paths)
        score_events: list[dict[str, Any]] = []
        for score in scores:
            event_attrs: dict[str, Any] = {
                "eval.metric_name": score.name,
                "eval.score.value": score.value,
                "eval.score.threshold": score.threshold,
                "eval.score.passed": score.passed,
            }
            dimension = (score.metadata or {}).get("dimension")
            if dimension:
                event_attrs["eval.dimension"] = dimension
            if score.reason:
                event_attrs["eval.score.reason"] = score.reason
            score_events.append(event_attrs)

        if item_context is not None:
            # Decorate the engine's existing item span with score events.
            # The engine owns the span — we don't end it.
            item_span = trace_api.get_current_span(item_context)
            item_span.set_attribute("eval.item.passed", all_passed)
            for event_attrs in score_events:
                item_span.add_event("eval.score", attributes=event_attrs)
        else:
            # Create our own child span under root.
            item_attrs: dict[str, Any] = {
                "eval.run_id": self._run_id,
                "eval.item.index": item_index,
                "eval.item.passed": all_passed,
                **self._extra_attributes,
                **self._item_span_attributes,
            }
            if scores:
                span_meta = scores[0].metadata or {}
                for mk, mv in span_meta.items():
                    if isinstance(mv, (str, int, float, bool)):
                        item_attrs[f"eval.meta.{mk}"] = mv if isinstance(mv, str) else str(mv)
                if self._model:
                    item_attrs["model"] = self._model
            if eval_case.input is not None:
                item_attrs["eval.item.input"] = _truncate(eval_case.input)
            if eval_case.output is not None:
                item_attrs["eval.item.output"] = _truncate(eval_case.output)
            if eval_case.expected is not None:
                item_attrs["eval.item.expected"] = _truncate(eval_case.expected)
            if eval_case.latency_ms is not None:
                item_attrs["eval.item.latency_ms"] = eval_case.latency_ms
            if eval_case.token_count is not None:
                item_attrs["eval.item.token_count"] = eval_case.token_count
            if eval_case.cost_usd is not None:
                item_attrs["eval.item.cost_usd"] = eval_case.cost_usd

            ctx = trace_api.set_span_in_context(self._root_span)
            item_span = self._tracer.start_span("eval-item", context=ctx, attributes=item_attrs)
            for event_attrs in score_events:
                item_span.add_event("eval.score", attributes=event_attrs)
            item_span.end()

    def finalize(self) -> None:
        """End the root span with summary attributes and flush both providers."""
        if self._finalized:
            return
        self._finalized = True

        with self._lock:
            root = self._root_span
            item_count = self._item_count
            items_passed = self._items_passed
            score_values = {k: list(v) for k, v in self._score_values.items()}
            dimension_values = {k: list(v) for k, v in self._dimension_values.items()}
            dimension_passed = dict(self._dimension_passed)

        if root is not None:
            root.set_attribute("eval.summary.items_total", item_count)
            root.set_attribute("eval.summary.items_passed", items_passed)
            root.set_attribute("eval.summary.status", "completed")
            # Per-metric averages as summary event
            summary_attrs: dict[str, Any] = {}
            for name, values in score_values.items():
                summary_attrs[f"eval.summary.{name}.mean"] = round(sum(values) / len(values), 4)
            # Per-dimension averages (ADR-009), plus Safety violation count as a
            # hard-constraint signal (ADR-003). Set on the root span so a single
            # span query returns the whole radar shape.
            for dimension, values in dimension_values.items():
                mean = round(sum(values) / len(values), 4)
                root.set_attribute(f"eval.summary.dimension.{dimension}.mean", mean)
                summary_attrs[f"eval.summary.dimension.{dimension}.mean"] = mean
                if dimension == SAFETY_DIMENSION:
                    violations = len(values) - dimension_passed.get(dimension, 0)
                    root.set_attribute("eval.summary.dimension.safety.violations", violations)
                    summary_attrs["eval.summary.dimension.safety.violations"] = violations
            if summary_attrs:
                root.add_event("eval.summary", attributes=summary_attrs)
            root.set_status(StatusCode.OK)
            root.end()

        if self._owns_trace_provider:
            self._trace_provider.force_flush()
        if self._owns_meter_provider:
            self._meter_provider.force_flush()

    def shutdown(self) -> None:
        """Finalize (if needed), then release resources for owned providers."""
        self.finalize()
        if self._owns_meter_provider:
            self._meter_provider.shutdown()
        if self._owns_trace_provider:
            self._trace_provider.shutdown()
