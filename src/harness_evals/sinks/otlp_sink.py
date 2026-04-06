"""OTLP sink — export eval scores as OpenTelemetry metrics and traces."""

from __future__ import annotations

import json
import threading
import uuid
import warnings
from typing import Any

try:
    from opentelemetry import trace as trace_api
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

_MAX_ATTR_LEN = 1000
_VALID_PROTOCOLS = {"grpc", "http"}


def _truncate(value: Any, max_len: int = _MAX_ATTR_LEN) -> str:
    """Convert value to string and truncate to max_len."""
    s = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return s[:max_len] if len(s) > max_len else s


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


def _load_http_exporters(endpoint: str, *, headers: dict[str, str] | None = None) -> tuple[Any, Any]:
    """Import and instantiate HTTP metric + span exporters."""
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as exc:
        raise ImportError(
            "HTTP exporters require opentelemetry-exporter-otlp-proto-http. "
            "Install with: pip install harness-evals[otlp]"
        ) from exc
    kwargs: dict[str, Any] = {"endpoint": endpoint}
    if headers:
        kwargs["headers"] = headers
    return OTLPMetricExporter(**kwargs), OTLPSpanExporter(**kwargs)


class OtlpSink(BaseSink):
    """Export eval scores as OpenTelemetry metrics and traces to an OTLP endpoint.

    **Metrics**: Each ``Score`` becomes a gauge observation on ``eval.score``.
    ``EvalCase`` runtime fields (latency, tokens, cost) are recorded as histograms.

    **Traces**: A root ``eval-run`` span contains child ``eval-item`` spans
    (one per ``write()`` call), each with ``eval.score`` events per metric.

    Deployment-specific attributes (environment, team, project) are injected by the
    caller via ``resource_attributes`` and ``extra_attributes`` — the sink itself
    uses only generic ``eval.*`` attribute names.

    Call ``finalize()`` after all writes to end the root span and flush data.
    ``shutdown()`` calls ``finalize()`` if it hasn't been called, then releases
    resources. The caller must ensure all ``write()`` calls complete before
    calling ``finalize()`` — concurrent write and finalize is not supported.

    The ``insecure`` parameter controls TLS for ``protocol="grpc"`` only.
    For ``protocol="http"``, TLS is determined by the URL scheme (``https://``
    vs ``http://``).

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
    ) -> None:
        if protocol not in _VALID_PROTOCOLS:
            raise ValueError(f"Unsupported protocol {protocol!r}, must be one of {_VALID_PROTOCOLS}")

        self.endpoint = endpoint
        self.service_name = service_name
        self._run_id = run_id or str(uuid.uuid4())
        self._model = model
        self._extra_attributes = extra_attributes or {}
        self._finalized = False

        # --- Resource (shared by metrics + traces) ---
        res_attrs = {"service.name": service_name, **(resource_attributes or {})}
        resource = Resource.create(res_attrs)

        # --- Exporters ---
        if protocol == "http":
            if not insecure:
                warnings.warn(
                    "insecure=False has no effect with protocol='http'. "
                    "TLS is controlled by the URL scheme (https:// vs http://).",
                    stacklevel=2,
                )
            metric_exporter, span_exporter = _load_http_exporters(endpoint, headers=headers)
        else:
            metric_exporter, span_exporter = _load_grpc_exporters(endpoint, insecure, headers=headers)

        # --- Metrics ---
        reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=1000)
        self._meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        meter = self._meter_provider.get_meter(service_name)

        self._score_gauge = meter.create_gauge(
            name="eval.score",
            description="Evaluation metric score",
            unit="ratio",
        )
        self._latency_hist = meter.create_histogram(
            name="eval.item.latency",
            description="Target invocation latency per eval item",
            unit="ms",
        )
        self._token_hist = meter.create_histogram(
            name="eval.item.tokens",
            description="Token count per eval item",
            unit="tokens",
        )
        self._cost_hist = meter.create_histogram(
            name="eval.item.cost",
            description="Cost per eval item",
            unit="usd",
        )

        # --- Traces ---
        self._trace_provider = TracerProvider(resource=resource)
        self._trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        self._tracer = self._trace_provider.get_tracer(service_name)

        # Root span is created lazily on first write() so that its start time
        # aligns with actual evaluation start, not sink construction.
        self._root_span = None

        # --- Summary accumulators (for finalize) ---
        self._lock = threading.Lock()
        self._item_count = 0
        self._items_passed = 0
        self._score_values: dict[str, list[float]] = {}

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
            attributes=root_attrs,
        )

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
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
            # Propagate generic score.metadata as OTel attributes (prefixed with "eval.meta.")
            # Callers control what's in metadata — the sink doesn't interpret it.
            meta = score.metadata or {}
            for mk, mv in meta.items():
                if isinstance(mv, (str, int, float, bool)):
                    metric_attrs[f"eval.meta.{mk}"] = mv if isinstance(mv, str) else str(mv)
            # dimension gets a first-class attribute (generic eval concept)
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

        # Build child span attributes
        item_attrs: dict[str, Any] = {
            "eval.run_id": self._run_id,
            "eval.item.index": item_index,
            "eval.item.passed": all_passed,
            **self._extra_attributes,
        }
        # Propagate generic score metadata to span (first score as representative)
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

        # Create child span under root, add score events, end immediately.
        # _root_span is safe to read outside the lock: it is set once (write-once
        # invariant) inside _ensure_root_span and never reassigned after that.
        ctx = trace_api.set_span_in_context(self._root_span)
        item_span = self._tracer.start_span("eval-item", context=ctx, attributes=item_attrs)
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

        if root is not None:
            root.set_attribute("eval.summary.items_total", item_count)
            root.set_attribute("eval.summary.items_passed", items_passed)
            root.set_attribute("eval.summary.status", "completed")
            # Per-metric averages as summary event
            summary_attrs: dict[str, Any] = {}
            for name, values in score_values.items():
                summary_attrs[f"eval.summary.{name}.mean"] = round(sum(values) / len(values), 4)
            if summary_attrs:
                root.add_event("eval.summary", attributes=summary_attrs)
            root.set_status(StatusCode.OK)
            root.end()

        self._trace_provider.force_flush()
        self._meter_provider.force_flush()

    def shutdown(self) -> None:
        """Finalize (if needed), then release resources for both providers."""
        self.finalize()
        self._meter_provider.shutdown()
        self._trace_provider.shutdown()
