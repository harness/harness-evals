"""OTLP sink — export scores as OpenTelemetry metrics to an OTLP-compatible backend."""

from __future__ import annotations

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
except ImportError as _err:
    raise ImportError(
        "OtlpSink requires the opentelemetry packages. Install them with: pip install harness-evals[otlp]"
    ) from _err

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink


class OtlpSink(BaseSink):
    """Export scores as OpenTelemetry gauge metrics to an OTLP endpoint.

    Each ``Score`` becomes a gauge observation with attributes for
    ``metric_name``, ``threshold``, ``passed``, and any ``eval_case.tags``.

    Requires ``pip install harness-evals[otlp]``.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        service_name: str = "harness-evals",
        insecure: bool = True,
    ) -> None:
        self.endpoint = endpoint
        self.service_name = service_name

        exporter = OTLPMetricExporter(endpoint=endpoint, insecure=insecure)
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=1000)
        self._provider = MeterProvider(metric_readers=[reader])

        meter = self._provider.get_meter(service_name)
        self._gauge = meter.create_gauge(
            name="harness_evals.score",
            description="Evaluation metric score",
            unit="ratio",
        )

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        tags = eval_case.tags or {}
        for score in scores:
            attributes = {
                "metric_name": score.name,
                "threshold": str(score.threshold),
                "passed": str(score.passed),
                **{f"tag.{k}": v for k, v in tags.items()},
            }
            self._gauge.set(score.value, attributes=attributes)

    def shutdown(self) -> None:
        """Flush pending metrics and shut down the meter provider."""
        self._provider.shutdown()
