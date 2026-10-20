from harness_evals.sinks.csv_sink import CsvSink
from harness_evals.sinks.json_sink import JsonSink
from harness_evals.sinks.junit_sink import JUnitSink
from harness_evals.sinks.stdout import StdoutSink

# OtlpSink is intentionally excluded — it requires the heavy `opentelemetry`
# optional dependency (pip install harness-evals[otlp]).  Import it directly:
#   from harness_evals.sinks.otlp_sink import OtlpSink
#
# LangfuseSink is also excluded — requires `langfuse` (pip install harness-evals[langfuse]):
#   from harness_evals.sinks.langfuse_sink import LangfuseSink
#
# HarnessSink is also excluded — requires `httpx` (pip install harness-evals[harness]):
#   from harness_evals.sinks.harness_sink import HarnessSink
__all__ = ["StdoutSink", "JsonSink", "CsvSink", "JUnitSink"]
