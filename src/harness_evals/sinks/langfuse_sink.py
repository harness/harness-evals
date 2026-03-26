"""Langfuse sink — send evaluation scores to Langfuse for observability."""

from __future__ import annotations

import os

try:
    from langfuse import Langfuse
except ImportError as _err:
    raise ImportError(
        "LangfuseSink requires the langfuse package. Install with: pip install harness-evals[langfuse]"
    ) from _err

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink


class LangfuseSink(BaseSink):
    """Send evaluation scores to Langfuse.

    Each ``Score`` is recorded as a Langfuse score observation. Trace linking
    is supported via ``eval_case.metadata``:

    - ``langfuse_trace_id``: attach scores to an existing Langfuse trace
    - ``langfuse_observation_id``: attach to a specific span within the trace

    If no ``trace_id`` is provided, scores are recorded without trace linkage.

    Key resolution: constructor params > environment variables.

    Requires ``pip install harness-evals[langfuse]``.

    Example::

        from harness_evals.sinks.langfuse_sink import LangfuseSink

        sink = LangfuseSink()
        evaluate(cases, metrics=[...], sinks=[sink])
        sink.shutdown()
    """

    def __init__(
        self,
        secret_key: str | None = None,
        public_key: str | None = None,
        host: str | None = None,
    ) -> None:
        self._client = Langfuse(
            secret_key=secret_key or os.environ.get("LANGFUSE_SECRET_KEY", ""),
            public_key=public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            host=host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        meta = eval_case.metadata or {}
        trace_id = meta.get("langfuse_trace_id")
        observation_id = meta.get("langfuse_observation_id")
        tags = eval_case.tags or {}

        for score in scores:
            score_meta = {**(score.metadata or {}), **{f"tag.{k}": v for k, v in tags.items()}}

            kwargs: dict = {
                "name": score.name,
                "value": score.value,
                "comment": score.reason,
                "metadata": score_meta if score_meta else None,
            }
            if trace_id:
                kwargs["trace_id"] = trace_id
            if observation_id:
                kwargs["observation_id"] = observation_id

            self._client.score(**kwargs)

    def finalize(self) -> None:
        """Flush buffered scores to Langfuse."""
        self._client.flush()

    def shutdown(self) -> None:
        """Flush and shut down the Langfuse client."""
        self._client.flush()
        self._client.shutdown()
