"""Base class for AI security benchmark adapters."""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from typing import Any

from harness_evals.benchmarks._security_utils import compute_security_aggregate_metrics
from harness_evals.benchmarks.base import BaseBenchmark, BenchmarkResult
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class SecurityBenchmark(BaseBenchmark):
    """Base for security/red-team benchmarks.

    Score semantics: 1.0 = safe (attack blocked / refusal), 0.0 = unsafe.
    ``BenchmarkResult.accuracy`` is the safety pass rate.
    """

    DEFAULT_THRESHOLD = 1.0

    def __init__(self, name: str, *, default_shots: int = 0) -> None:
        super().__init__(name=name, default_shots=default_shots)
        self._category_field = "category"
        self._extra_aggregate_fields: list[str] = []

    @abstractmethod
    def _security_metadata(self, item: dict) -> dict[str, Any]:
        """Return normalized security taxonomy metadata for an item."""

    def item_to_golden(self, item: dict, *, shots: int = 0) -> Golden:
        """Convert a dataset item to a Golden with security taxonomy metadata."""
        return Golden(
            input=self.format_prompt(item, shots=shots),
            expected=self._get_expected(item),
            metadata=self._security_metadata(item),
        )

    async def load_goldens(self, *, offline: bool = False, shots: int = 0) -> list[Golden]:
        """Load dataset items as Golden objects with security metadata."""
        items = await self.load_dataset(offline=offline)
        return [self.item_to_golden(item, shots=shots) for item in items]

    def _make_security_score(self, value: float, reason: str | None) -> Score:
        return Score(
            name=self.name,
            value=value,
            threshold=self.DEFAULT_THRESHOLD,
            reason=reason,
            metadata={"dimension": "safety"},
        )

    def _get_item_metadata(self, item: dict) -> dict[str, Any] | None:
        return self._security_metadata(item)

    def _compute_security_metrics(
        self,
        scores: list[Score],
        eval_cases: list[EvalCase],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        return compute_security_aggregate_metrics(
            scores,
            eval_cases,
            category_field=self._category_field,
            extra_fields=self._extra_aggregate_fields,
        )

    async def run(
        self,
        model: BaseLLM,
        *,
        sinks: list[BaseSink] | None = None,
        shots: int | None = None,
        limit: int | None = None,
        offline: bool = False,
        concurrency: int = 10,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Run the security benchmark end-to-end."""
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")

        effective_shots = self.default_shots if shots is None else shots

        dataset = await self.load_dataset(offline=offline)
        if limit is not None:
            dataset = dataset[:limit]

        semaphore = asyncio.Semaphore(concurrency)
        scores: list[Score] = []
        eval_cases: list[EvalCase] = []
        num_correct = 0

        async def evaluate_item(item: dict) -> tuple[Score, EvalCase]:
            async with semaphore:
                prompt = self.format_prompt(item, shots=effective_shots)
                response = await model.generate(prompt, **kwargs)
                value, reason = await self.a_score_response(item, response)
                score = self._make_security_score(value, reason)
                eval_case = EvalCase(
                    input=prompt,
                    output=response,
                    expected=self._get_expected(item),
                    metadata=self._get_item_metadata(item),
                )
                return score, eval_case

        tasks = [evaluate_item(item) for item in dataset]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for raw in raw_results:
            if isinstance(raw, BaseException) and not isinstance(raw, Exception):
                raise raw
            if isinstance(raw, BaseException):
                logger.warning("Benchmark %s: item errored: %r", self.name, raw)
                score = self._make_security_score(0.0, f"Error: {raw}")
                eval_case = EvalCase(input="", output="", expected="")
                scores.append(score)
                eval_cases.append(eval_case)
            else:
                score, eval_case = raw
                scores.append(score)
                eval_cases.append(eval_case)
                if score.value >= 1.0:
                    num_correct += 1

        if sinks:
            for score, eval_case in zip(scores, eval_cases, strict=True):
                for sink in sinks:
                    sink.write([score], eval_case)
            for sink in sinks:
                sink.finalize()
                sink.shutdown()

        accuracy = num_correct / len(dataset) if dataset else 0.0
        security_metrics, nested_meta = self._compute_security_metrics(scores, eval_cases)
        extra = self._compute_metrics(scores)
        metrics = {**extra, **security_metrics, "accuracy": accuracy}

        result_metadata = self._get_result_metadata()
        result_metadata.update(nested_meta)

        return BenchmarkResult(
            name=self.name,
            accuracy=accuracy,
            num_correct=num_correct,
            num_total=len(dataset),
            scores=scores,
            eval_cases=eval_cases,
            metadata=result_metadata,
            metrics=metrics,
        )
