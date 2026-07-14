"""Base classes for academic benchmark evaluation."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.llm.base import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Aggregate result of running a benchmark suite."""

    name: str
    accuracy: float
    num_correct: int
    num_total: int
    scores: list[Score] = field(default_factory=list)
    eval_cases: list[EvalCase] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def pass_at_1(self) -> float:
        """Alias for accuracy, used by code generation benchmarks."""
        return self.accuracy

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "scores": [s.to_dict() for s in self.scores],
        }


class BaseBenchmark(ABC):
    """Abstract base for academic benchmarks.

    Subclasses implement dataset loading, prompt formatting, and scoring.
    The ``run()`` method orchestrates evaluation with concurrency control
    and sink integration.
    """

    name: str
    default_shots: int = 0

    def __init__(self, name: str, *, default_shots: int = 0) -> None:
        self.name = name
        self.default_shots = default_shots

    @abstractmethod
    async def load_dataset(self, *, offline: bool = False) -> list[dict]:
        """Load benchmark dataset items. Each item is a dict with benchmark-specific fields."""
        ...

    @abstractmethod
    def format_prompt(self, item: dict, *, shots: int = 0) -> str:
        """Format a dataset item into a prompt string for the model."""
        ...

    @abstractmethod
    def score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        """Score a model response against the expected answer.

        Returns (value, reason) where value is 0.0 or 1.0 for accuracy-based benchmarks.
        """
        ...

    async def a_score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        """Async scoring hook. Override for benchmarks that need async scoring (e.g. LLM judge).

        Default delegates to synchronous score_response().
        """
        return self.score_response(item, response)

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
        """Run the benchmark end-to-end.

        Args:
            model: LLM to evaluate.
            sinks: Output sinks to write results to.
            shots: Number of few-shot examples. None uses the benchmark's default.
            limit: Max items to evaluate (None = full dataset).
            offline: If True, only use cached data (no network).
            concurrency: Max concurrent LLM calls (must be >= 1).
        """
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

                score = Score(name=self.name, value=value, threshold=0.5, reason=reason)
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
                score = Score(name=self.name, value=0.0, threshold=0.5, reason=f"Error: {raw}")
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
        metrics = self._compute_metrics(scores)
        metrics["accuracy"] = accuracy

        return BenchmarkResult(
            name=self.name,
            accuracy=accuracy,
            num_correct=num_correct,
            num_total=len(dataset),
            scores=scores,
            eval_cases=eval_cases,
            metadata=self._get_result_metadata(),
            metrics=metrics,
        )

    def _get_expected(self, item: dict) -> str | None:
        """Extract expected answer from a dataset item. Override for custom behavior."""
        return item.get("answer") or item.get("expected")

    def _get_item_metadata(self, item: dict) -> dict[str, Any] | None:
        """Extract metadata from a dataset item for the EvalCase."""
        return None

    def _get_result_metadata(self) -> dict[str, Any]:
        """Return benchmark-level metadata for the BenchmarkResult."""
        return {}

    def _compute_metrics(self, scores: list[Score]) -> dict[str, float]:
        """Compute aggregate metrics from scores. Override for benchmark-specific metrics."""
        return {}
