"""Importer adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from harness_evals.core.eval_case import EvalCase
from harness_evals.refs import ResourceRef

if TYPE_CHECKING:
    from harness_evals.config.schema import EvalConfig


class BaseEvalCaseSource(ABC):
    """Fetch already-produced eval cases from an external platform (traces, runs, etc.).

    The output is a list of :class:`~harness_evals.core.eval_case.EvalCase` objects that
    are ready to be scored with ``evaluate_cases()``.

    Concrete subclasses **must** define a class-level ``name: str`` attribute (e.g.
    ``name = "langfuse"``). This is enforced at class-creation time for non-abstract
    subclasses.

    Example::

        from harness_evals.importers.langfuse import LangfuseEvalCaseSource
        from harness_evals.refs import resolve
        from harness_evals import evaluate_cases
        from harness_evals.metrics import LatencyMetric

        source = LangfuseEvalCaseSource(client)
        cases = await source.fetch(resolve("langfuse://trace-abc-123"))
        scores = evaluate_cases(cases, metrics=[LatencyMetric(max_ms=3000)])
    """

    name: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None) and not isinstance(cls.__dict__.get("name"), str):
            raise TypeError(f"{cls.__name__} must define a class-level 'name: str' attribute")

    @abstractmethod
    async def fetch(self, ref: ResourceRef) -> list[EvalCase]:
        """Return eval cases identified by ``ref``."""

    async def close(self) -> None:
        """Release any adapter-owned resources."""
        return None

    async def __aenter__(self) -> BaseEvalCaseSource:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class BaseEvalConfigSource(ABC):
    """Translate a platform-native eval definition into an :class:`EvalConfig`.

    The output is an ``EvalConfig`` ready to be executed with ``run_config()``.
    Translation is best-effort; platform metrics with no local catalog equivalent
    raise :class:`~harness_evals.errors.UnmappedMetricError`.

    Concrete subclasses **must** define a class-level ``name: str`` attribute (e.g.
    ``name = "harness"``). This is enforced at class-creation time for non-abstract
    subclasses.

    Example::

        from harness_evals.importers.harness import HarnessEvalConfigSource
        from harness_evals.refs import resolve
        from harness_evals.config.runner import run_config

        source = HarnessEvalConfigSource()
        cfg = await source.fetch(resolve("harness://evals/my-eval@2"))
        run_config(cfg)
    """

    name: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None) and not isinstance(cls.__dict__.get("name"), str):
            raise TypeError(f"{cls.__name__} must define a class-level 'name: str' attribute")

    @abstractmethod
    async def fetch(self, ref: ResourceRef) -> EvalConfig:
        """Return an :class:`EvalConfig` translated from the platform eval at ``ref``."""

    async def close(self) -> None:
        """Release any adapter-owned resources."""
        return None

    async def __aenter__(self) -> BaseEvalConfigSource:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
