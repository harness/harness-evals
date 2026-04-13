"""Input generator — generate synthetic evaluation inputs from descriptions or seeds."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.datasets import Dataset
from harness_evals.input_generator.adversarial import AdversarialStrategy
from harness_evals.input_generator.base import BaseInputStrategy
from harness_evals.input_generator.complexity_ladder import ComplexityLadderStrategy
from harness_evals.input_generator.rephrase import RephraseStrategy
from harness_evals.input_generator.use_case import UseCaseStrategy
from harness_evals.llm.base import BaseLLM

_STRATEGIES: dict[str, type[BaseInputStrategy]] = {
    "use_case": UseCaseStrategy,
    "rephrase": RephraseStrategy,
    "adversarial": AdversarialStrategy,
    "complexity_ladder": ComplexityLadderStrategy,
}


class InputGenerator:
    """High-level facade for generating synthetic evaluation inputs.

    Dispatches to the appropriate strategy based on ``strategy``.

    Example::

        from harness_evals.input_generator import InputGenerator
        from harness_evals.llm.openai import OpenAILLM

        gen = InputGenerator(llm=OpenAILLM())
        goldens = await gen.generate(
            strategy="use_case",
            count=20,
            description="A DevOps assistant that creates CI/CD pipelines",
        )
    """

    def __init__(self, llm: BaseLLM, batch_size: int = 10) -> None:
        self.llm = llm
        self.batch_size = batch_size

    async def generate(
        self,
        strategy: str,
        count: int,
        description: str | None = None,
        seed_inputs: list[str] | None = None,
        **kwargs,
    ) -> Dataset:
        """Generate *count* synthetic inputs using the given *strategy*.

        Args:
            strategy: One of ``"use_case"``, ``"rephrase"``,
                ``"adversarial"``, ``"complexity_ladder"``.
            count: Number of inputs to generate.
            description: Use case description (required for use_case,
                adversarial, complexity_ladder).
            seed_inputs: Existing inputs to rephrase (required for rephrase).
            **kwargs: Strategy-specific options (e.g., ``levels`` for
                complexity_ladder).

        Returns:
            A ``Dataset`` (``list[Golden]``).
        """
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose from: {', '.join(sorted(_STRATEGIES))}"
            )

        impl = _STRATEGIES[strategy](self.llm, batch_size=self.batch_size)
        return await impl.generate(
            count=count,
            description=description,
            seed_inputs=seed_inputs,
            **kwargs,
        )

    def generate_sync(
        self,
        strategy: str,
        count: int,
        description: str | None = None,
        seed_inputs: list[str] | None = None,
        **kwargs,
    ) -> Dataset:
        """Synchronous wrapper around :meth:`generate`."""
        return _run_async(
            self.generate(strategy, count, description, seed_inputs, **kwargs)
        )


__all__ = [
    "InputGenerator",
    "BaseInputStrategy",
    "UseCaseStrategy",
    "RephraseStrategy",
    "AdversarialStrategy",
    "ComplexityLadderStrategy",
]
