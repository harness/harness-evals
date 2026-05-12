"""Dataset synthesizer — generate Golden datasets from source documents."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.datasets import Dataset
from harness_evals.llm.base import BaseLLM
from harness_evals.synthesizer.base import BaseSynthesizer
from harness_evals.synthesizer.conversation import ConversationSynthesizer, ScriptedConversationSynthesizer
from harness_evals.synthesizer.extraction import ExtractionSynthesizer
from harness_evals.synthesizer.qa import QASynthesizer
from harness_evals.synthesizer.structured import StructuredOutputSynthesizer
from harness_evals.synthesizer.summarization import SummarizationSynthesizer

_GENERATORS: dict[str, type[BaseSynthesizer]] = {
    "qa": QASynthesizer,
    "summarization": SummarizationSynthesizer,
    "extraction": ExtractionSynthesizer,
    "structured_output": StructuredOutputSynthesizer,
    "conversation": ConversationSynthesizer,
    "conversation_scripted": ScriptedConversationSynthesizer,
}


class Synthesizer:
    """High-level facade for generating golden datasets from documents.

    Dispatches to the appropriate task-type generator based on
    ``task_type``.

    Example::

        from harness_evals.synthesizer import Synthesizer
        from harness_evals.llm import OpenAILLM

        synth = Synthesizer(llm=OpenAILLM())
        goldens = await synth.generate(
            documents=[open("docs/guide.md").read()],
            n=30,
            task_type="qa",
        )
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_chunk_chars: int = 4000,
        batch_size: int = 10,
    ) -> None:
        self.llm = llm
        self._max_chunk_chars = max_chunk_chars
        self._batch_size = batch_size

    async def generate(
        self,
        documents: list[str],
        n: int = 20,
        task_type: str = "qa",
        difficulty: str = "mixed",
    ) -> list:
        """Generate *n* goldens from *documents* for the given *task_type*.

        Args:
            documents: Source document strings to generate from.
            n: Number of goldens to generate.
            task_type: One of ``"qa"``, ``"summarization"``,
                ``"extraction"``, ``"structured_output"``,
                ``"conversation"``, ``"conversation_scripted"``.
                Conversation task types return ``list[ConversationGolden]``
                instead of ``Dataset`` (``list[Golden]``).
            difficulty: One of ``"easy"``, ``"medium"``, ``"hard"``,
                ``"mixed"``.

        Returns:
            A ``Dataset`` (``list[Golden]``) for standard task types, or
            ``list[ConversationGolden]`` for conversation task types.
        """
        if task_type not in _GENERATORS:
            raise ValueError(f"Unknown task_type {task_type!r}. Choose from: {', '.join(sorted(_GENERATORS))}")

        generator = _GENERATORS[task_type](
            self.llm,
            max_chunk_chars=self._max_chunk_chars,
            batch_size=self._batch_size,
        )
        return await generator.generate(documents, n, difficulty)

    def generate_sync(
        self,
        documents: list[str],
        n: int = 20,
        task_type: str = "qa",
        difficulty: str = "mixed",
    ) -> Dataset:
        """Synchronous wrapper around :meth:`generate`."""
        return _run_async(self.generate(documents, n, task_type, difficulty))


__all__ = [
    "Synthesizer",
    "BaseSynthesizer",
    "QASynthesizer",
    "SummarizationSynthesizer",
    "ExtractionSynthesizer",
    "StructuredOutputSynthesizer",
    "ConversationSynthesizer",
    "ScriptedConversationSynthesizer",
]
