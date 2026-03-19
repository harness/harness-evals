"""Summarization synthesizer — generates "summarize this" tasks from documents."""

from __future__ import annotations

from harness_evals.core.golden import Golden
from harness_evals.synthesizer.base import BaseSynthesizer

__all__ = ["SummarizationSynthesizer"]

_PROMPT_TEMPLATE = """\
You are an expert at creating evaluation datasets for AI systems.
Given the following document, generate exactly {n} summarization tasks.
Each task consists of a passage and its expected summary (key points).

**Difficulty level**: {difficulty}
- easy: Short, single-paragraph passage with a straightforward summary.
- medium: Multi-paragraph passage requiring synthesis of main ideas.
- hard: Complex passage requiring identification of nuanced arguments and relationships.
- mixed: A mix of easy, medium, and hard tasks.

**Document**:
{chunk}

For each task, select a meaningful passage from the document and write a concise \
summary capturing the key points.

Respond with JSON:
{{"pairs": [{{"passage": "the passage to summarize", "summary": "the expected summary", "difficulty": "easy|medium|hard"}}]}}
"""

_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "required": ["pairs"],
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["passage", "summary", "difficulty"],
                "properties": {
                    "passage": {"type": "string"},
                    "summary": {"type": "string"},
                    "difficulty": {"type": "string"},
                },
            },
        }
    },
}


class SummarizationSynthesizer(BaseSynthesizer):
    """Generate summarization golden pairs from source documents."""

    task_type = "summarization"

    def _build_prompt(self, chunk: str, n: int, difficulty: str) -> str:
        return _PROMPT_TEMPLATE.format(n=n, difficulty=difficulty, chunk=chunk)

    def _response_schema(self) -> dict:
        return _RESPONSE_SCHEMA

    def _parse_response(
        self,
        response: dict,
        doc_index: int,
        difficulty: str,
    ) -> list[Golden]:
        pairs = response.get("pairs", [])
        goldens: list[Golden] = []
        for pair in pairs:
            passage = pair.get("passage", "")
            summary = pair.get("summary", "")
            if not passage or not summary:
                continue
            goldens.append(
                Golden(
                    input=passage,
                    expected=summary,
                    metadata={
                        "task_type": self.task_type,
                        "difficulty": pair.get("difficulty", difficulty),
                        "source_document_index": doc_index,
                    },
                )
            )
        return goldens
