"""Extraction synthesizer — generates "extract all X" tasks from documents."""

from __future__ import annotations

from harness_evals.core.golden import Golden
from harness_evals.synthesizer.base import BaseSynthesizer

__all__ = ["ExtractionSynthesizer"]

_PROMPT_TEMPLATE = """\
You are an expert at creating evaluation datasets for AI systems.
Given the following document, generate exactly {n} information extraction tasks.
Each task provides a text passage and the expected extracted information \
(entities, facts, or structured fields).

**Difficulty level**: {difficulty}
- easy: Extract a single, clearly stated entity or fact from the text.
- medium: Extract multiple related entities or facts from a paragraph.
- hard: Extract and structure complex, interrelated information from the text.
- mixed: A mix of easy, medium, and hard tasks.

**Document**:
{chunk}

For each task, select a passage and specify what information should be extracted.

Respond with JSON:
{{"pairs": [{{"text": "the source text", "extracted": "the expected extracted information", "difficulty": "easy|medium|hard"}}]}}
"""

_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "required": ["pairs"],
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "extracted", "difficulty"],
                "properties": {
                    "text": {"type": "string"},
                    "extracted": {"type": "string"},
                    "difficulty": {"type": "string"},
                },
            },
        }
    },
}


class ExtractionSynthesizer(BaseSynthesizer):
    """Generate information extraction golden pairs from source documents."""

    task_type = "extraction"

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
            text = pair.get("text", "")
            extracted = pair.get("extracted", "")
            if not text or not extracted:
                continue
            goldens.append(
                Golden(
                    input=text,
                    expected=extracted,
                    metadata={
                        "task_type": self.task_type,
                        "difficulty": pair.get("difficulty", difficulty),
                        "source_document_index": doc_index,
                    },
                )
            )
        return goldens
