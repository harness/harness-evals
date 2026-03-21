"""QA synthesizer — generates question-answer pairs from documents."""

from __future__ import annotations

from harness_evals.core.golden import Golden
from harness_evals.synthesizer.base import BaseSynthesizer

__all__ = ["QASynthesizer"]

_PROMPT_TEMPLATE = """\
You are an expert at creating evaluation datasets for AI systems.
Given the following document, generate exactly {n} question-answer pairs.

**Difficulty level**: {difficulty}
- easy: Simple factual questions answerable from a single sentence in the document.
- medium: Questions requiring understanding of a full paragraph.
- hard: Questions requiring reasoning across multiple parts of the document.
- mixed: A mix of easy, medium, and hard questions.

**Document**:
{chunk}

Generate exactly {n} question-answer pairs. Each answer should be concise and \
directly supported by the document.

Respond with JSON:
{{"pairs": [{{"question": "the question", "answer": "the answer", "difficulty": "easy|medium|hard"}}]}}
"""

_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "required": ["pairs"],
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["question", "answer", "difficulty"],
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "difficulty": {"type": "string"},
                },
            },
        }
    },
}


class QASynthesizer(BaseSynthesizer):
    """Generate question-answer golden pairs from source documents."""

    task_type = "qa"

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
            question = pair.get("question", "")
            answer = pair.get("answer", "")
            if not question or not answer:
                continue
            goldens.append(
                Golden(
                    input=question,
                    expected=answer,
                    metadata={
                        "task_type": self.task_type,
                        "difficulty": pair.get("difficulty", difficulty),
                        "source_document_index": doc_index,
                    },
                )
            )
        return goldens
