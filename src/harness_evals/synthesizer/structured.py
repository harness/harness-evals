"""Structured output synthesizer — generates "produce JSON/YAML for X" tasks."""

from __future__ import annotations

import json
import logging

from harness_evals.core.golden import Golden
from harness_evals.synthesizer.base import BaseSynthesizer

logger = logging.getLogger(__name__)

__all__ = ["StructuredOutputSynthesizer"]


def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM text that may contain markdown fences or trailing text."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
        text = text.strip()
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text, start)
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("Expected JSON object, got " + type(obj).__name__, text, start)
    return obj


_PROMPT_TEMPLATE = """\
You are an expert at creating evaluation datasets for AI systems.
Given the following document as domain context, generate exactly {n} structured \
output tasks.  Each task describes what JSON to produce and provides the expected \
JSON output.

**Difficulty level**: {difficulty}
- easy: Simple flat JSON object with 2-3 fields.
- medium: Nested JSON with arrays or objects, 4-6 fields total.
- hard: Complex, deeply nested JSON with conditional fields and relationships.
- mixed: A mix of easy, medium, and hard tasks.

**Document (domain context)**:
{chunk}

For each task, write a natural-language prompt requesting a specific JSON output, \
and provide the correct expected JSON object.

Respond with JSON:
{{"pairs": [{{"prompt": "description of the JSON to generate", \
"expected_output": {{"key": "value"}}, \
"difficulty": "easy|medium|hard"}}]}}
"""

_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "required": ["pairs"],
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["prompt", "expected_output", "difficulty"],
                "properties": {
                    "prompt": {"type": "string"},
                    "expected_output": {"type": "object"},
                    "difficulty": {"type": "string"},
                },
            },
        }
    },
}


class StructuredOutputSynthesizer(BaseSynthesizer):
    """Generate structured output golden pairs from source documents.

    Unlike other synthesizers, ``expected`` is a ``dict`` (not a string).
    The other synthesizers use ``generate_json()`` with strict JSON schemas
    for reliable structured output.  That approach doesn't work here
    because ``expected_output`` is a free-form JSON object whose keys
    are not known ahead of time.  Strict schema APIs (OpenAI, Anthropic)
    require every object to declare its properties with
    ``additionalProperties: false``, which is fundamentally incompatible
    with arbitrary JSON.  Therefore this synthesizer overrides
    ``_call_llm`` to use plain ``generate()`` and parses the JSON from
    the text response instead.
    """

    task_type = "structured_output"

    def _build_prompt(self, chunk: str, n: int, difficulty: str) -> str:
        return _PROMPT_TEMPLATE.format(n=n, difficulty=difficulty, chunk=chunk)

    def _response_schema(self) -> dict:
        # Not used at runtime (see _call_llm override), but satisfies the
        # abstract contract and documents the expected response shape.
        return _RESPONSE_SCHEMA

    async def _call_llm(self, prompt: str, schema: dict) -> dict:
        """Use plain ``generate()`` and parse JSON manually.

        Strict JSON schema APIs (OpenAI, Anthropic) reject free-form
        ``{"type": "object"}`` without declared properties, so we bypass
        structured output mode for this synthesizer.
        """
        text = await self.llm.generate(prompt)
        return _extract_json(text)

    def _parse_response(
        self,
        response: dict,
        doc_index: int,
        difficulty: str,
    ) -> list[Golden]:
        pairs = response.get("pairs", [])
        goldens: list[Golden] = []
        for pair in pairs:
            prompt = pair.get("prompt", "")
            expected_output = pair.get("expected_output")
            if not prompt or not expected_output:
                continue
            goldens.append(
                Golden(
                    input=prompt,
                    expected=expected_output,
                    metadata={
                        "task_type": self.task_type,
                        "difficulty": pair.get("difficulty", difficulty),
                        "source_document_index": doc_index,
                    },
                )
            )
        return goldens
