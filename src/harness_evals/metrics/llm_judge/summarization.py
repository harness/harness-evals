"""Summarization metric — evaluates factual correctness and detail coverage of summaries."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_ALIGNMENT_PROMPT = """You are an expert evaluator assessing summary accuracy.

**Original text**: {input}
**Summary**: {output}

Does the summary contain any hallucinated or contradictory information that is NOT in the original text?

Respond with JSON:
{{"reasoning": "analysis of factual alignment", "score": <float 0.0-1.0 where 1.0 means perfectly aligned with no hallucinations>}}
"""

_COVERAGE_PROMPT = """You are an expert evaluator assessing summary completeness.

**Original text**: {input}
**Summary**: {output}

For each of the following questions about the original text, determine if the summary preserves this information:

{questions_text}

Respond with JSON:
{{"answers": [{{"question": "q", "covered": true/false}}], "score": <float 0.0-1.0 representing fraction of key information preserved>}}
"""

_GENERATE_QUESTIONS_PROMPT = """You are an expert evaluator. Generate {n} yes/no questions about the key information in the following text. These questions should test whether a summary preserves the most important details.

**Text**: {input}

Respond with JSON:
{{"questions": ["question 1?", "question 2?", ...]}}
"""

_SCORE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

_COVERAGE_SCHEMA = {
    "type": "object",
    "required": ["answers", "score"],
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"question": {"type": "string"}, "covered": {"type": "boolean"}},
            },
        },
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

_QUESTIONS_SCHEMA = {
    "type": "object",
    "required": ["questions"],
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
}


class SummarizationMetric(BaseMetric):
    """Evaluates whether a summary is factually correct and includes key details.

    Score = min(alignment_score, coverage_score).
    - Alignment: does the summary hallucinate or contradict the source?
    - Coverage: does the summary preserve important information?

    The ``input`` field should contain the original text, and ``output`` the summary.
    """

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 0.5,
        n: int = 5,
        assessment_questions: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(name="summarization", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.llm = llm
        self.n = n
        self.assessment_questions = assessment_questions

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        input_text = str(eval_case.input)
        output_text = str(eval_case.output)

        # Step 1: Alignment score (hallucination check)
        alignment_prompt = _ALIGNMENT_PROMPT.format(input=input_text, output=output_text)
        alignment_result = await self.llm.generate_json(alignment_prompt, _SCORE_SCHEMA)
        alignment_score = max(0.0, min(1.0, float(alignment_result.get("score", 0.0))))
        alignment_reasoning = alignment_result.get("reasoning", "")

        # Step 2: Generate assessment questions if not provided
        questions = self.assessment_questions
        if not questions:
            gen_prompt = _GENERATE_QUESTIONS_PROMPT.format(n=self.n, input=input_text)
            gen_result = await self.llm.generate_json(gen_prompt, _QUESTIONS_SCHEMA)
            questions = gen_result.get("questions", [])

        # Step 3: Coverage score
        if not questions:
            coverage_score = 1.0
            coverage_reasoning = "No assessment questions generated"
        else:
            questions_text = "\n".join(f"- {q}" for q in questions)
            coverage_prompt = _COVERAGE_PROMPT.format(
                input=input_text, output=output_text, questions_text=questions_text
            )
            coverage_result = await self.llm.generate_json(coverage_prompt, _COVERAGE_SCHEMA)
            coverage_score = max(0.0, min(1.0, float(coverage_result.get("score", 0.0))))
            coverage_reasoning = str(coverage_result.get("answers", []))

        # Final score = min of both
        value = min(alignment_score, coverage_score)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"alignment={alignment_score:.2f} ({alignment_reasoning}), coverage={coverage_score:.2f}",
            metadata={
                "alignment_score": alignment_score,
                "coverage_score": coverage_score,
                "questions": questions,
                "alignment_reasoning": alignment_reasoning,
                "coverage_detail": coverage_reasoning,
            },
        )
