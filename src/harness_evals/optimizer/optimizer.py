"""Automated prompt optimizer — iteratively rewrites prompts to maximize metric scores."""

from __future__ import annotations

import asyncio
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.metric import BaseMetric
from harness_evals.core.runner import a_evaluate
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM
from harness_evals.prompts.template import PromptTemplate

_DIAGNOSE_PROMPT = PromptTemplate(
    template=(
        "You are an expert prompt engineer. A prompt was tested against a dataset and some cases failed.\n\n"
        "Current prompt:\n{{prompt}}\n\n"
        "Failing cases (input → expected vs actual output):\n{{failures}}\n\n"
        "Identify the common failure patterns. Be specific about what the prompt is missing or "
        "getting wrong. Respond with a concise diagnosis (3-5 bullet points)."
    ),
    input_variables=["prompt", "failures"],
)

_REWRITE_PROMPT = PromptTemplate(
    template=(
        "You are an expert prompt engineer. Rewrite the prompt below to fix the diagnosed issues.\n\n"
        "Current prompt:\n{{prompt}}\n\n"
        "Diagnosis of failures:\n{{diagnosis}}\n\n"
        "Generate exactly {{num_candidates}} distinct improved prompt variants. "
        "Each variant should address the diagnosed issues differently.\n"
        "The prompt uses {{placeholders}} as input placeholders — preserve them."
    ),
    input_variables=["prompt", "diagnosis", "num_candidates", "placeholders"],
)


@dataclass
class OptimizationResult:
    """Result of a prompt optimization run.

    ``score_history`` and ``prompt_history`` record the running best score and
    prompt at the end of each iteration (both monotonically track the best seen
    so far). ``candidate_scores`` records the raw best-candidate score per
    iteration — use this to diagnose regressions or oscillation.
    Use ``initial_score`` for the pre-optimization baseline.
    """

    best_prompt: PromptTemplate
    best_score: float
    initial_score: float
    score_history: list[float]
    candidate_scores: list[float]
    prompt_history: list[PromptTemplate]
    iterations: int
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_prompt": self.best_prompt.template,
            "best_score": self.best_score,
            "initial_score": self.initial_score,
            "score_history": self.score_history,
            "candidate_scores": self.candidate_scores,
            "prompt_history": [p.template for p in self.prompt_history],
            "iterations": self.iterations,
            "converged": self.converged,
        }

    def save(self, path: str) -> None:
        """Persist the optimization result as a JSON file (JsonSink-compatible format)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
            f.write("\n")


@dataclass
class _EvalResult:
    score: float
    failures: list[dict[str, str]] = field(default_factory=list)


class PromptOptimizer:
    """Iteratively rewrites a prompt to maximize metric performance.

    Uses a separate judge LLM to diagnose failures and generate candidate
    rewrites. ``model`` and ``judge`` must be different object instances
    (identity check via ``is``; two logically-equivalent instances with the
    same config will pass the check).

    ``max_concurrency`` caps total simultaneous ``model.generate()`` calls
    across all concurrent evaluations (including parallel candidate evaluation)
    via a shared semaphore.
    """

    def __init__(
        self,
        *,
        model: BaseLLM,
        judge: BaseLLM,
        metrics: list[BaseMetric],
        target_score: float = 0.85,
        max_iterations: int = 10,
        num_candidates: int = 3,
        patience: int = 3,
        max_concurrency: int | None = 10,
        diagnose_prompt: PromptTemplate | None = None,
        rewrite_prompt: PromptTemplate | None = None,
    ) -> None:
        if model is judge:
            raise ValueError(
                "model and judge must be different BaseLLM instances (identity check) — "
                "using the same model to evaluate itself produces biased rewrites"
            )
        if not metrics:
            raise ValueError("metrics must not be empty")
        if not (0.0 < target_score <= 1.0):
            raise ValueError("target_score must be in (0.0, 1.0]")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        if patience < 1:
            raise ValueError("patience must be >= 1")

        self._model = model
        self._judge = judge
        self._metrics = metrics
        self._target_score = target_score
        self._max_iterations = max_iterations
        self._num_candidates = num_candidates
        self._patience = patience
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        self._diagnose_prompt = diagnose_prompt or _DIAGNOSE_PROMPT
        self._rewrite_prompt = rewrite_prompt or _REWRITE_PROMPT

    async def optimize(
        self,
        prompt: PromptTemplate,
        goldens: list[Golden],
        on_iteration: Callable[[int, float, PromptTemplate], None] | None = None,
    ) -> OptimizationResult:
        """Run the optimization loop and return the best prompt found.

        ``on_iteration`` is called at the end of each iteration with
        (iteration_number, best_score_so_far, best_prompt_so_far).
        """
        if not goldens:
            raise ValueError("goldens must not be empty")

        current_prompt = prompt
        score_history: list[float] = []
        candidate_scores: list[float] = []
        prompt_history: list[PromptTemplate] = []

        initial_result = await self._evaluate_prompt(current_prompt, goldens)
        initial_score = initial_result.score
        best_score = initial_score
        best_prompt = current_prompt

        if best_score >= self._target_score:
            return OptimizationResult(
                best_prompt=best_prompt,
                best_score=best_score,
                initial_score=initial_score,
                score_history=[],
                candidate_scores=[],
                prompt_history=[],
                iterations=0,
                converged=True,
            )

        no_improvement_count = 0
        current_eval = initial_result
        fallback_count = 0

        for iteration in range(self._max_iterations):
            if not current_eval.failures:
                break

            diagnosis = await self._diagnose(current_prompt, current_eval.failures)
            candidates = await self._rewrite(current_prompt, diagnosis)

            if len(candidates) == 1 and candidates[0] is current_prompt:
                fallback_count += 1
                if fallback_count >= 3:
                    warnings.warn(
                        "Judge repeatedly failed to produce valid candidates; stopping early",
                        stacklevel=2,
                    )
                    break

            candidate_results = await asyncio.gather(*[self._evaluate_prompt(c, goldens) for c in candidates])

            best_candidate_idx = max(range(len(candidate_results)), key=lambda i: candidate_results[i].score)
            best_candidate_score = candidate_results[best_candidate_idx].score
            best_candidate = candidates[best_candidate_idx]

            if best_candidate_score > best_score:
                best_score = best_candidate_score
                best_prompt = best_candidate
                current_prompt = best_candidate
                current_eval = candidate_results[best_candidate_idx]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            candidate_scores.append(best_candidate_score)
            score_history.append(best_score)
            prompt_history.append(best_prompt)

            if on_iteration is not None:
                on_iteration(iteration + 1, best_score, best_prompt)

            if best_score >= self._target_score:
                return OptimizationResult(
                    best_prompt=best_prompt,
                    best_score=best_score,
                    initial_score=initial_score,
                    score_history=score_history,
                    candidate_scores=candidate_scores,
                    prompt_history=prompt_history,
                    iterations=iteration + 1,
                    converged=True,
                )

            if no_improvement_count >= self._patience:
                break

        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            initial_score=initial_score,
            score_history=score_history,
            candidate_scores=candidate_scores,
            prompt_history=prompt_history,
            iterations=len(score_history),
            converged=False,
        )

    @staticmethod
    def _build_render_kwargs(prompt: PromptTemplate, golden: Golden) -> dict[str, str]:
        """Map a golden's input to the prompt's variables.

        - If golden.input is a dict, use it directly (must cover all variables).
        - If the prompt has one variable, map the string input to it.
        - For multi-variable prompts with string input, map to the first variable
          and fill remaining variables from golden.context (positionally).
        """
        if isinstance(golden.input, dict):
            missing = set(prompt.input_variables) - golden.input.keys()
            if missing:
                raise ValueError(
                    f"golden.input dict is missing variables required by the prompt: {sorted(missing)}"
                )
            return {k: str(v) for k, v in golden.input.items()}

        input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
        variables = prompt.input_variables

        if len(variables) == 1:
            return {variables[0]: input_str}

        kwargs: dict[str, str] = {variables[0]: input_str}
        context = golden.context or []
        extra_vars = variables[1:]
        if len(context) < len(extra_vars):
            warnings.warn(
                f"golden.context has {len(context)} items but prompt needs "
                f"{len(extra_vars)} extra variables {extra_vars[len(context):]}; "
                f"filling with empty strings",
                stacklevel=3,
            )
        for i, var in enumerate(extra_vars):
            kwargs[var] = context[i] if i < len(context) else ""
        return kwargs

    async def _evaluate_prompt(self, prompt: PromptTemplate, goldens: list[Golden]) -> _EvalResult:
        """Render prompt for each golden, call model, score with metrics."""

        async def _run_one(golden: Golden) -> tuple[float, dict[str, str] | None]:
            render_kwargs = self._build_render_kwargs(prompt, golden)
            input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
            rendered = prompt.render(**render_kwargs)

            if self._semaphore:
                async with self._semaphore:
                    output = await self._model.generate(rendered)
            else:
                output = await self._model.generate(rendered)

            eval_case = EvalCase(
                input=input_str,
                output=output,
                expected=golden.expected,
                context=golden.context,
            )
            scores: list[Score] = await a_evaluate(eval_case, self._metrics)
            avg = sum(s.value for s in scores) / len(scores) if scores else 0.0

            failure_info = None
            if avg < self._target_score and golden.expected is not None:
                expected_str = golden.expected if isinstance(golden.expected, str) else str(golden.expected)
                failure_info = {
                    "input": input_str,
                    "expected": expected_str,
                    "actual": output,
                }
            return avg, failure_info

        results = await asyncio.gather(*[_run_one(g) for g in goldens])
        total_score = sum(r[0] for r in results) / len(results)
        failures = [r[1] for r in results if r[1] is not None]

        return _EvalResult(score=total_score, failures=failures)

    async def _diagnose(self, prompt: PromptTemplate, failures: list[dict[str, str]]) -> str:
        """Ask the judge to identify failure patterns."""
        failures_text = "\n".join(
            f"- Input: {f['input']}\n  Expected: {f['expected']}\n  Got: {f['actual']}"
            for f in failures[:10]  # cap to avoid token overflow
        )
        diagnosis_prompt = self._diagnose_prompt.render(prompt=prompt.template, failures=failures_text)
        return await self._judge.generate(diagnosis_prompt)

    async def _rewrite(self, prompt: PromptTemplate, diagnosis: str) -> list[PromptTemplate]:
        """Ask the judge to produce candidate rewrites."""
        placeholders = ", ".join(f"{{{{{v}}}}}" for v in prompt.input_variables)
        rewrite_rendered = self._rewrite_prompt.render(
            prompt=prompt.template,
            diagnosis=diagnosis,
            num_candidates=str(self._num_candidates),
            placeholders=placeholders,
        )
        result = await self._judge.generate_json(
            rewrite_rendered,
            schema={
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["candidates"],
            },
        )

        candidates: list[PromptTemplate] = []
        for template_str in result.get("candidates", [])[: self._num_candidates]:
            if not isinstance(template_str, str) or not template_str.strip():
                warnings.warn(f"Skipping empty/non-string candidate: {template_str!r}", stacklevel=2)
                continue
            try:
                candidates.append(
                    PromptTemplate(
                        template=template_str,
                        input_variables=list(prompt.input_variables),
                    )
                )
            except (ValueError, TypeError) as e:
                warnings.warn(f"Skipping malformed candidate: {e}", stacklevel=2)
                continue

        if not candidates:
            warnings.warn("All candidates were malformed; falling back to current prompt", stacklevel=2)
            candidates.append(prompt)

        return candidates
