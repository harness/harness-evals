"""Tests for PromptOptimizer."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM
from harness_evals.optimizer import OptimizationResult, PromptOptimizer
from harness_evals.prompts.template import PromptTemplate


class StubModel(BaseLLM):
    """Model that returns a fixed response or cycles through responses."""

    def __init__(self, responses: list[str] | None = None, default: str = "output"):
        self._responses = list(responses) if responses else []
        self._default = default
        self._call_idx = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        if self._call_idx < len(self._responses):
            result = self._responses[self._call_idx]
            self._call_idx += 1
            return result
        return self._default

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return {}


class StubJudge(BaseLLM):
    """Judge that returns canned diagnosis and rewrites."""

    def __init__(self, candidates: list[list[str]] | None = None):
        self._candidates = candidates or []
        self._rewrite_idx = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        return "The prompt lacks specificity and does not instruct the model to be concise."

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        if self._rewrite_idx < len(self._candidates):
            result = self._candidates[self._rewrite_idx]
            self._rewrite_idx += 1
            return {"candidates": result}
        return {"candidates": ["Answer concisely: {{input}}"]}


class ScoreControlMetric(BaseMetric):
    """Metric whose score is controlled by a callable."""

    def __init__(self, score_fn):
        super().__init__(name="controlled", dimension=Dimension.CORRECTNESS, threshold=0.5)
        self._score_fn = score_fn

    def measure(self, eval_case: EvalCase) -> Score:
        value = self._score_fn(eval_case)
        return Score(name=self.name, value=value, threshold=self.threshold)


class FixedScoreMetric(BaseMetric):
    """Returns a fixed score for all cases."""

    def __init__(self, value: float):
        super().__init__(name="fixed", dimension=Dimension.CORRECTNESS, threshold=0.5)
        self._value = value

    def measure(self, eval_case: EvalCase) -> Score:
        return Score(name=self.name, value=self._value, threshold=self.threshold)


def _make_goldens(n: int = 3) -> list[Golden]:
    return [Golden(input=f"question {i}", expected=f"answer {i}") for i in range(n)]


class TestPromptOptimizerValidation:
    def test_same_model_and_judge_raises(self):
        model = StubModel()
        with pytest.raises(ValueError, match="model and judge must be different"):
            PromptOptimizer(
                model=model,
                judge=model,
                metrics=[FixedScoreMetric(0.5)],
            )

    def test_empty_metrics_raises(self):
        with pytest.raises(ValueError, match="metrics must not be empty"):
            PromptOptimizer(
                model=StubModel(),
                judge=StubJudge(),
                metrics=[],
            )

    def test_invalid_max_iterations_raises(self):
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            PromptOptimizer(
                model=StubModel(),
                judge=StubJudge(),
                metrics=[FixedScoreMetric(0.5)],
                max_iterations=0,
            )

    @pytest.mark.asyncio
    async def test_empty_goldens_raises(self):
        optimizer = PromptOptimizer(
            model=StubModel(),
            judge=StubJudge(),
            metrics=[FixedScoreMetric(0.5)],
        )
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        with pytest.raises(ValueError, match="goldens must not be empty"):
            await optimizer.optimize(prompt=prompt, goldens=[])


class TestPromptOptimizerConvergence:
    @pytest.mark.asyncio
    async def test_already_at_target(self):
        """If initial prompt already meets target, return immediately."""
        optimizer = PromptOptimizer(
            model=StubModel(default="answer 0"),
            judge=StubJudge(),
            metrics=[FixedScoreMetric(0.9)],
            target_score=0.85,
        )
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        goldens = _make_goldens(2)

        result = await optimizer.optimize(prompt=prompt, goldens=goldens)

        assert result.converged is True
        assert result.iterations == 0
        assert result.best_score >= 0.85
        assert result.best_prompt.template == "{{input}}"

    @pytest.mark.asyncio
    async def test_converges_after_rewrite(self):
        """Optimizer converges when judge provides a better prompt."""
        call_count = {"n": 0}

        def score_fn(eval_case: EvalCase) -> float:
            call_count["n"] += 1
            if "concisely" in str(eval_case.input) or call_count["n"] > 6:
                return 0.9
            return 0.4

        optimizer = PromptOptimizer(
            model=StubModel(),
            judge=StubJudge(candidates=[["Answer concisely: {{input}}"]]),
            metrics=[ScoreControlMetric(score_fn)],
            target_score=0.85,
            max_iterations=5,
        )
        prompt = PromptTemplate(template="Answer: {{input}}", input_variables=["input"])
        goldens = _make_goldens(2)

        result = await optimizer.optimize(prompt=prompt, goldens=goldens)

        assert result.best_score >= 0.85
        assert result.iterations > 0
        assert len(result.score_history) >= 1


class TestPromptOptimizerEarlyStopping:
    @pytest.mark.asyncio
    async def test_patience_exhausted(self):
        """Stops early when no improvement for `patience` iterations."""
        optimizer = PromptOptimizer(
            model=StubModel(default="wrong"),
            judge=StubJudge(
                candidates=[
                    ["Try this: {{input}}"],
                    ["Maybe this: {{input}}"],
                    ["Or this: {{input}}"],
                    ["Last try: {{input}}"],
                ]
            ),
            metrics=[FixedScoreMetric(0.3)],
            target_score=0.9,
            max_iterations=10,
            patience=2,
        )
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        goldens = _make_goldens(2)

        result = await optimizer.optimize(prompt=prompt, goldens=goldens)

        assert result.converged is False
        assert result.iterations <= 4

    @pytest.mark.asyncio
    async def test_no_improvement_plateau(self):
        """Score stays flat — optimizer stops via patience."""
        optimizer = PromptOptimizer(
            model=StubModel(default="same"),
            judge=StubJudge(),
            metrics=[FixedScoreMetric(0.5)],
            target_score=0.95,
            max_iterations=10,
            patience=3,
        )
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        goldens = _make_goldens(3)

        result = await optimizer.optimize(prompt=prompt, goldens=goldens)

        assert result.converged is False
        assert result.best_score == pytest.approx(0.5)


class TestPromptOptimizerEdgeCases:
    @pytest.mark.asyncio
    async def test_single_golden(self):
        """Works with a single golden."""
        optimizer = PromptOptimizer(
            model=StubModel(default="answer 0"),
            judge=StubJudge(),
            metrics=[FixedScoreMetric(0.9)],
            target_score=0.85,
        )
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        goldens = [Golden(input="q", expected="a")]

        result = await optimizer.optimize(prompt=prompt, goldens=goldens)

        assert isinstance(result, OptimizationResult)
        assert result.converged is True

    @pytest.mark.asyncio
    async def test_result_to_dict(self):
        """OptimizationResult serializes to dict."""
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        result = OptimizationResult(
            best_prompt=prompt,
            best_score=0.9,
            initial_score=0.4,
            score_history=[0.7, 0.9],
            candidate_scores=[0.7, 0.9],
            prompt_history=[prompt, prompt],
            iterations=2,
            converged=True,
        )
        d = result.to_dict()
        assert d["best_score"] == 0.9
        assert d["converged"] is True
        assert len(d["score_history"]) == 2
        assert len(d["candidate_scores"]) == 2
        assert d["best_prompt"] == "{{input}}"

    @pytest.mark.asyncio
    async def test_result_save(self, tmp_path):
        """OptimizationResult.save() writes a JSON file."""
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        result = OptimizationResult(
            best_prompt=prompt,
            best_score=0.9,
            initial_score=0.4,
            score_history=[0.7, 0.9],
            candidate_scores=[0.7, 0.9],
            prompt_history=[prompt, prompt],
            iterations=2,
            converged=True,
        )
        out = str(tmp_path / "result.json")
        result.save(out)
        import json

        with open(out) as f:
            data = json.load(f)
        assert data["best_score"] == 0.9

    def test_invalid_target_score_raises(self):
        """target_score outside (0, 1] raises."""
        with pytest.raises(ValueError, match="target_score"):
            PromptOptimizer(
                model=StubModel(),
                judge=StubJudge(),
                metrics=[FixedScoreMetric(0.5)],
                target_score=0.0,
            )
        with pytest.raises(ValueError, match="target_score"):
            PromptOptimizer(
                model=StubModel(),
                judge=StubJudge(),
                metrics=[FixedScoreMetric(0.5)],
                target_score=1.5,
            )

    @pytest.mark.asyncio
    async def test_multi_variable_prompt_with_dict_input(self):
        """Optimizer handles multi-variable prompts when golden.input is a dict."""
        optimizer = PromptOptimizer(
            model=StubModel(default="answer"),
            judge=StubJudge(),
            metrics=[FixedScoreMetric(0.9)],
            target_score=0.85,
        )
        prompt = PromptTemplate(
            template="{{question}} given {{context}}",
            input_variables=["question", "context"],
        )
        goldens = [
            Golden(input={"question": "What is 2+2?", "context": "math"}, expected="4"),
        ]
        result = await optimizer.optimize(prompt=prompt, goldens=goldens)
        assert result.converged is True

    @pytest.mark.asyncio
    async def test_multi_variable_dict_missing_key_raises(self):
        """Dict input missing a required variable raises a clear error."""
        optimizer = PromptOptimizer(
            model=StubModel(default="answer"),
            judge=StubJudge(),
            metrics=[FixedScoreMetric(0.9)],
            target_score=0.85,
        )
        prompt = PromptTemplate(
            template="{{question}} given {{context}}",
            input_variables=["question", "context"],
        )
        goldens = [Golden(input={"question": "What?"}, expected="4")]
        with pytest.raises(ValueError, match="missing variables"):
            await optimizer.optimize(prompt=prompt, goldens=goldens)

    @pytest.mark.asyncio
    async def test_multi_variable_prompt_with_context_fallback(self):
        """Multi-variable prompt fills extra vars from golden.context when input is a string."""
        optimizer = PromptOptimizer(
            model=StubModel(default="answer"),
            judge=StubJudge(),
            metrics=[FixedScoreMetric(0.9)],
            target_score=0.85,
        )
        prompt = PromptTemplate(
            template="{{question}} given {{context}}",
            input_variables=["question", "context"],
        )
        goldens = [
            Golden(input="What is 2+2?", expected="4", context=["math"]),
        ]
        result = await optimizer.optimize(prompt=prompt, goldens=goldens)
        assert result.converged is True

    @pytest.mark.asyncio
    async def test_malformed_candidates_warns(self):
        """When all candidates are malformed, a warning is emitted and the current prompt is used."""

        class BadJudge(BaseLLM):
            async def generate(self, prompt: str, **kwargs) -> str:
                return "diagnosis"

            async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
                return {"candidates": [None, 123, "", "   "]}

        optimizer = PromptOptimizer(
            model=StubModel(default="wrong"),
            judge=BadJudge(),
            metrics=[FixedScoreMetric(0.3)],
            target_score=0.9,
            max_iterations=1,
            patience=1,
        )
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        goldens = _make_goldens(2)

        with pytest.warns(UserWarning, match="empty/non-string candidate|All candidates were malformed"):
            result = await optimizer.optimize(prompt=prompt, goldens=goldens)
        assert result.iterations >= 1

    @pytest.mark.asyncio
    async def test_on_iteration_callback(self):
        """on_iteration callback is called with iteration number, best score, and best prompt."""
        calls = []

        optimizer = PromptOptimizer(
            model=StubModel(default="wrong"),
            judge=StubJudge(),
            metrics=[FixedScoreMetric(0.5)],
            target_score=0.9,
            max_iterations=3,
            patience=2,
        )
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        goldens = _make_goldens(2)

        result = await optimizer.optimize(
            prompt=prompt, goldens=goldens, on_iteration=lambda i, s, p: calls.append((i, s, p))
        )

        assert len(calls) == result.iterations
        assert all(isinstance(i, int) and isinstance(s, float) for i, s, _ in calls)
        assert all(isinstance(p, PromptTemplate) for _, _, p in calls)

    @pytest.mark.asyncio
    async def test_score_history_is_monotonic(self):
        """score_history records the running best, so it must be non-decreasing."""
        call_count = {"n": 0}

        def score_fn(eval_case: EvalCase) -> float:
            call_count["n"] += 1
            if call_count["n"] <= 4:
                return 0.4
            return 0.7

        optimizer = PromptOptimizer(
            model=StubModel(),
            judge=StubJudge(
                candidates=[
                    ["Try A: {{input}}"],
                    ["Try B: {{input}}"],
                    ["Try C: {{input}}"],
                ]
            ),
            metrics=[ScoreControlMetric(score_fn)],
            target_score=0.95,
            max_iterations=3,
            patience=3,
        )
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        goldens = _make_goldens(2)

        result = await optimizer.optimize(prompt=prompt, goldens=goldens)

        for i in range(1, len(result.score_history)):
            assert result.score_history[i] >= result.score_history[i - 1]

    @pytest.mark.asyncio
    async def test_candidate_scores_track_raw_values(self):
        """candidate_scores records the actual best-candidate score each iteration."""
        optimizer = PromptOptimizer(
            model=StubModel(default="wrong"),
            judge=StubJudge(),
            metrics=[FixedScoreMetric(0.5)],
            target_score=0.9,
            max_iterations=3,
            patience=3,
        )
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        goldens = _make_goldens(2)

        result = await optimizer.optimize(prompt=prompt, goldens=goldens)

        assert len(result.candidate_scores) == result.iterations
        assert len(result.candidate_scores) == len(result.score_history)


class TestPromptOptimizerConcurrency:
    @pytest.mark.asyncio
    async def test_max_concurrency_limits_parallel_calls(self):
        """Semaphore limits concurrent model.generate() calls."""
        import asyncio

        peak_concurrent = {"value": 0}
        current_concurrent = {"value": 0}

        class TrackedModel(BaseLLM):
            async def generate(self, prompt: str, **kwargs) -> str:
                current_concurrent["value"] += 1
                if current_concurrent["value"] > peak_concurrent["value"]:
                    peak_concurrent["value"] = current_concurrent["value"]
                await asyncio.sleep(0.01)
                current_concurrent["value"] -= 1
                return "output"

            async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
                return {}

        optimizer = PromptOptimizer(
            model=TrackedModel(),
            judge=StubJudge(),
            metrics=[FixedScoreMetric(0.9)],
            target_score=0.85,
            max_concurrency=2,
        )
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        goldens = _make_goldens(5)

        await optimizer.optimize(prompt=prompt, goldens=goldens)

        assert peak_concurrent["value"] <= 2

    @pytest.mark.asyncio
    async def test_semaphore_shared_across_candidate_evaluations(self):
        """Concurrent candidate evaluations share the same semaphore."""
        import asyncio

        peak_concurrent = {"value": 0}
        current_concurrent = {"value": 0}

        class TrackedModel(BaseLLM):
            async def generate(self, prompt: str, **kwargs) -> str:
                current_concurrent["value"] += 1
                if current_concurrent["value"] > peak_concurrent["value"]:
                    peak_concurrent["value"] = current_concurrent["value"]
                await asyncio.sleep(0.01)
                current_concurrent["value"] -= 1
                return "wrong"

            async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
                return {}

        optimizer = PromptOptimizer(
            model=TrackedModel(),
            judge=StubJudge(
                candidates=[
                    ["Candidate A: {{input}}", "Candidate B: {{input}}", "Candidate C: {{input}}"],
                ]
            ),
            metrics=[FixedScoreMetric(0.3)],
            target_score=0.9,
            max_iterations=1,
            num_candidates=3,
            max_concurrency=2,
            patience=1,
        )
        prompt = PromptTemplate(template="{{input}}", input_variables=["input"])
        goldens = _make_goldens(6)

        await optimizer.optimize(prompt=prompt, goldens=goldens)

        # 3 candidates * 6 goldens = 18 concurrent coroutines, but semaphore caps at 2
        assert peak_concurrent["value"] <= 2
