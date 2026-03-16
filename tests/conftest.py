"""Shared test fixtures for harness-evals."""

import pytest

from harness_evals.core.eval_case import EvalCase


@pytest.fixture
def simple_eval_case() -> EvalCase:
    return EvalCase(
        input="What is 2+2?",
        output="4",
        expected="4",
    )


@pytest.fixture
def json_eval_case() -> EvalCase:
    return EvalCase(
        input="Generate a K8s deployment",
        output={"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "web"}},
        expected={"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "web"}},
    )


@pytest.fixture
def operational_eval_case() -> EvalCase:
    return EvalCase(
        input="List pods",
        output="kubectl get pods",
        expected="kubectl get pods",
        latency_ms=1200,
        token_count=350,
        cost_usd=0.003,
        retry_count=0,
    )


@pytest.fixture
def multi_run_eval_case() -> EvalCase:
    runs = [
        EvalCase(input="task", output="result_a", token_count=100),
        EvalCase(input="task", output="result_a", token_count=110),
        EvalCase(input="task", output="result_a", token_count=105),
        EvalCase(input="task", output="result_b", token_count=95),
        EvalCase(input="task", output="result_a", token_count=102),
    ]
    return EvalCase(input="task", output="result_a", runs=runs)
