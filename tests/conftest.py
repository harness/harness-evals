"""Shared test fixtures for harness-evals."""

import pytest

from harness_evals.core.test_case import TestCase


@pytest.fixture
def simple_test_case() -> TestCase:
    return TestCase(
        input="What is 2+2?",
        actual_output="4",
        expected_output="4",
    )


@pytest.fixture
def json_test_case() -> TestCase:
    return TestCase(
        input="Generate a K8s deployment",
        actual_output={"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "web"}},
        expected_output={"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "web"}},
    )


@pytest.fixture
def operational_test_case() -> TestCase:
    return TestCase(
        input="List pods",
        actual_output="kubectl get pods",
        expected_output="kubectl get pods",
        metadata={"latency_ms": 1200, "token_usage": 350, "cost_usd": 0.003, "retry_count": 0},
    )


@pytest.fixture
def multi_run_test_case() -> TestCase:
    runs = [
        TestCase(input="task", actual_output="result_a", metadata={"token_usage": 100}),
        TestCase(input="task", actual_output="result_a", metadata={"token_usage": 110}),
        TestCase(input="task", actual_output="result_a", metadata={"token_usage": 105}),
        TestCase(input="task", actual_output="result_b", metadata={"token_usage": 95}),
        TestCase(input="task", actual_output="result_a", metadata={"token_usage": 102}),
    ]
    return TestCase(input="task", actual_output="result_a", runs=runs)
