"""Tests for DAG metric."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import Dimension
from harness_evals.metrics.llm_judge.dag import (
    BinaryJudgementNode,
    DAGMetric,
    DeepAcyclicGraph,
    NonBinaryJudgementNode,
    TaskNode,
    VerdictNode,
)
from tests.conftest import MockLLM


@pytest.mark.unit
class TestVerdictNode:
    def test_valid_score(self):
        v = VerdictNode(verdict="good", score=0.8)
        assert v.score == 0.8

    def test_invalid_score_raises(self):
        with pytest.raises(ValueError, match="0.0-1.0"):
            VerdictNode(verdict="bad", score=1.5)


@pytest.mark.unit
class TestBinaryJudgementNode:
    def test_requires_true_and_false(self):
        with pytest.raises(ValueError, match="True and one False"):
            BinaryJudgementNode(
                criteria="test",
                children=[VerdictNode(verdict=True, score=1.0)],
            )

    def test_get_child(self):
        node = BinaryJudgementNode(
            criteria="test",
            children=[
                VerdictNode(verdict=True, score=1.0),
                VerdictNode(verdict=False, score=0.0),
            ],
        )
        assert node.get_child(True).score == 1.0
        assert node.get_child(False).score == 0.0


@pytest.mark.unit
class TestNonBinaryJudgementNode:
    def test_requires_at_least_two(self):
        with pytest.raises(ValueError, match="at least 2"):
            NonBinaryJudgementNode(
                criteria="test",
                children=[VerdictNode(verdict="only_one", score=0.5)],
            )

    def test_get_child_exact_match(self):
        node = NonBinaryJudgementNode(
            criteria="test",
            children=[
                VerdictNode(verdict="good", score=1.0),
                VerdictNode(verdict="bad", score=0.0),
            ],
        )
        assert node.get_child("good").score == 1.0

    def test_get_child_fallback_to_lowest(self):
        node = NonBinaryJudgementNode(
            criteria="test",
            children=[
                VerdictNode(verdict="good", score=1.0),
                VerdictNode(verdict="bad", score=0.2),
            ],
        )
        assert node.get_child("unknown").score == 0.2


@pytest.mark.unit
class TestDAGMetric:
    def test_dimension_is_correctness(self):
        llm = MockLLM()
        dag = DeepAcyclicGraph(root_nodes=[])
        metric = DAGMetric(llm=llm, dag=dag)
        assert metric.dimension == Dimension.CORRECTNESS

    async def test_simple_binary_pass(self):
        """Binary node returns True → score 1.0."""
        llm = MockLLM(responses=[{"reasoning": "yes", "verdict": True}])
        dag = DeepAcyclicGraph(
            root_nodes=[
                BinaryJudgementNode(
                    criteria="Is the answer correct?",
                    children=[
                        VerdictNode(verdict=True, score=1.0),
                        VerdictNode(verdict=False, score=0.0),
                    ],
                )
            ]
        )
        metric = DAGMetric(llm=llm, dag=dag)
        ec = EvalCase(input="What is 2+2?", output="4", expected="4")
        score = await metric.a_measure(ec)
        assert score.value == 1.0
        assert score.passed

    async def test_simple_binary_fail(self):
        """Binary node returns False → score 0.0."""
        llm = MockLLM(responses=[{"reasoning": "no", "verdict": False}])
        dag = DeepAcyclicGraph(
            root_nodes=[
                BinaryJudgementNode(
                    criteria="Is the answer correct?",
                    children=[
                        VerdictNode(verdict=True, score=1.0),
                        VerdictNode(verdict=False, score=0.0),
                    ],
                )
            ]
        )
        metric = DAGMetric(llm=llm, dag=dag)
        ec = EvalCase(input="What is 2+2?", output="5")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert not score.passed

    async def test_nonbinary_mid_score(self):
        """NonBinary node with partial match."""
        llm = MockLLM(responses=[{"reasoning": "partial", "verdict": "partially correct"}])
        dag = DeepAcyclicGraph(
            root_nodes=[
                NonBinaryJudgementNode(
                    criteria="How correct is the answer?",
                    children=[
                        VerdictNode(verdict="fully correct", score=1.0),
                        VerdictNode(verdict="partially correct", score=0.5),
                        VerdictNode(verdict="incorrect", score=0.0),
                    ],
                )
            ]
        )
        metric = DAGMetric(llm=llm, dag=dag)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.5

    async def test_task_then_binary(self):
        """TaskNode extracts data, then BinaryJudgementNode judges it."""
        llm = MockLLM(
            responses=[
                {"result": "intro, body, conclusion"},  # task extraction
                {"reasoning": "all present", "verdict": True},  # binary judgment
            ]
        )
        dag = DeepAcyclicGraph(
            root_nodes=[
                TaskNode(
                    instructions="Extract headings",
                    output_label="headings",
                    children=[
                        BinaryJudgementNode(
                            criteria="Are all required headings present?",
                            children=[
                                VerdictNode(verdict=True, score=1.0),
                                VerdictNode(verdict=False, score=0.0),
                            ],
                        )
                    ],
                )
            ]
        )
        metric = DAGMetric(llm=llm, dag=dag)
        ec = EvalCase(input="Write a report", output="## Intro\n## Body\n## Conclusion")
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    async def test_chained_verdict_to_child(self):
        """VerdictNode with child continues traversal."""
        llm = MockLLM(
            responses=[
                {"reasoning": "yes", "verdict": True},  # first binary
                {"reasoning": "ordered", "verdict": "in order"},  # second nonbinary
            ]
        )
        order_node = NonBinaryJudgementNode(
            criteria="Are they in order?",
            children=[
                VerdictNode(verdict="in order", score=1.0),
                VerdictNode(verdict="out of order", score=0.3),
            ],
        )
        dag = DeepAcyclicGraph(
            root_nodes=[
                BinaryJudgementNode(
                    criteria="Does it have headings?",
                    children=[
                        VerdictNode(verdict=True, score=1.0, child=order_node),
                        VerdictNode(verdict=False, score=0.0),
                    ],
                )
            ]
        )
        metric = DAGMetric(llm=llm, dag=dag)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    async def test_empty_dag_returns_zero(self):
        llm = MockLLM()
        dag = DeepAcyclicGraph(root_nodes=[])
        metric = DAGMetric(llm=llm, dag=dag)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
