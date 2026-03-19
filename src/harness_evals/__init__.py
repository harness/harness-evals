"""harness-evals: Open-source AI evaluation framework."""

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.metric import BaseMetric, ReliabilityMetric, SafetyMetric
from harness_evals.core.runner import (
    a_evaluate,
    assert_test,
    evaluate,
    evaluate_batch_metrics,
    evaluate_cases,
    evaluate_dataset,
)
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.datasets import Dataset, load_dataset, save_dataset
from harness_evals.reporting import EvalResult, HtmlReporter, HtmlSink
from harness_evals.sinks import JsonSink, StdoutSink

__all__ = [
    "Golden",
    "EvalCase",
    "Score",
    "BaseMetric",
    "ReliabilityMetric",
    "SafetyMetric",
    "BaseSink",
    "StdoutSink",
    "JsonSink",
    "evaluate",
    "a_evaluate",
    "assert_test",
    "evaluate_cases",
    "evaluate_dataset",
    "evaluate_batch_metrics",
    "Dataset",
    "load_dataset",
    "save_dataset",
    "EvalResult",
    "HtmlReporter",
    "HtmlSink",
]
