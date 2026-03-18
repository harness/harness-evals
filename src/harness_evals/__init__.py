"""harness-evals: Open-source AI evaluation framework."""

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.metric import BaseMetric, ReliabilityMetric, SafetyMetric
from harness_evals.core.runner import assert_test, evaluate, evaluate_cases, evaluate_dataset
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.datasets import Dataset, load_dataset, save_dataset
from harness_evals.reporting import EvalResult, HtmlReporter, HtmlSink

__all__ = [
    "Golden",
    "EvalCase",
    "Score",
    "BaseMetric",
    "ReliabilityMetric",
    "SafetyMetric",
    "BaseSink",
    "evaluate",
    "assert_test",
    "evaluate_cases",
    "evaluate_dataset",
    "Dataset",
    "load_dataset",
    "save_dataset",
    "EvalResult",
    "HtmlReporter",
    "HtmlSink",
]
