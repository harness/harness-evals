"""harness-evals: Open-source AI evaluation framework."""

from harness_evals.baseline import (
    BaselineResult,
    BaselineStore,
    JsonBaselineStore,
    MetricDelta,
    compare_to_baseline,
)
from harness_evals.catalog import CatalogEntry, catalog
from harness_evals.config import EvalConfig, load_config, loads_config, run_config
from harness_evals.conversation import (
    ConversationGolden,
    ConversationSimulator,
    evaluate_conversation,
    evaluate_conversations,
    load_conversation_dataset,
    save_conversation_dataset,
)
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.metric import BaseMetric, Dimension, ReliabilityMetric, SafetyMetric
from harness_evals.core.runner import (
    a_evaluate,
    assert_test,
    evaluate,
    evaluate_batch_metrics,
    evaluate_cases,
    evaluate_dataset,
    evaluate_dataset_pair,
)
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.core.types import Message, ToolCall
from harness_evals.datasets import Dataset, load_dataset, loads_dataset, save_dataset
from harness_evals.errors import HarnessEvalsError, MissingAdapterError, TargetInvocationError
from harness_evals.eval import run_eval
from harness_evals.importers.base import BaseEvalCaseSource, BaseEvalConfigSource
from harness_evals.input_generator import InputGenerator
from harness_evals.logging_config import configure_logging
from harness_evals.logging_config import init_from_env as _init_logging_from_env
from harness_evals.metrics.operational.turn_latency import TurnLatencyMetric
from harness_evals.metrics.operational.turn_token_cost import TurnTokenCostMetric
from harness_evals.optimizer import OptimizationResult, PromptOptimizer
from harness_evals.refs import ResourceRef, resolve
from harness_evals.reporting import EvalResult, HtmlReporter, HtmlSink
from harness_evals.sinks import CsvSink, JsonSink, JUnitSink, StdoutSink
from harness_evals.summary import (
    SAFETY_DIMENSION,
    UNKNOWN_DIMENSION,
    DimensionSummary,
    MetricSummary,
    ScoreSummary,
    summarize,
)
from harness_evals.synthesizer import ConversationSynthesizer, ScriptedConversationSynthesizer, Synthesizer
from harness_evals.targets import (
    ApiKeyAuth,
    AuthConfig,
    BaseTarget,
    BasicAuth,
    BearerAuth,
    HttpTarget,
    NoAuth,
    PromptTarget,
)
from harness_evals.testing import Fault, FaultInjector

# Honor HARNESS_EVALS_LOG_LEVEL for library consumers (no CLI) at import time.
_init_logging_from_env()

__all__ = [
    # Core types
    "Golden",
    "EvalCase",
    "Score",
    "BaseMetric",
    "CatalogEntry",
    "Dimension",
    "catalog",
    "ResourceRef",
    "resolve",
    "HarnessEvalsError",
    "MissingAdapterError",
    "TargetInvocationError",
    # Importers
    "BaseEvalCaseSource",
    "BaseEvalConfigSource",
    "ReliabilityMetric",
    "SafetyMetric",
    "Message",
    "ToolCall",
    # Runner
    "BaseSink",
    "StdoutSink",
    "JsonSink",
    "CsvSink",
    "JUnitSink",
    "evaluate",
    "a_evaluate",
    "assert_test",
    "evaluate_cases",
    "evaluate_dataset",
    "evaluate_dataset_pair",
    "evaluate_batch_metrics",
    "summarize",
    "MetricSummary",
    "DimensionSummary",
    "ScoreSummary",
    "SAFETY_DIMENSION",
    "UNKNOWN_DIMENSION",
    # Baseline
    "BaselineStore",
    "JsonBaselineStore",
    "BaselineResult",
    "MetricDelta",
    "compare_to_baseline",
    # Datasets
    "Dataset",
    "load_dataset",
    "loads_dataset",
    "save_dataset",
    # Targets
    "BaseTarget",
    "PromptTarget",
    "HttpTarget",
    "AuthConfig",
    "NoAuth",
    "BearerAuth",
    "ApiKeyAuth",
    "BasicAuth",
    # Reporting
    "EvalResult",
    "HtmlReporter",
    "HtmlSink",
    # Synthesis & testing
    "InputGenerator",
    "Synthesizer",
    "ConversationSynthesizer",
    "ScriptedConversationSynthesizer",
    "Fault",
    "FaultInjector",
    "TurnLatencyMetric",
    "TurnTokenCostMetric",
    # Optimizer
    "PromptOptimizer",
    "OptimizationResult",
    # Conversation
    "ConversationGolden",
    "ConversationSimulator",
    "evaluate_conversation",
    "evaluate_conversations",
    "load_conversation_dataset",
    "save_conversation_dataset",
    # Config & runner
    "EvalConfig",
    "load_config",
    "loads_config",
    "run_config",
    "run_eval",
    # Logging
    "configure_logging",
]
