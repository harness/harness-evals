"""Built-in metric catalog — introspects all shipped metrics and returns their metadata.

Platform services (e.g., ai-evals) call ``catalog()`` at startup to discover
available metrics and sync them into their persistence layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness_evals.core.metric import BaseMetric, Dimension, SafetyMetric


@dataclass(frozen=True)
class CatalogEntry:
    """Metadata for a single built-in metric."""

    kind: str
    name: str
    metric_class: type[BaseMetric]
    dimension: Dimension
    category: str
    default_threshold: float
    requires_llm: bool
    requires_embedding: bool
    description: str


def _category_from_module(cls: type) -> str:
    """Extract category from module path: harness_evals.metrics.<category>.xxx → category."""
    parts = cls.__module__.split(".")
    # e.g., harness_evals.metrics.deterministic.exact_match → deterministic
    try:
        idx = parts.index("metrics")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return "unknown"


def _requires_param(cls: type, param: str) -> bool:
    """Check if __init__ accepts a given parameter name."""
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
        return param in sig.parameters
    except (ValueError, TypeError):
        return False


def _get_dimension(cls: type) -> Dimension:
    """Get the dimension for a metric class by inspecting its default or instantiating a stub."""
    # SafetyMetric subclasses default to SAFETY
    if issubclass(cls, SafetyMetric):
        return Dimension.SAFETY

    # For metrics that need LLM/embedding, inspect the __init__ source for dimension=
    import inspect

    try:
        source = inspect.getsource(cls.__init__)
        for dim in Dimension:
            if f"Dimension.{dim.name}" in source:
                return dim
    except (TypeError, OSError):
        pass

    return Dimension.CORRECTNESS  # fallback


def _get_default_threshold(cls: type) -> float:
    """Extract default threshold from __init__ signature."""
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
        param = sig.parameters.get("threshold")
        if param and param.default is not inspect.Parameter.empty:
            return float(param.default)
    except (ValueError, TypeError):
        pass
    return 1.0


# ---------------------------------------------------------------------------
# Registry: kind → metric class
# Mirrors the ai-evals HEURISTIC_METRIC_REGISTRY but is the canonical source.
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, type[BaseMetric]]:
    """Build kind → class mapping from all shipped metrics."""
    from harness_evals.metrics import (
        AnswerCorrectnessMetric,
        AnswerRelevancyMetric,
        AnswerSimilarityMetric,
        ArgumentCorrectnessMetric,
        BiasMetric,
        BLEUMetric,
        BrierScoreMetric,
        CalibrationMetric,
        ContainsMetric,
        ContextEntityRecallMetric,
        ContextPrecisionMetric,
        ContextRecallMetric,
        ContextRelevancyMetric,
        ConversationCoherenceMetric,
        ConversationCompletenessMetric,
        ConversationResolutionMetric,
        CostEfficiencyMetric,
        DAGMetric,
        DiscriminationMetric,
        EmbeddingSimilarityMetric,
        EnvironmentRobustnessMetric,
        ExactMatchMetric,
        FaithfulnessMetric,
        FaultRobustnessMetric,
        GEvalMetric,
        GoalAccuracyMetric,
        HallucinationMetric,
        JsonDiffMetric,
        KnowledgeRetentionMetric,
        LatencyMetric,
        LevenshteinMetric,
        ListContainsMetric,
        MCPTraceCompletenessMetric,
        NumericDiffMetric,
        OutcomeConsistencyMetric,
        PairwiseMetric,
        PIIMetric,
        PlanAdherenceMetric,
        PlanQualityMetric,
        PromptAlignmentMetric,
        PromptInjectionMetric,
        PromptRobustnessMetric,
        RegexMetric,
        ResourceConsistencyMetric,
        RetryCountMetric,
        RoleAdherenceMetric,
        RubricJudgeMetric,
        SchemaValidationMetric,
        StepEfficiencyMetric,
        SummarizationMetric,
        TaskCompletionMetric,
        TokenCostMetric,
        ToolCorrectnessMetric,
        ToolSelectionAccuracyMetric,
        ToolUseMetric,
        TopicAdherenceMetric,
        ToxicityMetric,
        TrajectoryConsistencyMetric,
        TurnEfficiencyMetric,
        TurnRelevancyMetric,
    )
    from harness_evals.metrics.security import (
        ActionabilityMetric,
        CodeQualityMetric,
        CodeSafetyMetric,
        ExplanationQualityMetric,
        RootCauseAnalysisMetric,
        SecurityCompletenessMetric,
        VulnerabilityCorrectnessMetric,
    )

    return {
        # Deterministic
        "exact_match": ExactMatchMetric,
        "contains": ContainsMetric,
        "list_contains": ListContainsMetric,
        "regex": RegexMetric,
        "numeric_diff": NumericDiffMetric,
        # Structural
        "json_diff": JsonDiffMetric,
        "schema_validation": SchemaValidationMetric,
        # Similarity
        "levenshtein": LevenshteinMetric,
        "bleu": BLEUMetric,
        "embedding_similarity": EmbeddingSimilarityMetric,
        # Operational
        "latency": LatencyMetric,
        "token_cost": TokenCostMetric,
        "cost_efficiency": CostEfficiencyMetric,
        "retry_count": RetryCountMetric,
        # LLM Judge
        "geval": GEvalMetric,
        "rubric_judge": RubricJudgeMetric,
        "pairwise": PairwiseMetric,
        "dag": DAGMetric,
        "summarization": SummarizationMetric,
        "prompt_alignment": PromptAlignmentMetric,
        # RAG
        "faithfulness": FaithfulnessMetric,
        "answer_relevancy": AnswerRelevancyMetric,
        "answer_correctness": AnswerCorrectnessMetric,
        "answer_similarity": AnswerSimilarityMetric,
        "context_precision": ContextPrecisionMetric,
        "context_recall": ContextRecallMetric,
        "context_relevancy": ContextRelevancyMetric,
        "context_entity_recall": ContextEntityRecallMetric,
        # Safety
        "pii": PIIMetric,
        "toxicity": ToxicityMetric,
        "prompt_injection": PromptInjectionMetric,
        "hallucination": HallucinationMetric,
        "bias": BiasMetric,
        # Agent
        "task_completion": TaskCompletionMetric,
        "tool_correctness": ToolCorrectnessMetric,
        "argument_correctness": ArgumentCorrectnessMetric,
        "plan_adherence": PlanAdherenceMetric,
        "plan_quality": PlanQualityMetric,
        "step_efficiency": StepEfficiencyMetric,
        # Conversation
        "conversation_coherence": ConversationCoherenceMetric,
        "conversation_completeness": ConversationCompletenessMetric,
        "conversation_resolution": ConversationResolutionMetric,
        "goal_accuracy": GoalAccuracyMetric,
        "knowledge_retention": KnowledgeRetentionMetric,
        "role_adherence": RoleAdherenceMetric,
        "tool_use": ToolUseMetric,
        "topic_adherence": TopicAdherenceMetric,
        "turn_efficiency": TurnEfficiencyMetric,
        "turn_relevancy": TurnRelevancyMetric,
        # MCP
        "tool_selection": ToolSelectionAccuracyMetric,
        "mcp_trace_completeness": MCPTraceCompletenessMetric,
        # Reliability
        "outcome_consistency": OutcomeConsistencyMetric,
        "resource_consistency": ResourceConsistencyMetric,
        "trajectory_consistency": TrajectoryConsistencyMetric,
        "brier_score": BrierScoreMetric,
        "calibration": CalibrationMetric,
        "discrimination": DiscriminationMetric,
        "prompt_robustness": PromptRobustnessMetric,
        "environment_robustness": EnvironmentRobustnessMetric,
        "fault_robustness": FaultRobustnessMetric,
        # Security Remediation
        "vulnerability_correctness": VulnerabilityCorrectnessMetric,
        "security_completeness": SecurityCompletenessMetric,
        "code_safety": CodeSafetyMetric,
        "code_quality": CodeQualityMetric,
        "explanation_quality": ExplanationQualityMetric,
        "root_cause_analysis": RootCauseAnalysisMetric,
        "actionability": ActionabilityMetric,
    }


def catalog() -> list[CatalogEntry]:
    """Return metadata for all built-in metrics.

    Each entry includes the metric's kind (string identifier), dimension,
    category, default threshold, and capability requirements (LLM, embedding).

    This is the canonical source of truth for the metric catalog. Platform
    services should call this at startup to sync built-in metrics.
    """
    registry = _build_registry()
    entries: list[CatalogEntry] = []

    for kind, cls in registry.items():
        entries.append(
            CatalogEntry(
                kind=kind,
                name=cls.__name__,
                metric_class=cls,
                dimension=_get_dimension(cls),
                category=_category_from_module(cls),
                default_threshold=_get_default_threshold(cls),
                requires_llm=_requires_param(cls, "llm"),
                requires_embedding=_requires_param(cls, "embedding"),
                description=(cls.__doc__ or "").split("\n")[0].strip(),
            )
        )

    return entries
