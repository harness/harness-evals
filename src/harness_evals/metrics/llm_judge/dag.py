"""DAG metric — graph-based deterministic LLM evaluation.

Instead of a single LLM prompt, evaluation is decomposed into a directed
acyclic graph of nodes: extract data (TaskNode), make binary or multi-choice
judgments (JudgementNodes), and assign scores (VerdictNode).  Each node
focuses on one decision, reducing LLM variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


@dataclass
class VerdictNode:
    """Leaf node that assigns a score (0.0-1.0)."""

    verdict: str | bool
    score: float
    child: TaskNode | BinaryJudgementNode | NonBinaryJudgementNode | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"VerdictNode score must be 0.0-1.0, got {self.score}")


@dataclass
class BinaryJudgementNode:
    """Yes/no decision node. Routes to exactly two children (True/False)."""

    criteria: str
    children: list[VerdictNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        verdicts = {v.verdict for v in self.children}
        if verdicts != {True, False}:
            raise ValueError("BinaryJudgementNode requires exactly one True and one False VerdictNode")

    def get_child(self, result: bool) -> VerdictNode:
        for v in self.children:
            if v.verdict == result:
                return v
        raise ValueError(f"No child for verdict={result}")


@dataclass
class NonBinaryJudgementNode:
    """Multi-choice decision node. Routes based on string verdict match."""

    criteria: str
    children: list[VerdictNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.children) < 2:
            raise ValueError("NonBinaryJudgementNode requires at least 2 VerdictNodes")

    def get_child(self, verdict_str: str) -> VerdictNode:
        # Exact match first
        for v in self.children:
            if str(v.verdict).lower().strip() == verdict_str.lower().strip():
                return v
        # Fallback: lowest-scoring child
        return min(self.children, key=lambda v: v.score)


@dataclass
class TaskNode:
    """Processing node — asks the LLM to extract/transform data."""

    instructions: str
    output_label: str = "extracted"
    children: list[BinaryJudgementNode | NonBinaryJudgementNode] = field(default_factory=list)


@dataclass
class DeepAcyclicGraph:
    """Container for the DAG root nodes."""

    root_nodes: list[TaskNode | BinaryJudgementNode | NonBinaryJudgementNode]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_TASK_PROMPT = """You are an expert evaluator. Extract or process the following information.

**Instructions**: {instructions}

**Input**: {input}
**Output**: {output}
{context_section}

Respond with JSON:
{{"result": "<your extracted/processed data as a string>"}}
"""

_BINARY_PROMPT = """You are an expert evaluator making a yes/no judgment.

**Criteria**: {criteria}

{context_section}

Answer strictly with JSON:
{{"reasoning": "brief reasoning", "verdict": true or false}}
"""

_NONBINARY_PROMPT = """You are an expert evaluator. Choose the best matching verdict.

**Criteria**: {criteria}

**Possible verdicts**: {verdicts}

{context_section}

Answer strictly with JSON:
{{"reasoning": "brief reasoning", "verdict": "<one of the possible verdicts exactly>"}}
"""


# ---------------------------------------------------------------------------
# DAG Metric
# ---------------------------------------------------------------------------


class DAGMetric(BaseMetric):
    """Graph-based deterministic LLM evaluation metric.

    Decomposes evaluation into a DAG of task, judgment, and verdict nodes.
    Each LLM call focuses on a single decision, producing more deterministic
    results than a single-prompt approach like GEval.
    """

    def __init__(
        self,
        llm: BaseLLM,
        dag: DeepAcyclicGraph,
        name: str = "dag",
        threshold: float = 0.7,
        **kwargs: object,
    ) -> None:
        super().__init__(name=name, dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.llm = llm
        self.dag = dag

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        context: dict[str, Any] = {
            "input": str(eval_case.input),
            "output": str(eval_case.output),
        }
        if eval_case.expected:
            context["expected"] = str(eval_case.expected)

        scores: list[float] = []
        trace: list[str] = []

        for root in self.dag.root_nodes:
            score_val, path = await self._traverse(root, context)
            scores.append(score_val)
            trace.extend(path)

        final = sum(scores) / len(scores) if scores else 0.0

        return Score(
            name=self.name,
            value=max(0.0, min(1.0, final)),
            threshold=self.threshold,
            reason=" → ".join(trace),
            metadata={"node_scores": scores, "trace": trace},
        )

    async def _traverse(
        self,
        node: TaskNode | BinaryJudgementNode | NonBinaryJudgementNode | VerdictNode,
        context: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """Walk the DAG, executing LLM calls at each node."""

        if isinstance(node, VerdictNode):
            if node.child is not None:
                return await self._traverse(node.child, context)
            return node.score, [f"verdict({node.verdict})={node.score}"]

        if isinstance(node, TaskNode):
            result = await self._execute_task(node, context)
            context[node.output_label] = result
            trace = [f"task({node.output_label})"]

            # Traverse all children, average scores
            child_scores: list[float] = []
            for child in node.children:
                s, t = await self._traverse(child, context)
                child_scores.append(s)
                trace.extend(t)

            avg = sum(child_scores) / len(child_scores) if child_scores else 0.0
            return avg, trace

        if isinstance(node, BinaryJudgementNode):
            result = await self._execute_binary(node, context)
            verdict_node = node.get_child(result)
            trace = [f"binary({result})"]
            s, t = await self._traverse(verdict_node, context)
            trace.extend(t)
            return s, trace

        if isinstance(node, NonBinaryJudgementNode):
            result = await self._execute_nonbinary(node, context)
            verdict_node = node.get_child(result)
            trace = [f"choice({result})"]
            s, t = await self._traverse(verdict_node, context)
            trace.extend(t)
            return s, trace

        return 0.0, ["unknown_node"]

    async def _execute_task(self, node: TaskNode, context: dict[str, Any]) -> str:
        context_section = "\n".join(f"**{k}**: {v}" for k, v in context.items())
        prompt = _TASK_PROMPT.format(
            instructions=node.instructions,
            input=context.get("input", ""),
            output=context.get("output", ""),
            context_section=context_section,
        )
        schema = {"type": "object", "required": ["result"], "properties": {"result": {"type": "string"}}}
        result = await self.llm.generate_json(prompt, schema)
        return str(result.get("result", ""))

    async def _execute_binary(self, node: BinaryJudgementNode, context: dict[str, Any]) -> bool:
        context_section = "\n".join(f"**{k}**: {v}" for k, v in context.items())
        prompt = _BINARY_PROMPT.format(criteria=node.criteria, context_section=context_section)
        schema = {
            "type": "object",
            "required": ["reasoning", "verdict"],
            "properties": {"reasoning": {"type": "string"}, "verdict": {"type": "boolean"}},
        }
        result = await self.llm.generate_json(prompt, schema)
        return bool(result.get("verdict", False))

    async def _execute_nonbinary(self, node: NonBinaryJudgementNode, context: dict[str, Any]) -> str:
        verdicts = [str(v.verdict) for v in node.children]
        context_section = "\n".join(f"**{k}**: {v}" for k, v in context.items())
        prompt = _NONBINARY_PROMPT.format(
            criteria=node.criteria,
            verdicts=", ".join(f'"{v}"' for v in verdicts),
            context_section=context_section,
        )
        schema = {
            "type": "object",
            "required": ["reasoning", "verdict"],
            "properties": {"reasoning": {"type": "string"}, "verdict": {"type": "string"}},
        }
        result = await self.llm.generate_json(prompt, schema)
        return str(result.get("verdict", verdicts[0]))
