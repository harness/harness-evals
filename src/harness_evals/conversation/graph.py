"""SimulationGraph — declarative DAG for scripted multi-turn conversation scenarios."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from harness_evals.core.types import Message


@dataclass
class ScriptedNode:
    """Returns a fixed user message without LLM calls.

    If no outgoing edge matches after the agent responds, the node re-executes
    and the same message is sent again (until a predicate matches or max_turns).
    """

    message: str


@dataclass
class LLMNode:
    """Generates next user turn via LLM, guided by a goal prompt.

    If no outgoing edge matches after the agent responds, the node re-executes
    and generates a new message using the same goal (with updated history).
    """

    goal: str


@dataclass
class StopNode:
    """Terminates the conversation."""


@dataclass
class BranchNode:
    """Pure routing node — produces no message, evaluates outgoing edges to choose next node."""


SimulationNode = ScriptedNode | LLMNode | StopNode | BranchNode


@dataclass
class Edge:
    """Directed edge from source to target, optionally guarded by a named predicate."""

    source: str
    target: str
    predicate: str | None = None


@dataclass
class SimulationGraph:
    """DAG controlling user turn generation in multi-turn conversation simulation.

    Nodes produce user messages (or stop/route), edges connect nodes with optional
    predicate guards. Predicates are stored separately by name for serialization.

    Cycles are structurally prohibited — even if predicates would make a cycle safe
    at runtime, validation rejects back-edges. For retry/loop behavior, use the
    "stay on current node" mechanic (a node with no matching outgoing edge re-executes
    until a predicate matches or max_turns is reached).
    """

    nodes: dict[str, SimulationNode] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    predicates: dict[str, Callable[[Message], bool]] = field(default_factory=dict)
    start: str = ""

    def __post_init__(self) -> None:
        if self.nodes and self.start:
            self._validate()

    def _validate(self) -> None:
        if self.start not in self.nodes:
            raise ValueError(f"start node '{self.start}' not found in nodes")
        if isinstance(self.nodes[self.start], BranchNode):
            raise ValueError("start node cannot be a BranchNode (no prior agent response to route on)")
        for edge in self.edges:
            if edge.source not in self.nodes:
                raise ValueError(f"edge source '{edge.source}' not found in nodes")
            if edge.target not in self.nodes:
                raise ValueError(f"edge target '{edge.target}' not found in nodes")
            if isinstance(self.nodes[edge.source], StopNode):
                raise ValueError(f"StopNode '{edge.source}' cannot have outgoing edges")
            if edge.predicate is not None and edge.predicate not in self.predicates:
                raise ValueError(f"predicate '{edge.predicate}' not found in predicates dict")

        default_edges: dict[str, int] = {}
        for edge in self.edges:
            if edge.predicate is None:
                default_edges[edge.source] = default_edges.get(edge.source, 0) + 1
                if default_edges[edge.source] > 1:
                    raise ValueError(f"node '{edge.source}' has multiple unconditional edges; at most one is allowed")

        self._detect_cycles()

    def _detect_cycles(self) -> None:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {nid: WHITE for nid in self.nodes}

        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for edge in self.edges:
            adj[edge.source].append(edge.target)

        for start in self.nodes:
            if color[start] != WHITE:
                continue
            stack: list[tuple[str, int]] = [(start, 0)]
            color[start] = GRAY
            while stack:
                node_id, idx = stack[-1]
                neighbors = adj[node_id]
                if idx < len(neighbors):
                    stack[-1] = (node_id, idx + 1)
                    neighbor = neighbors[idx]
                    if color[neighbor] == GRAY:
                        raise ValueError(f"cycle detected: edge from '{node_id}' to '{neighbor}' creates a cycle")
                    if color[neighbor] == WHITE:
                        color[neighbor] = GRAY
                        stack.append((neighbor, 0))
                else:
                    color[node_id] = BLACK
                    stack.pop()

    def get_outgoing_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_id]

    def resolve_next(self, node_id: str, last_response: Message) -> str | None:
        """Determine next node given current node and last agent response.

        Evaluates predicated edges in order; first match wins.
        An edge with predicate=None acts as default fallback.
        Returns None if no edge matches (stay on current node).
        """
        outgoing = self.get_outgoing_edges(node_id)
        default_target: str | None = None
        for edge in outgoing:
            if edge.predicate is None:
                default_target = edge.target
            else:
                if self.predicates[edge.predicate](last_response):
                    return edge.target
        return default_target

    def to_dict(self) -> dict:
        """Serialize graph structure. Predicate functions are stored by name only."""
        serialized_nodes = []
        for node_id, node in self.nodes.items():
            entry: dict = {"id": node_id}
            if isinstance(node, LLMNode):
                entry["type"] = "llm"
                entry["goal"] = node.goal
            elif isinstance(node, ScriptedNode):
                entry["type"] = "scripted"
                entry["message"] = node.message
            elif isinstance(node, StopNode):
                entry["type"] = "stop"
            elif isinstance(node, BranchNode):
                entry["type"] = "branch"
            serialized_nodes.append(entry)

        serialized_edges = []
        for edge in self.edges:
            e: dict = {"source": edge.source, "target": edge.target}
            if edge.predicate is not None:
                e["predicate"] = edge.predicate
            serialized_edges.append(e)

        return {
            "start": self.start,
            "nodes": serialized_nodes,
            "edges": serialized_edges,
        }

    @classmethod
    def from_dict(cls, data: dict, predicates: dict[str, Callable[[Message], bool]] | None = None) -> SimulationGraph:
        """Reconstruct graph from serialized dict and caller-supplied predicates.

        Predicate functions are not serializable, so edges with predicate guards
        require the caller to supply a matching predicates dict. If the serialized
        data references predicates but none are provided, validation will raise
        ValueError. For graphs loaded from golden.graph_config without explicit
        predicates, only unconditional edges are supported.
        """
        for key in ("start", "nodes", "edges"):
            if key not in data:
                raise ValueError(f"graph data missing required key '{key}'")

        _NODE_SCHEMA: dict[str, tuple[set[str], set[str]]] = {
            # type: (required_keys, all_valid_keys)
            "llm": ({"id", "type", "goal"}, {"id", "type", "goal"}),
            "scripted": ({"id", "type", "message"}, {"id", "type", "message"}),
            "stop": ({"id", "type"}, {"id", "type"}),
            "branch": ({"id", "type"}, {"id", "type"}),
        }

        nodes: dict[str, SimulationNode] = {}
        for n in data["nodes"]:
            ntype = n.get("type")
            nid = n.get("id")
            if not nid:
                raise ValueError(f"node is missing a valid 'id': {n}")
            if ntype not in _NODE_SCHEMA:
                raise ValueError(f"unknown node type '{ntype}'")
            required, valid = _NODE_SCHEMA[ntype]
            missing = required - set(n.keys())
            if missing:
                raise ValueError(f"node '{nid}' missing required keys {missing} for type '{ntype}'")
            unexpected = set(n.keys()) - valid
            if unexpected:
                raise ValueError(f"node '{nid}' has unexpected keys {unexpected} for type '{ntype}'")
            if ntype == "llm":
                nodes[nid] = LLMNode(goal=n["goal"])
            elif ntype == "scripted":
                nodes[nid] = ScriptedNode(message=n["message"])
            elif ntype == "stop":
                nodes[nid] = StopNode()
            elif ntype == "branch":
                nodes[nid] = BranchNode()

        edges = [Edge(source=e["source"], target=e["target"], predicate=e.get("predicate")) for e in data["edges"]]

        if not predicates:
            predicated = [e.predicate for e in edges if e.predicate is not None]
            if predicated:
                raise ValueError(
                    f"edges reference predicates {predicated} but no predicates dict was supplied. "
                    f"Pass a predicates dict to from_dict() or use only unconditional edges."
                )

        return cls(
            nodes=nodes,
            edges=edges,
            predicates=predicates or {},
            start=data["start"],
        )
