"""Shared dotted-path lookup over ``EvalCase.metadata``.

Coding-agent pipelines typically stash a pre-computed grading artifact (suite
results, coverage) into ``EvalCase.metadata`` before scoring; these helpers
resolve it without the metric knowing the pipeline's module layout.
"""

from __future__ import annotations

from typing import Any


def resolve_metadata(metadata: Any, dotted_path: str) -> Any:
    """Traverse ``metadata`` by ``.``-separated keys; None when anything is missing."""
    if not dotted_path:
        return None
    node: Any = metadata
    for part in dotted_path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def suite_pass_fraction(node: Any) -> tuple[float, str] | None:
    """Interpret a suite-result node; returns (fraction, reason) or None.

    Accepted node shapes:
      ``{"ran": bool, "passed": int, "total": int, ...}`` — grader-style result
      ``{"pass_fraction": float}`` or a bare number in [0, 1] — ready fraction
    """
    if node is None:
        return None
    if isinstance(node, int | float) and not isinstance(node, bool):
        return float(node), "precomputed fraction"
    if not isinstance(node, dict):
        return None
    if "pass_fraction" in node and node["pass_fraction"] is not None:
        return float(node["pass_fraction"]), "pass_fraction"
    if "ran" in node or "total" in node or "passed" in node:
        total = int(node.get("total") or 0)
        if not node.get("ran") or total <= 0:
            return 0.0, node.get("reason") or "suite did not run"
        passed = int(node.get("passed") or 0)
        return passed / total, f"{passed}/{total} passed"
    return None  # unrecognizable dict: skip rather than score a phantom zero
