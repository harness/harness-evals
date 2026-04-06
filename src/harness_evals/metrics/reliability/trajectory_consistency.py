"""TrajectoryConsistency metric — action-path similarity across K runs (Rabanser et al.)."""

from __future__ import annotations

import math
from collections import Counter

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import Dimension, ReliabilityMetric
from harness_evals.core.score import Score


def _cosine_similarity(a: Counter, b: Counter) -> float:
    """Cosine similarity between two Counter vectors.

    Two empty vectors are treated as identical (returns 1.0).
    If only one is empty, returns 0.0.
    """
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0 if (norm_a == 0.0 and norm_b == 0.0) else 0.0
    return dot / (norm_a * norm_b)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of longest common subsequence (dynamic programming).

    Orients the shorter sequence on the inner axis and uses two
    pre-allocated rows swapped in-place (no per-row allocation).
    Space: O(min(m, n)).
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    if n > m:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def _normalized_lcs(a: list[str], b: list[str]) -> float:
    """LCS normalized by the maximum trajectory length."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return _lcs_length(a, b) / max_len


class TrajectoryConsistencyMetric(ReliabilityMetric):
    """Action-path similarity across K runs.

    Each run in ``eval_case.runs`` must have ``tool_calls`` — a list of
    ``ToolCall`` objects.  Tool names are extracted to form the trajectory.

    Two modes:

    - ``"distributional"`` (default): Cosine similarity of action frequency
      histograms, averaged over all ``K*(K-1)/2`` pairs.  Maps to C_traj_d
      from Rabanser et al.
    - ``"sequential"``: Longest Common Subsequence normalized by max
      trajectory length, averaged pairwise.  Maps to C_traj_s.

    For long trajectories, set ``max_trajectory_length`` to truncate each
    trajectory to its last *N* actions before comparison.  This bounds
    sequential-mode cost to O(K^2 * L^2) where L is the cap.
    """

    def __init__(
        self,
        mode: str = "distributional",
        threshold: float = 0.8,
        k: int = 5,
        max_trajectory_length: int | None = None,
        **kwargs: object,
    ) -> None:
        if mode not in ("distributional", "sequential"):
            raise ValueError(f"mode must be 'distributional' or 'sequential', got '{mode}'")
        if max_trajectory_length is not None and max_trajectory_length < 1:
            raise ValueError(f"max_trajectory_length must be >= 1 or None, got {max_trajectory_length}")
        super().__init__(name="trajectory_consistency", dimension=Dimension.TRAJECTORY, threshold=threshold, k=k, **kwargs)
        self.mode = mode
        self.max_trajectory_length = max_trajectory_length

    def _truncate(self, trajectory: list[str]) -> list[str]:
        if self.max_trajectory_length is not None and len(trajectory) > self.max_trajectory_length:
            return trajectory[-self.max_trajectory_length :]
        return trajectory

    def measure_runs(self, eval_case: EvalCase) -> Score:
        runs = eval_case.runs or []

        trajectories: list[list[str]] = []
        for run in runs:
            if run.tool_calls is not None:
                traj = [tc.name for tc in run.tool_calls]
                trajectories.append(self._truncate(traj))

        if len(trajectories) < 2:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Need at least 2 runs with trajectories, got {len(trajectories)}",
            )

        n = len(trajectories)
        pair_scores: list[float] = []

        if self.mode == "distributional":
            counters = [Counter(t) for t in trajectories]
            for i in range(n):
                for j in range(i + 1, n):
                    pair_scores.append(_cosine_similarity(counters[i], counters[j]))
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    pair_scores.append(_normalized_lcs(trajectories[i], trajectories[j]))

        value = max(0.0, min(1.0, sum(pair_scores) / len(pair_scores)))

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            metadata={
                "mode": self.mode,
                "k": n,
                "n_pairs": len(pair_scores),
            },
        )
