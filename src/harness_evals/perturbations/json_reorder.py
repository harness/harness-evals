"""JSON field reordering perturbation — deterministic, zero dependencies."""

from __future__ import annotations

import json
import random

from harness_evals.perturbations.base import BasePerturbation


class JsonFieldReorder(BasePerturbation):
    """Produce JSON variants with reordered fields.

    Tests whether an agent is sensitive to the order of keys in JSON input.
    Each perturbation shuffles the top-level keys (and optionally nested keys)
    into a different order.
    """

    def __init__(self, seed: int | None = None, recursive: bool = False) -> None:
        self._seed = seed
        self._recursive = recursive

    async def perturb(self, input_text: str, n: int = 5) -> list[str]:
        try:
            obj = json.loads(input_text)
        except (json.JSONDecodeError, TypeError):
            return [input_text] * n

        if not isinstance(obj, dict):
            return [input_text] * n

        rng = random.Random(self._seed)
        results: list[str] = []
        seen: set[str] = set()

        for _ in range(n * 10):  # try extra times to get unique permutations
            if len(results) >= n:
                break
            reordered = self._shuffle_dict(obj, rng)
            serialized = json.dumps(reordered, ensure_ascii=False)
            if serialized not in seen:
                seen.add(serialized)
                results.append(serialized)

        # If we can't get enough unique permutations, pad with what we have
        while len(results) < n:
            results.append(results[len(results) % max(1, len(results) - 1)] if results else input_text)

        return results[:n]

    def _shuffle_dict(self, obj: dict, rng: random.Random) -> dict:
        keys = list(obj.keys())
        rng.shuffle(keys)
        result = {}
        for k in keys:
            v = obj[k]
            if self._recursive and isinstance(v, dict):
                result[k] = self._shuffle_dict(v, rng)
            else:
                result[k] = v
        return result
