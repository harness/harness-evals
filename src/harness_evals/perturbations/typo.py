"""Typo injection perturbation — inject realistic typos at configurable rate."""

from __future__ import annotations

import random

from harness_evals.perturbations.base import BasePerturbation

# Common typo patterns: (original char, replacement char)
_KEYBOARD_NEIGHBORS: dict[str, str] = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "rw", "f": "dg",
    "g": "fh", "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k;",
    "m": "n,", "n": "bm", "o": "ip", "p": "o[", "q": "wa", "r": "et",
    "s": "ad", "t": "ry", "u": "yi", "v": "cb", "w": "qe", "x": "zc",
    "y": "tu", "z": "x",
}


class TypoInjection(BasePerturbation):
    """Inject realistic typos into text at a configurable rate.

    Applies character-level perturbations:
    - Swap adjacent characters
    - Replace with keyboard neighbor
    - Drop a character
    - Double a character

    Tests whether an agent is robust to minor input typos.
    """

    def __init__(self, rate: float = 0.05, seed: int | None = None) -> None:
        """Args:
            rate: Fraction of characters to perturb (0.0-1.0). Default 0.05 (5%).
            seed: Random seed for reproducibility.
        """
        self.rate = max(0.0, min(1.0, rate))
        self._seed = seed

    async def perturb(self, input_text: str, n: int = 5) -> list[str]:
        if not input_text:
            return [input_text] * n

        rng = random.Random(self._seed)
        results: list[str] = []

        for _ in range(n):
            chars = list(input_text)
            num_typos = max(1, int(len(chars) * self.rate))

            positions = rng.sample(range(len(chars)), min(num_typos, len(chars)))

            for pos in sorted(positions, reverse=True):
                op = rng.choice(["swap", "neighbor", "drop", "double"])

                if op == "swap" and pos < len(chars) - 1:
                    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                elif op == "neighbor":
                    c = chars[pos].lower()
                    neighbors = _KEYBOARD_NEIGHBORS.get(c)
                    if neighbors:
                        replacement = rng.choice(list(neighbors))
                        chars[pos] = replacement if chars[pos].islower() else replacement.upper()
                elif op == "drop":
                    chars.pop(pos)
                elif op == "double":
                    chars.insert(pos, chars[pos])

            results.append("".join(chars))

        return results
