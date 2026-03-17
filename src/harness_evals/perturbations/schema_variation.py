"""Schema variation perturbation — add/remove optional fields, change casing."""

from __future__ import annotations

import json
import random

from harness_evals.perturbations.base import BasePerturbation


class SchemaVariation(BasePerturbation):
    """Produce JSON variants with schema-level changes.

    Applies one or more of:
    - Remove a random optional field
    - Add an extra unknown field
    - Change key casing (camelCase <-> snake_case)
    - Wrap/unwrap values in arrays

    Tests whether an agent is sensitive to minor schema differences.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed

    async def perturb(self, input_text: str, n: int = 5) -> list[str]:
        try:
            obj = json.loads(input_text)
        except (json.JSONDecodeError, TypeError):
            return [input_text] * n

        if not isinstance(obj, dict):
            return [input_text] * n

        rng = random.Random(self._seed)
        results: list[str] = []

        transforms = [
            self._remove_field,
            self._add_extra_field,
            self._change_casing,
            self._wrap_value_in_array,
        ]

        for i in range(n):
            modified = dict(obj)
            transform = transforms[i % len(transforms)]
            modified = transform(modified, rng)
            results.append(json.dumps(modified, ensure_ascii=False))

        return results

    def _remove_field(self, obj: dict, rng: random.Random) -> dict:
        if len(obj) <= 1:
            return obj
        result = dict(obj)
        key = rng.choice(list(result.keys()))
        del result[key]
        return result

    def _add_extra_field(self, obj: dict, rng: random.Random) -> dict:
        result = dict(obj)
        extra_key = f"_extra_{rng.randint(0, 999)}"
        result[extra_key] = "unknown"
        return result

    def _change_casing(self, obj: dict, rng: random.Random) -> dict:
        result = {}
        for k, v in obj.items():
            if "_" in k:
                # snake_case -> camelCase
                parts = k.split("_")
                new_key = parts[0] + "".join(p.capitalize() for p in parts[1:])
            else:
                # camelCase -> snake_case
                new_key = ""
                for c in k:
                    if c.isupper() and new_key:
                        new_key += "_" + c.lower()
                    else:
                        new_key += c.lower()
            result[new_key] = v
        return result

    def _wrap_value_in_array(self, obj: dict, rng: random.Random) -> dict:
        if not obj:
            return obj
        result = dict(obj)
        key = rng.choice(list(result.keys()))
        val = result[key]
        if isinstance(val, list):
            if val:
                result[key] = val[0]
        else:
            result[key] = [val]
        return result
