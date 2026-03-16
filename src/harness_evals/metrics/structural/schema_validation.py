from __future__ import annotations

import json
from typing import Any

import jsonschema

from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.test_case import TestCase


class SchemaValidationMetric(BaseMetric):
    """Score 1.0 if actual_output conforms to the JSON schema in expected_output.

    expected_output must be a JSON Schema (dict or JSON string).
    actual_output is the data to validate (dict, list, or JSON string).
    """

    def __init__(self, threshold: float = 1.0, **kwargs: object) -> None:
        super().__init__(name="schema_validation", threshold=threshold, **kwargs)

    def _parse(self, value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        return value

    def measure(self, test_case: TestCase) -> Score:
        try:
            data = self._parse(test_case.actual_output)
            schema = self._parse(test_case.expected_output)
        except (json.JSONDecodeError, TypeError) as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                success=False,
                reason=f"JSON parse error: {e}",
            )

        try:
            jsonschema.validate(instance=data, schema=schema)
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                success=True,
            )
        except jsonschema.ValidationError as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                success=False,
                reason=f"Validation failed: {e.message}",
            )
        except jsonschema.SchemaError as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                success=False,
                reason=f"Invalid schema: {e.message}",
            )
