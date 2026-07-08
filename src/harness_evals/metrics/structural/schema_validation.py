from __future__ import annotations

import json

import jsonschema

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class SchemaValidationMetric(BaseMetric):
    """Score 1.0 if output conforms to a JSON Schema.

    The schema is passed via the constructor, keeping ``expected`` free
    for its standard meaning (ground truth answer).
    ``output`` is the data to validate (dict, list, or JSON string).
    """

    def __init__(self, schema: dict | str, threshold: float = 1.0, **kwargs: object) -> None:
        super().__init__(name="schema_validation", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        if isinstance(schema, str):
            schema = json.loads(schema)
        self.schema = schema

    def measure(self, eval_case: EvalCase) -> Score:
        try:
            data = json.loads(eval_case.output) if isinstance(eval_case.output, str) else eval_case.output
        except (json.JSONDecodeError, TypeError) as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Output could not be parsed as valid JSON ({e})",
            )

        try:
            jsonschema.validate(instance=data, schema=self.schema)
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason="Output conforms to the expected JSON Schema",
            )
        except jsonschema.ValidationError as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Validation failed — output does not conform to the expected schema ({e.message})",
            )
        except jsonschema.SchemaError as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"The provided JSON Schema itself is invalid ({e.message})",
            )
