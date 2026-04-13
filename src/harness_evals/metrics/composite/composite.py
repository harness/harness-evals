from __future__ import annotations

import json
import logging
from typing import Any

import yaml

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.metrics.composite.operators import OPERATORS, OperatorError
from harness_evals.utils.path import extract_path

logger = logging.getLogger(__name__)


class CompositeMetric(BaseMetric):
    """Declarative composite metric supporting weighted sub-checks."""

    def __init__(
        self,
        sub_scores: list[dict[str, Any]],
        output_format: str = "json",  # json | yaml
        threshold: float = 0.85,
        **kwargs: object,
    ) -> None:
        super().__init__(name="composite", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        for i, sub in enumerate(sub_scores):
            if "name" not in sub:
                raise ValueError(f"sub_scores[{i}] missing required 'name' key")
            if "check" not in sub:
                raise ValueError(f"sub_scores[{i}] ({sub['name']}) missing required 'check' key")
            if "type" not in sub.get("check", {}):
                raise ValueError(f"sub_scores[{i}] ({sub['name']}) check missing required 'type' key")
        self.sub_scores = sub_scores
        self.output_format = output_format.lower()

    def _parse(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            if self.output_format == "json":
                return json.loads(value)
            elif self.output_format == "yaml":
                return yaml.safe_load(value)
        except Exception:
            return value
        return value

    def measure(self, eval_case: EvalCase) -> Score:
        raw_actual = getattr(eval_case, "output", None)
        raw_expected = getattr(eval_case, "expected", None)

        if raw_actual is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Missing actual output",
            )

        actual = self._parse(raw_actual)
        expected = self._parse(raw_expected)

        if isinstance(raw_actual, str) and isinstance(actual, str) and self.output_format in ("json", "yaml"):
            # Parse failed
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Failed to parse actual output as {self.output_format}",
            )

        eval_case_dict = eval_case.to_dict()
        eval_case_dict["output"] = actual
        eval_case_dict["expected"] = expected

        details = {"sub_scores": {}, "effective_weights": {}}
        total_score = 0.0
        active_weight_sum = 0.0

        for sub in self.sub_scores:
            sub_name = sub.get("name", "unknown")
            weight = float(sub.get("weight", 0.0))
            check_config = sub.get("check", {})
            op_type = check_config.get("type")
            skip_when_missing = sub.get("skip_when_missing", False)

            if not op_type or op_type not in OPERATORS:
                details["sub_scores"][sub_name] = {
                    "value": 0.0,
                    "status": "error",
                    "reason": f"Unknown operator type: {op_type}",
                }
                active_weight_sum += weight
                continue

            operator_fn = OPERATORS[op_type]

            # Handle skip_when_missing explicitly if a specific field is defined
            # If the check specifies a main "field", see if it's missing
            main_field = check_config.get("field")
            if main_field and skip_when_missing:
                val = extract_path(eval_case_dict, main_field)
                if val is None:
                    details["sub_scores"][sub_name] = {
                        "value": None,
                        "status": "skipped",
                        "reason": "field missing, skip_when_missing=true",
                    }
                    continue

            try:
                val = operator_fn(eval_case_dict, check_config)
                details["sub_scores"][sub_name] = {
                    "value": val,
                    "status": "ok",
                }
                total_score += val * weight
            except OperatorError as e:
                details["sub_scores"][sub_name] = {
                    "value": 0.0,
                    "status": "error",
                    "reason": str(e),
                }
            except Exception as e:
                details["sub_scores"][sub_name] = {
                    "value": 0.0,
                    "status": "error",
                    "reason": f"Unexpected error: {e}",
                }

            active_weight_sum += weight

        final_score = 0.0
        if active_weight_sum > 0:
            final_score = total_score / active_weight_sum

        # Calculate effective weights
        if active_weight_sum > 0:
            for sub in self.sub_scores:
                sub_name = sub.get("name", "unknown")
                if details["sub_scores"][sub_name]["status"] != "skipped":
                    orig_weight = float(sub.get("weight", 0.0))
                    details["effective_weights"][sub_name] = round(orig_weight / active_weight_sum, 4)

        return Score(
            name=self.name,
            value=final_score,
            threshold=self.threshold,
            metadata=details,
        )
