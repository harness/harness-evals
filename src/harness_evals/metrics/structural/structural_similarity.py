from __future__ import annotations

import json
import logging
import importlib.metadata
from typing import Any

import yaml
from deepdiff import DeepDiff
import jsonschema

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.utils.path import extract_path

logger = logging.getLogger(__name__)


def _load_entry_point(group: str, name: str) -> Any:
    eps = importlib.metadata.entry_points(group=group)
    for ep in eps:
        if ep.name == name:
            return ep.load()
    raise ValueError(f"Entry point {name} not found in group {group}")


class StructuralSimilarityMetric(BaseMetric):
    """Configurable structural similarity between outputs (YAML/JSON).

    Supports raw deepdiff, structural (with keys ignored/required fields),
    and schema validation via JSON Schema or entry point validators.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        format: str = "yaml",  # yaml | json
        level: str = "raw",    # raw | structural | schema_validated
        ignore_keys: list[str] | None = None,
        required_fields: dict[str, Any] | None = None,
        extra_keys: str = "ignore", # ignore | penalize
        extra_keys_penalty: float = 0.8,  # multiplier when extra_keys=penalize
        schema_validator: dict[str, Any] | None = None,
        expected_field: str | None = None,
        output_field: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(name="structural_similarity", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.format = format.lower()
        self.level = level.lower()
        self.ignore_keys = ignore_keys or []
        self.required_fields = required_fields or {}
        self.extra_keys = extra_keys.lower()
        self.extra_keys_penalty = extra_keys_penalty
        self.schema_validator = schema_validator
        self.expected_field = expected_field
        self.output_field = output_field

    def _parse(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
            
        try:
            if self.format == "json":
                return json.loads(value)
            elif self.format == "yaml":
                # PyYAML safe_load
                return yaml.safe_load(value)
        except Exception:
            return value  # If parse fails, return raw string to be caught later

        return value

    def _get_field(self, eval_case: EvalCase, field_path: str | None, default_field: str) -> Any:
        """Extract field from EvalCase.
        
        If field_path resolves to None but the default field (output or expected)
        is a non-null string, fall back to the string (handles online mode).
        """
        raw_val = getattr(eval_case, default_field, None)
        
        if not field_path:
            return raw_val
            
        case_dict = eval_case.to_dict()
        extracted = extract_path(case_dict, field_path)
        
        if extracted is None and isinstance(raw_val, str):
            return raw_val
            
        return extracted

    def _validate_schema(self, value: Any, raw_string: str) -> tuple[bool, str | None]:
        """Run schema validation if configured."""
        if not self.schema_validator:
            return True, None

        val_type = self.schema_validator.get("type")
        
        if val_type == "json_schema":
            schema = self.schema_validator.get("schema")
            if not schema:
                return False, "JSON Schema config missing 'schema'"
            try:
                jsonschema.validate(instance=value, schema=schema)
                return True, None
            except jsonschema.exceptions.ValidationError as e:
                return False, f"Schema validation failed: {e.message}"
                
        elif val_type == "entry_point":
            name = self.schema_validator.get("name")
            if not name:
                return False, "Entry point config missing 'name'"
            try:
                validator_fn = _load_entry_point("harness_evals.validators", name)
                # Entry point validators expect the raw string
                result = validator_fn(raw_string)
                if not result.get("valid"):
                    errors = result.get("errors", ["Unknown validation error"])
                    return False, f"Schema validation failed: {', '.join(errors)}"
                return True, None
            except Exception as e:
                return False, f"Validator execution failed: {e}"

        return False, f"Unknown schema_validator type: {val_type}"

    def measure(self, eval_case: EvalCase) -> Score:
        raw_actual = self._get_field(eval_case, self.output_field, "output")
        raw_expected = self._get_field(eval_case, self.expected_field, "expected")

        if raw_actual is None or raw_expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Missing actual or expected output",
            )

        # 1. Level: schema_validated
        if self.level == "schema_validated":
            is_valid, err = self._validate_schema(
                self._parse(raw_actual) if isinstance(raw_actual, str) else raw_actual,
                raw_actual if isinstance(raw_actual, str) else json.dumps(raw_actual)
            )
            if not is_valid:
                return Score(
                    name=self.name,
                    value=0.0,
                    threshold=self.threshold,
                    reason=err,
                )

        # 2. Parse for structural comparison
        actual = self._parse(raw_actual)
        expected = self._parse(raw_expected)

        if isinstance(raw_actual, str) and isinstance(actual, str):
            # Parse failed
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Failed to parse actual output as {self.format}",
            )

        # 3. Level: structural (ignore keys, check required fields)
        exclude_paths = None
        if self.level in ("structural", "schema_validated"):
            if self.ignore_keys:
                # convert to deepdiff exclude_paths (e.g. root['name'])
                # This is simplified; true implementation might use regex or deeper paths
                exclude_paths = [f"root['{k}']" for k in self.ignore_keys] + [
                    f"root{path}" for path in self.ignore_keys if path.startswith("[")
                ]

        # 4. DeepDiff
        diff = DeepDiff(
            expected,
            actual,
            ignore_order=True,
            exclude_paths=exclude_paths,
            get_deep_distance=True,
            threshold_to_diff_deeper=0,
        )

        distance = diff.get("deep_distance", 0.0)
        value = max(0.0, min(1.0, 1.0 - distance))

        # Handle extra keys penalization
        if self.level in ("structural", "schema_validated") and self.extra_keys == "penalize":
            if "dictionary_item_added" in diff:
                value *= self.extra_keys_penalty

        reason = None
        if diff:
            changes = {k: v for k, v in diff.items() if k != "deep_distance"}
            if changes:
                reason = f"Differences: {list(changes.keys())}"

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reason,
        )
