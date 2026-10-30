from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score

_SINGLE_FIELD_PATH = re.compile(r"(?:\$\.)?([A-Za-z_]\w*)(?:\.(?:-?\w+|\*)|\[[^\]$]*\])*")
_SCORING_METHODS = ("measure", "a_measure", "measure_runs", "a_measure_runs", "measure_dataset")


def _overrides_concrete(cls: type, method: str) -> bool:
    """Whether ``cls`` replaces an inherited implementation rather than an abstract hook."""
    if method not in cls.__dict__:
        return False
    inherited = next((base.__dict__[method] for base in cls.__mro__[1:] if method in base.__dict__), None)
    return inherited is not None and not getattr(inherited, "__isabstractmethod__", False)


class Dimension(str, Enum):
    """The five evaluation dimensions — every metric belongs to exactly one.

    See ADR-009 for rationale. Adding a new dimension requires an ADR and
    community consensus.
    """

    CORRECTNESS = "correctness"
    GROUNDEDNESS = "groundedness"
    SAFETY = "safety"
    TRAJECTORY = "trajectory"
    PERFORMANCE = "performance"


class MetricInput(str, Enum):
    """A top-level :class:`EvalCase` field a metric may read.

    Nested data travels with its parent field: per-turn latency, tokens, and
    retrieval context are part of ``MESSAGES``; repeated-run fields are part
    of ``RUNS``.
    """

    INPUT = "input"
    OUTPUT = "output"
    EXPECTED = "expected"
    CONTEXT = "context"
    LATENCY_MS = "latency_ms"
    TOKEN_COUNT = "token_count"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    COST_USD = "cost_usd"
    RETRY_COUNT = "retry_count"
    CONFIDENCE = "confidence"
    MESSAGES = "messages"
    TOOL_CALLS = "tool_calls"
    EXPECTED_TOOLS = "expected_tools"
    EXPECTED_TOOL_CALLS = "expected_tool_calls"
    TAGS = "tags"
    METADATA = "metadata"
    RUNS = "runs"

    @classmethod
    def from_path(cls, path: str) -> MetricInput | None:
        """Return the field an ``EvalCase.to_dict()`` path reads.

        Accepts the dot and JSONPath forms used by configurable metrics, such
        as ``"output"``, ``"metadata.user.id"``, and ``"$.messages[0]"``.
        Returns ``None`` unless the whole path descends from one named field,
        for example for ``"$..id"``, ``"$.*"``, or ``"$.output | $.expected"``.
        """
        match = _SINGLE_FIELD_PATH.fullmatch(path)
        if match is None:
            return None
        try:
            return cls(match.group(1))
        except ValueError:
            return None


def _coerce_inputs(values: Iterable[MetricInput | str], argument: str) -> frozenset[MetricInput]:
    if isinstance(values, str):
        raise TypeError(f"{argument} must be an iterable of MetricInput values, not a string")
    return frozenset(MetricInput(value) for value in values)


def _inputs_from_paths(paths: Iterable[str], argument: str) -> frozenset[MetricInput] | None:
    if isinstance(paths, str):
        raise TypeError(f"{argument} must be an iterable of paths, not a string")
    inputs = set()
    for path in paths:
        field = MetricInput.from_path(path)
        if field is None:
            return None
        inputs.add(field)
    return frozenset(inputs)


@dataclass(frozen=True, init=False)
class MetricInputs:
    """The :class:`EvalCase` fields a metric reads.

    All ``required`` fields and at least one all-of group in ``any_of`` must
    be available for scope eligibility. An empty ``any_of`` adds no constraint.
    ``optional`` fields are read when present. Adapters materialize every field,
    including all alternatives, not just the first satisfiable group.
    These declarations describe field dependencies, not value-level validation:
    eligibility does not guarantee a particular case contains usable data.
    """

    required: frozenset[MetricInput]
    optional: frozenset[MetricInput]
    any_of: tuple[frozenset[MetricInput], ...]

    def __init__(
        self,
        required: Iterable[MetricInput | str] = (),
        optional: Iterable[MetricInput | str] = (),
        *,
        any_of: Iterable[Iterable[MetricInput | str]] = (),
    ) -> None:
        required_inputs = _coerce_inputs(required, "required")
        optional_inputs = _coerce_inputs(optional, "optional")
        if overlap := required_inputs & optional_inputs:
            names = ", ".join(sorted(field.value for field in overlap))
            raise ValueError(f"inputs cannot be both required and optional: {names}")
        if isinstance(any_of, str):
            raise TypeError("any_of must be an iterable of field groups, not a string")
        groups = {_coerce_inputs(group, "any_of group") - required_inputs for group in any_of}
        if frozenset() in groups:
            # An empty alternative is always satisfiable. Keep the other
            # alternatives materialized even though they impose no constraint.
            optional_inputs |= frozenset().union(*groups) - required_inputs
            groups = set()
        normalized = tuple(sorted(groups, key=lambda group: tuple(sorted(field.value for field in group))))
        object.__setattr__(self, "required", required_inputs)
        object.__setattr__(self, "optional", optional_inputs)
        object.__setattr__(self, "any_of", normalized)

    @classmethod
    def unknown(cls) -> MetricInputs:
        """Treat every field as required; the default for undeclared metrics."""
        return cls(required=MetricInput)

    @classmethod
    def from_paths(
        cls,
        required: Iterable[str] = (),
        optional: Iterable[str] = (),
        *,
        any_of: Iterable[Iterable[str]] = (),
    ) -> MetricInputs:
        """Declare inputs from ``EvalCase.to_dict()`` paths.

        Any path that does not resolve to a single field makes every field
        required, because the metric could read anything.
        """
        required_inputs = _inputs_from_paths(required, "required")
        optional_inputs = _inputs_from_paths(optional, "optional")
        if isinstance(any_of, str):
            raise TypeError("any_of must be an iterable of path groups, not a string")
        groups = [_inputs_from_paths(group, "any_of group") for group in any_of]
        if required_inputs is None or optional_inputs is None or any(group is None for group in groups):
            return cls.unknown()
        return cls(
            required=required_inputs,
            optional=optional_inputs - required_inputs,
            any_of=(group for group in groups if group is not None),
        )

    @classmethod
    def union(cls, *declarations: MetricInputs) -> MetricInputs:
        """Require every declaration, distributing their alternatives with a Cartesian product."""
        required = frozenset().union(*(declaration.required for declaration in declarations))
        optional = frozenset().union(*(declaration.optional for declaration in declarations))
        alternatives: set[frozenset[MetricInput]] = {frozenset()}
        for declaration in declarations:
            if declaration.any_of:
                alternatives = {left | right for left in alternatives for right in declaration.any_of}
        return cls(required=required, optional=optional - required, any_of=alternatives)

    @property
    def fields(self) -> frozenset[MetricInput]:
        """Every field the metric may read."""
        return self.required | self.optional | frozenset().union(*self.any_of)


class EvaluationMode(str, Enum):
    """Execution modes in which a metric can be evaluated."""

    OFFLINE = "offline"
    ONLINE = "online"


class OnlineScope(str, Enum):
    """Target cardinality for an online evaluation."""

    TRACE = "trace"
    SPAN = "span"
    GROUP = "group"


@dataclass(frozen=True)
class MetricApplicability:
    """Execution contexts a metric can support.

    Offline capability and online target scope are separate axes: a metric can
    run offline without being eligible for online scoring, and an online metric
    can support one or more target scopes.
    """

    modes: frozenset[EvaluationMode] = frozenset({EvaluationMode.OFFLINE})
    online_scopes: frozenset[OnlineScope] = frozenset()

    def __post_init__(self) -> None:
        has_online_mode = EvaluationMode.ONLINE in self.modes
        if self.online_scopes and not has_online_mode:
            raise ValueError("online_scopes requires EvaluationMode.ONLINE")
        if has_online_mode and not self.online_scopes:
            raise ValueError("EvaluationMode.ONLINE requires at least one online scope")


class BaseMetric(ABC):
    """Base class for all evaluation metrics.

    Subclasses implement ``measure()`` (sync) which takes an EvalCase and
    returns a Score — or ``None`` to signal that the metric does not apply
    to this particular case (e.g. missing metadata).  Skipped (``None``)
    results are excluded from aggregation and do not count as failures.

    For I/O-bound metrics (LLM-judged), override ``a_measure()`` instead —
    the default calls ``measure()`` synchronously.

    Declare the ``EvalCase`` fields the metric reads with a class-level
    ``eval_inputs``; assign ``self.eval_inputs`` in ``__init__`` when they
    depend on configuration. Undeclared metrics are treated as reading every
    field. A subclass that overrides a concrete scoring method without
    declaring ``eval_inputs`` reverts to that default rather than inheriting
    its parent's declaration; redeclare ``eval_inputs`` when overriding any
    other method that reads the eval case.

    ``catalog_applicability`` optionally advertises contexts supported by at
    least one valid configuration, for discovery only. It does not change a
    configured instance's input-derived ``applicability``. Subclasses that
    change input declarations or override scoring must redeclare this override
    if it still applies.
    """

    eval_inputs: MetricInputs = MetricInputs.unknown()
    catalog_applicability: MetricApplicability | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        declared = cls.__dict__.get("eval_inputs")
        if declared is None:
            if any(_overrides_concrete(cls, method) for method in _SCORING_METHODS):
                cls.eval_inputs = MetricInputs.unknown()
        elif not isinstance(declared, MetricInputs):
            raise TypeError(
                f"{cls.__qualname__}.eval_inputs must be a MetricInputs instance, got {type(declared).__name__}"
            )
        if "eval_inputs" in cls.__dict__ and "catalog_applicability" not in cls.__dict__:
            cls.catalog_applicability = None

    def __init__(self, name: str, dimension: Dimension, threshold: float = 1.0, **kwargs: object) -> None:
        self.name = name
        self.dimension = dimension
        self.threshold = threshold

    def _configure_inputs(self, owner: type[BaseMetric], inputs: MetricInputs) -> None:
        """Replace ``owner``'s class-level ``eval_inputs`` for this instance's configuration.

        Subclasses that declare their own inputs, or that reverted to the
        default by overriding a scoring method, keep their declaration.
        """
        declaration_owner = next(base for base in type(self).__mro__ if "eval_inputs" in base.__dict__)
        if declaration_owner is owner:
            self.eval_inputs = inputs

    @abstractmethod
    def measure(self, eval_case: EvalCase) -> Score | None:
        """Evaluate the case and return a Score, or ``None`` to skip."""
        ...

    async def a_measure(self, eval_case: EvalCase) -> Score | None:
        """Async variant. Override for I/O-bound metrics (LLM-judged). Default calls measure()."""
        return self.measure(eval_case)

    @property
    def applicability(self) -> MetricApplicability:
        """Return conservative scope eligibility, not validation of a particular case."""
        from harness_evals.applicability import applicability_for_inputs

        return applicability_for_inputs(self.eval_inputs)

    def measure_dataset(self, cases: list[EvalCase], outcomes: list[bool]) -> Score | None:
        """Evaluate across a full dataset with known outcomes.

        Override for dataset-level metrics (e.g. calibration, discrimination)
        that need the complete set of cases to compute a meaningful score.
        Returns ``None`` by default, signalling the runner should fall back
        to per-case evaluation.
        """
        return None


class SafetyMetric(BaseMetric):
    """Marker base class for safety metrics.

    Safety metrics are reported separately and never averaged into an overall
    score (hard-constraint design per Rabanser et al.). Use ``isinstance(m,
    SafetyMetric)`` to identify safety metrics programmatically.

    Defaults to ``Dimension.SAFETY`` — subclasses do not need to pass it.
    """

    def __init__(
        self, name: str, threshold: float = 1.0, *, dimension: Dimension = Dimension.SAFETY, **kwargs: object
    ) -> None:
        super().__init__(name=name, dimension=dimension, threshold=threshold, **kwargs)


class ReliabilityMetric(BaseMetric):
    """Base class for metrics that evaluate across multiple runs.

    Expects ``eval_case.runs`` to contain K repeated runs of the same task.
    Subclasses implement ``measure_runs()`` instead of ``measure()``.
    """

    eval_inputs = MetricInputs(required={MetricInput.RUNS})

    def __init__(self, name: str, dimension: Dimension, threshold: float = 1.0, k: int = 5, **kwargs: object) -> None:
        super().__init__(name=name, dimension=dimension, threshold=threshold, **kwargs)
        self.k = k

    @abstractmethod
    def measure_runs(self, eval_case: EvalCase) -> Score | None:
        """Evaluate across eval_case.runs. Called by measure() when runs are present."""
        ...

    async def a_measure_runs(self, eval_case: EvalCase) -> Score | None:
        """Async variant of measure_runs(). Default calls measure_runs()."""
        return self.measure_runs(eval_case)

    def measure(self, eval_case: EvalCase) -> Score | None:
        if eval_case.runs:
            return self.measure_runs(eval_case)
        return Score(
            name=self.name,
            value=0.0,
            threshold=self.threshold,
            reason=f"No runs provided — cannot evaluate reliability (expected {self.k} repeated runs)",
        )

    async def a_measure(self, eval_case: EvalCase) -> Score | None:
        if eval_case.runs:
            return await self.a_measure_runs(eval_case)
        return Score(
            name=self.name,
            value=0.0,
            threshold=self.threshold,
            reason=f"No runs provided — cannot evaluate reliability (expected {self.k} repeated runs)",
        )
