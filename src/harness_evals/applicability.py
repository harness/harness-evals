"""Conservative scope eligibility, derived from declared field dependencies.

Every metric can run offline against authored eval cases. A metric is eligible
for an online scope when its adapter can supply all declared required fields.
This is a source capability check, not proof that a particular trace contains
usable values or authored metadata keys. Metrics retain their own missing-data
behavior. Alternative input sources can make declarations conservative; see
``MetricInputs``. The Group adapter assembles message-scoped evidence, but
Group metric compatibility and actual-input sufficiency require a separate
reviewed gate. Group is deliberately not advertised by this capability table.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from harness_evals.core.metric import EvaluationMode, MetricApplicability, MetricInput, MetricInputs, OnlineScope

TRACE_INPUTS: frozenset[MetricInput] = frozenset(
    {
        MetricInput.INPUT,
        MetricInput.OUTPUT,
        MetricInput.MESSAGES,
        MetricInput.TOOL_CALLS,
        MetricInput.LATENCY_MS,
        MetricInput.TOKEN_COUNT,
        MetricInput.INPUT_TOKENS,
        MetricInput.OUTPUT_TOKENS,
        MetricInput.COST_USD,
    }
)
"""EvalCase fields the trace adapters populate from spans.

``metadata`` is excluded: adapted metadata carries trace correlation labels,
not the authored keys that metrics read from it.
"""

ONLINE_SCOPE_INPUTS: Mapping[OnlineScope, frozenset[MetricInput]] = MappingProxyType(
    {
        OnlineScope.TRACE: TRACE_INPUTS,
        OnlineScope.SPAN: TRACE_INPUTS,
    }
)
"""EvalCase fields each online scope's adapter supplies."""


def applicability_for_inputs(inputs: MetricInputs) -> MetricApplicability:
    """Return scope eligibility from adapter capabilities, without inspecting case values."""
    scopes = frozenset(scope for scope, supplied in ONLINE_SCOPE_INPUTS.items() if inputs.required <= supplied)
    if not scopes:
        return MetricApplicability()
    return MetricApplicability(
        modes=frozenset({EvaluationMode.OFFLINE, EvaluationMode.ONLINE}),
        online_scopes=scopes,
    )
