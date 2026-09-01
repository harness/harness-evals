"""Shared validation for negated deterministic comparison metrics.

``contains``, ``exact_match``, and ``regex`` all accept ``negate``/``forbidden``
with identical rules. Keeping the checks here stops the three from drifting.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ``build_metric()`` declares ``threshold: float = 0.0`` as an "unspecified"
# sentinel and does not forward a threshold to composite sub-metrics, so a
# negated metric can legitimately arrive here with 0.0. Scores are bounded to
# [0.0, 1.0], so a negated metric at 0.0 would pass on every input — including
# the one it exists to reject. Fall back to the strict default instead of
# rejecting a supported composition.
DEFAULT_NEGATED_THRESHOLD = 1.0


def resolve_negation(
    *,
    negate: bool,
    forbidden: object,
    threshold: float,
    allow_empty_forbidden: bool = False,
) -> tuple[str | None, float]:
    """Validate negation options and return ``(forbidden, effective_threshold)``.

    ``allow_empty_forbidden`` is set by ``exact_match``, where ``forbidden: ""``
    is the meaningful "output must not be empty" assertion. For substring and
    pattern checks an empty value would match everything, so it is rejected.
    """
    # Truthiness is not enough: a Harness pipeline expression interpolates to a
    # string, so ``negate: <+...>`` rendering to "false" would enable negation.
    # Nothing upstream catches it — ``_validate_constructor_options`` checks
    # option names, not types, and ``normalize_metric_config`` only diffs keys.
    # Reject rather than coerce, matching the ``forbidden`` guard below.
    if not isinstance(negate, bool):
        raise ValueError(
            f"negate must be a boolean, got {type(negate).__name__}. "
            'Do not quote the value in YAML (negate: true, not negate: "true").'
        )

    if not negate:
        if forbidden is not None:
            raise ValueError("forbidden is only valid when negate=True")
        return None, threshold

    if forbidden is None:
        raise ValueError("forbidden must be provided when negate=True")

    # A non-string slips through ``_validate_constructor_options`` (it checks
    # option names, not types) and ``normalize_metric_config`` (advisory only).
    # Left un-coerced it would compare unequal to every string output, so a
    # "must not match" assertion would silently always pass.
    if not isinstance(forbidden, str):
        raise ValueError(
            f"forbidden must be a string, got {type(forbidden).__name__}. "
            'Quote the value in YAML (e.g. forbidden: "404").'
        )

    if not allow_empty_forbidden and forbidden == "":
        raise ValueError("forbidden must not be empty when negate=True")

    if threshold > 1.0:
        raise ValueError(f"threshold must be at most 1.0 when negate=True, got {threshold}")

    if threshold == 0.0:
        # 0.0 is ``build_metric()``'s "unspecified" sentinel, and it does not
        # forward a threshold to composite sub-metrics, so a negated sub-metric
        # lands here on a fully supported path with nothing the author could
        # change: a ``threshold`` in a metric's ``options`` is rejected outright
        # as conflicting with a factory-supplied argument. Warning on every such
        # build would be un-silenceable noise, so record it at debug level.
        logger.debug("Negated metric received no threshold; using %s.", DEFAULT_NEGATED_THRESHOLD)
        threshold = DEFAULT_NEGATED_THRESHOLD
    elif threshold < 0.0:
        # A negative value cannot be the sentinel, so it was written by hand.
        logger.warning(
            "Negated metric received threshold=%s, which would pass on every input; using %s instead.",
            threshold,
            DEFAULT_NEGATED_THRESHOLD,
        )
        threshold = DEFAULT_NEGATED_THRESHOLD

    return forbidden, threshold
