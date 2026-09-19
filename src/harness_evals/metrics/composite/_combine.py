"""Shared weighted-sum fold for composite metrics.

Not a public import surface — internal to ``metrics/composite/``.
"""

from __future__ import annotations

from typing import Any


def fold_sub_scores(
    sub_scores: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Combine per-sub-check results into a weighted-average score.

    ``results[name]`` must already carry ``"status"`` (``"ok"``, ``"skipped"``,
    or ``"error"``) and ``"value"`` for ``"ok"`` entries. Sub-checks with status
    ``"skipped"`` are excluded from the active weight entirely; ``"error"``
    entries count their weight but contribute ``0.0``.
    """
    details: dict[str, Any] = {"sub_scores": results, "effective_weights": {}}
    total_score = 0.0
    active_weight_sum = 0.0

    for sub in sub_scores:
        sub_name = sub.get("name", "unknown")
        weight = float(sub.get("weight", 0.0))
        result = results.get(sub_name, {})
        if result.get("status") == "skipped":
            continue

        active_weight_sum += weight
        if result.get("status") == "ok":
            total_score += float(result.get("value") or 0.0) * weight

    final_score = total_score / active_weight_sum if active_weight_sum > 0 else 0.0

    if active_weight_sum > 0:
        for sub in sub_scores:
            sub_name = sub.get("name", "unknown")
            if results.get(sub_name, {}).get("status") != "skipped":
                weight = float(sub.get("weight", 0.0))
                details["effective_weights"][sub_name] = round(weight / active_weight_sum, 4)

    return final_score, details
