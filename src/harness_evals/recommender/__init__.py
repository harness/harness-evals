"""Eval Recommender — suggest metrics, goldens, and EvalConfig for any agent input."""

from harness_evals.recommender.engine import RECOMMENDATION_SCHEMA, recommend

__all__ = ["recommend", "RECOMMENDATION_SCHEMA"]
