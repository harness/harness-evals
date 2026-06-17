"""Local filesystem dataset source."""

from __future__ import annotations

import asyncio
from pathlib import Path

from harness_evals.core.golden import Golden
from harness_evals.datasets.base import BaseDatasetSource
from harness_evals.datasets.io import load_dataset
from harness_evals.plugins import register_dataset_source
from harness_evals.refs import ResourceRef


@register_dataset_source("local")
class LocalDatasetSource(BaseDatasetSource):
    """Fetch a dataset from a local JSON or JSONL file."""

    name = "local"

    async def fetch(self, ref: ResourceRef) -> list[Golden]:
        return await asyncio.to_thread(load_dataset, ref.id, format=_dataset_format(ref))


def _dataset_format(ref: ResourceRef) -> str:
    explicit_format = ref.extra.get("format")
    if explicit_format is not None:
        return str(explicit_format)
    if Path(ref.id).suffix.lower() == ".json":
        return "json"
    return "jsonl"
