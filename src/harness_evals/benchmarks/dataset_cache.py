"""Dataset caching for HuggingFace Hub datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "harness_evals" / "benchmarks"


def get_cache_path(benchmark: str, split: str, *, cache_dir: Path | None = None) -> Path:
    """Return the local cache file path for a benchmark split."""
    base = cache_dir or DEFAULT_CACHE_DIR
    return base / benchmark / f"{split}.jsonl"


def load_cached(benchmark: str, split: str, *, cache_dir: Path | None = None) -> list[dict] | None:
    """Load cached dataset if available, otherwise return None."""
    path = get_cache_path(benchmark, split, cache_dir=cache_dir)
    if not path.exists():
        return None
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_to_cache(items: list[dict], benchmark: str, split: str, *, cache_dir: Path | None = None) -> Path:
    """Save dataset items to local cache."""
    path = get_cache_path(benchmark, split, cache_dir=cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(path)
    logger.info("Cached %d items to %s", len(items), path)
    return path


async def fetch_hf_dataset(
    repo_id: str,
    split: str = "test",
    *,
    config: str | None = None,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> list[dict]:
    """Fetch a dataset from HuggingFace Hub, with caching.

    Uses the HF datasets API (rows endpoint) to avoid pulling in the
    full `datasets` library or PyArrow.

    Args:
        repo_id: HuggingFace dataset ID (e.g., "openai/gsm8k").
        split: Dataset split to load.
        config: Dataset config/subset name.
        cache_dir: Override cache directory.
        offline: If True, only use cache (raise if not cached).
    """
    cache_key = f"{repo_id.replace('/', '__')}"
    if config:
        cache_key += f"__{config}"

    cached = load_cached(cache_key, split, cache_dir=cache_dir)
    if cached is not None:
        return cached

    if offline:
        raise FileNotFoundError(
            f"Dataset '{repo_id}' (split={split}, config={config}) not found in cache. "
            f"Run once without offline=True to download."
        )

    items = await _fetch_all_rows(repo_id, split, config)
    save_to_cache(items, cache_key, split, cache_dir=cache_dir)
    return items


async def fetch_github_json(
    url: str,
    cache_key: str,
    *,
    split: str = "default",
    cache_dir: Path | None = None,
    offline: bool = False,
) -> list[dict]:
    """Fetch a JSON or JSONL resource from a URL (e.g. GitHub raw), with caching."""
    cached = load_cached(cache_key, split, cache_dir=cache_dir)
    if cached is not None:
        return cached

    if offline:
        raise FileNotFoundError(
            f"Dataset cache key '{cache_key}' (split={split}) not found in cache. "
            f"Run once without offline=True to download."
        )

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        text = resp.text.strip()

    items: list[dict]
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array from {url}, got {type(parsed).__name__}")
        items = [row for row in parsed if isinstance(row, dict)]
    else:
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    items.append(row)

    save_to_cache(items, cache_key, split, cache_dir=cache_dir)
    return items


async def _fetch_all_rows(repo_id: str, split: str, config: str | None) -> list[dict[str, Any]]:
    """Paginate through HuggingFace datasets API to fetch all rows."""
    base_url = "https://datasets-server.huggingface.co/rows"
    items: list[dict[str, Any]] = []
    offset = 0
    page_size = 100

    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            params: dict[str, Any] = {
                "dataset": repo_id,
                "split": split,
                "offset": offset,
                "length": page_size,
            }
            if config:
                params["config"] = config

            resp = await client.get(base_url, params=params)
            resp.raise_for_status()
            data = resp.json()

            rows = data.get("rows", [])
            if not rows:
                break

            for row in rows:
                items.append(row["row"])

            num_rows_total = data.get("num_rows_total", 0)
            offset += len(rows)
            if offset >= num_rows_total:
                break

    logger.info("Fetched %d rows from %s/%s", len(items), repo_id, split)
    return items
