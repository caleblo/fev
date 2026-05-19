"""CatalogBenchmark — generate fev.Benchmark from catalog.db queries."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
import fev

from .db import CatalogDB
from .task_factory import CatalogTask

logger = logging.getLogger("catalog")


class CatalogBenchmark:
    """Generate fev.Benchmark from catalog.db queries.

    Only includes datasets that have been downloaded.

    Parameters
    ----------
    db : CatalogDB
        Database interface.
    """

    def __init__(self, db: CatalogDB):
        self.db = db
        self.task_factory = CatalogTask(db)

    def from_query(self, tier: int | None = None, limit: int | None = None,
                   source_name: str | None = None, **task_kwargs: Any) -> fev.Benchmark:
        """Build Benchmark from eligible downloaded datasets."""
        downloaded = self.db.get_downloaded_datasets()
        downloaded_ids = {d["id"] for d in downloaded}

        eligible = self.db.get_eligible_datasets(tier=tier)
        rows = [r for r in eligible if r["id"] in downloaded_ids]

        if source_name:
            sl = source_name.lower()
            rows = [r for r in rows if sl in (r.get("source_name") or "").lower()]
        if limit:
            rows = rows[:limit]

        tasks = []
        for row in rows:
            try:
                tasks.append(self.task_factory._build_task(row, **task_kwargs))
            except Exception as e:
                logger.warning(f"Skipping {row['unified_id']}: {e}")
        return fev.Benchmark(tasks=tasks)

    def from_dataset_ids(self, ids: list[int], **task_kwargs: Any) -> fev.Benchmark:
        tasks = []
        for did in ids:
            try:
                tasks.append(self.task_factory.from_dataset_id(did, **task_kwargs))
            except Exception as e:
                logger.warning(f"Skipping ID {did}: {e}")
        return fev.Benchmark(tasks=tasks)

    def forecast_benchmark(self, limit: int | None = None, **task_kwargs: Any) -> fev.Benchmark:
        return self.from_query(tier=1, limit=limit, **task_kwargs)

    def full_benchmark(self, limit: int | None = None, **task_kwargs: Any) -> fev.Benchmark:
        return self.from_query(tier=None, limit=limit, **task_kwargs)

    def to_yaml(self, path: str | Path, tier: int | None = None, limit: int | None = None) -> None:
        downloaded_ids = {d["id"] for d in self.db.get_downloaded_datasets()}
        rows = [r for r in self.db.get_eligible_datasets(tier=tier) if r["id"] in downloaded_ids]
        if limit:
            rows = rows[:limit]
        task_configs = []
        for row in rows:
            try:
                task_configs.append(self.task_factory._build_task(row).to_dict())
            except Exception as e:
                logger.warning(f"Skipping {row['unified_id']}: {e}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump({"tasks": task_configs}, f, default_flow_style=False, sort_keys=False)
