"""CatalogTask — factory to convert catalog.db rows into fev.Task objects."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import fev

from . import config
from .db import CatalogDB
from .dataset_loader import LocalDatasetLoader


class CatalogTask:
    """Factory: catalog.db dataset row → fev.Task.

    Downloads are at E:\\datasets\\ and NEVER altered. The loader reads
    them in-memory and converts to FEV's expected HF Dataset schema.

    Parameters
    ----------
    db : CatalogDB
        Database interface.
    cache_dir : Path | None
        Directory for cached parquet conversions.
    """

    def __init__(self, db: CatalogDB, cache_dir: Path | None = None):
        self.db = db
        self.cache_dir = cache_dir or (config.DOWNLOAD_ROOT / "_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def from_dataset_id(self, dataset_id: int, **task_kwargs: Any) -> fev.Task:
        """Create an fev.Task from a catalog dataset ID."""
        row = self.db.get_dataset_by_id(dataset_id)
        if row is None:
            raise ValueError(f"Dataset ID {dataset_id} not found in catalog.db")
        return self._build_task(row, **task_kwargs)

    def from_unified_id(self, unified_id: str, **task_kwargs: Any) -> fev.Task:
        """Create an fev.Task from a unified_id (e.g. 'CHR-001')."""
        row = self.db.get_dataset_by_uid(unified_id)
        if row is None:
            raise ValueError(f"Unified ID '{unified_id}' not found in catalog.db")
        return self._build_task(row, **task_kwargs)

    def _build_task(self, row: dict, **task_kwargs: Any) -> fev.Task:
        """Core builder: catalog row → parquet cache → fev.Task.

        The original downloaded file is never touched. We load it via
        LocalDatasetLoader, convert to FEV schema, and cache as parquet.
        """
        local_path = row.get("dataset_storage_local")
        if not local_path:
            raise ValueError(
                f"Dataset '{row['unified_id']}' has no local file. "
                "Download it first via DatasetDownloader."
            )

        # Load original file (read-only) and convert to FEV schema
        loader = LocalDatasetLoader(local_path)
        hf_ds = loader.to_fev_dataset()

        # Cache as parquet for FEV (original file untouched)
        uid = row["unified_id"]
        parquet_dir = self.cache_dir / uid
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = parquet_dir / "data.parquet"
        hf_ds.to_parquet(str(parquet_path))

        # Build task params
        params = self._infer_task_params(row)
        params["dataset_path"] = str(parquet_path)
        params["task_name"] = uid
        params.update(task_kwargs)

        col_info = self._resolve_columns(loader.load_df())
        for key in ("id_column", "timestamp_column", "target"):
            if key not in params and key in col_info:
                params[key] = col_info[key]

        return fev.Task(**params)

    def _infer_task_params(self, row: dict) -> dict:
        freq = self.db.get_frequency(row["id"])
        if freq:
            freq_lower = freq.lower().replace("-", "_").replace(" ", "_")
            for key, defaults in config.FREQ_DEFAULTS.items():
                if key in freq_lower:
                    return {
                        "horizon": defaults["horizon"],
                        "seasonality": defaults["seasonality"],
                        "eval_metric": "MASE",
                    }
        return {"horizon": 12, "seasonality": 1, "eval_metric": "MASE"}

    @staticmethod
    def _resolve_columns(df) -> dict:
        import numpy as np
        import pandas as pd
        result: dict = {}
        for c in ("id", "series_id", "item_id", "unique_id", "ts_id"):
            if c in df.columns:
                result["id_column"] = c
                break
        for c in ("timestamp", "date", "time", "datetime", "ds"):
            if c in df.columns:
                result["timestamp_column"] = c
                break
        else:
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    result["timestamp_column"] = c
                    break
        exclude = set(result.values())
        targets = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        if len(targets) == 1:
            result["target"] = targets[0]
        elif targets:
            result["target"] = targets
        return result
