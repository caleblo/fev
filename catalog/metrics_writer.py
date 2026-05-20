"""MetricsWriter — write FEV evaluation results to catalog.db.

Metrics are computed by FEV in this flow:
    task = fev.Task(...)
    predictions = model.fit_predict(task)
    summary = task.evaluation_summary(predictions, model_name="...")
    #        ↓ fev/metrics.py: MAE.compute, MASE.compute, MSE.compute, etc.
    writer = MetricsWriter(db)
    writer.write_summary(summary, dataset_id=42)  # → catalog.db.evaluation_metrics
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .db import CatalogDB

logger = logging.getLogger("catalog")

KNOWN_METRICS = {
    "MAE", "WAPE", "MASE", "MSE", "RMSE", "RMSSE",
    "RMSLE", "MAPE", "SMAPE", "MQL", "WQL", "SQL",
}


class MetricsWriter:
    """Write FEV evaluation results to catalog.db."""

    def __init__(self, db: CatalogDB):
        self.db = db

    def write_summary(self, summary: dict[str, Any], dataset_id: int,
                      model_name: str | None = None) -> int:
        """Write one evaluation_summary dict. Returns number of metric rows."""
        model = model_name or summary.get("model_name", "unknown")
        horizon = summary.get("horizon")
        count = 0
        import math as _math
        for key, value in summary.items():
            if key.upper() in KNOWN_METRICS and isinstance(value, (int, float)):
                if not _math.isfinite(value):
                    continue  # skip NaN/Inf (e.g. all-zero series, empty windows)
                self.db.upsert_metric(
                    dataset_id=dataset_id, model_name=model,
                    metric_name=key.upper(), metric_value=float(value),
                    prediction_length=horizon, computed_by="fev",
                    source_file=summary.get("dataset_path"),
                )
                count += 1
        self.db.record_model_run(
            dataset_id=dataset_id, model_name=model,
            training_time_s=summary.get("training_time_s"),
            inference_time_s=summary.get("inference_time_s"),
            num_forecasts=summary.get("num_forecasts"),
            fev_version=summary.get("fev_version"),
            trained_on_this=summary.get("trained_on_this_dataset", False),
        )
        return count

    def write_from_result_file(self, file_path: str, model_name: str | None = None) -> int:
        """Parse CSV/JSON/Parquet results file and write metrics."""
        p = file_path.lower()
        if p.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif p.endswith((".json", ".jsonl")):
            try:
                df = pd.read_json(file_path)
            except Exception:
                df = pd.read_json(file_path, lines=True)
        elif p.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            return 0

        col_map = {c: c.upper() for c in df.columns}
        df = df.rename(columns=col_map)
        metric_cols = [c for c in df.columns if c in KNOWN_METRICS]
        if not metric_cols:
            return 0

        name_col = next((c for c in ("DATASET", "DATASET_NAME", "TASK", "TASK_NAME", "NAME")
                         if c in df.columns), None)
        if not name_col:
            return 0
        model_col = next((c for c in ("MODEL", "MODEL_NAME", "METHOD") if c in df.columns), None)

        total = 0
        for _, row in df.iterrows():
            ds = self.db.search_dataset_by_title(str(row[name_col]))
            if not ds:
                continue
            m = model_name or (str(row[model_col]) if model_col else "unknown")
            for mc in metric_cols:
                if pd.notna(row[mc]):
                    self.db.upsert_metric(
                        dataset_id=ds["id"], model_name=m, metric_name=mc,
                        metric_value=float(row[mc]), source_file=file_path,
                        computed_by="extracted",
                    )
                    total += 1
        return total

    def read_metrics(self, model_name: str | None = None,
                     metric_name: str | None = None) -> pd.DataFrame:
        rows = self.db.get_metrics(model_name=model_name, metric_name=metric_name)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
