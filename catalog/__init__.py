"""Catalog bridge — connects catalog.db to the FEV forecasting framework.

Architecture
------------
- **catalog.db** (E:\\z_dataset lists\\catalog.db) — single source of truth.
  Metadata + download tracking + metrics + dataloader flags.
- **E:\\datasets\\** — downloaded files, NEVER altered after download.
  Dataloaders handle all transformation in memory.

Metrics Flow
------------
Metrics are computed by FEV (fev/metrics.py → MAE, MASE, MSE, etc.):

    task = fev.Task(...)                             # from CatalogTask
    predictions = model.fit_predict(task)             # ForecastingModel subclass
    summary = task.evaluation_summary(predictions)    # EvaluationWindow.compute_metrics
    writer = MetricsWriter(db)
    writer.write_summary(summary, dataset_id=42)      # → catalog.db.evaluation_metrics
"""
from .benchmark_factory import CatalogBenchmark
from .darts_loader import DartsDatasetLoader, download_all_darts, df_to_timeseries, timeseries_to_df
from .dataset_loader import LocalDatasetLoader
from .db import CatalogDB
from .download import DatasetDownloader
from .hf_loader import HFDatasetLoader, download_explicit_targets
from .metrics_writer import MetricsWriter
from .task_factory import CatalogTask

__all__ = [
    "CatalogDB",
    "CatalogTask",
    "CatalogBenchmark",
    "MetricsWriter",
    "LocalDatasetLoader",
    "DatasetDownloader",
    "HFDatasetLoader",
    "download_explicit_targets",
    "DartsDatasetLoader",
    "download_all_darts",
    "df_to_timeseries",
    "timeseries_to_df",
]
