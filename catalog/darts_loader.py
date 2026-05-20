"""DartsDatasetLoader — integrate Darts built-in datasets into the catalog pipeline.

Uses the Darts library (https://github.com/unit8co/darts) to load its 27 built-in
datasets, convert to pandas DataFrames, and save as parquet to E:\\datasets\\.

All 27 Darts datasets are also registered in the catalog with their unified_ids
so they can be tracked alongside HF and Zenodo datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import config

try:
    import darts
    import darts.datasets as darts_ds
    from darts import TimeSeries
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    TimeSeries = None  # type: ignore


# ── Registry of all Darts built-in datasets ──────────────────────────────────
# (class_name, catalog_uid, horizon, seasonality, description)
DARTS_REGISTRY: list[dict[str, Any]] = [
    {"class": "AirPassengersDataset",          "uid": "TFM-009", "horizon": 12, "seasonality": 12,
     "title": "AirPassengersDataset (Darts)",  "freq": "monthly"},
    {"class": "AusBeerDataset",                "uid": "TFM-010", "horizon": 8,  "seasonality": 4,
     "title": "AusBeerDataset (Darts)",        "freq": "quarterly"},
    {"class": "AustralianTourismDataset",      "uid": None,      "horizon": 12, "seasonality": 12,
     "title": "AustralianTourismDataset (Darts)", "freq": "monthly"},
    {"class": "ETTh1Dataset",                  "uid": None,      "horizon": 24, "seasonality": 24,
     "title": "ETTh1Dataset (Darts)",          "freq": "hourly"},
    {"class": "ETTh2Dataset",                  "uid": None,      "horizon": 24, "seasonality": 24,
     "title": "ETTh2Dataset (Darts)",          "freq": "hourly"},
    {"class": "ETTm1Dataset",                  "uid": None,      "horizon": 96, "seasonality": 96,
     "title": "ETTm1Dataset (Darts)",          "freq": "15min"},
    {"class": "ETTm2Dataset",                  "uid": None,      "horizon": 96, "seasonality": 96,
     "title": "ETTm2Dataset (Darts)",          "freq": "15min"},
    {"class": "ElectricityConsumptionZurichDataset", "uid": None, "horizon": 24, "seasonality": 24,
     "title": "ElectricityConsumptionZurich (Darts)", "freq": "hourly"},
    {"class": "ElectricityDataset",            "uid": None,      "horizon": 96, "seasonality": 96,
     "title": "ElectricityDataset (Darts)",    "freq": "15min"},
    {"class": "EnergyDataset",                 "uid": None,      "horizon": 24, "seasonality": 24,
     "title": "EnergyDataset (Darts)",         "freq": "hourly"},
    {"class": "ExchangeRateDataset",           "uid": None,      "horizon": 30, "seasonality": 1,
     "title": "ExchangeRateDataset (Darts)",   "freq": "daily"},
    {"class": "GasRateCO2Dataset",             "uid": "TFM-011", "horizon": 12, "seasonality": 1,
     "title": "GasRateCO2Dataset (Darts)",     "freq": "monthly"},
    {"class": "HeartRateDataset",              "uid": "TFM-016", "horizon": 60, "seasonality": 1,
     "title": "HeartRateDataset (Darts)",      "freq": "seconds"},
    {"class": "ILINetDataset",                 "uid": None,      "horizon": 4,  "seasonality": 52,
     "title": "ILINetDataset (Darts)",         "freq": "weekly"},
    {"class": "IceCreamHeaterDataset",         "uid": None,      "horizon": 12, "seasonality": 12,
     "title": "IceCreamHeaterDataset (Darts)", "freq": "monthly"},
    {"class": "MonthlyMilkDataset",            "uid": "TFM-012", "horizon": 12, "seasonality": 12,
     "title": "MonthlyMilkDataset (Darts)",    "freq": "monthly"},
    {"class": "MonthlyMilkIncompleteDataset",  "uid": None,      "horizon": 12, "seasonality": 12,
     "title": "MonthlyMilkIncompleteDataset (Darts)", "freq": "monthly"},
    {"class": "SunspotsDataset",               "uid": "TFM-013", "horizon": 12, "seasonality": 12,
     "title": "SunspotsDataset (Darts)",       "freq": "monthly"},
    {"class": "TaxiNewYorkDataset",            "uid": None,      "horizon": 48, "seasonality": 48,
     "title": "TaxiNewYorkDataset (Darts)",    "freq": "30min"},
    {"class": "TaylorDataset",                 "uid": None,      "horizon": 48, "seasonality": 48,
     "title": "TaylorDataset (Darts)",         "freq": "30min"},
    {"class": "TemperatureDataset",            "uid": None,      "horizon": 14, "seasonality": 7,
     "title": "TemperatureDataset (Darts)",    "freq": "daily"},
    {"class": "TrafficDataset",                "uid": None,      "horizon": 24, "seasonality": 24,
     "title": "TrafficDataset (Darts)",        "freq": "hourly"},
    {"class": "USGasolineDataset",             "uid": None,      "horizon": 8,  "seasonality": 52,
     "title": "USGasolineDataset (Darts)",     "freq": "weekly"},
    {"class": "UberTLCDataset",                "uid": None,      "horizon": 24, "seasonality": 24,
     "title": "UberTLCDataset (Darts)",        "freq": "hourly"},
    {"class": "WeatherDataset",                "uid": None,      "horizon": 14, "seasonality": 7,
     "title": "WeatherDataset (Darts)",        "freq": "daily"},
    {"class": "WineDataset",                   "uid": "TFM-014", "horizon": 12, "seasonality": 12,
     "title": "WineDataset (Darts)",           "freq": "monthly"},
    {"class": "WoolyDataset",                  "uid": "TFM-015", "horizon": 8,  "seasonality": 4,
     "title": "WoolyDataset (Darts)",          "freq": "quarterly"},
]


class DartsDatasetLoader:
    """Load a Darts built-in dataset, save to E:\\datasets\\, and integrate with catalog.

    Parameters
    ----------
    class_name : str
        Darts dataset class name, e.g. 'AirPassengersDataset'.
    unified_id : str | None
        Override catalog unified_id. Auto-looked up from DARTS_REGISTRY if None.
    save_dir : Path | None
        Override save directory. Defaults to E:\\datasets\\{unified_id or class_name}.
    """

    def __init__(
        self,
        class_name: str,
        unified_id: str | None = None,
        save_dir: Path | None = None,
    ):
        if not DARTS_AVAILABLE:
            raise ImportError("pip install darts")

        self.class_name = class_name
        self._meta = next(
            (r for r in DARTS_REGISTRY if r["class"] == class_name), {}
        )
        self.unified_id = unified_id or self._meta.get("uid") or class_name
        self.save_dir = save_dir or (config.DOWNLOAD_ROOT / self.unified_id)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def save_path(self) -> Path:
        return self.save_dir / f"{self.class_name}.parquet"

    @property
    def horizon(self) -> int:
        return self._meta.get("horizon", 12)

    @property
    def seasonality(self) -> int:
        return self._meta.get("seasonality", 1)

    def load_timeseries(self) -> "TimeSeries":
        """Load as a Darts TimeSeries object (original format)."""
        cls = getattr(darts_ds, self.class_name)
        return cls().load()

    def load_df(self) -> pd.DataFrame:
        """Load as a pandas DataFrame with time index reset to a column."""
        ts = self.load_timeseries()
        df = ts.pd_dataframe()
        df.index.name = "timestamp"
        return df.reset_index()

    def download(self, force: bool = False) -> Path:
        """Load from Darts and save as parquet to E:\\datasets\\.

        The saved file is the sacred copy — never modified after this.

        Returns
        -------
        Path to saved parquet file.
        """
        if self.save_path.exists() and not force:
            return self.save_path

        print(f"  Loading {self.class_name} via Darts...")
        df = self.load_df()
        df.to_parquet(self.save_path, index=False)
        size_kb = self.save_path.stat().st_size / 1024
        print(f"  Saved: {self.save_path.name} ({size_kb:.0f} KB, shape={df.shape})")
        return self.save_path

    def to_fev_task(self, **task_kwargs):
        """Download if needed, then create an fev.Task for evaluation."""
        import fev

        self.download()

        # Load and reshape to FEV's wide-array schema via LocalDatasetLoader
        from .dataset_loader import LocalDatasetLoader
        loader = LocalDatasetLoader(self.save_path)

        cache_dir = config.DOWNLOAD_ROOT / "_cache" / self.unified_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        eval_path = cache_dir / "data.parquet"

        if not eval_path.exists():
            hf_ds = loader.to_fev_dataset()
            hf_ds.to_parquet(str(eval_path))

        params: dict[str, Any] = {
            "dataset_path": str(eval_path),
            "horizon": self.horizon,
            "seasonality": self.seasonality,
            "eval_metric": "MASE",
            "extra_metrics": ["MAE", "MSE", "RMSE"],
            "task_name": self.unified_id,
        }
        params.update(task_kwargs)
        return fev.Task(**params)

    @classmethod
    def from_catalog_uid(cls, uid: str) -> "DartsDatasetLoader":
        """Create a loader from a catalog unified_id (e.g. 'TFM-009')."""
        meta = next((r for r in DARTS_REGISTRY if r.get("uid") == uid), None)
        if meta is None:
            raise ValueError(f"No Darts dataset with uid '{uid}' in DARTS_REGISTRY")
        return cls(meta["class"], unified_id=uid)

    @classmethod
    def all_loaders(cls) -> list["DartsDatasetLoader"]:
        """Return loaders for all 27 registered Darts datasets."""
        return [cls(r["class"]) for r in DARTS_REGISTRY]


# ── Batch download helper ─────────────────────────────────────────────────────

def download_all_darts(
    db: "CatalogDB",
    report_every: int = 10,
    skip_existing: bool = True,
) -> list[dict]:
    """Download all 27 Darts built-in datasets.

    Saves to E:\\datasets\\{uid|class_name}\\{class_name}.parquet.
    Updates catalog.db download tracking for datasets with known UIDs.

    Returns list of result dicts: {class_name, uid, status, file_kb, error}
    """
    results = []

    for i, meta in enumerate(DARTS_REGISTRY, 1):
        class_name = meta["class"]
        uid = meta.get("uid") or class_name
        dataset_id = None

        if meta.get("uid"):
            row = db.get_dataset_by_uid(meta["uid"])
            if row:
                dataset_id = row["id"]
                if skip_existing and db.get_download_status(dataset_id) == "success":
                    print(f"[{i:2d}/27] SKIP {uid:12s} {class_name}")
                    results.append({"class_name": class_name, "uid": uid, "status": "skipped"})
                    continue

        loader = DartsDatasetLoader(class_name)
        try:
            path = loader.download()
            file_size = path.stat().st_size

            if dataset_id:
                db.upsert_download(
                    dataset_id, status="success",
                    local_path=str(path),
                    file_size_bytes=file_size,
                    download_url=f"darts://{class_name}",
                    url_type="darts",
                )
                db.add_to_budget(file_size)
                db.set_dataloader_created(dataset_id, 1)

            result = {"class_name": class_name, "uid": uid, "status": "success",
                      "file_kb": file_size / 1024}
            print(f"[{i:2d}/27] OK   {uid:12s} {class_name:40s} {result['file_kb']:.0f} KB")

        except Exception as e:
            err = str(e)[:120]
            if dataset_id:
                db.upsert_download(dataset_id, status="failed", error_message=err)
            result = {"class_name": class_name, "uid": uid, "status": "failed", "error": err}
            print(f"[{i:2d}/27] FAIL {uid:12s} {class_name:40s} {err[:50]}")

        results.append(result)

    ok   = sum(1 for r in results if r["status"] == "success")
    fail = sum(1 for r in results if r["status"] == "failed")
    skip = sum(1 for r in results if r["status"] == "skipped")
    print(f"\nDarts batch: {ok} ok / {fail} fail / {skip} skip")
    return results


def timeseries_to_df(ts: "TimeSeries") -> pd.DataFrame:
    """Convert a Darts TimeSeries to a long-format pandas DataFrame.

    Returns a DataFrame with columns: [timestamp, component, value]
    compatible with LocalDatasetLoader's column detection heuristics.
    """
    df = ts.pd_dataframe()
    df.index.name = "timestamp"
    return df.reset_index()


def df_to_timeseries(
    df: pd.DataFrame,
    value_cols: str | list[str],
    time_col: str = "timestamp",
    freq: str | None = None,
) -> "TimeSeries":
    """Convert a pandas DataFrame to a Darts TimeSeries.

    Parameters
    ----------
    df : pd.DataFrame
    value_cols : str | list[str]
        Column(s) to use as target values.
    time_col : str
        Column to use as time index.
    freq : str | None
        Pandas offset string (e.g. 'h', 'D', 'ME').
    """
    if not DARTS_AVAILABLE:
        raise ImportError("pip install darts")

    if time_col in df.columns:
        df = df.set_index(time_col)
        df.index = pd.to_datetime(df.index)

    if isinstance(value_cols, str):
        value_cols = [value_cols]

    return TimeSeries.from_dataframe(
        df[value_cols],
        freq=freq,
        fill_missing_dates=True,
    )
