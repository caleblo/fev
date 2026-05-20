"""HFDatasetLoader — load datasets via the HuggingFace pipeline and save to E:\\datasets\\."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

from . import config

try:
    import datasets as hf_datasets
except ImportError as e:  # noqa: F841
    hf_datasets = None  # type: ignore[assignment]


# ── HF repo → (config_name_key, target_col, horizon, seasonality) defaults ───
# For datasets where the target column is not 'target'
HF_TARGET_OVERRIDES: dict[str, dict[str, Any]] = {
    # ── autogluon/chronos_datasets ────────────────────────────────────────────
    "ushcn_daily":                  {"target": "TMAX",          "horizon": 14,  "seasonality": 7},
    "weatherbench_daily":           {"target": "target",        "horizon": 14,  "seasonality": 7,  "max_series": 10000},
    "wiki_daily_100k":              {"target": "target",        "horizon": 7,   "seasonality": 7,  "max_series": 50000},

    # ── autogluon/fev_datasets — non-standard target columns ─────────────────
    # ETT datasets: use 'OT' (oil temperature) as primary forecast target
    "ETT_15T":                      {"target": "OT",            "horizon": 96,  "seasonality": 96},
    "ETT_1H":                       {"target": "OT",            "horizon": 24,  "seasonality": 24},
    "ETT_1D":                       {"target": "OT",            "horizon": 14,  "seasonality": 7},
    "ETT_1W":                       {"target": "OT",            "horizon": 8,   "seasonality": 1},

    # bizitobs: multivariate (target_0 .. target_6) — use target_0
    "bizitobs_l2c_1H":              {"target": "target_0",      "horizon": 24,  "seasonality": 24},
    "bizitobs_l2c_5T":              {"target": "target_0",      "horizon": 144, "seasonality": 144},

    # boomlet: observability multivariate (target_0 .. target_N) — use target_0
    "boomlet_285":                  {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_619":                  {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_772":                  {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_963":                  {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_1062":                 {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_1209":                 {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_1225":                 {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_1230":                 {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_1282":                 {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_1487":                 {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_1631":                 {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_1676":                 {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_1855":                 {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_1975":                 {"target": "target_0",      "horizon": 12,  "seasonality": 1},
    "boomlet_2187":                 {"target": "target_0",      "horizon": 12,  "seasonality": 1},

    # favorita: sales forecasting
    "favorita_stores_1D":           {"target": "sales",         "horizon": 14,  "seasonality": 7},
    "favorita_transactions_1D":     {"target": "transactions",  "horizon": 14,  "seasonality": 7},
    "favorita_stores_1W":           {"target": "sales",         "horizon": 8,   "seasonality": 1},
    "favorita_stores_1M":           {"target": "sales",         "horizon": 12,  "seasonality": 12},

    # FRED: macro datasets — very wide multivariate, use INDPRO (industrial production)
    "fred_md_2025":                 {"target": "INDPRO",        "horizon": 12,  "seasonality": 12},
    "fred_qd_2025":                 {"target": "GDPC1",         "horizon": 8,   "seasonality": 4},

    # GVAR: use 'y' (output) as target
    "gvar":                         {"target": "y",             "horizon": 8,   "seasonality": 4},

    # KDD Cup 2022: wind power — use 'Patv' (active power)
    "kdd_cup_2022_10T":             {"target": "Patv",          "horizon": 144, "seasonality": 144},
    "kdd_cup_2022_30T":             {"target": "Patv",          "horizon": 48,  "seasonality": 48},
    "kdd_cup_2022_1D":              {"target": "Patv",          "horizon": 14,  "seasonality": 7},

    # Rohlik: orders and sales
    "rohlik_orders_1D":             {"target": "orders",        "horizon": 14,  "seasonality": 7},
    "rohlik_orders_1W":             {"target": "orders",        "horizon": 8,   "seasonality": 1},
    "rohlik_sales_1D":              {"target": "sales",         "horizon": 14,  "seasonality": 7},
    "rohlik_sales_1W":              {"target": "sales",         "horizon": 8,   "seasonality": 1},

    # Rossmann: store sales
    "rossmann_1D":                  {"target": "Sales",         "horizon": 14,  "seasonality": 7},
    "rossmann_1W":                  {"target": "Sales",         "horizon": 8,   "seasonality": 1},

    # UCI Air Quality: CO concentration
    "uci_air_quality_1D":           {"target": "CO(GT)",        "horizon": 14,  "seasonality": 7},
    "uci_air_quality_1H":           {"target": "CO(GT)",        "horizon": 24,  "seasonality": 24},

    # UK COVID: new cases
    "uk_covid_nation_1D":           {"target": "new_cases",     "horizon": 14,  "seasonality": 7},
    "uk_covid_nation_1W":           {"target": "new_cases",     "horizon": 8,   "seasonality": 1},
    "uk_covid_utla_1D":             {"target": "new_cases",     "horizon": 14,  "seasonality": 7},
    "uk_covid_utla_1W":             {"target": "new_cases",     "horizon": 8,   "seasonality": 1},
}

# fev_datasets frequency heuristics (suffix → horizon, seasonality)
FEV_FREQ_DEFAULTS: dict[str, tuple[int, int]] = {
    "10T": (144, 144),   # 10-minute
    "15T": (96,  96),    # 15-minute
    "30T": (48,  48),    # 30-minute
    "1H":  (24,  24),    # hourly
    "1D":  (14,  7),     # daily
    "1W":  (8,   1),     # weekly
    "1M":  (12,  12),    # monthly
    "1Q":  (8,   4),     # quarterly
    "1Y":  (1,   1),     # yearly
}


def _infer_horizon_season(config_name: str) -> tuple[int, int]:
    """Infer horizon and seasonality from config name suffix."""
    name_upper = config_name.upper()
    for suffix, (h, s) in FEV_FREQ_DEFAULTS.items():
        if name_upper.endswith(suffix):
            return h, s
    return 12, 1


class HFDatasetLoader:
    """Download and load a HuggingFace dataset to E:\\datasets\\.

    Parameters
    ----------
    hf_repo : str
        HuggingFace dataset repository, e.g. 'autogluon/fev_datasets'
    config_name : str
        Dataset config/subset name, e.g. 'ETT_15T'
    save_dir : Path | None
        Override save location. Defaults to E:\\datasets\\{unified_id}.
    unified_id : str | None
        Dataset unified_id from catalog.db (used for directory naming).
    """

    HF_CACHE = config.DOWNLOAD_ROOT / "_hf_cache"

    def __init__(
        self,
        hf_repo: str,
        config_name: str,
        save_dir: Path | None = None,
        unified_id: str | None = None,
    ):
        if hf_datasets is None:
            raise ImportError("pip install datasets")

        self.hf_repo = hf_repo
        self.config_name = config_name
        self.unified_id = unified_id or config_name
        self.save_dir = save_dir or (config.DOWNLOAD_ROOT / self.unified_id)

        # Set HF cache to E:\datasets\_hf_cache to avoid filling C:
        os.environ.setdefault("HF_HOME", str(self.HF_CACHE))
        self.HF_CACHE.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def save_path(self) -> Path:
        return self.save_dir / f"{self.config_name}.parquet"

    @property
    def target_overrides(self) -> dict[str, Any]:
        return HF_TARGET_OVERRIDES.get(self.config_name, {})

    @property
    def max_series(self) -> int | None:
        return self.target_overrides.get("max_series")

    def download(self, force: bool = False) -> Path:
        """Download from HuggingFace and save as parquet.

        The saved file is the sacred copy — never modified after this.

        Parameters
        ----------
        force : bool
            Re-download even if file already exists.

        Returns
        -------
        Path to saved parquet file.
        """
        if self.save_path.exists() and not force:
            return self.save_path

        t0 = time.time()
        print(f"  Downloading {self.hf_repo}/{self.config_name}...")

        ds = hf_datasets.load_dataset(
            self.hf_repo,
            self.config_name,
            split="train",
            num_proc=1,
        )
        n_series = len(ds)
        print(f"  {n_series:,} series ({time.time() - t0:.1f}s)")

        # Save full dataset (sacred copy)
        ds.to_parquet(str(self.save_path))
        print(f"  Saved: {self.save_path.name} ({self.save_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return self.save_path

    def load_hf(self, for_eval: bool = False) -> "hf_datasets.Dataset":
        """Load the saved parquet back as an HF Dataset.

        Parameters
        ----------
        for_eval : bool
            If True and max_series is set, return truncated version for evaluation.
        """
        if hf_datasets is None:
            raise ImportError("pip install datasets")

        if not self.save_path.exists():
            raise FileNotFoundError(
                f"Dataset not downloaded yet: {self.save_path}\n"
                f"Call .download() first, or use .download_and_load()."
            )

        ds = hf_datasets.Dataset.from_parquet(str(self.save_path))
        if for_eval and self.max_series and len(ds) > self.max_series:
            ds = ds.select(range(self.max_series))
        return ds

    def load_df(self) -> "pd.DataFrame":
        """Load the saved parquet as a pandas DataFrame."""
        import pandas as pd
        return pd.read_parquet(self.save_path)

    def download_and_load(self) -> "hf_datasets.Dataset":
        """Download if needed, then load as HF Dataset."""
        self.download()
        return self.load_hf()

    def to_fev_task(self, **task_kwargs):
        """Create an fev.Task from this HF dataset.

        Downloads if needed, saves eval-ready parquet to _cache/,
        returns fev.Task pointing at it.
        """
        import fev
        from .config import DOWNLOAD_ROOT

        # Download sacred copy
        self.download()

        # Eval copy (truncated if needed)
        cache_dir = DOWNLOAD_ROOT / "_cache" / self.unified_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        eval_path = cache_dir / "data.parquet"

        if not eval_path.exists():
            ds = self.load_hf(for_eval=True)
            ds.to_parquet(str(eval_path))

        # Defaults from frequency suffix
        horizon, seasonality = _infer_horizon_season(self.config_name)
        overrides = self.target_overrides

        params: dict[str, Any] = {
            "dataset_path": str(eval_path),
            "horizon": overrides.get("horizon", horizon),
            "seasonality": overrides.get("seasonality", seasonality),
            "eval_metric": "MASE",
            "extra_metrics": ["MAE", "MSE", "RMSE"],
            "task_name": self.unified_id,
        }
        if "target" in overrides:
            params["target"] = overrides["target"]

        params.update(task_kwargs)
        return fev.Task(**params)

    @classmethod
    def from_catalog(cls, db: "CatalogDB", unified_id: str) -> "HFDatasetLoader":
        """Create an HFDatasetLoader from a catalog.db unified_id.

        Looks up the homepage URL and parses hf_repo + config_name from it.
        Raises ValueError if the dataset is not a HuggingFace dataset.
        """
        row = db.get_dataset_by_uid(unified_id)
        if row is None:
            raise ValueError(f"'{unified_id}' not found in catalog.db")

        url = row.get("homepage", "") or ""
        repo, config_name = _parse_hf_url(url)
        if not repo:
            raise ValueError(
                f"'{unified_id}' homepage is not a HuggingFace URL: {url}"
            )

        return cls(repo, config_name, unified_id=unified_id)


# ── Batch helpers ─────────────────────────────────────────────────────────────

# Full list of explicitly requested HF configs
EXPLICIT_HF_TARGETS: list[tuple[str, str, str]] = [
    # (hf_repo, config_name, unified_id)
    ("autogluon/chronos_datasets", "ushcn_daily",                "CHR-012"),
    ("autogluon/chronos_datasets", "weatherbench_daily",         "CHR-013"),
    ("autogluon/chronos_datasets", "wiki_daily_100k",            "CHR-016"),
    ("autogluon/fev_datasets",     "ETT_15T",                    "FEV-001"),
    ("autogluon/fev_datasets",     "LOOP_SEATTLE_1D",            "FEV-005"),
    ("autogluon/fev_datasets",     "M_DENSE_1D",                 "FEV-008"),
    ("autogluon/fev_datasets",     "SZ_TAXI_15T",                "FEV-010"),
    ("autogluon/fev_datasets",     "australian_tourism",         "FEV-012"),
    ("autogluon/fev_datasets",     "bizitobs_l2c_1H",            "FEV-013"),
    ("autogluon/fev_datasets",     "boomlet_1062",               "FEV-015"),
    ("autogluon/fev_datasets",     "boomlet_1209",               "FEV-016"),
    ("autogluon/fev_datasets",     "boomlet_1225",               "FEV-017"),
    ("autogluon/fev_datasets",     "boomlet_1230",               "FEV-018"),
    ("autogluon/fev_datasets",     "boomlet_1282",               "FEV-019"),
    ("autogluon/fev_datasets",     "boomlet_1487",               "FEV-020"),
    ("autogluon/fev_datasets",     "boomlet_1631",               "FEV-021"),
    ("autogluon/fev_datasets",     "boomlet_1676",               "FEV-022"),
    ("autogluon/fev_datasets",     "boomlet_1855",               "FEV-023"),
    ("autogluon/fev_datasets",     "boomlet_1975",               "FEV-024"),
    ("autogluon/fev_datasets",     "boomlet_2187",               "FEV-025"),
    ("autogluon/fev_datasets",     "boomlet_285",                "FEV-026"),
    ("autogluon/fev_datasets",     "boomlet_619",                "FEV-027"),
    ("autogluon/fev_datasets",     "boomlet_772",                "FEV-028"),
    ("autogluon/fev_datasets",     "boomlet_963",                "FEV-029"),
    ("autogluon/fev_datasets",     "ecdc_ili",                   "FEV-030"),
    ("autogluon/fev_datasets",     "entsoe_15T",                 "FEV-031"),
    ("autogluon/fev_datasets",     "epf_be",                     "FEV-034"),
    ("autogluon/fev_datasets",     "epf_de",                     "FEV-035"),
    ("autogluon/fev_datasets",     "epf_fr",                     "FEV-036"),
    ("autogluon/fev_datasets",     "epf_np",                     "FEV-037"),
    ("autogluon/fev_datasets",     "epf_pjm",                    "FEV-038"),
    ("autogluon/fev_datasets",     "ercot_1D",                   "FEV-039"),
    ("autogluon/fev_datasets",     "favorita_stores_1D",         "FEV-043"),
    ("autogluon/fev_datasets",     "favorita_transactions_1D",   "FEV-046"),
    ("autogluon/fev_datasets",     "fred_md_2025",               "FEV-049"),
    ("autogluon/fev_datasets",     "fred_qd_2025",               "FEV-051"),
    ("autogluon/fev_datasets",     "gvar",                       "FEV-053"),
    ("autogluon/fev_datasets",     "hermes",                     "FEV-054"),
    ("autogluon/fev_datasets",     "hierarchical_sales_1D",      "FEV-055"),
    ("autogluon/fev_datasets",     "hospital_admissions_1D",     "FEV-058"),
    ("autogluon/fev_datasets",     "kdd_cup_2022_10T",           "FEV-063"),
    ("autogluon/fev_datasets",     "m5_1D",                      "FEV-066"),
    ("autogluon/fev_datasets",     "proenfo_gfc12",              "FEV-069"),
    ("autogluon/fev_datasets",     "proenfo_gfc14",              "FEV-070"),
    ("autogluon/fev_datasets",     "proenfo_gfc17",              "FEV-071"),
    ("autogluon/fev_datasets",     "redset_15T",                 "FEV-072"),
    ("autogluon/fev_datasets",     "restaurant",                 "FEV-075"),
    ("autogluon/fev_datasets",     "rohlik_orders_1D",           "FEV-076"),
    ("autogluon/fev_datasets",     "rohlik_sales_1D",            "FEV-078"),
    ("autogluon/fev_datasets",     "rossmann_1D",                "FEV-080"),
    ("autogluon/fev_datasets",     "solar_with_weather_15T",     "FEV-084"),
    ("autogluon/fev_datasets",     "uci_air_quality_1D",         "FEV-086"),
    ("autogluon/fev_datasets",     "uk_covid_nation_1D",         "FEV-088"),
    ("autogluon/fev_datasets",     "uk_covid_utla_1D",           "FEV-092"),
    ("autogluon/fev_datasets",     "us_consumption_1M",          "FEV-094"),
    ("autogluon/fev_datasets",     "walmart",                    "FEV-097"),
    ("autogluon/fev_datasets",     "world_co2_emissions",        "FEV-098"),
    ("autogluon/fev_datasets",     "world_life_expectancy",      "FEV-099"),
    ("autogluon/fev_datasets",     "world_tourism",              "FEV-100"),
]


def download_explicit_targets(
    db: "CatalogDB",
    report_every: int = 10,
    skip_existing: bool = True,
) -> list[dict]:
    """Download all 59 explicitly requested HF datasets.

    Updates catalog.db download tracking for each dataset.
    Reports progress every `report_every` datasets.

    Parameters
    ----------
    db : CatalogDB
    report_every : int
    skip_existing : bool
        Skip datasets already downloaded (status='success' in catalog.db).

    Returns
    -------
    List of result dicts: {uid, config, status, file_mb, error}
    """
    results = []
    batch = []
    total = len(EXPLICIT_HF_TARGETS)

    for i, (repo, config_name, uid) in enumerate(EXPLICIT_HF_TARGETS, 1):
        row = db.get_dataset_by_uid(uid)
        dataset_id = row["id"] if row else None

        if skip_existing and db.get_download_status(dataset_id) == "success":
            print(f"[{i:2d}/{total}] SKIP (already downloaded): {uid} {config_name}")
            results.append({"uid": uid, "config": config_name, "status": "skipped"})
            batch.append(results[-1])
        else:
            loader = HFDatasetLoader(repo, config_name, unified_id=uid)
            try:
                path = loader.download()
                file_size = path.stat().st_size

                if dataset_id:
                    db.upsert_download(
                        dataset_id, status="success",
                        local_path=str(path),
                        file_size_bytes=file_size,
                        download_url=f"hf://{repo}/{config_name}",
                        url_type="huggingface",
                    )
                    db.add_to_budget(file_size)
                    db.set_dataloader_created(dataset_id, 1)

                result = {"uid": uid, "config": config_name, "status": "success",
                          "file_mb": file_size / 1024 / 1024, "path": str(path)}
                print(f"[{i:2d}/{total}] OK  {uid:10s} {config_name:30s} "
                      f"{result['file_mb']:7.1f} MB")

            except Exception as e:
                err = str(e)[:120]
                if dataset_id:
                    db.upsert_download(dataset_id, status="failed", error_message=err)
                result = {"uid": uid, "config": config_name, "status": "failed", "error": err}
                print(f"[{i:2d}/{total}] FAIL {uid:10s} {config_name:30s} {err[:60]}")

            results.append(result)
            batch.append(result)

        # Progress report every N
        if len(batch) >= report_every or i == total:
            _print_batch_report(batch, i, db)
            batch = []

    return results


def _print_batch_report(batch: list[dict], i: int, db: "CatalogDB") -> None:
    used, budget = db.get_budget()
    downloaded = db.get_downloaded_datasets()
    ok = sum(1 for r in batch if r["status"] == "success")
    fail = sum(1 for r in batch if r["status"] == "failed")
    skip = sum(1 for r in batch if r["status"] == "skipped")
    print(f"\n{'─'*70}")
    print(f"Progress: {i}/{len(EXPLICIT_HF_TARGETS)} | "
          f"Batch: {ok} ok / {fail} fail / {skip} skip | "
          f"Total downloaded: {len(downloaded)} | "
          f"Budget: {used/1024/1024:.0f} MB")
    print(f"{'─'*70}\n")


def _parse_hf_url(url: str) -> tuple[str | None, str | None]:
    """Parse a HuggingFace URL into (repo, config_name).

    Examples:
      https://huggingface.co/datasets/autogluon/fev_datasets/viewer/ETT_15T
        → ('autogluon/fev_datasets', 'ETT_15T')
      https://huggingface.co/datasets/autogluon/chronos_datasets/tree/main/ushcn_daily
        → ('autogluon/chronos_datasets', 'ushcn_daily')
    """
    if "huggingface.co/datasets/" not in url:
        return None, None

    path = url.split("huggingface.co/datasets/")[1].rstrip("/")
    parts = path.split("/")

    if len(parts) < 2:
        return None, None

    repo = "/".join(parts[:2])  # e.g. autogluon/fev_datasets

    # viewer/CONFIG or tree/main/CONFIG
    if len(parts) >= 4 and parts[2] in ("viewer", "tree"):
        config_name = parts[-1]
    elif len(parts) == 2:
        config_name = None
    else:
        config_name = parts[-1]

    return repo, config_name
