"""TempusBenchLoader — download and load TempusBench datasets from GitHub.

TempusBench (https://github.com/Smlcrm/TempusBench) provides 73 forecasting
tasks across 3 types:
  - univariate (25 tasks): single target variable
  - multivariate (22 tasks): multiple target variables
  - covariate (26 tasks): targets + known external covariates

CSV schema (4 columns):
  variable_name | timestamps            | values                | variable_type
  target_1      | ["2020-01-01Z", ...] | [1.0, 2.5, null, ...] | target
  covariate_1   | ["2020-01-01Z", ...] | [0.5, 0.6, 0.7, ...]  | covariate

Files are downloaded once from GitHub (raw) and stored at
E:\\datasets\\{unified_id}\\{task_dir_name}.csv (sacred copy, never altered).
"""
from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from . import config

# ── GitHub raw base URL ───────────────────────────────────────────────────────
_GH_RAW = (
    "https://raw.githubusercontent.com/Smlcrm/TempusBench/"
    "prod/tempus_bench/tasks"
)
_GH_API = (
    "https://api.github.com/repos/Smlcrm/TempusBench/contents/"
    "tempus_bench/tasks/{task_type}/{task_dir}"
)

# ── Complete task registry: (task_dir, task_type, unified_id) ────────────────
# Maps all 73 TempusBench task directories to catalog.db unified_ids (TBN-*)
TEMPUSBENCH_REGISTRY: list[dict[str, str]] = [
    # ── Univariate (25) ───────────────────────────────────────────────────
    {"task_dir": "absent_binary_univariate",              "task_type": "univariate",   "uid": "TBN-065"},
    {"task_dir": "chickenpox_dense_univariate",           "task_type": "univariate",   "uid": "TBN-062"},
    {"task_dir": "coinbase_days_univariate",              "task_type": "univariate",   "uid": "TBN-008"},
    {"task_dir": "coinbase_economics_univariate",         "task_type": "univariate",   "uid": "TBN-032"},
    {"task_dir": "delhi_climate_univariate",              "task_type": "univariate",   "uid": "TBN-028"},
    {"task_dir": "electricity_energy_univariate",         "task_type": "univariate",   "uid": "TBN-001"},
    {"task_dir": "employees_healthcare_univariate",       "task_type": "univariate",   "uid": "TBN-039"},
    {"task_dir": "federal_funds_weeks_univariate",        "task_type": "univariate",   "uid": "TBN-021"},
    {"task_dir": "german_houses_sales_univariate",        "task_type": "univariate",   "uid": "TBN-050"},
    {"task_dir": "german_quarterly_univariate",           "task_type": "univariate",   "uid": "TBN-019"},
    {"task_dir": "inventories_manufacturing_univariate",  "task_type": "univariate",   "uid": "TBN-041"},
    {"task_dir": "inventories_months_univariate",         "task_type": "univariate",   "uid": "TBN-017"},
    {"task_dir": "madrid_transport_univariate",           "task_type": "univariate",   "uid": "TBN-058"},
    {"task_dir": "occupancy_count_univariate",            "task_type": "univariate",   "uid": "TBN-071"},
    {"task_dir": "patient_sparse_univariate",             "task_type": "univariate",   "uid": "TBN-064"},
    {"task_dir": "power_consumption_years_univariate",    "task_type": "univariate",   "uid": "TBN-022"},
    {"task_dir": "retail_categorical_univariate",         "task_type": "univariate",   "uid": "TBN-066"},
    {"task_dir": "software_nonstationary_univariate",     "task_type": "univariate",   "uid": "TBN-002"},
    {"task_dir": "soil_nature_univariate",                "task_type": "univariate",   "uid": "TBN-045"},
    {"task_dir": "sw_job_postings_software_univariate",   "task_type": "univariate",   "uid": "TBN-053"},
    {"task_dir": "synthetic_additive2_univariate",        "task_type": "univariate",   "uid": "TBN-005"},
    {"task_dir": "synthetic_cyclic_univariate",           "task_type": "univariate",   "uid": "TBN-025"},
    {"task_dir": "synthetic_multiplicative_univariate",   "task_type": "univariate",   "uid": "TBN-006"},
    {"task_dir": "synthetic_nonstationary_univariate",    "task_type": "univariate",   "uid": "TBN-026"},
    {"task_dir": "web_traffic_univariate",                "task_type": "univariate",   "uid": "TBN-060"},
    # ── Multivariate (22) ─────────────────────────────────────────────────
    {"task_dir": "baggage_100_multivariate",              "task_type": "multivariate", "uid": "TBN-055"},
    {"task_dir": "baggage_months_multivariate",           "task_type": "multivariate", "uid": "TBN-016"},
    {"task_dir": "baggage_sales_multivariate",            "task_type": "multivariate", "uid": "TBN-049"},
    {"task_dir": "batadal_software_multivariate",         "task_type": "multivariate", "uid": "TBN-052"},
    {"task_dir": "gold_india_continuous_multivariate",    "task_type": "multivariate", "uid": "TBN-067"},
    {"task_dir": "gold_india_dense_multivariate",         "task_type": "multivariate", "uid": "TBN-061"},
    {"task_dir": "gold_india_economics_multivariate",     "task_type": "multivariate", "uid": "TBN-030"},
    {"task_dir": "gold_india_real_multivariate",          "task_type": "multivariate", "uid": "TBN-031"},
    {"task_dir": "india_gold_days_multivariate",          "task_type": "multivariate", "uid": "TBN-007"},
    {"task_dir": "lt_stock_longest_multivariate",         "task_type": "multivariate", "uid": "TBN-013"},
    {"task_dir": "lt_stock_minutes_multivariate",         "task_type": "multivariate", "uid": "TBN-012"},
    {"task_dir": "madrid_count_multivariate",             "task_type": "multivariate", "uid": "TBN-070"},
    {"task_dir": "madrid_cyclical_multivariate",          "task_type": "multivariate", "uid": "TBN-024"},
    {"task_dir": "madrid_hours_multivariate",             "task_type": "multivariate", "uid": "TBN-010"},
    {"task_dir": "madrid_noisy_multivariate",             "task_type": "multivariate", "uid": "TBN-057"},
    {"task_dir": "madrid_transport_multivariate",         "task_type": "multivariate", "uid": "TBN-056"},
    {"task_dir": "nyc_covid_healthcare_multivariate",     "task_type": "multivariate", "uid": "TBN-038"},
    {"task_dir": "soil_500_multivariate",                 "task_type": "multivariate", "uid": "TBN-044"},
    {"task_dir": "soil_nature_multivariate",              "task_type": "multivariate", "uid": "TBN-043"},
    {"task_dir": "split_smart_energy_multivariate",       "task_type": "multivariate", "uid": "TBN-035"},
    {"task_dir": "utah_manufacturing_multivariate",       "task_type": "multivariate", "uid": "TBN-020"},
    # ── Covariate (26) ────────────────────────────────────────────────────
    {"task_dir": "advertising_sales_covariate",           "task_type": "covariate",    "uid": "TBN-051"},
    {"task_dir": "building_manufacturing_covariate",      "task_type": "covariate",    "uid": "TBN-042"},
    {"task_dir": "california_energy_covariate",           "task_type": "covariate",    "uid": "TBN-037"},
    {"task_dir": "california_hourly_covariate",           "task_type": "covariate",    "uid": "TBN-011"},
    {"task_dir": "currency_dense_covariate",              "task_type": "covariate",    "uid": "TBN-063"},
    {"task_dir": "currency_monthly_covariate",            "task_type": "covariate",    "uid": "TBN-018"},
    {"task_dir": "cybersecurity_count_covariate",         "task_type": "covariate",    "uid": "TBN-072"},
    {"task_dir": "cybersecurity_software_covariate",      "task_type": "covariate",    "uid": "TBN-054"},
    {"task_dir": "gdp_noisy_covariate",                   "task_type": "covariate",    "uid": "TBN-004"},  # approx
    {"task_dir": "gdp_years_covariate",                   "task_type": "covariate",    "uid": "TBN-023"},
    {"task_dir": "mobility_transport_covariate",          "task_type": "covariate",    "uid": "TBN-059"},
    {"task_dir": "nifty_longest_covariate",               "task_type": "covariate",    "uid": "TBN-014"},
    {"task_dir": "nifty_minutes_covariate",               "task_type": "covariate",    "uid": "TBN-003"},
    {"task_dir": "nifty_nonstationary_covariate",         "task_type": "covariate",    "uid": "TBN-003"},  # same src
    {"task_dir": "nyc_covid_healthcare_covariate",        "task_type": "covariate",    "uid": "TBN-040"},
    {"task_dir": "solar_100_covariate",                   "task_type": "covariate",    "uid": "TBN-047"},
    {"task_dir": "solar_500_covariate",                   "task_type": "covariate",    "uid": "TBN-046"},
    {"task_dir": "solar_nature_covariate",                "task_type": "covariate",    "uid": "TBN-048"},
    {"task_dir": "stocks_continuous_covariate",           "task_type": "covariate",    "uid": "TBN-069"},
    {"task_dir": "stocks_daily_covariate",                "task_type": "covariate",    "uid": "TBN-009"},
    {"task_dir": "stocks_economics_covariate",            "task_type": "covariate",    "uid": "TBN-033"},
    {"task_dir": "stocks_real_covariate",                 "task_type": "covariate",    "uid": "TBN-034"},
    {"task_dir": "weather_climate_covariate",             "task_type": "covariate",    "uid": "TBN-029"},
    {"task_dir": "weather_cyclical_covariate",            "task_type": "covariate",    "uid": "TBN-027"},
]

# ── Frequency heuristics from task_dir suffix ─────────────────────────────────
_FREQ_MAP = {
    "minutes": ("min", 60, 60),   # freq, horizon, seasonality
    "hours":   ("h",   24, 24),
    "days":    ("D",   14,  7),
    "weeks":   ("W",    8,  1),
    "months":  ("ME",  12, 12),
    "quarterly": ("QE",  8,  4),
    "years":   ("YE",   1,  1),
    "seconds": ("s",   60,  1),
    "dense":   ("ME",  12, 12),   # assume monthly when no explicit freq
    "sparse":  ("ME",  12, 12),
    "continuous": ("D", 14, 7),
    "noisy":   ("h",   24, 24),
    "cyclical": ("h",  24, 24),
    "longest": ("D",   14,  7),
    "sales":   ("ME",  12, 12),
    "transport": ("h", 24, 24),
    "software": ("ME", 12, 12),
    "energy":  ("h",   24, 24),
    "healthcare": ("ME", 12, 12),
    "nature":  ("ME",  12, 12),
    "manufacturing": ("ME", 12, 12),
    "economics": ("ME", 12, 12),
    "climate": ("ME",  12, 12),
    "real":    ("D",   14,  7),
    "count":   ("h",   24, 24),
    "binary":  ("ME",  12, 12),
    "categorical": ("ME", 12, 12),
}


def _infer_freq(task_dir: str, task_yaml: dict) -> tuple[str, int, int]:
    """Return (pandas_freq, horizon, seasonality) from task dir name + yaml."""
    horizon = task_yaml.get("task", {}).get("forecast_horizon", 12)
    # Try to find a freq keyword in the task_dir name
    name_lower = task_dir.lower()
    for key, (freq, h, s) in _FREQ_MAP.items():
        if key in name_lower:
            return freq, horizon, s
    return "ME", horizon, 1  # monthly fallback


class TempusBenchLoader:
    """Download and load a single TempusBench task from GitHub.

    Each task contains:
    - task.yaml: context_window, forecast_horizon, csv filename, preprocessing
    - {name}.csv: time series data in TempusBench format

    The CSV format uses JSON-encoded arrays for timestamps and values:
      variable_name, timestamps, values, variable_type
      "target_1", '["2020-01T","2020-02T"]', '[1.0,2.5]', "target"

    Parameters
    ----------
    task_dir : str
        Task directory name, e.g. 'chickenpox_dense_univariate'
    task_type : str
        One of 'univariate', 'multivariate', 'covariate'
    unified_id : str | None
        catalog.db unified_id (e.g. 'TBN-062')
    save_dir : Path | None
        Override save directory. Defaults to E:\\datasets\\{uid}
    """

    def __init__(
        self,
        task_dir: str,
        task_type: str,
        unified_id: str | None = None,
        save_dir: Path | None = None,
    ):
        self.task_dir = task_dir
        self.task_type = task_type
        self.unified_id = unified_id or task_dir
        self.save_dir = save_dir or (config.DOWNLOAD_ROOT / self.unified_id)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._task_yaml: dict | None = None

    @property
    def csv_url(self) -> str:
        """Raw GitHub URL for the CSV file."""
        csv_name = f"{self.task_dir}.csv"
        return f"{_GH_RAW}/{self.task_type}/{self.task_dir}/{csv_name}"

    @property
    def yaml_url(self) -> str:
        """Raw GitHub URL for task.yaml."""
        return f"{_GH_RAW}/{self.task_type}/{self.task_dir}/task.yaml"

    @property
    def save_path(self) -> Path:
        return self.save_dir / f"{self.task_dir}.csv"

    def load_task_yaml(self) -> dict:
        """Fetch and parse task.yaml from GitHub."""
        if self._task_yaml is not None:
            return self._task_yaml
        resp = requests.get(self.yaml_url, timeout=30)
        resp.raise_for_status()
        self._task_yaml = yaml.safe_load(resp.text)
        return self._task_yaml

    def download(self, force: bool = False) -> Path:
        """Download the CSV from GitHub and save as sacred copy.

        Returns
        -------
        Path to the saved CSV file.
        """
        if self.save_path.exists() and not force:
            return self.save_path

        resp = requests.get(
            self.csv_url,
            timeout=60,
            headers=config.HEADERS,
        )
        resp.raise_for_status()
        self.save_path.write_bytes(resp.content)
        size_kb = self.save_path.stat().st_size / 1024
        print(f"  Saved: {self.save_path.name} ({size_kb:.0f} KB)")
        return self.save_path

    def load_raw_df(self) -> pd.DataFrame:
        """Load the raw CSV as-is (4 columns: variable_name, timestamps, values, variable_type).

        The file is never altered — this returns the original TempusBench format.
        """
        if not self.save_path.exists():
            raise FileNotFoundError(
                f"Not downloaded yet: {self.save_path}\nCall .download() first."
            )
        return pd.read_csv(self.save_path)

    def load_df(self, variable_type: str = "target") -> pd.DataFrame:
        """Load and parse into a wide pandas DataFrame.

        Parses the JSON-encoded timestamps and values arrays, then pivots
        to wide format: columns = variable names, index = datetime.

        Parameters
        ----------
        variable_type : 'target' | 'covariate' | 'all'
            Which rows to return. Use 'all' to include both targets and covariates.

        Returns
        -------
        pd.DataFrame with datetime index and one column per variable.
        """
        df_raw = self.load_raw_df()

        if variable_type != "all":
            df_raw = df_raw[df_raw["variable_type"] == variable_type]

        if df_raw.empty:
            return pd.DataFrame()

        frames = []
        for _, row in df_raw.iterrows():
            # Parse JSON arrays
            timestamps = _parse_json_array(row["timestamps"])
            values = _parse_json_array(row["values"])
            ts = pd.to_datetime(timestamps, utc=True).tz_localize(None)
            s = pd.Series(values, index=ts, name=row["variable_name"], dtype=float)
            frames.append(s)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=1).sort_index()

    def load_targets_df(self) -> pd.DataFrame:
        """Load only target variables as wide DataFrame."""
        return self.load_df("target")

    def load_covariates_df(self) -> pd.DataFrame:
        """Load only covariate variables as wide DataFrame."""
        return self.load_df("covariate")

    @property
    def horizon(self) -> int:
        cfg = self.load_task_yaml()
        return cfg.get("task", {}).get("forecast_horizon", 12)

    @property
    def context_window(self) -> int:
        cfg = self.load_task_yaml()
        return cfg.get("task", {}).get("context_window", 128)

    @property
    def seasonality(self) -> int:
        cfg = self.load_task_yaml()
        _, _, s = _infer_freq(self.task_dir, cfg)
        return s

    @property
    def pandas_freq(self) -> str:
        cfg = self.load_task_yaml()
        freq, _, _ = _infer_freq(self.task_dir, cfg)
        return freq

    def metadata(self) -> dict:
        """Return task metadata dict."""
        cfg = self.load_task_yaml()
        df = self.load_raw_df() if self.save_path.exists() else None
        n_targets = len(df[df["variable_type"] == "target"]) if df is not None else None
        n_covariates = len(df[df["variable_type"] == "covariate"]) if df is not None else None
        return {
            "unified_id": self.unified_id,
            "task_dir": self.task_dir,
            "task_type": self.task_type,
            "forecast_horizon": self.horizon,
            "context_window": self.context_window,
            "seasonality": self.seasonality,
            "pandas_freq": self.pandas_freq,
            "n_targets": n_targets,
            "n_covariates": n_covariates,
            "handle_missing": cfg.get("task", {}).get("dataset", {}).get("handle_missing"),
            "normalize": cfg.get("task", {}).get("dataset", {}).get("normalize"),
        }

    def to_fev_task(self, **task_kwargs):
        """Download if needed, convert to wide FEV format, return fev.Task.

        Handles target-only tasks as multi-series (one row per variable → FEV
        expects one row per series). For covariate tasks targets are used as
        the forecast target.
        """
        import fev
        import datasets as hf_datasets

        self.download()
        cfg = self.load_task_yaml()
        horizon, seasonality = self.horizon, self.seasonality

        targets_df = self.load_targets_df()
        if targets_df.empty:
            raise ValueError(f"No target variables in {self.task_dir}")

        # Infer actual frequency from modal time diff and resample to fill gaps.
        # This is required for FEV's timestamp validation (pd.infer_freq fails on gaps).
        freq_str = _infer_freq_from_df(targets_df)
        if freq_str:
            targets_df = targets_df.resample(freq_str).mean()

        # Build FEV long format: one row per target variable
        records = []
        for col in targets_df.columns:
            s = targets_df[col]
            if s.isna().all():
                continue
            records.append({
                "id": str(col),
                "timestamp": s.index.tolist(),
                "target": s.tolist(),
            })

        features = hf_datasets.Features({
            "id": hf_datasets.Value("string"),
            "timestamp": hf_datasets.Sequence(hf_datasets.Value("timestamp[us]")),
            "target": hf_datasets.Sequence(hf_datasets.Value("float64")),
        })
        hf_ds = hf_datasets.Dataset.from_list(records, features=features)

        # Cache
        cache_dir = config.DOWNLOAD_ROOT / "_cache" / self.unified_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        eval_path = cache_dir / "data.parquet"
        hf_ds.to_parquet(str(eval_path))

        params: dict[str, Any] = {
            "dataset_path": str(eval_path),
            "horizon": horizon,
            "seasonality": seasonality,
            "eval_metric": "MASE",
            "extra_metrics": ["MAE", "MSE", "RMSE"],
            "task_name": self.unified_id,
            "target": "target",
        }
        params.update(task_kwargs)
        return fev.Task(**params)

    @classmethod
    def from_uid(cls, uid: str) -> "TempusBenchLoader":
        """Create loader by catalog unified_id (e.g. 'TBN-062')."""
        entry = next((e for e in TEMPUSBENCH_REGISTRY if e["uid"] == uid), None)
        if entry is None:
            raise ValueError(f"No TempusBench task with uid '{uid}'")
        return cls(entry["task_dir"], entry["task_type"], unified_id=uid)

    @classmethod
    def from_task_dir(cls, task_dir: str) -> "TempusBenchLoader":
        """Create loader by task directory name."""
        entry = next((e for e in TEMPUSBENCH_REGISTRY if e["task_dir"] == task_dir), None)
        uid = entry["uid"] if entry else task_dir
        task_type = entry["task_type"] if entry else "univariate"
        return cls(task_dir, task_type, unified_id=uid)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_freq_from_df(df: pd.DataFrame) -> str | None:
    """Infer pandas frequency string from modal time difference in a DataFrame."""
    if len(df) < 2:
        return None
    try:
        diffs = df.index.to_series().diff().dropna()
        modal_diff = diffs.mode()[0]
        total_seconds = modal_diff.total_seconds()
        if total_seconds < 60:
            return "s"
        elif total_seconds < 3600:
            mins = int(total_seconds / 60)
            return f"{mins}min" if mins > 1 else "min"
        elif total_seconds < 86400:
            hours = int(total_seconds / 3600)
            return f"{hours}h" if hours > 1 else "h"
        elif total_seconds < 86400 * 7:
            days = int(total_seconds / 86400)
            return f"{days}D" if days > 1 else "D"
        elif total_seconds < 86400 * 32:
            return "W"
        elif total_seconds < 86400 * 100:
            return "ME"
        elif total_seconds < 86400 * 200:
            return "QE"
        else:
            return "YE"
    except Exception:
        return None


def _parse_json_array(val) -> list:
    """Parse a JSON-encoded array from a CSV cell."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("["):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(val)
                except Exception:
                    pass
    return []


# ── Batch download helper ─────────────────────────────────────────────────────

def download_all_tempusbench(
    db: "CatalogDB",
    report_every: int = 10,
    skip_existing: bool = True,
    rate_limit_secs: float = 0.5,
) -> list[dict]:
    """Download all 73 TempusBench tasks from GitHub.

    Saves CSVs to E:\\datasets\\{uid}\\{task_dir}.csv.
    Updates catalog.db download tracking for each task.

    Parameters
    ----------
    db : CatalogDB
    report_every : int
        Print progress report every N tasks.
    skip_existing : bool
        Skip tasks already downloaded (status='success').
    rate_limit_secs : float
        Pause between GitHub requests to avoid rate limiting.

    Returns
    -------
    List of result dicts: {uid, task_dir, status, file_kb, error}
    """
    results = []
    total = len(TEMPUSBENCH_REGISTRY)

    for i, entry in enumerate(TEMPUSBENCH_REGISTRY, 1):
        uid = entry["uid"]
        task_dir = entry["task_dir"]
        task_type = entry["task_type"]

        row = db.get_dataset_by_uid(uid)
        dataset_id = row["id"] if row else None

        if skip_existing and dataset_id:
            if db.get_download_status(dataset_id) == "success":
                print(f"[{i:2d}/{total}] SKIP {uid:10s} {task_dir}")
                results.append({"uid": uid, "task_dir": task_dir, "status": "skipped"})
                if i % report_every == 0 or i == total:
                    _print_progress(results, i, total, db)
                continue

        loader = TempusBenchLoader(task_dir, task_type, unified_id=uid)

        try:
            path = loader.download()
            file_size = path.stat().st_size

            if dataset_id:
                db.upsert_download(
                    dataset_id, status="success",
                    local_path=str(path),
                    file_size_bytes=file_size,
                    download_url=loader.csv_url,
                    url_type="github_raw",
                )
                db.add_to_budget(file_size)
                db.set_dataloader_created(dataset_id, 1)

            result = {"uid": uid, "task_dir": task_dir, "status": "success",
                      "file_kb": file_size / 1024}
            print(f"[{i:2d}/{total}] OK   {uid:10s} {task_dir:45s} {result['file_kb']:.0f} KB")

        except Exception as e:
            err = str(e)[:120]
            if dataset_id:
                db.upsert_download(dataset_id, status="failed", error_message=err)
            result = {"uid": uid, "task_dir": task_dir, "status": "failed", "error": err}
            print(f"[{i:2d}/{total}] FAIL {uid:10s} {task_dir:45s} {err[:50]}")

        results.append(result)
        time.sleep(rate_limit_secs)

        if i % report_every == 0 or i == total:
            _print_progress(results, i, total, db)

    return results


def _print_progress(results: list[dict], i: int, total: int, db: "CatalogDB") -> None:
    ok   = sum(1 for r in results if r["status"] == "success")
    fail = sum(1 for r in results if r["status"] == "failed")
    skip = sum(1 for r in results if r["status"] == "skipped")
    used, budget = db.get_budget()
    downloaded = db.get_downloaded_datasets()
    print(f"\n{'─'*72}")
    print(f"Progress {i}/{total} | {ok} ok / {fail} fail / {skip} skip | "
          f"Total DL: {len(downloaded)} | Budget: {used/1024/1024:.0f} MB")
    print(f"{'─'*72}\n")
