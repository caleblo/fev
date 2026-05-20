#!/usr/bin/env python3
"""
Download and create dataloaders for:
  TFM-002  Wiki Pageviews  (multi-resolution: daily/weekly/monthly from CHR-016)
  TFM-007  LibCity 15-min traffic  (SZ-Taxi 15T from autogluon/fev_datasets)
  TMO-117  Global Temp Benchmark  (1000 sequences from Google Drive JSONL)

All files saved to E:\\datasets\\{uid}\\ (sacred copies, never altered).
"""
import sys, os, json, math, time, shutil
from pathlib import Path

os.environ.setdefault("HF_HOME", r"E:\datasets\_hf_cache")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import fev
import datasets as hf_datasets
from catalog import CatalogDB, MetricsWriter
from catalog.config import DOWNLOAD_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# Seasonal naive helper
# ─────────────────────────────────────────────────────────────────────────────
def seasonal_naive(past_data, target_col, horizon, seasonality):
    preds = []
    for j in range(len(past_data)):
        ts = past_data[j][target_col]
        clean = [v for v in ts if v is not None and not math.isnan(v) and not math.isinf(v)]
        if not clean: clean = [0.0]
        s = max(seasonality, 1)
        pat = clean[-s:] if len(clean) >= s else clean
        preds.append((pat * ((horizon // len(pat)) + 1))[:horizon])
    return preds


def run_fev_eval(hf_ds, uid, horizon, seasonality, writer, dataset_id):
    """Build fev.Task from an HF Dataset, run SeasonalNaive, write metrics."""
    cache_dir = DOWNLOAD_ROOT / "_cache" / uid
    cache_dir.mkdir(parents=True, exist_ok=True)
    eval_path = cache_dir / "data.parquet"
    hf_ds.to_parquet(str(eval_path))

    task = fev.Task(
        dataset_path=str(eval_path), horizon=horizon, seasonality=seasonality,
        eval_metric="MASE", extra_metrics=["MAE", "MSE", "RMSE"],
        task_name=uid, target="target",
    )
    t0 = time.time()
    preds_per_window = []
    for window in task.iter_windows():
        past_data, _ = window.get_input_data()
        p = seasonal_naive(past_data, task.target_columns[0], horizon, seasonality)
        pred_ds = hf_datasets.Dataset.from_dict({"predictions": p})
        preds_per_window.append(hf_datasets.DatasetDict({task.target_columns[0]: pred_ds}))

    summary = task.evaluation_summary(
        preds_per_window, model_name="SeasonalNaive",
        inference_time_s=time.time() - t0, trained_on_this_dataset=False,
    )
    print(f"  MASE={summary['MASE']:.4f}  MAE={summary['MAE']:.4f}  RMSE={summary['RMSE']:.4f}")
    if dataset_id:
        writer.write_summary(summary, dataset_id=dataset_id)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# TFM-002: Wiki Pageviews
#   Source: CHR-016 wiki_daily_100k.parquet (100K Wikipedia articles, daily)
#   We also generate weekly and monthly aggregations.
# ─────────────────────────────────────────────────────────────────────────────
def process_tfm002(db, writer):
    print("\n" + "="*65)
    print("TFM-002  Wiki Pageviews (multi-resolution)")
    print("="*65)

    uid = "TFM-002"
    dataset_id = db.get_dataset_by_uid(uid)["id"]
    save_dir = DOWNLOAD_ROOT / uid
    save_dir.mkdir(parents=True, exist_ok=True)

    # Source: CHR-016 already downloaded
    chr016_path = DOWNLOAD_ROOT / "CHR-016" / "wiki_daily_100k.parquet"
    if not chr016_path.exists():
        raise FileNotFoundError(f"CHR-016 source not found: {chr016_path}\nRun the main pipeline first.")

    # ── Daily (copy of CHR-016) ──────────────────────────────────────────────
    daily_path = save_dir / "wiki_pageviews_daily.parquet"
    if not daily_path.exists():
        shutil.copy2(str(chr016_path), str(daily_path))
    print(f"  Daily : {daily_path.name} ({daily_path.stat().st_size // 1024} KB) — copied from CHR-016")

    # ── Weekly & Monthly aggregations ────────────────────────────────────────
    daily_ds = hf_datasets.Dataset.from_parquet(str(chr016_path))
    # CHR-016 has columns: id, timestamp, target  (each row = one series, wide array)

    for resample_freq, suffix, horizon, seasonality in [
        ("W",  "weekly",  52, 52),
        ("ME", "monthly", 12, 12),
    ]:
        out_path = save_dir / f"wiki_pageviews_{suffix}.parquet"
        if not out_path.exists():
            records = []
            for i in range(len(daily_ds)):
                row = daily_ds[i]
                ts = pd.Series(
                    row["target"],
                    index=pd.to_datetime(row["timestamp"]),
                    dtype=float,
                )
                agg = ts.resample(resample_freq).mean().dropna()
                if len(agg) < horizon * 2:
                    continue
                records.append({
                    "id": row["id"],
                    "timestamp": agg.index.tolist(),
                    "target": agg.tolist(),
                })
                if (i + 1) % 10000 == 0:
                    print(f"    {suffix}: processed {i+1}/{len(daily_ds)} series...")

            features = hf_datasets.Features({
                "id": hf_datasets.Value("string"),
                "timestamp": hf_datasets.Sequence(hf_datasets.Value("timestamp[us]")),
                "target": hf_datasets.Sequence(hf_datasets.Value("float64")),
            })
            agg_ds = hf_datasets.Dataset.from_list(records, features=features)
            agg_ds.to_parquet(str(out_path))
        size_kb = out_path.stat().st_size // 1024
        print(f"  {suffix.capitalize()}: {out_path.name} ({size_kb} KB)")

    # ── Dataloader ────────────────────────────────────────────────────────────
    loader_path = save_dir / "dataloader.py"
    loader_path.write_text('''"""
TFM-002 — Wiki Pageviews dataloader
Source: Wikipedia pageviews (100K articles), aggregated from daily CHR-016 data.
Files: wiki_pageviews_daily.parquet, wiki_pageviews_weekly.parquet,
       wiki_pageviews_monthly.parquet
Schema: id (article name), timestamp (sequence), target (pageview count sequence)
"""
from pathlib import Path
import datasets as hf_datasets
import pandas as pd

DATA_DIR = Path(__file__).parent


def load(resolution="daily") -> hf_datasets.Dataset:
    """Load Wiki Pageviews dataset.

    Parameters
    ----------
    resolution : "daily" | "weekly" | "monthly"

    Returns
    -------
    HF Dataset with columns: id, timestamp, target
    Each row is one Wikipedia article time series.
    """
    fname = f"wiki_pageviews_{resolution}.parquet"
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}. Run download_tfm_tmo_datasets.py first.")
    return hf_datasets.Dataset.from_parquet(str(path))


def load_df(resolution="daily") -> pd.DataFrame:
    """Load as a long-format pandas DataFrame: id, timestamp, target."""
    ds = load(resolution)
    rows = []
    for r in ds:
        s = pd.Series(r["target"], index=pd.to_datetime(r["timestamp"]), name=r["id"])
        rows.append(s)
    return pd.concat(rows, axis=1).T  # shape: (n_series, n_timestamps)


def metadata() -> dict:
    return {
        "unified_id": "TFM-002",
        "title": "Wiki Pageviews (hourly/daily/weekly/monthly)",
        "resolutions": ["daily", "weekly", "monthly"],
        "source": "autogluon/chronos_datasets::wiki_daily_100k",
        "daily_path": str(DATA_DIR / "wiki_pageviews_daily.parquet"),
        "weekly_path": str(DATA_DIR / "wiki_pageviews_weekly.parquet"),
        "monthly_path": str(DATA_DIR / "wiki_pageviews_monthly.parquet"),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(metadata(), indent=2))
    for res in ["daily", "weekly", "monthly"]:
        ds = load(res)
        print(f"  {res}: {len(ds)} series")
''', encoding="utf-8")
    print(f"  Loader: {loader_path}")

    # ── Update DB ─────────────────────────────────────────────────────────────
    db.upsert_download(
        dataset_id, status="success",
        local_path=str(daily_path), file_size_bytes=daily_path.stat().st_size,
        download_url="derived:CHR-016/wiki_daily_100k.parquet", url_type="derived",
    )
    db.set_dataloader_created(dataset_id, 1)

    # ── FEV evaluation on daily (horizon=7, seasonality=7) ───────────────────
    print("  Evaluating (daily, h=7, s=7)...")
    daily_ds_eval = hf_datasets.Dataset.from_parquet(str(daily_path))
    # Use a sample (50K series is large) — first 1000 for speed
    sample = daily_ds_eval.select(range(min(1000, len(daily_ds_eval))))
    run_fev_eval(sample, uid, horizon=7, seasonality=7, writer=writer, dataset_id=dataset_id)
    print("  TFM-002 complete.")


# ─────────────────────────────────────────────────────────────────────────────
# TFM-007: LibCity 15-min Traffic
#   Source: autogluon/fev_datasets::SZ_TAXI_15T (already downloaded as FEV-010)
# ─────────────────────────────────────────────────────────────────────────────
def process_tfm007(db, writer):
    print("\n" + "="*65)
    print("TFM-007  LibCity 15-min Traffic")
    print("="*65)

    uid = "TFM-007"
    dataset_id = db.get_dataset_by_uid(uid)["id"]
    save_dir = DOWNLOAD_ROOT / uid
    save_dir.mkdir(parents=True, exist_ok=True)

    # Source: FEV-010 SZ_TAXI_15T already downloaded (156 series, 15-min)
    fev010_path = DOWNLOAD_ROOT / "FEV-010" / "SZ_TAXI_15T.parquet"
    if not fev010_path.exists():
        # Fallback: download from HF
        print("  FEV-010 not found, downloading from HF...")
        hf_ds = hf_datasets.load_dataset(
            "autogluon/fev_datasets", "SZ_TAXI_15T", split="train", num_proc=1
        )
        fev010_path.parent.mkdir(parents=True, exist_ok=True)
        hf_ds.to_parquet(str(fev010_path))

    # Copy / symlink to TFM-007 dir
    dest_path = save_dir / "libcity_sz_taxi_15min.parquet"
    if not dest_path.exists():
        shutil.copy2(str(fev010_path), str(dest_path))
    size_kb = dest_path.stat().st_size // 1024
    print(f"  SZ-Taxi 15-min: {dest_path.name} ({size_kb} KB)")

    # ── Dataloader ────────────────────────────────────────────────────────────
    loader_path = save_dir / "dataloader.py"
    loader_path.write_text(f'''"""
TFM-007 — LibCity 15-minute Traffic dataloader
Source: autogluon/fev_datasets::SZ_TAXI_15T (SZ-Taxi, Shenzhen Taxi dataset)
        from LibCity benchmark (https://github.com/LibCity/Bigscity-LibCity)
File: libcity_sz_taxi_15min.parquet
Schema: id (taxi zone ID), timestamp (15-min intervals), target (ride count)
156 time series, 15-minute resolution
"""
from pathlib import Path
import datasets as hf_datasets
import pandas as pd

DATA_DIR = Path(__file__).parent
DATA_FILE = DATA_DIR / "libcity_sz_taxi_15min.parquet"


def load() -> hf_datasets.Dataset:
    """Load SZ-Taxi 15-min traffic dataset as HF Dataset.

    Returns
    -------
    HF Dataset with columns: id, timestamp, target
    Each row is one taxi zone time series at 15-min frequency.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Not found: {{DATA_FILE}}. Run download_tfm_tmo_datasets.py.")
    return hf_datasets.Dataset.from_parquet(str(DATA_FILE))


def load_df() -> pd.DataFrame:
    """Load as wide pandas DataFrame: index=datetime, columns=taxi zone IDs."""
    ds = load()
    frames = []
    for r in ds:
        s = pd.Series(r["target"], index=pd.to_datetime(r["timestamp"]),
                      name=r["id"], dtype=float)
        frames.append(s)
    return pd.concat(frames, axis=1)


def metadata() -> dict:
    ds = load()
    return {{
        "unified_id": "TFM-007",
        "title": "LibCity 15-minute Traffic (SZ-Taxi)",
        "n_series": len(ds),
        "frequency": "15-min",
        "source": "autogluon/fev_datasets::SZ_TAXI_15T",
        "file": str(DATA_FILE),
    }}


if __name__ == "__main__":
    import json
    print(json.dumps(metadata(), indent=2, default=str))
''', encoding="utf-8")
    print(f"  Loader: {loader_path}")

    # ── Update DB ─────────────────────────────────────────────────────────────
    db.upsert_download(
        dataset_id, status="success",
        local_path=str(dest_path), file_size_bytes=dest_path.stat().st_size,
        download_url="hf://autogluon/fev_datasets/SZ_TAXI_15T", url_type="huggingface",
    )
    db.set_dataloader_created(dataset_id, 1)

    # ── FEV evaluation (h=96, s=96 for 15-min) ────────────────────────────────
    print("  Evaluating (h=96, s=96, 15-min)...")
    raw_ds = hf_datasets.Dataset.from_parquet(str(fev010_path))
    # SZ_TAXI_15T has 'target' column already
    run_fev_eval(raw_ds, uid, horizon=96, seasonality=96, writer=writer, dataset_id=dataset_id)
    print("  TFM-007 complete.")


# ─────────────────────────────────────────────────────────────────────────────
# TMO-117: Global Temp Benchmark
#   Source: Google Drive (Time-MoE benchmark) — 1000 sequences in JSONL
#   Format: {"sequence": [float, ...]}  — each line is one temperature series
# ─────────────────────────────────────────────────────────────────────────────
def process_tmo117(db, writer):
    print("\n" + "="*65)
    print("TMO-117  Global Temp (Evaluation Benchmark)")
    print("="*65)

    uid = "TMO-117"
    dataset_id = db.get_dataset_by_uid(uid)["id"]
    save_dir = DOWNLOAD_ROOT / uid
    save_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = save_dir / "global_temp_test.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} not found.\n"
            "Download from Google Drive: https://drive.google.com/drive/folders/1KjnAYr9X3D-jyJpo4yM7Giyq5V1Hga_7\n"
            "Run: python -c \"import gdown; gdown.download_folder("
            "'https://drive.google.com/drive/folders/1KjnAYr9X3D-jyJpo4yM7Giyq5V1Hga_7',"
            f" output='{save_dir}/', quiet=False, use_cookies=False)\""
        )

    size_mb = jsonl_path.stat().st_size / 1024 / 1024
    print(f"  Source: {jsonl_path.name} ({size_mb:.0f} MB)")

    # ── Convert JSONL → HF Dataset ──────────────────────────────────────────
    parquet_path = save_dir / "global_temp_test.parquet"
    if not parquet_path.exists():
        print("  Converting JSONL → parquet...")
        records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                d = json.loads(line.strip())
                seq = d["sequence"]
                # Synthetic integer index — sequences vary in length (can be 1000s of steps)
                # Use daily freq from 2000-01-01 to avoid pandas timestamp overflow
                n = len(seq)
                idx = pd.date_range("2000-01-01", periods=n, freq="D")
                records.append({
                    "id": f"global_temp_{i:04d}",
                    "timestamp": idx.tolist(),
                    "target": [float(v) for v in seq],
                })

        features = hf_datasets.Features({
            "id": hf_datasets.Value("string"),
            "timestamp": hf_datasets.Sequence(hf_datasets.Value("timestamp[us]")),
            "target": hf_datasets.Sequence(hf_datasets.Value("float64")),
        })
        hf_ds = hf_datasets.Dataset.from_list(records, features=features)
        hf_ds.to_parquet(str(parquet_path))
        print(f"  Saved: {parquet_path.name} ({parquet_path.stat().st_size // 1024} KB)")

    # ── Dataloader ────────────────────────────────────────────────────────────
    loader_path = save_dir / "dataloader.py"
    loader_path.write_text('''"""
TMO-117 — Global Temp Evaluation Benchmark dataloader
Source: Time-MoE Google Drive benchmark suite
        https://drive.google.com/drive/folders/1KjnAYr9X3D-jyJpo4yM7Giyq5V1Hga_7
File: global_temp_test.jsonl / global_temp_test.parquet
Schema: 1000 temperature sequences, each a variable-length list of floats.
        Timestamps are synthetic (monthly index from 2000-01-01).
"""
from pathlib import Path
import json
import datasets as hf_datasets
import pandas as pd

DATA_DIR = Path(__file__).parent
PARQUET_FILE = DATA_DIR / "global_temp_test.parquet"
JSONL_FILE = DATA_DIR / "global_temp_test.jsonl"


def load() -> hf_datasets.Dataset:
    """Load Global Temp benchmark as HF Dataset.

    Returns
    -------
    HF Dataset with columns: id, timestamp, target
    1000 rows, each a global temperature time series.
    Timestamps use a synthetic monthly index (no real date metadata in source).
    """
    if PARQUET_FILE.exists():
        return hf_datasets.Dataset.from_parquet(str(PARQUET_FILE))
    raise FileNotFoundError(f"Not found: {PARQUET_FILE}. Run download_tfm_tmo_datasets.py.")


def load_raw() -> list[list[float]]:
    """Load raw sequences from the original JSONL (list of float lists)."""
    sequences = []
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            sequences.append(d["sequence"])
    return sequences


def load_df() -> pd.DataFrame:
    """Load as wide DataFrame: rows=sequences, columns=time steps."""
    sequences = load_raw()
    max_len = max(len(s) for s in sequences)
    return pd.DataFrame(
        [s + [float("nan")] * (max_len - len(s)) for s in sequences]
    )


def metadata() -> dict:
    seqs = load_raw()
    return {
        "unified_id": "TMO-117",
        "title": "Global Temp (Evaluation Benchmark)",
        "n_sequences": len(seqs),
        "seq_lengths": {"min": min(len(s) for s in seqs),
                        "max": max(len(s) for s in seqs),
                        "mean": sum(len(s) for s in seqs) // len(seqs)},
        "source": "Time-MoE Google Drive benchmark",
        "jsonl_file": str(JSONL_FILE),
        "parquet_file": str(PARQUET_FILE),
    }


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(metadata(), indent=2))
''', encoding="utf-8")
    print(f"  Loader: {loader_path}")

    # ── Update DB ─────────────────────────────────────────────────────────────
    db.upsert_download(
        dataset_id, status="success",
        local_path=str(jsonl_path), file_size_bytes=jsonl_path.stat().st_size,
        download_url="https://drive.google.com/drive/folders/1KjnAYr9X3D-jyJpo4yM7Giyq5V1Hga_7",
        url_type="google_drive",
    )
    db.add_to_budget(jsonl_path.stat().st_size)
    db.set_dataloader_created(dataset_id, 1)

    # ── FEV evaluation (h=12, s=12 monthly) ──────────────────────────────────
    print("  Evaluating (h=12, s=12)...")
    hf_ds = hf_datasets.Dataset.from_parquet(str(parquet_path))
    run_fev_eval(hf_ds, uid, horizon=12, seasonality=12, writer=writer, dataset_id=dataset_id)
    print("  TMO-117 complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    db = CatalogDB()
    db.run_migrations()
    writer = MetricsWriter(db)

    process_tfm002(db, writer)
    process_tfm007(db, writer)
    process_tmo117(db, writer)

    # Final state
    used, budget = db.get_budget()
    print(f"\n{'='*65}")
    print(f"Done. Budget: {used/1024/1024:.0f} MB | Metrics: {len(db.get_metrics())} rows")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
