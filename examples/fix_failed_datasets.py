#!/usr/bin/env python3
"""Fix the 5 failed datasets: custom target columns + weatherbench chunking."""
import sys
import os
import math
import time
from pathlib import Path

os.environ["HF_HOME"] = r"E:\datasets\_hf_cache"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fev
import numpy as np
import datasets as hf_datasets
from catalog import CatalogDB, MetricsWriter

# Target column mapping for non-standard datasets
FIX_DATASETS = [
    {
        "hf_config": "electricity_15min",
        "catalog_uid": "CHR-019",
        "target": "consumption_kW",
        "horizon": 96,
        "seasonality": 96,
    },
    {
        "hf_config": "monash_rideshare",
        "catalog_uid": "CHR-031",
        "target": "price_mean",  # primary forecast target
        "horizon": 24,
        "seasonality": 24,
    },
    {
        "hf_config": "monash_saugeenday",
        "catalog_uid": "CHR-032",
        "target": "T1",
        "horizon": 14,
        "seasonality": 7,
    },
    {
        "hf_config": "monash_temperature_rain",
        "catalog_uid": "CHR-033",
        "target": "t_mean",
        "horizon": 14,
        "seasonality": 7,
    },
    {
        # Weatherbench daily: has 'target' column but 225K series causes Arrow overflow.
        # Fix: limit to first 10K series for evaluation.
        "hf_config": "weatherbench_daily",
        "catalog_uid": "CHR-013",
        "target": "target",
        "horizon": 14,
        "seasonality": 7,
        "max_series": 10000,
    },
]


def seasonal_naive(past_data, target_col, horizon, seasonality):
    preds = []
    for j in range(len(past_data)):
        ts = past_data[j][target_col]
        clean = [v for v in ts if v is not None and not math.isnan(v) and not math.isinf(v)]
        if not clean:
            clean = [0.0]
        s = max(seasonality, 1)
        if len(clean) >= s:
            pat = clean[-s:]
            forecast = (list(pat) * ((horizon // s) + 1))[:horizon]
        else:
            forecast = [clean[-1]] * horizon
        preds.append(forecast)
    return preds


def main():
    db = CatalogDB()
    writer = MetricsWriter(db)

    print("=" * 75)
    print(f"FIXING {len(FIX_DATASETS)} FAILED DATASETS")
    print("=" * 75)

    for i, cfg in enumerate(FIX_DATASETS, 1):
        config = cfg["hf_config"]
        uid = cfg["catalog_uid"]
        target = cfg["target"]
        horizon = cfg["horizon"]
        seasonality = cfg["seasonality"]
        max_series = cfg.get("max_series")

        print(f"\n{'-' * 75}")
        print(f"[{i}/{len(FIX_DATASETS)}] {uid} -- {config} (target={target})")
        print(f"{'-' * 75}")

        try:
            # 1. Download
            print("  1. Downloading...")
            t0 = time.time()
            hf_ds = hf_datasets.load_dataset(
                "autogluon/chronos_datasets", config, split="train", num_proc=1
            )
            dl_time = time.time() - t0
            n_total = len(hf_ds)
            print(f"     {n_total} series ({dl_time:.1f}s)")

            # Truncate if needed (weatherbench)
            if max_series and n_total > max_series:
                hf_ds = hf_ds.select(range(max_series))
                print(f"     Truncated to {max_series} series for evaluation")

            # Save sacred copy (full dataset, not truncated)
            save_dir = Path(r"E:\datasets") / uid
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{config}.parquet"
            if not save_path.exists():
                # Save full dataset if not already saved
                full_ds = hf_datasets.load_dataset(
                    "autogluon/chronos_datasets", config, split="train", num_proc=1
                ) if max_series else hf_ds
                full_ds.to_parquet(str(save_path))
            file_size = save_path.stat().st_size
            print(f"     File: {save_path.name} ({file_size/1024/1024:.1f} MB)")

            # Update catalog.db
            cat_row = db.get_dataset_by_uid(uid)
            dataset_id = cat_row["id"] if cat_row else None
            if dataset_id:
                db.upsert_download(
                    dataset_id, status="success", local_path=str(save_path),
                    file_size_bytes=file_size,
                    download_url=f"hf://autogluon/chronos_datasets/{config}",
                    url_type="huggingface",
                )
                if db.get_download_status(dataset_id) != "success":
                    db.add_to_budget(file_size)
                db.set_dataloader_created(dataset_id, 1)

            # 2. For evaluation: save truncated version as temp parquet
            eval_path = save_dir / f"{config}_eval.parquet"
            hf_ds.to_parquet(str(eval_path))

            # 3. Create Task with custom target
            print(f"  2. fev.Task (target={target}, h={horizon}, s={seasonality})")
            task = fev.Task(
                dataset_path=str(eval_path),
                horizon=horizon,
                seasonality=seasonality,
                eval_metric="MASE",
                extra_metrics=["MAE", "MSE", "RMSE"],
                target=target,
                task_name=uid,
            )

            # 4. Predict
            print("  3. Seasonal Naive...")
            t0 = time.time()
            preds_per_window = []
            for window in task.iter_windows():
                past_data, future_data = window.get_input_data()
                p = seasonal_naive(past_data, task.target_columns[0], horizon, seasonality)
                pred_ds = hf_datasets.Dataset.from_dict({"predictions": p})
                preds_per_window.append(
                    hf_datasets.DatasetDict({task.target_columns[0]: pred_ds})
                )
            infer_time = time.time() - t0

            # 5. Evaluate
            print("  4. Metrics...")
            summary = task.evaluation_summary(
                preds_per_window, model_name="SeasonalNaive",
                inference_time_s=infer_time, trained_on_this_dataset=False,
            )
            print(f"     MASE={summary['MASE']:.4f}  MAE={summary['MAE']:.4f}  "
                  f"MSE={summary['MSE']:.4f}  RMSE={summary['RMSE']:.4f}")

            # 6. Write to DB
            if dataset_id:
                n = writer.write_summary(summary, dataset_id=dataset_id)
                print(f"  5. Wrote {n} metrics to catalog.db")

            # Cleanup eval temp file
            if eval_path != save_path:
                eval_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Final report
    used, budget = db.get_budget()
    all_metrics = db.get_metrics()
    downloaded = db.get_downloaded_datasets()

    print(f"\n{'=' * 75}")
    print("FINAL STATE")
    print(f"{'=' * 75}")
    print(f"Downloaded:  {len(downloaded)} datasets")
    print(f"Metrics:     {len(all_metrics)} rows")
    print(f"Budget:      {used/1024/1024/1024:.2f} GB / {budget/1e9:.0f} GB")

    # Show any remaining failures
    import sqlite3
    conn = sqlite3.connect(str(db._path))
    fails = conn.execute("""
        SELECT d.unified_id, d.title, dl.error_message
        FROM dataset_downloads dl JOIN datasets d ON d.id=dl.dataset_id
        WHERE dl.status='failed'
    """).fetchall()
    conn.close()
    if fails:
        print(f"\nRemaining failures ({len(fails)}):")
        for uid, title, err in fails:
            print(f"  {uid} {title[:40]} — {(err or '')[:60]}")
    else:
        print("\nNo remaining failures!")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
