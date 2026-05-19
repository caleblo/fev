#!/usr/bin/env python3
"""Real dataset pipeline: HF download → save → model → metrics → catalog.db"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fev
import numpy as np
import datasets as hf_datasets
from catalog import CatalogDB, MetricsWriter


TEST_DATASETS = [
    {
        "hf_path": "autogluon/chronos_datasets",
        "hf_config": "monash_cif_2016",
        "catalog_uid": "CHR-045",
        "horizon": 12,
        "seasonality": 1,
    },
    {
        "hf_path": "autogluon/chronos_datasets",
        "hf_config": "monash_hospital",
        "catalog_uid": "CHR-043",
        "horizon": 12,
        "seasonality": 12,
    },
    {
        "hf_path": "autogluon/chronos_datasets",
        "hf_config": "monash_m1_monthly",
        "catalog_uid": "CHR-047",
        "horizon": 12,
        "seasonality": 12,
    },
    {
        "hf_path": "autogluon/chronos_datasets",
        "hf_config": "monash_tourism_monthly",
        "catalog_uid": "CHR-052",
        "horizon": 24,
        "seasonality": 12,
    },
    {
        "hf_path": "autogluon/chronos_datasets",
        "hf_config": "monash_nn5_weekly",
        "catalog_uid": "CHR-050",
        "horizon": 8,
        "seasonality": 1,
    },
]


def seasonal_naive_predict(past_data, task, horizon, seasonality):
    """Simple seasonal naive: tile last season_length values."""
    preds_list = []
    for j in range(len(past_data)):
        ts = past_data[j][task.target_columns[0]]
        season = max(seasonality, 1)
        if len(ts) >= season:
            pattern = ts[-season:]
            repeats = (horizon // season) + 1
            forecast = (list(pattern) * repeats)[:horizon]
        else:
            forecast = [ts[-1]] * horizon
        preds_list.append(forecast)
    return preds_list


def main():
    print("=" * 70)
    print("REAL DATASET PIPELINE: Download -> Save -> Model -> Metrics -> DB")
    print("=" * 70)

    db = CatalogDB()
    db.run_migrations()
    writer = MetricsWriter(db)
    all_summaries = []

    for i, cfg in enumerate(TEST_DATASETS, 1):
        uid = cfg["catalog_uid"]
        config_name = cfg["hf_config"]
        print(f"\n{'-' * 70}")
        print(f"[{i}/{len(TEST_DATASETS)}] {uid} -- {config_name}")
        print(f"{'-' * 70}")

        # Step 1: Download via HF
        print(f"  1. Downloading from HuggingFace...")
        t0 = time.time()
        hf_ds = hf_datasets.load_dataset(
            cfg["hf_path"], config_name, split="train", num_proc=1
        )
        dl_time = time.time() - t0
        print(f"     {len(hf_ds)} series ({dl_time:.1f}s)")

        # Save to E:\datasets\ (sacred copy)
        save_dir = Path(r"E:\datasets") / uid
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{config_name}.parquet"
        hf_ds.to_parquet(str(save_path))
        file_size = save_path.stat().st_size
        print(f"     Saved: {save_path.name} ({file_size / 1024:.0f} KB)")

        # Update catalog.db
        cat_row = db.get_dataset_by_uid(uid)
        if not cat_row:
            cat_row = db.search_dataset_by_title(
                config_name.replace("monash_", "").replace("_", " ")
            )
        dataset_id = cat_row["id"] if cat_row else None
        if dataset_id:
            db.upsert_download(
                dataset_id, status="success",
                local_path=str(save_path),
                file_size_bytes=file_size,
                download_url=f"hf://{cfg['hf_path']}/{config_name}",
                url_type="huggingface",
            )
            db.add_to_budget(file_size)
            db.set_dataloader_created(dataset_id, 1)
            print(f"     catalog.db updated (id={dataset_id})")
        else:
            print(f"     WARNING: {uid} not in catalog")

        # Step 2: fev.Task
        print(f"  2. fev.Task (h={cfg['horizon']}, s={cfg['seasonality']})")
        task = fev.Task(
            dataset_path=cfg["hf_path"],
            dataset_config=config_name,
            horizon=cfg["horizon"],
            seasonality=cfg["seasonality"],
            eval_metric="MASE",
            extra_metrics=["MAE", "MSE", "RMSE"],
            task_name=uid,
        )

        # Step 3: Seasonal Naive
        print(f"  3. Seasonal Naive...")
        t0 = time.time()
        predictions_per_window = []
        for window in task.iter_windows():
            past_data, future_data = window.get_input_data()
            preds = seasonal_naive_predict(
                past_data, task, cfg["horizon"], cfg["seasonality"]
            )
            pred_ds = hf_datasets.Dataset.from_dict({"predictions": preds})
            predictions_per_window.append(
                hf_datasets.DatasetDict({task.target_columns[0]: pred_ds})
            )
        infer_time = time.time() - t0

        # Step 4: Evaluate
        print(f"  4. Metrics via fev...")
        summary = task.evaluation_summary(
            predictions_per_window,
            model_name="SeasonalNaive",
            inference_time_s=infer_time,
            trained_on_this_dataset=False,
        )
        print(f"     MASE={summary['MASE']:.4f}  MAE={summary['MAE']:.4f}  "
              f"MSE={summary['MSE']:.4f}  RMSE={summary['RMSE']:.4f}")

        # Step 5: Write to DB
        if dataset_id:
            n = writer.write_summary(summary, dataset_id=dataset_id)
            print(f"  5. Wrote {n} metrics to catalog.db")

        all_summaries.append({"uid": uid, "config": config_name, "summary": summary})

    # Final report
    print(f"\n{'=' * 70}")
    print("FINAL REPORT")
    print(f"{'=' * 70}")
    print(f"\n{'Dataset':<14} {'MASE':>8} {'MAE':>12} {'MSE':>14} {'RMSE':>12}")
    print("-" * 62)
    for r in all_summaries:
        s = r["summary"]
        print(f"{r['uid']:<14} {s['MASE']:>8.4f} {s['MAE']:>12.4f} "
              f"{s['MSE']:>14.4f} {s['RMSE']:>12.4f}")

    used, budget = db.get_budget()
    print(f"\nBudget: {used / 1024:.1f} KB / {budget / 1e9:.0f} GB")
    print(f"Metric rows: {len(db.get_metrics())}")
    print(f"Downloaded:  {len(db.get_downloaded_datasets())}")
    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
