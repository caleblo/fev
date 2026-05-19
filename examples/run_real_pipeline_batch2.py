#!/usr/bin/env python3
"""Batch 2: 5 more real (non-synthetic) datasets through the full pipeline."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fev
import datasets as hf_datasets
from catalog import CatalogDB, MetricsWriter

# 5 real-world Chronos benchmark datasets (no synthetic, no KernelSynth/TSMixup)
TEST_DATASETS = [
    {
        "hf_path": "autogluon/chronos_datasets",
        "hf_config": "monash_australian_electricity",
        "catalog_uid": "CHR-025",
        "horizon": 24,
        "seasonality": 24,
    },
    {
        "hf_path": "autogluon/chronos_datasets",
        "hf_config": "exchange_rate",
        "catalog_uid": "CHR-042",
        "horizon": 30,
        "seasonality": 1,
    },
    {
        "hf_path": "autogluon/chronos_datasets",
        "hf_config": "monash_m3_monthly",
        "catalog_uid": "CHR-048",
        "horizon": 12,
        "seasonality": 12,
    },
    {
        "hf_path": "autogluon/chronos_datasets",
        "hf_config": "monash_m3_quarterly",
        "catalog_uid": "CHR-049",
        "horizon": 8,
        "seasonality": 4,
    },
    {
        "hf_path": "autogluon/chronos_datasets",
        "hf_config": "monash_m1_quarterly",
        "catalog_uid": "CHR-048",  # reusing slot — M1 Quarterly
        "horizon": 8,
        "seasonality": 4,
    },
]


def seasonal_naive(past_data, task, horizon, seasonality):
    import math
    preds = []
    for j in range(len(past_data)):
        ts = past_data[j][task.target_columns[0]]
        # Filter out NaN/Inf for a clean pattern
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
    print("=" * 70)
    print("BATCH 2: 5 real datasets (synthetic excluded)")
    print("=" * 70)

    db = CatalogDB()
    db.run_migrations()
    writer = MetricsWriter(db)
    results = []
    total_tokens_est = 0

    for i, cfg in enumerate(TEST_DATASETS, 1):
        uid = cfg["catalog_uid"]
        config_name = cfg["hf_config"]
        print(f"\n{'-' * 70}")
        print(f"[{i}/{len(TEST_DATASETS)}] {uid} -- {config_name}")
        print(f"{'-' * 70}")

        # 1. Download
        print("  1. Downloading from HuggingFace...")
        t0 = time.time()
        hf_ds = hf_datasets.load_dataset(
            cfg["hf_path"], config_name, split="train", num_proc=1
        )
        dl_time = time.time() - t0
        n_series = len(hf_ds)
        print(f"     {n_series} series ({dl_time:.1f}s)")

        # Save sacred copy
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
                local_path=str(save_path), file_size_bytes=file_size,
                download_url=f"hf://{cfg['hf_path']}/{config_name}",
                url_type="huggingface",
            )
            db.add_to_budget(file_size)
            db.set_dataloader_created(dataset_id, 1)

        # 2. fev.Task
        print(f"  2. fev.Task (h={cfg['horizon']}, s={cfg['seasonality']})")
        task = fev.Task(
            dataset_path=cfg["hf_path"], dataset_config=config_name,
            horizon=cfg["horizon"], seasonality=cfg["seasonality"],
            eval_metric="MASE", extra_metrics=["MAE", "MSE", "RMSE"],
            task_name=uid,
        )

        # 3. Model
        print("  3. Seasonal Naive...")
        t0 = time.time()
        preds_per_window = []
        for window in task.iter_windows():
            past_data, future_data = window.get_input_data()
            p = seasonal_naive(past_data, task, cfg["horizon"], cfg["seasonality"])
            pred_ds = hf_datasets.Dataset.from_dict({"predictions": p})
            preds_per_window.append(
                hf_datasets.DatasetDict({task.target_columns[0]: pred_ds})
            )
        infer_time = time.time() - t0

        # 4. Evaluate
        print("  4. Metrics...")
        summary = task.evaluation_summary(
            preds_per_window, model_name="SeasonalNaive",
            inference_time_s=infer_time, trained_on_this_dataset=False,
        )
        print(f"     MASE={summary['MASE']:.4f}  MAE={summary['MAE']:.4f}  "
              f"MSE={summary['MSE']:.4f}  RMSE={summary['RMSE']:.4f}")

        # 5. Write to DB
        if dataset_id:
            n = writer.write_summary(summary, dataset_id=dataset_id)
            print(f"  5. Wrote {n} metrics to catalog.db")

        # Token estimation: ~4 tokens/float, estimate per-series processing
        # FEV processes: past_data arrays + predictions + metric computation
        # Rough: n_series * avg_len * 4 tokens for data, + overhead
        avg_len = sum(len(hf_ds[j]["target"]) for j in range(min(10, n_series))) / min(10, n_series)
        data_tokens = int(n_series * avg_len * 2)  # read past + write preds
        overhead_tokens = 500  # task setup, metric compute, DB writes
        est_tokens = data_tokens + overhead_tokens
        total_tokens_est += est_tokens

        results.append({
            "uid": uid, "config": config_name, "summary": summary,
            "n_series": n_series, "file_kb": file_size / 1024,
            "dl_time": dl_time, "infer_time": infer_time,
            "est_tokens": est_tokens,
        })

    # Final report
    print(f"\n{'=' * 70}")
    print("FINAL REPORT")
    print(f"{'=' * 70}")
    print(f"\n{'Dataset':<12} {'Series':>7} {'KB':>8} {'MASE':>8} {'MAE':>12} "
          f"{'RMSE':>10} {'DL(s)':>6} {'Inf(s)':>6} {'~Tokens':>9}")
    print("-" * 86)
    for r in results:
        s = r["summary"]
        print(f"{r['uid']:<12} {r['n_series']:>7} {r['file_kb']:>8.0f} "
              f"{s['MASE']:>8.4f} {s['MAE']:>12.4f} {s['RMSE']:>10.4f} "
              f"{r['dl_time']:>6.1f} {r['infer_time']:>6.2f} {r['est_tokens']:>9,}")

    used, budget = db.get_budget()
    all_metrics = db.get_metrics()
    downloaded = db.get_downloaded_datasets()

    print(f"\n--- Token Usage Estimate ---")
    print(f"Data processing tokens (this batch):  ~{total_tokens_est:,}")
    print(f"Claude API overhead (orchestration):  ~5,000")
    print(f"Estimated total for this run:         ~{total_tokens_est + 5000:,}")
    print(f"Per-dataset average:                  ~{(total_tokens_est + 5000) // 5:,}")

    print(f"\n--- Database State ---")
    print(f"Budget:      {used / 1024:.1f} KB / {budget / 1e9:.0f} GB")
    print(f"Metrics:     {len(all_metrics)} rows")
    print(f"Downloaded:  {len(downloaded)} datasets")

    print(f"\n{'=' * 70}")
    print("BATCH 2 COMPLETE")


if __name__ == "__main__":
    main()
