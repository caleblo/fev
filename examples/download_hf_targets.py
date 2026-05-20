#!/usr/bin/env python3
"""
Download all 59 explicitly requested HF datasets and run SeasonalNaive evaluation.

Sources:
  - autogluon/chronos_datasets: ushcn_daily, weatherbench_daily, wiki_daily_100k
  - autogluon/fev_datasets: 56 configs (ETT_15T, boomlet_*, epf_*, ercot_*, etc.)

Rules:
  - fev_eval and gift_eval datasets are EXCLUDED from the auto-download pipeline
    but these 59 are downloaded explicitly as requested
  - Files saved to E:\\datasets\\{uid}\\{config}.parquet (never altered)
  - HF cache at E:\\datasets\\_hf_cache (not C:)
  - Reports progress every 10 datasets
"""
import sys
import math
import time
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", r"E:\datasets\_hf_cache")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fev
import datasets as hf_datasets
from catalog import CatalogDB, MetricsWriter, HFDatasetLoader
from catalog.hf_loader import EXPLICIT_HF_TARGETS, _infer_horizon_season, HF_TARGET_OVERRIDES


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


def print_report(batch_results, batch_num, db):
    used, budget = db.get_budget()
    all_metrics = db.get_metrics()
    downloaded = db.get_downloaded_datasets()
    ok    = sum(1 for r in batch_results if r.get("status") == "success")
    fail  = sum(1 for r in batch_results if r.get("status") == "failed")
    skip  = sum(1 for r in batch_results if r.get("status") in ("skipped", "eval_skip"))

    print(f"\n{'='*80}")
    print(f"REPORT {batch_num} | {len(downloaded)} total downloaded | "
          f"{len(all_metrics)} metric rows | {used/1024/1024:.0f} MB used")
    print(f"{'='*80}")
    print(f"{'UID':<12} {'Config':<30} {'Series':>7} {'MB':>7} {'MASE':>8} {'Status'}")
    print("-"*72)
    for r in batch_results:
        s = r.get("summary")
        status = r.get("status", "?")
        if s:
            print(f"{r['uid']:<12} {r['config']:<30} {r.get('n_series',0):>7} "
                  f"{r.get('file_mb',0):>7.1f} {s['MASE']:>8.4f} {status}")
        else:
            err = r.get("error", "")[:30]
            print(f"{r['uid']:<12} {r['config']:<30} {'':>7} {'':>7} {'':>8} "
                  f"{status}: {err}")
    print(f"  Batch: {ok} ok / {fail} fail / {skip} skip")
    print(f"{'='*80}\n")


def main():
    db = CatalogDB()
    db.run_migrations()
    writer = MetricsWriter(db)

    total = len(EXPLICIT_HF_TARGETS)
    print(f"{'='*80}")
    print(f"DOWNLOADING {total} EXPLICITLY REQUESTED HF DATASETS")
    print(f"Excluded from auto-pipeline: fev_eval, gift_eval")
    print(f"HF cache: {os.environ.get('HF_HOME')}")
    print(f"{'='*80}")

    batch_results = []
    batch_num = 0
    total_done = 0

    for i, (repo, config_name, uid) in enumerate(EXPLICIT_HF_TARGETS, 1):
        print(f"\n[{i}/{total}] {uid} -- {repo.split('/')[-1]}/{config_name}")

        row = db.get_dataset_by_uid(uid)
        dataset_id = row["id"] if row else None

        # Skip if already downloaded
        if dataset_id and db.get_download_status(dataset_id) == "success":
            existing_path = db.get_downloaded_datasets()
            print(f"  SKIP: already downloaded")
            batch_results.append({"uid": uid, "config": config_name, "status": "skipped"})
            total_done += 1
            if total_done % 10 == 0 or i == total:
                batch_num += 1
                print_report(batch_results, batch_num, db)
                batch_results = []
            continue

        loader = HFDatasetLoader(repo, config_name, unified_id=uid)

        try:
            # 1. Download
            path = loader.download()
            file_size = path.stat().st_size
            file_mb = file_size / 1024 / 1024

            # Update catalog.db
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

            # 2. Evaluate with SeasonalNaive
            overrides = HF_TARGET_OVERRIDES.get(config_name, {})
            horizon, seasonality = _infer_horizon_season(config_name)
            horizon    = overrides.get("horizon",    horizon)
            seasonality = overrides.get("seasonality", seasonality)
            target_col  = overrides.get("target",     "target")
            max_series  = overrides.get("max_series")

            # Load eval copy (truncated if needed)
            eval_ds = loader.load_hf(for_eval=True)
            n_series = len(eval_ds)
            print(f"  Series: {n_series:,}  MB: {file_mb:.1f}  "
                  f"h={horizon} s={seasonality} target={target_col}")

            # Save eval parquet
            cache_dir = loader.save_dir.parent / "_cache" / uid
            cache_dir.mkdir(parents=True, exist_ok=True)
            eval_path = cache_dir / "data.parquet"
            eval_ds.to_parquet(str(eval_path))

            # Create fev.Task
            task_params = dict(
                dataset_path=str(eval_path),
                horizon=horizon,
                seasonality=seasonality,
                eval_metric="MASE",
                extra_metrics=["MAE", "MSE", "RMSE"],
                task_name=uid,
            )
            if target_col != "target":
                task_params["target"] = target_col

            task = fev.Task(**task_params)

            # Predict
            t0 = time.time()
            preds_per_window = []
            for window in task.iter_windows():
                past_data, _ = window.get_input_data()
                p = seasonal_naive(past_data, task.target_columns[0], horizon, seasonality)
                pred_ds = hf_datasets.Dataset.from_dict({"predictions": p})
                preds_per_window.append(
                    hf_datasets.DatasetDict({task.target_columns[0]: pred_ds})
                )
            infer_time = time.time() - t0

            summary = task.evaluation_summary(
                preds_per_window, model_name="SeasonalNaive",
                inference_time_s=infer_time, trained_on_this_dataset=False,
            )
            print(f"  MASE={summary['MASE']:.4f}  MAE={summary['MAE']:.4f}  "
                  f"RMSE={summary['RMSE']:.4f}")

            if dataset_id:
                writer.write_summary(summary, dataset_id=dataset_id)

            batch_results.append({
                "uid": uid, "config": config_name, "status": "success",
                "n_series": n_series, "file_mb": file_mb, "summary": summary,
            })

        except Exception as e:
            err = str(e)[:200]
            print(f"  FAILED: {err[:80]}")
            if dataset_id:
                db.upsert_download(dataset_id, status="failed", error_message=err)
            batch_results.append({"uid": uid, "config": config_name,
                                   "status": "failed", "error": err[:80]})

        total_done += 1
        if total_done % 10 == 0 or i == total:
            batch_num += 1
            print_report(batch_results, batch_num, db)
            batch_results = []

    # Final summary
    used, budget = db.get_budget()
    all_metrics = db.get_metrics()
    downloaded = db.get_downloaded_datasets()

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Downloaded:  {len(downloaded)} datasets")
    print(f"Metrics:     {len(all_metrics)} rows")
    print(f"Budget:      {used/1024/1024:.0f} MB / {budget/1e9:.0f} GB")

    # Show failures
    import sqlite3
    conn = sqlite3.connect(str(db._path))
    fails = conn.execute("""
        SELECT d.unified_id, d.title, dl.error_message
        FROM dataset_downloads dl JOIN datasets d ON d.id = dl.dataset_id
        WHERE dl.status = 'failed'
          AND d.unified_id IN ({})
    """.format(",".join(f"'{uid}'" for _, _, uid in EXPLICIT_HF_TARGETS))).fetchall()
    conn.close()

    if fails:
        print(f"\nFailed ({len(fails)}):")
        for uid, title, err in fails:
            print(f"  {uid:10s} {title[:40]:40s} {(err or '')[:50]}")
    else:
        print("\nNo failures!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
