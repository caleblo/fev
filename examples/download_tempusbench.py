#!/usr/bin/env python3
"""Download all 73 TempusBench tasks from GitHub and run SeasonalNaive evaluation."""
import sys
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fev
import datasets as hf_datasets
from catalog import CatalogDB, MetricsWriter, TempusBenchLoader
from catalog.tempusbench_loader import TEMPUSBENCH_REGISTRY


def seasonal_naive(past_data, target_col, horizon, seasonality):
    preds = []
    for j in range(len(past_data)):
        ts = past_data[j][target_col]
        clean = [v for v in ts if v is not None and not math.isnan(v) and not math.isinf(v)]
        if not clean:
            clean = [0.0]
        s = max(seasonality, 1)
        pat = clean[-s:] if len(clean) >= s else clean
        preds.append((pat * ((horizon // len(pat)) + 1))[:horizon])
    return preds


def main():
    db = CatalogDB()
    db.run_migrations()
    writer = MetricsWriter(db)
    total = len(TEMPUSBENCH_REGISTRY)

    print(f"{'='*75}")
    print(f"TEMPUSBENCH: {total} tasks from github.com/Smlcrm/TempusBench")
    print(f"{'='*75}")

    results = []
    batch = []

    for i, entry in enumerate(TEMPUSBENCH_REGISTRY, 1):
        uid = entry["uid"]
        task_dir = entry["task_dir"]
        task_type = entry["task_type"]

        row = db.get_dataset_by_uid(uid)
        dataset_id = row["id"] if row else None

        # Skip if already downloaded and evaluated
        if dataset_id and db.get_download_status(dataset_id) == "success":
            print(f"[{i:2d}/{total}] SKIP {uid:10s} {task_dir}")
            results.append({"uid": uid, "task_dir": task_dir, "status": "skipped"})
            batch.append(results[-1])
            if len(batch) >= 10 or i == total:
                _report(batch, i, total, db)
                batch = []
            continue

        print(f"\n[{i:2d}/{total}] {uid} -- {task_type}/{task_dir}")
        loader = TempusBenchLoader(task_dir, task_type, unified_id=uid)

        try:
            # 1. Download CSV from GitHub
            path = loader.download()
            file_size = path.stat().st_size

            if dataset_id:
                db.upsert_download(
                    dataset_id, status="success",
                    local_path=str(path), file_size_bytes=file_size,
                    download_url=loader.csv_url, url_type="github_raw",
                )
                db.add_to_budget(file_size)
                db.set_dataloader_created(dataset_id, 1)

            # 2. Build fev.Task
            horizon = loader.horizon
            seasonality = loader.seasonality
            print(f"  h={horizon} s={seasonality} type={task_type}")

            task = loader.to_fev_task()

            # 3. SeasonalNaive
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

            results.append({"uid": uid, "task_dir": task_dir, "status": "success",
                             "file_kb": file_size / 1024, "summary": summary})

        except Exception as e:
            err = str(e).encode("ascii", errors="replace").decode()[:100]
            print(f"  FAILED: {err}")
            if dataset_id:
                db.upsert_download(dataset_id, status="failed", error_message=str(e)[:200])
            results.append({"uid": uid, "task_dir": task_dir, "status": "failed", "error": err})

        batch.append(results[-1])
        time.sleep(0.3)  # gentle GitHub rate limiting

        if len(batch) >= 10 or i == total:
            _report(batch, i, total, db)
            batch = []

    # Final
    ok   = sum(1 for r in results if r["status"] == "success")
    fail = sum(1 for r in results if r["status"] == "failed")
    skip = sum(1 for r in results if r["status"] == "skipped")
    used, budget = db.get_budget()

    print(f"\n{'='*75}")
    print(f"FINAL: {ok} ok / {fail} fail / {skip} skip")
    print(f"Budget: {used/1024/1024:.0f} MB | Metrics: {len(db.get_metrics())} rows")
    if fail:
        print("\nFailed:")
        for r in results:
            if r["status"] == "failed":
                print(f"  {r['uid']:10s} {r['task_dir']:40s} {r.get('error','')[:50]}")
    print(f"{'='*75}")


def _report(batch, i, total, db):
    ok   = sum(1 for r in batch if r["status"] == "success")
    fail = sum(1 for r in batch if r["status"] == "failed")
    skip = sum(1 for r in batch if r["status"] == "skipped")
    used, budget = db.get_budget()
    downloaded = db.get_downloaded_datasets()

    print(f"\n{'='*75}")
    print(f"REPORT {i}/{total} | {len(downloaded)} total DL | "
          f"{len(db.get_metrics())} metrics | {used/1024/1024:.0f} MB")
    print(f"{'='*75}")
    print(f"{'UID':<12} {'Task dir':<40} {'KB':>7} {'MASE':>8} {'Status'}")
    print("-"*75)
    for r in batch:
        s = r.get("summary")
        if s:
            print(f"{r['uid']:<12} {r['task_dir']:<40} {r.get('file_kb',0):>7.0f} "
                  f"{s['MASE']:>8.4f} OK")
        else:
            print(f"{r['uid']:<12} {r['task_dir']:<40} {'':>7} {'':>8} "
                  f"{r['status'].upper()}: {r.get('error','')[:25]}")
    print(f"  Batch: {ok} ok / {fail} fail / {skip} skip\n")


if __name__ == "__main__":
    main()
