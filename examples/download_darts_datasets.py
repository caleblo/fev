#!/usr/bin/env python3
"""Download all 27 Darts built-in datasets and run SeasonalNaive evaluation."""
import sys, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fev
import datasets as hf_datasets
from catalog import CatalogDB, MetricsWriter, DartsDatasetLoader
from catalog.darts_loader import DARTS_REGISTRY


def seasonal_naive(past_data, target_col, horizon, seasonality):
    preds = []
    for j in range(len(past_data)):
        ts = past_data[j][target_col]
        clean = [v for v in ts if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))]
        if not clean: clean = [0.0]
        s = max(seasonality, 1)
        pat = clean[-s:] if len(clean) >= s else clean
        preds.append((pat * ((horizon // len(pat)) + 1))[:horizon])
    return preds


def main():
    db = CatalogDB()
    writer = MetricsWriter(db)
    total = len(DARTS_REGISTRY)

    print(f"{'='*70}")
    print(f"DOWNLOADING ALL {total} DARTS BUILT-IN DATASETS")
    print(f"{'='*70}")

    results = []
    for i, meta in enumerate(DARTS_REGISTRY, 1):
        class_name = meta["class"]
        uid = meta.get("uid") or class_name
        horizon = meta["horizon"]
        seasonality = meta["seasonality"]

        # Skip if already done
        row = db.get_dataset_by_uid(uid) if meta.get("uid") else None
        dataset_id = row["id"] if row else None
        if dataset_id and db.get_download_status(dataset_id) == "success":
            print(f"[{i:2d}/{total}] SKIP {uid:12s} {class_name}")
            results.append({"class_name": class_name, "uid": uid, "status": "skipped"})
            continue

        print(f"\n[{i:2d}/{total}] {uid:12s} {class_name}")
        loader = DartsDatasetLoader(class_name, unified_id=uid)

        try:
            # Download + save
            path = loader.download()
            file_size = path.stat().st_size
            print(f"  Saved: {path.name} ({file_size/1024:.0f} KB)")

            if dataset_id:
                db.upsert_download(dataset_id, status="success",
                                   local_path=str(path), file_size_bytes=file_size,
                                   download_url=f"darts://{class_name}", url_type="darts")
                db.add_to_budget(file_size)
                db.set_dataloader_created(dataset_id, 1)

            # Evaluate via fev
            task = loader.to_fev_task()
            t0 = time.time()
            preds_per_window = []
            for window in task.iter_windows():
                past_data, _ = window.get_input_data()
                p = seasonal_naive(past_data, task.target_columns[0], horizon, seasonality)
                pred_ds = hf_datasets.Dataset.from_dict({"predictions": p})
                preds_per_window.append(
                    hf_datasets.DatasetDict({task.target_columns[0]: pred_ds})
                )

            summary = task.evaluation_summary(
                preds_per_window, model_name="SeasonalNaive",
                inference_time_s=time.time()-t0, trained_on_this_dataset=False,
            )
            print(f"  MASE={summary['MASE']:.4f}  MAE={summary['MAE']:.4f}  RMSE={summary['RMSE']:.4f}")

            if dataset_id:
                writer.write_summary(summary, dataset_id=dataset_id)

            results.append({"class_name": class_name, "uid": uid, "status": "success",
                             "file_kb": file_size/1024, "mase": summary["MASE"]})
        except Exception as e:
            print(f"  FAILED: {str(e).encode('ascii', errors='replace').decode('ascii')}")
            if dataset_id:
                db.upsert_download(dataset_id, status="failed", error_message=str(e)[:200])
            results.append({"class_name": class_name, "uid": uid, "status": "failed", "error": str(e)[:80]})

    # Summary
    ok   = sum(1 for r in results if r["status"] == "success")
    skip = sum(1 for r in results if r["status"] == "skipped")
    fail = sum(1 for r in results if r["status"] == "failed")
    used, budget = db.get_budget()

    print(f"\n{'='*70}")
    print(f"Darts: {ok} ok / {skip} skip / {fail} fail")
    print(f"Budget: {used/1024/1024:.0f} MB | Metrics: {len(db.get_metrics())} rows")
    print(f"{'='*70}")
    if fail:
        for r in results:
            if r["status"] == "failed":
                print(f"  FAIL {r['uid']:12s} {r['class_name']:40s} {r.get('error','')[:50]}")


if __name__ == "__main__":
    main()
