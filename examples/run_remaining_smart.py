#!/usr/bin/env python3
"""Smart retry: skip giant/problematic datasets, handle non-standard columns."""
import sys
import os
import math
import time
from pathlib import Path

# Redirect HF cache to E: drive
os.environ["HF_HOME"] = r"E:\datasets\_hf_cache"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fev
import datasets as hf_datasets
from catalog import CatalogDB, MetricsWriter

# Skip these configs: too large (>5GB generated), synthetic, or non-standard schema
SKIP_CONFIGS = {
    "training_corpus_tsmixup_10m",   # 10M series, synthetic, massive
    "training_corpus_kernel_synth_1m", # 1M synthetic
    "weatherbench_hourly",  # not a real config (has sub-configs)
    "wiki_daily_100k",      # 4 GB generated
    "monash_london_smart_meters",  # 2.5 GB generated
}

# Non-standard column datasets: specify target column
CUSTOM_TARGETS = {
    "solar": "power_mw",
    "solar_1h": "power_mw",
    "ushcn_daily": "TMAX",  # Use max temp as target
}

FREQ_MAP = {
    "15min": (96, 96), "30min": (48, 48), "hourly": (24, 24), "1h": (24, 24),
    "daily": (14, 7), "weekly": (8, 1), "monthly": (12, 12),
    "quarterly": (8, 4), "yearly": (1, 1),
}


def guess_horizon_season(config_name, title):
    text = f"{config_name} {title}".lower()
    for key, (h, s) in FREQ_MAP.items():
        if key in text:
            return h, s
    return 12, 1


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
    print(f"\n{'=' * 85}")
    print(f"REPORT {batch_num} | {len(downloaded)} downloaded | {len(all_metrics)} metrics | {used/1024/1024:.0f} MB used")
    print(f"{'=' * 85}")
    print(f"{'UID':<12} {'Config':<35} {'Series':>7} {'MB':>7} {'MASE':>8} {'Status':<8}")
    print("-" * 85)
    for r in batch_results:
        s = r.get("summary")
        if s:
            print(f"{r['uid']:<12} {r['config'][:35]:<35} {r['n_series']:>7} "
                  f"{r['file_mb']:>7.1f} {s['MASE']:>8.4f} OK")
        else:
            print(f"{r['uid']:<12} {r['config'][:35]:<35} {'':>7} {'':>7} {'':>8} "
                  f"FAIL: {r.get('error', '?')[:35]}")
    print(f"{'=' * 85}\n")


def main():
    db = CatalogDB()
    db.run_migrations()
    writer = MetricsWriter(db)

    # Reset failed to pending
    import sqlite3
    conn = sqlite3.connect(str(db._path))
    conn.execute("UPDATE dataset_downloads SET status='pending' WHERE status='failed'")
    conn.commit()
    conn.close()

    downloaded_ids = {d['id'] for d in db.get_downloaded_datasets()}
    eligible = db.get_eligible_datasets()

    candidates = []
    for ds in eligible:
        if ds['id'] in downloaded_ids:
            continue
        url = ds['homepage'] or ''
        if 'autogluon/chronos_datasets' in url and 'tree/main/' in url:
            config = url.split('tree/main/')[-1].rstrip('/')
            if config in SKIP_CONFIGS:
                print(f"  SKIP {ds['unified_id']} -- {config} (blocklisted)")
                db.upsert_download(ds['id'], status="skipped",
                                   error_message=f"blocklisted: {config}")
                continue
            candidates.append({
                'uid': ds['unified_id'], 'id': ds['id'],
                'title': ds['title'], 'config': config,
            })

    print(f"{'=' * 85}")
    print(f"SMART RETRY: {len(candidates)} datasets | {len(downloaded_ids)} already done")
    print(f"{'=' * 85}")

    batch_results = []
    batch_num = 0
    total_done = 0

    for i, cand in enumerate(candidates, 1):
        uid, config, dataset_id = cand['uid'], cand['config'], cand['id']
        title = cand['title']
        print(f"\n[{i}/{len(candidates)}] {uid} -- {config}")

        try:
            # Download
            t0 = time.time()
            hf_ds = hf_datasets.load_dataset(
                "autogluon/chronos_datasets", config, split="train", num_proc=1
            )
            dl_time = time.time() - t0
            n_series = len(hf_ds)

            save_dir = Path(r"E:\datasets") / uid
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{config}.parquet"
            hf_ds.to_parquet(str(save_path))
            file_size = save_path.stat().st_size
            print(f"  DL: {n_series} series, {file_size/1024/1024:.1f} MB ({dl_time:.1f}s)")

            db.upsert_download(
                dataset_id, status="success", local_path=str(save_path),
                file_size_bytes=file_size,
                download_url=f"hf://autogluon/chronos_datasets/{config}",
                url_type="huggingface",
            )
            db.add_to_budget(file_size)
            db.set_dataloader_created(dataset_id, 1)

            # Task — handle custom target columns
            horizon, seasonality = guess_horizon_season(config, title)
            target_col = CUSTOM_TARGETS.get(config, "target")

            task = fev.Task(
                dataset_path="autogluon/chronos_datasets",
                dataset_config=config,
                horizon=horizon,
                seasonality=seasonality,
                eval_metric="MASE",
                extra_metrics=["MAE", "MSE", "RMSE"],
                target=target_col,
                task_name=uid,
            )

            # Predict
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

            # Evaluate
            summary = task.evaluation_summary(
                preds_per_window, model_name="SeasonalNaive",
                inference_time_s=infer_time, trained_on_this_dataset=False,
            )
            print(f"  MASE={summary['MASE']:.4f}  MAE={summary['MAE']:.4f}  RMSE={summary['RMSE']:.4f}")

            n = writer.write_summary(summary, dataset_id=dataset_id)
            print(f"  Wrote {n} metrics")

            batch_results.append({
                'uid': uid, 'config': config, 'n_series': n_series,
                'file_mb': file_size / 1024 / 1024, 'summary': summary,
            })

        except Exception as e:
            err = str(e)[:200]
            print(f"  FAILED: {err[:80]}")
            db.upsert_download(dataset_id, status="failed", error_message=err)
            batch_results.append({'uid': uid, 'config': config, 'error': err[:80]})

        total_done += 1
        if total_done % 10 == 0 or i == len(candidates):
            batch_num += 1
            print_report(batch_results, batch_num, db)
            batch_results = []

    # Final
    used, budget = db.get_budget()
    downloaded = db.get_downloaded_datasets()
    all_metrics = db.get_metrics()
    failed = []
    conn = sqlite3.connect(str(db._path))
    for row in conn.execute("""
        SELECT d.unified_id, d.title, dl.error_message
        FROM dataset_downloads dl JOIN datasets d ON d.id=dl.dataset_id
        WHERE dl.status='failed'
    """):
        failed.append(row)
    conn.close()

    print(f"\n{'=' * 85}")
    print("FINAL SUMMARY")
    print(f"{'=' * 85}")
    print(f"Downloaded:  {len(downloaded)} datasets")
    print(f"Metrics:     {len(all_metrics)} rows")
    print(f"Budget:      {used/1024/1024:.0f} MB / {budget/1e9:.0f} GB")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for uid, title, err in failed:
            print(f"  {uid:12s} {title[:40]:40s} {(err or '')[:50]}")
    print(f"{'=' * 85}")


if __name__ == "__main__":
    main()
