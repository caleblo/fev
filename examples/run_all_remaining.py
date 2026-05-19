#!/usr/bin/env python3
"""Download and evaluate ALL remaining HF chronos datasets. Report every 10."""
import sys
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fev
import datasets as hf_datasets
from catalog import CatalogDB, MetricsWriter

# Frequency heuristics for horizon/seasonality
FREQ_MAP = {
    "15min": (96, 96),
    "30min": (48, 48),
    "hourly": (24, 24),
    "1h": (24, 24),
    "daily": (14, 7),
    "weekly": (8, 1),
    "monthly": (12, 12),
    "quarterly": (8, 4),
    "yearly": (1, 1),
}

def guess_horizon_season(config_name, title):
    """Guess horizon and seasonality from config/title."""
    text = f"{config_name} {title}".lower()
    for key, (h, s) in FREQ_MAP.items():
        if key in text:
            return h, s
    # Default
    return 12, 1


def seasonal_naive(past_data, task, horizon, seasonality):
    preds = []
    for j in range(len(past_data)):
        ts = past_data[j][task.target_columns[0]]
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
    """Print a report for the last batch of datasets."""
    used, budget = db.get_budget()
    all_metrics = db.get_metrics()
    downloaded = db.get_downloaded_datasets()

    print(f"\n{'=' * 80}")
    print(f"PROGRESS REPORT — After batch {batch_num} ({len(downloaded)} total downloaded)")
    print(f"{'=' * 80}")
    print(f"\n{'UID':<12} {'Dataset':<35} {'Series':>7} {'KB':>8} {'MASE':>8} {'MAE':>12} {'Status':<8}")
    print("-" * 96)
    for r in batch_results:
        s = r.get("summary")
        if s:
            print(f"{r['uid']:<12} {r['config'][:35]:<35} {r['n_series']:>7} "
                  f"{r['file_kb']:>8.0f} {s['MASE']:>8.4f} {s['MAE']:>12.4f} OK")
        else:
            print(f"{r['uid']:<12} {r['config'][:35]:<35} {'':>7} {'':>8} {'':>8} "
                  f"{'':>12} FAIL: {r.get('error','?')[:30]}")

    print(f"\nBudget:     {used / 1024 / 1024:.1f} MB / {budget / 1e9:.0f} GB")
    print(f"Metrics:    {len(all_metrics)} rows")
    print(f"Downloaded: {len(downloaded)} datasets")
    print(f"{'=' * 80}\n")


def main():
    db = CatalogDB()
    db.run_migrations()
    writer = MetricsWriter(db)

    # Find all remaining HF chronos configs
    downloaded_ids = {d['id'] for d in db.get_downloaded_datasets()}
    eligible = db.get_eligible_datasets()

    candidates = []
    for ds in eligible:
        if ds['id'] in downloaded_ids:
            continue
        url = ds['homepage'] or ''
        if 'autogluon/chronos_datasets' in url and 'tree/main/' in url:
            config = url.split('tree/main/')[-1].rstrip('/')
            candidates.append({
                'uid': ds['unified_id'],
                'id': ds['id'],
                'title': ds['title'],
                'config': config,
            })

    print(f"{'=' * 80}")
    print(f"DOWNLOADING ALL REMAINING: {len(candidates)} datasets")
    print(f"Already downloaded: {len(downloaded_ids)}")
    print(f"{'=' * 80}")

    batch_results = []
    batch_num = 0
    total_done = 0

    for i, cand in enumerate(candidates, 1):
        uid = cand['uid']
        config = cand['config']
        dataset_id = cand['id']
        title = cand['title']

        print(f"\n[{i}/{len(candidates)}] {uid} -- {config}")

        try:
            # 1. Download
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
            print(f"  DL: {n_series} series, {file_size/1024:.0f} KB ({dl_time:.1f}s)")

            # Update catalog.db
            db.upsert_download(
                dataset_id, status="success",
                local_path=str(save_path), file_size_bytes=file_size,
                download_url=f"hf://autogluon/chronos_datasets/{config}",
                url_type="huggingface",
            )
            db.add_to_budget(file_size)
            db.set_dataloader_created(dataset_id, 1)

            # 2. Create Task
            horizon, seasonality = guess_horizon_season(config, title)
            task = fev.Task(
                dataset_path="autogluon/chronos_datasets",
                dataset_config=config,
                horizon=horizon,
                seasonality=seasonality,
                eval_metric="MASE",
                extra_metrics=["MAE", "MSE", "RMSE"],
                task_name=uid,
            )

            # 3. Predict
            preds_per_window = []
            t0 = time.time()
            for window in task.iter_windows():
                past_data, future_data = window.get_input_data()
                p = seasonal_naive(past_data, task, horizon, seasonality)
                pred_ds = hf_datasets.Dataset.from_dict({"predictions": p})
                preds_per_window.append(
                    hf_datasets.DatasetDict({task.target_columns[0]: pred_ds})
                )
            infer_time = time.time() - t0

            # 4. Evaluate
            summary = task.evaluation_summary(
                preds_per_window, model_name="SeasonalNaive",
                inference_time_s=infer_time, trained_on_this_dataset=False,
            )
            print(f"  MASE={summary['MASE']:.4f}  MAE={summary['MAE']:.4f}  RMSE={summary['RMSE']:.4f}")

            # 5. Write metrics
            n = writer.write_summary(summary, dataset_id=dataset_id)
            print(f"  Wrote {n} metrics")

            batch_results.append({
                'uid': uid, 'config': config, 'n_series': n_series,
                'file_kb': file_size / 1024, 'summary': summary,
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            db.upsert_download(dataset_id, status="failed", error_message=str(e)[:200])
            batch_results.append({
                'uid': uid, 'config': config, 'error': str(e)[:80],
            })

        total_done += 1

        # Report every 10
        if total_done % 10 == 0 or i == len(candidates):
            batch_num += 1
            print_report(batch_results, batch_num, db)
            batch_results = []

    # Final summary
    used, budget = db.get_budget()
    all_metrics = db.get_metrics()
    downloaded = db.get_downloaded_datasets()

    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total downloaded:  {len(downloaded)}")
    print(f"Total metrics:     {len(all_metrics)} rows")
    print(f"Budget used:       {used / 1024 / 1024:.1f} MB / {budget / 1e9:.0f} GB")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
