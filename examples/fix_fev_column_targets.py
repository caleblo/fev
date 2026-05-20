#!/usr/bin/env python3
"""Fix 29 failed fev_datasets by using correct non-standard target columns."""
import sys, os, math, time
from pathlib import Path
os.environ.setdefault("HF_HOME", r"E:\datasets\_hf_cache")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fev
import datasets as hf_datasets
from catalog import CatalogDB, MetricsWriter
from catalog.hf_loader import HF_TARGET_OVERRIDES, _infer_horizon_season

# All 29 failed configs from the previous run
FAILED_CONFIGS = [
    ("autogluon/fev_datasets", "ETT_15T",                 "FEV-001"),
    ("autogluon/fev_datasets", "bizitobs_l2c_1H",         "FEV-013"),
    ("autogluon/fev_datasets", "boomlet_1062",            "FEV-015"),
    ("autogluon/fev_datasets", "boomlet_1209",            "FEV-016"),
    ("autogluon/fev_datasets", "boomlet_1225",            "FEV-017"),
    ("autogluon/fev_datasets", "boomlet_1230",            "FEV-018"),
    ("autogluon/fev_datasets", "boomlet_1282",            "FEV-019"),
    ("autogluon/fev_datasets", "boomlet_1487",            "FEV-020"),
    ("autogluon/fev_datasets", "boomlet_1631",            "FEV-021"),
    ("autogluon/fev_datasets", "boomlet_1676",            "FEV-022"),
    ("autogluon/fev_datasets", "boomlet_1855",            "FEV-023"),
    ("autogluon/fev_datasets", "boomlet_1975",            "FEV-024"),
    ("autogluon/fev_datasets", "boomlet_2187",            "FEV-025"),
    ("autogluon/fev_datasets", "boomlet_285",             "FEV-026"),
    ("autogluon/fev_datasets", "boomlet_619",             "FEV-027"),
    ("autogluon/fev_datasets", "boomlet_772",             "FEV-028"),
    ("autogluon/fev_datasets", "boomlet_963",             "FEV-029"),
    ("autogluon/fev_datasets", "favorita_stores_1D",      "FEV-043"),
    ("autogluon/fev_datasets", "favorita_transactions_1D","FEV-046"),
    ("autogluon/fev_datasets", "fred_md_2025",            "FEV-049"),
    ("autogluon/fev_datasets", "fred_qd_2025",            "FEV-051"),
    ("autogluon/fev_datasets", "gvar",                    "FEV-053"),
    ("autogluon/fev_datasets", "kdd_cup_2022_10T",        "FEV-063"),
    ("autogluon/fev_datasets", "rohlik_orders_1D",        "FEV-076"),
    ("autogluon/fev_datasets", "rohlik_sales_1D",         "FEV-078"),
    ("autogluon/fev_datasets", "rossmann_1D",             "FEV-080"),
    ("autogluon/fev_datasets", "uci_air_quality_1D",      "FEV-086"),
    ("autogluon/fev_datasets", "uk_covid_nation_1D",      "FEV-088"),
    ("autogluon/fev_datasets", "uk_covid_utla_1D",        "FEV-092"),
]


def seasonal_naive(past_data, target_col, horizon, seasonality):
    preds = []
    for j in range(len(past_data)):
        ts = past_data[j][target_col]
        clean = [v for v in ts if v is not None and not math.isnan(v) and not math.isinf(v)]
        if not clean:
            clean = [0.0]
        s = max(seasonality, 1)
        pat = clean[-s:] if len(clean) >= s else clean
        forecast = (pat * ((horizon // len(pat)) + 1))[:horizon]
        preds.append(forecast)
    return preds


def main():
    db = CatalogDB()
    writer = MetricsWriter(db)
    total = len(FAILED_CONFIGS)

    print(f"{'='*70}")
    print(f"FIXING {total} FAILED fev_datasets (non-standard target columns)")
    print(f"{'='*70}")

    results = []
    for i, (repo, config, uid) in enumerate(FAILED_CONFIGS, 1):
        row = db.get_dataset_by_uid(uid)
        dataset_id = row["id"] if row else None
        overrides = HF_TARGET_OVERRIDES.get(config, {})
        target = overrides.get("target", "target")
        horizon, seasonality = _infer_horizon_season(config)
        horizon    = overrides.get("horizon", horizon)
        seasonality = overrides.get("seasonality", seasonality)
        max_series = overrides.get("max_series")

        print(f"\n[{i}/{total}] {uid} {config} -> target='{target}'")

        try:
            # Load from saved parquet (already downloaded)
            save_path = Path(r"E:\datasets") / uid / f"{config}.parquet"
            if not save_path.exists():
                print(f"  Re-downloading...")
                ds = hf_datasets.load_dataset(repo, config, split="train", num_proc=1)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                ds.to_parquet(str(save_path))

            eval_ds = hf_datasets.Dataset.from_parquet(str(save_path))
            if max_series and len(eval_ds) > max_series:
                eval_ds = eval_ds.select(range(max_series))

            n_series = len(eval_ds)
            print(f"  {n_series} series, target='{target}', h={horizon}, s={seasonality}")

            # Save eval copy
            cache_dir = Path(r"E:\datasets") / "_cache" / uid
            cache_dir.mkdir(parents=True, exist_ok=True)
            eval_path = cache_dir / "data.parquet"
            eval_ds.to_parquet(str(eval_path))

            task_params = dict(
                dataset_path=str(eval_path),
                horizon=horizon, seasonality=seasonality,
                eval_metric="MASE", extra_metrics=["MAE", "MSE", "RMSE"],
                task_name=uid,
            )
            if target != "target":
                task_params["target"] = target

            task = fev.Task(**task_params)
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
                # Update download status to success
                file_size = save_path.stat().st_size
                db.upsert_download(dataset_id, status="success",
                                   local_path=str(save_path), file_size_bytes=file_size,
                                   download_url=f"hf://{repo}/{config}", url_type="huggingface")
                db.set_dataloader_created(dataset_id, 1)
                writer.write_summary(summary, dataset_id=dataset_id)
                print(f"  DB updated")

            results.append({"uid": uid, "config": config, "status": "success",
                             "mase": summary["MASE"]})
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"uid": uid, "config": config, "status": "failed", "error": str(e)[:80]})

    ok   = sum(1 for r in results if r["status"] == "success")
    fail = sum(1 for r in results if r["status"] == "failed")
    print(f"\n{'='*70}")
    print(f"Fixed: {ok}/{total} ok, {fail} failed")

    used, budget = db.get_budget()
    all_metrics = db.get_metrics()
    downloaded = db.get_downloaded_datasets()
    print(f"Total downloaded: {len(downloaded)} | Metrics: {len(all_metrics)} | Budget: {used/1024/1024:.0f} MB")

    if fail:
        print("\nStill failing:")
        for r in results:
            if r["status"] == "failed":
                print(f"  {r['uid']:10s} {r['config']:30s} {r.get('error','')[:60]}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
