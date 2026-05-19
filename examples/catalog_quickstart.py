#!/usr/bin/env python3
"""
Catalog Quickstart — single DB architecture.

- catalog.db = single source of truth (metadata + downloads + metrics + loaders)
- E:\\datasets\\ = downloaded files (NEVER altered — dataloaders handle transformation)

Metrics flow:
    fev.Task → model.fit_predict → task.evaluation_summary → MetricsWriter → catalog.db
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import CatalogDB, CatalogBenchmark, DatasetDownloader, MetricsWriter


def main():
    print("=" * 60)

    # 1. Connect
    db = CatalogDB()
    db.run_migrations()
    print("1. Connected to catalog.db")

    # 2. Show eligible datasets
    eligible = db.get_eligible_datasets()
    tiers = {}
    for ds in eligible:
        t = ds["priority_tier"]
        tiers[t] = tiers.get(t, 0) + 1
    print(f"\n2. Eligible: {len(eligible)} datasets")
    for t in sorted(tiers):
        print(f"   Tier {t}: {tiers[t]}")

    # 3. Dry-run download
    print("\n3. Download dry run (3 datasets)...")
    dl = DatasetDownloader(db)
    dl.download_all(tier=1, dry_run=True, max_datasets=3)

    # 4. Budget
    used, budget = db.get_budget()
    print(f"\n4. Budget: {used/1e9:.2f} GB / {budget/1e9:.0f} GB")

    # 5. Check metrics
    writer = MetricsWriter(db)
    df = writer.read_metrics()
    print(f"\n5. Metrics: {len(df)} rows")

    print("\n" + "=" * 60 + "\nDone!")


if __name__ == "__main__":
    main()
