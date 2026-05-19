"""CatalogDB — single database interface to catalog.db.

catalog.db owns everything:
  - Metadata: datasets, data_sources, dedup decisions, frequencies, categories
  - Downloads: dataset_downloads, download_budget
  - Metrics: evaluation_metrics, model_runs
  - Loaders: datasets.dataloader_created, datasets.dataset_storage_local

Downloaded files at E:\\datasets\\ are NEVER altered — dataloaders handle
all transformation in memory.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


class CatalogDB:
    """Thread-safe interface to catalog.db (single source of truth).

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    """

    def __init__(self, db_path: Path | str = config.DB_PATH):
        self._path = Path(db_path)
        self._lock = threading.Lock()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Migrations ────────────────────────────────────────────────────────────

    def run_migrations(self) -> None:
        """Execute all .sql files in migrations/ (idempotent)."""
        migration_dir = config.MIGRATIONS_DIR
        if not migration_dir.exists():
            return
        with self._lock, self._conn() as conn:
            for f in sorted(migration_dir.glob("*.sql")):
                conn.executescript(f.read_text(encoding="utf-8"))

    # ══════════════════════════════════════════════════════════════════════════
    # METADATA READS (datasets, sources, dedup, frequencies)
    # ══════════════════════════════════════════════════════════════════════════

    def get_eligible_datasets(self, tier: int | None = None) -> list[dict]:
        """Return KEEP datasets, excluding gift_eval/fev_eval and
        language/classification-only. Ordered by priority tier."""
        sql = """
        SELECT
            d.id, d.unified_id, d.title, d.homepage, d.data_type,
            d.dataset_origin, d.dataset_storage_local, d.dataloader_created,
            ds.name AS source_name, dsr.source_url AS paper_url,
            CASE
                WHEN EXISTS (
                    SELECT 1 FROM dataset_tasks dt2
                    JOIN primary_tasks pt2 ON pt2.id = dt2.task_id
                    WHERE dt2.dataset_id = d.id
                      AND LOWER(pt2.name) IN ('forecasting','time series forecasting',
                          'probabilistic forecasting','anomaly detection')
                ) THEN 1
                WHEN EXISTS (
                    SELECT 1 FROM dataset_tasks dt2
                    JOIN primary_tasks pt2 ON pt2.id = dt2.task_id
                    WHERE dt2.dataset_id = d.id
                      AND LOWER(pt2.name) IN ('regression','imputation','prediction')
                ) THEN 2
                ELSE 3
            END AS priority_tier
        FROM datasets d
        JOIN data_sources ds ON ds.id = d.data_source_id
        JOIN deduplication_decisions dd ON dd.dataset_id = d.id
        LEFT JOIN dataset_sources_raw dsr ON dsr.dataset_id = d.id
        LEFT JOIN dataset_downloads dl ON dl.dataset_id = d.id
        WHERE
            d.enrich_skip = 0
            AND d.homepage IS NOT NULL AND d.homepage != ''
            AND dd.action = 'KEEP'
            AND ds.name NOT LIKE 'gift_eval%'
            AND ds.name NOT LIKE 'fev_eval%'
            AND COALESCE(dl.status, 'pending') NOT IN ('success', 'running')
            AND COALESCE(LOWER(d.dataset_origin), '') != 'synthetic'
            AND NOT (
                EXISTS (
                    SELECT 1 FROM dataset_tasks dt
                    JOIN primary_tasks pt ON pt.id = dt.task_id
                    WHERE dt.dataset_id = d.id
                )
                AND NOT EXISTS (
                    SELECT 1 FROM dataset_tasks dt
                    JOIN primary_tasks pt ON pt.id = dt.task_id
                    WHERE dt.dataset_id = d.id
                      AND LOWER(pt.name) NOT IN (
                          'classification','image classification','text classification',
                          'object detection','language modeling','nlp',
                          'sentiment analysis','machine translation','question answering',
                          'named entity recognition','text generation'
                      )
                )
            )
        ORDER BY priority_tier ASC, d.id ASC
        """
        with self._conn() as conn:
            rows = conn.execute(sql).fetchall()
        results = [dict(r) for r in rows]
        if tier is not None:
            results = [r for r in results if r["priority_tier"] == tier]
        return results

    def get_dataset_by_id(self, dataset_id: int) -> dict | None:
        sql = """
        SELECT d.*, ds.name AS source_name, dsr.source_url AS paper_url
        FROM datasets d
        JOIN data_sources ds ON ds.id = d.data_source_id
        LEFT JOIN dataset_sources_raw dsr ON dsr.dataset_id = d.id
        WHERE d.id = ?
        """
        with self._conn() as conn:
            row = conn.execute(sql, (dataset_id,)).fetchone()
        return dict(row) if row else None

    def get_dataset_by_uid(self, unified_id: str) -> dict | None:
        sql = """
        SELECT d.*, ds.name AS source_name, dsr.source_url AS paper_url
        FROM datasets d
        JOIN data_sources ds ON ds.id = d.data_source_id
        LEFT JOIN dataset_sources_raw dsr ON dsr.dataset_id = d.id
        WHERE d.unified_id = ?
        """
        with self._conn() as conn:
            row = conn.execute(sql, (unified_id,)).fetchone()
        return dict(row) if row else None

    def get_frequency(self, dataset_id: int) -> str | None:
        sql = "SELECT temporal_resolution FROM dataset_frequencies WHERE dataset_id = ? LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql, (dataset_id,)).fetchone()
        return row["temporal_resolution"] if row else None

    def search_dataset_by_title(self, title: str) -> dict | None:
        sql = """
        SELECT d.id, d.unified_id, d.title, d.homepage, ds.name AS source_name
        FROM datasets d
        JOIN data_sources ds ON ds.id = d.data_source_id
        WHERE LOWER(d.title) = LOWER(?) OR LOWER(d.title) LIKE ?
        LIMIT 1
        """
        with self._conn() as conn:
            row = conn.execute(sql, (title, f"%{title}%")).fetchone()
        return dict(row) if row else None

    def get_downloaded_datasets(self) -> list[dict]:
        """All successfully downloaded datasets."""
        sql = """
        SELECT d.id, d.unified_id, d.title, d.dataset_storage_local, d.dataloader_created,
               dl.local_path, dl.file_size_bytes, dl.downloaded_at
        FROM datasets d
        JOIN dataset_downloads dl ON dl.dataset_id = d.id
        WHERE dl.status = 'success'
        ORDER BY d.id
        """
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql).fetchall()]

    # ══════════════════════════════════════════════════════════════════════════
    # DOWNLOAD TRACKING
    # ══════════════════════════════════════════════════════════════════════════

    def get_download_status(self, dataset_id: int) -> str:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM dataset_downloads WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        return row["status"] if row else "pending"

    def upsert_download(self, dataset_id: int, *, status: str, **kwargs: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        fields = {"dataset_id": dataset_id, "status": status, "downloaded_at": now}
        fields.update({k: v for k, v in kwargs.items() if v is not None})

        cols = ", ".join(fields.keys())
        placeholders = ", ".join(["?"] * len(fields))
        updates = ", ".join(f"{k} = excluded.{k}" for k in fields if k != "dataset_id")

        sql = f"""
        INSERT INTO dataset_downloads ({cols}) VALUES ({placeholders})
        ON CONFLICT(dataset_id) DO UPDATE SET {updates}
        """
        with self._lock, self._conn() as conn:
            conn.execute(sql, list(fields.values()))
            # Also update datasets.dataset_storage_local
            if status == "success" and "local_path" in kwargs:
                conn.execute(
                    "UPDATE datasets SET dataset_storage_local = ? WHERE id = ?",
                    (kwargs["local_path"], dataset_id),
                )

    # ══════════════════════════════════════════════════════════════════════════
    # BUDGET
    # ══════════════════════════════════════════════════════════════════════════

    def get_budget(self) -> tuple[int, int]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT total_bytes_used, budget_bytes FROM download_budget WHERE id = 1"
            ).fetchone()
        return (row["total_bytes_used"], row["budget_bytes"]) if row else (0, config.BUDGET_BYTES)

    def add_to_budget(self, bytes_added: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE download_budget SET total_bytes_used = total_bytes_used + ?, last_updated = ? WHERE id = 1",
                (bytes_added, now),
            )

    # ══════════════════════════════════════════════════════════════════════════
    # DATALOADER TRACKING (uses existing datasets.dataloader_created column)
    # ══════════════════════════════════════════════════════════════════════════

    def set_dataloader_created(self, dataset_id: int, value: int = 1) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE datasets SET dataloader_created = ? WHERE id = ?",
                (value, dataset_id),
            )

    def get_datasets_needing_loaders(self) -> list[dict]:
        """Downloaded datasets that don't have dataloaders yet."""
        sql = """
        SELECT d.id, d.unified_id, d.title, dl.local_path
        FROM datasets d
        JOIN dataset_downloads dl ON dl.dataset_id = d.id
        WHERE dl.status = 'success' AND d.dataloader_created = 0
        ORDER BY d.id
        """
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql).fetchall()]

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION METRICS
    # ══════════════════════════════════════════════════════════════════════════

    def upsert_metric(
        self,
        dataset_id: int,
        model_name: str,
        metric_name: str,
        metric_value: float,
        *,
        prediction_length: int | None = None,
        context_window: int | None = None,
        dataset_split: str = "test",
        source_file: str | None = None,
        computed_by: str = "fev",
        notes: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO evaluation_metrics
            (dataset_id, model_name, metric_name, metric_value,
             prediction_length, context_window, dataset_split,
             source_file, computed_by, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id, model_name, metric_name,
                    prediction_length, context_window, dataset_split)
        DO UPDATE SET
            metric_value = excluded.metric_value,
            source_file  = COALESCE(excluded.source_file, source_file),
            computed_by  = excluded.computed_by,
            notes        = COALESCE(excluded.notes, notes),
            created_at   = excluded.created_at
        """
        with self._lock, self._conn() as conn:
            conn.execute(sql, (
                dataset_id, model_name, metric_name, metric_value,
                prediction_length, context_window, dataset_split,
                source_file, computed_by, notes, now,
            ))

    def get_metrics(
        self,
        dataset_id: int | None = None,
        model_name: str | None = None,
        metric_name: str | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM evaluation_metrics WHERE 1=1"
        params: list[Any] = []
        if dataset_id is not None:
            sql += " AND dataset_id = ?"
            params.append(dataset_id)
        if model_name is not None:
            sql += " AND model_name LIKE ?"
            params.append(model_name)
        if metric_name is not None:
            sql += " AND metric_name = ?"
            params.append(metric_name)
        sql += " ORDER BY dataset_id, model_name, metric_name"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL RUNS
    # ══════════════════════════════════════════════════════════════════════════

    def record_model_run(
        self,
        dataset_id: int,
        model_name: str,
        *,
        training_time_s: float | None = None,
        inference_time_s: float | None = None,
        num_forecasts: int | None = None,
        fev_version: str | None = None,
        trained_on_this: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO model_runs
            (dataset_id, model_name, training_time_s, inference_time_s,
             num_forecasts, fev_version, trained_on_this, ran_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id, model_name) DO UPDATE SET
            training_time_s  = excluded.training_time_s,
            inference_time_s = excluded.inference_time_s,
            num_forecasts    = excluded.num_forecasts,
            fev_version      = excluded.fev_version,
            ran_at           = excluded.ran_at
        """
        with self._lock, self._conn() as conn:
            conn.execute(sql, (
                dataset_id, model_name,
                training_time_s, inference_time_s, num_forecasts,
                fev_version, int(trained_on_this), now,
            ))
