"""Paths, constants, and configuration for the catalog bridge."""
from __future__ import annotations

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Single DB — catalog.db owns everything: metadata + download status + metrics
DB_PATH = Path(r"E:\z_dataset lists\catalog.db")

# Downloaded files live here — NEVER altered after download
DOWNLOAD_ROOT = Path(r"E:\datasets")
LOG_PATH = DOWNLOAD_ROOT / "_logs" / "download_log.md"
MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# ── Download budget ───────────────────────────────────────────────────────────
BUDGET_BYTES = 200 * 1024**3  # 200 GB

# ── Download settings ─────────────────────────────────────────────────────────
MAX_WORKERS = 3
REQUEST_TIMEOUT = 60
CHUNK_SIZE = 1024 * 1024  # 1 MB
RATE_LIMIT_SECS = 1.5
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": "DatasetPipeline/1.0 (catalog-fev-bridge)",
    "Accept": "*/*",
}

# ── URL classification ────────────────────────────────────────────────────────
PAPER_URL_HOSTS = frozenset({
    "arxiv.org", "semanticscholar.org", "openreview.net",
    "dl.acm.org", "ieeexplore.ieee.org", "researchgate.net",
})

DOWNLOADABLE_EXTS = frozenset({
    ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".csv", ".parquet", ".h5", ".hdf5",
    ".pkl", ".pickle", ".json", ".jsonl",
    ".tsv", ".arrow", ".feather",
})

# ── Task-type filtering ──────────────────────────────────────────────────────
EXCLUDE_TASKS = frozenset({
    "classification", "image classification", "text classification",
    "object detection", "language modeling", "nlp",
    "sentiment analysis", "machine translation", "question answering",
    "named entity recognition", "text generation",
})

FORECAST_TASKS = frozenset({
    "forecasting", "time series forecasting",
    "probabilistic forecasting", "anomaly detection",
})

REGRESSION_TASKS = frozenset({
    "regression", "imputation", "prediction",
})

# ── Frequency → forecast defaults ────────────────────────────────────────────
FREQ_DEFAULTS: dict[str, dict] = {
    "minutely":   {"horizon": 60,  "seasonality": 60},
    "15_minutes": {"horizon": 96,  "seasonality": 96},
    "hourly":     {"horizon": 24,  "seasonality": 24},
    "daily":      {"horizon": 7,   "seasonality": 7},
    "weekly":     {"horizon": 4,   "seasonality": 52},
    "monthly":    {"horizon": 12,  "seasonality": 12},
    "quarterly":  {"horizon": 4,   "seasonality": 4},
    "yearly":     {"horizon": 1,   "seasonality": 1},
}
