-- Operational tables added to catalog.db (idempotent)

CREATE TABLE IF NOT EXISTS dataset_downloads (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id       INTEGER NOT NULL UNIQUE REFERENCES datasets(id),
    status           TEXT NOT NULL DEFAULT 'pending',
    download_url     TEXT,
    url_type         TEXT,
    local_path       TEXT,
    file_size_bytes  INTEGER,
    downloaded_at    TEXT,
    error_message    TEXT,
    retry_count      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS download_budget (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    total_bytes_used INTEGER DEFAULT 0,
    budget_bytes     INTEGER DEFAULT 214748364800,
    last_updated     TEXT
);
INSERT OR IGNORE INTO download_budget (id) VALUES (1);

CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id        INTEGER NOT NULL REFERENCES datasets(id),
    model_name        TEXT NOT NULL,
    metric_name       TEXT NOT NULL,
    metric_value      REAL NOT NULL,
    prediction_length INTEGER,
    context_window    INTEGER,
    dataset_split     TEXT DEFAULT 'test',
    source_file       TEXT,
    computed_by       TEXT DEFAULT 'fev',
    notes             TEXT,
    created_at        TEXT,
    UNIQUE(dataset_id, model_name, metric_name,
           prediction_length, context_window, dataset_split)
);
CREATE INDEX IF NOT EXISTS idx_em_dataset ON evaluation_metrics(dataset_id);
CREATE INDEX IF NOT EXISTS idx_em_model   ON evaluation_metrics(model_name);

CREATE TABLE IF NOT EXISTS model_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id       INTEGER NOT NULL REFERENCES datasets(id),
    model_name       TEXT NOT NULL,
    training_time_s  REAL,
    inference_time_s REAL,
    num_forecasts    INTEGER,
    fev_version      TEXT,
    trained_on_this  INTEGER DEFAULT 0,
    ran_at           TEXT,
    UNIQUE(dataset_id, model_name)
);
