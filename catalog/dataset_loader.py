"""LocalDatasetLoader — bridge local files to HF datasets.Dataset for FEV."""
from __future__ import annotations

import json
import zipfile
import tarfile
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

try:
    import datasets as hf_datasets
except ImportError:
    hf_datasets = None  # type: ignore[assignment]


class LocalDatasetLoader:
    """Load a local dataset file into a pandas DataFrame or HF Dataset.

    Supports: csv, tsv, parquet, hdf5, json, jsonl, pickle, feather, arrow,
    zip archives (extracts and finds tabular files inside).

    Parameters
    ----------
    local_path : str | Path
        Path to the dataset file on disk.
    """

    LOADERS: dict[str, str] = {
        ".csv":     "_load_csv",
        ".tsv":     "_load_tsv",
        ".parquet": "_load_parquet",
        ".json":    "_load_json",
        ".jsonl":   "_load_jsonl",
        ".h5":      "_load_hdf5",
        ".hdf5":    "_load_hdf5",
        ".pkl":     "_load_pickle",
        ".pickle":  "_load_pickle",
        ".feather": "_load_feather",
        ".arrow":   "_load_arrow",
        ".zip":     "_load_archive",
        ".tar":     "_load_archive",
        ".gz":      "_load_archive",
        ".bz2":     "_load_archive",
        ".xz":      "_load_archive",
    }

    def __init__(self, local_path: str | Path):
        self.path = Path(local_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

    @property
    def format(self) -> str:
        return self.path.suffix.lower()

    def load_df(self) -> pd.DataFrame:
        """Load the file into a pandas DataFrame (auto-detect format)."""
        loader_name = self.LOADERS.get(self.format)
        if loader_name is None:
            raise ValueError(f"Unsupported file format: {self.format}")
        loader = getattr(self, loader_name)
        return loader()

    def load_hf(self) -> "hf_datasets.Dataset":
        """Load as an HF datasets.Dataset (flat table, not FEV schema)."""
        if hf_datasets is None:
            raise ImportError("pip install datasets")
        df = self.load_df()
        return hf_datasets.Dataset.from_pandas(df)

    def to_fev_dataset(
        self,
        id_col: str | None = None,
        timestamp_col: str | None = None,
        target_cols: list[str] | None = None,
    ) -> "hf_datasets.Dataset":
        """Load and reshape into FEV's expected wide-array schema.

        FEV expects each row to be one time series:
          - id: str
          - timestamp: Sequence(Value('timestamp[ns]'))
          - target: Sequence(Value('float64'))

        This method auto-detects columns and pivots long-format data
        into the wide-array format.
        """
        if hf_datasets is None:
            raise ImportError("pip install datasets")

        df = self.load_df()

        # Auto-detect columns
        id_col = id_col or self._detect_id_column(df)
        timestamp_col = timestamp_col or self._detect_timestamp_column(df)
        target_cols = target_cols or self._detect_target_columns(df, exclude=[id_col, timestamp_col])

        if not target_cols:
            raise ValueError(f"No numeric target columns found. Columns: {df.columns.tolist()}")

        # Parse timestamps
        if timestamp_col and timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values([id_col, timestamp_col] if id_col else [timestamp_col])

        # Build wide-array format
        if id_col and id_col in df.columns:
            groups = df.groupby(id_col, sort=False)
        else:
            # Single series — wrap in a group
            df["_id"] = "series_0"
            id_col = "_id"
            groups = df.groupby(id_col, sort=False)

        records = []
        for series_id, group in groups:
            record: dict = {"id": str(series_id)}
            if timestamp_col and timestamp_col in group.columns:
                record["timestamp"] = group[timestamp_col].tolist()
            for col in target_cols:
                record[col] = group[col].astype(float).tolist()
            records.append(record)

        # Build HF Dataset with Sequence features
        features_dict: dict = {"id": hf_datasets.Value("string")}
        if timestamp_col:
            features_dict["timestamp"] = hf_datasets.Sequence(hf_datasets.Value("timestamp[ns]"))
        for col in target_cols:
            features_dict[col] = hf_datasets.Sequence(hf_datasets.Value("float64"))

        features = hf_datasets.Features(features_dict)
        return hf_datasets.Dataset.from_list(records, features=features)

    # ── Column detection heuristics ───────────────────────────────────────────

    @staticmethod
    def _detect_id_column(df: pd.DataFrame) -> str | None:
        """Find the series ID column."""
        candidates = ["id", "series_id", "item_id", "unique_id", "ts_id", "ID"]
        for c in candidates:
            if c in df.columns:
                return c
        # First string/object column with reasonable cardinality
        for c in df.columns:
            if df[c].dtype == object and df[c].nunique() < len(df) * 0.5:
                return c
        return None

    @staticmethod
    def _detect_timestamp_column(df: pd.DataFrame) -> str | None:
        """Find the timestamp column."""
        candidates = ["timestamp", "date", "time", "datetime", "ds", "Date", "Timestamp"]
        for c in candidates:
            if c in df.columns:
                return c
        # First datetime column
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        # Try parsing string columns
        for c in df.columns:
            if df[c].dtype == object:
                try:
                    pd.to_datetime(df[c].head(5))
                    return c
                except Exception:
                    continue
        return None

    @staticmethod
    def _detect_target_columns(df: pd.DataFrame, exclude: list[str | None]) -> list[str]:
        """All numeric columns not in exclude list."""
        exclude_set = {c for c in exclude if c is not None}
        return [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude_set
        ]

    # ── Format-specific loaders ───────────────────────────────────────────────

    def _load_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.path, low_memory=False)

    def _load_tsv(self) -> pd.DataFrame:
        return pd.read_csv(self.path, sep="\t", low_memory=False)

    def _load_parquet(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)

    def _load_json(self) -> pd.DataFrame:
        try:
            return pd.read_json(self.path)
        except Exception:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame([data])

    def _load_jsonl(self) -> pd.DataFrame:
        return pd.read_json(self.path, lines=True)

    def _load_hdf5(self) -> pd.DataFrame:
        with pd.HDFStore(str(self.path), "r") as store:
            keys = store.keys()
        if len(keys) == 1:
            return pd.read_hdf(str(self.path), key=keys[0])
        # Return first key for simplicity
        return pd.read_hdf(str(self.path), key=keys[0])

    def _load_pickle(self) -> pd.DataFrame:
        obj = pd.read_pickle(self.path)
        if isinstance(obj, pd.DataFrame):
            return obj
        raise ValueError(f"Pickle contains {type(obj)}, expected DataFrame")

    def _load_feather(self) -> pd.DataFrame:
        return pd.read_feather(self.path)

    def _load_arrow(self) -> pd.DataFrame:
        import pyarrow.ipc as ipc
        with ipc.open_file(str(self.path)) as reader:
            return reader.read_pandas()

    def _load_archive(self) -> pd.DataFrame:
        """Extract archive and load first tabular file found."""
        extract_dir = self.path.parent / "_extracted"
        extract_dir.mkdir(exist_ok=True)

        suffix = self.path.suffix.lower()
        name = self.path.name.lower()

        if suffix == ".zip":
            with zipfile.ZipFile(self.path) as zf:
                zf.extractall(extract_dir)
        elif suffix in (".tar", ".gz", ".bz2", ".xz") or ".tar" in name:
            with tarfile.open(self.path) as tf:
                tf.extractall(extract_dir)
        else:
            # .gz single file (not tar.gz)
            import gzip, shutil
            out_file = extract_dir / self.path.stem
            with gzip.open(self.path, "rb") as f_in, open(out_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            return LocalDatasetLoader(out_file).load_df()

        # Find first tabular file
        for pattern in ("**/*.parquet", "**/*.csv", "**/*.json", "**/*.tsv"):
            found = sorted(extract_dir.glob(pattern))
            if found:
                return LocalDatasetLoader(found[0]).load_df()

        raise ValueError(f"No tabular file found in archive: {self.path.name}")
