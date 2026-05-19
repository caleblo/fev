"""DatasetDownloader — download catalog URLs to E:\\datasets\\ (files never altered)."""
from __future__ import annotations

import hashlib, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from tqdm import tqdm

from . import config
from .db import CatalogDB

_log_lock = threading.Lock()

def _log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    try:
        config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, open(config.LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


class DatasetDownloader:
    """Download datasets from catalog.db URLs to E:\\datasets\\."""

    def __init__(self, db: CatalogDB):
        self.db = db
        self.session = requests.Session()
        self.session.headers.update(config.HEADERS)

    def download_all(self, tier: int | None = None, dry_run: bool = False,
                     max_datasets: int | None = None) -> dict:
        _log("=" * 60)
        _log("Download Agent starting")
        eligible = self.db.get_eligible_datasets(tier=tier)
        if max_datasets:
            eligible = eligible[:max_datasets]
        _log(f"Found {len(eligible)} eligible datasets")
        used, budget = self.db.get_budget()
        _log(f"Budget: {used/1e9:.2f} GB / {budget/1e9:.0f} GB")

        if dry_run:
            for ds in eligible:
                url, ut = self._resolve_url(ds["homepage"], ds.get("paper_url"))
                _log(f"  [DRY] {ds['unified_id']}: {ut} → {url}")
            return {"success": 0, "failed": 0, "skipped": 0, "bytes_total": 0}

        stats = {"success": 0, "failed": 0, "skipped": 0, "bytes_total": 0}
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
            futs = {pool.submit(self._process_one, ds): ds for ds in eligible}
            for fut in as_completed(futs):
                ds = futs[fut]
                try:
                    r = fut.result()
                    stats[r["status"]] += 1
                    stats["bytes_total"] += r.get("bytes", 0)
                except Exception as e:
                    _log(f"Unhandled: {ds['unified_id']}: {e}", "ERROR")
                    self.db.upsert_download(ds["id"], status="failed", error_message=str(e))
                    stats["failed"] += 1
        _log(f"Done. {stats}")
        return stats

    def download_one(self, dataset_id: int) -> Path | None:
        row = self.db.get_dataset_by_id(dataset_id)
        if not row:
            raise ValueError(f"ID {dataset_id} not found")
        r = self._process_one(row)
        return Path(r["local_path"]) if r["status"] == "success" else None

    def _process_one(self, ds: dict) -> dict:
        uid, did = ds["unified_id"], ds["id"]
        used, budget = self.db.get_budget()
        if used >= budget:
            self.db.upsert_download(did, status="skipped", error_message="budget_exhausted")
            return {"status": "skipped", "bytes": 0}

        url, ut = self._resolve_url(ds["homepage"], ds.get("paper_url"))
        _log(f"→ [{uid}] {ut}: {url}")
        self.db.upsert_download(did, status="running", download_url=url, url_type=ut)

        last_err = None
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                dest, size = self._stream(url, uid)
                _log(f"✓ [{uid}] {dest.name} ({size:,} B)")
                self.db.upsert_download(did, status="success", local_path=str(dest),
                                        file_size_bytes=size, download_url=url, url_type=ut)
                self.db.add_to_budget(size)
                time.sleep(config.RATE_LIMIT_SECS)
                return {"status": "success", "bytes": size, "local_path": str(dest)}
            except _BudgetExceeded as e:
                self.db.upsert_download(did, status="skipped", error_message=str(e))
                return {"status": "skipped", "bytes": 0}
            except Exception as e:
                last_err = str(e)
                _log(f"  [{uid}] attempt {attempt}: {e}", "WARN")
                if attempt < config.MAX_RETRIES:
                    time.sleep(2 ** attempt)
        self.db.upsert_download(did, status="failed", error_message=last_err)
        return {"status": "failed", "bytes": 0}

    # ── URL resolution ────────────────────────────────────────────────────────
    def _resolve_url(self, homepage: str, paper_url: str | None) -> tuple[str, str]:
        if self._is_dl(homepage): return homepage, "homepage"
        r = self._gh_raw(homepage)
        if r: return r, "github_raw"
        r = self._arxiv(homepage)
        if r: return r, "arxiv_pdf"
        if paper_url:
            if self._is_dl(paper_url): return paper_url, "paper_url"
            r = self._gh_raw(paper_url)
            if r: return r, "github_raw"
        return homepage, "homepage"

    @staticmethod
    def _is_dl(u): return os.path.splitext(urlparse(u).path)[1].lower() in config.DOWNLOADABLE_EXTS
    @staticmethod
    def _gh_raw(u):
        m = re.match(r"https?://github\.com/([^/]+/[^/]+)/blob/(.+)", u, re.I)
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}" if m else None
    @staticmethod
    def _arxiv(u):
        m = re.match(r"https?://arxiv\.org/abs/(.+)", u, re.I)
        return f"https://arxiv.org/pdf/{m.group(1)}" if m else None

    # ── Download ──────────────────────────────────────────────────────────────
    def _stream(self, url: str, uid: str) -> tuple[Path, int]:
        head = self.session.head(url, timeout=config.REQUEST_TIMEOUT, allow_redirects=True)
        head.raise_for_status()
        cl = int(head.headers.get("Content-Length", 0)) or None
        dest = self._dest(uid, head.url, head.headers)
        resume = dest.stat().st_size if dest.exists() else 0
        if cl and resume >= cl:
            return dest, resume
        if cl:
            used, budget = self.db.get_budget()
            if used + (cl - resume) > budget:
                raise _BudgetExceeded(f"{cl:,} B exceeds budget")
        hdrs = dict(config.HEADERS)
        if resume: hdrs["Range"] = f"bytes={resume}-"
        with self.session.get(url, headers=hdrs, stream=True, timeout=config.REQUEST_TIMEOUT,
                              allow_redirects=True) as r:
            r.raise_for_status()
            written = resume
            with open(dest, "ab" if resume else "wb") as f, \
                 tqdm(total=cl, initial=resume, unit="B", unit_scale=True,
                      desc=uid[:30], leave=False) as bar:
                for chunk in r.iter_content(chunk_size=config.CHUNK_SIZE):
                    if chunk:
                        f.write(chunk); written += len(chunk); bar.update(len(chunk))
        return dest, written

    @staticmethod
    def _dest(uid: str, url: str, headers: dict) -> Path:
        cd = headers.get("Content-Disposition", "")
        m = re.search(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';\s]+)', cd, re.I)
        fn = unquote(m.group(1)) if m else os.path.basename(urlparse(url).path) or None
        if not fn: fn = hashlib.md5(url.encode()).hexdigest()[:12] + ".bin"
        d = config.DOWNLOAD_ROOT / uid
        d.mkdir(parents=True, exist_ok=True)
        return d / fn

class _BudgetExceeded(Exception): pass
