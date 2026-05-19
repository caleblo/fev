#!/usr/bin/env python3
"""
Update catalog.db:
1. Mark monash_tsf (TFM-017..034) as SKIP duplicates with canonical IDs
2. Populate publisher/organization/year from 12 known papers (via source_url arxiv ID)
3. Infer publisher from DOI prefix patterns
4. Populate doi from source_url where source is a doi.org link
"""
import sqlite3
from datetime import datetime, timezone

DB_PATH = r"E:\z_dataset lists\catalog.db"

PRIORITY_SOURCES = [
    'gift_eval/gift_eval', 'gift_eval_monash', 'fev_eval',
    'ashokarxiv2025/chronos', 'ashokarxiv2025/sun',
]

# arXiv ID → (publisher/venue, organization, year)
PAPER_META = {
    '2105.06643': ('NeurIPS 2021 Datasets & Benchmarks', 'Monash University', 2021),
    '2403.07815': ('ICML 2024', 'Amazon Web Services', 2024),
    '2410.10393': ('arXiv 2024', 'Salesforce AI Research', 2024),
    '2509.26468': ('arXiv 2026', 'Amazon Web Services', 2026),
    '2402.03885': ('ICML 2024', 'CMU AutonLab', 2024),
    '2110.03224': ('JMLR 2022', 'Unit8 Co. Ltd.', 2022),
    '1703.07015': ('SIGIR 2018', 'Carnegie Mellon University', 2018),
    '2501.15942': ('arXiv 2025', 'Peking University', 2025),
    '2409.12915': ('ICML 2025', 'CMU AutonLab', 2025),
    '2402.02592': ('ICML 2024', 'Salesforce AI Research', 2024),
    '2106.13008': ('NeurIPS 2021', 'Tsinghua University', 2021),
    '2012.07436': ('AAAI 2021', 'Beihang University', 2021),
}

DOI_PUBLISHERS = [
    ('10.1007/', 'Springer'),
    ('10.1145/', 'ACM'),
    ('10.1609/', 'AAAI Press'),
    ('10.18653/', 'ACL Anthology'),
    ('10.1162/', 'MIT Press'),
    ('10.1109/', 'IEEE'),
    ('10.24963/', 'IJCAI'),
]


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # ── 1. Dedup: mark TFM-017..034 as SKIP ───────────────────────────────────
    monash_tsf_uids = [f'TFM-{i:03d}' for i in range(17, 35)]
    dedup_updates = 0

    for uid in monash_tsf_uids:
        row = conn.execute(
            'SELECT id, title FROM datasets WHERE unified_id=?', (uid,)
        ).fetchone()
        if not row:
            continue
        did, title = row
        clean_title = (
            title.replace(' (Monash)', '').replace('(Monash)', '').strip().lower()
        )

        matches = conn.execute("""
            SELECT d.id, d.unified_id, ds.name
            FROM datasets d
            JOIN data_sources ds ON ds.id = d.data_source_id
            LEFT JOIN deduplication_decisions dd ON dd.dataset_id = d.id
            WHERE d.id != ? AND dd.action = 'KEEP'
              AND LOWER(REPLACE(REPLACE(d.title,' (Monash)',''),'(Monash)','')) = ?
            ORDER BY d.id
        """, (did, clean_title)).fetchall()

        if not matches:
            print(f"  WARN: no KEEP match for {uid} '{title}' - leaving as KEEP")
            continue

        best = matches[0]
        for m in matches:
            for prio in PRIORITY_SOURCES:
                if prio in m[2]:
                    best = m
                    break

        conn.execute("""
            INSERT OR REPLACE INTO deduplication_decisions
                (dataset_id, action, reason, canonical_id)
            VALUES (?, 'SKIP', 'SKIP_SAME_PAPER', ?)
        """, (did, best[0]))
        dedup_updates += 1
        print(f"  SKIP {uid}: '{title[:35]}' → canonical {best[1]} ({best[2]})")

    conn.commit()
    print(f"\n1. Dedup: {dedup_updates} monash_tsf datasets marked SKIP")

    # ── 2. Publisher/org/year from arXiv paper source_urls ────────────────────
    pub_updates = 0
    for arxiv_id, (publisher, organization, year) in PAPER_META.items():
        rows = conn.execute("""
            SELECT DISTINCT d.id
            FROM datasets d
            JOIN dataset_sources_raw dsr ON dsr.dataset_id = d.id
            WHERE dsr.source_url LIKE ? OR dsr.source_url LIKE ?
        """, (
            f'%arxiv.org/abs/{arxiv_id}%',
            f'%arxiv.org/pdf/{arxiv_id}%',
        )).fetchall()

        if not rows:
            print(f"  WARN: no datasets found for arxiv:{arxiv_id}")
            continue

        dataset_ids = [r[0] for r in rows]
        conn.executemany("""
            UPDATE datasets SET
                publisher    = CASE WHEN publisher    IS NULL THEN ? ELSE publisher    END,
                organization = CASE WHEN organization IS NULL THEN ? ELSE organization END,
                year         = CASE WHEN year         IS NULL THEN ? ELSE year         END
            WHERE id = ?
        """, [(publisher, organization, year, did) for did in dataset_ids])
        pub_updates += len(dataset_ids)

    conn.commit()
    print(f"2. Publisher/org/year: updated {pub_updates} datasets from 12 papers")

    # ── 3. Publisher from DOI prefix ──────────────────────────────────────────
    doi_pub_updates = 0
    for prefix, pub in DOI_PUBLISHERS:
        n = conn.execute("""
            UPDATE datasets SET publisher = ?
            WHERE publisher IS NULL AND doi LIKE ?
        """, (pub, f'%{prefix}%')).rowcount
        doi_pub_updates += n
    conn.commit()
    print(f"3. DOI-based publisher: updated {doi_pub_updates} datasets")

    # ── 4. DOI from source_url ─────────────────────────────────────────────────
    doi_from_url = 0
    rows = conn.execute("""
        SELECT d.id, dsr.source_url
        FROM datasets d
        JOIN dataset_sources_raw dsr ON dsr.dataset_id = d.id
        WHERE d.doi IS NULL
          AND (dsr.source_url LIKE 'https://doi.org/10.%'
               OR dsr.source_url LIKE 'http://doi.org/10.%')
    """).fetchall()
    for did, url in rows:
        doi = url.replace('https://doi.org/', '').replace('http://doi.org/', '').strip()
        if doi:
            conn.execute(
                "UPDATE datasets SET doi=? WHERE id=? AND doi IS NULL", (doi, did)
            )
            doi_from_url += 1
    conn.commit()
    print(f"4. DOI from source_url: updated {doi_from_url} datasets")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=== Final stats ===")
    for col in ['publisher', 'organization', 'year', 'doi']:
        nn = conn.execute(
            f'SELECT COUNT(*) FROM datasets WHERE {col} IS NOT NULL'
        ).fetchone()[0]
        nl = conn.execute(
            f'SELECT COUNT(*) FROM datasets WHERE {col} IS NULL'
        ).fetchone()[0]
        print(f"  {col:15s}: {nn:4d} filled, {nl} null")

    for action in ['KEEP', 'SKIP']:
        n = conn.execute(
            "SELECT COUNT(*) FROM deduplication_decisions WHERE action=?", (action,)
        ).fetchone()[0]
        print(f"  dedup {action:5s}: {n}")

    print()
    print("=== Updated TFM-017..034 dedup ===")
    for r in conn.execute("""
        SELECT d.unified_id, dd.action, dd.reason, d2.unified_id as canonical_uid
        FROM deduplication_decisions dd
        JOIN datasets d ON d.id = dd.dataset_id
        LEFT JOIN datasets d2 ON d2.id = dd.canonical_id
        WHERE d.unified_id LIKE 'TFM-%'
          AND CAST(SUBSTR(d.unified_id, 5) AS INTEGER) BETWEEN 17 AND 34
        ORDER BY d.unified_id
    """).fetchall():
        print(f"  {r[0]:12s} {r[1]:6s} {r[2]:25s} -> {r[3]}")

    print()
    print("=== Sample publisher updates (Monash paper) ===")
    for r in conn.execute("""
        SELECT d.unified_id, d.publisher, d.organization, d.year
        FROM datasets d
        JOIN dataset_sources_raw dsr ON dsr.dataset_id = d.id
        WHERE dsr.source_url LIKE '%2105.06643%'
          AND d.publisher IS NOT NULL
        LIMIT 8
    """).fetchall():
        print(f"  {r[0]:12s} | {r[1]:40s} | {r[2]:30s} | {r[3]}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
