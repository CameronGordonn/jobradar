"""Storage: one DuckDB file.

raw_<source>  untouched API payloads, one row per source job id. A rerun updates rows
              in place, so loads are idempotent. ingested_at is the latest load and
              first_ingested_at the first, both UTC. dbt builds from these.
jobs          parsed postings after stage 0, deduped across sources on org + title +
              location. The ranking stages read this.
scores        stage 2 output. digested: what has already gone out in a digest.
ingest_runs   rows fetched and new per source per run, for the data quality monitor.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .models import Job

log = logging.getLogger(__name__)
RAW_SOURCES = ("usajobs", "greenhouse", "lever", "ashby")

_RAW_TABLE = """
CREATE TABLE IF NOT EXISTS raw_{source} (
    source_id VARCHAR PRIMARY KEY,
    board VARCHAR,                 -- board token, or "keyword|location" for USAJOBS
    board_name VARCHAR,
    payload JSON,
    payload_hash VARCHAR,
    first_ingested_at TIMESTAMP,
    ingested_at TIMESTAMP,
    run_id VARCHAR
);
"""

SCHEMA = "".join(_RAW_TABLE.format(source=s) for s in RAW_SOURCES) + """
CREATE TABLE IF NOT EXISTS jobs (
    key VARCHAR PRIMARY KEY,
    source VARCHAR, source_id VARCHAR, title VARCHAR, org VARCHAR, location VARCHAR, url VARCHAR,
    description VARCHAR, posted VARCHAR, closes VARCHAR, extra VARCHAR,
    filter_reason VARCHAR,         -- NULL means it passed hard filters
    recall DOUBLE,                 -- stage-1 similarity to best resume variant
    first_seen VARCHAR, last_seen VARCHAR
);
CREATE TABLE IF NOT EXISTS scores (
    key VARCHAR PRIMARY KEY,
    fit INTEGER, recommend VARCHAR, variant VARCHAR, evidence VARCHAR,
    gaps VARCHAR, dealbreakers VARCHAR, pitch VARCHAR, model VARCHAR, scored_at VARCHAR
);
CREATE TABLE IF NOT EXISTS labels (
    key VARCHAR PRIMARY KEY,
    label INTEGER,                 -- 1 = would apply, 0 = would skip
    labeled_at VARCHAR
);
CREATE TABLE IF NOT EXISTS digested (
    key VARCHAR PRIMARY KEY,
    digest_date VARCHAR
);
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id VARCHAR, source VARCHAR,
    started_at TIMESTAMP, finished_at TIMESTAMP,
    rows_fetched INTEGER,          -- raw postings returned by the API this run
    rows_new INTEGER,              -- source job ids never seen before
    PRIMARY KEY (run_id, source)
);
"""

_JOB_COLS = ("key", "source", "source_id", "title", "org", "location", "url", "description",
             "posted", "closes", "extra", "filter_reason", "first_seen", "last_seen")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utcnow() -> datetime:
    """Naive UTC, for TIMESTAMP columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


class DB:
    """Thin wrapper so callers get rows as dicts, like sqlite3.Row did."""

    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = str(path)
        self.con = duckdb.connect(self.path, read_only=read_only)

    def execute(self, sql: str, params=()) -> "DB":
        self.con.execute(sql, params)
        return self

    def query(self, sql: str, params=()) -> list[dict]:
        cur = self.con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def scalar(self, sql: str, params=()):
        r = self.con.execute(sql, params).fetchone()
        return r[0] if r else None

    def insert_rows(self, table: str, rows: list[tuple], chunk: int = 500) -> None:
        """Multi-row VALUES inserts: ~10x faster than executemany in DuckDB."""
        if not rows:
            return
        ph = "(" + ",".join("?" * len(rows[0])) + ")"
        for i in range(0, len(rows), chunk):
            part = rows[i:i + chunk]
            self.con.execute(f"INSERT INTO {table} VALUES " + ",".join([ph] * len(part)),
                             [v for r in part for v in r])

    def commit(self) -> None:
        """DuckDB autocommits each statement; kept so callers read like before."""

    def close(self) -> None:
        self.con.close()


def connect(path: str | Path) -> DB:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fresh = not p.exists()
    con = DB(p)
    con.con.execute(SCHEMA)
    legacy = p.with_suffix(".db")
    if fresh and legacy.exists() and legacy != p:
        migrate_sqlite(con, legacy)
    return con


def migrate_sqlite(con: DB, sqlite_path: str | Path) -> dict:
    """One-time copy of jobs, scores, labels and digests from the old SQLite store.
    The SQLite file is left in place."""
    src = sqlite3.connect(sqlite_path)
    counts = {}
    for table in ("jobs", "scores", "labels", "digested"):
        cur = src.execute(f"SELECT * FROM {table}")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        if rows:
            # Name the columns so an older column order can't misalign.
            con.execute(f"CREATE TEMP TABLE _mig AS SELECT * FROM {table} LIMIT 0")
            con.insert_rows(f"_mig ({', '.join(cols)})", rows)
            con.execute(f"INSERT OR REPLACE INTO {table} BY NAME SELECT * FROM _mig")
            con.execute("DROP TABLE _mig")
        counts[table] = len(rows)
    src.close()
    log.info("migrated %s from %s", counts, sqlite_path)
    return counts


def upsert_raw(con: DB, source: str, rows: list[tuple[str, str, str | None, dict]], run_id: str) -> int:
    """rows: (source_id, board, board_name, payload). Returns how many source ids are new.
    The same id can arrive twice in one run (USAJOBS matches several queries); the last wins."""
    if source not in RAW_SOURCES:
        raise ValueError(f"unknown source {source!r}")
    t = utcnow()
    con.execute("CREATE OR REPLACE TEMP TABLE _raw_in (seq INTEGER, source_id VARCHAR, board VARCHAR, "
                "board_name VARCHAR, payload VARCHAR)")
    con.insert_rows("_raw_in", [(i, sid, b, bn, json.dumps(p, sort_keys=True))
                                for i, (sid, b, bn, p) in enumerate(rows)])
    new = con.scalar(f"SELECT COUNT(DISTINCT source_id) FROM _raw_in "
                     f"WHERE source_id NOT IN (SELECT source_id FROM raw_{source})")
    con.execute(f"""
        INSERT INTO raw_{source}
        SELECT source_id, board, board_name, payload::JSON, md5(payload), ?, ?, ?
        FROM (SELECT * FROM _raw_in QUALIFY row_number() OVER (PARTITION BY source_id ORDER BY seq DESC) = 1)
        ON CONFLICT (source_id) DO UPDATE SET
            board = excluded.board, board_name = excluded.board_name, payload = excluded.payload,
            payload_hash = excluded.payload_hash, ingested_at = excluded.ingested_at, run_id = excluded.run_id
    """, (t, t, run_id))
    con.execute("DROP TABLE _raw_in")
    return new


def upsert_jobs(con: DB, items: list[tuple[Job, str | None]]) -> int:
    """Insert or refresh parsed jobs with their stage 0 result. Returns how many keys are new."""
    t = now()
    cols = ", ".join(f"{c} VARCHAR" for c in _JOB_COLS)
    con.execute(f"CREATE OR REPLACE TEMP TABLE _jobs_in (seq INTEGER, {cols})")
    con.insert_rows("_jobs_in", [
        (i, j.key, j.source, j.source_id, j.title, j.org, j.location, j.url, j.description,
         j.posted, j.closes, json.dumps(j.extra), reason, t, t)
        for i, (j, reason) in enumerate(items)])
    new = con.scalar("SELECT COUNT(DISTINCT key) FROM _jobs_in WHERE key NOT IN (SELECT key FROM jobs)")
    names = ", ".join(_JOB_COLS)
    con.execute(f"""
        INSERT INTO jobs ({names})
        SELECT {names} FROM (SELECT * FROM _jobs_in QUALIFY row_number() OVER (PARTITION BY key ORDER BY seq DESC) = 1)
        ON CONFLICT (key) DO UPDATE SET
            description = excluded.description, closes = excluded.closes, extra = excluded.extra,
            filter_reason = excluded.filter_reason, last_seen = excluded.last_seen
    """)
    con.execute("DROP TABLE _jobs_in")
    return new


def set_recall(con: DB, pairs: list[tuple[str, float]]) -> None:
    con.execute("CREATE OR REPLACE TEMP TABLE _recall_in (key VARCHAR, recall DOUBLE)")
    con.insert_rows("_recall_in", pairs)
    con.execute("UPDATE jobs SET recall = r.recall FROM _recall_in r WHERE jobs.key = r.key")
    con.execute("DROP TABLE _recall_in")


def log_run(con: DB, run_id: str, source: str, started: datetime, fetched: int, new: int) -> None:
    con.execute("INSERT OR REPLACE INTO ingest_runs VALUES (?,?,?,?,?,?)",
                (run_id, source, started, utcnow(), fetched, new))


def row_to_job(r: dict) -> Job:
    return Job(source=r["source"], source_id=r["source_id"], title=r["title"], org=r["org"],
               location=r["location"], url=r["url"], description=r["description"],
               posted=r["posted"], closes=r["closes"], extra=json.loads(r["extra"] or "{}"))


def save_score(con: DB, key: str, result: dict, model: str) -> None:
    con.execute(
        """INSERT OR REPLACE INTO scores
           (key, fit, recommend, variant, evidence, gaps, dealbreakers, pitch, model, scored_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (key, result["fit_score"], result["recommend"], result["resume_variant"],
         result["strongest_evidence"], json.dumps(result.get("gaps", [])),
         json.dumps(result.get("dealbreakers", [])), result["pitch"], model, now()),
    )
