"""Storage: one DuckDB file.

raw_<source>  untouched API payloads, one row per source job id. A rerun updates rows
              in place, so loads are idempotent. ingested_at is the latest load and
              first_ingested_at the first, both UTC. dbt builds from these.
jobs          parsed postings with their stage 0 result, one row per source job id
              (key = "<source>:<source_id>"). dedupe_key groups the same org + title +
              location so a duplicate is never sent to Claude twice. The ranking stages read this.
scores        stage 2 output. digested: what has already gone out in a digest.
ingest_runs   rows fetched and new per source per run, for the data quality monitor.

Hand labels live in a separate file; see labeling.py.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
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

# Public warehouse tables: everything the scheduled pipeline and dbt need, nothing personal.
WAREHOUSE_SCHEMA = "".join(_RAW_TABLE.format(source=s) for s in RAW_SOURCES) + """
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id VARCHAR, source VARCHAR,
    started_at TIMESTAMP, finished_at TIMESTAMP,
    rows_fetched INTEGER,          -- raw postings returned by the API this run
    rows_new INTEGER,              -- source job ids never seen before
    PRIMARY KEY (run_id, source)
);
CREATE TABLE IF NOT EXISTS quality_checks (
    run_id VARCHAR, source VARCHAR, checked_at TIMESTAMP,
    rows_fetched INTEGER,
    trailing_runs INTEGER,         -- earlier runs the median is taken over (up to 7)
    trailing_median DOUBLE,
    ratio DOUBLE,                  -- rows_fetched / trailing_median
    flagged BOOLEAN,
    reason VARCHAR,
    PRIMARY KEY (run_id, source)
);
-- Near-duplicate clusters from embedding similarity (pipeline.dedupe). Empty until a threshold is set.
CREATE TABLE IF NOT EXISTS job_clusters (
    job_id VARCHAR PRIMARY KEY, cluster_id VARCHAR, threshold DOUBLE, clustered_at TIMESTAMP
);
"""

# Personal tables: the ranking pipeline's state. Never created in the public warehouse.
APP_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    key VARCHAR PRIMARY KEY,       -- "<source>:<source_id>"
    source VARCHAR, source_id VARCHAR, title VARCHAR, org VARCHAR, location VARCHAR, url VARCHAR,
    description VARCHAR, posted VARCHAR, closes VARCHAR, extra VARCHAR,
    filter_reason VARCHAR,         -- NULL means it passed hard filters
    recall DOUBLE,                 -- stage-1 similarity to best resume variant
    first_seen VARCHAR, last_seen VARCHAR,
    dedupe_key VARCHAR             -- hash of org + title + location
);
CREATE TABLE IF NOT EXISTS scores (
    key VARCHAR PRIMARY KEY,
    fit INTEGER, recommend VARCHAR, variant VARCHAR, evidence VARCHAR,
    gaps VARCHAR, dealbreakers VARCHAR, pitch VARCHAR, model VARCHAR, scored_at VARCHAR
);
CREATE TABLE IF NOT EXISTS digested (
    key VARCHAR PRIMARY KEY,
    digest_date VARCHAR
);
"""
SCHEMA = WAREHOUSE_SCHEMA + APP_SCHEMA

_JOB_COLS = ("key", "source", "source_id", "title", "org", "location", "url", "description",
             "posted", "closes", "extra", "filter_reason", "first_seen", "last_seen", "dedupe_key")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def utcnow() -> datetime:
    """Naive UTC, for TIMESTAMP columns."""
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


class DB:
    """Thin wrapper so callers get rows as dicts, like sqlite3.Row did."""

    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = str(path)
        self.con = duckdb.connect(self.path, read_only=read_only)

    def execute(self, sql: str, params=()) -> DB:
        self.con.execute(sql, params)
        return self

    def query(self, sql: str, params=()) -> list[dict]:
        cur = self.con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

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

    def reopen(self, wait_seconds: int = 1800) -> None:
        """Reconnect after close(). Another process (a labeling session) may hold the write
        lock, so keep trying for a while before giving up."""
        import time
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                self.con = duckdb.connect(self.path)
                return
            except duckdb.IOException:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{self.path} stayed locked by another process; rerun to resume") from None
                time.sleep(10)

    def commit(self) -> None:
        """DuckDB autocommits each statement; kept so callers read like before."""

    def close(self) -> None:
        self.con.close()


def connect(path: str | Path, app_tables: bool = True) -> DB:
    """app_tables=False gives the public warehouse: raw tables and run logs only."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fresh = not p.exists()
    con = DB(p)
    con.con.execute(SCHEMA if app_tables else WAREHOUSE_SCHEMA)
    if not app_tables:
        return con
    legacy = p.with_suffix(".db")
    if fresh and legacy.exists() and legacy != p:
        migrate_sqlite(con, legacy)
    migrate_job_keys(con)
    return con


def migrate_job_keys(con: DB) -> int:
    """Rekey jobs from the old org+title+location hash to "<source>:<source_id>". The hash
    becomes dedupe_key. Where an edited posting left several rows for one source id, the
    latest row wins; scores and digest marks move to the new key (latest score wins).
    Old hash keys contain no ":", which is how old rows are recognized. Returns rows rekeyed."""
    con.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS dedupe_key VARCHAR")
    if not con.scalar("SELECT COUNT(*) FROM jobs WHERE key NOT LIKE '%:%'"):
        return 0
    n = con.scalar("SELECT COUNT(*) FROM jobs")
    con.execute("BEGIN")
    try:
        con.execute("""CREATE TEMP TABLE _keymap AS SELECT key AS old_key, source || ':' || source_id AS new_key,
                       last_seen FROM jobs""")
        con.execute("""CREATE TEMP TABLE _jobs AS SELECT * REPLACE (source || ':' || source_id AS key,
                       coalesce(dedupe_key, key) AS dedupe_key) FROM jobs
                       QUALIFY row_number() OVER (PARTITION BY source, source_id ORDER BY last_seen DESC, key) = 1""")
        con.execute("""CREATE TEMP TABLE _scores AS SELECT s.* REPLACE (m.new_key AS key) FROM scores s
                       JOIN _keymap m ON m.old_key = s.key
                       QUALIFY row_number() OVER (PARTITION BY m.new_key ORDER BY s.scored_at DESC) = 1""")
        con.execute("""CREATE TEMP TABLE _digested AS SELECT d.* REPLACE (m.new_key AS key) FROM digested d
                       JOIN _keymap m ON m.old_key = d.key
                       QUALIFY row_number() OVER (PARTITION BY m.new_key ORDER BY d.digest_date) = 1""")
        for t in ("jobs", "scores", "digested"):
            con.execute(f"DELETE FROM {t}")
            con.execute(f"INSERT INTO {t} BY NAME SELECT * FROM _{t}")
            con.execute(f"DROP TABLE _{t}")
        con.execute("DROP TABLE _keymap")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    after = con.scalar("SELECT COUNT(*) FROM jobs")
    log.info("rekeyed jobs on source id: %d rows -> %d (older versions of edited postings merged)", n, after)
    return n


def migrate_sqlite(con: DB, sqlite_path: str | Path) -> dict:
    """One-time copy of jobs, scores and digests from the old SQLite store.
    The SQLite file is left in place. Its binary labels are not carried over (labels
    are now apply/maybe/no on frozen sets), so a warning says how many there were."""
    src = sqlite3.connect(sqlite_path)
    counts = {}
    old_labels = src.execute("SELECT COUNT(*) FROM labels").fetchone()[0] if src.execute(
        "SELECT 1 FROM sqlite_master WHERE name='labels'").fetchone() else 0
    if old_labels:
        log.warning("%d old binary labels in %s were not migrated", old_labels, sqlite_path)
    for table in ("jobs", "scores", "digested"):
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


def upsert_raw(con: DB, source: str, rows: list[tuple[str, str, str | None, dict]], run_id: str,
               at: datetime | None = None) -> int:
    """rows: (source_id, board, board_name, payload). Returns how many source ids are new.
    The same id can arrive twice in one run (USAJOBS matches several queries); the last wins.
    `at` overrides the load time (fixture loading)."""
    if source not in RAW_SOURCES:
        raise ValueError(f"unknown source {source!r}")
    t = at or utcnow()
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
    """Insert or refresh parsed jobs with their stage 0 result. Returns how many keys are new.
    An edited posting updates its row in place, title and location included."""
    t = now()
    cols = ", ".join(f"{c} VARCHAR" for c in _JOB_COLS)
    con.execute(f"CREATE OR REPLACE TEMP TABLE _jobs_in (seq INTEGER, {cols})")
    con.insert_rows("_jobs_in", [
        (i, j.key, j.source, j.source_id, j.title, j.org, j.location, j.url, j.description,
         j.posted, j.closes, json.dumps(j.extra), reason, t, t, j.dedupe_key)
        for i, (j, reason) in enumerate(items)])
    new = con.scalar("SELECT COUNT(DISTINCT key) FROM _jobs_in WHERE key NOT IN (SELECT key FROM jobs)")
    names = ", ".join(_JOB_COLS)
    con.execute(f"""
        INSERT INTO jobs ({names})
        SELECT {names} FROM (SELECT * FROM _jobs_in QUALIFY row_number() OVER (PARTITION BY key ORDER BY seq DESC) = 1)
        ON CONFLICT (key) DO UPDATE SET
            title = excluded.title, org = excluded.org, location = excluded.location, url = excluded.url,
            description = excluded.description, closes = excluded.closes, extra = excluded.extra,
            filter_reason = excluded.filter_reason, last_seen = excluded.last_seen, dedupe_key = excluded.dedupe_key
    """)
    con.execute("DROP TABLE _jobs_in")
    return new


def set_recall(con: DB, pairs: list[tuple[str, float]]) -> None:
    con.execute("CREATE OR REPLACE TEMP TABLE _recall_in (key VARCHAR, recall DOUBLE)")
    con.insert_rows("_recall_in", pairs)
    con.execute("UPDATE jobs SET recall = r.recall FROM _recall_in r WHERE jobs.key = r.key")
    con.execute("DROP TABLE _recall_in")


def log_run(con: DB, run_id: str, source: str, started: datetime, fetched: int, new: int,
            finished: datetime | None = None) -> None:
    con.execute("INSERT OR REPLACE INTO ingest_runs VALUES (?,?,?,?,?,?)",
                (run_id, source, started, finished or utcnow(), fetched, new))


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
