from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import Job

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    key TEXT PRIMARY KEY,
    source TEXT, source_id TEXT, title TEXT, org TEXT, location TEXT, url TEXT,
    description TEXT, posted TEXT, closes TEXT, extra TEXT,
    filter_reason TEXT,            -- NULL means it passed hard filters
    recall REAL,                   -- stage-1 similarity to best resume variant
    first_seen TEXT, last_seen TEXT
);
CREATE TABLE IF NOT EXISTS scores (
    key TEXT PRIMARY KEY REFERENCES jobs(key),
    fit INTEGER, recommend TEXT, variant TEXT, evidence TEXT,
    gaps TEXT, dealbreakers TEXT, pitch TEXT, model TEXT, scored_at TEXT
);
CREATE TABLE IF NOT EXISTS labels (
    key TEXT PRIMARY KEY REFERENCES jobs(key),
    label INTEGER,                 -- 1 = would apply, 0 = would skip
    labeled_at TEXT
);
CREATE TABLE IF NOT EXISTS digested (
    key TEXT PRIMARY KEY REFERENCES jobs(key),
    digest_date TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def upsert_job(con: sqlite3.Connection, job: Job, filter_reason: str | None) -> bool:
    """Insert or refresh a job. Returns True if it is new."""
    t = now()
    existed = con.execute("SELECT 1 FROM jobs WHERE key=?", (job.key,)).fetchone()
    con.execute(
        """INSERT INTO jobs (key, source, source_id, title, org, location, url, description,
                             posted, closes, extra, filter_reason, first_seen, last_seen)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(key) DO UPDATE SET
             description=excluded.description, closes=excluded.closes,
             extra=excluded.extra, filter_reason=excluded.filter_reason,
             last_seen=excluded.last_seen""",
        (job.key, job.source, job.source_id, job.title, job.org, job.location, job.url,
         job.description, job.posted, job.closes, json.dumps(job.extra), filter_reason, t, t),
    )
    return existed is None


def row_to_job(r: sqlite3.Row) -> Job:
    return Job(source=r["source"], source_id=r["source_id"], title=r["title"], org=r["org"],
               location=r["location"], url=r["url"], description=r["description"],
               posted=r["posted"], closes=r["closes"], extra=json.loads(r["extra"] or "{}"))


def save_score(con: sqlite3.Connection, key: str, result: dict, model: str) -> None:
    con.execute(
        """INSERT OR REPLACE INTO scores
           (key, fit, recommend, variant, evidence, gaps, dealbreakers, pitch, model, scored_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (key, result["fit_score"], result["recommend"], result["resume_variant"],
         result["strongest_evidence"], json.dumps(result.get("gaps", [])),
         json.dumps(result.get("dealbreakers", [])), result["pitch"], model, now()),
    )
