"""Job-hunt tracker: a local page over the personal database.

    python -m jobradar.cli tracker            # http://127.0.0.1:8765

Inbox (scored postings not yet acted on), the application pipeline, and weekly stats. Status
changes go to `applications` and `application_events`. Everything stays on this machine: the
server binds to localhost and opens the database only for the length of each request, so the
7am run can still take the write lock.
"""
from __future__ import annotations

import json
import re
import statistics
import uuid
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb

from . import db
from .db import DB
from .models import Job

PAGE = Path(__file__).parent / "tracker.html"
STATUSES = ["saved", "applied", "interviewing", "offer", "rejected", "withdrawn", "skipped"]
APPLIED = {"applied", "interviewing", "offer", "rejected"}  # reaching any of these means it was sent
RESPONSES = {"interviewing", "offer", "rejected"}
INTERVIEWS = {"interviewing", "offer"}
FOLLOW_UP_DAYS = 7    # applied, and no change since: time to nudge
OPEN_GRACE_DAYS = 2   # a board posting not seen in the last N days of runs is probably closed
NEW_DAYS = 2

# For postings without a tagged board (USAJOBS, added by hand). First match wins.
_AREA_RULES = [
    ("geospatial", r"\b(gis|geospatial|remote sensing|geograph\w*|cartograph\w*|imagery|lidar|mapping)\b"),
    ("climate", r"\b(climate|weather|meteorolog\w*|carbon|emissions?|environment\w*|wildfire|hydrolog\w*)\b"),
    ("energy", r"\b(energy|grid|utilit\w*|solar|battery|batteries|power systems?)\b"),
    ("cv", r"\b(computer vision|perception|robot\w*|autonomous|image processing)\b"),
    ("sports", r"\b(sports?|athlet\w*)\b"),
]


# ---------- lookups ----------

def board_areas(sources: dict) -> dict[str, str]:
    """Display name (the posting's org) -> area, from the `area` tags in sources.yaml."""
    return {b["name"].lower(): b["area"] for v in sources.values() for b in v.get("boards", [])
            if b.get("name") and b.get("area")}


def area_for(org: str, title: str, areas: dict[str, str]) -> str:
    if (org or "").lower() in areas:
        return areas[org.lower()]
    text = f"{title} {org}".lower()
    return next((a for a, pat in _AREA_RULES if re.search(pat, text)), "general")


def referral_for(org: str, referrals: dict[str, str]) -> str | None:
    """referrals: company -> contact, from config.yaml (tracker.referrals). Whole-word match."""
    o = (org or "").lower()
    return next((who for co, who in referrals.items()
                 if re.search(rf"(?<![a-z0-9]){re.escape(co.lower())}(?![a-z0-9])", o)), None)


def _cutoffs(con: DB) -> tuple[str, str]:
    """(still-open cutoff, new cutoff) as ISO strings, relative to the latest board run, so a
    missed day of runs doesn't make everything look closed."""
    latest = con.scalar("SELECT max(last_seen) FROM jobs WHERE source != 'manual'")
    ref = datetime.fromisoformat(latest) if latest else datetime.now().astimezone()
    return ((ref - timedelta(days=OPEN_GRACE_DAYS)).isoformat(timespec="seconds"),
            (ref - timedelta(days=NEW_DAYS)).isoformat(timespec="seconds"))


def _decorate(r: dict, areas: dict, referrals: dict) -> dict:
    for k in ("gaps", "dealbreakers"):
        if k in r:
            r[k] = json.loads(r[k] or "[]")
    r["area"] = area_for(r["org"], r["title"], areas)
    r["referral"] = r.get("referral") or referral_for(r["org"], referrals)
    return r


# ---------- reads ----------

def inbox(con: DB, cfg: dict, min_fit: int = 5) -> list[dict]:
    """Scored postings still open that nothing has been done with yet, best first. A duplicate
    group (same org, title, location) disappears once any member of it has a status."""
    open_cut, new_cut = _cutoffs(con)
    rows = con.query("""
        SELECT j.key, j.title, j.org, j.location, j.url, j.closes, j.first_seen, j.source,
               s.fit, s.recommend, s.variant, s.evidence, s.gaps, s.dealbreakers, s.pitch, s.model,
               j.first_seen >= ? AS is_new
        FROM jobs j JOIN scores s USING (key)
        WHERE j.filter_reason IS NULL AND s.fit >= ? AND j.last_seen >= ? AND (j.closes IS NULL OR j.closes >= ?)
          AND NOT EXISTS (SELECT 1 FROM applications a JOIN jobs j2 USING (key)
                          WHERE j2.dedupe_key = j.dedupe_key)
        ORDER BY s.fit DESC, j.first_seen DESC, j.key""", (new_cut, min_fit, open_cut, date.today().isoformat()))
    t = cfg.get("tracker", {})
    areas, referrals = board_areas(cfg["sources"]), t.get("referrals") or {}
    return [_decorate(r, areas, referrals) for r in rows]


def pipeline(con: DB, cfg: dict) -> list[dict]:
    open_cut, _ = _cutoffs(con)
    rows = con.query("""
        SELECT a.key, a.status, a.note, a.referral, a.created_at, a.updated_at, a.applied_at,
               j.title, j.org, j.location, j.url, j.source, j.last_seen, j.closes, s.fit, s.variant,
               (SELECT max(happened_at) FROM application_events e WHERE e.key = a.key) AS last_event
        FROM applications a JOIN jobs j USING (key) LEFT JOIN scores s USING (key)
        WHERE a.status != 'skipped'
        ORDER BY a.updated_at DESC""")
    t = cfg.get("tracker", {})
    areas, referrals = board_areas(cfg["sources"]), t.get("referrals") or {}
    now = datetime.now().astimezone()
    for r in rows:
        _decorate(r, areas, referrals)
        idle = (now - datetime.fromisoformat(r["last_event"] or r["updated_at"])).days
        r["days_idle"] = idle
        r["follow_up"] = r["status"] == "applied" and idle >= FOLLOW_UP_DAYS
        r["posting_closed"] = r["source"] != "manual" and (r["last_seen"] or "") < open_cut
    return rows


def _week(iso: str) -> str:
    d = datetime.fromisoformat(iso).astimezone().date()  # local week, matching "this week"
    return (d - timedelta(days=d.weekday())).isoformat()


def stats(con: DB, cfg: dict) -> dict:
    """Applied, responded (any reply: interview, offer or rejection) and interviewed, by week of
    applying, by area and by resume variant. Plus median days from applying to the first reply."""
    t = cfg.get("tracker", {})
    areas = board_areas(cfg["sources"])
    apps = con.query("""SELECT a.key, a.status, a.applied_at, j.org, j.title, s.variant
                        FROM applications a JOIN jobs j USING (key) LEFT JOIN scores s USING (key)
                        WHERE a.applied_at IS NOT NULL""")
    events = {}
    for e in con.query("SELECT key, status, happened_at FROM application_events ORDER BY happened_at"):
        events.setdefault(e["key"], []).append(e)

    def summarize(group: list[dict]) -> dict:
        n = len(group)
        resp = sum(a["responded"] for a in group)
        inter = sum(a["interviewed"] for a in group)
        return {"applied": n, "responded": resp, "interviewed": inter,
                "response_rate": resp / n if n else None, "interview_rate": inter / n if n else None}

    waits = []
    for a in apps:
        after = [e for e in events.get(a["key"], []) if e["happened_at"] >= a["applied_at"]]
        reply = next((e for e in after if e["status"] in RESPONSES), None)
        a["responded"] = reply is not None
        a["interviewed"] = any(e["status"] in INTERVIEWS for e in after)
        a["area"] = area_for(a["org"], a["title"], areas)
        if reply:
            waits.append((datetime.fromisoformat(reply["happened_at"]) - datetime.fromisoformat(a["applied_at"])).days)

    weeks = {}
    for a in apps:
        weeks.setdefault(_week(a["applied_at"]), []).append(a)
    by = {}
    for field in ("area", "variant"):
        groups = {}
        for a in apps:
            groups.setdefault(a[field] or "added by hand", []).append(a)
        by[field] = sorted(({"name": k, **summarize(v)} for k, v in groups.items()), key=lambda r: -r["applied"])
    counts = {r["status"]: r["n"] for r in con.query("SELECT status, COUNT(*) n FROM applications GROUP BY 1")}
    today = date.today()
    this_week = (today - timedelta(days=today.weekday())).isoformat()
    return {"total": summarize(apps), "by_week": [{"week": w, **summarize(v)} for w, v in sorted(weeks.items())],
            "by_area": by["area"], "by_variant": by["variant"], "status_counts": counts,
            "median_days_to_reply": statistics.median(waits) if waits else None,
            "applied_this_week": len(weeks.get(this_week, [])), "weekly_goal": t.get("weekly_goal")}


def job_detail(con: DB, key: str) -> dict | None:
    rows = con.query("SELECT key, title, org, location, url, description, posted, closes FROM jobs WHERE key=?",
                     (key,))
    return rows[0] if rows else None


# ---------- writes ----------

def set_status(con: DB, key: str, status: str, note: str | None = None, referral: str | None = None) -> None:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if not con.scalar("SELECT COUNT(*) FROM jobs WHERE key=?", (key,)):
        raise KeyError(key)
    t = db.now()
    applied = t if status in APPLIED else None
    con.execute("""INSERT INTO applications VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT (key) DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at,
                       applied_at = coalesce(applications.applied_at, excluded.applied_at),
                       note = coalesce(excluded.note, applications.note),
                       referral = coalesce(excluded.referral, applications.referral)""",
                (key, status, note, referral, t, t, applied))
    con.execute("INSERT INTO application_events VALUES (?,?,?)", (key, status, t))


def update_fields(con: DB, key: str, note: str | None = None, referral: str | None = None) -> None:
    """Edit the note or referral without logging a status change. "" clears a field."""
    for col, val in (("note", note), ("referral", referral)):
        if val is not None:
            con.execute(f"UPDATE applications SET {col} = ?, updated_at = ? WHERE key = ?",
                        (val or None, db.now(), key))


def add_manual(con: DB, title: str, org: str, url: str = "", location: str = "", note: str | None = None,
               referral: str | None = None, status: str = "saved") -> str:
    """A role that came through a person, not a board. filter_reason 'manual' keeps it out of
    scoring and the digest."""
    if not title.strip() or not org.strip():
        raise ValueError("title and company are required")
    job = Job("manual", uuid.uuid4().hex[:12], title.strip(), org.strip(), location.strip(), url.strip(), "")
    t = db.now()
    con.execute("""INSERT INTO jobs (key, source, source_id, title, org, location, url, description, posted, closes,
                                     extra, filter_reason, recall, first_seen, last_seen, dedupe_key)
                   VALUES (?,?,?,?,?,?,?,?,NULL,NULL,'{}','manual',NULL,?,?,?)""",
                (job.key, job.source, job.source_id, job.title, job.org, job.location, job.url, "", t, t,
                 job.dedupe_key))
    set_status(con, job.key, status, note or None, referral or None)
    return job.key


# ---------- server ----------

def _handler(cfg: dict):
    path = cfg["db_path"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet: one line per request is noise in a terminal
            pass

        def _send(self, code: int, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _with_db(self, fn):
            try:
                con = DB(path)
            except duckdb.IOException:
                return self._send(503, {"error": "The database is busy, probably the daily run. "
                                                 "Try again in a minute."})
            try:
                return self._send(200, fn(con))
            except (KeyError, ValueError) as e:
                return self._send(400, {"error": str(e)})
            finally:
                con.close()

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            routes = {
                "/api/inbox": lambda c: inbox(c, cfg, int(q.get("min_fit", 5))),
                "/api/pipeline": lambda c: pipeline(c, cfg),
                "/api/stats": lambda c: stats(c, cfg),
                "/api/job": lambda c: job_detail(c, q.get("key", "")),
            }
            if u.path not in routes:
                return self._send(404, {"error": "not found"})
            return self._with_db(routes[u.path])

        def do_POST(self):
            # JSON only: a cross-site form post can't send it without a CORS preflight,
            # which this server never answers.
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(415, {"error": "JSON only"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            routes = {
                "/api/status": lambda c: set_status(c, body["key"], body["status"], body.get("note"),
                                                    body.get("referral")),
                "/api/fields": lambda c: update_fields(c, body["key"], body.get("note"), body.get("referral")),
                "/api/manual": lambda c: {"key": add_manual(c, body.get("title", ""), body.get("org", ""),
                                                            body.get("url", ""), body.get("location", ""),
                                                            body.get("note"), body.get("referral"),
                                                            body.get("status", "saved"))},
            }
            if urlparse(self.path).path not in routes:
                return self._send(404, {"error": "not found"})
            return self._with_db(lambda c: routes[urlparse(self.path).path](c) or {"ok": True})

    return Handler


def serve(cfg: dict, port: int = 8765) -> None:
    db.connect(cfg["db_path"]).close()  # creates the tracker tables if this is the first run
    server = HTTPServer(("127.0.0.1", port), _handler(cfg))
    print(f"Tracker at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
