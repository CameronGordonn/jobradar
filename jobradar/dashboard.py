"""Static dashboard for GitHub Pages, built from the warehouse marts.

Aggregates of public job postings only: no labels, rubric, resumes or ranking scores exist
in the warehouse, and nothing here reads the personal database.
"""
from __future__ import annotations

import json
import shutil
from datetime import UTC
from pathlib import Path

import yaml

from .db import DB, utcnow

TEMPLATE = Path(__file__).parent / "dashboard_template.html"
SOURCE_ORDER = ["greenhouse", "usajobs", "ashby", "lever"]  # fixed color slots 1-4


def export(con: DB, sources_path: Path) -> dict:
    src = yaml.safe_load(Path(sources_path).read_text())["sources"]
    boards = {k: len(v.get("boards", [])) for k, v in src.items() if k != "usajobs"}
    us = src.get("usajobs", {})
    q = con.query
    open_rows = q("SELECT COUNT(*) n, SUM(variants) raw, COUNT(DISTINCT company) companies, "
                  "AVG(is_remote::int) remote, AVG((salary_min_annual IS NOT NULL)::int) salary "
                  "FROM marts.mart_open_postings")[0]
    weekly = q("""SELECT date_trunc('week', day)::date AS week, source, SUM(new_postings_deduped) AS n
                  FROM marts.mart_daily_postings
                  WHERE day >= current_date - 7 * 27 AND date_trunc('week', day) < date_trunc('week', current_date)
                  GROUP BY 1, 2 ORDER BY 1, 2""")
    feed = q("""SELECT day, source, postings_in_feed AS n FROM marts.mart_daily_postings
                WHERE postings_in_feed IS NOT NULL ORDER BY 1, 2""")
    family = q("""SELECT role_family, COUNT(*) AS open, SUM(is_remote::int) AS remote
                  FROM marts.mart_open_postings GROUP BY 1 ORDER BY 2 DESC""")
    companies = q("""SELECT company, COUNT(*) AS open FROM marts.mart_open_postings
                     GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 15""")
    health = q("""SELECT source, started_at, rows_fetched, trailing_median, flagged, reason FROM (
                      SELECT *, row_number() OVER (PARTITION BY source ORDER BY started_at DESC) rn
                      FROM marts.mart_source_health) WHERE rn = 1 ORDER BY source""")
    lifecycle = q("""SELECT role_family, COUNT(*) AS n, median(days_listed) AS median_days
                     FROM marts.mart_posting_lifecycle GROUP BY 1 HAVING COUNT(*) >= 10 ORDER BY 3 DESC""")
    closed = con.scalar("SELECT COUNT(*) FROM marts.mart_posting_lifecycle")
    changes = q("""SELECT COUNT(*) AS tracked, COUNT(*) FILTER (WHERE title_changes + location_changes
                          + salary_changes + workplace_changes > 0) AS edited,
                          COUNT(*) FILTER (WHERE salary_changes > 0) AS salary_edited
                   FROM marts.mart_posting_changes""")[0]
    salary = q("""SELECT role_family, COUNT(*) AS n,
                         median((salary_min_annual + salary_max_annual) / 2) AS median_mid
                  FROM marts.mart_open_postings
                  WHERE salary_currency = 'USD' AND salary_min_annual IS NOT NULL
                  GROUP BY 1 HAVING COUNT(*) >= 10 ORDER BY 3 DESC""")
    return {
        "generated_at": utcnow().replace(tzinfo=UTC).isoformat(timespec="minutes"),
        "scope": {"usajobs_queries": len(us.get("keywords", [])) * len(us.get("locations", [None])),
                  "usajobs_keywords": us.get("keywords", []), "boards": boards,
                  "board_count": sum(boards.values())},
        "tiles": {"open_postings": open_rows["n"], "open_before_dedupe": int(open_rows["raw"] or 0),
                  "companies": open_rows["companies"], "remote_share": open_rows["remote"],
                  "salary_share": open_rows["salary"]},
        "source_order": SOURCE_ORDER,
        "weekly_by_source": [{"x": str(r["week"]), "source": r["source"], "n": int(r["n"])} for r in weekly],
        "feed_by_source": [{"x": str(r["day"]), "source": r["source"], "n": int(r["n"])} for r in feed],
        "open_by_family": [{"label": r["role_family"], "value": r["open"]} for r in family],
        "remote_by_family": [{"label": r["role_family"], "value": r["remote"] / r["open"], "n": r["open"]}
                             for r in family if r["open"] >= 10],
        "top_companies": [{"label": r["company"], "value": r["open"]} for r in companies],
        "salary_by_family": [{"label": r["role_family"], "value": r["median_mid"], "n": r["n"]} for r in salary],
        "lifecycle_by_family": [{"label": r["role_family"], "value": float(r["median_days"]), "n": r["n"]}
                                for r in lifecycle],
        "closed_postings": closed,
        "changes": changes,
        "health": [{"source": r["source"], "last_run": str(r["started_at"])[:16], "rows": r["rows_fetched"],
                    "trailing_median": r["trailing_median"], "flagged": bool(r["flagged"]), "reason": r["reason"]}
                   for r in health],
    }


def build(con: DB, sources_path: Path, out: Path) -> dict:
    data = export(con, sources_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, indent=1, default=str))
    shutil.copy(TEMPLATE, out / "index.html")
    (out / ".nojekyll").write_text("")
    return data
