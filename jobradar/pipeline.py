"""The public ELT pipeline: raw ingest into DuckDB, data quality checks, near-duplicate
clustering and the dashboard export. Needs only sources.yaml, never the personal config,
so it runs unattended in GitHub Actions.

    python -m jobradar.pipeline ingest   --db data/warehouse.duckdb
    python -m jobradar.pipeline quality  --db data/warehouse.duckdb
    (cd warehouse && dbt build)          # staging -> intermediate -> marts, with tests
    python -m jobradar.pipeline dashboard --db data/warehouse.duckdb --out site
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import yaml

from . import db
from .db import DB, utcnow
from .models import Job
from .sources import ashby, greenhouse, lever, usajobs

log = logging.getLogger("jobradar.pipeline")
FETCHERS = {"usajobs": usajobs.fetch_raw, "greenhouse": greenhouse.fetch_raw,
            "lever": lever.fetch_raw, "ashby": ashby.fetch_raw}
PARSERS = {"usajobs": usajobs.parse, "greenhouse": greenhouse.parse,
           "lever": lever.parse, "ashby": ashby.parse}

QUALITY_WINDOW = 7       # trailing runs the median is taken over
QUALITY_MIN_HISTORY = 3  # fewer earlier runs than this: too little history to judge
QUALITY_DROP = 0.5       # flag when a source returns less than half its trailing median


def ingest_raw(con: DB, sources: dict, fetchers: dict[str, Callable] | None = None,
               run_id: str | None = None, at: datetime | None = None) -> tuple[str, dict[str, list[Job]]]:
    """Fetch every enabled source, upsert its raw payloads and log the run.
    Returns the run id and the parsed jobs per source (the ranking pipeline filters those)."""
    fetchers = fetchers or FETCHERS
    run_id = run_id or uuid.uuid4().hex[:12]
    out = {}
    for name, src_cfg in sources.items():
        if not src_cfg.get("enabled", True) or name not in fetchers:
            continue
        started = at or utcnow()
        raw, jobs = [], []
        for board, board_name, item in fetchers[name](src_cfg):
            job = PARSERS[name](item, board, board_name)
            raw.append((job.source_id, board, board_name, item))
            jobs.append(job)
        new = db.upsert_raw(con, name, raw, run_id, at=at)
        db.log_run(con, run_id, name, started, len(raw), new, finished=at)
        log.info("ingest %s: %d fetched, %d new source ids", name, len(raw), new)
        out[name] = jobs
    return run_id, out


def check_quality(con: DB, run_id: str | None = None) -> list[dict]:
    """Compare each source's row count in a run with the median of its previous runs (up to 7).
    Flags a drop below half the median, and a source with history that logged nothing."""
    if run_id is None:
        run_id = con.scalar("SELECT run_id FROM ingest_runs ORDER BY started_at DESC LIMIT 1")
        if run_id is None:
            return []
    started = con.scalar("SELECT min(started_at) FROM ingest_runs WHERE run_id=?", (run_id,))
    this = {r["source"]: r["rows_fetched"] for r in
            con.query("SELECT source, rows_fetched FROM ingest_runs WHERE run_id=?", (run_id,))}
    history = {}
    for r in con.query("""SELECT source, rows_fetched FROM (
                              SELECT *, row_number() OVER (PARTITION BY source ORDER BY started_at DESC) rn
                              FROM ingest_runs WHERE started_at < ? AND run_id != ?)
                          WHERE rn <= ? ORDER BY source""", (started, run_id, QUALITY_WINDOW)):
        history.setdefault(r["source"], []).append(r["rows_fetched"])
    checks = []
    for source in sorted(set(this) | set(history)):
        prev = history.get(source, [])
        n = this.get(source)
        med = statistics.median(prev) if prev else None
        if n is None:
            flagged, reason = True, "no rows logged this run (source disabled or the run crashed)"
        elif len(prev) < QUALITY_MIN_HISTORY:
            flagged, reason = False, f"only {len(prev)} earlier runs; need {QUALITY_MIN_HISTORY} to judge"
        elif n < QUALITY_DROP * med:
            flagged, reason = True, f"{n} rows vs trailing median {med:g}: likely an API change or outage"
        else:
            flagged, reason = False, "ok"
        checks.append({"run_id": run_id, "source": source, "checked_at": utcnow(), "rows_fetched": n,
                       "trailing_runs": len(prev), "trailing_median": med,
                       "ratio": (n / med) if (n is not None and med) else None,
                       "flagged": flagged, "reason": reason})
    con.execute("DELETE FROM quality_checks WHERE run_id=?", (run_id,))
    con.insert_rows("quality_checks", [tuple(c.values()) for c in checks])
    return checks


# ---------- fixtures (CI) ----------

def load_fixture_runs(con: DB, path: Path) -> int:
    """Replay a committed synthetic fixture: several daily runs of API responses, each loaded
    through the same upsert path with its load time set to N days ago."""
    runs = json.loads(Path(path).read_text())
    now = utcnow()
    for i, run in enumerate(runs):
        at = now - timedelta(days=run["days_ago"])
        resp = run["responses"]

        def fetcher(name):
            def f(cfg):
                if name == "usajobs":
                    for item in resp.get("usajobs", []):
                        yield "fixture|", None, item
                else:
                    for token, board in resp.get(name, {}).items():
                        for item in board["jobs"]:
                            yield token, board["name"], item
            return f
        sources = {s: {"enabled": True} for s in FETCHERS}
        ingest_raw(con, sources, {s: fetcher(s) for s in FETCHERS}, run_id=f"fixture{i:02d}", at=at)
        check_quality(con, f"fixture{i:02d}")
    return len(runs)


# ---------- entry ----------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="jobradar.pipeline")
    ap.add_argument("--db", default="data/warehouse.duckdb")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ip = sub.add_parser("ingest", help="fetch every source in sources.yaml into the raw tables")
    ip.add_argument("--sources", default="sources.yaml")
    sub.add_parser("quality", help="flag sources whose row count dropped versus their trailing median")
    fp = sub.add_parser("load-fixtures", help="build a warehouse from the committed synthetic fixture")
    fp.add_argument("--path", default="tests/fixtures/warehouse/runs.json")
    sub.add_parser("init", help="create empty warehouse tables")
    dp = sub.add_parser("dedupe-candidates", help="draw 50 same-company pairs across the similarity range to mark")
    dp.add_argument("--out", default="data/dedupe_pairs.csv")
    mp = sub.add_parser("dedupe-mark", help="mark candidate pairs same/different (similarity hidden)")
    mp.add_argument("--pairs", default="data/dedupe_pairs.csv")
    ep = sub.add_parser("dedupe-eval", help="pick the threshold from the marks; precision, recall, share collapsed")
    ep.add_argument("--pairs", default="data/dedupe_pairs.csv")
    ap_ = sub.add_parser("dedupe", help="cluster postings at the threshold in sources.yaml into job_clusters")
    ap_.add_argument("--sources", default="sources.yaml")
    ap_.add_argument("--threshold", type=float, default=None)
    bp = sub.add_parser("dashboard", help="export the marts to a static site (aggregates only)")
    bp.add_argument("--sources", default="sources.yaml")
    bp.add_argument("--out", default="site")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    if args.cmd == "dedupe-mark":  # works on the CSV only
        from . import dedupe
        print(f"marked {dedupe.mark_loop(Path(args.pairs))} pairs")
        return
    con = db.connect(args.db, app_tables=False)
    if args.cmd == "ingest":
        sources = yaml.safe_load(Path(args.sources).read_text())["sources"]
        run_id, _ = ingest_raw(con, sources)
        print(run_id)
    elif args.cmd == "quality":
        checks = check_quality(con)
        for c in checks:
            line = f"{c['source']}: {c['rows_fetched']} rows, trailing median {c['trailing_median']}, {c['reason']}"
            # GitHub Actions shows ::warning:: lines as annotations on the run.
            print(f"::warning title=Data quality::{line}" if c["flagged"] else line)
    elif args.cmd == "load-fixtures":
        print(f"loaded {load_fixture_runs(con, Path(args.path))} fixture runs into {args.db}")
    elif args.cmd == "dedupe-candidates":
        from . import dedupe
        if Path(args.out).exists():
            sys.exit(f"{args.out} exists; candidates are drawn once so marks stay valid")
        jobs = dedupe.load_jobs(con)
        E = dedupe.embed(jobs)
        pairs = dedupe.company_pairs(jobs, E, dedupe.LOW)
        rows = dedupe.sample_candidates(jobs, pairs)
        dedupe.write_pairs(Path(args.out), rows)
        print(f"{len(jobs)} open postings, {len(pairs)} same-company pairs with similarity >= {dedupe.LOW}; "
              f"wrote {len(rows)} candidates to {args.out}")
    elif args.cmd == "dedupe-eval":
        from . import dedupe
        res = dedupe.evaluate(dedupe.read_pairs(Path(args.pairs)))
        if res["status"] != "ok":
            print(f"pending: {res['reason']}")
        else:
            c = res["chosen"]
            jobs = dedupe.load_jobs(con)
            mapping = dedupe.cluster(jobs, dedupe.embed(jobs), c["threshold"])
            exact = {j["job_id"]: j["exact_key"] for j in jobs}
            res.update(postings=len(jobs), collapsed_share=dedupe.collapsed_share(mapping),
                       collapsed_share_exact_only=dedupe.collapsed_share(exact))
            out = Path("results/dedupe_eval.json")
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(res, indent=2))
            print(f"{res['marked']} pairs marked ({res['same']} same). Threshold {c['threshold']:.4f} ({res['rule']}): "
                  f"weighted precision {c['precision_w']:.3f}, weighted recall {c['recall_w']:.3f} "
                  f"(raw tp={c['tp']} fp={c['fp']} fn={c['fn']}). Collapses {res['collapsed_share']:.1%} of "
                  f"{len(jobs)} open postings ({res['collapsed_share_exact_only']:.1%} from exact matches alone). "
                  f"Wrote {out}.")
    elif args.cmd == "dashboard":
        from . import dashboard
        d = dashboard.build(con, Path(args.sources), Path(args.out))
        print(f"wrote {args.out}/index.html and data.json: {d['tiles']['open_postings']} open postings")
    elif args.cmd == "dedupe":
        from . import dedupe
        t = args.threshold
        if t is None:
            t = (yaml.safe_load(Path(args.sources).read_text()).get("dedupe") or {}).get("threshold")
        if t is None:
            dedupe.write_clusters(con, {}, None)
            print("dedupe threshold not set yet (pending hand-marked pairs); exact matches only")
        else:
            jobs = dedupe.load_jobs(con, open_only=False)
            mapping = dedupe.cluster(jobs, dedupe.embed(jobs), t)
            dedupe.write_clusters(con, mapping, t)
            print(f"clustered {len(jobs)} postings at {t}: {dedupe.collapsed_share(mapping):.1%} collapsed")
    con.close()


if __name__ == "__main__":
    main()
