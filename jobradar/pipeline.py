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
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

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


def check_quality(con: DB, run_id: str | None = None, sources: dict | None = None) -> list[dict]:
    """Compare each source's row count in a run with the median of its previous runs (up to 7).
    Flags a drop below half the median, a run that returned no rows at all (an enabled source
    is never legitimately empty, and a source stuck at zero would otherwise drag its own median
    to zero and read as fine), and a source with history that logged nothing. With `sources`
    (the sources.yaml mapping), also flags each configured board that loaded nothing."""
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
        elif n == 0:
            flagged, reason = True, "returned no rows: missing credentials, an API change or an outage"
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
    if sources:
        checks += _empty_boards(con, run_id, sources, {c["source"] for c in checks if not c["flagged"]})
    con.execute("DELETE FROM quality_checks WHERE run_id=?", (run_id,))
    con.insert_rows("quality_checks", [tuple(c.values()) for c in checks])
    return checks


def _empty_boards(con: DB, run_id: str, sources: dict, healthy: set[str]) -> list[dict]:
    """A check per configured board that loaded nothing this run, as "<source>/<token>". A source's
    total hides one empty board (Overstory's Greenhouse board sat empty after it moved to Ashby).
    An empty board looks the same whether the company moved or just has no openings, so only the
    run where a board goes from postings to none is flagged: one alert, not one a day. Only for
    sources that are fine overall, so an outage is one flag."""
    out = []
    started = con.scalar("SELECT min(started_at) FROM ingest_runs WHERE run_id=?", (run_id,))
    for source in sorted(healthy & set(db.RAW_SOURCES)):
        tokens = [b["token"] for b in sources.get(source, {}).get("boards", [])
                  if not b["token"].startswith("REPLACE")]
        prev = con.scalar("""SELECT run_id FROM ingest_runs WHERE source=? AND started_at < ? AND run_id != ?
                             ORDER BY started_at DESC LIMIT 1""", (source, started, run_id))
        # A load moves a posting's run_id forward, so rows still on prev are the ones that left this run.
        by_run = {}
        for r in con.query(f"SELECT DISTINCT board, run_id FROM raw_{source} WHERE run_id IN (?, ?)",
                           (run_id, prev or run_id)):
            by_run.setdefault(r["run_id"], set()).add(r["board"])
        for t in tokens:
            if t in by_run.get(run_id, set()):
                continue
            went_empty = prev is not None and t in by_run.get(prev, set())
            out.append({"run_id": run_id, "source": f"{source}/{t}", "checked_at": utcnow(), "rows_fetched": 0,
                        "trailing_runs": None, "trailing_median": None, "ratio": None, "flagged": went_empty,
                        "reason": "board went empty this run: renamed, moved to another job board, or no openings"
                        if went_empty else "board loaded no postings (not newly: check the token if it never has)"})
    return out


# ---------- backup ----------

BACKUP_TABLES = [f"raw_{s}" for s in db.RAW_SOURCES] + ["ingest_runs", "quality_checks", "job_clusters"]
SNAPSHOT_TABLE = "snapshots.snap_postings"


def export_parquet(con: DB, out: Path) -> dict:
    """Everything that can't be rebuilt from the APIs: raw tables, run logs, clusters and the
    snapshot history. dbt models are rebuilt from these, so they aren't exported."""
    out.mkdir(parents=True, exist_ok=True)
    tables = list(BACKUP_TABLES)
    if con.scalar("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='snapshots' "
                  "AND table_name='snap_postings'"):
        tables.append(SNAPSHOT_TABLE)
    counts = {}
    for t in tables:
        path = out / f"{t.replace('.', '__')}.parquet"
        con.execute(f"COPY (SELECT * FROM {t}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
        counts[t] = con.scalar(f"SELECT COUNT(*) FROM {t}")
    manifest = {"exported_at": utcnow().isoformat() + "Z", "rows": counts}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def restore_parquet(con: DB, src: Path) -> dict:
    """Load an export into a warehouse whose tables are empty. Refuses to overwrite data."""
    manifest = json.loads((src / "manifest.json").read_text())
    for t in BACKUP_TABLES:
        if con.scalar(f"SELECT COUNT(*) FROM {t}"):
            raise RuntimeError(f"{t} already has rows; restore needs an empty warehouse")
    for t in manifest["rows"]:
        path = src / f"{t.replace('.', '__')}.parquet"
        if t == SNAPSHOT_TABLE:
            con.execute("CREATE SCHEMA IF NOT EXISTS snapshots")
            con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{path}')")
        else:
            con.execute(f"INSERT INTO {t} BY NAME SELECT * FROM read_parquet('{path}')")
        got = con.scalar(f"SELECT COUNT(*) FROM {t}")
        if got != manifest["rows"][t]:
            raise RuntimeError(f"{t}: restored {got} rows, manifest says {manifest['rows'][t]}")
    return manifest


# ---------- fixtures (CI) ----------

def load_fixture_runs(con: DB, path: Path) -> int:
    """Replay a committed synthetic fixture: several daily runs of API responses, each loaded
    through the same upsert path with its load time set to N days ago."""
    runs = json.loads(Path(path).read_text())
    now = utcnow()
    for i, run in enumerate(runs):
        at = now - timedelta(days=run["days_ago"])
        resp = run["responses"]

        def fetcher(name, resp=resp):  # bind this run's responses, not the loop variable
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
    qp = sub.add_parser("quality", help="flag sources whose row count dropped versus their trailing median, "
                                         "and boards that loaded nothing")
    qp.add_argument("--flags-out", help="write the flagged checks here as JSON (for the alerting step)")
    qp.add_argument("--sources", default="sources.yaml")
    fp = sub.add_parser("load-fixtures", help="build a warehouse from the committed synthetic fixture")
    fp.add_argument("--path", default="tests/fixtures/warehouse/runs.json")
    sub.add_parser("init", help="create empty warehouse tables")
    xp = sub.add_parser("export", help="write raw tables, run logs and snapshot history to Parquet")
    xp.add_argument("--out", default="export")
    rp = sub.add_parser("restore", help="load a Parquet export into an empty warehouse")
    rp.add_argument("--dir", default="export")
    dp = sub.add_parser("dedupe-candidates", help="draw 50 same-company pairs across the similarity range to mark")
    dp.add_argument("--out", default="data/dedupe_pairs.csv")
    mp = sub.add_parser("dedupe-mark", help="mark candidate pairs same/different (similarity hidden)")
    mp.add_argument("--pairs", default="data/dedupe_pairs.csv")
    ep = sub.add_parser("dedupe-eval", help="pick the threshold from the marks; precision, recall, share collapsed")
    ep.add_argument("--pairs", nargs="+", default=["data/dedupe_pairs.csv"])
    ep.add_argument("--rule", choices=["embedding", "company_title"], default="embedding")
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
        checks = check_quality(con, sources=yaml.safe_load(Path(args.sources).read_text())["sources"])
        for c in checks:
            line = f"{c['source']}: {c['rows_fetched']} rows, trailing median {c['trailing_median']}, {c['reason']}"
            # GitHub Actions shows ::warning:: lines as annotations on the run.
            print(f"::warning title=Data quality::{line}" if c["flagged"] else line)
        if args.flags_out:
            Path(args.flags_out).write_text(json.dumps(
                [{k: c[k] for k in ("run_id", "source", "rows_fetched", "trailing_median", "reason")}
                 for c in checks if c["flagged"]], indent=2))
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
        if args.rule == "company_title":
            jobs = dedupe.load_jobs(con)
            res = {"rule": "company_title", "files": {}}
            for p in args.pairs:
                res["files"][p] = dedupe.evaluate_title_rule(dedupe.read_pairs(Path(p)))
            exact = {j["job_id"]: j["exact_key"] for j in jobs}
            res.update(postings=len(jobs), collapsed_share=dedupe.collapsed_share(dedupe.cluster_by_title(jobs)),
                       collapsed_share_exact_only=dedupe.collapsed_share(exact))
            out = Path("results/dedupe_title_rule.json")
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(res, indent=2))
            for p, r in res["files"].items():
                lo = f", precision >= {r['precision_lower_95']:.3f} (95%)" if r["precision_lower_95"] else ""
                print(f"{p}: {r['marked']} marked, tp={r['tp']} fp={r['fp']} fn={r['fn']} tn={r['tn']}{lo}")
            print(f"Collapses {res['collapsed_share']:.1%} of {len(jobs)} open postings "
                  f"({res['collapsed_share_exact_only']:.1%} from exact matches alone). Wrote {out}.")
            return
        # rule: embedding, the rejected method; reproduces results/dedupe_eval.json.
        res = dedupe.evaluate(dedupe.read_pairs(Path(args.pairs[0])))
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
    elif args.cmd == "export":
        m = export_parquet(con, Path(args.out))
        print(f"exported {sum(m['rows'].values())} rows from {len(m['rows'])} tables to {args.out}")
    elif args.cmd == "restore":
        m = restore_parquet(con, Path(args.dir))
        print(f"restored {sum(m['rows'].values())} rows from the export made {m['exported_at']}")
    elif args.cmd == "dashboard":
        from . import dashboard
        d = dashboard.build(con, Path(args.sources), Path(args.out))
        print(f"wrote {args.out}/index.html and data.json: {d['tiles']['open_postings']} open postings")
    elif args.cmd == "dedupe":
        from . import dedupe
        dcfg = yaml.safe_load(Path(args.sources).read_text()).get("dedupe") or {}
        t = args.threshold
        if t is None and dcfg.get("rule") == "company_title":
            jobs = dedupe.load_jobs(con, open_only=False)
            mapping = dedupe.cluster_by_title(jobs)
            dedupe.write_clusters(con, mapping, None)
            print(f"clustered {len(jobs)} postings by company + title: {dedupe.collapsed_share(mapping):.1%} collapsed")
            return
        # rule: embedding. Rejected by the hand marks (results/dedupe_eval.json); kept so the
        # comparison can be rerun, not for production use.
        if t is None:
            t = dcfg.get("threshold")
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
