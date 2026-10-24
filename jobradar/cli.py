from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

import yaml

from . import db, digest, filters, labeling, recall, rerank
from .db import row_to_job
from .sources import ashby, greenhouse, lever, usajobs

log = logging.getLogger("jobradar")
# Raw fetchers yield (board, board name, payload); parsers turn a payload into a Job.
SOURCES = {"usajobs": usajobs.fetch_raw, "greenhouse": greenhouse.fetch_raw,
           "lever": lever.fetch_raw, "ashby": ashby.fetch_raw}
PARSERS = {"usajobs": usajobs.parse, "greenhouse": greenhouse.parse,
           "lever": lever.parse, "ashby": ashby.parse}
BOARDS = {"greenhouse": greenhouse.fetch_board, "lever": lever.fetch_board, "ashby": ashby.fetch_board}


# ---------- config ----------

def load_config(path: str) -> dict:
    p = Path(path).resolve()
    cfg = yaml.safe_load(p.read_text())
    base = p.parent
    cfg["_base"] = base
    if "sources" not in cfg:  # public source list lives in its own committed file
        cfg["sources"] = yaml.safe_load((base / cfg.get("sources_path", "sources.yaml")).read_text())["sources"]
    cfg["db_path"] = str(base / cfg.get("db_path", "data/jobradar.duckdb"))
    cfg["labels_db_path"] = str(base / cfg.get("labels_db_path", "data/labels.duckdb"))
    cfg["rubric_text"] = (base / cfg["rubric_path"]).read_text()
    cfg["resume_texts"] = {k: (base / v).read_text() for k, v in cfg["resumes"].items()}
    cfg.setdefault("digest", {})["out_dir"] = str(base / cfg["digest"].get("out_dir", "digests"))
    # Older configs kept the usajobs grade and hiring-path rules under sources.usajobs.
    cfg["filters"].setdefault("usajobs", cfg["sources"].get("usajobs", {}))
    return cfg


# ---------- stages ----------

def cmd_ingest(cfg, con, args=None):
    run_id = uuid.uuid4().hex[:12]
    new = kept = total = 0
    for name, src_cfg in cfg["sources"].items():
        if not src_cfg.get("enabled", True) or name not in SOURCES:
            continue
        started = db.utcnow()
        raw, parsed = [], []
        for board, board_name, item in SOURCES[name](src_cfg):
            job = PARSERS[name](item, board, board_name)
            raw.append((job.source_id, board, board_name, item))
            parsed.append((job, filters.check(job, cfg["filters"])))
        new_ids = db.upsert_raw(con, name, raw, run_id)
        new += db.upsert_jobs(con, parsed)
        db.log_run(con, run_id, name, started, len(raw), new_ids)
        total += len(raw)
        kept += sum(reason is None for _, reason in parsed)
        log.info("ingest %s: %d fetched, %d new source ids", name, len(raw), new_ids)
    log.info("ingest: %d fetched, %d new rows, %d passed filters this run", total, new, kept)
    _update_recall(cfg, con)


def _update_recall(cfg, con):
    rows = con.query("SELECT * FROM jobs WHERE filter_reason IS NULL ORDER BY key")
    if not rows:
        return
    jobs = [row_to_job(r) for r in rows]
    scores = recall.score([j.text() for j in jobs], list(cfg["resume_texts"].values()),
                          cfg.get("recall", {}).get("backend", "tfidf"))
    db.set_recall(con, [(r["key"], float(s)) for s, r in zip(scores, rows)])


def _client_and_prompt(cfg):
    import anthropic
    client = anthropic.Anthropic()
    system = rerank.build_system(cfg["rubric_text"], cfg["resume_texts"])
    tool = rerank.build_tool(list(cfg["resume_texts"].keys()))
    return client, system, tool


def _rerank_rows(cfg, con, rows, client=None):
    if not rows:
        return 0
    rcfg = cfg.get("rerank", {})
    model = rcfg.get("model", "claude-haiku-4-5-20251001")
    if client is None:
        client, system, tool = _client_and_prompt(cfg)
    else:
        system = rerank.build_system(cfg["rubric_text"], cfg["resume_texts"])
        tool = rerank.build_tool(list(cfg["resume_texts"].keys()))
    done, last_err = 0, None
    for r in rows:
        try:
            result = rerank.score_job(client, model, system, tool, row_to_job(r),
                                      rcfg.get("max_desc_chars", 8000))
            db.save_score(con, r["key"], result, model)
            done += 1
            log.info("scored %s | %s -> %s/10 %s", r["title"], r["org"], result["fit_score"], result["recommend"])
        except Exception as e:  # keep going; one bad posting shouldn't kill the run
            # ...but a bad key or account fails every call the same way, so stop now.
            if getattr(e, "status_code", None) in (401, 403):
                raise RuntimeError(f"rerank stopped: Anthropic rejected the request ({e})") from e
            log.error("rerank failed for %s: %s", r["key"], e)
            last_err = e
    if done == 0:
        # Every call failed. Report the run as failed rather than writing an empty digest.
        raise RuntimeError(f"rerank failed for all {len(rows)} jobs; last error: {last_err}")
    return done


def cmd_score(cfg, con, args=None, client=None):
    top_k = cfg.get("recall", {}).get("top_k", 40)
    min_recall = cfg.get("recall", {}).get("min_recall", 0.0)
    rows = con.query(
        """SELECT j.* FROM jobs j LEFT JOIN scores s ON s.key = j.key
           WHERE j.filter_reason IS NULL AND s.key IS NULL AND COALESCE(j.recall, 0) >= ?
           ORDER BY j.recall DESC, j.key LIMIT ?""", (min_recall, top_k))
    n = _rerank_rows(cfg, con, rows, client)
    log.info("score: reranked %d jobs", n)


def cmd_digest(cfg, con, args=None):
    path = digest.write(con, cfg["digest"])
    print(path or "No new roles above threshold.")


def cmd_run(cfg, con, args=None):
    cmd_ingest(cfg, con)
    cmd_score(cfg, con)
    cmd_digest(cfg, con)


# ---------- calibration ----------

def cmd_label_sample(cfg, con, args):
    """Draw the frozen label sets once. Recall is refreshed first so the deciles match today's pool."""
    _update_recall(cfg, con)
    lab = labeling.connect(cfg["labels_db_path"])
    n = labeling.create_sets(con, lab, args.seed, cfg.get("recall", {}).get("backend", "tfidf"))
    print(f"Drew label sets from {n['pool']} postings that passed stage 0: "
          f"a={n['a']}, b={n['b']}, c={n['c']} (c stays locked until 7 days after a is done).")


def cmd_label(cfg, con, args):
    """Blind labeling: title, company, location, excerpt and link only."""
    lab = labeling.connect(cfg["labels_db_path"])
    sets = args.set.split(",") if args.set else (["a", "b"] if args.include_filtered else ["a"])
    labeling.label_loop(lab, sets, args.n)


def cmd_label_status(cfg, con, args):
    print("\n".join(labeling.status(labeling.connect(cfg["labels_db_path"]))))


def cmd_eval(cfg, con, args, client=None):
    """Quick check on whatever is labeled so far. Spends nothing: postings without a cached
    Claude score are left out of the rerank line."""
    from sklearn.metrics import roc_auc_score

    lab = labeling.connect(cfg["labels_db_path"])
    rows = lab.query("""SELECT i.set_name, i.job_key, i.title, i.org, i.filter_reason, i.recall, l.label
                        FROM labels l JOIN label_items i USING (set_name, source, source_id)
                        WHERE l.set_name IN ('a', 'b')""")
    fn = [r for r in rows if r["set_name"] == "b" and r["label"] == "apply"]
    a = [r for r in rows if r["set_name"] == "a"]
    y = [int(r["label"] == "apply") for r in a]
    print(f"\nSet a labeled: {len(a)}  (apply: {sum(y)})")
    if fn:
        print(f"\nFilter false negatives ({len(fn)}): loosen these rules")
        for r in fn:
            print(f"  - {r['title']} | {r['org']}  [{r['filter_reason']}]")
    if len(set(y)) < 2:
        print("Need both apply and non-apply labels for AUC. Label more.")
        return
    print(f"Recall stage AUC (frozen scores): {roc_auc_score(y, [r['recall'] for r in a]):.3f}")

    scores = {r["key"]: r["fit"] for r in con.query("SELECT key, fit FROM scores")}
    scored = [(t, scores[r["job_key"]]) for t, r in zip(y, a) if r["job_key"] in scores]
    if len({t for t, _ in scored}) == 2:
        print(f"Rerank fit-score AUC (cached scores only, n={len(scored)}): "
              f"{roc_auc_score([t for t, _ in scored], [f for _, f in scored]):.3f}")
    else:
        print(f"Rerank: {len(scored)} labeled postings have cached scores; not enough for AUC.")


def cmd_probe(cfg, con, args):
    jobs = list(BOARDS[getattr(args, "source", "greenhouse")](args.token))
    passing = [j for j in jobs if filters.check(j, cfg["filters"]) is None]
    print(f"{args.token}: {len(jobs)} open jobs, {len(passing)} pass filters")
    for j in (passing or jobs)[:10]:
        print(f"  - {j.title} | {j.location}")


# ---------- entry ----------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="jobradar")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "score", "digest", "run", "eval"):
        sub.add_parser(name)
    sp = sub.add_parser("label-sample", help="draw the frozen label sets a, b, c (once)")
    sp.add_argument("--seed", type=int, default=20261023)
    lp = sub.add_parser("label", help="blind labeling of a frozen set")
    lp.add_argument("--set", help="comma-separated: a, b, a,b or c (default a)")
    lp.add_argument("-n", type=int, default=None, help="stop after this many labels")
    lp.add_argument("--include-filtered", action="store_true",
                    help="same as --set a,b: mix in postings stage 0 dropped")
    sub.add_parser("label-status")
    pp = sub.add_parser("probe", help="check a job board token before adding it to config.yaml")
    pp.add_argument("source", choices=sorted(BOARDS))
    pp.add_argument("token")
    sub.add_parser("probe-greenhouse").add_argument("token")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    cfg = load_config(args.config)
    # Labeling only touches the labels file, so a long session never holds the main DB's
    # write lock (DuckDB allows one writer) while the daily run needs it.
    con = None if args.cmd in ("label", "label-status") else db.connect(cfg["db_path"])
    {"ingest": cmd_ingest, "score": cmd_score, "digest": cmd_digest, "run": cmd_run,
     "label": cmd_label, "label-sample": cmd_label_sample, "label-status": cmd_label_status,
     "eval": cmd_eval, "probe": cmd_probe, "probe-greenhouse": cmd_probe}[args.cmd](cfg, con, args)


if __name__ == "__main__":
    main()
