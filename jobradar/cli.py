from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import yaml

from . import db, digest, filters, recall, rerank
from .db import row_to_job
from .sources import ashby, greenhouse, lever, usajobs

log = logging.getLogger("jobradar")
SOURCES = {"usajobs": usajobs.fetch, "greenhouse": greenhouse.fetch,
           "lever": lever.fetch, "ashby": ashby.fetch}
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
    new = kept = total = 0
    for name, src_cfg in cfg["sources"].items():
        if not src_cfg.get("enabled", True) or name not in SOURCES:
            continue
        for job in SOURCES[name](src_cfg):
            total += 1
            reason = filters.check(job, cfg["filters"])
            new += db.upsert_job(con, job, reason)
            kept += reason is None
    con.commit()
    log.info("ingest: %d fetched, %d new rows, %d passed filters this run", total, new, kept)
    _update_recall(cfg, con)


def _update_recall(cfg, con):
    rows = con.execute("SELECT * FROM jobs WHERE filter_reason IS NULL").fetchall()
    if not rows:
        return
    jobs = [row_to_job(r) for r in rows]
    scores = recall.score([j.text() for j in jobs], list(cfg["resume_texts"].values()),
                          cfg.get("recall", {}).get("backend", "tfidf"))
    con.executemany("UPDATE jobs SET recall=? WHERE key=?",
                    [(float(s), r["key"]) for s, r in zip(scores, rows)])
    con.commit()


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
            con.commit()
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
    rows = con.execute(
        """SELECT j.* FROM jobs j LEFT JOIN scores s ON s.key = j.key
           WHERE j.filter_reason IS NULL AND s.key IS NULL AND COALESCE(j.recall, 0) >= ?
           ORDER BY j.recall DESC LIMIT ?""", (min_recall, top_k)).fetchall()
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

def cmd_label(cfg, con, args):
    """Stratified sample across the recall distribution, so you label jobs the
    pipeline would never show you, not just its favorites. That's what makes
    `eval` measure the pipeline instead of flattering it."""
    where = "" if args.include_filtered else "WHERE j.filter_reason IS NULL"
    rows = con.execute(
        f"""SELECT j.* FROM jobs j LEFT JOIN labels l ON l.key = j.key
            {where} {'AND' if where else 'WHERE'} l.key IS NULL
            ORDER BY COALESCE(j.recall, 0)""").fetchall()
    if not rows:
        print("Nothing left to label.")
        return
    n = min(args.n, len(rows))
    bins = [rows[i * len(rows) // n:(i + 1) * len(rows) // n] for i in range(n)]
    sample = [random.choice(b) for b in bins if b]
    random.shuffle(sample)

    for i, r in enumerate(sample, 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(sample)}] {r['title']} | {r['org']} | {r['location']}")
        print(r["url"])
        if r["filter_reason"]:
            print(f"(filtered: {r['filter_reason']})")
        print("-" * 80)
        print((r["description"] or "")[:1500])
        ans = input("\nWould you apply? [y]es / [n]o / [s]kip / [q]uit: ").strip().lower()
        if ans == "q":
            break
        if ans in ("y", "n"):
            con.execute("INSERT OR REPLACE INTO labels (key, label, labeled_at) VALUES (?,?,?)",
                        (r["key"], int(ans == "y"), db.now()))
            con.commit()


def cmd_eval(cfg, con, args, client=None):
    from sklearn.metrics import roc_auc_score

    # Filter-stage false negatives: jobs you'd apply to that stage 0 threw away.
    fn = con.execute("""SELECT j.title, j.org, j.filter_reason FROM labels l JOIN jobs j ON j.key=l.key
                        WHERE l.label=1 AND j.filter_reason IS NOT NULL""").fetchall()

    # Make sure every labeled, unfiltered job has a Claude score (costs one call each).
    missing = con.execute("""SELECT j.* FROM labels l JOIN jobs j ON j.key=l.key
                             LEFT JOIN scores s ON s.key=j.key
                             WHERE j.filter_reason IS NULL AND s.key IS NULL""").fetchall()
    if missing:
        log.info("eval: scoring %d labeled jobs that were never reranked", len(missing))
        _rerank_rows(cfg, con, missing, client)

    rows = con.execute("""SELECT l.label, j.recall, s.fit, s.recommend FROM labels l
                          JOIN jobs j ON j.key=l.key JOIN scores s ON s.key=j.key
                          WHERE j.filter_reason IS NULL""").fetchall()
    y = [r["label"] for r in rows]
    print(f"\nLabeled & unfiltered: {len(rows)}  (positives: {sum(y)})")
    if fn:
        print(f"\nFilter false negatives ({len(fn)}): loosen these rules")
        for r in fn:
            print(f"  - {r['title']} | {r['org']}  [{r['filter_reason']}]")
    if len(set(y)) < 2:
        print("Need both positive and negative labels for AUC. Label more.")
        return

    print(f"\nRecall stage AUC ({cfg.get('recall', {}).get('backend', 'tfidf')}): "
          f"{roc_auc_score(y, [r['recall'] or 0 for r in rows]):.3f}")
    print(f"Rerank fit-score AUC:   {roc_auc_score(y, [r['fit'] for r in rows]):.3f}")

    pred = [r["recommend"] == "apply" for r in rows]
    tp = sum(p and t for p, t in zip(pred, y))
    fp = sum(p and not t for p, t in zip(pred, y))
    fn_ = sum((not p) and t for p, t in zip(pred, y))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn_) if tp + fn_ else float("nan")
    print(f"'apply' recommendation: precision {prec:.2f}, recall {rec:.2f}  (tp={tp} fp={fp} fn={fn_})")

    top_k = cfg.get("recall", {}).get("top_k", 40)
    n_pass = con.execute("SELECT COUNT(*) FROM jobs WHERE filter_reason IS NULL").fetchone()[0]
    if n_pass:
        cutoff = con.execute("""SELECT recall FROM jobs WHERE filter_reason IS NULL
                                ORDER BY recall DESC LIMIT 1 OFFSET ?""",
                             (min(top_k, n_pass) - 1,)).fetchone()[0]
        lost = sum(1 for r in rows if r["label"] == 1 and (r["recall"] or 0) < cutoff)
        print(f"Positives below today's top-{top_k} recall cutoff: {lost}/{sum(y)} "
              f"(these would never reach Claude in a single batch)")


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
    lp = sub.add_parser("label")
    lp.add_argument("-n", type=int, default=30)
    lp.add_argument("--include-filtered", action="store_true",
                    help="also sample filtered-out jobs, to catch over-aggressive filters")
    pp = sub.add_parser("probe", help="check a job board token before adding it to config.yaml")
    pp.add_argument("source", choices=sorted(BOARDS))
    pp.add_argument("token")
    sub.add_parser("probe-greenhouse").add_argument("token")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    cfg = load_config(args.config)
    con = db.connect(cfg["db_path"])
    {"ingest": cmd_ingest, "score": cmd_score, "digest": cmd_digest, "run": cmd_run,
     "label": cmd_label, "eval": cmd_eval, "probe": cmd_probe, "probe-greenhouse": cmd_probe}[args.cmd](cfg, con, args)


if __name__ == "__main__":
    main()
