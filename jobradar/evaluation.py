"""Phase 2 evaluation: score the frozen label sets with each ranker and measure them
against the hand labels.

Rankers
  tfidf      stage 1 as it ran when the sets were drawn (frozen in labels.pool)
  emb:<m>    local sentence-transformers similarity, max over resume variants
  claude:<m> stage 2 fit score (0-10) from the production prompt, scored on the frozen text

Everything here reads and writes data/labels.duckdb, which is gitignored. Only aggregate
numbers leave it, through results/.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass

import numpy as np

from . import rerank
from .db import DB, utcnow
from .models import Job

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS ranker_scores (
    ranker VARCHAR, job_key VARCHAR, score DOUBLE, created_at TIMESTAMP,
    PRIMARY KEY (ranker, job_key)
);
CREATE TABLE IF NOT EXISTS eval_scores (
    model VARCHAR, run INTEGER, job_key VARCHAR,
    fit INTEGER, recommend VARCHAR, result_json VARCHAR,
    input_tokens INTEGER, output_tokens INTEGER,
    cache_creation_input_tokens INTEGER, cache_read_input_tokens INTEGER,
    cost_usd DOUBLE,               -- what this call actually cost (batch discount applied)
    origin VARCHAR,                -- batch | sync | production_cache
    scored_at TIMESTAMP,
    PRIMARY KEY (model, run, job_key)
);
CREATE TABLE IF NOT EXISTS eval_failures (
    model VARCHAR, run INTEGER, job_key VARCHAR, error VARCHAR, cost_usd DOUBLE, failed_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS eval_batches (
    batch_id VARCHAR PRIMARY KEY, model VARCHAR, run INTEGER, n_requests INTEGER,
    created_at TIMESTAMP, ended_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS eval_prompt_tokens (
    model VARCHAR PRIMARY KEY, system_tokens INTEGER, counted_at TIMESTAMP
);
"""

STABILITY_N = 50
STABILITY_SEED = 50
DEFAULT_BUDGET_USD = 5.0


def init(lab: DB) -> DB:
    lab.con.execute(SCHEMA)
    return lab


def item_job(r: dict) -> Job:
    return Job(source=r["source"], source_id=r["source_id"], title=r["title"], org=r["org"],
               location=r["location"], url=r["url"], description=r["description"],
               extra=json.loads(r["extra"] or "{}"))


# ---------- which postings each run scores ----------

def items_for_run(lab: DB, run: int) -> list[dict]:
    """Run 1 scores all of set a. Run 2 rescores a fixed random 50 of them, for stability."""
    items = lab.query("SELECT * FROM label_items WHERE set_name='a' ORDER BY source, source_id")
    if run == 1:
        return items
    return random.Random(STABILITY_SEED + run).sample(items, min(STABILITY_N, len(items)))


def pending_items(lab: DB, model: str, run: int) -> list[dict]:
    done = {r["job_key"] for r in lab.query("SELECT job_key FROM eval_scores WHERE model=? AND run=?",
                                            (model, run))}
    return [r for r in items_for_run(lab, run) if r["job_key"] not in done]


def total_spend(lab: DB) -> float:
    a = lab.scalar("SELECT COALESCE(SUM(cost_usd), 0) FROM eval_scores") or 0.0
    b = lab.scalar("SELECT COALESCE(SUM(cost_usd), 0) FROM eval_failures") or 0.0
    return float(a) + float(b)


# ---------- reuse of production scores ----------

def reuse_production_scores(lab: DB, main: DB, model: str, prompt_changed_at) -> int:
    """Copy daily-run scores for set a postings when the same model scored them after the
    rubric and resumes last changed. These cost nothing now; their tokens weren't logged."""
    rows = main.query("""SELECT s.*, j.source, j.source_id FROM scores s JOIN jobs j USING (key)
                         WHERE s.model=? AND s.scored_at >= ?""", (model, prompt_changed_at.isoformat()))
    # Match on source id: the label sets keep the key the posting had when they were drawn.
    have = {(r["source"], r["source_id"]): r["job_key"] for r in items_for_run(lab, 1)}
    done = {r["job_key"] for r in lab.query("SELECT job_key FROM eval_scores WHERE model=? AND run=1", (model,))}
    n = 0
    for r in rows:
        key = have.get((r["source"], r["source_id"]))
        if key and key not in done:
            result = {"fit_score": r["fit"], "recommend": r["recommend"], "resume_variant": r["variant"],
                      "strongest_evidence": r["evidence"], "gaps": json.loads(r["gaps"] or "[]"),
                      "dealbreakers": json.loads(r["dealbreakers"] or "[]"), "pitch": r["pitch"]}
            _save(lab, model, 1, key, result, {}, 0.0, "production_cache")
            done.add(key)
            n += 1
    return n


def _save(lab, model, run, key, result, usage, cost, origin):
    lab.execute("INSERT OR REPLACE INTO eval_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (model, run, key, result["fit_score"], result["recommend"], json.dumps(result),
                 usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                 usage.get("cache_creation_input_tokens", 0), usage.get("cache_read_input_tokens", 0),
                 cost, origin, utcnow()))


# ---------- cost estimate ----------

@dataclass
class Estimate:
    n: int
    system_tokens: int                 # rubric + resumes, the part production caches
    mean_input_tokens: float           # whole request: system + tool + posting
    mean_output_tokens: float | None   # None until real calls have measured it
    batch: bool
    model: str

    @property
    def usd(self) -> float | None:
        if self.mean_output_tokens is None:
            return None
        # Conservative: assumes no cache hits (batch caching is best effort).
        per = rerank.cost_usd(self.model, {"input_tokens": self.mean_input_tokens,
                                           "output_tokens": self.mean_output_tokens}, batch=self.batch)
        return per * self.n


def count_system_tokens(client, lab: DB, model: str, system, tool) -> int:
    cached = lab.scalar("SELECT system_tokens FROM eval_prompt_tokens WHERE model=?", (model,))
    if cached:
        return cached
    probe = Job("x", "x", "x", "x", "x", "x", "x")
    p = rerank.request_params(model, system, tool, probe, 10)
    with_sys = client.messages.count_tokens(model=model, system=system, tools=p["tools"],
                                            tool_choice=p["tool_choice"], messages=p["messages"]).input_tokens
    no_sys = client.messages.count_tokens(model=model, tools=p["tools"], tool_choice=p["tool_choice"],
                                          messages=p["messages"]).input_tokens
    n = with_sys - no_sys
    lab.execute("INSERT OR REPLACE INTO eval_prompt_tokens VALUES (?,?,?)", (model, n, utcnow()))
    return n


def estimate(client, lab: DB, model: str, items: list[dict], system, tool, max_chars: int,
             batch: bool, sample: int = 20) -> Estimate:
    """Token counts come from the free count_tokens endpoint; output tokens from earlier real calls."""
    sys_tokens = count_system_tokens(client, lab, model, system, tool)
    totals = []
    for r in random.Random(0).sample(items, min(sample, len(items))):
        p = rerank.request_params(model, system, tool, item_job(r), max_chars)
        totals.append(client.messages.count_tokens(model=model, system=system, tools=p["tools"],
                                                   tool_choice=p["tool_choice"], messages=p["messages"]).input_tokens)
    out = lab.scalar("SELECT AVG(output_tokens) FROM eval_scores WHERE model=? AND origin != 'production_cache'",
                     (model,))
    return Estimate(len(items), sys_tokens, float(np.mean(totals)) if totals else 0.0,
                    float(out) if out is not None else None, batch, model)


# ---------- scoring ----------

def score_sync(client, lab: DB, model: str, run: int, items: list[dict], system, tool, max_chars: int) -> int:
    done = 0
    for r in items:
        msg = None
        try:
            msg = client.messages.create(**rerank.request_params(model, system, tool, item_job(r), max_chars))
            usage = rerank.usage_dict(msg)
            result = rerank.parse_result(msg, r["job_key"])
            _save(lab, model, run, r["job_key"], result, usage, rerank.cost_usd(model, usage), "sync")
            done += 1
        except Exception as e:
            if getattr(e, "status_code", None) in (401, 403):
                raise
            cost = rerank.cost_usd(model, rerank.usage_dict(msg)) if msg is not None else 0.0
            lab.execute("INSERT INTO eval_failures VALUES (?,?,?,?,?,?)",
                        (model, run, r["job_key"], str(e)[:500], cost, utcnow()))
            log.error("eval score failed for %s: %s", r["job_key"], e)
    return done


def score_batch(client, lab: DB, model: str, run: int, items: list[dict], system, tool, max_chars: int,
                poll_seconds: int = 30) -> int:
    """Message Batches API: half price, results within 24h (usually minutes). Resumes an
    unfinished batch for this model and run instead of submitting a second one."""
    open_batch = lab.scalar("SELECT batch_id FROM eval_batches WHERE model=? AND run=? AND ended_at IS NULL",
                            (model, run))
    if open_batch:
        batch_id = open_batch
        log.info("resuming batch %s", batch_id)
    else:
        by_id = {f"r{i}": r for i, r in enumerate(items)}
        reqs = [{"custom_id": cid, "params": rerank.request_params(model, system, tool, item_job(r), max_chars)}
                for cid, r in by_id.items()]
        lab.execute("CREATE TABLE IF NOT EXISTS eval_batch_items "
                    "(batch_id VARCHAR, custom_id VARCHAR, job_key VARCHAR)")
        batch = client.messages.batches.create(requests=reqs)
        batch_id = batch.id
        lab.execute("BEGIN")  # the batch and its id map land together, or a resume can't read the results
        lab.execute("INSERT INTO eval_batches VALUES (?,?,?,?,?,NULL)", (batch_id, model, run, len(reqs), utcnow()))
        lab.insert_rows("eval_batch_items", [(batch_id, cid, r["job_key"]) for cid, r in by_id.items()])
        lab.execute("COMMIT")
        log.info("submitted batch %s with %d requests", batch_id, len(reqs))
    lab.close()  # don't hold the labels file's write lock while waiting; labeling can go on
    try:
        while client.messages.batches.retrieve(batch_id).processing_status != "ended":
            time.sleep(poll_seconds)
    finally:
        lab.reopen()
    keys = {r["custom_id"]: r["job_key"] for r in
            lab.query("SELECT custom_id, job_key FROM eval_batch_items WHERE batch_id=?", (batch_id,))}
    done = 0
    for res in client.messages.batches.results(batch_id):
        key = keys[res.custom_id]
        if res.result.type != "succeeded":
            lab.execute("INSERT INTO eval_failures VALUES (?,?,?,?,?,?)",
                        (model, run, key, f"batch result {res.result.type}", 0.0, utcnow()))
            continue
        msg = res.result.message
        usage = rerank.usage_dict(msg)
        cost = rerank.cost_usd(model, usage, batch=True)
        try:
            _save(lab, model, run, key, rerank.parse_result(msg, key), usage, cost, "batch")
            done += 1
        except Exception as e:
            lab.execute("INSERT INTO eval_failures VALUES (?,?,?,?,?,?)",
                        (model, run, key, str(e)[:500], cost, utcnow()))
    lab.execute("UPDATE eval_batches SET ended_at=? WHERE batch_id=?", (utcnow(), batch_id))
    return done


# ---------- local embeddings ----------

EMBEDDING_MODELS = {
    # Pre-registered primary: 512-token window (postings are long) and a retrieval-tuned model.
    "bge-small": "BAAI/bge-small-en-v1.5",
    # Secondary, for reference: the common default, 256-token window.
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


def embed_pool(lab: DB, resumes: list[str], name: str, encode=None) -> int:
    """Score every posting in the frozen pool. `encode` is injectable for tests."""
    texts = lab.query("SELECT job_key, title, org, location, description FROM pool ORDER BY job_key")
    if encode is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODELS[name])
        encode = lambda xs: model.encode(xs, normalize_embeddings=True, batch_size=32)  # noqa: E731
    j = np.asarray(encode([f"{r['title']}\n{r['org']}\n{r['location']}\n\n{r['description']}" for r in texts]))
    v = np.asarray(encode(resumes))
    scores = (j @ v.T).max(axis=1)
    lab.execute("DELETE FROM ranker_scores WHERE ranker=?", (f"emb:{name}",))
    lab.insert_rows("ranker_scores", [(f"emb:{name}", r["job_key"], float(s), utcnow())
                                      for r, s in zip(texts, scores, strict=True)])
    return len(texts)


# ---------- metrics ----------

def precision_curve(y: np.ndarray, s: np.ndarray, ks) -> np.ndarray:
    """Expected precision@k for each k, under random tie-breaking. Claude's fit score is an
    integer 0-10, so ties at the cutoff are common and the order within a tie is arbitrary."""
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order].astype(float)
    cum = np.concatenate([[0.0], np.cumsum(yy)])
    # Each position's tie group as [start, end).
    change = np.concatenate([[True], ss[1:] != ss[:-1]])
    starts = np.maximum.accumulate(np.where(change, np.arange(len(ss)), 0))
    ends = np.concatenate([np.flatnonzero(change)[1:], [len(ss)]])[np.cumsum(change) - 1]
    ks = np.minimum(np.asarray(ks), len(ss))
    g0, g1 = starts[ks - 1], ends[ks - 1]
    group_rate = (cum[g1] - cum[g0]) / (g1 - g0)
    return (cum[g0] + (ks - g0) * group_rate) / ks


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    return float(precision_curve(y, s, [k])[0]) if len(y) else float("nan")


def metrics(y: np.ndarray, s: np.ndarray, ks=(10, 20)) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {f"p@{k}": precision_at_k(y, s, k) for k in ks}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, s))
        out["ap"] = float(average_precision_score(y, s))
    else:
        out["auc"] = out["ap"] = float("nan")
    return out


def bootstrap(y: np.ndarray, rankers: dict[str, np.ndarray], n_boot: int = 2000, seed: int = 0,
              ks=(10, 20), curve_k: int = 50, baseline: str | None = "tfidf") -> dict:
    """Percentile 95% CIs from resampling postings with replacement. Every ranker is scored on
    the same resample, so the paired differences against the baseline are on matched samples."""
    rng = np.random.default_rng(seed)
    n = len(y)
    names = list(rankers)
    point = {m: metrics(y, rankers[m], ks) for m in names}
    curve_ks = list(range(1, min(curve_k, n) + 1))
    point_curve = {m: precision_curve(y, rankers[m], curve_ks).tolist() for m in names}
    samples = {m: {k: [] for k in point[m]} for m in names}
    curves = {m: [] for m in names}
    diffs = {m: {k: [] for k in point[m]} for m in names if baseline in rankers and m != baseline}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        mb = {m: metrics(yb, rankers[m][idx], ks) for m in names}
        for m in names:
            for k, v in mb[m].items():
                samples[m][k].append(v)
            curves[m].append(precision_curve(yb, rankers[m][idx], curve_ks))
        for m in diffs:
            for k in diffs[m]:
                diffs[m][k].append(mb[m][k] - mb[baseline][k])

    def ci(v):
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2

    out = {"n": int(n), "positives": int(y.sum()), "n_boot": n_boot, "rankers": {}}
    for m in names:
        c = np.asarray(curves[m])
        out["rankers"][m] = {
            "point": point[m],
            "ci": {k: ci(v) for k, v in samples[m].items()},
            "curve": {"k": curve_ks, "p": point_curve[m],
                      "lo": np.nanpercentile(c, 2.5, axis=0).tolist(),
                      "hi": np.nanpercentile(c, 97.5, axis=0).tolist()},
        }
        if m in diffs:
            out["rankers"][m]["diff_vs_" + baseline] = {
                k: {"point": point[m][k] - point[baseline][k], "ci": ci(v)} for k, v in diffs[m].items()}
    return out


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b))
