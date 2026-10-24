"""Near-duplicate postings: embedding similarity on "title | company | location", compared
only within a company (postings at different companies are different jobs).

1. candidates: 50 same-company pairs spanning the similarity range, stratified by bin
2. mark: label each pair same / different, blind to the similarity score
3. evaluate: pick the threshold from the marks, report precision and recall
4. cluster: connected components over pairs at or above the threshold -> job_clusters
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .db import DB, utcnow

MODEL = "BAAI/bge-small-en-v1.5"
LOW, BINS, PER_BIN = 0.85, 10, 5
TARGET_PRECISION = 0.95
FIELDS = ["pair_id", "sim", "bin", "weight", "job_a", "job_b", "company", "title_a", "location_a", "url_a",
          "title_b", "location_b", "url_b", "same"]


def load_jobs(con: DB, open_only: bool = True) -> list[dict]:
    where = "WHERE is_open" if open_only else ""
    return con.query(f"""SELECT job_id, company, title_clean AS title, coalesce(location, '') AS location,
                                url, exact_key
                         FROM intermediate.int_jobs {where} ORDER BY job_id""")


def embed(jobs: list[dict], encode=None) -> np.ndarray:
    if encode is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL)
        encode = lambda xs: model.encode(xs, normalize_embeddings=True, batch_size=64)  # noqa: E731
    return np.asarray(encode([f"{j['title']} | {j['company']} | {j['location']}" for j in jobs]))


def company_pairs(jobs: list[dict], E: np.ndarray, low: float) -> list[tuple[int, int, float]]:
    """Same-company pairs with similarity >= low, skipping exact company+title+location matches
    (those collapse without embeddings)."""
    by_co = defaultdict(list)
    for i, j in enumerate(jobs):
        by_co[j["company"].strip().lower()].append(i)
    out = []
    for idx in by_co.values():
        if len(idx) < 2:
            continue
        idx = np.array(idx)
        S = E[idx] @ E[idx].T
        a, b = np.triu_indices(len(idx), 1)
        keep = S[a, b] >= low
        for x, y, s in zip(idx[a[keep]], idx[b[keep]], S[a, b][keep], strict=True):
            if jobs[x]["exact_key"] != jobs[y]["exact_key"]:
                out.append((int(x), int(y), float(s)))
    return out


def sample_candidates(jobs, pairs, seed: int = 50, low: float = LOW, bins: int = BINS,
                      per: int = PER_BIN) -> list[dict]:
    rng = random.Random(seed)
    width = (1.0 - low) / bins
    binned = defaultdict(list)
    for p in pairs:
        binned[min(int((p[2] - low) / width), bins - 1)].append(p)
    rows = []
    for b in range(bins):
        pool = binned.get(b, [])
        for x, y, s in rng.sample(pool, min(per, len(pool))):
            rows.append({"sim": round(s, 4), "bin": b, "weight": len(pool) / min(per, len(pool)),
                         "job_a": jobs[x]["job_id"], "job_b": jobs[y]["job_id"], "company": jobs[x]["company"],
                         "title_a": jobs[x]["title"], "location_a": jobs[x]["location"], "url_a": jobs[x]["url"],
                         "title_b": jobs[y]["title"], "location_b": jobs[y]["location"], "url_b": jobs[y]["url"],
                         "same": ""})
    rng.shuffle(rows)  # marking order carries no hint of similarity
    for i, r in enumerate(rows):
        r["pair_id"] = i + 1
    return rows


def write_pairs(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_pairs(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mark_loop(path: Path, ask=input, out=print) -> int:
    rows = read_pairs(path)
    todo = [r for r in rows if r["same"] not in ("y", "n")]
    out("Same posting, or different jobs? y = same, n = different, q = quit. "
        "Same role listed for several cities: decide once and stay consistent.")
    done = 0
    for r in todo:
        out("\n" + "=" * 80 + f"\n{r['company']}")
        out(f"  A: {r['title_a']} | {r['location_a']}\n     {r['url_a']}")
        out(f"  B: {r['title_b']} | {r['location_b']}\n     {r['url_b']}")
        ans = ""
        while ans not in ("y", "n", "q"):
            ans = ask(f"[{rows.index(r) + 1}/{len(rows)}] same? ").strip().lower()
        if ans == "q":
            break
        r["same"] = ans
        done += 1
        write_pairs(path, rows)  # save after every answer
    return done


def evaluate(rows: list[dict]) -> dict:
    """Precision and recall of "sim >= t means same" over the marked pairs, raw and weighted by
    bin size (the sample is stratified, so weights turn it into an estimate for all candidates
    with similarity >= LOW). Rule fixed before marking: the lowest threshold with weighted
    precision >= 0.95; if none reaches it, the threshold with the best weighted F1."""
    marked = [r for r in rows if r["same"] in ("y", "n")]
    if not marked:
        return {"status": "pending", "reason": "no pairs marked yet"}
    sims = np.array([float(r["sim"]) for r in marked])
    y = np.array([r["same"] == "y" for r in marked])
    w = np.array([float(r["weight"]) for r in marked])
    table = []
    for t in sorted(set(sims)):
        pred = sims >= t
        tp, fp, fn = (w * (pred & y)).sum(), (w * (pred & ~y)).sum(), (w * (~pred & y)).sum()
        p = tp / (tp + fp) if tp + fp else float("nan")
        r_ = tp / (tp + fn) if tp + fn else float("nan")
        table.append({"threshold": float(t), "precision_w": float(p), "recall_w": float(r_),
                      "f1_w": float(2 * p * r_ / (p + r_)) if p + r_ else 0.0,
                      "tp": int((pred & y).sum()), "fp": int((pred & ~y).sum()), "fn": int((~pred & y).sum())})
    ok = [row for row in table if row["precision_w"] >= TARGET_PRECISION]
    pick = ok[0] if ok else max(table, key=lambda row: row["f1_w"])
    return {"status": "ok", "marked": len(marked), "same": int(y.sum()), "rule": "lowest threshold with weighted "
            f"precision >= {TARGET_PRECISION}" if ok else "no threshold reached the precision target; best weighted F1",
            "chosen": pick, "table": table}


def cluster(jobs: list[dict], E: np.ndarray, threshold: float) -> dict[str, str]:
    """Union-find over exact matches and same-company pairs at or above the threshold.
    Cluster id = smallest job id in the cluster, so it is stable while that posting lives."""
    parent = list(range(len(jobs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first = {}
    for i, j in enumerate(jobs):
        if j["exact_key"] in first:
            union(first[j["exact_key"]], i)
        else:
            first[j["exact_key"]] = i
    for a, b, _ in company_pairs(jobs, E, threshold):
        union(a, b)
    return {j["job_id"]: jobs[find(i)]["job_id"] for i, j in enumerate(jobs)}


def collapsed_share(mapping: dict[str, str]) -> float:
    return 1 - len(set(mapping.values())) / len(mapping) if mapping else 0.0


def write_clusters(con: DB, mapping: dict[str, str], threshold: float | None) -> None:
    con.execute("DELETE FROM job_clusters")
    t = utcnow()
    con.insert_rows("job_clusters", [(k, v, threshold, t) for k, v in mapping.items()])
