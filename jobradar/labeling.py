"""Hand labels for evaluation, kept in their own DuckDB file (gitignored).

Sets are drawn once and frozen with the posting text, so the labels, the text the
rankers score, and the stage 1 scores used for sampling never drift apart:

  a  200 postings that passed stage 0, 20 from each stage 1 recall decile
  b  30 postings stage 0 dropped, stratified by rule family
  c  40 postings drawn at random from a, relabeled 7+ days after a is finished,
     to measure self-agreement

Labeling is blind: no scores, ranks, deciles, filter reasons or set names are shown.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from datetime import timedelta

from .db import DB, utcnow

LABELS = ("apply", "maybe", "no")
SET_A, SET_B, SET_C = 200, 30, 40
RELABEL_WAIT = timedelta(days=7)
EXCERPT_CHARS = 1500
REQ_CHARS = 700
# Where the role itself starts, and where its requirements start. Used only to pick the excerpt.
_ROLE = re.compile(r"responsibilit|what you.ll do|you will|about the role|the role|duties|"
                   r"what we.re looking|you may be a good fit|qualifications|requirements", re.I)
_REQS = re.compile(r"qualifications|requirements|what we.re looking for|you may be a good fit|"
                   r"you have|minimum|education", re.I)

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id VARCHAR PRIMARY KEY, created_at TIMESTAMP, seed INTEGER,
    recall_backend VARCHAR, n_pool INTEGER, n_filtered INTEGER
);
-- Every posting that passed stage 0 when the sets were drawn, with its stage 1 score.
-- Rankers in `eval` are refit on this frozen corpus.
CREATE TABLE IF NOT EXISTS pool (
    snapshot_id VARCHAR, job_key VARCHAR, source VARCHAR, source_id VARCHAR,
    title VARCHAR, org VARCHAR, location VARCHAR, url VARCHAR, description VARCHAR, extra VARCHAR,
    posted VARCHAR, first_seen VARCHAR, last_seen VARCHAR, recall DOUBLE,
    PRIMARY KEY (snapshot_id, job_key)
);
CREATE TABLE IF NOT EXISTS label_items (
    set_name VARCHAR, item_no INTEGER,       -- item_no is the (shuffled) display order
    snapshot_id VARCHAR, job_key VARCHAR, source VARCHAR, source_id VARCHAR,
    title VARCHAR, org VARCHAR, location VARCHAR, url VARCHAR, description VARCHAR, extra VARCHAR,
    filter_reason VARCHAR, recall DOUBLE,
    stratum VARCHAR,                         -- recall decile (a, c) or filter rule family (b)
    weight DOUBLE,                           -- postings in the stratum / postings sampled from it
    PRIMARY KEY (set_name, source, source_id)
);
CREATE TABLE IF NOT EXISTS labels (
    set_name VARCHAR, source VARCHAR, source_id VARCHAR, job_key VARCHAR,
    label VARCHAR CHECK (label IN ('apply', 'maybe', 'no')),
    note VARCHAR, labeled_at TIMESTAMP,
    PRIMARY KEY (set_name, source, source_id)
);
"""

_ITEM_COLS = ("job_key", "source", "source_id", "title", "org", "location", "url", "description", "extra")


def connect(path) -> DB:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = DB(path)
    con.con.execute(SCHEMA)
    return con


# ---------- sampling ----------

def deciles(scores: list[float]) -> list[int]:
    """Equal-count bins, 1 = lowest recall, 10 = highest. Input order is preserved."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    out = [0] * len(scores)
    for rank, i in enumerate(order):
        out[i] = rank * 10 // len(scores) + 1
    return out


def allocate(counts: dict[str, int], total: int, floor: int = 2) -> dict[str, int]:
    """Split `total` samples across strata: at least `floor` each (or all it has), the rest
    in proportion to sqrt(stratum size), so rare rules are seen without big ones vanishing."""
    alloc = {k: min(floor, n) for k, n in counts.items()}
    left = total - sum(alloc.values())
    while left > 0:
        room = {k: n - alloc[k] for k, n in counts.items() if n > alloc[k]}
        if not room:
            break
        w = {k: math.sqrt(counts[k]) for k in room}
        shares = {k: left * w[k] / sum(w.values()) for k in room}
        step = {k: min(room[k], int(shares[k])) for k in room}
        if sum(step.values()) == 0:  # hand out the remainder one at a time, largest share first
            k = max(room, key=lambda k: (shares[k] - int(shares[k]), counts[k]))
            step = {k: 1}
        for k, v in step.items():
            alloc[k] += v
        left -= sum(step.values())
    return alloc


def create_sets(main: DB, lab: DB, seed: int, backend: str) -> dict[str, int]:
    """Draw sets a, b and c from the current jobs table. Call after recall is fresh."""
    if lab.scalar("SELECT COUNT(*) FROM snapshots"):
        raise RuntimeError("label sets already exist; they are drawn once so labels stay comparable")
    rng = random.Random(seed)
    snap = f"s{utcnow():%Y%m%dT%H%M%S}"

    # A posting edited between runs (new title or location) gets a new dedupe key but keeps its
    # source id, so jobs can hold old versions. Keep the latest version of each.
    latest = main.query("""SELECT * FROM jobs
                           QUALIFY row_number() OVER (PARTITION BY source, source_id ORDER BY last_seen DESC, key) = 1
                           ORDER BY key""")
    passing = [r for r in latest if r["filter_reason"] is None and r["recall"] is not None]
    filtered = [r for r in latest if r["filter_reason"] is not None]
    if len(passing) < SET_A:
        raise RuntimeError(f"only {len(passing)} postings passed stage 0; need {SET_A}")

    lab.execute("BEGIN")  # all or nothing: a half-written draw would block a clean redraw
    try:
        _write_sets(lab, rng, snap, seed, backend, passing, filtered)
        lab.execute("COMMIT")
    except Exception:
        lab.execute("ROLLBACK")
        raise
    return {k: lab.scalar(f"SELECT COUNT(*) FROM {t}") for k, t in
            [("pool", "pool"), ("a", "label_items WHERE set_name='a'"),
             ("b", "label_items WHERE set_name='b'"), ("c", "label_items WHERE set_name='c'")]}


def _write_sets(lab: DB, rng: random.Random, snap: str, seed: int, backend: str,
                passing: list[dict], filtered: list[dict]) -> None:
    lab.insert_rows("snapshots", [(snap, utcnow(), seed, backend, len(passing), len(filtered))])
    lab.insert_rows("pool", [(snap, r["key"], r["source"], r["source_id"], r["title"], r["org"], r["location"],
                              r["url"], r["description"], r["extra"], r["posted"], r["first_seen"],
                              r["last_seen"], r["recall"]) for r in passing])

    # a: 20 per recall decile
    bins = defaultdict(list)
    for r, d in zip(passing, deciles([r["recall"] for r in passing]), strict=True):
        bins[d].append(r)
    per = SET_A // 10
    set_a = [(r, f"decile_{d:02d}", len(bins[d]) / per) for d in sorted(bins) for r in rng.sample(bins[d], per)]

    # b: stratified by stage 0 rule family
    fams = defaultdict(list)
    for r in filtered:
        fams[r["filter_reason"].split(":")[0]].append(r)
    alloc = allocate({k: len(v) for k, v in fams.items()}, SET_B)
    set_b = [(r, f, len(fams[f]) / alloc[f]) for f in sorted(fams) for r in rng.sample(fams[f], alloc[f])]

    # c: a random subset of a. Same text, separate labels.
    set_c = rng.sample(set_a, SET_C)

    # a and b share one shuffled order so they can be labeled interleaved.
    order = list(range(len(set_a) + len(set_b)))
    rng.shuffle(order)
    c_order = list(range(len(set_c)))
    rng.shuffle(c_order)

    def rows(name, items, numbers):
        return [(name, n, snap, r["key"], r["source"], r["source_id"], r["title"], r["org"], r["location"],
                 r["url"], r["description"], r["extra"], r["filter_reason"], r["recall"], stratum, weight)
                for (r, stratum, weight), n in zip(items, numbers, strict=True)]

    lab.insert_rows("label_items", rows("a", set_a, order[:len(set_a)])
                    + rows("b", set_b, order[len(set_a):]) + rows("c", set_c, c_order))


# ---------- relabel lock ----------

def relabel_unlock_at(lab: DB):
    """When set c opens: 7 days after the last label in a, once all of a is labeled. None if a isn't done."""
    n_items = lab.scalar("SELECT COUNT(*) FROM label_items WHERE set_name='a'")
    n_done, last = lab.query("SELECT COUNT(*) n, max(labeled_at) last FROM labels WHERE set_name='a'")[0].values()
    if not n_items or n_done < n_items:
        return None
    return last + RELABEL_WAIT


def status(lab: DB) -> list[str]:
    out = []
    for s, n, done in lab.con.execute("""
            SELECT i.set_name, COUNT(*), COUNT(l.label) FROM label_items i
            LEFT JOIN labels l USING (set_name, source, source_id) GROUP BY 1 ORDER BY 1""").fetchall():
        out.append(f"set {s}: {done}/{n} labeled")
    if not out:
        return ["No label sets yet. Run: python -m jobradar.cli label-sample"]
    unlock = relabel_unlock_at(lab)
    out.append("set c (relabel): locked until set a is fully labeled" if unlock is None
               else f"set c (relabel): {'open' if utcnow() >= unlock else 'locked'} from {unlock:%Y-%m-%d %H:%M} UTC")
    return out


# ---------- the labeling loop ----------

def excerpt(desc: str) -> str:
    """Skip a long company intro, show the role, then the requirements if they come later.
    The same rule for every posting, so it doesn't favor any of them."""
    if len(desc) <= EXCERPT_CHARS + REQ_CHARS:
        return desc
    m = _ROLE.search(desc)
    start = m.start() if m and m.start() > 300 else 0
    parts = [("[intro skipped] " if start else "") + desc[start:start + EXCERPT_CHARS]]
    end = start + EXCERPT_CHARS
    r = _REQS.search(desc, end)
    if r:
        parts.append("[...]\n" + desc[r.start():r.start() + REQ_CHARS])
        end = r.start() + REQ_CHARS
    return "\n".join(parts) + ("\n[... f shows the full posting]" if end < len(desc) else "")


def _show(item: dict, full: bool, out) -> None:
    out("\n" + "=" * 80)
    out(f"{item['title']} | {item['org']} | {item['location']}")
    out(item["url"])
    if item["source"] == "usajobs":  # posting content the reranker also sees, not a pipeline score
        e = json.loads(item["extra"] or "{}")
        out(f"Grade: GS-{e.get('low_grade')} to GS-{e.get('high_grade')} | "
            f"Hiring paths: {', '.join(e.get('hiring_paths') or [])}")
    out("-" * 80)
    # Tidy for the screen only: "-" bullets split from their text, and runs of blank lines.
    desc = re.sub(r"(?m)^-\s*\n\s*", "- ", item["description"] or "")
    desc = re.sub(r"\n{3,}", "\n\n", re.sub(r"\n\s*\n(?=- )", "\n", desc))
    out(desc if full else excerpt(desc))


def parse_answer(ans: str) -> tuple[str, str | None] | None:
    """'a', 'apply', 'n too senior' -> (label, note). None if it isn't a label."""
    head, _, note = ans.strip().partition(" ")
    label = {"a": "apply", "m": "maybe", "n": "no"}.get(head.lower(), head.lower())
    if label not in LABELS:
        return None
    return label, (" ".join(note.split()) or None)


def label_loop(lab: DB, sets: list[str], limit: int | None = None, ask=input, out=print) -> int:
    """Label unlabeled items from `sets` in their stored order. Returns how many were labeled."""
    if "c" in sets:
        unlock = relabel_unlock_at(lab)
        if unlock is None or utcnow() < unlock:
            out("Set c is locked. " + ("Finish set a first." if unlock is None
                                       else f"It opens {unlock:%Y-%m-%d %H:%M} UTC."))
            return 0
    ph = ",".join("?" * len(sets))
    items = lab.query(f"""SELECT i.* FROM label_items i LEFT JOIN labels l USING (set_name, source, source_id)
                          WHERE i.set_name IN ({ph}) AND l.label IS NULL ORDER BY i.item_no, i.set_name""", sets)
    total = lab.scalar(f"SELECT COUNT(*) FROM label_items WHERE set_name IN ({ph})", sets)
    if not items:
        out("Nothing left to label in " + ", ".join(sets) + ".")
        return 0
    out("Judge each posting as if it were open today. a = apply, m = maybe, n = no; "
        "add a short note after a space (n too senior). f = full text, u = undo last, q = quit.")
    done, history, i, full = 0, [], 0, False
    while i < len(items) and (limit is None or done < limit):
        item = items[i]
        _show(item, full, out)
        ans = ask(f"\n[{total - len(items) + i + 1}/{total}] label: ").strip()
        full = False
        if ans.lower() == "q":
            break
        if ans.lower() == "f":
            full = True
            continue
        if ans.lower() == "u":
            if history:
                prev = history.pop()
                lab.execute("DELETE FROM labels WHERE set_name=? AND source=? AND source_id=?",
                            (prev["set_name"], prev["source"], prev["source_id"]))
                done -= 1
                i -= 1
            continue
        parsed = parse_answer(ans)
        if parsed is None:
            out("Type a, m or n (optionally followed by a note), or f, u, q.")
            continue
        lab.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?)",
                    (item["set_name"], item["source"], item["source_id"], item["job_key"],
                     parsed[0], parsed[1], utcnow()))
        history.append(item)
        done += 1
        i += 1
    out(f"\nLabeled {done} this session.")
    return done
