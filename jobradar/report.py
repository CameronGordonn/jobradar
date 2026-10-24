"""`python -m jobradar.cli eval --report`: every eval number in one regenerated summary.

Public outputs (results/, committed) hold aggregates only. Anything that names a posting
alongside a label goes to data/eval_details.md, which is gitignored.
"""
from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import numpy as np

from . import evaluation as ev
from . import labeling, rerank
from .db import DB, utcnow

RANKER_NAMES = {
    "tfidf": "TF-IDF (current stage 1)",
    "emb:bge-small": "Embeddings, bge-small-en-v1.5",
    "emb:minilm": "Embeddings, all-MiniLM-L6-v2",
    "claude:claude-haiku-4-5-20251001": "Claude Haiku 4.5 rerank",
    "claude:claude-sonnet-5-5": "Claude Sonnet 5.5 rerank",
    "claude:claude-haiku-5-5": "Claude Haiku 5.5 rerank",
}
FIGURE_RANKERS = ["tfidf", "emb:bge-small", "claude:claude-haiku-4-5-20251001", "claude:claude-sonnet-5-5"]
# Reference categorical palette, light mode, slots 1-4 in fixed order (dataviz skill).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
POSITIVES = {"apply": {"apply"}, "apply_or_maybe": {"apply", "maybe"}}
COMMAND = "python -m jobradar.cli eval --report"


def _name(r: str) -> str:
    return RANKER_NAMES.get(r, r)


def _fmt(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def ranker_scores(lab: DB) -> dict[str, dict[str, float]]:
    out = {"tfidf": {r["job_key"]: r["recall"] for r in lab.query("SELECT job_key, recall FROM pool")}}
    for r in lab.query("SELECT ranker, job_key, score FROM ranker_scores"):
        out.setdefault(r["ranker"], {})[r["job_key"]] = r["score"]
    for r in lab.query("SELECT model, job_key, fit FROM eval_scores WHERE run=1"):
        out.setdefault(f"claude:{r['model']}", {})[r["job_key"]] = float(r["fit"])
    return out


def labeled_a(lab: DB) -> list[dict]:
    return lab.query("""SELECT i.job_key, l.label FROM labels l JOIN label_items i USING (set_name, source, source_id)
                        WHERE l.set_name='a' ORDER BY i.job_key""")


def compare(lab: DB, positive: str, n_boot: int) -> dict:
    labs = labeled_a(lab)
    scores = ranker_scores(lab)
    order = [r for r in RANKER_NAMES if r in scores] + sorted(set(scores) - set(RANKER_NAMES))
    # Claude rankers join only once they cover the labeled postings; otherwise they'd shrink the shared set.
    keys = {r["job_key"] for r in labs}
    use = [r for r in order if r == "tfidf" or len(keys & set(scores[r])) >= 0.95 * len(keys)]
    common = [r for r in labs if all(r["job_key"] in scores[m] for m in use)]
    y = np.array([int(r["label"] in POSITIVES[positive]) for r in common])
    if len(common) < 10 or y.sum() == 0 or y.sum() == len(y):
        return {"status": "pending", "reason": f"{len(labs)} postings in set a labeled, {int(y.sum())} positive; "
                "need both positive and negative labels", "excluded": [r for r in order if r not in use]}
    res = ev.bootstrap(y, {m: np.array([scores[m][r["job_key"]] for r in common]) for m in use}, n_boot=n_boot)
    res["status"] = "ok"
    res["excluded"] = [r for r in order if r not in use]
    return res


def production_cost(lab: DB, model: str, top_k: int) -> dict | None:
    """Cost of one daily run (top_k sequential calls, standard prices) from measured per-call tokens.
    Production sends calls back to back, so the system prompt caches after the first call when it
    is long enough for the model."""
    r = lab.query("""SELECT COUNT(*) n, AVG(input_tokens + cache_creation_input_tokens + cache_read_input_tokens) inp,
                            AVG(output_tokens) outp FROM eval_scores
                     WHERE model=? AND origin != 'production_cache'""", (model,))[0]
    sys_tokens = lab.scalar("SELECT system_tokens FROM eval_prompt_tokens WHERE model=?", (model,))
    if not r["n"] or sys_tokens is None:
        return None
    fam = rerank._family(model)
    caches = sys_tokens >= rerank.MIN_CACHEABLE[fam]
    rest = r["inp"] - sys_tokens
    if caches:
        first = {"cache_creation_input_tokens": sys_tokens, "input_tokens": rest, "output_tokens": r["outp"]}
        later = {"cache_read_input_tokens": sys_tokens, "input_tokens": rest, "output_tokens": r["outp"]}
        usd = rerank.cost_usd(model, first) + (top_k - 1) * rerank.cost_usd(model, later)
    else:
        usd = top_k * rerank.cost_usd(model, {"input_tokens": r["inp"], "output_tokens": r["outp"]})
    return {"calls_measured": int(r["n"]), "mean_input_tokens": float(r["inp"]), "mean_output_tokens": float(r["outp"]),
            "system_tokens": int(sys_tokens), "min_cacheable": rerank.MIN_CACHEABLE[fam], "system_caches": caches,
            "top_k": top_k, "usd_per_run": usd}


def stability(lab: DB) -> dict:
    out = {}
    for m in [r["model"] for r in lab.query("SELECT DISTINCT model FROM eval_scores WHERE run=2 ORDER BY 1")]:
        rows = lab.query("""SELECT a.fit f1, b.fit f2, a.recommend r1, b.recommend r2 FROM eval_scores a
                            JOIN eval_scores b USING (model, job_key)
                            WHERE a.model=? AND a.run=1 AND b.run=2 AND a.origin != 'production_cache'""", (m,))
        if len(rows) < 10:
            continue
        f1, f2 = [r["f1"] for r in rows], [r["f2"] for r in rows]
        out[m] = {"n": len(rows), "spearman": ev.spearman(f1, f2),
                  "exact_fit_match": float(np.mean([a == b for a, b in zip(f1, f2, strict=True)])),
                  "within_1": float(np.mean([abs(a - b) <= 1 for a, b in zip(f1, f2, strict=True)])),
                  "recommend_match": float(np.mean([r["r1"] == r["r2"] for r in rows]))}
    return out


def stage0_audit(lab: DB) -> tuple[dict, list[dict]]:
    rows = lab.query("""SELECT i.stratum, i.weight, i.title, i.org, i.filter_reason, l.label FROM label_items i
                        LEFT JOIN labels l USING (set_name, source, source_id) WHERE i.set_name='b'
                        ORDER BY i.stratum, i.title""")
    done = [r for r in rows if r["label"]]
    fams = {}
    for r in rows:
        f = fams.setdefault(r["stratum"], {"sampled": 0, "labeled": 0, "apply": 0, "maybe": 0, "weight": r["weight"]})
        f["sampled"] += 1
        if r["label"]:
            f["labeled"] += 1
            if r["label"] in ("apply", "maybe"):
                f[r["label"]] += 1
    est = None
    if done:
        w = np.array([r["weight"] for r in done])
        est = float((w * np.array([r["label"] == "apply" for r in done])).sum() / w.sum())
    summary = {"sampled": len(rows), "labeled": len(done), "apply": sum(r["label"] == "apply" for r in done),
               "maybe": sum(r["label"] == "maybe" for r in done), "by_rule": fams,
               "weighted_apply_share_of_filtered": est}
    return summary, [r for r in done if r["label"] in ("apply", "maybe")]


def topk_audit(lab: DB, main: DB | None, top_k: int) -> tuple[dict, list[dict]]:
    pos = lab.query("""SELECT i.job_key, i.source, i.source_id, i.title, i.org FROM labels l
                       JOIN label_items i USING (set_name, source, source_id)
                       WHERE l.set_name='a' AND l.label='apply'""")
    if not pos:
        return {"status": "pending", "reason": "no apply labels in set a yet"}, []
    scores = ranker_scores(lab)
    out = {"status": "ok", "positives": len(pos), "top_k": top_k, "pool": lab.scalar("SELECT COUNT(*) FROM pool"),
           "below_cutoff_if_one_batch": {}}
    lost = []
    for m in ("tfidf", "emb:bge-small", "emb:minilm"):
        if m not in scores:
            continue
        ranked = sorted(scores[m], key=lambda k: -scores[m][k])
        rank = {k: i + 1 for i, k in enumerate(ranked)}
        below = [p for p in pos if rank.get(p["job_key"], 10**9) > top_k]
        out["below_cutoff_if_one_batch"][m] = len(below)
        if m == "tfidf":
            lost = [dict(p, rank=rank.get(p["job_key"])) for p in below]
    if main is not None:
        scored = {(r["source"], r["source_id"]) for r in
                  main.query("SELECT j.source, j.source_id FROM scores s JOIN jobs j USING (key)")}
        out["reached_claude_in_daily_runs"] = sum((p["source"], p["source_id"]) in scored for p in pos)
    return out, lost


def self_agreement(lab: DB) -> dict:
    rows = lab.query("""SELECT a.label la, c.label lc FROM labels c JOIN labels a USING (source, source_id)
                        WHERE c.set_name='c' AND a.set_name='a'""")
    n_c = lab.scalar("SELECT COUNT(*) FROM label_items WHERE set_name='c'")
    if not n_c or len(rows) < n_c:
        unlock = labeling.relabel_unlock_at(lab)
        when = f"opens {unlock:%Y-%m-%d} UTC" if unlock else "opens 7 days after set a is fully labeled"
        return {"status": "pending", "reason": f"{len(rows)}/{n_c} of set c relabeled; set c {when}"}
    a, c = [r["la"] for r in rows], [r["lc"] for r in rows]
    bin_a, bin_c = [x == "apply" for x in a], [x == "apply" for x in c]
    return {"status": "ok", "n": len(rows),
            "agreement_3class": float(np.mean([x == y for x, y in zip(a, c, strict=True)])),
            "kappa_3class": ev.cohen_kappa(a, c),
            "agreement_apply": float(np.mean([x == y for x, y in zip(bin_a, bin_c, strict=True)])),
            "kappa_apply": ev.cohen_kappa([str(x) for x in bin_a], [str(x) for x in bin_c])}


def agreement(lab: DB, top_k: int) -> dict:
    """Needs no labels. How alike the rankers order set a, and how many postings each Claude
    model recommends "apply" that stage 1 ranks below its top_k cutoff across the whole pool
    (in a single batch those would never reach Claude)."""
    scores = ranker_scores(lab)
    keys = [r["job_key"] for r in items_for_set_a(lab)]
    names = [m for m in RANKER_NAMES if m in scores and sum(k in scores[m] for k in keys) >= 0.95 * len(keys)]
    common = [k for k in keys if all(k in scores[m] for m in names)]
    out = {"n": len(common), "spearman": {}, "claude_apply_below_cutoff": {}}
    if len(common) < 10:
        return {"status": "pending", "reason": "fewer than 10 set a postings scored by every ranker"}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["spearman"][f"{a} | {b}"] = ev.spearman([scores[a][k] for k in common], [scores[b][k] for k in common])
    ranks = {}
    for m in ("tfidf", "emb:bge-small", "emb:minilm"):
        if m in scores:
            order = sorted(scores[m], key=lambda k: -scores[m][k])
            ranks[m] = {k: i + 1 for i, k in enumerate(order)}
    for model in [r["model"] for r in lab.query("SELECT DISTINCT model FROM eval_scores WHERE run=1 ORDER BY 1")]:
        picks = [r["job_key"] for r in lab.query(
            "SELECT job_key FROM eval_scores WHERE model=? AND run=1 AND recommend='apply'", (model,))]
        scored = lab.scalar("SELECT COUNT(*) FROM eval_scores WHERE model=? AND run=1", (model,))
        out["claude_apply_below_cutoff"][model] = {
            "scored": scored, "apply": len(picks),
            **{m: sum(r.get(k, 10**9) > top_k for k in picks) for m, r in ranks.items()}}
    out["status"] = "ok"
    out["top_k"] = top_k
    out["pool"] = lab.scalar("SELECT COUNT(*) FROM pool")
    return out


def items_for_set_a(lab: DB) -> list[dict]:
    return lab.query("SELECT job_key FROM label_items WHERE set_name='a' ORDER BY job_key")


def spend(lab: DB) -> dict:
    by = {r["model"]: float(r["usd"]) for r in lab.query(
        """SELECT model, SUM(cost_usd) usd FROM (SELECT model, cost_usd FROM eval_scores
           UNION ALL SELECT model, cost_usd FROM eval_failures) GROUP BY 1 ORDER BY 1""")}
    return {"total_usd": sum(by.values()), "by_model": by}


# ---------- figure ----------

def figure(res: dict, path: Path) -> bool:
    if res.get("status") != "ok":
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [m for m in FIGURE_RANKERS if m in res["rankers"]]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    base = res["positives"] / res["n"]
    ax.axhline(base, color=INK_2, lw=1)
    ax.text(1.5, base + 0.012, f"labeled base rate {base:.2f}", color=INK_2, fontsize=8, va="bottom")
    for m in present:  # color follows the ranker's fixed slot, not its rank
        color = SERIES[FIGURE_RANKERS.index(m)]
        c = res["rankers"][m]["curve"]
        ax.fill_between(c["k"], c["lo"], c["hi"], color=color, alpha=0.10, lw=0)
        ax.plot(c["k"], c["p"], color=color, lw=2, solid_capstyle="round", label=_name(m))
        ax.plot(c["k"][-1], c["p"][-1], "o", color=color, ms=5, mec=SURFACE, mew=2)
    for x in (10, 20):
        ax.axvline(x, color=GRID, lw=1, zorder=0)
    ax.set_xlim(1, max(res["rankers"][present[0]]["curve"]["k"]) + 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("k (top-ranked labeled postings)", color=INK_2)
    ax.set_ylabel("precision@k", color=INK_2)
    ax.set_title(f"Precision@k on {res['n']} hand-labeled postings, 95% bootstrap bands",
                 color=INK, loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


# ---------- markdown ----------

def _metric_table(res: dict) -> list[str]:
    if res.get("status") != "ok":
        return [f"Pending: {res.get('reason')}."]
    lines = [f"n = {res['n']} labeled postings, {res['positives']} positive, {res['n_boot']} bootstrap resamples.", "",
             "| Ranker | ROC AUC | Average precision | Precision@10 | Precision@20 |", "|---|---|---|---|---|"]
    for m, r in res["rankers"].items():
        cells = [f"{_fmt(r['point'][k])} [{_fmt(r['ci'][k][0])}, {_fmt(r['ci'][k][1])}]"
                 for k in ("auc", "ap", "p@10", "p@20")]
        lines.append(f"| {_name(m)} | " + " | ".join(cells) + " |")
    diffs = [(m, r["diff_vs_tfidf"]) for m, r in res["rankers"].items() if "diff_vs_tfidf" in r]
    if diffs:
        lines += ["", "Paired difference against TF-IDF on the same resamples (95% CI):", "",
                  "| Ranker | ROC AUC | Precision@10 |", "|---|---|---|"]
        for m, d in diffs:
            lines.append(f"| {_name(m)} | " + " | ".join(
                f"{d[k]['point']:+.3f} [{d[k]['ci'][0]:+.3f}, {d[k]['ci'][1]:+.3f}]" for k in ("auc", "p@10")) + " |")
    if res.get("excluded"):
        lines += ["", "Not yet scored on enough labeled postings to compare: "
                  + ", ".join(_name(m) for m in res["excluded"]) + "."]
    return lines


def build(cfg: dict, lab: DB, main: DB | None, out_dir: Path, private_path: Path, n_boot: int = 2000) -> dict:
    ev.init(lab)
    top_k = cfg.get("recall", {}).get("top_k", 40)
    snap = (lab.query("SELECT * FROM snapshots") or [{}])[0]
    counts = {r["set_name"]: (r["n"], r["done"]) for r in lab.query(
        """SELECT i.set_name, COUNT(*) n, COUNT(l.label) done FROM label_items i
           LEFT JOIN labels l USING (set_name, source, source_id) GROUP BY 1""")}
    results = {pos: compare(lab, pos, n_boot) for pos in POSITIVES}
    costs = {m: production_cost(lab, m, top_k) for m in
             [r["model"] for r in lab.query("SELECT DISTINCT model FROM eval_scores ORDER BY 1")]}
    stab = stability(lab)
    s0, s0_rows = stage0_audit(lab)
    tk, tk_rows = topk_audit(lab, main, top_k)
    sa = self_agreement(lab)
    ag = agreement(lab, top_k)
    sp = spend(lab)
    fig_ok = figure(results["apply"], out_dir / "precision_at_k.png")

    metrics = {"generated_at": utcnow().replace(tzinfo=UTC).isoformat(), "command": COMMAND,
               "snapshot": {k: (str(v) if k == "created_at" else v) for k, v in snap.items()},
               "label_counts": {k: {"items": n, "labeled": d} for k, (n, d) in counts.items()},
               "comparison": results, "production_cost": costs, "stability": stab, "stage0_audit": s0,
               "topk_audit": tk, "self_agreement": sa, "ranker_agreement": ag, "api_spend": sp}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    L = ["# Ranking evaluation", "",
         f"Generated by `{COMMAND}` on {metrics['generated_at'][:10]}. Every number below comes from that command; "
         "\"pending\" means the data for it doesn't exist yet.", ""]
    L += ["## Data", "",
          f"- Postings that passed stage 0 when the label sets were drawn: {snap.get('n_pool', 'n/a')} "
          f"(snapshot {str(snap.get('created_at', 'n/a'))[:10]}, seed {snap.get('seed', 'n/a')}).",
          "- Set a: 200 of them, 20 from each stage 1 recall decile. Because the deciles are equal-sized, "
          "this is close to a uniform sample of the pool, so metrics over set a describe the pool.",
          "- Labels: apply / maybe / no, given blind (no scores, ranks or filter reasons shown).",
          "- Labeled so far: " + ", ".join(f"set {k} {d}/{n}" for k, (n, d) in sorted(counts.items())) + ".", ""]
    if fig_ok:
        L += ["![Precision at k](precision_at_k.png)", ""]
    L += ["## Rankers compared", "",
          "- **TF-IDF**: stage 1 as it ran (word and bigram TF-IDF, cosine against each resume, max), "
          "scores frozen when the sets were drawn.",
          "- **Embeddings, bge-small-en-v1.5**: the pre-registered local embedding ranker. Picked over "
          "all-MiniLM-L6-v2 before seeing results because postings are long (median about 6,700 characters) "
          "and bge-small reads 512 tokens against MiniLM's 256, and because it is trained for retrieval. "
          "MiniLM is reported for reference. Both truncate long postings.",
          "- **Claude rerank**: the production prompt (rubric plus resumes, one posting per call, structured "
          "tool output), scored on the same frozen text that was labeled. The ranking score is the 0-10 fit score. "
          "Precision@k uses the expected value over random order within tied scores.", ""]
    for pos, title in (("apply", "Positive = apply"), ("apply_or_maybe", "Positive = apply or maybe")):
        L += [f"## {title}", ""] + _metric_table(results[pos]) + [""]
    L += ["## Cost per daily run", ""]
    if any(costs.values()):
        L += [f"One daily run sends the top {top_k} postings by recall to Claude, one call each, at standard "
              "prices. Computed from token counts measured on the eval calls.", "",
              "| Model | Calls measured | Mean input tokens | Mean output tokens | System prompt caches? "
              "| USD per run |",
              "|---|---|---|---|---|---|"]
        for m, c in costs.items():
            if c:
                caches = f"{'yes' if c['system_caches'] else 'no'} ({c['system_tokens']} tokens, " \
                         f"minimum {c['min_cacheable']})"
                L.append(f"| {m} | {c['calls_measured']} | {c['mean_input_tokens']:.0f} | "
                         f"{c['mean_output_tokens']:.0f} | {caches} | "
                         f"${c['usd_per_run']:.3f} |")
    else:
        L.append("Pending: no Claude calls measured yet.")
    L += ["", "## LLM stability", "",
          "The rerank is pointwise (one posting per call), so stability is measured by scoring the same "
          f"{ev.STABILITY_N} postings twice per model.", ""]
    if stab:
        L += ["| Model | n | Spearman (fit) | Exact fit match | Within 1 point | Same recommendation |",
              "|---|---|---|---|---|---|"]
        for m, s in stab.items():
            L.append(f"| {m} | {s['n']} | {s['spearman']:.3f} | {s['exact_fit_match']:.0%} | {s['within_1']:.0%} | "
                     f"{s['recommend_match']:.0%} |")
    else:
        L.append("Pending: second scoring run not done yet.")
    L += ["", "## Stage 0 audit (filter false negatives)", ""]
    if s0["labeled"]:
        L += [f"{s0['labeled']} of {s0['sampled']} filtered postings labeled: "
              f"{s0['apply']} apply, {s0['maybe']} maybe.",
              "", "| Rule | Sampled | Labeled | Apply | Maybe |", "|---|---|---|---|---|"]
        for f, d in sorted(s0["by_rule"].items()):
            L.append(f"| {f} | {d['sampled']} | {d['labeled']} | {d.get('apply', 0)} | {d.get('maybe', 0)} |")
        L += ["", f"Weighted estimate of the share of all filtered postings you'd apply to: "
              f"{s0['weighted_apply_share_of_filtered']:.1%} (set b is stratified by rule, so each label is "
              "weighted by its rule's size)."]
    else:
        L.append("Pending: set b not labeled yet.")
    L += ["", "## Top-k audit", ""]
    if tk["status"] == "ok":
        L += [f"{tk['positives']} apply labels in set a. With the whole pool of {tk['pool']} as one batch, "
              f"positives ranked below the top {top_k} (they would never reach Claude):", ""]
        for m, n in tk["below_cutoff_if_one_batch"].items():
            L.append(f"- {_name(m)}: {n} of {tk['positives']}")
        if "reached_claude_in_daily_runs" in tk:
            L += ["", f"In the actual daily runs so far, {tk['reached_claude_in_daily_runs']} of {tk['positives']} "
                  "apply-labeled postings were sent to Claude (daily runs see only that day's unscored postings, "
                  "so this depends on timing as well as rank)."]
    else:
        L.append(f"Pending: {tk['reason']}.")
    L += ["", "## Ranker agreement (no labels needed)", ""]
    if ag["status"] == "ok":
        L += [f"Spearman correlation of the scores each ranker gives the same {ag['n']} set a postings:", "",
              "| Rankers | Spearman |", "|---|---|"]
        for pair, rho in ag["spearman"].items():
            a, b = pair.split(" | ")
            L.append(f"| {_name(a)} vs {_name(b)} | {rho:.3f} |")
        if ag["claude_apply_below_cutoff"]:
            L += ["", f"Postings each Claude model recommends \"apply\" that a recall ranker puts below its top "
                  f"{ag['top_k']} across the whole pool of {ag['pool']} "
                  "(in a single batch they would never reach Claude):", "",
                  "| Model | Scored | Apply | Below TF-IDF top-k | Below bge-small top-k | Below MiniLM top-k |",
                  "|---|---|---|---|---|---|"]
            for m, c in ag["claude_apply_below_cutoff"].items():
                L.append(f"| {m} | {c['scored']} | {c['apply']} | {c.get('tfidf', 'n/a')} | "
                         f"{c.get('emb:bge-small', 'n/a')} | {c.get('emb:minilm', 'n/a')} |")
            L += ["", "This compares rankers with each other, not with the labels: it shows how much the recall stage "
                  "and the reranker disagree, not which one is right."]
    else:
        L.append(f"Pending: {ag['reason']}.")
    L += ["", "## Self-agreement ceiling", ""]
    if sa["status"] == "ok":
        L.append(f"On {sa['n']} postings relabeled 7+ days later: {sa['agreement_3class']:.0%} identical labels "
                 f"(Cohen's kappa {sa['kappa_3class']:.2f}); apply vs not: {sa['agreement_apply']:.0%} "
                 f"(kappa {sa['kappa_apply']:.2f}). No ranker can be expected to agree with the labels more than "
                 "the labeler agrees with themself.")
    else:
        L.append(f"Pending: {sa['reason']}.")
    L += ["", "## API spend", "", f"Total spent on eval calls: ${sp['total_usd']:.2f}" +
          ("".join(f"; {m} ${v:.2f}" for m, v in sp["by_model"].items())) + ".", "",
          "## Reproduce", "", "```bash",
          "python -m jobradar.cli label-sample           # once: draw the frozen sets",
          "python -m jobradar.cli label --set a,b        # blind labeling",
          "python -m jobradar.cli eval-embed             # local embedding rankers",
          "python -m jobradar.cli eval-score --model claude-haiku-4-5-20251001 --run 1 --yes",
          "python -m jobradar.cli eval-score --model claude-sonnet-5-5 --run 1 --yes",
          "python -m jobradar.cli eval-score --model <model> --run 2 --yes   # stability",
          f"{COMMAND}", "```", "",
          "Labels, resumes and the rubric are personal and not in the repository, so the commands above "
          "reproduce the numbers only with them present.", ""]
    (out_dir / "eval_summary.md").write_text("\n".join(L))

    P = ["# Eval details (private, gitignored)", "", "## Filtered postings labeled apply or maybe", ""]
    P += [f"- [{r['label']}] {r['title']} | {r['org']}  rule: {r['filter_reason']}" for r in s0_rows] or ["- none"]
    P += ["", f"## Apply-labeled postings below the TF-IDF top {top_k}", ""]
    P += [f"- rank {r['rank']}: {r['title']} | {r['org']}" for r in tk_rows] or ["- none"]
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text("\n".join(P) + "\n")
    return metrics
