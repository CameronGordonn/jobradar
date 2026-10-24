import itertools
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jobradar import db, evaluation as ev, labeling, report
from jobradar.models import Job
from test_labeling import make_main

SYSTEM = [{"type": "text", "text": "rubric and resumes"}]
TOOL = {"name": "record_fit", "input_schema": {"type": "object"}}


def brute_precision(y, s, k):
    vals = []
    for perm in itertools.permutations(range(len(y))):
        order = sorted(range(len(y)), key=lambda i: (-s[i], perm.index(i)))
        vals.append(np.mean([y[i] for i in order[:k]]))
    return np.mean(vals)


def test_precision_curve_matches_brute_force_over_tie_orders():
    rng = np.random.default_rng(1)
    for _ in range(5):
        y = rng.integers(0, 2, 6)
        s = rng.integers(0, 3, 6).astype(float)
        got = ev.precision_curve(y, s, range(1, 7))
        assert np.allclose(got, [brute_precision(y, s, k) for k in range(1, 7)])


def test_bootstrap_paired_and_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 120)
    good = y + rng.normal(0, 0.5, 120)
    noise = rng.normal(0, 1, 120)
    res = ev.bootstrap(y, {"tfidf": noise, "better": good}, n_boot=300)
    b = res["rankers"]["better"]
    assert b["ci"]["auc"][0] <= b["point"]["auc"] <= b["ci"]["auc"][1]
    assert b["diff_vs_tfidf"]["auc"]["ci"][0] > 0          # clearly better on matched resamples
    assert len(b["curve"]["k"]) == 50 and len(b["curve"]["lo"]) == 50


class FakeAPI:
    """Messages, count_tokens and batches, scoring by the number in the title so fits are deterministic."""

    def __init__(self):
        self.messages = self
        self.batches = self
        self.calls = 0
        self._batches = {}

    def create(self, **kw):
        if "requests" in kw:  # batches.create
            bid = f"b{len(self._batches)}"
            self._batches[bid] = kw["requests"]
            return SimpleNamespace(id=bid)
        self.calls += 1
        text = kw["messages"][0]["content"]
        n = int(text.split("Data Scientist ")[1].split()[0]) if "Data Scientist " in text else 0
        fit = n % 11
        return SimpleNamespace(stop_reason="tool_use", usage=SimpleNamespace(
            input_tokens=6000, output_tokens=300, cache_creation_input_tokens=0, cache_read_input_tokens=0),
            content=[SimpleNamespace(type="tool_use", input={
                "fit_score": fit, "recommend": "apply" if fit >= 7 else "skip", "resume_variant": "v",
                "strongest_evidence": "e", "gaps": [], "dealbreakers": [], "pitch": "p"})])

    def count_tokens(self, **kw):
        return SimpleNamespace(input_tokens=(5000 if kw.get("system") else 0) + 100
                               + len(kw["messages"][0]["content"]) // 4)

    def retrieve(self, bid):
        return SimpleNamespace(processing_status="ended")

    def results(self, bid):
        return [SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(
            type="succeeded", message=self.create(**r["params"]))) for r in self._batches[bid]]


@pytest.fixture
def world(tmp_path):
    main = make_main(tmp_path)
    lab = ev.init(labeling.connect(tmp_path / "labels.duckdb"))
    labeling.create_sets(main, lab, seed=3, backend="tfidf")
    return main, lab


def label_all(lab):
    """Synthetic labels: apply when the title number is >= 150, plus a few maybes."""
    for set_name in ("a", "b"):
        for r in lab.query("SELECT * FROM label_items WHERE set_name=?", (set_name,)):
            digits = "".join(c for c in r["title"] if c.isdigit())
            n = int(digits or 0)
            label = "apply" if (set_name == "a" and n >= 150) or (set_name == "b" and n % 10 == 0) else (
                "maybe" if n % 7 == 0 else "no")
            lab.execute("INSERT INTO labels VALUES (?,?,?,?,?,?,?)",
                        (set_name, r["source"], r["source_id"], r["job_key"], label, None, db.utcnow()))


def test_score_sync_and_batch_record_tokens_and_cost(world):
    main, lab = world
    api = FakeAPI()
    items = ev.items_for_run(lab, 1)
    assert ev.score_sync(api, lab, "claude-haiku-4-5-20251001", 1, items[:5], SYSTEM, TOOL, 8000) == 5
    assert ev.score_batch(api, lab, "claude-haiku-4-5-20251001", 1, ev.pending_items(lab, "claude-haiku-4-5-20251001", 1),
                          SYSTEM, TOOL, 8000, poll_seconds=0) == 195
    rows = lab.query("SELECT origin, COUNT(*) n, SUM(cost_usd) usd FROM eval_scores GROUP BY 1 ORDER BY 1")
    # sync: 6000 in, 300 out at $1/$5 per MTok = $0.0075 per call; batch is half
    assert rows == [{"origin": "batch", "n": 195, "usd": pytest.approx(195 * 0.00375)},
                    {"origin": "sync", "n": 5, "usd": pytest.approx(5 * 0.0075)}]
    assert ev.pending_items(lab, "claude-haiku-4-5-20251001", 1) == []
    assert len(ev.items_for_run(lab, 2)) == 50
    assert ev.items_for_run(lab, 2) == ev.items_for_run(lab, 2)  # same stability subset every time


def test_estimate_uses_measured_output(world):
    main, lab = world
    api = FakeAPI()
    items = ev.items_for_run(lab, 1)
    e = ev.estimate(api, lab, "claude-sonnet-5-5", items, SYSTEM, TOOL, 8000, batch=True)
    assert e.system_tokens == 5000 and e.usd is None   # nothing measured yet
    ev.score_sync(api, lab, "claude-sonnet-5-5", 1, items[:2], SYSTEM, TOOL, 8000)
    e = ev.estimate(api, lab, "claude-sonnet-5-5", items, SYSTEM, TOOL, 8000, batch=True)
    assert e.mean_output_tokens == 300 and e.usd > 0


def test_reuse_production_scores_only_after_prompt_change(world):
    main, lab = world
    keys = [r["job_key"] for r in ev.items_for_run(lab, 1)[:3]]
    for k, when in zip(keys, ["2026-10-01T00:00:00+00:00", "2026-10-22T00:00:00+00:00", "2026-10-22T00:00:00+00:00"]):
        main.execute("INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (k, 8, "apply", "v", "e", "[]", "[]", "p", "claude-haiku-4-5-20251001", when))
    n = ev.reuse_production_scores(lab, main, "claude-haiku-4-5-20251001", datetime(2026, 10, 20))
    assert n == 2
    assert lab.scalar("SELECT SUM(cost_usd) FROM eval_scores") == 0


def test_report_end_to_end_and_private_data_stays_private(world, tmp_path):
    main, lab = world
    label_all(lab)
    api = FakeAPI()
    for model in ("claude-haiku-4-5-20251001", "claude-sonnet-5-5"):
        ev.score_sync(api, lab, model, 1, ev.items_for_run(lab, 1), SYSTEM, TOOL, 8000)
        ev.score_sync(api, lab, model, 2, ev.items_for_run(lab, 2), SYSTEM, TOOL, 8000)
    ev.embed_pool(lab, ["resume"], "bge-small", encode=lambda xs: np.array(
        [[(sum(map(ord, x)) % 97) / 97.0, 1.0] for x in xs]) / np.sqrt(2))
    ev.count_system_tokens(api, lab, "claude-haiku-4-5-20251001", SYSTEM, TOOL)
    ev.count_system_tokens(api, lab, "claude-sonnet-5-5", SYSTEM, TOOL)

    out, private = tmp_path / "results", tmp_path / "data/eval_details.md"
    m = report.build({"recall": {"top_k": 40}}, lab, main, out, private, n_boot=200)
    assert m["comparison"]["apply"]["status"] == "ok"
    assert set(m["comparison"]["apply"]["rankers"]) == {"tfidf", "emb:bge-small", "claude:claude-haiku-4-5-20251001",
                                                       "claude:claude-sonnet-5-5"}
    assert m["stability"]["claude-sonnet-5-5"]["spearman"] == pytest.approx(1.0)  # the fake is deterministic
    assert m["stage0_audit"]["labeled"] == 30
    assert m["self_agreement"]["status"] == "pending"
    ag = m["ranker_agreement"]
    assert ag["status"] == "ok" and ag["n"] == 200
    assert ag["spearman"]["claude:claude-haiku-4-5-20251001 | claude:claude-sonnet-5-5"] == pytest.approx(1.0)
    c = ag["claude_apply_below_cutoff"]["claude-sonnet-5-5"]
    assert c["scored"] == 200 and 0 <= c["tfidf"] <= c["apply"]
    # Haiku 4.5 needs 4096 tokens to cache; the fake system prompt is 5000, so it caches.
    assert m["production_cost"]["claude-haiku-4-5-20251001"]["system_caches"] is True
    assert (out / "precision_at_k.png").stat().st_size > 10_000

    public = (out / "eval_summary.md").read_text() + (out / "eval_metrics.json").read_text()
    titles = [r["title"] for r in lab.query("SELECT title FROM label_items")]
    assert not any(t in public for t in titles)
    assert "Filtered" in private.read_text() or "none" in private.read_text()
    json.loads((out / "eval_metrics.json").read_text())


def test_report_with_relabels_reports_self_agreement(world, tmp_path):
    main, lab = world
    label_all(lab)
    for r in lab.query("SELECT i.*, l.label FROM label_items i JOIN labels l ON l.set_name='a' "
                       "AND l.source=i.source AND l.source_id=i.source_id WHERE i.set_name='c'"):
        lab.execute("INSERT INTO labels VALUES ('c',?,?,?,?,?,?)",
                    (r["source"], r["source_id"], r["job_key"], r["label"], None, db.utcnow()))
    sa = report.self_agreement(lab)
    assert sa["status"] == "ok" and sa["agreement_3class"] == 1.0 and sa["kappa_3class"] == pytest.approx(1.0)
