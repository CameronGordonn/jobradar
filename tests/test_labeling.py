from collections import Counter
from datetime import timedelta

import pytest

from jobradar import db, labeling
from jobradar.models import Job


def make_main(tmp_path, n_pass=250, filtered=None):
    filtered = filtered or {"title:senior": 40, "location": 30, "years:5": 8, "hiring_path:fed": 3, "text:ts/sci": 1}
    con = db.connect(tmp_path / "main.duckdb")
    items = [(Job("greenhouse", f"b:{i}", f"Data Scientist {i}", "Org", "CA", f"u{i}", f"desc {i}"), None)
             for i in range(n_pass)]
    k = 0
    for reason, n in filtered.items():
        for _ in range(n):
            items.append((Job("lever", f"f:{k}", f"Filtered {k}", "Org", "CA", f"v{k}", f"desc f{k}"), reason))
            k += 1
    db.upsert_jobs(con, items)
    keys = [r["key"] for r in con.query("SELECT key FROM jobs WHERE filter_reason IS NULL ORDER BY title")]
    db.set_recall(con, [(key, i / len(keys)) for i, key in enumerate(keys)])
    return con


def test_deciles_equal_count():
    scores = [(i * 7) % 30 for i in range(30)]  # 0..29 shuffled
    d = labeling.deciles(scores)
    assert Counter(d) == {i: 3 for i in range(1, 11)}
    assert d[scores.index(0)] == 1 and d[scores.index(29)] == 10  # input order preserved


def test_allocate_floor_and_total():
    a = labeling.allocate({"title": 2952, "location": 2159, "years": 313, "hiring_path": 67, "grade": 20, "text": 6}, 30)
    assert sum(a.values()) == 30
    assert min(a.values()) >= 2
    assert a["title"] > a["years"] > a["text"]
    assert labeling.allocate({"x": 1, "y": 100}, 10) == {"x": 1, "y": 9}  # can't take more than a stratum has


def test_create_sets(tmp_path):
    main = make_main(tmp_path)
    lab = labeling.connect(tmp_path / "labels.duckdb")
    n = labeling.create_sets(main, lab, seed=1, backend="tfidf")
    assert n == {"pool": 250, "a": 200, "b": 30, "c": 40}
    strata = Counter(r["stratum"] for r in lab.query("SELECT stratum FROM label_items WHERE set_name='a'"))
    assert strata == {f"decile_{d:02d}": 20 for d in range(1, 11)}
    assert lab.scalar("SELECT COUNT(*) FROM label_items WHERE set_name='b' AND filter_reason IS NULL") == 0
    assert lab.scalar("""SELECT COUNT(*) FROM label_items c JOIN label_items a USING (source, source_id)
                         WHERE c.set_name='c' AND a.set_name='a'""") == 40
    # a and b share one shuffled order with no repeats
    nums = [r["item_no"] for r in lab.query("SELECT item_no FROM label_items WHERE set_name IN ('a','b')")]
    assert sorted(nums) == list(range(230))
    assert lab.scalar("SELECT COUNT(*) FROM pool") == 250
    with pytest.raises(RuntimeError, match="already exist"):
        labeling.create_sets(main, lab, seed=1, backend="tfidf")


def test_create_sets_is_reproducible(tmp_path):
    picks = []
    for sub in ("one", "two"):
        (tmp_path / sub).mkdir()
        main = make_main(tmp_path / sub)
        lab = labeling.connect(tmp_path / sub / "labels.duckdb")
        labeling.create_sets(main, lab, seed=7, backend="tfidf")
        picks.append(lab.query("SELECT set_name, item_no, source_id FROM label_items ORDER BY ALL"))
    assert picks[0] == picks[1]


def test_parse_answer():
    assert labeling.parse_answer("a") == ("apply", None)
    assert labeling.parse_answer("n   too  senior ") == ("no", "too senior")
    assert labeling.parse_answer("Maybe") == ("maybe", None)
    assert labeling.parse_answer("x") is None


def scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_label_loop_is_blind_and_resumable(tmp_path):
    main = make_main(tmp_path)
    lab = labeling.connect(tmp_path / "labels.duckdb")
    labeling.create_sets(main, lab, seed=1, backend="tfidf")
    shown = []
    done = labeling.label_loop(lab, ["a", "b"], ask=scripted(["a", "zzz", "f", "n too senior", "u", "m", "q"]),
                               out=shown.append)
    assert done == 2
    rows = lab.query("SELECT label, note FROM labels ORDER BY labeled_at, label")
    assert sorted(r["label"] for r in rows) == ["apply", "maybe"]  # the "no" was undone and replaced
    text = "\n".join(shown)
    for leak in ("recall", "decile", "filter", "title:senior", "set a", "weight"):
        assert leak not in text.lower()
    # Next session picks up where this one stopped.
    assert labeling.label_loop(lab, ["a", "b"], limit=1, ask=scripted(["n"]), out=lambda s: None) == 1
    assert lab.scalar("SELECT COUNT(*) FROM labels") == 3


def test_relabel_set_locked_until_a_done_plus_7_days(tmp_path):
    main = make_main(tmp_path)
    lab = labeling.connect(tmp_path / "labels.duckdb")
    labeling.create_sets(main, lab, seed=1, backend="tfidf")
    msgs = []
    assert labeling.label_loop(lab, ["c"], ask=scripted([]), out=msgs.append) == 0
    assert "Finish set a" in msgs[-1]

    labeling.label_loop(lab, ["a"], ask=scripted(["n"] * 200), out=lambda s: None)
    assert labeling.label_loop(lab, ["c"], ask=scripted([]), out=msgs.append) == 0
    assert "opens" in msgs[-1]

    lab.execute("UPDATE labels SET labeled_at = labeled_at - INTERVAL 8 DAY")
    assert labeling.label_loop(lab, ["c"], limit=1, ask=scripted(["a"]), out=lambda s: None) == 1
    # The relabel is stored separately; the original a label is untouched.
    assert lab.scalar("SELECT COUNT(*) FROM labels WHERE set_name='a'") == 200
    assert "open" in labeling.status(lab)[-1]
    assert labeling.relabel_unlock_at(lab) <= db.utcnow() - timedelta(days=1)
