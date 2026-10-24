from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from jobradar import db, pipeline

FIXTURE = Path(__file__).parent / "fixtures/warehouse/runs.json"


def log_runs(con, source, counts):
    t0 = db.utcnow() - timedelta(days=len(counts))
    for i, n in enumerate(counts):
        db.log_run(con, f"r{i:02d}", source, t0 + timedelta(days=i), n, 0, finished=t0 + timedelta(days=i))


def test_warehouse_has_no_personal_tables(tmp_path):
    con = db.connect(tmp_path / "w.duckdb", app_tables=False)
    tables = {r["table_name"] for r in con.query("SELECT table_name FROM information_schema.tables")}
    assert {"raw_usajobs", "raw_greenhouse", "raw_lever", "raw_ashby", "ingest_runs",
            "quality_checks", "job_clusters"} <= tables
    assert not tables & {"jobs", "scores", "digested", "labels"}


def test_quality_flags_drop_below_half_of_trailing_median(tmp_path):
    con = db.connect(tmp_path / "w.duckdb", app_tables=False)
    log_runs(con, "lever", [100, 90, 110, 100, 95, 105, 100, 40])
    c = pipeline.check_quality(con, "r07")
    assert c[0]["flagged"] and c[0]["trailing_runs"] == 7 and c[0]["trailing_median"] == 100
    assert con.scalar("SELECT flagged FROM quality_checks WHERE run_id='r07'") is True


def test_quality_ok_and_too_little_history(tmp_path):
    con = db.connect(tmp_path / "w.duckdb", app_tables=False)
    log_runs(con, "ashby", [100, 100, 100, 60])        # 60% of median: fine
    assert pipeline.check_quality(con, "r03")[0]["flagged"] is False
    con2 = db.connect(tmp_path / "w2.duckdb", app_tables=False)
    log_runs(con2, "ashby", [100, 1])                  # one earlier run: don't judge
    c = pipeline.check_quality(con2, "r01")[0]
    assert c["flagged"] is False and "need 3" in c["reason"]


def test_quality_flags_a_source_stuck_at_zero(tmp_path):
    con = db.connect(tmp_path / "w.duckdb", app_tables=False)
    log_runs(con, "usajobs", [800, 0, 0, 0, 0, 0, 0, 0, 0])   # trailing median is now 0
    c = pipeline.check_quality(con, "r08")[0]
    assert c["flagged"] and "no rows" in c["reason"]
    con2 = db.connect(tmp_path / "w2.duckdb", app_tables=False)
    log_runs(con2, "usajobs", [0])                           # even on the very first run
    assert pipeline.check_quality(con2, "r00")[0]["flagged"]


def test_quality_flags_source_missing_from_run(tmp_path):
    con = db.connect(tmp_path / "w.duckdb", app_tables=False)
    log_runs(con, "usajobs", [50, 50, 50])
    log_runs(con, "lever", [10, 10, 10, 10])           # r03 has lever only
    c = {x["source"]: x for x in pipeline.check_quality(con, "r03")}
    assert c["usajobs"]["flagged"] and "no rows logged" in c["usajobs"]["reason"]
    assert not c["lever"]["flagged"]


def test_fixture_replay_is_idempotent_and_catches_the_outage(tmp_path):
    con = db.connect(tmp_path / "w.duckdb", app_tables=False)
    assert pipeline.load_fixture_runs(con, FIXTURE) == 8
    counts = {s: con.scalar(f"SELECT COUNT(*) FROM raw_{s}") for s in db.RAW_SOURCES}
    assert counts == {"usajobs": 12, "greenhouse": 50, "lever": 14, "ashby": 14}
    flagged = con.query("SELECT run_id, source FROM quality_checks WHERE flagged")
    assert flagged == [{"run_id": "fixture05", "source": "lever"}]
    # Latest load is "now", so dbt source freshness passes in CI.
    assert con.scalar("SELECT max(ingested_at) FROM raw_greenhouse") >= db.utcnow() - timedelta(minutes=5)


def fake_jobs():
    """Two companies; Acme lists one role under two titles, plus an unrelated role."""
    rows = [("a1", "Acme", "Machine Learning Engineer", "SF"), ("a2", "Acme", "ML Engineer", "SF"),
            ("a3", "Acme", "Accountant", "SF"), ("a4", "Acme", "Accountant", "SF"),
            ("b1", "Beta", "ML Engineer", "SF")]
    return [{"job_id": j, "company": c, "title": t, "location": loc, "url": "u", "exact_key": f"{c}|{t}|{loc}"}
            for j, c, t, loc in rows]


def fake_embed(jobs):
    vecs = {"Machine Learning Engineer": [1, 0.05, 0], "ML Engineer": [1, 0, 0], "Accountant": [0, 0, 1]}
    E = np.array([vecs[j["title"]] for j in jobs], dtype=float)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def test_dedupe_clusters_within_company_only():
    from jobradar import dedupe
    jobs = fake_jobs()
    E = fake_embed(jobs)
    pairs = dedupe.company_pairs(jobs, E, 0.5)
    assert [(jobs[a]["job_id"], jobs[b]["job_id"]) for a, b, _ in pairs] == [("a1", "a2")]  # a3/a4 exact, b1 other co
    m = dedupe.cluster(jobs, E, 0.99)
    assert m == {"a1": "a1", "a2": "a1", "a3": "a3", "a4": "a3", "b1": "b1"}
    assert dedupe.collapsed_share(m) == pytest.approx(2 / 5)


def test_dedupe_threshold_rule_and_blind_marking(tmp_path):
    from jobradar import dedupe
    rows = [{"pair_id": i, "sim": s, "bin": 0, "weight": 1.0, "job_a": "x", "job_b": "y", "company": "C",
             "title_a": "T", "location_a": "L", "url_a": "u", "title_b": "T2", "location_b": "L", "url_b": "u",
             "same": ""} for i, s in enumerate([0.86, 0.90, 0.93, 0.96, 0.99])]
    path = tmp_path / "pairs.csv"
    dedupe.write_pairs(path, rows)
    shown = []
    answers = iter(["n", "n", "y", "y", "y"])
    assert dedupe.mark_loop(path, ask=lambda p: next(answers), out=shown.append) == 5
    assert "0.9" not in "\n".join(shown)  # similarity never shown
    res = dedupe.evaluate(dedupe.read_pairs(path))
    assert res["chosen"]["threshold"] == 0.93 and res["chosen"]["precision_w"] == 1.0
    assert res["chosen"]["recall_w"] == 1.0


def test_parquet_export_restore_round_trip(tmp_path):
    con = db.connect(tmp_path / "w.duckdb", app_tables=False)
    pipeline.load_fixture_runs(con, FIXTURE)
    con.execute("CREATE SCHEMA snapshots")
    con.execute("CREATE TABLE snapshots.snap_postings AS SELECT source_id AS job_id, 1 AS v FROM raw_lever")
    m = pipeline.export_parquet(con, tmp_path / "export")
    assert m["rows"]["raw_greenhouse"] == 50 and m["rows"]["snapshots.snap_postings"] == 14

    fresh = db.connect(tmp_path / "restored.duckdb", app_tables=False)
    pipeline.restore_parquet(fresh, tmp_path / "export")
    for t in pipeline.BACKUP_TABLES + ["snapshots.snap_postings"]:
        assert fresh.scalar(f"SELECT COUNT(*) FROM {t}") == con.scalar(f"SELECT COUNT(*) FROM {t}")
    assert fresh.query("SELECT payload FROM raw_ashby ORDER BY source_id LIMIT 1") == \
        con.query("SELECT payload FROM raw_ashby ORDER BY source_id LIMIT 1")
    with pytest.raises(RuntimeError, match="empty warehouse"):
        pipeline.restore_parquet(fresh, tmp_path / "export")   # never overwrites
