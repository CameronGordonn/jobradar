from datetime import timedelta
from pathlib import Path

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
