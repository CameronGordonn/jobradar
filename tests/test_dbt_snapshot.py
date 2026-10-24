"""Two dbt builds with one changed run in between: the snapshot must record the edits and
closures, and the change and lifecycle marts must count them. Needs dbt (requirements-pipeline.txt);
skipped where it isn't installed. CI's dbt job runs it."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from jobradar import db, pipeline

ROOT = Path(__file__).parent.parent
DBT = shutil.which("dbt") or str(Path(sys.executable).parent / "dbt")
pytestmark = pytest.mark.skipif(not Path(DBT).exists(), reason="dbt not installed")


def dbt_build(path):
    env = dict(os.environ, JOBRADAR_WAREHOUSE=str(path))
    r = subprocess.run([DBT, "build", "--profiles-dir", ".", "--no-use-colors"], cwd=ROOT / "warehouse",
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:]


def test_snapshot_records_edits_and_closures(tmp_path):
    w = tmp_path / "w.duckdb"
    con = db.connect(w, app_tables=False)
    pipeline.load_fixture_runs(con, ROOT / "tests/fixtures/warehouse/runs.json")
    con.close()
    dbt_build(w)
    con = db.connect(w, app_tables=False)
    pipeline.load_fixture_runs(con, ROOT / "tests/fixtures/warehouse/changes.json")
    con.close()
    dbt_build(w)

    con = db.DB(w, read_only=True)
    totals = con.query("""SELECT SUM(title_changes) titles, SUM(salary_changes) salaries, SUM(closures) closures
                          FROM marts.mart_posting_changes""")[0]
    assert totals == {"titles": 2, "salaries": 1, "closures": 3}
    assert con.scalar("SELECT COUNT(*) FROM marts.mart_posting_lifecycle WHERE days_listed >= 0") >= 3
