import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

from jobradar import db, tracker
from jobradar.models import Job

CFG = {"sources": {"greenhouse": {"boards": [{"token": "acme", "name": "Acme", "area": "climate"}]}},
       "tracker": {"referrals": {"Acme": "Pat (eng manager)", "EY": "Sam"}}}


def job(sid, title, org="Acme", location="Remote - US"):
    return Job("greenhouse", sid, title, org, location, f"https://x/{sid}", "desc")


def score(con, key, fit, variant="data_science"):
    db.save_score(con, key, {"fit_score": fit, "recommend": "apply" if fit >= 7 else "maybe",
                             "resume_variant": variant, "strongest_evidence": "built X",
                             "gaps": ["Spark"], "dealbreakers": [], "pitch": "I built X."}, "m")


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "j.duckdb")
    db.upsert_jobs(c, [(job("1", "Data Scientist"), None), (job("2", "ML Engineer"), None),
                       (job("3", "Analyst"), None), (job("4", "Old Role"), None)])
    for key, fit in [("greenhouse:1", 8), ("greenhouse:2", 4), ("greenhouse:3", 6), ("greenhouse:4", 9)]:
        score(c, key, fit)
    c.execute("UPDATE jobs SET last_seen = '2020-01-01T00:00:00+00:00' WHERE key = 'greenhouse:4'")  # closed
    db.upsert_jobs(c, [(job("5", "Expired Role"), None)])
    score(c, "greenhouse:5", 9)
    c.execute("UPDATE jobs SET closes = '2020-06-30' WHERE key = 'greenhouse:5'")  # still listed, deadline passed
    return c


def test_inbox_open_scored_untouched_best_first(con):
    rows = tracker.inbox(con, CFG, min_fit=5)
    assert [r["key"] for r in rows] == ["greenhouse:1", "greenhouse:3"]   # fit 4 below min, fit 9 closed
    r = rows[0]
    assert r["area"] == "climate" and r["referral"] == "Pat (eng manager)" and r["gaps"] == ["Spark"]
    tracker.set_status(con, "greenhouse:1", "saved")
    assert [r["key"] for r in tracker.inbox(con, CFG, min_fit=5)] == ["greenhouse:3"]


def test_a_status_on_one_duplicate_clears_the_group(con):
    db.upsert_jobs(con, [(Job("lever", "9", "Analyst", "Acme", "Remote - US", "u", "d"), None)])  # same dedupe key as 3
    score(con, "lever:9", 6)
    tracker.set_status(con, "lever:9", "skipped", note="not my area")
    assert [r["key"] for r in tracker.inbox(con, CFG, min_fit=5)] == ["greenhouse:1"]


def test_status_history_and_stats(con):
    for s in ("saved", "applied", "interviewing", "rejected"):
        tracker.set_status(con, "greenhouse:1", s)
    tracker.set_status(con, "greenhouse:3", "applied")
    a = con.query("SELECT * FROM applications WHERE key = 'greenhouse:1'")[0]
    assert a["status"] == "rejected" and a["applied_at"] is not None
    assert con.scalar("SELECT COUNT(*) FROM application_events WHERE key = 'greenhouse:1'") == 4
    s = tracker.stats(con, CFG)
    assert s["total"] == {"applied": 2, "responded": 1, "interviewed": 1, "response_rate": 0.5, "interview_rate": 0.5}
    assert s["by_area"][0]["name"] == "climate" and s["applied_this_week"] == 2
    with pytest.raises(ValueError):
        tracker.set_status(con, "greenhouse:1", "ghosted")


def test_applied_at_is_kept_from_the_first_time(con):
    tracker.set_status(con, "greenhouse:1", "applied")
    first = con.scalar("SELECT applied_at FROM applications WHERE key = 'greenhouse:1'")
    con.execute("UPDATE applications SET applied_at = '2026-01-05T00:00:00+00:00'")
    tracker.set_status(con, "greenhouse:1", "interviewing")
    assert first and con.scalar("SELECT applied_at FROM applications") == "2026-01-05T00:00:00+00:00"


def test_follow_up_and_closed_posting(con):
    tracker.set_status(con, "greenhouse:1", "applied")
    tracker.set_status(con, "greenhouse:4", "saved")
    con.execute("UPDATE application_events SET happened_at = '2026-01-01T00:00:00+00:00' WHERE key = 'greenhouse:1'")
    p = {r["key"]: r for r in tracker.pipeline(con, CFG)}
    assert p["greenhouse:1"]["follow_up"] and not p["greenhouse:1"]["posting_closed"]
    assert p["greenhouse:4"]["posting_closed"] and not p["greenhouse:4"]["follow_up"]


def test_manual_job_skips_scoring_and_shows_in_pipeline(con):
    key = tracker.add_manual(con, "Data Engineer", "Accenture", referral="Tim", status="applied")
    assert con.scalar("SELECT filter_reason FROM jobs WHERE key = ?", (key,)) == "manual"
    p = {r["key"]: r for r in tracker.pipeline(con, CFG)}[key]
    assert p["referral"] == "Tim" and p["status"] == "applied" and not p["posting_closed"]
    with pytest.raises(ValueError):
        tracker.add_manual(con, " ", "Acme")


def test_area_rules_and_referral_matching():
    assert tracker.area_for("Bureau of Land Management", "Physical Scientist (Remote Sensing)", {}) == "geospatial"
    assert tracker.area_for("NOAA", "Meteorologist", {}) == "climate"
    assert tracker.area_for("Somewhere", "Accountant", {}) == "general"
    assert tracker.referral_for("EY", CFG["tracker"]["referrals"]) == "Sam"
    assert tracker.referral_for("Keystone Energy", CFG["tracker"]["referrals"]) is None  # whole word only


def test_server_reads_json_and_refuses_form_posts(tmp_path, con):
    cfg = {**CFG, "db_path": con.path}
    con.close()
    server = HTTPServer(("127.0.0.1", 0), tracker._handler(cfg))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        rows = json.load(urllib.request.urlopen(f"{base}/api/inbox?min_fit=7"))
        assert [r["key"] for r in rows] == ["greenhouse:1"]
        form = urllib.request.Request(f"{base}/api/status", data=b"key=greenhouse:1&status=skipped",
                                      headers={"Content-Type": "application/x-www-form-urlencoded"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(form)
        assert e.value.code == 415
        ok = urllib.request.Request(f"{base}/api/status", data=json.dumps({"key": "greenhouse:1", "status": "saved"})
                                    .encode(), headers={"Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(ok)) == {"ok": True}
    finally:
        server.shutdown()
