import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jobradar import cli, db, filters, labeling
from jobradar.models import Job
from jobradar.sources import ashby, greenhouse, lever, usajobs

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent


class FakeResp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, data):
        self.data, self.calls = data, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        return FakeResp(self.data)


class FakeClaude:
    """Scores by keyword so tests are deterministic."""
    def __init__(self):
        self.messages = self
        self.n = 0

    def create(self, **kw):
        self.n += 1
        assert kw["tool_choice"]["type"] == "tool"
        text = kw["messages"][0]["content"].lower()
        fit = 8 if "computer vision" in text else 3
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input={
            "fit_score": fit, "recommend": "apply" if fit >= 7 else "skip",
            "resume_variant": "geospatial_cv", "strongest_evidence": "solar CV pipeline",
            "gaps": ["gap"], "dealbreakers": [], "pitch": "I built a rooftop solar detector."})])


def write_resumes(base):
    (base / "resumes").mkdir()
    (base / "resumes/geospatial_cv.txt").write_text(
        "Computer vision engineer. Rooftop solar panel detection with PyTorch on aerial and satellite imagery. Geospatial raster pipelines.")
    (base / "resumes/data_science.txt").write_text(
        "Data scientist. A/B testing, SQL, Python, statistics, diffusion models, forecasting.")
    (base / "resumes/software_eng.txt").write_text(
        "Full-stack engineer. Next.js, React, TypeScript, FastAPI, Docker, CI.")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    # config.yaml and rubric.md are personal and gitignored; tests use committed stand-ins.
    shutil.copy(FIX / "config.yaml", tmp_path / "config.yaml")
    shutil.copy(ROOT / "rubric.example.md", tmp_path / "rubric.md")
    write_resumes(tmp_path)
    monkeypatch.setenv("USAJOBS_API_KEY", "k")
    monkeypatch.setenv("USAJOBS_EMAIL", "me@example.com")
    return cli.load_config(str(tmp_path / "config.yaml"))


def patch_sources(monkeypatch):
    us = FakeSession(json.loads((FIX / "usajobs.json").read_text()))
    gh = FakeSession(json.loads((FIX / "greenhouse.json").read_text()))
    monkeypatch.setitem(cli.SOURCES, "usajobs", lambda c: usajobs.fetch_raw(c, us))
    monkeypatch.setitem(cli.SOURCES, "greenhouse", lambda c: greenhouse.fetch_raw(c, gh))
    return us, gh


def test_usajobs_parse_and_headers(cfg):
    s = FakeSession(json.loads((FIX / "usajobs.json").read_text()))
    jobs = list(usajobs.fetch({"keywords": ["ml"], "locations": ["California"]}, s))
    assert len(jobs) == 2
    assert s.calls[0][2]["Authorization-Key"] == "k"
    assert s.calls[0][1]["LocationName"] == "California"
    j = jobs[0]
    assert j.extra["hiring_paths"] == ["public", "graduates"]
    assert "segmentation" in j.description
    assert jobs[1].description.count("Supervise staff") == 1  # string MajorDuties handled


def test_greenhouse_html_cleaned(cfg):
    s = FakeSession(json.loads((FIX / "greenhouse.json").read_text()))
    jobs = list(greenhouse.fetch_board("solarco", "SolarCo", s))
    assert "<" not in jobs[0].description and "&lt;" not in jobs[0].description
    assert "- 1+ years" in jobs[0].description


def test_filters(cfg):
    s = FakeSession(json.loads((FIX / "greenhouse.json").read_text()))
    gh = {j.title: filters.check(j, cfg["filters"]) for j in greenhouse.fetch_board("solarco", "SolarCo", s)}
    assert gh["Machine Learning Engineer, Computer Vision"] is None
    assert gh["Senior Staff Software Engineer"].startswith("title:")
    assert gh["Data Scientist"] == "location"
    assert gh["Data Analyst, Grid Operations"] is None  # min(5, 2) = 2 <= 3

    s = FakeSession(json.loads((FIX / "usajobs.json").read_text()))
    us = {j.title: filters.check(j, cfg["filters"]) for j in usajobs.fetch({"keywords": ["x"]}, s)}
    assert us["Physical Scientist (Remote Sensing)"] is None
    assert us["Supervisory Data Scientist"] is not None


def test_non_us_remote_blocked(cfg):
    def loc(s):
        return filters.check(greenhouse.parse_job({"id": 9, "title": "Data Scientist", "location": {"name": s}},
                                                  "x", "X"), cfg["filters"])
    assert loc("Remote - Canada") == "location:Canada"
    assert loc("Remote - US") is None
    assert loc("San Francisco, CA | London, UK") is None   # has a concrete US location
    assert loc("Remote") is None


def test_years_regex():
    f = filters.min_years_required
    assert f("Requires 3+ years of professional experience") == 3
    # Lenient by design: the MS path lowers the minimum even though he has no MS.
    # The rerank is responsible for flagging that as a dealbreaker.
    assert f("5 or more years of relevant industry experience, or 1 year with an MS") == 1
    assert f("BS + 4 years experience or MS + 2 years experience") == 2
    assert f("no requirement here") is None


def test_dedupe_across_runs(cfg, monkeypatch):
    patch_sources(monkeypatch)
    con = db.connect(cfg["db_path"])
    cli.cmd_ingest(cfg, con)
    n1 = con.scalar("SELECT COUNT(*) FROM jobs")
    cli.cmd_ingest(cfg, con)
    assert con.scalar("SELECT COUNT(*) FROM jobs") == n1


def test_end_to_end(cfg, monkeypatch, capsys):
    patch_sources(monkeypatch)
    con = db.connect(cfg["db_path"])
    cli.cmd_ingest(cfg, con)

    passing = con.query("SELECT title, recall FROM jobs WHERE filter_reason IS NULL ORDER BY recall DESC")
    assert passing[0]["title"] in ("Machine Learning Engineer, Computer Vision", "Physical Scientist (Remote Sensing)")

    fake = FakeClaude()
    cli.cmd_score(cfg, con, client=fake)
    assert fake.n == len(passing)

    path = cli.digest.write(con, cfg["digest"])
    text = path.read_text()
    assert "Machine Learning Engineer, Computer Vision" in text
    assert "Grid Operations" not in text  # fit 3, below min_fit
    assert cli.digest.write(con, cfg["digest"]) is None  # already digested

    # eval reads the frozen label sets; build tiny ones by hand
    lab = labeling.connect(cfg["labels_db_path"])
    for t, label, set_name in [("Machine Learning Engineer, Computer Vision", "apply", "a"),
                               ("Data Analyst, Grid Operations", "no", "a"),
                               ("Physical Scientist (Remote Sensing)", "apply", "a"),
                               ("Senior Staff Software Engineer", "apply", "b")]:
        r = con.query("SELECT * FROM jobs WHERE title=?", (t,))[0]
        lab.insert_rows("label_items", [(set_name, 0, "s", r["key"], r["source"], r["source_id"], r["title"],
                                         r["org"], r["location"], r["url"], r["description"], r["extra"],
                                         r["filter_reason"], r["recall"], "x", 1.0)])
        lab.execute("INSERT INTO labels VALUES (?,?,?,?,?,?,?)",
                    (set_name, r["source"], r["source_id"], r["key"], label, None, db.utcnow()))
    cli.cmd_eval(cfg, con, SimpleNamespace(), client=fake)
    out = capsys.readouterr().out
    assert "Senior Staff Software Engineer" in out and "Recall stage AUC" in out
    assert fake.n == len(passing)  # eval never calls Claude


def test_usajobs_multiple_locations_expanded(cfg):
    data = json.loads((FIX / "usajobs.json").read_text())
    d = data["SearchResult"]["SearchResultItems"][0]["MatchedObjectDescriptor"]
    d["PositionLocationDisplay"] = "Multiple Locations"
    d["PositionLocation"] = [{"LocationName": "Denver, Colorado"}, {"LocationName": "San Diego, California"}]
    j = usajobs.parse_item(data["SearchResult"]["SearchResultItems"][0])
    assert "San Diego" in j.location
    assert filters.check(j, cfg["filters"]) is None


def test_lever_parse_and_filters(cfg):
    s = FakeSession(json.loads((FIX / "lever.json").read_text()))
    jobs = list(lever.fetch({"boards": [{"token": "sunco", "name": "SunCo"}]}, s))
    assert [j.org for j in jobs] == ["SunCo"] * 3
    assert "- 1+ years" in jobs[0].description and "<li>" not in jobs[0].description
    assert jobs[0].posted.startswith("2026-")
    res = [filters.check(j, cfg["filters"]) for j in jobs]
    assert res[0] is None                      # San Diego
    assert jobs[1].location == "Remote (US)" and res[1] is None
    assert res[2] == "location"                # UK remote must not pass as "Remote"


def test_ashby_parse_and_filters(cfg):
    s = FakeSession(json.loads((FIX / "ashby.json").read_text()))
    jobs = list(ashby.fetch({"boards": [{"token": "gridco", "name": "GridCo"}]}, s))
    assert [j.title for j in jobs] == ["Computer Vision Engineer", "Software Engineer", "Engineer"]  # unlisted dropped
    res = [filters.check(j, cfg["filters"]) for j in jobs]
    assert res[0] is None and res[1] is None
    assert jobs[2].location == "Berlin; Munich" and res[2] == "location"

    # A bare "Remote" location from a non-US company must not pass on the word alone.
    de = ashby.parse_job({"id": "x", "title": "Technical Analyst", "location": "Remote", "isRemote": True,
                          "address": {"postalAddress": {"addressCountry": "Germany"}},
                          "descriptionPlain": "HomeOffice"}, "enpal", "Enpal")
    assert filters.check(de, cfg["filters"]) == "location:country=germany"


def test_usajobs_query_failure_does_not_abort(cfg):
    import requests

    class Flaky(FakeSession):
        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append((url, params, headers))
            if len(self.calls) == 1:
                raise requests.ConnectionError("boom")
            return FakeResp(self.data)

    s = Flaky(json.loads((FIX / "usajobs.json").read_text()))
    jobs = list(usajobs.fetch({"keywords": ["a", "b"], "locations": [None]}, s))
    assert len(jobs) == 2 and len(s.calls) == 2


class FailingClaude:
    def __init__(self, status_code):
        self.messages = self
        self.n = 0
        self.status_code = status_code

    def create(self, **kw):
        self.n += 1
        err = Exception(f"HTTP {self.status_code}")
        err.status_code = self.status_code
        raise err


def test_bad_api_key_fails_the_run_after_one_call(cfg, monkeypatch):
    patch_sources(monkeypatch)
    con = db.connect(cfg["db_path"])
    cli.cmd_ingest(cfg, con)
    fake = FailingClaude(401)
    with pytest.raises(RuntimeError, match="rejected"):
        cli.cmd_score(cfg, con, client=fake)
    assert fake.n == 1


def test_every_rerank_failing_fails_the_run(cfg, monkeypatch):
    patch_sources(monkeypatch)
    con = db.connect(cfg["db_path"])
    cli.cmd_ingest(cfg, con)
    fake = FailingClaude(500)
    with pytest.raises(RuntimeError, match="failed for all"):
        cli.cmd_score(cfg, con, client=fake)
    assert fake.n > 1  # transient errors still get tried on every job


def test_example_config_reads_public_sources(tmp_path):
    for f in ("config.example.yaml", "sources.yaml"):
        shutil.copy(ROOT / f, tmp_path / f)
    shutil.copy(ROOT / "rubric.example.md", tmp_path / "rubric.md")
    (tmp_path / "resumes").mkdir()
    (tmp_path / "resumes/main.txt").write_text("resume")
    c = cli.load_config(str(tmp_path / "config.example.yaml"))
    assert set(c["sources"]) == {"usajobs", "greenhouse", "lever", "ashby"}
    assert c["filters"]["usajobs"]["max_low_grade"] == 9
    # The public file holds queries only, no personal filter rules.
    assert "max_low_grade" not in c["sources"]["usajobs"]


def test_raw_tables_idempotent(cfg, monkeypatch):
    patch_sources(monkeypatch)
    con = db.connect(cfg["db_path"])
    cli.cmd_ingest(cfg, con)
    first = {s: con.scalar(f"SELECT COUNT(*) FROM raw_{s}") for s in db.RAW_SOURCES}
    t1 = con.scalar("SELECT max(ingested_at) FROM raw_greenhouse")
    cli.cmd_ingest(cfg, con)
    assert {s: con.scalar(f"SELECT COUNT(*) FROM raw_{s}") for s in db.RAW_SOURCES} == first
    assert first["usajobs"] == 2 and first["greenhouse"] > 0
    # First load time is kept; latest load time moves forward or stays equal.
    r = con.query("SELECT first_ingested_at, ingested_at, payload FROM raw_greenhouse LIMIT 1")[0]
    assert r["first_ingested_at"] <= r["ingested_at"] and r["ingested_at"] >= t1
    assert '"title"' in r["payload"]  # untouched API payload
    runs = con.query("SELECT source, rows_fetched, rows_new FROM ingest_runs ORDER BY started_at, source")
    assert [x["rows_new"] for x in runs if x["source"] == "usajobs"] == [2, 0]


def test_raw_upsert_same_id_twice_in_one_batch(tmp_path):
    con = db.connect(tmp_path / "x.duckdb")
    n = db.upsert_raw(con, "usajobs", [("1", "a|", None, {"v": 1}), ("1", "b|", None, {"v": 2})], "r1")
    assert n == 1
    assert con.query("SELECT board, payload FROM raw_usajobs") == [{"board": "b|", "payload": '{"v": 2}'}]


def test_migrates_legacy_sqlite(tmp_path):
    import sqlite3
    legacy = sqlite3.connect(tmp_path / "jobradar.db")
    legacy.executescript("""
        CREATE TABLE jobs (key TEXT PRIMARY KEY, source TEXT, source_id TEXT, title TEXT, org TEXT,
            location TEXT, url TEXT, description TEXT, posted TEXT, closes TEXT, extra TEXT,
            filter_reason TEXT, recall REAL, first_seen TEXT, last_seen TEXT);
        CREATE TABLE scores (key TEXT PRIMARY KEY, fit INTEGER, recommend TEXT, variant TEXT, evidence TEXT,
            gaps TEXT, dealbreakers TEXT, pitch TEXT, model TEXT, scored_at TEXT);
        CREATE TABLE labels (key TEXT PRIMARY KEY, label INTEGER, labeled_at TEXT);
        CREATE TABLE digested (key TEXT PRIMARY KEY, digest_date TEXT);
        INSERT INTO jobs VALUES ('k1','lever','a:1','DS','A','CA','u','d',NULL,NULL,'{}',NULL,0.5,'t','t');
        INSERT INTO scores VALUES ('k1',7,'apply','v','e','[]','[]','p','m','t');
    """)
    legacy.commit()
    legacy.close()
    con = db.connect(tmp_path / "jobradar.duckdb")
    assert con.scalar("SELECT recall FROM jobs WHERE key='k1'") == 0.5
    assert con.scalar("SELECT fit FROM scores") == 7
    assert (tmp_path / "jobradar.db").exists()  # left in place
