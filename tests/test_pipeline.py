import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jobradar import cli, db, filters
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


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    shutil.copy(ROOT / "config.yaml", tmp_path / "config.yaml")
    shutil.copy(ROOT / "rubric.example.md", tmp_path / "rubric.md")  # rubric.md is personal and gitignored
    (tmp_path / "resumes").mkdir()
    (tmp_path / "resumes/geospatial_cv.txt").write_text(
        "Computer vision engineer. Rooftop solar panel detection with PyTorch on aerial and satellite imagery. Geospatial raster pipelines.")
    (tmp_path / "resumes/data_science.txt").write_text(
        "Data scientist. A/B testing, SQL, Python, statistics, diffusion models, forecasting.")
    (tmp_path / "resumes/software_eng.txt").write_text(
        "Full-stack engineer. Next.js, React, TypeScript, FastAPI, Docker, CI.")
    raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
    raw["sources"]["greenhouse"]["boards"] = [{"token": "solarco", "name": "SolarCo"}]
    for src in ("lever", "ashby"):  # keep tests offline; their adapters have their own tests
        raw["sources"][src]["boards"] = []
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("USAJOBS_API_KEY", "k")
    monkeypatch.setenv("USAJOBS_EMAIL", "me@example.com")
    return cli.load_config(str(tmp_path / "config.yaml"))


def patch_sources(monkeypatch):
    us = FakeSession(json.loads((FIX / "usajobs.json").read_text()))
    gh = FakeSession(json.loads((FIX / "greenhouse.json").read_text()))
    monkeypatch.setitem(cli.SOURCES, "usajobs", lambda c: usajobs.fetch(c, us))
    monkeypatch.setitem(cli.SOURCES, "greenhouse", lambda c: greenhouse.fetch(c, gh))
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
    n1 = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    cli.cmd_ingest(cfg, con)
    assert con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == n1


def test_end_to_end(cfg, monkeypatch):
    patch_sources(monkeypatch)
    con = db.connect(cfg["db_path"])
    cli.cmd_ingest(cfg, con)

    passing = con.execute("SELECT title, recall FROM jobs WHERE filter_reason IS NULL ORDER BY recall DESC").fetchall()
    assert passing[0]["title"] in ("Machine Learning Engineer, Computer Vision", "Physical Scientist (Remote Sensing)")

    fake = FakeClaude()
    cli.cmd_score(cfg, con, client=fake)
    assert fake.n == len(passing)

    path = cli.digest.write(con, cfg["digest"])
    text = path.read_text()
    assert "Machine Learning Engineer, Computer Vision" in text
    assert "Grid Operations" not in text  # fit 3, below min_fit
    assert cli.digest.write(con, cfg["digest"]) is None  # already digested

    # eval with hand labels
    keys = {r["title"]: r["key"] for r in con.execute("SELECT key, title FROM jobs")}
    for t, lab in [("Machine Learning Engineer, Computer Vision", 1), ("Data Analyst, Grid Operations", 0),
                   ("Physical Scientist (Remote Sensing)", 1), ("Senior Staff Software Engineer", 1)]:
        con.execute("INSERT INTO labels VALUES (?,?,?)", (keys[t], lab, db.now()))
    con.commit()
    cli.cmd_eval(cfg, con, SimpleNamespace(), client=fake)


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
