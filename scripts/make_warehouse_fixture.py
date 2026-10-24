"""Write tests/fixtures/warehouse/runs.json (eight synthetic daily runs of API responses) and
changes.json (one later run with edits and closures, for the snapshot test). Companies,
titles and numbers are made up. Deterministic (seeded).

    python scripts/make_warehouse_fixture.py
"""
from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests/fixtures/warehouse/runs.json"
CHANGES = OUT.with_name("changes.json")
RUNS = 8
TODAY = datetime(2026, 10, 1, 12, tzinfo=UTC)  # posted dates are relative to this

TITLES = ["Machine Learning Engineer", "ML Engineer", "Data Scientist", "Senior Data Scientist",
          "Data Engineer", "Analytics Engineer", "Software Engineer, Backend", "Backend Software Engineer",
          "Geospatial Analyst", "Remote Sensing Scientist", "Product Manager", "Account Executive",
          "Operations Research Analyst", "Hydrologist", "Mathematical Statistician",
          "Research Scientist, Computer Vision"]
PLACES = ["San Diego, CA", "Remote (US)", "Denver, CO", "Austin, TX", "Boston, MA", "Remote"]

BOARDS = {
    "greenhouse": {"solarco": "SolarCo", "gridworks": "GridWorks"},
    "lever": {"sunco": "SunCo"},
    "ashby": {"voltline": "Voltline"},
}


def postings(rng: random.Random, source: str, n: int) -> list[dict]:
    out = []
    for i in range(n):
        title = rng.choice(TITLES)
        out.append({"i": i, "title": title, "place": rng.choice(PLACES),
                    "posted_days_ago": rng.randint(0, 60), "open_from_run": rng.randint(0, 3),
                    "open_to_run": rng.choice([RUNS - 1, RUNS - 1, RUNS - 1, rng.randint(3, RUNS - 1)]),
                    "salary": rng.choice([None, (90000, 130000), (120000, 170000), (40, 65)])})
    return out


def iso(days_ago: int) -> str:
    return (TODAY - timedelta(days=days_ago)).isoformat()


def greenhouse_job(p, token):
    job = {"id": 1000 + p["i"], "title": p["title"], "location": {"name": p["place"]},
           "absolute_url": f"https://example.com/{token}/{p['i']}", "first_published": iso(p["posted_days_ago"]),
           "updated_at": iso(0), "company_name": token, "departments": [{"name": "Engineering"}],
           "content": "&lt;p&gt;Synthetic posting.&lt;/p&gt;"}
    if p["salary"]:
        lo, hi = p["salary"]
        job["pay_input_ranges"] = [{"min_cents": lo * 100, "max_cents": hi * 100, "currency_type": "USD",
                                    "title": "Hourly Pay" if hi < 1000 else "Annual Salary"}]
    return job


def lever_job(p, site):
    job = {"id": f"lv-{p['i']}", "text": p["title"], "hostedUrl": f"https://example.com/{site}/{p['i']}",
           "createdAt": int((TODAY - timedelta(days=p["posted_days_ago"])).timestamp() * 1000),
           "categories": {"location": p["place"], "allLocations": [p["place"]], "team": "Data",
                          "commitment": "Full-time"},
           "workplaceType": "remote" if "Remote" in p["place"] else "onsite", "country": "US",
           "descriptionPlain": "Synthetic posting."}
    if p["salary"]:
        lo, hi = p["salary"]
        job["salaryRange"] = {"min": lo, "max": hi, "currency": "USD",
                              "interval": "per-hour-wage" if hi < 1000 else "per-year-salary"}
    return job


def ashby_job(p, name):
    comp = []
    if p["salary"]:
        lo, hi = p["salary"]
        comp = [{"compensationType": "Salary", "currencyCode": "USD", "interval": "1 HOUR" if hi < 1000 else "1 YEAR",
                 "minValue": lo, "maxValue": hi}]
    return {"id": f"ash-{p['i']}", "title": p["title"], "location": p["place"], "secondaryLocations": [],
            "isRemote": "Remote" in p["place"], "workplaceType": "Remote" if "Remote" in p["place"] else "OnSite",
            "publishedAt": iso(p["posted_days_ago"]), "jobUrl": f"https://example.com/{name}/{p['i']}",
            "department": "Engineering", "team": "Data", "employmentType": "FullTime", "isListed": True,
            "address": {"postalAddress": {"addressCountry": "United States"}},
            "compensation": {"summaryComponents": comp}, "descriptionPlain": "Synthetic posting."}


def usajobs_item(p):
    lo, hi = p["salary"] if p["salary"] and p["salary"][1] > 1000 else (60000, 95000)
    return {"MatchedObjectId": str(5000 + p["i"]), "MatchedObjectDescriptor": {
        "PositionID": f"FIX-{p['i']}", "PositionTitle": p["title"], "OrganizationName": "Example Survey",
        "DepartmentName": "Department of Examples", "PositionLocationDisplay": p["place"],
        "PositionLocation": [{"LocationName": p["place"]}], "PositionURI": f"https://example.com/usajobs/{p['i']}",
        "PublicationStartDate": iso(p["posted_days_ago"])[:19], "ApplicationCloseDate": "2099-12-31T23:59:59",
        "PositionRemuneration": [{"MinimumRange": str(lo), "MaximumRange": str(hi), "RateIntervalCode": "PA"}],
        "QualificationSummary": "Synthetic.",
        "UserArea": {"Details": {"JobSummary": "Synthetic posting.", "HiringPath": ["public"], "LowGrade": "7",
                                 "HighGrade": "9", "RemoteIndicator": "Remote" in p["place"],
                                 "TeleworkEligible": True}}}}


def main():
    rng = random.Random(20261001)
    pools = {(s, t): postings(rng, s, 25 if s == "greenhouse" else 14) for s, boards in BOARDS.items() for t in boards}
    fed = postings(rng, "usajobs", 12)
    runs = []
    for r in range(RUNS):
        resp = {"usajobs": [usajobs_item(p) for p in fed if p["open_from_run"] <= r <= p["open_to_run"]]}
        for s, boards in BOARDS.items():
            resp[s] = {}
            for token, name in boards.items():
                live = [p for p in pools[(s, token)] if p["open_from_run"] <= r <= p["open_to_run"]]
                if s == "lever" and r == RUNS - 3:
                    live = live[:2]  # simulated outage: the quality monitor should flag this run
                make = {"greenhouse": greenhouse_job, "lever": lever_job, "ashby": ashby_job}[s]
                resp[s][token] = {"name": name, "jobs": [make(p, token) for p in live]}
        runs.append({"days_ago": RUNS - 1 - r, "responses": resp})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(runs, sort_keys=True, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")

    # One more run after the eight, for the snapshot test: two retitles, a salary change, three closures.
    nxt = json.loads(json.dumps(runs[-1]))
    nxt["days_ago"] = 0
    gh = nxt["responses"]["greenhouse"]
    for job in gh["solarco"]["jobs"][:2]:
        job["title"] += " II"
    gh["gridworks"]["jobs"] = gh["gridworks"]["jobs"][:-3]
    for comp in nxt["responses"]["ashby"]["voltline"]["jobs"]:
        if comp["compensation"]["summaryComponents"]:
            comp["compensation"]["summaryComponents"][0]["maxValue"] += 10000
            break
    CHANGES.write_text(json.dumps([nxt], sort_keys=True, separators=(",", ":")))
    print(f"wrote {CHANGES}")


if __name__ == "__main__":
    main()
