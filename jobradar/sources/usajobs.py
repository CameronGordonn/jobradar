"""USAJOBS Search API adapter.

Free API key: https://developer.usajobs.gov/apirequest/
Docs: https://developer.usajobs.gov/api-reference/get-api-search
Hiring path codes: https://data.usajobs.gov/api/codelist/hiringpaths
"""
from __future__ import annotations

import logging
import os
from typing import Iterator

import requests

from ..models import Job

log = logging.getLogger(__name__)
BASE = "https://data.usajobs.gov/api/search"
PAGE_SIZE = 500  # API max


def parse_item(item: dict) -> Job:
    d = item["MatchedObjectDescriptor"]
    details = d.get("UserArea", {}).get("Details", {}) or {}

    duties = details.get("MajorDuties") or []
    if isinstance(duties, str):
        duties = [duties]
    parts = [
        details.get("JobSummary", ""),
        "Major duties:\n" + "\n".join(f"- {x}" for x in duties) if duties else "",
        "Qualifications:\n" + (d.get("QualificationSummary") or ""),
        "Education:\n" + (details.get("Education") or ""),
        "Requirements:\n" + (details.get("Requirements") or ""),
    ]
    locations = [loc.get("LocationName", "") for loc in d.get("PositionLocation", [])]
    pay = (d.get("PositionRemuneration") or [{}])[0]

    return Job(
        source="usajobs",
        source_id=str(d.get("PositionID") or item.get("MatchedObjectId")),
        title=d.get("PositionTitle", ""),
        org=f"{d.get('OrganizationName', '')} ({d.get('DepartmentName', '')})",
        # "Multiple Locations" hides the real list from the location filter, so expand it.
        location=(display if (display := d.get("PositionLocationDisplay")) and display != "Multiple Locations"
                  else "; ".join(locations) or display or ""),
        url=d.get("PositionURI", ""),
        description="\n\n".join(p for p in parts if p.strip()),
        posted=d.get("PublicationStartDate"),
        closes=d.get("ApplicationCloseDate"),
        extra={
            "hiring_paths": details.get("HiringPath") or [],
            "low_grade": details.get("LowGrade"),
            "high_grade": details.get("HighGrade"),
            "pay_min": pay.get("MinimumRange"),
            "pay_max": pay.get("MaximumRange"),
            "pay_interval": pay.get("RateIntervalCode"),
        },
    )


def parse(item: dict, board: str, board_name: str | None = None) -> Job:
    return parse_item(item)


def fetch(cfg: dict, session: requests.Session | None = None) -> Iterator[Job]:
    for board, name, item in fetch_raw(cfg, session):
        yield parse(item, board, name)


def fetch_raw(cfg: dict, session: requests.Session | None = None) -> Iterator[tuple[str, None, dict]]:
    """Yield (query, None, raw item). The query stands in for a board: "keyword|location"."""
    key, email = os.environ.get("USAJOBS_API_KEY"), os.environ.get("USAJOBS_EMAIL")
    if not key or not email:
        log.warning("USAJOBS_API_KEY / USAJOBS_EMAIL not set; skipping USAJOBS")
        return
    s = session or requests.Session()
    headers = {"Host": "data.usajobs.gov", "User-Agent": email, "Authorization-Key": key}

    for kw in cfg.get("keywords", []):
        for loc in cfg.get("locations", [None]):
            page = 1
            while True:
                params = {"Keyword": kw, "ResultsPerPage": PAGE_SIZE, "Page": page,
                          "DatePosted": cfg.get("date_posted_days", 14)}
                if loc:
                    params["LocationName"] = loc
                try:
                    r = s.get(BASE, params=params, headers=headers, timeout=30)
                    r.raise_for_status()
                    result = r.json().get("SearchResult", {})
                except (requests.RequestException, ValueError) as e:
                    log.error("usajobs kw=%r loc=%r page=%d failed: %s", kw, loc, page, e)
                    break
                items = result.get("SearchResultItems", [])
                for it in items:
                    yield f"{kw}|{loc or ''}", None, it
                total = int(result.get("SearchResultCountAll", 0))
                log.info("usajobs kw=%r loc=%r page=%d got=%d total=%d", kw, loc, page, len(items), total)
                if not items or page * PAGE_SIZE >= total:
                    break
                page += 1
