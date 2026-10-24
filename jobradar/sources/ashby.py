"""Ashby public Job Board API adapter (no auth needed).

Find a company's board name from its careers URL:  jobs.ashbyhq.com/<name>
Verify with:  python -m jobradar.cli probe ashby <name>
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

import requests

from ..models import Job
from .text import html_to_text

log = logging.getLogger(__name__)
BASE = "https://api.ashbyhq.com/posting-api/job-board/{name}"
_US = {"united states", "usa", "us", ""}


def _country(loc: dict) -> str:
    return (((loc.get("address") or {}).get("postalAddress") or {}).get("addressCountry") or "").strip().lower()


def _location(j: dict) -> str:
    locs = [j.get("location") or ""] + [s.get("location", "") for s in j.get("secondaryLocations") or []]
    loc = "; ".join(x for x in locs if x)
    # Only tag remote as allowed-anywhere when it's US remote; "Remote" alone would pass the location filter.
    if j.get("isRemote") and _country(j) in _US and "remote" not in loc.lower():
        loc = f"Remote (US); {loc}" if loc else "Remote (US)"
    return loc


def parse_job(j: dict, name: str, display_name: str | None) -> Job:
    return Job(
        source="ashby",
        source_id=f"{name}:{j['id']}",
        title=j.get("title", ""),
        org=display_name or name,
        location=_location(j),
        url=j.get("jobUrl", ""),
        description=(j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml", ""))).strip(),
        posted=j.get("publishedAt"),
        extra={"country": _country(j) or None, "department": j.get("department"), "team": j.get("team"),
               "employment_type": j.get("employmentType"), "workplace": j.get("workplaceType")},
    )


def fetch_board(name: str, display_name: str | None = None,
                session: requests.Session | None = None) -> Iterator[Job]:
    for j in fetch_board_raw(name, session):
        yield parse_job(j, name, display_name)


def fetch_board_raw(name: str, session: requests.Session | None = None) -> Iterator[dict]:
    s = session or requests.Session()
    # includeCompensation keeps salary in the raw payload for the warehouse
    r = s.get(BASE.format(name=name), params={"includeCompensation": "true"}, timeout=30)
    if r.status_code == 404:
        log.warning("ashby board %r not found (wrong name or company left Ashby)", name)
        return
    r.raise_for_status()
    jobs = [j for j in r.json().get("jobs", []) if j.get("isListed", True)]
    log.info("ashby %s: %d jobs", name, len(jobs))
    yield from jobs


def parse(item: dict, board: str, board_name: str | None = None) -> Job:
    return parse_job(item, board, board_name)


def fetch(cfg: dict, session: requests.Session | None = None) -> Iterator[Job]:
    for board, name, item in fetch_raw(cfg, session):
        yield parse(item, board, name)


def fetch_raw(cfg: dict, session: requests.Session | None = None) -> Iterator[tuple[str, str | None, dict]]:
    """Yield (board token, display name, raw posting) for every configured board."""
    for b in cfg.get("boards", []):
        try:
            for item in fetch_board_raw(b["token"], session):
                yield b["token"], b.get("name"), item
        except requests.RequestException as e:
            log.error("ashby %s failed: %s", b["token"], e)
