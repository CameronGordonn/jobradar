"""Lever public Postings API adapter (no auth needed).

Find a company's site name from its careers URL:  jobs.lever.co/<site>
Verify with:  python -m jobradar.cli probe lever <site>
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterator

import requests

from ..models import Job
from .text import html_to_text

log = logging.getLogger(__name__)
BASE = "https://api.lever.co/v0/postings/{site}"
_US = {"US", "USA", None, ""}


def _location(j: dict) -> str:
    cats = j.get("categories") or {}
    locs = cats.get("allLocations") or ([cats["location"]] if cats.get("location") else [])
    loc = "; ".join(locs)
    # Only tag remote as allowed-anywhere when it's US remote; "Remote" alone would pass the location filter.
    if j.get("workplaceType") == "remote" and j.get("country") in _US and "remote" not in loc.lower():
        loc = f"Remote (US); {loc}" if loc else "Remote (US)"
    return loc


def parse_job(j: dict, site: str, display_name: str | None) -> Job:
    parts = [j.get("descriptionPlain") or html_to_text(j.get("description", ""))]
    for lst in j.get("lists") or []:
        parts.append(f"{lst.get('text', '')}:\n{html_to_text(lst.get('content', ''))}")
    parts.append(j.get("additionalPlain") or "")
    created = j.get("createdAt")
    return Job(
        source="lever",
        source_id=f"{site}:{j['id']}",
        title=j.get("text", ""),
        org=display_name or site,
        location=_location(j),
        url=j.get("hostedUrl", ""),
        description="\n\n".join(p.strip() for p in parts if p and p.strip()),
        posted=datetime.fromtimestamp(created / 1000, timezone.utc).isoformat() if created else None,
        extra={"country": j.get("country") or None,
               "team": (j.get("categories") or {}).get("team"),
               "commitment": (j.get("categories") or {}).get("commitment"),
               "workplace": j.get("workplaceType")},
    )


def fetch_board(site: str, display_name: str | None = None,
                session: requests.Session | None = None) -> Iterator[Job]:
    s = session or requests.Session()
    r = s.get(BASE.format(site=site), params={"mode": "json"}, timeout=30)
    if r.status_code == 404:
        log.warning("lever site %r not found (wrong name or company left Lever)", site)
        return
    r.raise_for_status()
    jobs = r.json()
    log.info("lever %s: %d jobs", site, len(jobs))
    for j in jobs:
        yield parse_job(j, site, display_name)


def fetch(cfg: dict, session: requests.Session | None = None) -> Iterator[Job]:
    for b in cfg.get("boards", []):
        try:
            yield from fetch_board(b["token"], b.get("name"), session)
        except requests.RequestException as e:
            log.error("lever %s failed: %s", b["token"], e)
