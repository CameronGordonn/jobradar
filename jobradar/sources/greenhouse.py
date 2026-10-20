"""Greenhouse public Job Board API adapter (no auth needed).

Find a company's board token from its careers URL:
  boards.greenhouse.io/<token>  or  job-boards.greenhouse.io/<token>
Verify with:  python -m jobradar.cli probe-greenhouse <token>
"""
from __future__ import annotations

import logging
from typing import Iterator

import requests

from ..models import Job
from .text import html_to_text

log = logging.getLogger(__name__)
BASE = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"


def parse_job(j: dict, token: str, display_name: str | None) -> Job:
    return Job(
        source="greenhouse",
        source_id=f"{token}:{j['id']}",
        title=j.get("title", ""),
        org=display_name or j.get("company_name") or token,
        location=(j.get("location") or {}).get("name", ""),
        url=j.get("absolute_url", ""),
        description=html_to_text(j.get("content", "")),
        posted=j.get("first_published") or j.get("updated_at"),
        extra={"departments": [d.get("name") for d in j.get("departments", [])]},
    )


def fetch_board(token: str, display_name: str | None = None,
                session: requests.Session | None = None) -> Iterator[Job]:
    s = session or requests.Session()
    r = s.get(BASE.format(token=token), params={"content": "true"}, timeout=30)
    if r.status_code == 404:
        log.warning("greenhouse board %r not found (wrong token or company left Greenhouse)", token)
        return
    r.raise_for_status()
    jobs = r.json().get("jobs", [])
    log.info("greenhouse %s: %d jobs", token, len(jobs))
    for j in jobs:
        yield parse_job(j, token, display_name)


def fetch(cfg: dict, session: requests.Session | None = None) -> Iterator[Job]:
    for b in cfg.get("boards", []):
        if b["token"].startswith("REPLACE"):
            continue
        try:
            yield from fetch_board(b["token"], b.get("name"), session)
        except requests.RequestException as e:
            log.error("greenhouse %s failed: %s", b["token"], e)
