"""Stage 0: cheap hard filters.

Bias these toward leniency. A job wrongly dropped here is never seen again,
while a job wrongly kept only costs one rerank call.
"""
from __future__ import annotations

import re

from .models import Job

# "3+ years of experience", "5 or more years of relevant experience", "2 years experience"
_YEARS = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:or more\s+)?years?\s+(?:of\s+)?(?:[a-z/\-]+\s+){0,4}?experience",
    re.I,
)
# Alternate paths that lower the bar: "..., or 2 years with a quantitative degree"
_ALT_YEARS = re.compile(r"\bor\s+(?:[a-z]+\s+\+\s+)?(\d{1,2})\s*\+?\s*years?\b", re.I)
_US_COUNTRY = {"us", "usa", "united states", "united states of america"}
_VAGUE = {"remote", "anywhere"}  # allow-list words that say nothing about the country


def _word_hit(needle: str, hay: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(needle.lower())}(?![a-z0-9])", hay.lower()) is not None


def min_years_required(text: str) -> int | None:
    """Smallest experience requirement mentioned. Using the minimum keeps
    'BS + 3 years or MS + 1 year' style postings in."""
    hits = [int(m.group(1)) for m in _YEARS.finditer(text)]
    hits += [int(m.group(1)) for m in _ALT_YEARS.finditer(text)]
    hits = [h for h in hits if 0 < h < 30]
    return min(hits) if hits else None


def check(job: Job, cfg: dict) -> str | None:
    """Return a rejection reason, or None if the job passes."""
    for kw in cfg.get("exclude_title", []):
        if _word_hit(kw, job.title):
            return f"title:{kw}"

    allow = cfg.get("location_allow", [])
    if allow and not any(_word_hit(a, job.location) for a in allow):
        return "location"
    # "Remote - Canada" passes on "Remote". Reject a non-US place unless a concrete US location is also listed.
    blocked = [b for b in cfg.get("location_block", []) if _word_hit(b, job.location)]
    if blocked and not any(_word_hit(a, job.location) for a in allow if a.lower() not in _VAGUE):
        return f"location:{blocked[0]}"
    # A bare "Remote" passes location_allow, so trust the source's country field when it has one.
    country = (job.extra.get("country") or "").strip().lower()
    if country and country not in _US_COUNTRY:
        return f"location:country={country}"

    for kw in cfg.get("exclude_text", []):
        if kw.lower() in job.description.lower():
            return f"text:{kw}"

    max_yrs = cfg.get("max_years_required")
    yrs = min_years_required(job.description)
    if max_yrs is not None and yrs is not None and yrs > max_yrs:
        return f"years:{yrs}"

    if job.source == "usajobs":
        return _check_usajobs(job, cfg.get("usajobs", {}))
    return None


def _check_usajobs(job: Job, cfg: dict) -> str | None:
    paths = set(job.extra.get("hiring_paths") or [])
    allowed = set(cfg.get("allowed_hiring_paths", []))
    if allowed and paths and not (paths & allowed):
        return f"hiring_path:{','.join(sorted(paths))}"

    max_low = cfg.get("max_low_grade")
    low = job.extra.get("low_grade")
    if max_low is not None and low not in (None, ""):
        try:
            if int(low) > max_low:
                return f"grade:{low}"
        except ValueError:
            pass
    return None
