from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


@dataclass
class Job:
    source: str
    source_id: str
    title: str
    org: str
    location: str
    url: str
    description: str
    posted: str | None = None
    closes: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Row identity: stable across edits to the title or location."""
        return f"{self.source}:{self.source_id}"

    @property
    def dedupe_key(self) -> str:
        """Same org + title + location, even on several boards or under several ids.
        Only one posting per dedupe key is sent to Claude."""
        raw = "|".join(_norm(x) for x in (self.org, self.title, self.location))
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def text(self) -> str:
        return f"{self.title}\n{self.org}\n{self.location}\n\n{self.description}"
