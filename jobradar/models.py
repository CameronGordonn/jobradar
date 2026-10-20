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
        """Cross-source dedupe key: same org + title + location collapses to one row,
        even if the posting shows up on several boards."""
        raw = "|".join(_norm(x) for x in (self.org, self.title, self.location))
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def text(self) -> str:
        return f"{self.title}\n{self.org}\n{self.location}\n\n{self.description}"
