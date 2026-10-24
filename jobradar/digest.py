from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def write(con, cfg: dict) -> Path | None:
    rows = con.query(
        """SELECT j.*, s.fit, s.recommend, s.variant, s.evidence, s.gaps, s.dealbreakers, s.pitch
           FROM jobs j JOIN scores s ON s.key = j.key
           LEFT JOIN digested d ON d.key = j.key
           WHERE d.key IS NULL AND s.fit >= ? AND s.recommend != 'skip'
           ORDER BY s.fit DESC, j.recall DESC
           LIMIT ?""",
        (cfg.get("min_fit", 6), cfg.get("top_n", 10)),
    )
    if not rows:
        return None

    today = date.today().isoformat()
    lines = [f"# Job digest {today}", "", f"{len(rows)} new roles above fit {cfg.get('min_fit', 6)}.", ""]
    for i, r in enumerate(rows, 1):
        gaps = json.loads(r["gaps"] or "[]")
        deal = json.loads(r["dealbreakers"] or "[]")
        lines += [
            f"## {i}. {r['title']} | {r['org']}",
            f"**Fit {r['fit']}/10, {r['recommend']}** | {r['location']} | resume: `{r['variant']}`"
            + (f" | closes {r['closes'][:10]}" if r["closes"] else ""),
            "",
            f"{r['url']}",
            "",
            f"- Strongest match: {r['evidence']}",
            f"- Gaps: {'; '.join(gaps) if gaps else 'none flagged'}",
        ]
        if deal:
            lines.append(f"- Possible dealbreakers: {'; '.join(deal)}")
        lines += [f"- Opening line: {r['pitch']}", ""]

    out = Path(cfg.get("out_dir", "digests"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{today}.md"
    path.write_text("\n".join(lines))
    con.insert_rows("digested", [(r["key"], today) for r in rows])
    return path
