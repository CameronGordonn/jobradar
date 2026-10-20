"""Check which candidate companies have a live Lever or Ashby board, and how many
of their jobs pass the hard filters. Run from the repo root:

    .venv/bin/python scripts/probe_candidates.py
"""
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jobradar import cli, filters  # noqa: E402

logging.disable(logging.CRITICAL)

# Energy, climate, geospatial, CV and sports companies that weren't on Greenhouse.
CANDIDATES = """aurorasolar arcadia palmetto kevala voltus camusenergy fervoenergy formenergy terabase heliogen
span leap crusoe watttime gridstatus octopusenergy stem enpal sunrun sense bidgely weavegrid gridmatic
pachama nearmap mapbox carto satellogic earthdaily overstory ecopia
teamworks zelusanalytics sportradar tomorrowio jupiterintelligence windborne windbornesystems
salientpredictions climateai rhodiumgroup carbonplan watershed persefoni
braincorp monarchtractor iunu farmwise bluerivertechnology
""".split()


def probe(args):
    source, token = args
    try:
        jobs = list(cli.BOARDS[source](token))
    except Exception:
        return source, token, None, None
    ok = [j for j in jobs if filters.check(j, cfg["filters"]) is None]
    return source, token, len(jobs), len(ok)


cfg = cli.load_config("config.yaml")
work = [(s, t) for t in CANDIDATES for s in ("lever", "ashby")]
with ThreadPoolExecutor(12) as ex:
    results = [r for r in ex.map(probe, work) if r[2]]
for source, token, n, ok in sorted(results, key=lambda r: -r[3]):
    print(f"{source:7} {token:22} {n:>5} total {ok:>4} pass filters")
