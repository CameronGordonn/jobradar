# jobradar

[![tests](https://github.com/CameronGordonn/jobradar/actions/workflows/tests.yml/badge.svg)](https://github.com/CameronGordonn/jobradar/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Finds and ranks jobs across sources the big boards miss. It never applies or sends anything; it hands you a short daily list and you decide.

The shipped config pulls from USAJOBS plus 24 company boards on Greenhouse, Lever and Ashby. Cheap filters and TF-IDF recall narrow the pool, Claude reranks at most 40 jobs per run against a written rubric, and the digest shows the top 10.

```
sources ──> stage 0: hard filters ──> stage 1: recall (tfidf/embeddings) ──> stage 2: Claude rerank ──> digest
 USAJOBS      title, location,           max cosine vs. resume variants        rubric + resumes,          top N markdown
 Greenhouse   clearance, years,          top_k go to stage 2                   forced tool call           with pitch line
 Lever        GS grade, hiring path
 Ashby
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env && chmod 600 .env   # then fill in the keys with an editor
```

`.env` needs `ANTHROPIC_API_KEY`, `USAJOBS_API_KEY` (free: https://developer.usajobs.gov/apirequest/) and `USAJOBS_EMAIL`. It is gitignored.

1. Paste plain-text resumes into `resumes/` and list each one under `resumes:` in `config.yaml`.
2. Copy `rubric.example.md` to `rubric.md` and fill in the candidate section. `rubric.md` is gitignored.
3. Add Greenhouse, Lever and Ashby boards to `config.yaml`, checking each token first:
   `python -m jobradar.cli probe greenhouse|lever|ashby <token>`

## Daily use

```bash
python -m jobradar.cli run      # ingest -> score -> digest; writes digests/YYYY-MM-DD.md
```

To run it every morning, schedule `scripts/daily_run.sh`. It loads `.env`, appends to `data/run.log`, refuses to overlap a run already in progress, and exits non-zero if scoring fails (a rejected API key stops at the first call instead of writing an empty digest). Claude spend is capped at `recall.top_k` calls per run.

Cron, 7am daily (macOS/Linux):

```
0 7 * * * /path/to/jobradar/scripts/daily_run.sh
```

On Windows with WSL, use Task Scheduler instead, because WSL's cron only runs while a WSL session is open. The action is `wsl.exe -d Ubuntu -- bash -lc /path/to/jobradar/scripts/daily_run.sh`, with "run as soon as possible after a missed start" turned on.

## Calibrate before you trust it

```bash
python -m jobradar.cli ingest                              # a few days of data first
python -m jobradar.cli label -n 50                         # stratified across the recall distribution
python -m jobradar.cli label -n 20 --include-filtered      # checks whether stage 0 is too aggressive
python -m jobradar.cli eval
```

`eval` reports:
- **Filter false negatives**: jobs you'd apply to that the hard filters dropped. Any here means a rule is too tight.
- **Recall AUC**: whether stage 1 ranks your positives above negatives. If it's weak, switch `recall.backend` to `embeddings` and rerun.
- **Rerank AUC, precision/recall of "apply"**: whether Claude agrees with you. Edit the rubric where it disagrees, then re-eval. Try `claude-sonnet-5-5` against Haiku on the same labels and see if the difference justifies the cost.
- **Positives below the top-k cutoff**: good jobs that would never reach Claude. If nonzero, raise `top_k`.

The labeling sample is stratified on purpose. If you only label what the pipeline shows you, every metric looks great and tells you nothing.

## Known limitations

- TF-IDF IDF weights shift as the corpus grows, so recall scores aren't comparable across weeks. Fine for ranking within a run.
- Dedupe keys on org + title + location. The same role posted as "ML Engineer" and "Machine Learning Engineer" won't collapse.
- USAJOBS hiring-path codes are from the public codelist; confirm against real results after your first ingest.

## Next adapters (in rough order of value per hour)

- **Workday**: many utilities and large employers use it. Most tenants expose an undocumented JSON search endpoint behind the careers page (look in browser devtools, Network tab, for a POST to `.../wday/cxs/...`).
- **NEOGOV / governmentjobs.com**: San Diego County, City of San Diego, and many CA agencies.
- **CalCareers**: no API, needs a scraper. Remember that most state jobs require passing an exam to get on the eligibility list first, so pull the exam list too.

## License

MIT. See [LICENSE](LICENSE).
