#!/usr/bin/env bash
# Daily entry point: ingest -> score -> digest. Run by a scheduler (cron, or
# Windows Task Scheduler calling into WSL). Keys come from .env, which is gitignored.
#
# Spend is bounded by recall.top_k in config.yaml: at most that many Claude calls per run.
# flock stops two runs overlapping if a missed run fires late while another is going.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data
LOG="data/run.log"

exec 9>data/run.lock
if ! flock -n 9; then
  echo "$(date -Is) skipped: another run holds the lock" >> "$LOG"
  exit 0
fi

# shellcheck disable=SC1091
. ./.env
echo "$(date -Is) start" >> "$LOG"
rc=0
.venv/bin/python -m jobradar.cli run >> "$LOG" 2>&1 || rc=$?
if [ "$rc" -eq 0 ]; then
  echo "$(date -Is) ok" >> "$LOG"
else
  echo "$(date -Is) FAILED (exit $rc)" >> "$LOG"
  exit "$rc"
fi
