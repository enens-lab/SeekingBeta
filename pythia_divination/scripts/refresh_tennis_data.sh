#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Tennis data refresh — keeps the curated WTA/ATP board current.
# ==============================================================================
# The live ESPN feed (prophecy: _apply_live_tennis_schedule) only corrects the
# DATES of tournaments already on the curated board. To pick up NEW tournaments
# and refresh model predictions/fields, the curated board itself must be rebuilt
# from fresh match data. This script does the full chain:
#
#   1. ingest latest ATP/WTA matches (Jeff Sackmann GitHub)
#   2. rebuild the normalized training dataset
#   3. re-export the frontend JSON (wta_upcoming_tournaments.json + backtests)
#
# It does NOT retrain the ranker model (artifacts are reused); add a train step
# if you want fresh weights. Run weekly via cron (see TENNIS_AND_TLS_RUNBOOK.md).
#
# Usage (from the divination dir, with its venv):
#   pythia_divination/scripts/refresh_tennis_data.sh [START_YEAR] [END_YEAR]
# ==============================================================================

DIV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIV_DIR"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
fi

START_YEAR="${1:-2020}"
END_YEAR="${2:-$(date -u +%Y)}"

echo "[refresh_tennis] $(date -u +%FT%TZ) ingest matches ${START_YEAR}-${END_YEAR} (ATP+WTA)"
"$PY" -m sports.wta.ingest --start-year "$START_YEAR" --end-year "$END_YEAR" --force

echo "[refresh_tennis] rebuild normalized training dataset"
"$PY" -m sports.wta.build_training_dataset

echo "[refresh_tennis] re-export frontend JSON"
"$PY" scripts/export_wta_frontend_data.py

echo "[refresh_tennis] done. Restart/redeploy prophecy-api (or rely on its file mtime cache) to serve the refreshed board."
