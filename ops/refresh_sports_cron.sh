#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Sports static-board refresh — cron wrapper for EC2.
# ==============================================================================
# soccer + mlb + tennis are served from PRECOMPUTED static JSON (the BFF reads
# files; it does NOT recompute per request — that was the cause of sitewide
# sports slowness / 504s under load). This job regenerates those static files.
#
# The divination Python deps (pandas/torch/sklearn/scipy) live ONLY inside the
# seekingbeta-divination-api image, and the exports must write to the host's
# pythia_prophecy/frontend/src/data so prophecy serves them. So we run the
# exports in a throwaway divination container with the HOST repo bind-mounted,
# then restart prophecy to pick up the refreshed files.
#
# Installed as a cron (see TENNIS_AND_TLS_RUNBOOK.md). Idempotent; safe to re-run.
# ==============================================================================

REPO="${SEEKINGBETA_REPO:-/home/ec2-user/seekingbeta}"
IMAGE="${DIVINATION_IMAGE:-seekingbeta-divination-api}"
cd "$REPO"
DC="docker compose"; $DC version >/dev/null 2>&1 || DC="docker-compose"

echo "[refresh_sports_cron] $(date -u +%FT%TZ) start"

# One container run does all three exports. Each export step is allowed to fail
# without aborting the others (a single sport's upstream hiccup shouldn't block
# the rest), but we surface failures in the log.
docker run --rm \
  -v "$REPO":/work \
  -w /work/pythia_divination \
  "$IMAGE" \
  bash -lc '
    set +e
    echo "--- tennis ---"
    python -m sports.wta.ingest --start-year 2020 --end-year "$(date -u +%Y)" --force \
      && python -m sports.wta.build_training_dataset \
      && python scripts/export_wta_frontend_data.py || echo "tennis refresh FAILED"
    echo "--- soccer ---"
    python scripts/export_soccer_frontend_data.py || echo "soccer refresh FAILED"
    echo "--- mlb ---"
    python scripts/export_mlb_frontend_data.py || echo "mlb refresh FAILED"
  '

echo "[refresh_sports_cron] restart prophecy-api to serve refreshed boards"
$DC restart prophecy-api

echo "[refresh_sports_cron] $(date -u +%FT%TZ) done"
