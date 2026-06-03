#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Tennis refresh — cron wrapper for EC2.
# ==============================================================================
# The divination code only has its Python deps (pandas/torch/sklearn/scipy)
# INSIDE the seekingbeta-divination-api image, and the export must write to the
# host's pythia_prophecy/frontend/src/data so the prophecy container serves it.
#
# So we run the refresh in a throwaway divination container with the HOST repo
# bind-mounted over /app's repo root, writing real files on the host. Then we
# bounce prophecy-api so it serves the refreshed board.
#
# Installed as a weekly cron (see TENNIS_AND_TLS_RUNBOOK.md).
# ==============================================================================

REPO="${SEEKINGBETA_REPO:-/home/ec2-user/seekingbeta}"
IMAGE="${DIVINATION_IMAGE:-seekingbeta-divination-api}"
cd "$REPO"
DC="docker compose"; $DC version >/dev/null 2>&1 || DC="docker-compose"

echo "[refresh_tennis_cron] $(date -u +%FT%TZ) start"

# Run ingest -> build dataset -> export inside the divination image, with the
# host repo mounted so outputs land in pythia_prophecy/frontend/src/data on disk.
# -w sets the working dir to the divination package root inside the mount.
docker run --rm \
  -v "$REPO":/work \
  -w /work/pythia_divination \
  "$IMAGE" \
  bash -lc '
    set -e
    python -m sports.wta.ingest --start-year 2020 --end-year "$(date -u +%Y)" --force
    python -m sports.wta.build_training_dataset
    python scripts/export_wta_frontend_data.py
  '

echo "[refresh_tennis_cron] redeploy prophecy-api to serve refreshed board"
$DC up -d --build prophecy-api

echo "[refresh_tennis_cron] $(date -u +%FT%TZ) done"
