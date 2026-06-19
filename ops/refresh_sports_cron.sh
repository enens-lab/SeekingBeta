#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Sports static-board refresh — cron wrapper for EC2.
# ==============================================================================
# Sports boards are served from PRECOMPUTED static JSON (the BFF reads files;
# it does NOT recompute per request — that was the cause of sitewide sports
# slowness / 504s under load). This job regenerates the static files for ALL
# sports: tennis, soccer, mlb, golf, basketball, football, olympics.
#
# NOTE: the cron entry must redirect its log somewhere ec2-user can write
# (e.g. ~/logs/refresh_sports.log). The original install pointed at
# /var/log/refresh_sports.log, which ec2-user cannot create — the shell died on
# the redirect before this script ever ran, so boards silently froze for weeks.
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
    echo "--- golf ---"
    python scripts/export_frontend_data.py || echo "golf refresh FAILED"
    echo "--- basketball ---"
    python scripts/export_basketball_frontend_data.py || echo "basketball refresh FAILED"
    echo "--- football ---"
    python scripts/export_football_frontend_data.py || echo "football refresh FAILED"
    echo "--- olympics ---"
    python scripts/export_olympics_frontend_data.py || echo "olympics refresh FAILED"
  '

# IMPORTANT: prophecy bakes frontend/src/data into the image at build time
# (Dockerfile: COPY frontend/src/data/ ./frontend_data/), and serves from that
# copy — NOT from the host dir. A plain `restart` will NOT pick up the files the
# export just wrote; the image must be rebuilt. (This is also why committing the
# static JSON to git matters: the build uses the repo copy.)
echo "[refresh_sports_cron] rebuild prophecy-api to bake in the refreshed boards"
# Narrow rebuild: build ONLY prophecy-api, then recreate it with --no-deps. `up --build
# prophecy-api` also rebuilds the heavy divination-api image, which fails pip install on a
# tight disk and leaves the boards stale (this silently broke sports refreshes).
$DC build prophecy-api && $DC up -d --no-deps prophecy-api

# prophecy-api was just recreated with a NEW container IP. nginx (frontend) caches
# upstream IPs at config-load, so without this it keeps proxying to the dead IP and
# 502s every API call until restarted. Graceful reload re-resolves; fall back to restart.
$DC exec -T frontend nginx -s reload || $DC restart frontend

echo "[refresh_sports_cron] $(date -u +%FT%TZ) done"
