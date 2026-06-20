#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Soccer-only static-board refresh — daily cron wrapper for EC2.
# ==============================================================================
# The weekly all-sports refresh (refresh_sports_cron.sh, Mondays) is too coarse
# during the FIFA World Cup, where matches finish DAILY: a finished match would
# not appear in the historical feed for up to a week. This job refreshes ONLY
# soccer (live Dixon-Coles upcoming boards + finished World Cup matches scored as
# out-of-sample backtest boards) and rebuilds prophecy so the completed boards
# land within ~a day of each match.
#
# Intended to run daily for the duration of the tournament window
# (see WORLD_CUP_WINDOW in scripts/export_soccer_frontend_data.py). It is safe to
# leave installed year-round — off-tournament it just refreshes league boards —
# but it can be removed once the World Cup ends; the weekly job keeps the rest.
#
# Same mechanics as refresh_sports_cron.sh: the divination Python deps live ONLY
# in the seekingbeta-divination-api image, and the export must write to the host
# repo's pythia_prophecy/frontend/src/data, so we run the export in a throwaway
# divination container with the HOST repo bind-mounted, then rebuild prophecy
# (which bakes frontend/src/data into its image at build time).
#
# Idempotent; safe to re-run.
# ==============================================================================

REPO="${SEEKINGBETA_REPO:-/home/ec2-user/seekingbeta}"
IMAGE="${DIVINATION_IMAGE:-seekingbeta-divination-api}"
cd "$REPO"
DC="docker compose"; $DC version >/dev/null 2>&1 || DC="docker-compose"

echo "[refresh_soccer_cron] $(date -u +%FT%TZ) start"

# HARD MEMORY CAP (--memory=3g --memory-swap=3g): confine the throwaway export
# container so a buggy export can never consume all host RAM/swap and global-OOM the
# instance (this took the live site down on 2026-06-19 via an uncapped export). The
# runaway process is OOM-killed inside its own 3 GB cgroup instead; --memory-swap=3g
# (== --memory) disables swap for the container so it can't thrash. See
# refresh_sports_cron.sh for the full incident note.
docker run --rm \
  --memory=3g --memory-swap=3g \
  -v "$REPO":/work \
  -w /work/pythia_divination \
  "$IMAGE" \
  bash -lc '
    set +e
    echo "--- soccer ---"
    python scripts/export_soccer_frontend_data.py || echo "soccer refresh FAILED"
  '

# prophecy bakes frontend/src/data into its image at build time and serves from
# that copy (NOT the host dir), so a plain restart will not pick up the freshly
# written JSON — the image must be rebuilt.
echo "[refresh_soccer_cron] rebuild prophecy-api to bake in the refreshed soccer boards"
# Narrow rebuild: build ONLY prophecy-api, then recreate it with --no-deps. `up --build
# prophecy-api` also rebuilds the heavy divination-api image, which fails pip install on a
# tight disk and leaves the boards stale (this silently broke sports refreshes).
$DC build prophecy-api && $DC up -d --no-deps prophecy-api

# prophecy-api was just recreated with a NEW container IP. nginx (frontend) caches
# upstream IPs at config-load, so without this it keeps proxying to the dead IP and
# 502s every API call until restarted. Graceful reload re-resolves; fall back to restart.
$DC exec -T frontend nginx -s reload || $DC restart frontend

echo "[refresh_soccer_cron] $(date -u +%FT%TZ) done"
