#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Restore a state bundle (ops/export_state_bundle.sh) onto a NEW box and start.
# ==============================================================================
# Run on the NEW box as ec2-user after ops/lightsail_bootstrap.sh:
#   bash ops/import_state_bundle.sh /home/ec2-user/seekingbeta-state-<ts>.tgz
#
# Restores .env, secrets, boards, predictions, artifacts, both docker volumes
# (Postgres + SQLite), /etc/letsencrypt, the crontab and the small home dirs,
# then builds and starts postgres, prophecy-api and frontend (no divination:
# stock predictions are the daily RunPod sweep, served from a file).
# Refuses to overwrite a volume that already has data unless FORCE=1.
# ==============================================================================
BUNDLE="${1:?path to seekingbeta-state-<ts>.tgz}"
REPO="${SEEKINGBETA_REPO:-/home/ec2-user/seekingbeta}"
cd "$REPO"
PROJECT="$(basename "$REPO")"
DC="docker compose"; $DC version >/dev/null 2>&1 || DC="docker-compose"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

log() { echo "[import-state] $(date -u +%FT%TZ) $*"; }

tar xzf "$BUNDLE" -C "$WORK"
B="$WORK/bundle"
[ -f "$B/MANIFEST" ] && { log "bundle:"; sed 's/^/  /' "$B/MANIFEST"; }

log "files"
cp "$B/.env" .env && chmod 600 .env
[ -d "$B/secrets" ] && { mkdir -p pythia_prophecy/secrets && cp -a "$B/secrets/." pythia_prophecy/secrets/ && chmod 600 pythia_prophecy/secrets/* 2>/dev/null || true; }
[ -d "$B/boards" ] && { mkdir -p pythia_prophecy/frontend/src/data && cp -a "$B/boards/." pythia_prophecy/frontend/src/data/; }
[ -d "$B/predictions" ] && { mkdir -p pythia_prophecy/data/predictions && cp -a "$B/predictions/." pythia_prophecy/data/predictions/; }
[ -f "$B/artifacts.tgz" ] && tar xzf "$B/artifacts.tgz" -C pythia_divination
[ -d "$B/home/archive" ] && cp -a "$B/home/archive" /home/ec2-user/
[ -d "$B/home/logs" ] && cp -a "$B/home/logs" /home/ec2-user/
[ -d "$B/repo-logs" ] && cp -a "$B/repo-logs/." logs/
mkdir -p logs backups/postgres /home/ec2-user/logs

log "/etc/letsencrypt"
sudo tar xzf "$B/letsencrypt.tgz" -C /etc
sudo test -f /etc/letsencrypt/live/seekingbeta.ai/fullchain.pem && log "  cert present: $(sudo openssl x509 -enddate -noout -in /etc/letsencrypt/live/seekingbeta.ai/fullchain.pem)"
# certbot renews with --standalone on :80 while nginx is stopped (pre/post hooks
# in the renewal conf); the timer is the distro's certbot-renew.timer.
sudo systemctl enable --now certbot-renew.timer >/dev/null 2>&1 || log "  WARN: certbot-renew.timer not enabled (check `systemctl list-timers`)"

log "docker volumes"
for vol in postgres_data prophecy_data; do
  src="$B/volumes/${vol}.tgz"; [ -f "$src" ] || { log "  no $vol in bundle; skipping"; continue; }
  name="${PROJECT}_${vol}"
  docker volume create "$name" >/dev/null
  if [ "${FORCE:-0}" != "1" ] && [ -n "$(docker run --rm -v "$name":/data alpine sh -c 'ls -A /data' 2>/dev/null)" ]; then
    log "  $name already has data; FORCE=1 to overwrite"; continue
  fi
  docker run --rm -v "$name":/data -v "$(dirname "$src")":/in:ro alpine sh -c "rm -rf /data/* /data/..?* /data/.[!.]* 2>/dev/null; tar xzf /in/${vol}.tgz -C /data"
  log "  restored $name"
done

log "build + start (postgres, prophecy-api, frontend)"
$DC build prophecy-api frontend 2>&1 | grep -E "naming to|ERROR" || true
$DC up -d postgres prophecy-api frontend
for i in $(seq 1 30); do $DC ps --format '{{.Name}} {{.Status}}' | grep pythia-prophecy | grep -q healthy && break; sleep 4; done
$DC ps --format 'table {{.Name}}\t{{.Status}}'

log "crontab"
if [ -f "$B/crontab.txt" ] && [ -s "$B/crontab.txt" ]; then
  crontab "$B/crontab.txt" && log "  installed $(grep -c . "$B/crontab.txt") lines"
fi

log "smoke"
curl -sk --max-time 20 https://localhost/healthz | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  healthz ok=%s predictions=%s" % (d.get("ok"), (d.get("stock_predictions") or {}).get("available")))' || log "  WARN healthz probe failed"
curl -sk --max-time 30 "https://localhost/api/sports/boards?sports=golf&include_backtests=false" -o /dev/null -w '  sports boards: HTTP %{http_code}\n' || true
curl -sk --max-time 20 https://localhost/predict/homepage | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  homepage available=%s as_of=%s" % (d.get("available"), d.get("as_of")))' || log "  WARN homepage probe failed"
log "done. Point DNS at this box, then watch: docker compose logs -f prophecy-api frontend"
