#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Restore a state bundle (ops/export_state_bundle.sh) onto a NEW box and start.
# ==============================================================================
# Run on the NEW box as its login user (ubuntu / ec2-user) after ops/lightsail_bootstrap.sh:
#   bash ops/import_state_bundle.sh ~/seekingbeta-state-<ts>.tgz
#
# Restores .env, secrets, boards, predictions, (artifacts if bundled), the
# prebuilt images, both docker volumes (Postgres + SQLite), /etc/letsencrypt,
# the crontab and the small home dirs, then starts postgres, prophecy-api and
# frontend (no divination: stock predictions are the daily RunPod sweep, served
# from a file). Nothing is built when the bundle carries images.
# Refuses to overwrite a volume that already has data unless FORCE=1.
# ==============================================================================
BUNDLE="${1:?path to seekingbeta-state-<ts>.tgz}"
HOME_DIR="$(getent passwd "$(id -un)" | cut -d: -f6)"
REPO="${SEEKINGBETA_REPO:-$HOME_DIR/seekingbeta}"
cd "$REPO"
# compose project name = directory name -> image tags <project>-prophecy-api and
# volumes <project>_postgres_data, the same on the old box (also "seekingbeta").
PROJECT="$(basename "$REPO")"
DC="docker compose"; $DC version >/dev/null 2>&1 || DC="docker-compose"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

log() { echo "[import-state] $(date -u +%FT%TZ) $*"; }

# Everything on the old box is addressed as /home/ec2-user/...: the crontab, the
# ops scripts' REPO default, the certbot renewal hooks. On a box whose user is
# not ec2-user, a symlink keeps all of that valid unchanged.
if [ "$HOME_DIR" != "/home/ec2-user" ] && [ ! -e /home/ec2-user ]; then
  sudo ln -s "$HOME_DIR" /home/ec2-user && log "symlinked /home/ec2-user -> $HOME_DIR"
fi

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
# in the renewal conf). The timer is certbot.timer on Ubuntu (apt package) and
# certbot-renew.timer on Amazon Linux.
TIMER_OK=0
for t in certbot.timer certbot-renew.timer; do
  if sudo systemctl enable --now "$t" >/dev/null 2>&1; then log "  renewal timer: $t"; TIMER_OK=1; break; fi
done
[ "$TIMER_OK" = 1 ] || log "  WARN: no certbot timer enabled (check: systemctl list-timers --all | grep certbot)"
sudo certbot renew --dry-run >/dev/null 2>&1 && log "  certbot renew --dry-run: OK" || log "  WARN: certbot renew --dry-run failed (fine before DNS points here; re-run after cutover)"

if [ -f "$B/images.tgz" ]; then
  log "docker load (prebuilt images from the old box; nothing is built here)"
  gunzip -c "$B/images.tgz" | docker load | sed 's/^/  /'
else
  log "no images in bundle -> building prophecy-api + frontend here (slow on 2 GB)"
  $DC build prophecy-api frontend 2>&1 | grep -E "naming to|ERROR" || true
fi
docker pull -q alpine >/dev/null 2>&1 || true

# Let compose create the named volumes (with its labels) BEFORE restoring into
# them: a volume created by hand makes `up` complain it "was not created by
# Docker Compose". --no-start creates containers + volumes without running them.
log "compose create (volumes + containers, not started)"
$DC up --no-start --no-build postgres prophecy-api frontend 2>&1 | grep -vE '^\s*$' | sed 's/^/  /' || true

log "docker volumes"
for vol in postgres_data prophecy_data; do
  src="$B/volumes/${vol}.tgz"; [ -f "$src" ] || { log "  no $vol in bundle; skipping"; continue; }
  name="${PROJECT}_${vol}"
  docker volume inspect "$name" >/dev/null 2>&1 || docker volume create "$name" >/dev/null
  if [ "${FORCE:-0}" != "1" ] && [ -n "$(docker run --rm -v "$name":/data alpine sh -c 'ls -A /data' 2>/dev/null)" ]; then
    log "  $name already has data; FORCE=1 to overwrite"; continue
  fi
  docker run --rm -v "$name":/data -v "$(dirname "$src")":/in:ro alpine sh -c "find /data -mindepth 1 -delete; tar xzf /in/${vol}.tgz -C /data"
  log "  restored $name ($(docker run --rm -v "$name":/data alpine sh -c 'du -sh /data | cut -f1'))"
done

log "start (postgres, prophecy-api, frontend)"
$DC up -d --no-build postgres prophecy-api frontend
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
