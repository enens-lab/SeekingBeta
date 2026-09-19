#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Export everything a NEW box needs that is not in git, as one tarball.
# ==============================================================================
# Run on the OLD box (EC2) as ec2-user, from the repo:
#   bash ops/export_state_bundle.sh                # -> /home/ec2-user/seekingbeta-state-<ts>.tgz
#   bash ops/export_state_bundle.sh --consistent   # stops the app containers first (~1 min),
#                                                  # so the Postgres/SQLite copies are exact
#
# Contents (all relative to the repo unless noted):
#   .env                                   secrets + config (never in git)
#   pythia_prophecy/secrets/               Play service account, Apple .p8
#   pythia_prophecy/frontend/src/data/     sports boards (118 MB; or re-sync from S3)
#   pythia_prophecy/data/predictions/      today's stock sweep
#   pythia_divination/artifacts/           model artifacts, ONLY with INCLUDE_ARTIFACTS=true
#                                          (393 MB; the web path no longer reads them)
#   images.tgz                             docker save of seekingbeta-prophecy-api,
#                                          seekingbeta-frontend, postgres:16-alpine, so the
#                                          2 GB target never has to `docker compose build`
#                                          (INCLUDE_IMAGES=false to skip)
#   volumes/postgres_data.tgz              docker volume seekingbeta_postgres_data (users' compliance
#                                          prefs, price cache, billing events, ...)
#   volumes/prophecy_data.tgz              docker volume seekingbeta_prophecy_data (SQLite users DB)
#   letsencrypt.tgz                        /etc/letsencrypt (certs + renewal conf, root-owned)
#   crontab.txt                            ec2-user crontab
#   home/archive/, home/logs/              graded-record archive, cron logs (small)
#
# The tarball contains live secrets: scp it straight to the new box over SSH and
# delete both copies afterwards. Do not put it in S3 unencrypted.
# ==============================================================================
REPO="${SEEKINGBETA_REPO:-/home/ec2-user/seekingbeta}"
cd "$REPO"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="${OUT:-/home/ec2-user/seekingbeta-state-${STAMP}.tgz}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
DC="docker compose"; $DC version >/dev/null 2>&1 || DC="docker-compose"
PROJECT="$(basename "$REPO")"   # compose project name = directory name -> volume prefix

log() { echo "[export-state] $(date -u +%FT%TZ) $*"; }

if [ "${1:-}" = "--consistent" ]; then
  log "stopping app containers for a consistent copy (frontend keeps serving the static site)"
  $DC stop prophecy-api postgres >/dev/null
  trap '$DC start postgres prophecy-api >/dev/null || true; rm -rf "$WORK"' EXIT
fi

mkdir -p "$WORK/bundle/volumes" "$WORK/bundle/home"
cp .env "$WORK/bundle/.env"
[ -d pythia_prophecy/secrets ] && cp -a pythia_prophecy/secrets "$WORK/bundle/secrets"
[ -d pythia_prophecy/data/predictions ] && cp -a pythia_prophecy/data/predictions "$WORK/bundle/predictions"
mkdir -p "$WORK/bundle/boards" && cp pythia_prophecy/frontend/src/data/*.json "$WORK/bundle/boards/" 2>/dev/null || true
if [ "${INCLUDE_ARTIFACTS:-false}" = "true" ] && [ -d pythia_divination/artifacts ]; then
  tar czf "$WORK/bundle/artifacts.tgz" -C pythia_divination artifacts
fi
if [ "${INCLUDE_IMAGES:-true}" = "true" ]; then
  # Ship the images that are actually running so the target does not build
  # anything: the tags compose would otherwise build (<project>-prophecy-api,
  # <project>-frontend) plus the pinned postgres. `up -d` reuses a present tag.
  IMAGES=()
  for img in "${PROJECT}-prophecy-api:latest" "${PROJECT}-frontend:latest" "postgres:16-alpine"; do
    docker image inspect "$img" >/dev/null 2>&1 && IMAGES+=("$img") || log "image $img not present; skipping"
  done
  if [ "${#IMAGES[@]}" -gt 0 ]; then
    log "docker save ${IMAGES[*]}"
    docker save "${IMAGES[@]}" | gzip -1 > "$WORK/bundle/images.tgz"
  fi
fi
crontab -l > "$WORK/bundle/crontab.txt" 2>/dev/null || true
[ -d /home/ec2-user/archive ] && cp -a /home/ec2-user/archive "$WORK/bundle/home/archive"
[ -d /home/ec2-user/logs ] && cp -a /home/ec2-user/logs "$WORK/bundle/home/logs"
[ -d logs ] && cp -a logs "$WORK/bundle/repo-logs"

for vol in postgres_data prophecy_data; do
  name="${PROJECT}_${vol}"
  docker volume inspect "$name" >/dev/null 2>&1 || { log "volume $name not found; skipping"; continue; }
  log "volume $name"
  docker run --rm -v "$name":/data:ro -v "$WORK/bundle/volumes":/out alpine sh -c "tar czf /out/${vol}.tgz -C /data ."
done

log "/etc/letsencrypt (sudo)"
sudo tar czf "$WORK/bundle/letsencrypt.tgz" -C /etc letsencrypt
sudo chown ec2-user:ec2-user "$WORK/bundle/letsencrypt.tgz"

printf 'exported_at=%s\nsource_host=%s\nrepo_commit=%s\nproject=%s\n' "$STAMP" "$(hostname)" "$(git rev-parse --short HEAD)" "$PROJECT" > "$WORK/bundle/MANIFEST"
tar czf "$OUT" -C "$WORK" bundle
chmod 600 "$OUT"
log "wrote $OUT ($(du -h "$OUT" | cut -f1))"
log "next: scp -i <key> $OUT ec2-user@<new-ip>:/home/ec2-user/ && (on the new box) bash ops/import_state_bundle.sh /home/ec2-user/$(basename "$OUT")"
