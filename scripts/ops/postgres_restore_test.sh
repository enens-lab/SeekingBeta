#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BACKUP_DIR="${BACKUP_DIR:-$ROOT_DIR/backups/postgres}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
SERVICE_NAME="${SERVICE_NAME:-postgres}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESTORE_DB="pythia_restore_test_${TIMESTAMP//[-:T]/}"
RESTORE_DB="${RESTORE_DB//Z/}"

BACKUP_FILE="${1:-}"
if [[ -z "$BACKUP_FILE" ]]; then
  BACKUP_FILE="$(ls -1t "$BACKUP_DIR"/pythia_*.dump.gz 2>/dev/null | head -1 || true)"
fi

if [[ -z "$BACKUP_FILE" || ! -f "$BACKUP_FILE" ]]; then
  echo "[restore-test] no backup file found"
  exit 1
fi

echo "[restore-test] using backup=${BACKUP_FILE}"

cleanup() {
  docker compose exec -T "$SERVICE_NAME" \
    psql -U "$POSTGRES_USER" -d postgres -v ON_ERROR_STOP=1 \
    -c "DROP DATABASE IF EXISTS \"${RESTORE_DB}\";" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker compose exec -T "$SERVICE_NAME" \
  psql -U "$POSTGRES_USER" -d postgres -v ON_ERROR_STOP=1 \
  -c "DROP DATABASE IF EXISTS \"${RESTORE_DB}\";"

docker compose exec -T "$SERVICE_NAME" \
  psql -U "$POSTGRES_USER" -d postgres -v ON_ERROR_STOP=1 \
  -c "CREATE DATABASE \"${RESTORE_DB}\";"

gunzip -c "$BACKUP_FILE" | docker compose exec -T "$SERVICE_NAME" sh -lc \
  "cat >/tmp/restore_test.dump && pg_restore -U '$POSTGRES_USER' -d '$RESTORE_DB' --no-owner --no-privileges /tmp/restore_test.dump && rm -f /tmp/restore_test.dump"

TABLE_COUNT="$(docker compose exec -T "$SERVICE_NAME" \
  psql -U "$POSTGRES_USER" -d "$RESTORE_DB" -Atc \
  "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';" | tr -d '\r')"

if [[ -z "$TABLE_COUNT" || "$TABLE_COUNT" -eq 0 ]]; then
  echo "[restore-test] failed: restored DB has no public tables"
  exit 1
fi

echo "[restore-test] success: restored db=${RESTORE_DB} public_tables=${TABLE_COUNT}"
