#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BACKUP_DIR="${BACKUP_DIR:-$ROOT_DIR/backups/postgres}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-pythia}"
SERVICE_NAME="${SERVICE_NAME:-postgres}"
PROPHECY_SERVICE="${PROPHECY_SERVICE:-prophecy-api}"
SQLITE_BACKUP_ENABLED="${SQLITE_BACKUP_ENABLED:-true}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$BACKUP_DIR"

OUT_DUMP="${BACKUP_DIR}/pythia_${TIMESTAMP}.dump"
OUT_GZ="${OUT_DUMP}.gz"
OUT_META="${BACKUP_DIR}/pythia_${TIMESTAMP}.meta"
SQLITE_OUT_DB="${BACKUP_DIR}/pythia_sqlite_${TIMESTAMP}.db"
SQLITE_OUT_GZ="${SQLITE_OUT_DB}.gz"

echo "[backup] starting postgres backup for db=${POSTGRES_DB} service=${SERVICE_NAME}"
docker compose ps "$SERVICE_NAME" >/dev/null

docker compose exec -T "$SERVICE_NAME" \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc > "$OUT_DUMP"

gzip -f "$OUT_DUMP"

# Validate archive readability inside postgres container.
gunzip -c "$OUT_GZ" | docker compose exec -T "$SERVICE_NAME" sh -lc \
  "cat >/tmp/backup_validate.dump && pg_restore -l /tmp/backup_validate.dump >/dev/null && rm -f /tmp/backup_validate.dump"

if command -v sha256sum >/dev/null 2>&1; then
  CHECKSUM="$(sha256sum "$OUT_GZ" | awk '{print $1}')"
else
  CHECKSUM="$(shasum -a 256 "$OUT_GZ" | awk '{print $1}')"
fi

cat > "$OUT_META" <<EOF
timestamp_utc=${TIMESTAMP}
backup_file=$(basename "$OUT_GZ")
postgres_db=${POSTGRES_DB}
postgres_user=${POSTGRES_USER}
sha256=${CHECKSUM}
size_bytes=$(wc -c < "$OUT_GZ")
EOF

find "$BACKUP_DIR" -name "pythia_*.dump.gz" -mtime +"$RETENTION_DAYS" -delete
find "$BACKUP_DIR" -name "pythia_sqlite_*.db.gz" -mtime +"$RETENTION_DAYS" -delete
find "$BACKUP_DIR" -name "pythia_*.meta" -mtime +"$RETENTION_DAYS" -delete

echo "[backup] done file=${OUT_GZ}"
echo "[backup] checksum=${CHECKSUM}"

if [[ "${SQLITE_BACKUP_ENABLED}" == "true" ]]; then
  echo "[backup] starting sqlite backup from service=${PROPHECY_SERVICE}"
  docker compose ps "$PROPHECY_SERVICE" >/dev/null

  docker compose exec -T "$PROPHECY_SERVICE" python - <<'PY'
import sqlite3
from pathlib import Path

src = Path("/app/data/pythia.db")
dst = Path("/tmp/pythia_db_backup.db")

if not src.exists():
    raise SystemExit(0)

if dst.exists():
    dst.unlink()

source_conn = sqlite3.connect(src.as_posix())
dest_conn = sqlite3.connect(dst.as_posix())
source_conn.backup(dest_conn)
dest_conn.close()
source_conn.close()

verify_conn = sqlite3.connect(dst.as_posix())
result = verify_conn.execute("PRAGMA integrity_check;").fetchone()
verify_conn.close()
if not result or str(result[0]).lower() != "ok":
    raise SystemExit("sqlite backup integrity check failed")
PY

  docker compose exec -T "$PROPHECY_SERVICE" sh -lc 'cat /tmp/pythia_db_backup.db' > "$SQLITE_OUT_DB"
  docker compose exec -T "$PROPHECY_SERVICE" rm -f /tmp/pythia_db_backup.db || true
  gzip -f "$SQLITE_OUT_DB"
  echo "[backup] sqlite done file=${SQLITE_OUT_GZ}"
fi
