#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BACKUP_DIR="${BACKUP_DIR:-$ROOT_DIR/backups/postgres}"
BACKUP_FILE="${1:-}"

if [[ -z "$BACKUP_FILE" ]]; then
  BACKUP_FILE="$(ls -1t "$BACKUP_DIR"/pythia_sqlite_*.db.gz 2>/dev/null | head -1 || true)"
fi

if [[ -z "$BACKUP_FILE" || ! -f "$BACKUP_FILE" ]]; then
  echo "[sqlite-restore-test] no sqlite backup file found"
  exit 1
fi

TMP_DB="$(mktemp /tmp/pythia_sqlite_restore_test.XXXXXX.db)"
trap 'rm -f "$TMP_DB"' EXIT

gunzip -c "$BACKUP_FILE" > "$TMP_DB"

python3 - "$TMP_DB" <<'PY'
import sqlite3
import sys

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)
integrity = conn.execute("PRAGMA integrity_check;").fetchone()
tables = conn.execute(
    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
).fetchone()
conn.close()

if not integrity or str(integrity[0]).lower() != "ok":
    raise SystemExit("[sqlite-restore-test] integrity check failed")

table_count = int(tables[0]) if tables else 0
if table_count == 0:
    raise SystemExit("[sqlite-restore-test] restored sqlite backup has no user tables")

print(f"[sqlite-restore-test] success: integrity=ok tables={table_count}")
PY
