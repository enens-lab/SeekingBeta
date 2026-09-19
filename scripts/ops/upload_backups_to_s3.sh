#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Mirror local Postgres/SQLite backups to S3 (offsite copy).
# ==============================================================================
# Runs hourly from the EC2 crontab right after scripts/ops/postgres_backup.sh:
#
#   S3_BUCKET=<bucket> S3_PREFIX=postgres S3_KMS_KEY_ARN=<arn> \
#     bash scripts/ops/upload_backups_to_s3.sh
#
# History: this script was referenced by the crontab but never committed, and the
# only copy lived untracked on the box. It vanished in the 2026-06-22 repo resync,
# after which every hourly run failed with "No such file or directory" and the
# offsite copy silently stopped at 2026-03-01 while local retention stayed at 3
# days. Nobody noticed because the backup step itself kept succeeding. Keep this
# file in git.
#
# Credentials: the cron does NOT source .env and ~/.aws on the box holds stale
# keys (InvalidClientTokenId), so source .env here explicitly -- the same way
# ops/refresh_sports_runpod.sh does. The pythia-app IAM user has PutObject on the
# backups bucket (that is how the March uploads landed).
# ==============================================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Import ONLY the AWS credentials from .env. Sourcing the whole file clobbered the
# S3_BUCKET the cron passes in (the .env S3_BUCKET is the ML-artifacts bucket), so
# the first run tried to write backups into pythia-ml-artifacts and was denied.
if [ -f .env ]; then
  for k in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION; do
    v="$(grep -E "^${k}=" .env | tail -1 | cut -d= -f2- | tr -d '"' || true)"
    [ -n "$v" ] && export "$k=$v"
  done
fi

S3_BUCKET="${S3_BUCKET:?S3_BUCKET is required}"
S3_PREFIX="${S3_PREFIX:-postgres}"
S3_KMS_KEY_ARN="${S3_KMS_KEY_ARN:-}"
AWS_REGION="${AWS_REGION:-us-east-2}"
BACKUP_DIR="${BACKUP_DIR:-$ROOT_DIR/backups/postgres}"

if [ ! -d "$BACKUP_DIR" ]; then
  echo "[upload] backup dir missing: $BACKUP_DIR" >&2
  exit 1
fi

SSE_ARGS=()
if [ -n "$S3_KMS_KEY_ARN" ]; then
  SSE_ARGS=(--sse aws:kms --sse-kms-key-id "$S3_KMS_KEY_ARN")
fi

# The pythia-app IAM user is allowed a plain PutObject but not the multipart
# upload flow, so anything above the CLI's 8MB default threshold (the ~13MB
# postgres dumps) was denied while the ~4MB SQLite copies went through. Raise the
# threshold via a throwaway CLI config rather than touching ~/.aws on the host.
AWS_CLI_TMP="$(mktemp -d)"; trap 'rm -rf "$AWS_CLI_TMP"' EXIT
printf '[default]\ns3 =\n  multipart_threshold = 256MB\n' > "$AWS_CLI_TMP/config"
export AWS_CONFIG_FILE="$AWS_CLI_TMP/config"

# sync is incremental: only files not yet in S3 are transferred, so this stays
# cheap even though it runs hourly. Deletions are never propagated (the IAM user
# cannot delete anyway), so S3 keeps history beyond local retention.
echo "[upload] $(date -u +%FT%TZ) syncing $BACKUP_DIR -> s3://$S3_BUCKET/$S3_PREFIX/"
aws s3 sync "$BACKUP_DIR" "s3://$S3_BUCKET/$S3_PREFIX/" \
  --region "$AWS_REGION" \
  --exclude "*" --include "pythia_*.dump.gz" --include "pythia_sqlite_*.db.gz" --include "pythia_*.meta" \
  --only-show-errors \
  "${SSE_ARGS[@]}"

NEWEST="$(ls -t "$BACKUP_DIR"/pythia_*.dump.gz 2>/dev/null | head -1 || true)"
if [ -n "$NEWEST" ]; then
  KEY="$S3_PREFIX/$(basename "$NEWEST")"
  if aws s3api head-object --bucket "$S3_BUCKET" --key "$KEY" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "[upload] ok: newest dump present in S3 ($KEY)"
  else
    echo "[upload] ERROR: newest dump not found in S3 after sync ($KEY)" >&2
    exit 1
  fi
fi
