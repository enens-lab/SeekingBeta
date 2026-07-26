#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Sports board refresh via RunPod serverless (replaces the on-box export)
# ==============================================================================
# Instead of running the heavy export inside a throwaway divination container on
# the EC2 box (the job that OOM'd the instance on 2026-06-19/20), this triggers the
# RunPod sports-export worker, waits for it, pulls the fresh board JSON from S3, and
# rebuilds prophecy. The multi-GB export memory now runs on RunPod, never on EC2.
#
# Flow:
#   1. POST /run to the RunPod endpoint with {"input":{"type":"sports_export","sports":[...]}}
#   2. poll /status/<id> until COMPLETED
#   3. aws s3 sync the fresh boards from s3://$S3_BUCKET/$BOARDS_S3_PREFIX/ into
#      pythia_prophecy/frontend/src/data/
#   4. rebuild prophecy-api (bakes the JSON) + reload nginx
#
# Requires in $REPO/.env (never committed): RUNPOD_API_KEY, RUNPOD_SPORTS_ENDPOINT_ID,
# S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION.
#
# Usage:
#   ops/refresh_sports_runpod.sh                 # default: ["mlb"]
#   ops/refresh_sports_runpod.sh '["mlb","golf"]'
#   ops/refresh_sports_runpod.sh '["all"]'
# ==============================================================================
REPO="${SEEKINGBETA_REPO:-/home/ec2-user/seekingbeta}"
cd "$REPO"
set -a; [ -f .env ] && . ./.env; set +a

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in .env}"
: "${RUNPOD_SPORTS_ENDPOINT_ID:?set RUNPOD_SPORTS_ENDPOINT_ID in .env}"
BUCKET="${S3_BUCKET:-pythia-ml-artifacts}"
REGION="${AWS_REGION:-us-east-1}"
BOARDS_PREFIX="${BOARDS_S3_PREFIX:-frontend-boards}"
SPORTS_JSON="${1:-[\"mlb\"]}"
POLL_INTERVAL="${RUNPOD_POLL_INTERVAL_SEC:-30}"
POLL_MAX="${RUNPOD_POLL_MAX:-120}"   # 120 * 30s = 60 min ceiling
BASE="https://api.runpod.ai/v2/${RUNPOD_SPORTS_ENDPOINT_ID}"
DC="docker compose"; $DC version >/dev/null 2>&1 || DC="docker-compose"

echo "[runpod-sports] $(date -u +%FT%TZ) submitting job: sports=$SPORTS_JSON"
RESP=$(curl -s -X POST "$BASE/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d "{\"input\":{\"type\":\"sports_export\",\"sports\":$SPORTS_JSON}}")
JOB_ID=$(printf '%s' "$RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("id",""))' 2>/dev/null || true)
[ -n "$JOB_ID" ] || { echo "[runpod-sports] no job id; response: $RESP"; exit 1; }
echo "[runpod-sports] job id: $JOB_ID; polling every ${POLL_INTERVAL}s..."

STATUS=""
for i in $(seq 1 "$POLL_MAX"); do
  sleep "$POLL_INTERVAL"
  ST=$(curl -s "$BASE/status/$JOB_ID" -H "Authorization: Bearer $RUNPOD_API_KEY")
  STATUS=$(printf '%s' "$ST" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("status",""))' 2>/dev/null || true)
  echo "  [$i] status=$STATUS"
  case "$STATUS" in
    COMPLETED) printf '%s' "$ST" | python3 -m json.tool 2>/dev/null | tail -40 || true; break;;
    FAILED|CANCELLED|TIMED_OUT) echo "[runpod-sports] job $STATUS: $ST"; exit 1;;
  esac
done
[ "$STATUS" = "COMPLETED" ] || { echo "[runpod-sports] timed out after $((POLL_INTERVAL*POLL_MAX))s"; exit 1; }

DATA_DIR="pythia_prophecy/frontend/src/data"

# Snapshot the currently-serving boards so a bad sync can be rolled back. The
# 2026-07-26 away-win bug shipped corrupt accuracy figures for months precisely
# because a sync had no way to be refused.
SNAPSHOT="$(mktemp -d)"
cp "$DATA_DIR"/*.json "$SNAPSHOT"/ 2>/dev/null || true

echo "[runpod-sports] syncing fresh boards: s3://$BUCKET/$BOARDS_PREFIX/ -> prophecy data dir"
aws s3 sync "s3://$BUCKET/$BOARDS_PREFIX/" "$DATA_DIR/" \
  --region "$REGION" --only-show-errors --exclude "*" --include "*.json"

echo "[runpod-sports] validating synced boards before baking them"
if ! python3 ops/validate_sports_boards.py "$DATA_DIR"; then
  echo "[runpod-sports] validation FAILED -- rolling back to the previously served boards"
  cp "$SNAPSHOT"/*.json "$DATA_DIR"/ 2>/dev/null || true
  rm -rf "$SNAPSHOT"
  echo "[runpod-sports] boards restored; prophecy-api NOT rebuilt (still serving good data)"
  exit 1
fi
rm -rf "$SNAPSHOT"

echo "[runpod-sports] rebuild prophecy-api (bake boards) + reload nginx"
$DC build prophecy-api && $DC up -d --no-deps prophecy-api
$DC exec -T frontend nginx -s reload || $DC restart frontend
echo "[runpod-sports] $(date -u +%FT%TZ) done"
