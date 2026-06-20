#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Options-features archive via RunPod serverless (replaces the on-box nightly sweep)
# ==============================================================================
# Triggers the RunPod options-archive worker (same endpoint as sports; type
# "options_archive"), waits for it, then ingests the day's parquet from S3 into
# Postgres. The worker snapshots the options features + writes parquet to
# s3://$S3_BUCKET/options_archive/dt=<date>/; this side (in the divination
# container, which has DATABASE_URL) upserts it into options_features_daily.
#
# This offloads the ~90 min / multi-GB nightly sweep off the EC2 box. The worker
# defaults to yfinance (Black-Scholes greeks) since it has no Schwab token; set a
# token + QUANT_OPTIONS_SOURCE=schwab on the endpoint for real greeks.
#
# IMPORTANT: when you switch to this, DISABLE the on-box archive so they don't
# both write Postgres — set OPTIONS_ARCHIVE_ENABLED=false in .env and recreate
# divination-api.
#
# Requires in .env: RUNPOD_API_KEY, RUNPOD_SPORTS_ENDPOINT_ID, S3_BUCKET, AWS creds.
# Usage:
#   ops/refresh_options_runpod.sh                  # full universe
#   ops/refresh_options_runpod.sh '["AAPL","MSFT"]'  # subset (testing)
# ==============================================================================
REPO="${SEEKINGBETA_REPO:-/home/ec2-user/seekingbeta}"
cd "$REPO"
set -a; [ -f .env ] && . ./.env; set +a
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in .env}"
: "${RUNPOD_SPORTS_ENDPOINT_ID:?set RUNPOD_SPORTS_ENDPOINT_ID in .env}"
BASE="https://api.runpod.ai/v2/${RUNPOD_SPORTS_ENDPOINT_ID}"
DC="docker compose"; $DC version >/dev/null 2>&1 || DC="docker-compose"
TICKERS_JSON="${1:-null}"
POLL_INTERVAL="${RUNPOD_POLL_INTERVAL_SEC:-30}"
POLL_MAX="${RUNPOD_OPTIONS_POLL_MAX:-260}"   # 260 * 30s ≈ 2.1h ceiling (full-universe sweep)

if [ "$TICKERS_JSON" = "null" ]; then
  BODY='{"input":{"type":"options_archive"}}'
else
  BODY="{\"input\":{\"type\":\"options_archive\",\"tickers\":$TICKERS_JSON}}"
fi
echo "[runpod-options] $(date -u +%FT%TZ) submitting options_archive (tickers=$TICKERS_JSON)"
RESP=$(curl -s -X POST "$BASE/run" -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" -d "$BODY")
JOB_ID=$(printf '%s' "$RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("id",""))' 2>/dev/null || true)
[ -n "$JOB_ID" ] || { echo "[runpod-options] no job id; response: $RESP"; exit 1; }
echo "[runpod-options] job id: $JOB_ID; polling every ${POLL_INTERVAL}s..."

STATUS=""
for i in $(seq 1 "$POLL_MAX"); do
  sleep "$POLL_INTERVAL"
  ST=$(curl -s "$BASE/status/$JOB_ID" -H "Authorization: Bearer $RUNPOD_API_KEY")
  STATUS=$(printf '%s' "$ST" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("status",""))' 2>/dev/null || true)
  echo "  [$i] status=$STATUS"
  case "$STATUS" in
    COMPLETED) printf '%s' "$ST" | python3 -m json.tool 2>/dev/null | tail -25 || true; break;;
    FAILED|CANCELLED|TIMED_OUT) echo "[runpod-options] job $STATUS: $ST"; exit 1;;
  esac
done
[ "$STATUS" = "COMPLETED" ] || { echo "[runpod-options] timed out after $((POLL_INTERVAL*POLL_MAX))s"; exit 1; }

echo "[runpod-options] ingest parquet -> Postgres (in divination container)"
$DC exec -T divination-api python scripts/ingest_options_archive.py
echo "[runpod-options] $(date -u +%FT%TZ) done"
