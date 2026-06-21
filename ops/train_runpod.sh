#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Model training via RunPod serverless (worker #3) — CPU, same endpoint as sports.
# ==============================================================================
# Triggers the RunPod 'train' job, waits for it, then pulls the freshly-trained
# artifacts from S3 to the box and restarts divination so /predict serves them.
# The worker runs `python -m models.train`, writes artifacts to S3
# (artifacts/artifacts/), and never touches Postgres (dummy DATABASE_URL, Yahoo data).
#
# Requires in .env: RUNPOD_API_KEY, RUNPOD_SPORTS_ENDPOINT_ID, S3_BUCKET, AWS creds.
# Usage:
#   ops/train_runpod.sh                                   # --all
#   ops/train_runpod.sh '{"model":"lstm_5d","task":"classifier"}'
#   ops/train_runpod.sh '{"all":true,"tickers":["AAPL","MSFT"]}'
# ==============================================================================
REPO="${SEEKINGBETA_REPO:-/home/ec2-user/seekingbeta}"
cd "$REPO"
set -a; [ -f .env ] && . ./.env; set +a
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in .env}"
: "${RUNPOD_SPORTS_ENDPOINT_ID:?set RUNPOD_SPORTS_ENDPOINT_ID in .env}"
BASE="https://api.runpod.ai/v2/${RUNPOD_SPORTS_ENDPOINT_ID}"
BUCKET="${S3_BUCKET:-pythia-ml-artifacts}"
REGION="${AWS_REGION:-us-east-1}"
DC="docker compose"; $DC version >/dev/null 2>&1 || DC="docker-compose"
SPEC="${1:-{\"all\":true}}"
POLL_INTERVAL="${RUNPOD_POLL_INTERVAL_SEC:-30}"
POLL_MAX="${RUNPOD_TRAIN_POLL_MAX:-150}"   # 150 * 30s = 75 min ceiling

BODY=$(python3 -c 'import sys,json; d=json.loads(sys.argv[1]); d["type"]="train"; print(json.dumps({"input":d}))' "$SPEC")
echo "[runpod-train] $(date -u +%FT%TZ) submitting: $BODY"
RESP=$(curl -s -X POST "$BASE/run" -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" -d "$BODY")
JOB_ID=$(printf '%s' "$RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("id",""))' 2>/dev/null || true)
[ -n "$JOB_ID" ] || { echo "[runpod-train] no job id; response: $RESP"; exit 1; }
echo "[runpod-train] job id: $JOB_ID; polling every ${POLL_INTERVAL}s..."

STATUS=""
for i in $(seq 1 "$POLL_MAX"); do
  sleep "$POLL_INTERVAL"
  ST=$(curl -s "$BASE/status/$JOB_ID" -H "Authorization: Bearer $RUNPOD_API_KEY")
  STATUS=$(printf '%s' "$ST" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("status",""))' 2>/dev/null || true)
  echo "  [$i] status=$STATUS"
  case "$STATUS" in
    COMPLETED) printf '%s' "$ST" | python3 -m json.tool 2>/dev/null | tail -30 || true; break;;
    FAILED|CANCELLED|TIMED_OUT) echo "[runpod-train] job $STATUS: $ST"; exit 1;;
  esac
done
[ "$STATUS" = "COMPLETED" ] || { echo "[runpod-train] timed out after $((POLL_INTERVAL*POLL_MAX))s"; exit 1; }

echo "[runpod-train] pull retrained artifacts -> box (volume mount) + restart divination"
aws s3 sync "s3://$BUCKET/artifacts/artifacts/" pythia_divination/artifacts/ --region "$REGION" --only-show-errors
$DC up -d --no-deps divination-api 2>&1 | tail -2
echo "[runpod-train] $(date -u +%FT%TZ) done"
