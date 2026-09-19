#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Daily stock-prediction sweep via RunPod (replaces the on-box divination server)
# ==============================================================================
# Since 2026-09-19 the web box runs no model server. This script is the whole
# stock-prediction pipeline from the box's point of view:
#
#   1. collect the extra tickers the curated universe must be extended with:
#      homepage tickers, the Free/Basic tier lists, every user's watchlist
#   2. POST /run {"type":"stock_predictions","universe":"core","tickers":[...]}
#      to the RunPod sports endpoint; poll /status until COMPLETED
#   3. aws s3 cp the result (frontend-boards/stock_predictions_latest.json) to a
#      temp file, validate it (ops/validate_stock_predictions.py), then atomically
#      move it into pythia_prophecy/data/predictions/ -- which docker-compose
#      bind-mounts read-only into prophecy-api. No rebuild, no restart: the
#      store re-reads the file on mtime change.
#   4. probe /predict/homepage through prophecy and report freshness
#
# A refused file leaves the previous one in place (prophecy keeps serving it
# until STOCK_PREDICTIONS_MAX_AGE_HOURS, then answers 503 honestly).
#
# Cron (weekdays after the close; the options archive runs 21:30, keep them apart):
#   15 22 * * 1-5 cd /home/ec2-user/seekingbeta && ops/refresh_stock_predictions_runpod.sh >> logs/refresh_stock_predictions.log 2>&1
#
# Requires in $REPO/.env: RUNPOD_API_KEY, RUNPOD_SPORTS_ENDPOINT_ID, S3_BUCKET,
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION.
#
# Usage:
#   ops/refresh_stock_predictions_runpod.sh              # core universe + extras
#   ops/refresh_stock_predictions_runpod.sh --sync-only  # no RunPod job: pull + validate + install what S3 has
#   STOCK_SWEEP_LIMIT=25 ops/refresh_stock_predictions_runpod.sh   # smoke run (first 25 tickers)
# ==============================================================================
REPO="${SEEKINGBETA_REPO:-/home/ec2-user/seekingbeta}"
cd "$REPO"
set -a; [ -f .env ] && . ./.env; set +a

BUCKET="${S3_BUCKET:-pythia-ml-artifacts}"
REGION="${AWS_REGION:-us-east-1}"
BOARDS_PREFIX="${BOARDS_S3_PREFIX:-frontend-boards}"
FILE_NAME="stock_predictions_latest.json"
DEST_DIR="pythia_prophecy/data/predictions"
POLL_INTERVAL="${RUNPOD_POLL_INTERVAL_SEC:-30}"
POLL_MAX="${RUNPOD_POLL_MAX:-80}"          # 80 * 30s = 40 min ceiling (sweep deadline is 25 min)
UNIVERSE="${STOCK_SWEEP_UNIVERSE:-core}"
MODELS="${STOCK_SWEEP_MODELS:-lstm_5d,lstm_jackpot,lstm_quant}"
HOMEPAGE="${HOME_CACHE_TICKERS:-AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META}"
DC="docker compose"; $DC version >/dev/null 2>&1 || DC="docker-compose"
SYNC_ONLY=0; [ "${1:-}" = "--sync-only" ] && SYNC_ONLY=1

log() { echo "[stock-sweep] $(date -u +%FT%TZ) $*"; }

if [ "$SYNC_ONLY" -eq 0 ]; then
  : "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in .env}"
  : "${RUNPOD_SPORTS_ENDPOINT_ID:?set RUNPOD_SPORTS_ENDPOINT_ID in .env}"
  BASE="https://api.runpod.ai/v2/${RUNPOD_SPORTS_ENDPOINT_ID}"

  # --- 1. extra tickers: homepage + tier lists + user watchlists (SQLite in prophecy) ---
  # Importing api.service emits JSON log lines on stdout, so keep only the LAST
  # line (the ticker list) and let everything else go to stderr/the log.
  EXTRA_JSON=$($DC exec -T prophecy-api python - "$HOMEPAGE" <<'PY' 2>/dev/null | tail -1 || echo '[]'
import json, sqlite3, sys
tickers = [t.strip().upper() for t in sys.argv[1].split(",") if t.strip()]
try:
    from api.models import TIER_CONFIG
    from api.service import _available_stocks_for_tier
    for cfg in TIER_CONFIG.values():
        if cfg.get("stocks_limit", -1) != -1:
            tickers += [str(t).upper() for t in _available_stocks_for_tier(cfg)]
except Exception as exc:  # noqa: BLE001
    print(f"tier lists unavailable: {exc}", file=sys.stderr)
try:
    con = sqlite3.connect("/app/data/pythia.db")
    for (w,) in con.execute("SELECT watchlist FROM user_oracle"):
        try:
            tickers += [str(t).upper() for t in json.loads(w or "[]")]
        except Exception:
            pass
except Exception as exc:  # noqa: BLE001
    print(f"watchlists unavailable: {exc}", file=sys.stderr)
seen, out = set(), []
for t in tickers:
    if t and t not in seen and len(t) <= 12:
        seen.add(t); out.append(t)
print(json.dumps(out))
PY
)
  if ! printf '%s' "$EXTRA_JSON" | python3 -c 'import sys,json; v=json.load(sys.stdin); assert isinstance(v, list)' 2>/dev/null; then
    log "WARN: extra-ticker collection returned non-JSON (${EXTRA_JSON:0:80}); continuing with the homepage list only"
    EXTRA_JSON=$(printf '%s' "$HOMEPAGE" | python3 -c 'import sys,json;print(json.dumps([t.strip().upper() for t in sys.stdin.read().split(",") if t.strip()]))')
  fi
  EXTRA_COUNT=$(printf '%s' "$EXTRA_JSON" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)))')
  log "extra tickers (homepage + tiers + watchlists): $EXTRA_COUNT"

  # --- 2. submit + poll ---
  LIMIT_JSON=""; [ -n "${STOCK_SWEEP_LIMIT:-}" ] && LIMIT_JSON=",\"limit\":${STOCK_SWEEP_LIMIT}"
  MODELS_JSON=$(printf '%s' "$MODELS" | python3 -c 'import sys,json;print(json.dumps([m for m in sys.stdin.read().strip().split(",") if m]))')
  HOME_JSON=$(printf '%s' "$HOMEPAGE" | python3 -c 'import sys,json;print(json.dumps([t.strip().upper() for t in sys.stdin.read().strip().split(",") if t.strip()]))')
  BODY="{\"input\":{\"type\":\"stock_predictions\",\"universe\":\"$UNIVERSE\",\"models\":$MODELS_JSON,\"homepage_tickers\":$HOME_JSON,\"tickers\":$EXTRA_JSON$LIMIT_JSON}}"
  log "submitting job: universe=$UNIVERSE models=$MODELS extras=$EXTRA_COUNT${STOCK_SWEEP_LIMIT:+ limit=$STOCK_SWEEP_LIMIT}"
  RESP=$(curl -s -X POST "$BASE/run" -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$BODY")
  JOB_ID=$(printf '%s' "$RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("id",""))' 2>/dev/null || true)
  [ -n "$JOB_ID" ] || { log "no job id; response: $RESP"; exit 1; }
  log "job id: $JOB_ID; polling every ${POLL_INTERVAL}s"

  STATUS=""
  for i in $(seq 1 "$POLL_MAX"); do
    sleep "$POLL_INTERVAL"
    ST=$(curl -s "$BASE/status/$JOB_ID" -H "Authorization: Bearer $RUNPOD_API_KEY")
    STATUS=$(printf '%s' "$ST" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("status",""))' 2>/dev/null || true)
    echo "  [$i] status=$STATUS"
    case "$STATUS" in
      COMPLETED)
        printf '%s' "$ST" | python3 -c 'import sys,json;o=json.load(sys.stdin).get("output") or {};print("  ok=%s summary=%s boards=%s" % (o.get("ok"), json.dumps(o.get("summary")), o.get("boards"))); print((o.get("error") or "")[:300])' 2>/dev/null || true
        OK=$(printf '%s' "$ST" | python3 -c 'import sys,json;print(str((json.load(sys.stdin).get("output") or {}).get("ok")).lower())' 2>/dev/null || echo false)
        [ "$OK" = "true" ] || { log "worker reported ok=false; not installing"; exit 1; }
        break;;
      FAILED|CANCELLED|TIMED_OUT) log "job $STATUS: $ST"; exit 1;;
    esac
  done
  [ "$STATUS" = "COMPLETED" ] || { log "timed out after $((POLL_INTERVAL*POLL_MAX))s"; exit 1; }
fi

# --- 3. pull, validate, install atomically ---
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
log "pulling s3://$BUCKET/$BOARDS_PREFIX/$FILE_NAME"
aws s3 cp "s3://$BUCKET/$BOARDS_PREFIX/$FILE_NAME" "$TMP/$FILE_NAME" --region "$REGION" --only-show-errors

if ! python3 ops/validate_stock_predictions.py "$TMP/$FILE_NAME" ${STOCK_SWEEP_LIMIT:+--min-tickers 1 --min-homepage 0}; then
  log "validation REFUSED the sweep; previous file left in place"
  exit 1
fi
mkdir -p "$DEST_DIR"
cp "$TMP/$FILE_NAME" "$DEST_DIR/.$FILE_NAME.tmp" && mv -f "$DEST_DIR/.$FILE_NAME.tmp" "$DEST_DIR/$FILE_NAME"
log "installed $DEST_DIR/$FILE_NAME ($(stat -c %s "$DEST_DIR/$FILE_NAME" 2>/dev/null || stat -f %z "$DEST_DIR/$FILE_NAME") bytes)"

# --- 4. verify prophecy picked it up (mtime reload, no restart) ---
sleep 2
curl -s --max-time 20 "http://localhost:8001/predict/cache/status" \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  prophecy: mode=%s available=%s generated_at=%s age_h=%s tickers=%s models=%s" % (d.get("mode"),d.get("available"),d.get("generated_at"),d.get("age_hours"),d.get("tickers"),d.get("models")))' \
  || log "WARN: prophecy status probe failed"
curl -s --max-time 20 "http://localhost:8001/predict/homepage" \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  homepage: available=%s as_of=%s rows=%s" % (d.get("available"),d.get("as_of"),[(r["model"],r["successful_predictions"],len(r["failures"])) for r in d.get("rows",[])]))' \
  || log "WARN: homepage probe failed"
log "done"
