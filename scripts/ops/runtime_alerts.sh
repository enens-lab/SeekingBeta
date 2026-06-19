#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WINDOW_MINUTES="${WINDOW_MINUTES:-10}"
HTTP_5XX_THRESHOLD="${HTTP_5XX_THRESHOLD:-20}"
WEBHOOK_FAILURE_THRESHOLD="${WEBHOOK_FAILURE_THRESHOLD:-1}"
MAX_ALLOWED_RESTARTS="${MAX_ALLOWED_RESTARTS:-0}"
ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"
# End-to-end synthetic probe: hits the public endpoints THROUGH nginx (like a real
# user), so it catches gateway-level outages (e.g. nginx 502 from a stale upstream
# IP) that the log-based 5xx check misses because the app itself logged 200s.
SYNTHETIC_PROBE="${SYNTHETIC_PROBE:-true}"
PROBE_BASE="${PROBE_BASE:-https://localhost}"

# Fall back to the webhook in .env (gitignored) without sourcing the whole file, so
# the cron line stays unchanged and the secret never lands in git.
if [[ -z "$ALERT_WEBHOOK_URL" && -f .env ]]; then
  ALERT_WEBHOOK_URL="$(grep -E '^ALERT_WEBHOOK_URL=' .env 2>/dev/null | head -1 | cut -d= -f2- | sed -e 's/^["'"'"']//' -e 's/["'"'"']$//')"
fi

SINCE_ARG="${WINDOW_MINUTES}m"
SERVICES=(prophecy-api divination-api frontend)
CONTAINERS=(pythia-prophecy pythia-divination pythia-frontend pythia-postgres)

logs="$(docker compose logs --since "$SINCE_ARG" "${SERVICES[@]}" 2>/dev/null || true)"

http_5xx_count="$(printf "%s\n" "$logs" | grep -E "\"status_code\": 5[0-9][0-9]" -c || true)"
webhook_failure_count="$(printf "%s\n" "$logs" | grep -Ei \
  "Stripe webhook processing failed|Invalid Stripe webhook signature|POST /api/webhooks/stripe - 5[0-9]{2}|POST /api/webhooks/ses-sns - 5[0-9]{2}" -c || true)"

window_seconds=$((WINDOW_MINUTES * 60))
now_epoch="$(date -u +%s)"
restart_alert=0
restart_lines=()
for container in "${CONTAINERS[@]}"; do
  if docker inspect "$container" >/dev/null 2>&1; then
    count="$(docker inspect -f '{{.RestartCount}}' "$container" 2>/dev/null || echo 0)"
    started="$(docker inspect -f '{{.State.StartedAt}}' "$container" 2>/dev/null || echo '')"
    started_epoch="$(date -u -d "$started" +%s 2>/dev/null || echo 0)"
    age=$((now_epoch - started_epoch))
    # Only alert on a restart WITHIN the window. A clean deploy resets RestartCount
    # to 0 (won't trip), and an old one-off restart ages out of the window — so this
    # flags active flapping, not stale cumulative counts.
    if [[ "$count" -gt "$MAX_ALLOWED_RESTARTS" && "$started_epoch" -gt 0 && "$age" -lt "$window_seconds" ]]; then
      restart_alert=1
      restart_lines+=("${container}=${count}@${age}s")
    fi
  fi
done

# --- End-to-end synthetic probe (through nginx) ---
synthetic_failures=0
synthetic_lines=()
if [[ "$SYNTHETIC_PROBE" == "true" ]]; then
  probe_code() { curl -sk -o /dev/null -w "%{http_code}" --max-time 15 "$@" 2>/dev/null; }
  # Healthy = a real HTTP response below 500. Down = 000 (no connection / timeout)
  # or any 5xx. Retry up to 3x so a one-off blip doesn't flap an alert.
  check_endpoint() {
    local label="$1"; shift
    local code=""
    for _ in 1 2 3; do
      code="$(probe_code "$@" || true)"; code="${code:-000}"
      if [[ "$code" != "000" && "$code" -lt 500 ]]; then return 0; fi
      sleep 2
    done
    synthetic_failures=$((synthetic_failures + 1))
    synthetic_lines+=("${label}=${code}")
  }
  # Critical user paths: sign-in (the App Store rejection vector) + sports boards
  # (both via prophecy-api) and a prediction (divination, nginx-cached).
  check_endpoint login "$PROBE_BASE/api/auth/login" -X POST -H "Content-Type: application/json" -d '{}'
  check_endpoint sports "$PROBE_BASE/api/sports/boards?sports=golf&include_backtests=false"
  check_endpoint predict "$PROBE_BASE/predict/lstm_5d/AAPL"
fi

status="ok"
reasons=()

if [[ "$http_5xx_count" -ge "$HTTP_5XX_THRESHOLD" ]]; then
  status="alert"
  reasons+=("5xx_spike")
fi

if [[ "$webhook_failure_count" -ge "$WEBHOOK_FAILURE_THRESHOLD" ]]; then
  status="alert"
  reasons+=("webhook_failures")
fi

if [[ "$restart_alert" -eq 1 ]]; then
  status="alert"
  reasons+=("container_restarts")
fi

if [[ "$synthetic_failures" -ge 1 ]]; then
  status="alert"
  reasons+=("endpoint_down")
fi

summary="status=${status} window=${WINDOW_MINUTES}m http_5xx=${http_5xx_count} webhook_failures=${webhook_failure_count} restarts=[${restart_lines[*]:-none}] probe=[${synthetic_lines[*]:-ok}] reasons=[${reasons[*]:-none}]"
echo "[runtime-alerts] ${summary}"

if [[ -n "$ALERT_WEBHOOK_URL" && "$status" == "alert" ]]; then
  # Discord webhooks use {"content":...}; Slack/generic use {"text":...}.
  if [[ "$ALERT_WEBHOOK_URL" == *"discord.com"* || "$ALERT_WEBHOOK_URL" == *"discordapp.com"* ]]; then
    key="content"
  else
    key="text"
  fi
  payload="$(printf '{"%s":"🚨 SeekingBeta alert — %s"}' "$key" "$(printf "%s" "$summary" | sed 's/"/\\"/g')")"
  curl -fsS -X POST -H "Content-Type: application/json" \
    -d "$payload" "$ALERT_WEBHOOK_URL" >/dev/null || true
fi

if [[ "$status" == "alert" ]]; then
  exit 2
fi
