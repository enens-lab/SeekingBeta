#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WINDOW_MINUTES="${WINDOW_MINUTES:-10}"
HTTP_5XX_THRESHOLD="${HTTP_5XX_THRESHOLD:-20}"
WEBHOOK_FAILURE_THRESHOLD="${WEBHOOK_FAILURE_THRESHOLD:-1}"
MAX_ALLOWED_RESTARTS="${MAX_ALLOWED_RESTARTS:-0}"
ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"

SINCE_ARG="${WINDOW_MINUTES}m"
SERVICES=(prophecy-api divination-api frontend)
CONTAINERS=(pythia-prophecy pythia-divination pythia-frontend pythia-postgres)

logs="$(docker compose logs --since "$SINCE_ARG" "${SERVICES[@]}" 2>/dev/null || true)"

http_5xx_count="$(printf "%s\n" "$logs" | grep -E "\"status_code\": 5[0-9][0-9]" -c || true)"
webhook_failure_count="$(printf "%s\n" "$logs" | grep -Ei \
  "Stripe webhook processing failed|Invalid Stripe webhook signature|POST /api/webhooks/stripe - 5[0-9]{2}|POST /api/webhooks/ses-sns - 5[0-9]{2}" -c || true)"

restart_alert=0
restart_lines=()
for container in "${CONTAINERS[@]}"; do
  if docker inspect "$container" >/dev/null 2>&1; then
    count="$(docker inspect -f '{{.RestartCount}}' "$container" 2>/dev/null || echo 0)"
    if [[ "$count" -gt "$MAX_ALLOWED_RESTARTS" ]]; then
      restart_alert=1
      restart_lines+=("${container}=${count}")
    fi
  fi
done

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

summary="status=${status} window=${WINDOW_MINUTES}m http_5xx=${http_5xx_count} webhook_failures=${webhook_failure_count} restarts=[${restart_lines[*]:-none}] reasons=[${reasons[*]:-none}]"
echo "[runtime-alerts] ${summary}"

if [[ -n "$ALERT_WEBHOOK_URL" && "$status" == "alert" ]]; then
  payload="$(printf '{"text":"%s"}' "$(printf "%s" "$summary" | sed 's/"/\\"/g')")"
  curl -fsS -X POST -H "Content-Type: application/json" \
    -d "$payload" "$ALERT_WEBHOOK_URL" >/dev/null || true
fi

if [[ "$status" == "alert" ]]; then
  exit 2
fi
