#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE_PATH="${COMPOSE_FILE_PATH:-$ROOT_DIR/docker-compose.yml}"
FRONTEND_NGINX_PATH="${FRONTEND_NGINX_PATH:-$ROOT_DIR/pythia_prophecy/frontend/nginx.conf}"
BACKTEST_DIR="${BACKTEST_DIR:-$ROOT_DIR/pythia_prophecy/data/backtests}"
REQUIRED_BACKTEST_FILES="${REQUIRED_BACKTEST_FILES:-production_trade_log.csv production_metrics.json jackpot_trade_log.csv jackpot_metrics.json}"
CHECK_LOCAL_CERT_FILES="${CHECK_LOCAL_CERT_FILES:-false}"
TLS_CERT_DIR="${TLS_CERT_DIR:-/etc/letsencrypt/live/seekingbeta.ai}"

fail() {
  echo "[pre-deploy-check] ERROR: $1" >&2
  exit 2
}

if [[ ! -f "$COMPOSE_FILE_PATH" ]]; then
  fail "compose file not found at $COMPOSE_FILE_PATH"
fi

if [[ ! -f "$FRONTEND_NGINX_PATH" ]]; then
  fail "frontend nginx config not found at $FRONTEND_NGINX_PATH"
fi

if [[ ! -d "$BACKTEST_DIR" ]]; then
  fail "backtest directory not found at $BACKTEST_DIR"
fi

read -r -a required_backtest_array <<< "$REQUIRED_BACKTEST_FILES"
missing_backtests=()
for file_name in "${required_backtest_array[@]}"; do
  if [[ ! -f "$BACKTEST_DIR/$file_name" ]]; then
    missing_backtests+=("$file_name")
  fi
done

if [[ "${#missing_backtests[@]}" -gt 0 ]]; then
  fail "missing backtest files: ${missing_backtests[*]} (dir: $BACKTEST_DIR)"
fi

if ! grep -q "./pythia_prophecy/data/backtests:/app/data/backtests:ro" "$COMPOSE_FILE_PATH"; then
  fail "docker-compose.yml is missing required backtest mount: ./pythia_prophecy/data/backtests:/app/data/backtests:ro"
fi

if grep -q "listen 443 ssl;" "$FRONTEND_NGINX_PATH"; then
  if ! grep -q '"443:443"' "$COMPOSE_FILE_PATH"; then
    fail "nginx requires TLS but compose is missing frontend port mapping 443:443"
  fi

  if ! grep -q "/etc/letsencrypt:/etc/letsencrypt:ro" "$COMPOSE_FILE_PATH"; then
    fail "nginx requires TLS certs but compose is missing letsencrypt mount"
  fi

  if [[ "$CHECK_LOCAL_CERT_FILES" == "true" ]]; then
    if [[ ! -f "$TLS_CERT_DIR/fullchain.pem" ]]; then
      fail "missing cert file: $TLS_CERT_DIR/fullchain.pem"
    fi

    if [[ ! -f "$TLS_CERT_DIR/privkey.pem" ]]; then
      fail "missing cert file: $TLS_CERT_DIR/privkey.pem"
    fi
  fi
fi

echo "[pre-deploy-check] OK: required backtests and deploy mounts are present"
