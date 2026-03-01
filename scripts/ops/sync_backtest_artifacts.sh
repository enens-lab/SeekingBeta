#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOURCE_DIR="${SOURCE_DIR:-$ROOT_DIR/backtest_results}"
DEST_DIR="${DEST_DIR:-$ROOT_DIR/pythia_prophecy/data/backtests}"
RESTART_PROPHECY_API="${RESTART_PROPHECY_API:-true}"
PROPHECY_SERVICE="${PROPHECY_SERVICE:-prophecy-api}"

declare -A REQUIRED_FILES=(
  ["production_trade_log.csv"]="production_trade_log.csv"
  ["production_metrics.json"]="production_metrics.json"
  ["jackpot_trade_log.csv"]="jackpot_trade_log.csv"
  ["jackpot_metrics.json"]="jackpot_metrics.json"
)

fail() {
  echo "[sync-backtests] ERROR: $1" >&2
  exit 2
}

if [[ ! -d "$SOURCE_DIR" ]]; then
  fail "source directory does not exist: $SOURCE_DIR"
fi

mkdir -p "$DEST_DIR"

for source_name in "${!REQUIRED_FILES[@]}"; do
  source_path="$SOURCE_DIR/$source_name"
  if [[ ! -f "$source_path" ]]; then
    fail "missing source file: $source_path"
  fi
  if [[ ! -s "$source_path" ]]; then
    fail "source file is empty: $source_path"
  fi
done

echo "[sync-backtests] source=$SOURCE_DIR"
echo "[sync-backtests] destination=$DEST_DIR"

for source_name in "${!REQUIRED_FILES[@]}"; do
  target_name="${REQUIRED_FILES[$source_name]}"
  source_path="$SOURCE_DIR/$source_name"
  target_path="$DEST_DIR/$target_name"
  install -m 0644 "$source_path" "$target_path"
  echo "[sync-backtests] copied $source_name -> $target_name"
done

ls -lah "$DEST_DIR" | sed -n '1,40p'

if [[ "$RESTART_PROPHECY_API" == "true" ]]; then
  echo "[sync-backtests] restarting ${PROPHECY_SERVICE} to clear in-memory performance cache"
  docker compose up -d --force-recreate "$PROPHECY_SERVICE"
fi

echo "[sync-backtests] done"
