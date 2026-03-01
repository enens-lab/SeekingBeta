#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BACKTEST_SCRIPT_PATH="${BACKTEST_SCRIPT_PATH:-/home/ec2-user/Stock_Prediction_Model/run_backtest_v4_1.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKTEST_CHOICE="${BACKTEST_CHOICE:-3}"  # 1=production, 2=jackpot, 3=both

if [[ ! -f "$BACKTEST_SCRIPT_PATH" ]]; then
  echo "[refresh-backtests] ERROR: backtest script not found at $BACKTEST_SCRIPT_PATH" >&2
  exit 2
fi

BACKTEST_SCRIPT_DIR="$(cd "$(dirname "$BACKTEST_SCRIPT_PATH")" && pwd)"
BACKTEST_WORKDIR="${BACKTEST_WORKDIR:-$BACKTEST_SCRIPT_DIR}"
SOURCE_DIR_DEFAULT="${BACKTEST_WORKDIR}/backtest_results"
SOURCE_DIR="${SOURCE_DIR:-$SOURCE_DIR_DEFAULT}"

if [[ ! -d "$BACKTEST_WORKDIR" ]]; then
  echo "[refresh-backtests] ERROR: backtest workdir not found at $BACKTEST_WORKDIR" >&2
  exit 2
fi

echo "[refresh-backtests] running $BACKTEST_SCRIPT_PATH with choice=$BACKTEST_CHOICE"
echo "[refresh-backtests] workdir=$BACKTEST_WORKDIR"
(
  cd "$BACKTEST_WORKDIR"
  printf '%s\n' "$BACKTEST_CHOICE" | "$PYTHON_BIN" "$BACKTEST_SCRIPT_PATH"
)

echo "[refresh-backtests] syncing artifacts from $SOURCE_DIR"
SOURCE_DIR="$SOURCE_DIR" bash scripts/ops/sync_backtest_artifacts.sh

echo "[refresh-backtests] done"
