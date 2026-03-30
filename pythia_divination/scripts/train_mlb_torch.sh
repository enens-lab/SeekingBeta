#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-torch"
PYTHON_EXE="${PYTHON_EXE:-${VENV_DIR}/bin/python}"

if [ ! -x "${PYTHON_EXE}" ]; then
  echo "[train-mlb-torch] python executable not found: ${PYTHON_EXE}" >&2
  echo "[train-mlb-torch] run scripts/setup_pga_torch_env.sh first" >&2
  exit 1
fi

cd "${ROOT_DIR}"

SEASON_START="${SEASON_START:-2020}"
SEASON_END="${SEASON_END:-2025}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
HIDDEN_WIDTH="${HIDDEN_WIDTH:-512}"
DEVICE="${DEVICE:-auto}"

exec "${PYTHON_EXE}" -m sports.mlb.train_torch \
  --rebuild-dataset \
  --season-start "${SEASON_START}" \
  --season-end "${SEASON_END}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-width "${HIDDEN_WIDTH}" \
  --device "${DEVICE}" \
  -v \
  "$@"
