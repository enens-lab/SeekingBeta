#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_EXE="${PYTHON_EXE:-${VENV_DIR}/bin/python}"

if [ ! -x "${PYTHON_EXE}" ]; then
  echo "[train-basketball-torch] python executable not found: ${PYTHON_EXE}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

LEAGUE="${LEAGUE:-nba}"
EPOCHS="${EPOCHS:-35}"
BATCH_SIZE="${BATCH_SIZE:-512}"
HIDDEN_WIDTH="${HIDDEN_WIDTH:-384}"
DEVICE="${DEVICE:-auto}"

exec "${PYTHON_EXE}" -m sports.basketball.train_torch \
  --league "${LEAGUE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-width "${HIDDEN_WIDTH}" \
  --device "${DEVICE}" \
  -v \
  "$@"
