#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-torch"
PYTHON_EXE="${PYTHON_EXE:-${VENV_DIR}/bin/python}"

if [ ! -x "${PYTHON_EXE}" ]; then
  echo "[train-pga-multitask-torch] python executable not found: ${PYTHON_EXE}" >&2
  echo "[train-pga-multitask-torch] run scripts/setup_pga_torch_env.sh first" >&2
  exit 1
fi

cd "${ROOT_DIR}"

SEASON_START="${SEASON_START:-2021}"
SEASON_END="${SEASON_END:-2026}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-12}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LSTM_UNITS="${LSTM_UNITS:-192}"
STATIC_WIDTH="${STATIC_WIDTH:-384}"
DENSE_WIDTH="${DENSE_WIDTH:-384}"
DEVICE="${DEVICE:-auto}"
PROFILE_WORKERS="${PROFILE_WORKERS:-12}"

exec "${PYTHON_EXE}" -m sports.pga.train_multitask_torch \
  --rebuild-dataset \
  --season-start "${SEASON_START}" \
  --season-end "${SEASON_END}" \
  --profile-workers "${PROFILE_WORKERS}" \
  --targets won top_10 made_cut \
  --sequence-length "${SEQUENCE_LENGTH}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lstm-units "${LSTM_UNITS}" \
  --static-width "${STATIC_WIDTH}" \
  --dense-width "${DENSE_WIDTH}" \
  --device "${DEVICE}" \
  -v \
  "$@"
