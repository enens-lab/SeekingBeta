#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-torch"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="${ROOT_DIR}/requirements-pga-torch.txt"

if [ ! -f "${REQ_FILE}" ]; then
  echo "[setup-pga-torch-env] missing requirements file: ${REQ_FILE}" >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[setup-pga-torch-env] creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ_FILE}"

python - <<'PY'
import sys
import torch
print('[setup-pga-torch-env] python =', sys.executable)
print('[setup-pga-torch-env] torch =', torch.__version__)
print('[setup-pga-torch-env] mps built =', torch.backends.mps.is_built())
print('[setup-pga-torch-env] mps available =', torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x = torch.randn(512, 512, device='mps')
    y = torch.randn(512, 512, device='mps')
    z = x @ y
    print('[setup-pga-torch-env] smoke device =', z.device)
else:
    print('[setup-pga-torch-env] warning: MPS is not available; training will fall back to CPU')
PY
