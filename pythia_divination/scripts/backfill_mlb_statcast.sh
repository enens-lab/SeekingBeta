#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-${ROOT_DIR}/.venv/bin/python}"

if [ ! -x "${PYTHON_EXE}" ]; then
  echo "[backfill-mlb-statcast] python executable not found: ${PYTHON_EXE}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

declare -A START_DATES=(
  [2020]="2020-07-23"
  [2021]="2021-04-01"
  [2022]="2022-04-07"
  [2023]="2023-03-30"
  [2024]="2024-03-28"
  [2025]="2025-03-18"
)

declare -A END_DATES=(
  [2020]="2020-09-27"
  [2021]="2021-10-03"
  [2022]="2022-10-05"
  [2023]="2023-10-01"
  [2024]="2024-09-30"
  [2025]="2025-09-30"
)

SEASON_START="${SEASON_START:-2020}"
SEASON_END="${SEASON_END:-2024}"
CHUNK_DAYS="${CHUNK_DAYS:-7}"

for ((season=SEASON_START; season<=SEASON_END; season++)); do
  start="${START_DATES[$season]:-}"
  end="${END_DATES[$season]:-}"
  if [ -z "${start}" ] || [ -z "${end}" ]; then
    echo "[backfill-mlb-statcast] no configured Statcast window for season ${season}" >&2
    exit 1
  fi

  echo "[backfill-mlb-statcast] collecting ${season} (${start} -> ${end})"
  "${PYTHON_EXE}" -m sports.mlb.collect_statcast \
    --start-date "${start}" \
    --end-date "${end}" \
    --chunk-days "${CHUNK_DAYS}" \
    "$@"
done
