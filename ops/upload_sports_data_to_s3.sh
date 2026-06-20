#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Upload normalized sport data + model artifacts to S3 for the RunPod sports worker
# ==============================================================================
# The RunPod sports-export worker (runpod/sports/handler.py) runs in RunPod's cloud
# and has no access to the EC2 filesystem, so it pulls its inputs from S3:
#   - normalized data: s3://$S3_BUCKET/$SPORTS_DATA_S3_PREFIX/<sport>/...
#   - model artifacts: s3://$S3_BUCKET/artifacts/artifacts/...   (same layout as
#     scripts/download_artifacts.sh)
# This script (run ON the EC2 box, where the data is produced) syncs both up.
#
# Needs WRITE creds to the bucket — the `pythia-artifact-uploader` user, NOT the
# read-only app user (see ops/README-iam-artifacts.md). Export them for this shell:
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1
#
# Usage:
#   ops/upload_sports_data_to_s3.sh                 # all sports + artifacts
#   ops/upload_sports_data_to_s3.sh mlb             # one sport's data + artifacts
#   ops/upload_sports_data_to_s3.sh "mlb pga" --no-artifacts
# ==============================================================================
REPO="${SEEKINGBETA_REPO:-/home/ec2-user/seekingbeta}"
cd "$REPO"
# Credentials: prefer creds already exported in the shell (e.g. the dedicated
# pythia-artifact-uploader keys); otherwise fall back to the box .env (pythia-app,
# which now has write access). Without this the aws CLI uses the ambient/default
# chain and can fail with InvalidAccessKeyId.
if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then set -a; [ -f .env ] && . ./.env; set +a; fi
BUCKET="${S3_BUCKET:-pythia-ml-artifacts}"
REGION="${AWS_REGION:-us-east-1}"
DATA_PREFIX="${SPORTS_DATA_S3_PREFIX:-sports-data}"

# data/sports subdir names (NOT the board prefixes): tennis lives under wta, golf under pga
SPORTS="${1:-mlb soccer pga wta basketball football olympics}"
WITH_ARTIFACTS=1
[ "${2:-}" = "--no-artifacts" ] && WITH_ARTIFACTS=0

for sub in $SPORTS; do
  src="pythia_divination/data/sports/$sub/"
  if [ ! -d "$src" ]; then echo "[upload] skip $sub (no $src)"; continue; fi
  echo "[upload] $src  ->  s3://$BUCKET/$DATA_PREFIX/$sub/"
  aws s3 sync "$src" "s3://$BUCKET/$DATA_PREFIX/$sub/" --region "$REGION" --only-show-errors
done

if [ "$WITH_ARTIFACTS" = "1" ]; then
  if [ -d pythia_divination/artifacts ]; then
    echo "[upload] artifacts  ->  s3://$BUCKET/artifacts/artifacts/"
    aws s3 sync pythia_divination/artifacts/ "s3://$BUCKET/artifacts/artifacts/" --region "$REGION" --only-show-errors
  else
    echo "[upload] skip artifacts (no pythia_divination/artifacts)"
  fi
fi
echo "[upload] done"
