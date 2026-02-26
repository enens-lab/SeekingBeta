#!/bin/bash
set -euo pipefail

BUCKET_NAME="${S3_BUCKET:-pythia-ml-artifacts}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/app/artifacts}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AUTO_DOWNLOAD_ARTIFACTS="${AUTO_DOWNLOAD_ARTIFACTS:-false}"
STRICT_ARTIFACT_DOWNLOAD="${STRICT_ARTIFACT_DOWNLOAD:-false}"

has_required_artifacts() {
  [ -d "$ARTIFACTS_DIR/gradient_boosting" ] || [ -d "$ARTIFACTS_DIR/lstm_5d" ] || [ -d "$ARTIFACTS_DIR/lstm_jackpot" ]
}

if [ "$AUTO_DOWNLOAD_ARTIFACTS" != "true" ]; then
  echo "AUTO_DOWNLOAD_ARTIFACTS is not enabled. Skipping S3 artifact download."
  exit 0
fi

echo "Ensuring ML artifacts are available from S3 bucket: $BUCKET_NAME"

# Create artifacts directory
mkdir -p "$ARTIFACTS_DIR"

if has_required_artifacts; then
  echo "✓ Artifacts already present at $ARTIFACTS_DIR"
  exit 0
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "AWS CLI is not installed in this image. Skipping artifact download."
  exit 0
fi

if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  echo "AWS credentials are missing. Skipping artifact download."
  exit 0
fi

# Try both layouts used historically in the bucket.
if aws s3 cp "s3://$BUCKET_NAME/artifacts/artifacts/" "$ARTIFACTS_DIR/" --region "$AWS_REGION" --recursive --quiet \
  || aws s3 cp "s3://$BUCKET_NAME/artifacts/" "$ARTIFACTS_DIR/" --region "$AWS_REGION" --recursive --quiet; then
  echo "✓ Successfully downloaded artifacts to $ARTIFACTS_DIR"
else
  if [ "$STRICT_ARTIFACT_DOWNLOAD" = "true" ]; then
    echo "✗ Error: failed to download artifacts from S3 with strict mode enabled."
    exit 1
  fi
  echo "Artifact download failed; continuing without S3 artifacts."
  exit 0
fi

if has_required_artifacts; then
  echo "✓ Artifact structure verified"
else
  if [ "$STRICT_ARTIFACT_DOWNLOAD" = "true" ]; then
    echo "✗ Error: artifacts not properly downloaded (strict mode)."
    exit 1
  fi
  echo "Artifacts still missing after download; continuing because strict mode is disabled."
fi
