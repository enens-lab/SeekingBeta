#!/bin/bash
set -e

BUCKET_NAME="${S3_BUCKET:-pythia-ml-artifacts}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/app/artifacts}"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo "Downloading ML artifacts from S3 bucket: $BUCKET_NAME"

# Create artifacts directory
mkdir -p "$ARTIFACTS_DIR"

# Download artifacts from S3
aws s3 cp "s3://$BUCKET_NAME/artifacts/" "$ARTIFACTS_DIR/" \
  --region "$AWS_REGION" \
  --recursive \
  --quiet

echo "✓ Successfully downloaded artifacts to $ARTIFACTS_DIR"

# Verify structure
if [ -d "$ARTIFACTS_DIR/gradient_boosting" ]; then
  echo "✓ Artifact structure verified"
else
  echo "✗ Error: artifacts not properly downloaded"
  exit 1
fi