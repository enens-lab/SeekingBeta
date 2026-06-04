#!/bin/bash
set -euo pipefail

BUCKET_NAME="${S3_BUCKET:-pythia-ml-artifacts}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/app/artifacts}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AUTO_DOWNLOAD_ARTIFACTS="${AUTO_DOWNLOAD_ARTIFACTS:-false}"
STRICT_ARTIFACT_DOWNLOAD="${STRICT_ARTIFACT_DOWNLOAD:-false}"

has_sports_artifacts() {
  # True if ANY sports artifact dir exists (the *_torch / *_baseline /
  # *_ranker_torch families). Uses a glob loop so a non-matching pattern doesn't
  # fail the check the way `ls -d a b c` would when only some patterns match.
  for d in "$ARTIFACTS_DIR"/*_torch "$ARTIFACTS_DIR"/*_baseline "$ARTIFACTS_DIR"/*_ranker_torch; do
    [ -d "$d" ] && return 0
  done
  return 1
}

has_required_artifacts() {
  # Require BOTH a stock model AND at least one sports artifact. Otherwise a box
  # that only has the stock models (the bucket's original contents) would short-
  # circuit the download and the sports boards would render empty.
  { [ -d "$ARTIFACTS_DIR/gradient_boosting" ] || [ -d "$ARTIFACTS_DIR/lstm_5d" ] || [ -d "$ARTIFACTS_DIR/lstm_jackpot" ]; } \
    && has_sports_artifacts
}

if [ "$AUTO_DOWNLOAD_ARTIFACTS" != "true" ]; then
  echo "AUTO_DOWNLOAD_ARTIFACTS is not enabled. Skipping S3 artifact download."
  exit 0
fi

echo "Ensuring ML artifacts are available from S3 bucket: $BUCKET_NAME"

# Create artifacts directory
mkdir -p "$ARTIFACTS_DIR"

# Fast path: if we have a complete-looking set AND no way/need to talk to S3,
# skip. We still prefer to run `sync` below (it's cheap and self-heals a
# partially-missing set), so this only short-circuits when tooling/creds are
# absent — never silently leaving a partial set when we COULD repair it.
if ! command -v aws >/dev/null 2>&1; then
  if has_required_artifacts; then
    echo "AWS CLI not installed but artifacts look present; continuing."
    exit 0
  fi
  echo "AWS CLI is not installed and artifacts are incomplete. Skipping download."
  [ "$STRICT_ARTIFACT_DOWNLOAD" = "true" ] && exit 1 || exit 0
fi

if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  if has_required_artifacts; then
    echo "AWS credentials missing but artifacts look present; continuing."
    exit 0
  fi
  echo "AWS credentials are missing and artifacts are incomplete. Skipping download."
  [ "$STRICT_ARTIFACT_DOWNLOAD" = "true" ] && exit 1 || exit 0
fi

# Always sync from S3: this pulls only missing/changed objects, so it self-heals
# a partial set (e.g. one model dir deleted) without re-downloading everything.
# Try the nested layout first (artifacts/artifacts/), then the flat one.
if aws s3 sync "s3://$BUCKET_NAME/artifacts/artifacts/" "$ARTIFACTS_DIR/" --region "$AWS_REGION" --only-show-errors \
  && [ -n "$(ls -A "$ARTIFACTS_DIR" 2>/dev/null)" ]; then
  echo "✓ Synced artifacts from s3://$BUCKET_NAME/artifacts/artifacts/"
elif aws s3 sync "s3://$BUCKET_NAME/artifacts/" "$ARTIFACTS_DIR/" --region "$AWS_REGION" --only-show-errors; then
  echo "✓ Synced artifacts from s3://$BUCKET_NAME/artifacts/"
else
  if [ "$STRICT_ARTIFACT_DOWNLOAD" = "true" ]; then
    echo "✗ Error: failed to sync artifacts from S3 with strict mode enabled."
    exit 1
  fi
  echo "Artifact sync failed; continuing without S3 artifacts."
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
