#!/usr/bin/env bash
set -euo pipefail

SRC="/Users/huyngo/Downloads/Stock_Prediction_Model/TrainingData/models"
DEST1="/Users/huyngo/Downloads/pythia/pythia_divination/artifacts/lstm_5d/classifier"
DEST2="/Users/huyngo/Downloads/pythia/pythia_divination/artifacts/lstm_jackpot/classifier"

mkdir -p "$DEST1" "$DEST2"

if [ -f "$SRC/feature_columns_production.json" ]; then
  mv "$SRC/feature_columns_production.json" "$DEST1/feature_columns.json"
  echo "Moved production feature map -> $DEST1/feature_columns.json"
else
  echo "Missing: $SRC/feature_columns_production.json" >&2
fi

if [ -f "$SRC/feature_columns_jackpot.json" ]; then
  mv "$SRC/feature_columns_jackpot.json" "$DEST2/feature_columns.json"
  echo "Moved jackpot feature map -> $DEST2/feature_columns.json"
else
  echo "Missing: $SRC/feature_columns_jackpot.json" >&2
fi

ls -la "$DEST1/feature_columns.json" "$DEST2/feature_columns.json" || true