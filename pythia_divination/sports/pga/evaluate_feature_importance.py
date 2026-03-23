"""Evaluate feature permutation importance for the tournament-aware PGA model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.pga.storage import write_json  # noqa: E402
from sports.pga.train_multitask_torch import (  # noqa: E402
    _build_examples,
    _resolve_device,
    _set_seed,
    _time_ordered_tournament_split,
    _transform_and_pad_sequences,
)
from sports.pga.train_tournament_ranker_torch import (  # noqa: E402
    AUX_TARGETS,
    PGATournamentRanker,
    TournamentDataset,
    _build_tournament_groups,
    _collate_tournaments,
    _predict,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate permutation importance for the PGA tournament ranker.")
    parser.add_argument("--dataset", required=True, help="Path to the PGA training dataset CSV/Parquet.")
    parser.add_argument("--artifact-dir", required=True, help="Path to the trained tournament ranker artifact directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutation.")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"], help="Device to use for evaluation.")
    parser.add_argument("--batch-size", type=int, default=32, help="Tournament batch size for evaluation.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top features to report in the summary.")
    return parser.parse_args()


def _evaluate_model(
    model: PGATournamentRanker,
    groups: list[Any],
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(
        TournamentDataset(groups),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_tournaments,
    )
    predictions = _predict(model, loader, device=device, auxiliary_targets=AUX_TARGETS)
    return float(roc_auc_score(predictions["won"], predictions["winner_probability"]))


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _set_seed(args.seed)
    device = _resolve_device(args.device)

    artifact_dir = Path(args.artifact_dir)
    with (artifact_dir / "sequence_feature_columns.json").open("r", encoding="utf-8") as f:
        sequence_columns = json.load(f)
    with (artifact_dir / "static_feature_columns.json").open("r", encoding="utf-8") as f:
        static_columns = json.load(f)

    metrics_path = artifact_dir / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        train_metrics = json.load(f)
    sequence_length = train_metrics["sequence_length"]
    min_history = train_metrics.get("min_history", 3)

    dataset_path = Path(args.dataset)
    dataset = pd.read_parquet(dataset_path) if dataset_path.suffix == ".parquet" else pd.read_csv(dataset_path, low_memory=False)

    sequences_raw, history_lengths_raw, static_raw, labels_all, meta = _build_examples(
        dataset,
        sequence_columns=sequence_columns,
        static_columns=static_columns,
        targets=["won", *AUX_TARGETS],
        sequence_length=sequence_length,
        min_history=min_history,
    )

    for target, values in labels_all.items():
        meta[target] = values.astype(int)

    train_ids, val_ids = _time_ordered_tournament_split(meta)
    val_mask = meta["tournament_id"].isin(val_ids).to_numpy()

    # We only care about validation rows for feature importance
    X_val_seq_raw = sequences_raw[val_mask]
    X_val_lengths = history_lengths_raw[val_mask]
    X_val_static_raw = static_raw[val_mask]
    y_val = {target: values[val_mask].astype(np.float32) for target, values in labels_all.items()}
    val_meta = meta.loc[val_mask, ["tournament_id", "tournament_name", "player_id", "player_name"]].reset_index(drop=True)

    seq_imputer = joblib.load(artifact_dir / "sequence_imputer.joblib")
    seq_scaler = joblib.load(artifact_dir / "sequence_scaler.joblib")
    static_imputer = joblib.load(artifact_dir / "static_imputer.joblib")
    static_scaler = joblib.load(artifact_dir / "static_scaler.joblib")

    checkpoint = torch.load(artifact_dir / "model.pt", map_location=device)
    model = PGATournamentRanker(
        sequence_feature_count=len(sequence_columns),
        static_feature_count=len(static_columns),
        lstm_units=checkpoint.get("lstm_units", 384),
        static_width=checkpoint.get("static_width", 512),
        dense_width=checkpoint.get("dense_width", 512),
        dropout=checkpoint.get("dropout", 0.3),
        auxiliary_targets=AUX_TARGETS,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def _get_val_groups(seq_data: np.ndarray, static_data: np.ndarray) -> list[Any]:
        return _build_tournament_groups(
            val_meta, seq_data, X_val_lengths, static_data, y_val, auxiliary_targets=AUX_TARGETS,
        )

    logger.info("Evaluating baseline validation AUC...")
    
    # Preprocess once
    X_val_seq_padded, _ = _transform_and_pad_sequences(
        X_val_seq_raw, X_val_lengths, seq_imputer, seq_scaler,
        sequence_length=sequence_length, sequence_columns=sequence_columns,
    )
    X_val_static_imputed = static_imputer.transform(X_val_static_raw)
    X_val_static_scaled = static_scaler.transform(X_val_static_imputed)

    baseline_groups = _get_val_groups(X_val_seq_padded, X_val_static_scaled)
    baseline_auc = _evaluate_model(model, baseline_groups, args.batch_size, device)
    logger.info("Baseline Validation AUC: %.4f", baseline_auc)

    feature_importances = []

    logger.info("Evaluating sequence features...")
    for idx, col in enumerate(sequence_columns):
        shuffled_seq = X_val_seq_padded.copy()
        # Shuffle across the batch dimension (axis=0) for each player-sequence step
        # To break associations, we shuffle the batch dimension for this feature
        shuffled_seq[:, :, idx] = np.random.permutation(shuffled_seq[:, :, idx])
        groups = _get_val_groups(shuffled_seq, X_val_static_scaled)
        auc = _evaluate_model(model, groups, args.batch_size, device)
        drop = baseline_auc - auc
        feature_importances.append({"feature": col, "type": "sequence", "auc_drop": drop})
        logger.info("Sequence feature %s AUC drop: %.5f", col, drop)

    logger.info("Evaluating static features...")
    for idx, col in enumerate(static_columns):
        shuffled_static = X_val_static_scaled.copy()
        shuffled_static[:, idx] = np.random.permutation(shuffled_static[:, idx])
        groups = _get_val_groups(X_val_seq_padded, shuffled_static)
        auc = _evaluate_model(model, groups, args.batch_size, device)
        drop = baseline_auc - auc
        feature_importances.append({"feature": col, "type": "static", "auc_drop": drop})
        logger.info("Static feature %s AUC drop: %.5f", col, drop)

    results_df = pd.DataFrame(feature_importances).sort_values("auc_drop", ascending=False).reset_index(drop=True)
    out_path = artifact_dir / "permutation_importance.csv"
    results_df.to_csv(out_path, index=False)
    
    print("\n" + "=" * 72)
    print(f"TOP {args.top_n} FEATURES BY PERMUTATION IMPORTANCE (AUC DROP)")
    print("=" * 72)
    for idx, row in results_df.head(args.top_n).iterrows():
        print(f"{idx+1:2d}. {row['feature'][:40]:<40} [{row['type'][:3]}] - Drop: {row['auc_drop']:.5f}")
    print("=" * 72)
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
