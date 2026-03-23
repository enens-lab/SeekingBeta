"""Train a first sequence-aware neural network for PGA tournament predictions."""

from __future__ import annotations

import argparse
import json
import math
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.pga.build_training_dataset import build_training_dataset  # noqa: E402
from sports.pga.feature_engineering import (  # noqa: E402
    neural_sequence_feature_columns,
    neural_static_feature_columns,
)
from sports.pga.storage import DEFAULT_PGA_DATA_ROOT, write_json  # noqa: E402

logger = logging.getLogger(__name__)

_IDENTIFIER_COLUMNS = {
    "season_year",
    "tournament_id",
    "tournament_name",
    "display_date",
    "month",
    "event_sort_key",
    "player_id",
    "player_name",
    "country",
    "feature_player_name",
    "feature_country_name",
    "course_name",
    "course_city",
    "course_state_code",
    "course_country",
    "course_key",
    "champion_name",
    "champion_player_id",
}
_LABEL_COLUMNS = {"won", "top_5", "top_10", "made_cut", "withdrawn"}
_LEAKY_COLUMNS = {
    "position",
    "position_numeric",
    "position_pct_in_field",
    "total_score",
    "total_score_sort",
    "score_to_par",
    "total_strokes",
    "thru",
    "round_score",
    "player_state",
    "current_round",
    "official_position",
    "official_position_sort",
    "projected_fedex_rank",
    "projected_fedex_rank_sort",
    "round_1_score",
    "round_2_score",
    "round_3_score",
    "round_4_score",
    "rounds_completed",
    "strokes_per_round",
    "score_to_par_vs_field",
    "strokes_per_round_vs_field",
    "event_avg_position",
    "event_avg_score_to_par",
    "event_avg_total_strokes",
    "event_avg_strokes_per_round",
    "event_best_score_to_par",
    "event_cut_rate",
    "event_top10_rate",
    "event_win_rate",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a first sequence-aware PGA neural model.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_PGA_DATA_ROOT / "normalized" / "pga_training_dataset_latest.csv"),
        help="Path to the PGA training dataset CSV.",
    )
    parser.add_argument(
        "--target",
        default="won",
        choices=["won", "top_5", "top_10", "made_cut"],
        help="Target label to train.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Single season to build if --rebuild-dataset is set.",
    )
    parser.add_argument(
        "--season-start",
        type=int,
        default=None,
        help="First season year to include if rebuilding the dataset.",
    )
    parser.add_argument(
        "--season-end",
        type=int,
        default=None,
        help="Last season year to include if rebuilding the dataset.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=None,
        help="Optional tournament limit per season when rebuilding the dataset.",
    )
    parser.add_argument(
        "--stat-limit",
        type=int,
        default=None,
        help="Optional PGA stat limit when rebuilding the dataset.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild the PGA dataset before training.",
    )
    parser.add_argument(
        "--snapshot-mode",
        default="prior_season",
        choices=["prior_season", "same_season", "none"],
        help="Snapshot mode to use if rebuilding the dataset.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8,
        help="Number of previous tournaments to encode per player.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=2,
        help="Minimum previous tournaments required to include a row in training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "pga_neural"),
        help="Directory for neural model artifacts.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _coerce_binary(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
        }
    )
    return mapped.fillna(0).astype(int)


def _time_ordered_tournament_split(dataset: pd.DataFrame) -> tuple[set[str], set[str]]:
    ordering_column = "event_sort_key" if "event_sort_key" in dataset.columns else "season_event_index"
    tournaments = (
        dataset[["tournament_id", ordering_column]]
        .drop_duplicates()
        .sort_values([ordering_column, "tournament_id"])
        .reset_index(drop=True)
    )
    unique_count = len(tournaments)
    if unique_count < 8:
        raise ValueError("Need at least 8 tournaments for a useful neural validation split.")
    holdout_count = max(3, math.ceil(unique_count * 0.25))
    train_ids = set(tournaments.iloc[:-holdout_count]["tournament_id"])
    val_ids = set(tournaments.iloc[-holdout_count:]["tournament_id"])
    return train_ids, val_ids


def _build_example_sequences(
    dataset: pd.DataFrame,
    *,
    sequence_columns: list[str],
    static_columns: list[str],
    target: str,
    sequence_length: int,
    min_history: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    frame = dataset.copy()
    frame[target] = _coerce_binary(frame[target])
    frame = frame.sort_values(["player_id", "event_sort_key", "tournament_id"]).reset_index(drop=True)

    examples: list[np.ndarray] = []
    static_rows: list[np.ndarray] = []
    labels: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    grouped = frame.groupby("player_id", sort=False)
    for _, player_frame in grouped:
        sequence_values = player_frame[sequence_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        static_values = player_frame[static_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        targets = player_frame[target].to_numpy(dtype=int)
        records = player_frame.to_dict(orient="records")
        for idx in range(len(player_frame)):
            history = sequence_values[max(0, idx - sequence_length):idx]
            if len(history) < min_history:
                continue
            examples.append(history.copy())
            static_rows.append(static_values[idx].copy())
            labels.append(int(targets[idx]))
            meta_rows.append(
                {
                    "tournament_id": records[idx]["tournament_id"],
                    "tournament_name": records[idx]["tournament_name"],
                    "player_id": records[idx]["player_id"],
                    "player_name": records[idx]["player_name"],
                    "event_sort_key": records[idx].get("event_sort_key"),
                    target: int(targets[idx]),
                    "history_length": int(len(history)),
                }
            )
    if not examples:
        raise ValueError("No training examples were created. Lower --min-history or expand seasons.")

    return (
        np.asarray(examples, dtype=object),
        np.asarray(static_rows, dtype=float),
        np.asarray(labels, dtype=int),
        pd.DataFrame(meta_rows),
    )


def _fit_sequence_preprocessors(
    train_sequences: np.ndarray,
    sequence_columns: list[str],
) -> tuple[SimpleImputer, StandardScaler]:
    flat_rows = [seq for seq in train_sequences if len(seq)]
    flat = np.concatenate(flat_rows, axis=0) if flat_rows else np.zeros((1, len(sequence_columns)))
    imputer = SimpleImputer(strategy="median")
    flat_imputed = imputer.fit_transform(flat)
    scaler = StandardScaler()
    scaler.fit(flat_imputed)
    return imputer, scaler


def _transform_and_pad_sequences(
    sequences: np.ndarray,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    *,
    sequence_length: int,
    sequence_columns: list[str],
) -> np.ndarray:
    result = np.zeros((len(sequences), sequence_length, len(sequence_columns)), dtype=np.float32)
    for idx, seq in enumerate(sequences):
        seq_array = np.asarray(seq, dtype=float)
        if seq_array.size == 0:
            continue
        seq_imputed = imputer.transform(seq_array)
        transformed = scaler.transform(seq_imputed)
        take = transformed[-sequence_length:]
        result[idx, -len(take):, :] = take
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_static_features(
    train_frame: np.ndarray,
    val_frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(train_frame)
    val_imputed = imputer.transform(val_frame)
    train_scaled = scaler.fit_transform(train_imputed)
    val_scaled = scaler.transform(val_imputed)
    return train_scaled.astype(np.float32), val_scaled.astype(np.float32), imputer, scaler


def _build_model(sequence_length: int, sequence_feature_count: int, static_feature_count: int) -> tf.keras.Model:
    seq_input = tf.keras.Input(shape=(sequence_length, sequence_feature_count), name="sequence")
    x = tf.keras.layers.Masking(mask_value=0.0)(seq_input)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    static_input = tf.keras.Input(shape=(static_feature_count,), name="static")
    s = tf.keras.layers.Dense(64, activation="relu")(static_input)
    s = tf.keras.layers.Dropout(0.2)(s)
    s = tf.keras.layers.Dense(32, activation="relu")(s)

    merged = tf.keras.layers.Concatenate()([x, s])
    merged = tf.keras.layers.Dense(64, activation="relu")(merged)
    merged = tf.keras.layers.Dropout(0.3)(merged)
    merged = tf.keras.layers.Dense(32, activation="relu")(merged)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(merged)

    model = tf.keras.Model(inputs=[seq_input, static_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def _class_weight_dict(y: np.ndarray) -> dict[int, float]:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return {0: 1.0, 1: 1.0}
    total = positives + negatives
    return {
        0: total / (2.0 * negatives),
        1: total / (2.0 * positives),
    }


def train_neural_model(args: argparse.Namespace) -> dict[str, Any]:
    if args.rebuild_dataset:
        build_training_dataset(
            argparse.Namespace(
                season=args.season,
                season_start=args.season_start,
                season_end=args.season_end,
                tour_code="R",
                history_limit=args.history_limit,
                stat_limit=args.stat_limit,
                snapshot_mode=args.snapshot_mode,
                output_root=None,
                snapshot_tag=None,
                verbose=args.verbose,
            )
        )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path, low_memory=False)
    if args.target not in dataset.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset")

    sequence_columns = neural_sequence_feature_columns(dataset)
    static_columns = [
        column
        for column in neural_static_feature_columns(dataset)
        if column not in _LABEL_COLUMNS
        and column not in _IDENTIFIER_COLUMNS
        and column not in _LEAKY_COLUMNS
    ]
    if not sequence_columns:
        raise ValueError("No usable sequence columns found for PGA neural model")
    if not static_columns:
        raise ValueError("No usable static feature columns found for PGA neural model")

    sequences_raw, static_raw, y_all, meta = _build_example_sequences(
        dataset,
        sequence_columns=sequence_columns,
        static_columns=static_columns,
        target=args.target,
        sequence_length=args.sequence_length,
        min_history=args.min_history,
    )
    meta["target"] = y_all
    train_ids, val_ids = _time_ordered_tournament_split(meta)
    train_mask = meta["tournament_id"].isin(train_ids).to_numpy()
    val_mask = meta["tournament_id"].isin(val_ids).to_numpy()
    non_empty_static_mask = ~np.all(np.isnan(static_raw[train_mask]), axis=0)
    static_raw = static_raw[:, non_empty_static_mask]
    static_columns = [
        column
        for column, keep in zip(static_columns, non_empty_static_mask)
        if keep
    ]
    if not static_columns:
        raise ValueError("No usable static feature columns remain after filtering empty train features.")

    seq_imputer, seq_scaler = _fit_sequence_preprocessors(sequences_raw[train_mask], sequence_columns)
    X_train_seq = _transform_and_pad_sequences(
        sequences_raw[train_mask],
        seq_imputer,
        seq_scaler,
        sequence_length=args.sequence_length,
        sequence_columns=sequence_columns,
    )
    X_val_seq = _transform_and_pad_sequences(
        sequences_raw[val_mask],
        seq_imputer,
        seq_scaler,
        sequence_length=args.sequence_length,
        sequence_columns=sequence_columns,
    )
    X_train_static, X_val_static, static_imputer, static_scaler = _prepare_static_features(
        static_raw[train_mask],
        static_raw[val_mask],
    )
    y_train = y_all[train_mask].astype(np.float32)
    y_val = y_all[val_mask].astype(np.float32)

    model = _build_model(args.sequence_length, len(sequence_columns), len(static_columns))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=6,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        {"sequence": X_train_seq, "static": X_train_static},
        y_train,
        validation_data=({"sequence": X_val_seq, "static": X_val_static}, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1 if args.verbose else 0,
        callbacks=callbacks,
        class_weight=_class_weight_dict(y_train.astype(int)),
    )

    val_prob = model.predict({"sequence": X_val_seq, "static": X_val_static}, verbose=0).reshape(-1)
    val_meta = meta.loc[val_mask, ["tournament_id", "tournament_name", "player_id", "player_name"]].copy()
    val_meta["raw_probability"] = val_prob
    group_sums = val_meta.groupby("tournament_id")["raw_probability"].transform("sum")
    field_sizes = val_meta.groupby("tournament_id")["player_id"].transform("count")
    val_meta["normalized_win_probability"] = (
        val_meta["raw_probability"] / group_sums.replace(0, np.nan)
    ).fillna(1.0 / field_sizes)
    val_meta["target"] = y_val.astype(int)

    auc = roc_auc_score(y_val, val_prob) if len(np.unique(y_val)) > 1 else None
    ll = log_loss(y_val, val_prob, labels=[0, 1])
    brier = brier_score_loss(y_val, val_prob)
    top_pick = (
        val_meta.sort_values(["tournament_id", "normalized_win_probability"], ascending=[True, False])
        .groupby("tournament_id")
        .head(1)
    )
    top_pick_hit_rate = float(top_pick["target"].mean()) if not top_pick.empty else 0.0

    artifact_dir = Path(args.output_dir) / args.target
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save(artifact_dir / "model.keras")
    joblib.dump(seq_imputer, artifact_dir / "sequence_imputer.joblib")
    joblib.dump(static_imputer, artifact_dir / "static_imputer.joblib")
    joblib.dump(static_scaler, artifact_dir / "static_scaler.joblib")
    joblib.dump(seq_scaler, artifact_dir / "sequence_scaler.joblib")
    with (artifact_dir / "static_feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(static_columns, handle, indent=2)
        handle.write("\n")
    with (artifact_dir / "sequence_feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(sequence_columns, handle, indent=2)
        handle.write("\n")
    val_meta.to_csv(artifact_dir / "validation_predictions.csv", index=False)

    feature_snapshot_mode = dataset["feature_snapshot_mode"].iloc[0] if "feature_snapshot_mode" in dataset.columns else "unknown"
    leakage_warning = (
        bool(dataset["feature_leakage_warning"].fillna(False).astype(bool).any())
        if "feature_leakage_warning" in dataset.columns
        else False
    )
    metrics = {
        "target": args.target,
        "dataset_path": str(dataset_path),
        "sequence_length": args.sequence_length,
        "min_history": args.min_history,
        "sequence_feature_count": len(sequence_columns),
        "static_feature_count": len(static_columns),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()),
        "train_tournaments": int(len(train_ids)),
        "validation_tournaments": int(len(val_ids)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_validation": float(y_val.mean()),
        "roc_auc": auc,
        "log_loss": float(ll),
        "brier_score": float(brier),
        "top_pick_hit_rate": top_pick_hit_rate,
        "best_val_auc": float(max(history.history.get("val_auc", [0.0]))),
        "artifact_dir": str(artifact_dir),
        "validation_prediction_path": str(artifact_dir / "validation_predictions.csv"),
        "feature_snapshot_mode": feature_snapshot_mode,
        "leakage_warning": leakage_warning,
        "notes": [
            "Sequence branch uses prior tournament outcomes only, padded per player.",
            "Static branch uses derived player-form, season-to-date, course-history, and curated public stat priors.",
            "Neural performance should still be treated as exploratory until we add auxiliary targets and better calibration.",
        ],
    }
    write_json(artifact_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    metrics = train_neural_model(args)
    print("=" * 72)
    print("PGA NEURAL TRAINING COMPLETE")
    print("=" * 72)
    print(f"Target:                 {metrics['target']}")
    print(f"Sequence length:        {metrics['sequence_length']}")
    print(f"Sequence features:      {metrics['sequence_feature_count']}")
    print(f"Static features:        {metrics['static_feature_count']}")
    print(f"Train tournaments:      {metrics['train_tournaments']}")
    print(f"Validation tournaments: {metrics['validation_tournaments']}")
    print(f"Validation ROC AUC:     {metrics['roc_auc']}")
    print(f"Validation log loss:    {metrics['log_loss']:.6f}")
    print(f"Validation Brier:       {metrics['brier_score']:.6f}")
    print(f"Top-pick hit rate:      {metrics['top_pick_hit_rate']:.2%}")
    print(f"Artifact dir:           {metrics['artifact_dir']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
