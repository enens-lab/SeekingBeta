"""Train a multi-task sequence-aware PGA neural model."""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    "profile_player_name",
    "course_name",
    "course_city",
    "course_state_code",
    "course_country",
    "course_key",
    "champion_name",
    "champion_player_id",
    "profile_source_path",
    "profile_first_name",
    "profile_last_name",
    "profile_country",
    "profile_country_code",
    "profile_born_date",
    "profile_birthplace",
    "profile_college",
    "profile_height_text",
    "profile_equipment_sponsor_primary",
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
    "event_field_size",
    "field_size",
    "implied_uniform_win_probability",
}
DEFAULT_TARGETS = ("won", "top_10", "made_cut")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-task PGA neural model.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_PGA_DATA_ROOT / "normalized" / "pga_training_dataset_latest.csv"),
        help="Path to the PGA training dataset CSV.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        choices=["won", "top_5", "top_10", "made_cut"],
        help="Binary labels to learn jointly.",
    )
    parser.add_argument("--season", type=int, default=None, help="Single season to build if --rebuild-dataset is set.")
    parser.add_argument("--season-start", type=int, default=None, help="First season year to include if rebuilding the dataset.")
    parser.add_argument("--season-end", type=int, default=None, help="Last season year to include if rebuilding the dataset.")
    parser.add_argument("--history-limit", type=int, default=None, help="Optional tournament limit per season when rebuilding the dataset.")
    parser.add_argument("--stat-limit", type=int, default=None, help="Optional PGA stat limit when rebuilding the dataset.")
    parser.add_argument("--profile-limit", type=int, default=None, help="Optional player profile limit when rebuilding the dataset.")
    parser.add_argument("--refresh-profiles", action="store_true", help="Refresh cached profile snapshots when rebuilding the dataset.")
    parser.add_argument("--profile-workers", type=int, default=8, help="Concurrent profile fetch workers when rebuilding the dataset.")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild the PGA dataset before training.")
    parser.add_argument(
        "--snapshot-mode",
        default="prior_season",
        choices=["prior_season", "same_season", "none"],
        help="Snapshot mode to use if rebuilding the dataset.",
    )
    parser.add_argument("--sequence-length", type=int, default=10, help="Number of previous tournaments to encode per player.")
    parser.add_argument("--min-history", type=int, default=3, help="Minimum previous tournaments required to include a row in training.")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lstm-units", type=int, default=128, help="Base hidden width for the sequence tower.")
    parser.add_argument("--static-width", type=int, default=256, help="Hidden width for the static tower.")
    parser.add_argument("--dense-width", type=int, default=256, help="Hidden width for the shared dense head.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout applied throughout the network.")
    parser.add_argument("--learning-rate", type=float, default=7.5e-4, help="Adam learning rate.")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision when the local TensorFlow build supports it.")
    parser.add_argument("--cache-dataset", action="store_true", help="Cache tf.data pipelines in memory for repeated epochs.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "pga_neural_multitask"),
        help="Directory for neural model artifacts.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _coerce_binary(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
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
    if unique_count < 10:
        raise ValueError("Need at least 10 tournaments for a useful multi-task validation split.")
    holdout_count = max(4, math.ceil(unique_count * 0.25))
    train_ids = set(tournaments.iloc[:-holdout_count]["tournament_id"])
    val_ids = set(tournaments.iloc[-holdout_count:]["tournament_id"])
    return train_ids, val_ids


def _build_examples(
    dataset: pd.DataFrame,
    *,
    sequence_columns: list[str],
    static_columns: list[str],
    targets: list[str],
    sequence_length: int,
    min_history: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    frame = dataset.copy()
    for target in targets:
        frame[target] = _coerce_binary(frame[target])
    frame = frame.sort_values(["player_id", "event_sort_key", "tournament_id"]).reset_index(drop=True)

    examples: list[np.ndarray] = []
    static_rows: list[np.ndarray] = []
    label_rows: list[list[int]] = []
    meta_rows: list[dict[str, Any]] = []

    grouped = frame.groupby("player_id", sort=False)
    for _, player_frame in grouped:
        sequence_values = player_frame[sequence_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        static_values = player_frame[static_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        target_values = player_frame[targets].to_numpy(dtype=int)
        records = player_frame.to_dict(orient="records")
        for idx in range(len(player_frame)):
            history = sequence_values[max(0, idx - sequence_length):idx]
            if len(history) < min_history:
                continue
            examples.append(history.copy())
            static_rows.append(static_values[idx].copy())
            label_rows.append(target_values[idx].astype(int).tolist())
            meta = {
                "tournament_id": records[idx]["tournament_id"],
                "tournament_name": records[idx]["tournament_name"],
                "player_id": records[idx]["player_id"],
                "player_name": records[idx]["player_name"],
                "event_sort_key": records[idx].get("event_sort_key"),
                "history_length": int(len(history)),
            }
            for target_index, target in enumerate(targets):
                meta[target] = int(target_values[idx][target_index])
            meta_rows.append(meta)

    if not examples:
        raise ValueError("No training examples were created. Lower --min-history or expand the season range.")

    label_matrix = np.asarray(label_rows, dtype=int)
    labels = {target: label_matrix[:, index].astype(np.float32) for index, target in enumerate(targets)}
    return (
        np.asarray(examples, dtype=object),
        np.asarray(static_rows, dtype=float),
        labels,
        pd.DataFrame(meta_rows),
    )


def _fit_sequence_preprocessors(train_sequences: np.ndarray, sequence_columns: list[str]) -> tuple[SimpleImputer, StandardScaler]:
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


def _prepare_static_features(train_frame: np.ndarray, val_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(train_frame)
    val_imputed = imputer.transform(val_frame)
    train_scaled = scaler.fit_transform(train_imputed)
    val_scaled = scaler.transform(val_imputed)
    return train_scaled.astype(np.float32), val_scaled.astype(np.float32), imputer, scaler


def _positive_weight(y: np.ndarray) -> float:
    positives = float(np.sum(y))
    negatives = float(len(y) - positives)
    if positives <= 0 or negatives <= 0:
        return 1.0
    return float(min(max(negatives / positives, 1.0), 25.0))


def _weighted_binary_crossentropy(pos_weight: float):
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        return tf.reduce_mean(
            -(pos_weight * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        )

    return loss


def _build_model(
    *,
    sequence_length: int,
    sequence_feature_count: int,
    static_feature_count: int,
    targets: list[str],
    lstm_units: int,
    static_width: int,
    dense_width: int,
    dropout: float,
    learning_rate: float,
    positive_weights: dict[str, float],
) -> tf.keras.Model:
    seq_input = tf.keras.Input(shape=(sequence_length, sequence_feature_count), name="sequence")
    x = tf.keras.layers.Masking(mask_value=0.0)(seq_input)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max(16, lstm_units // 2)))(x)
    x = tf.keras.layers.Dense(max(32, lstm_units), activation="relu")(x)

    static_input = tf.keras.Input(shape=(static_feature_count,), name="static")
    s = tf.keras.layers.Dense(static_width, activation="relu")(static_input)
    s = tf.keras.layers.BatchNormalization()(s)
    s = tf.keras.layers.Dropout(dropout)(s)
    s = tf.keras.layers.Dense(max(32, static_width // 2), activation="relu")(s)

    merged = tf.keras.layers.Concatenate()([x, s])
    merged = tf.keras.layers.Dense(dense_width, activation="relu")(merged)
    merged = tf.keras.layers.BatchNormalization()(merged)
    merged = tf.keras.layers.Dropout(dropout)(merged)
    merged = tf.keras.layers.Dense(max(32, dense_width // 2), activation="relu")(merged)

    outputs = {
        target: tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name=target)(merged)
        for target in targets
    }
    model = tf.keras.Model(inputs=[seq_input, static_input], outputs=outputs)
    losses = {target: _weighted_binary_crossentropy(positive_weights[target]) for target in targets}
    loss_weights = {target: 2.0 if target == "won" else 1.2 if target == "top_10" else 1.0 for target in targets}
    metrics = {target: [tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="accuracy")] for target in targets}
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return model


def _build_dataset(
    sequences: np.ndarray,
    static: np.ndarray,
    labels: dict[str, np.ndarray],
    *,
    batch_size: int,
    training: bool,
    cache: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(({"sequence": sequences, "static": static}, labels))
    if cache:
        dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(min(len(sequences), 4096), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_multitask_model(args: argparse.Namespace) -> dict[str, Any]:
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

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
                profile_limit=args.profile_limit,
                refresh_profiles=args.refresh_profiles,
                profile_workers=args.profile_workers,
                output_root=None,
                snapshot_tag=None,
                verbose=args.verbose,
            )
        )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path, low_memory=False)
    targets = [target for target in args.targets if target in dataset.columns]
    if not targets:
        raise ValueError("None of the requested targets exist in the dataset")

    sequence_columns = neural_sequence_feature_columns(dataset)
    static_columns = [
        column
        for column in neural_static_feature_columns(dataset)
        if column not in _LABEL_COLUMNS and column not in _IDENTIFIER_COLUMNS and column not in _LEAKY_COLUMNS
    ]
    if not sequence_columns:
        raise ValueError("No usable sequence columns found for PGA multi-task neural model")
    if not static_columns:
        raise ValueError("No usable static feature columns found for PGA multi-task neural model")

    sequences_raw, static_raw, y_all, meta = _build_examples(
        dataset,
        sequence_columns=sequence_columns,
        static_columns=static_columns,
        targets=targets,
        sequence_length=args.sequence_length,
        min_history=args.min_history,
    )
    for target, values in y_all.items():
        meta[target] = values.astype(int)

    train_ids, val_ids = _time_ordered_tournament_split(meta)
    train_mask = meta["tournament_id"].isin(train_ids).to_numpy()
    val_mask = meta["tournament_id"].isin(val_ids).to_numpy()

    non_empty_static_mask = ~np.all(np.isnan(static_raw[train_mask]), axis=0)
    static_raw = static_raw[:, non_empty_static_mask]
    static_columns = [column for column, keep in zip(static_columns, non_empty_static_mask) if keep]
    if not static_columns:
        raise ValueError("No usable static feature columns remain after filtering empty train features.")

    seq_imputer, seq_scaler = _fit_sequence_preprocessors(sequences_raw[train_mask], sequence_columns)
    X_train_seq = _transform_and_pad_sequences(
        sequences_raw[train_mask], seq_imputer, seq_scaler, sequence_length=args.sequence_length, sequence_columns=sequence_columns
    )
    X_val_seq = _transform_and_pad_sequences(
        sequences_raw[val_mask], seq_imputer, seq_scaler, sequence_length=args.sequence_length, sequence_columns=sequence_columns
    )
    X_train_static, X_val_static, static_imputer, static_scaler = _prepare_static_features(
        static_raw[train_mask], static_raw[val_mask]
    )
    y_train = {target: values[train_mask].astype(np.float32) for target, values in y_all.items()}
    y_val = {target: values[val_mask].astype(np.float32) for target, values in y_all.items()}

    positive_weights = {target: _positive_weight(y_train[target]) for target in targets}
    model = _build_model(
        sequence_length=args.sequence_length,
        sequence_feature_count=len(sequence_columns),
        static_feature_count=len(static_columns),
        targets=targets,
        lstm_units=args.lstm_units,
        static_width=args.static_width,
        dense_width=args.dense_width,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        positive_weights=positive_weights,
    )

    primary_target = "won" if "won" in targets else targets[0]
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=f"val_{primary_target}_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
        )
    ]

    train_ds = _build_dataset(X_train_seq, X_train_static, y_train, batch_size=args.batch_size, training=True, cache=args.cache_dataset)
    val_ds = _build_dataset(X_val_seq, X_val_static, y_val, batch_size=args.batch_size, training=False, cache=args.cache_dataset)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=1 if args.verbose else 0, callbacks=callbacks)

    predictions = model.predict({"sequence": X_val_seq, "static": X_val_static}, verbose=0)
    if isinstance(predictions, list):
        predictions = {target: prediction.reshape(-1) for target, prediction in zip(model.output_names, predictions)}
    else:
        predictions = {target: np.asarray(predictions[target]).reshape(-1) for target in model.output_names}

    val_meta = meta.loc[val_mask, ["tournament_id", "tournament_name", "player_id", "player_name"]].copy()
    per_target_metrics: dict[str, Any] = {}
    for target in targets:
        truth = y_val[target].astype(int)
        prob = predictions[target]
        val_meta[f"{target}_probability"] = prob
        auc = roc_auc_score(truth, prob) if len(np.unique(truth)) > 1 else None
        ll = log_loss(truth, prob, labels=[0, 1])
        brier = brier_score_loss(truth, prob)
        metric_entry = {
            "positive_rate_train": float(np.mean(y_train[target])),
            "positive_rate_validation": float(np.mean(truth)),
            "roc_auc": auc,
            "log_loss": float(ll),
            "brier_score": float(brier),
            "positive_weight": positive_weights[target],
        }
        if target == "won":
            group_sums = val_meta.groupby("tournament_id")[f"{target}_probability"].transform("sum")
            field_sizes = val_meta.groupby("tournament_id")["player_id"].transform("count")
            val_meta["won_normalized_probability"] = (
                val_meta[f"{target}_probability"] / group_sums.replace(0, np.nan)
            ).fillna(1.0 / field_sizes)
            winners = meta.loc[val_mask, ["tournament_id", target]].copy()
            top_pick = (
                val_meta.sort_values(["tournament_id", "won_normalized_probability"], ascending=[True, False])
                .groupby("tournament_id")
                .head(1)
            )
            top_pick = top_pick.merge(winners, on="tournament_id", how="left")
            metric_entry["top_pick_hit_rate"] = float(top_pick[target].mean()) if not top_pick.empty else 0.0
        per_target_metrics[target] = metric_entry

    artifact_dir = Path(args.output_dir)
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
    with (artifact_dir / "targets.json").open("w", encoding="utf-8") as handle:
        json.dump(targets, handle, indent=2)
        handle.write("\n")
    val_meta.to_csv(artifact_dir / "validation_predictions.csv", index=False)

    feature_snapshot_mode = dataset["feature_snapshot_mode"].iloc[0] if "feature_snapshot_mode" in dataset.columns else "unknown"
    leakage_warning = bool(dataset["feature_leakage_warning"].fillna(False).astype(bool).any()) if "feature_leakage_warning" in dataset.columns else False
    metrics = {
        "targets": targets,
        "dataset_path": str(dataset_path),
        "sequence_length": args.sequence_length,
        "min_history": args.min_history,
        "sequence_feature_count": len(sequence_columns),
        "static_feature_count": len(static_columns),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()),
        "train_tournaments": int(len(train_ids)),
        "validation_tournaments": int(len(val_ids)),
        "artifact_dir": str(artifact_dir),
        "validation_prediction_path": str(artifact_dir / "validation_predictions.csv"),
        "feature_snapshot_mode": feature_snapshot_mode,
        "leakage_warning": leakage_warning,
        "mixed_precision": bool(args.mixed_precision),
        "batch_size": int(args.batch_size),
        "lstm_units": int(args.lstm_units),
        "static_width": int(args.static_width),
        "dense_width": int(args.dense_width),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "best_primary_val_auc": float(max(history.history.get(f"val_{primary_target}_auc", [0.0]))),
        "per_target": per_target_metrics,
        "notes": [
            "Multi-task network jointly predicts won, top_10, and made_cut using shared sequence and static towers.",
            "Profile features include stable public player metadata plus equipment sponsor signals and availability proxies.",
            "Use wider season windows and higher batch sizes on Apple Silicon to take advantage of local hardware once dataset scale is sufficient.",
        ],
    }
    write_json(artifact_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    metrics = train_multitask_model(args)
    print("=" * 72)
    print("PGA MULTI-TASK NEURAL TRAINING COMPLETE")
    print("=" * 72)
    print(f"Targets:                {metrics['targets']}")
    print(f"Sequence length:        {metrics['sequence_length']}")
    print(f"Sequence features:      {metrics['sequence_feature_count']}")
    print(f"Static features:        {metrics['static_feature_count']}")
    print(f"Train tournaments:      {metrics['train_tournaments']}")
    print(f"Validation tournaments: {metrics['validation_tournaments']}")
    for target, target_metrics in metrics["per_target"].items():
        print(f"{target} ROC AUC:         {target_metrics['roc_auc']}")
        print(f"{target} log loss:        {target_metrics['log_loss']:.6f}")
        print(f"{target} brier:           {target_metrics['brier_score']:.6f}")
        if "top_pick_hit_rate" in target_metrics:
            print(f"{target} top-pick hit:    {target_metrics['top_pick_hit_rate']:.2%}")
    print(f"Artifact dir:           {metrics['artifact_dir']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
