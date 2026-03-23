"""Train a multi-task PGA neural model with PyTorch on Apple Silicon or CPU."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
    parser = argparse.ArgumentParser(description="Train a multi-task PGA neural model with PyTorch.")
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
    parser.add_argument("--refresh-source-data", action="store_true", help="Refresh schedules, results, and stat snapshots from the web instead of reusing the local store.")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild the PGA dataset before training.")
    parser.add_argument(
        "--snapshot-mode",
        default="prior_season",
        choices=["prior_season", "same_season", "none"],
        help="Snapshot mode to use if rebuilding the dataset.",
    )
    parser.add_argument("--sequence-length", type=int, default=12, help="Number of previous tournaments to encode per player.")
    parser.add_argument("--min-history", type=int, default=3, help="Minimum previous tournaments required to include a row in training.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--lstm-units", type=int, default=192, help="Base hidden width for the sequence tower.")
    parser.add_argument("--static-width", type=int, default=384, help="Hidden width for the static tower.")
    parser.add_argument("--dense-width", type=int, default=384, help="Hidden width for the shared dense head.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout applied throughout the network.")
    parser.add_argument("--learning-rate", type=float, default=4e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Optional label smoothing for binary targets.")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Keep 0 on macOS if you hit issues.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Device to train on. Defaults to auto and prefers MPS.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "pga_neural_multitask_torch"),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    frame = dataset.copy()
    for target in targets:
        frame[target] = _coerce_binary(frame[target])
    frame = frame.sort_values(["player_id", "event_sort_key", "tournament_id"]).reset_index(drop=True)

    sequence_rows: list[np.ndarray] = []
    history_lengths: list[int] = []
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
            sequence_rows.append(history.copy())
            history_lengths.append(int(len(history)))
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

    if not sequence_rows:
        raise ValueError("No training examples were created. Lower --min-history or expand the season range.")

    label_matrix = np.asarray(label_rows, dtype=np.float32)
    labels = {target: label_matrix[:, index].astype(np.float32) for index, target in enumerate(targets)}
    return (
        np.asarray(sequence_rows, dtype=object),
        np.asarray(history_lengths, dtype=np.int64),
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
    history_lengths: np.ndarray,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    *,
    sequence_length: int,
    sequence_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros((len(sequences), sequence_length, len(sequence_columns)), dtype=np.float32)
    clipped_lengths = np.clip(history_lengths.astype(np.int64), 0, sequence_length)
    for idx, seq in enumerate(sequences):
        seq_array = np.asarray(seq, dtype=float)
        if seq_array.size == 0:
            continue
        seq_imputed = imputer.transform(seq_array)
        transformed = scaler.transform(seq_imputed)
        take = transformed[-sequence_length:]
        result[idx, -len(take):, :] = take
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), clipped_lengths


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


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested MPS device, but PyTorch MPS is not available.")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


@dataclass
class PGABatch:
    sequence: torch.Tensor
    history_length: torch.Tensor
    static: torch.Tensor
    labels: dict[str, torch.Tensor]


class PGADataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        history_lengths: np.ndarray,
        static: np.ndarray,
        labels: dict[str, np.ndarray],
    ) -> None:
        self.sequences = torch.from_numpy(sequences.astype(np.float32))
        self.history_lengths = torch.from_numpy(history_lengths.astype(np.int64))
        self.static = torch.from_numpy(static.astype(np.float32))
        self.labels = {target: torch.from_numpy(values.astype(np.float32)) for target, values in labels.items()}

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> PGABatch:
        return PGABatch(
            sequence=self.sequences[index],
            history_length=self.history_lengths[index],
            static=self.static[index],
            labels={target: values[index] for target, values in self.labels.items()},
        )


def _collate_fn(batch: list[PGABatch]) -> PGABatch:
    return PGABatch(
        sequence=torch.stack([item.sequence for item in batch]),
        history_length=torch.stack([item.history_length for item in batch]),
        static=torch.stack([item.static for item in batch]),
        labels={
            target: torch.stack([item.labels[target] for item in batch])
            for target in batch[0].labels
        },
    )


class PGAMultiTaskTorchModel(nn.Module):
    def __init__(
        self,
        *,
        sequence_feature_count: int,
        static_feature_count: int,
        targets: list[str],
        lstm_units: int,
        static_width: int,
        dense_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.targets = targets
        self.seq_norm = nn.LayerNorm(sequence_feature_count)
        self.seq_lstm = nn.LSTM(
            input_size=sequence_feature_count,
            hidden_size=lstm_units,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        seq_out_width = lstm_units * 2
        self.seq_head = nn.Sequential(
            nn.Linear(seq_out_width, max(64, lstm_units)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.static_tower = nn.Sequential(
            nn.Linear(static_feature_count, static_width),
            nn.ReLU(),
            nn.BatchNorm1d(static_width),
            nn.Dropout(dropout),
            nn.Linear(static_width, max(64, static_width // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        merged_width = max(64, lstm_units) + max(64, static_width // 2)
        self.shared = nn.Sequential(
            nn.Linear(merged_width, dense_width),
            nn.ReLU(),
            nn.BatchNorm1d(dense_width),
            nn.Dropout(dropout),
            nn.Linear(dense_width, max(64, dense_width // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        head_width = max(64, dense_width // 2)
        self.heads = nn.ModuleDict({target: nn.Linear(head_width, 1) for target in targets})

    def forward(self, sequence: torch.Tensor, history_length: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.seq_norm(sequence)
        seq_out, _ = self.seq_lstm(x)
        valid_index = torch.clamp(history_length - 1, min=0).to(device=seq_out.device)
        batch_index = torch.arange(seq_out.shape[0], device=seq_out.device)
        seq_last = seq_out[batch_index, valid_index]
        seq_last = self.seq_head(seq_last)
        static_out = self.static_tower(static)
        merged = torch.cat([seq_last, static_out], dim=1)
        shared = self.shared(merged)
        return {target: self.heads[target](shared).squeeze(-1) for target in self.targets}


def _label_smooth(tensor: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return tensor
    return tensor * (1.0 - smoothing) + 0.5 * smoothing


def _train_epoch(
    model: PGAMultiTaskTorchModel,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    criterions: dict[str, nn.Module],
    loss_weights: dict[str, float],
    device: torch.device,
    label_smoothing: float,
    clip_grad_norm: float,
) -> float:
    model.train()
    running_loss = 0.0
    example_count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        sequence = batch.sequence.to(device)
        history_length = batch.history_length.to(device)
        static = batch.static.to(device)
        outputs = model(sequence, history_length, static)
        total_loss = torch.tensor(0.0, device=device)
        for target, criterion in criterions.items():
            truth = batch.labels[target].to(device)
            truth = _label_smooth(truth, label_smoothing)
            total_loss = total_loss + loss_weights[target] * criterion(outputs[target], truth)
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        batch_size = sequence.shape[0]
        running_loss += float(total_loss.detach().cpu()) * batch_size
        example_count += batch_size
    return running_loss / max(example_count, 1)


def _predict(
    model: PGAMultiTaskTorchModel,
    loader: DataLoader,
    *,
    device: torch.device,
    targets: list[str],
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    with torch.no_grad():
        for batch in loader:
            sequence = batch.sequence.to(device)
            history_length = batch.history_length.to(device)
            static = batch.static.to(device)
            outputs = model(sequence, history_length, static)
            for target in targets:
                probs = torch.sigmoid(outputs[target]).detach().cpu().numpy()
                collected[target].append(probs)
    return {target: np.concatenate(values, axis=0) if values else np.asarray([], dtype=np.float32) for target, values in collected.items()}


def train_multitask_torch(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    device = _resolve_device(args.device)

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
                refresh_source_data=args.refresh_source_data,
                output_root=None,
                snapshot_tag=None,
                verbose=args.verbose,
            )
        )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = pd.read_parquet(dataset_path) if dataset_path.suffix == ".parquet" else pd.read_csv(dataset_path, low_memory=False)
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
        raise ValueError("No usable sequence columns found for PGA multi-task torch model")
    if not static_columns:
        raise ValueError("No usable static feature columns found for PGA multi-task torch model")

    sequences_raw, history_lengths_raw, static_raw, y_all, meta = _build_examples(
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
    X_train_seq, X_train_lengths = _transform_and_pad_sequences(
        sequences_raw[train_mask],
        history_lengths_raw[train_mask],
        seq_imputer,
        seq_scaler,
        sequence_length=args.sequence_length,
        sequence_columns=sequence_columns,
    )
    X_val_seq, X_val_lengths = _transform_and_pad_sequences(
        sequences_raw[val_mask],
        history_lengths_raw[val_mask],
        seq_imputer,
        seq_scaler,
        sequence_length=args.sequence_length,
        sequence_columns=sequence_columns,
    )
    X_train_static, X_val_static, static_imputer, static_scaler = _prepare_static_features(
        static_raw[train_mask], static_raw[val_mask]
    )
    y_train = {target: values[train_mask].astype(np.float32) for target, values in y_all.items()}
    y_val = {target: values[val_mask].astype(np.float32) for target, values in y_all.items()}

    train_dataset = PGADataset(X_train_seq, X_train_lengths, X_train_static, y_train)
    val_dataset = PGADataset(X_val_seq, X_val_lengths, X_val_static, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=_collate_fn)

    model = PGAMultiTaskTorchModel(
        sequence_feature_count=len(sequence_columns),
        static_feature_count=len(static_columns),
        targets=targets,
        lstm_units=args.lstm_units,
        static_width=args.static_width,
        dense_width=args.dense_width,
        dropout=args.dropout,
    ).to(device)

    positive_weights = {target: _positive_weight(y_train[target]) for target in targets}
    criterions = {
        target: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weights[target]], device=device))
        for target in targets
    }
    loss_weights = {target: 2.0 if target == "won" else 1.2 if target == "top_10" else 1.0 for target in targets}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_state: dict[str, Any] | None = None
    best_val_auc = -float("inf")
    best_epoch = -1
    history_rows: list[dict[str, Any]] = []
    primary_target = "won" if "won" in targets else targets[0]
    patience = 6
    patience_left = patience

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterions=criterions,
            loss_weights=loss_weights,
            device=device,
            label_smoothing=args.label_smoothing,
            clip_grad_norm=args.clip_grad_norm,
        )
        predictions = _predict(model, val_loader, device=device, targets=targets)
        per_target_epoch_metrics: dict[str, dict[str, float | None]] = {}
        for target in targets:
            truth = y_val[target].astype(int)
            prob = predictions[target]
            auc = roc_auc_score(truth, prob) if len(np.unique(truth)) > 1 else None
            ll = log_loss(truth, prob, labels=[0, 1])
            brier = brier_score_loss(truth, prob)
            per_target_epoch_metrics[target] = {
                "roc_auc": auc,
                "log_loss": float(ll),
                "brier": float(brier),
            }
        current_primary_auc = per_target_epoch_metrics[primary_target]["roc_auc"] or -float("inf")
        current_primary_loss = per_target_epoch_metrics[primary_target]["log_loss"]
        scheduler.step(current_primary_loss)
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"{target}_auc": metrics["roc_auc"] for target, metrics in per_target_epoch_metrics.items()},
            **{f"{target}_log_loss": metrics["log_loss"] for target, metrics in per_target_epoch_metrics.items()},
        }
        history_rows.append(history_row)
        if args.verbose:
            logger.info(
                "epoch=%s train_loss=%.5f %s_auc=%.5f lr=%.6f",
                epoch,
                train_loss,
                primary_target,
                current_primary_auc if current_primary_auc > -float("inf") else float("nan"),
                optimizer.param_groups[0]["lr"],
            )
        if current_primary_auc > best_val_auc:
            best_val_auc = current_primary_auc
            best_epoch = epoch
            best_state = {
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Torch multi-task training finished without a valid checkpoint.")

    model.load_state_dict(best_state["model"])
    predictions = _predict(model, val_loader, device=device, targets=targets)
    val_meta = meta.loc[val_mask, ["tournament_id", "tournament_name", "player_id", "player_name"]].copy()
    per_target_metrics: dict[str, Any] = {}
    for target in targets:
        truth = y_val[target].astype(int)
        prob = predictions[target]
        val_meta[f"{target}_probability"] = prob
        val_meta[target] = truth
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
            top_pick = (
                val_meta.sort_values(["tournament_id", "won_normalized_probability"], ascending=[True, False])
                .groupby("tournament_id")
                .head(1)
            )
            metric_entry["top_pick_hit_rate"] = float(top_pick[target].mean()) if not top_pick.empty else 0.0
        per_target_metrics[target] = metric_entry

    artifact_dir = Path(args.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_feature_columns": sequence_columns,
            "static_feature_columns": static_columns,
            "targets": targets,
            "sequence_length": args.sequence_length,
            "lstm_units": args.lstm_units,
            "static_width": args.static_width,
            "dense_width": args.dense_width,
            "dropout": args.dropout,
        },
        artifact_dir / "model.pt",
    )
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
    pd.DataFrame(history_rows).to_csv(artifact_dir / "training_history.csv", index=False)
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
        "device": str(device),
        "batch_size": int(args.batch_size),
        "lstm_units": int(args.lstm_units),
        "static_width": int(args.static_width),
        "dense_width": int(args.dense_width),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "best_primary_val_auc": None if best_val_auc == -float("inf") else float(best_val_auc),
        "best_epoch": best_epoch,
        "per_target": per_target_metrics,
        "notes": [
            "PyTorch trainer prefers Apple MPS when available and falls back to CPU.",
            "Profile features include stable public player metadata plus equipment sponsor signals and availability proxies.",
            "This path is intended to unlock Apple Silicon training even when tensorflow-metal is unstable.",
        ],
    }
    write_json(artifact_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    metrics = train_multitask_torch(args)
    print("=" * 72)
    print("PGA MULTI-TASK TORCH TRAINING COMPLETE")
    print("=" * 72)
    print(f"Targets:                {metrics['targets']}")
    print(f"Device:                 {metrics['device']}")
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
