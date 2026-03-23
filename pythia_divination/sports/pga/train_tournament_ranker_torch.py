"""Train a tournament-aware PGA winner ranking model with PyTorch."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.pga.build_training_dataset import build_training_dataset  # noqa: E402
from sports.pga.feature_engineering import neural_sequence_feature_columns, neural_static_feature_columns  # noqa: E402
from sports.pga.storage import DEFAULT_PGA_DATA_ROOT, write_json  # noqa: E402
from sports.pga.train_multitask_torch import (  # noqa: E402
    _IDENTIFIER_COLUMNS,
    _LABEL_COLUMNS,
    _LEAKY_COLUMNS,
    _build_examples,
    _fit_sequence_preprocessors,
    _positive_weight,
    _prepare_static_features,
    _resolve_device,
    _set_seed,
    _time_ordered_tournament_split,
    _transform_and_pad_sequences,
)

logger = logging.getLogger(__name__)
AUX_TARGETS = ("top_10", "made_cut")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tournament-aware PGA winner ranker with PyTorch.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_PGA_DATA_ROOT / "normalized" / "pga_training_dataset_latest.csv"),
        help="Path to the PGA training dataset CSV.",
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
    parser.add_argument("--tournament-batch-size", type=int, default=12, help="Number of tournaments per optimization batch.")
    parser.add_argument("--lstm-units", type=int, default=192, help="Base hidden width for the sequence tower.")
    parser.add_argument("--static-width", type=int, default=384, help="Hidden width for the static tower.")
    parser.add_argument("--dense-width", type=int, default=384, help="Hidden width for the shared dense head.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout applied throughout the network.")
    parser.add_argument("--learning-rate", type=float, default=4e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Optional label smoothing for auxiliary BCE targets.")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"], help="Device to train on.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "pga_tournament_ranker_torch"),
        help="Directory for tournament-ranker artifacts.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


@dataclass
class TournamentGroup:
    tournament_id: str
    tournament_name: str
    player_ids: np.ndarray
    player_names: np.ndarray
    sequences: np.ndarray
    history_lengths: np.ndarray
    static: np.ndarray
    winner: np.ndarray
    auxiliary: dict[str, np.ndarray]


@dataclass
class TournamentBatch:
    sequence: torch.Tensor
    history_length: torch.Tensor
    static: torch.Tensor
    valid_mask: torch.Tensor
    winner: torch.Tensor
    auxiliary: dict[str, torch.Tensor]
    tournament_id: list[str]
    tournament_name: list[str]
    player_id: list[list[str]]
    player_name: list[list[str]]


class TournamentDataset(Dataset):
    def __init__(self, groups: list[TournamentGroup]) -> None:
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> TournamentGroup:
        return self.groups[index]


class PGATournamentRanker(nn.Module):
    def __init__(
        self,
        *,
        sequence_feature_count: int,
        static_feature_count: int,
        lstm_units: int,
        static_width: int,
        dense_width: int,
        dropout: float,
        auxiliary_targets: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.auxiliary_targets = auxiliary_targets
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
        self.winner_head = nn.Linear(head_width, 1)
        self.aux_heads = nn.ModuleDict({target: nn.Linear(head_width, 1) for target in auxiliary_targets})

    def forward(self, sequence: torch.Tensor, history_length: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, player_count, sequence_length, sequence_features = sequence.shape
        flat_sequence = sequence.reshape(batch_size * player_count, sequence_length, sequence_features)
        flat_history = history_length.reshape(batch_size * player_count)
        flat_static = static.reshape(batch_size * player_count, static.shape[-1])

        x = self.seq_norm(flat_sequence)
        seq_out, _ = self.seq_lstm(x)
        valid_index = torch.clamp(flat_history - 1, min=0).to(device=seq_out.device)
        batch_index = torch.arange(seq_out.shape[0], device=seq_out.device)
        seq_last = seq_out[batch_index, valid_index]
        seq_last = self.seq_head(seq_last)
        static_out = self.static_tower(flat_static)
        merged = torch.cat([seq_last, static_out], dim=1)
        shared = self.shared(merged)

        winner_logits = self.winner_head(shared).reshape(batch_size, player_count)
        auxiliary_logits = {
            target: head(shared).reshape(batch_size, player_count)
            for target, head in self.aux_heads.items()
        }
        return {"winner": winner_logits, **auxiliary_logits}


def _label_smooth(tensor: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return tensor
    return tensor * (1.0 - smoothing) + 0.5 * smoothing


def _build_tournament_groups(
    meta: pd.DataFrame,
    sequences: np.ndarray,
    history_lengths: np.ndarray,
    static: np.ndarray,
    labels: dict[str, np.ndarray],
    *,
    auxiliary_targets: tuple[str, ...],
) -> list[TournamentGroup]:
    grouped: list[TournamentGroup] = []
    ordered = meta.reset_index(drop=True).copy()
    ordered["_row_index"] = np.arange(len(ordered))
    for tournament_id, group in ordered.groupby("tournament_id", sort=False):
        row_indices = group["_row_index"].to_numpy(dtype=int)
        winner = labels["won"][row_indices].astype(np.float32)
        if int(np.round(winner.sum())) != 1:
            continue
        grouped.append(
            TournamentGroup(
                tournament_id=str(tournament_id),
                tournament_name=str(group["tournament_name"].iloc[0]),
                player_ids=group["player_id"].astype(str).to_numpy(),
                player_names=group["player_name"].fillna("").astype(str).to_numpy(),
                sequences=sequences[row_indices],
                history_lengths=history_lengths[row_indices],
                static=static[row_indices],
                winner=winner,
                auxiliary={target: labels[target][row_indices].astype(np.float32) for target in auxiliary_targets},
            )
        )
    return grouped


def _collate_tournaments(batch: list[TournamentGroup]) -> TournamentBatch:
    max_players = max(group.sequences.shape[0] for group in batch)
    sequence_length = batch[0].sequences.shape[1]
    sequence_features = batch[0].sequences.shape[2]
    static_features = batch[0].static.shape[1]

    sequence = torch.zeros((len(batch), max_players, sequence_length, sequence_features), dtype=torch.float32)
    history_length = torch.zeros((len(batch), max_players), dtype=torch.int64)
    static = torch.zeros((len(batch), max_players, static_features), dtype=torch.float32)
    valid_mask = torch.zeros((len(batch), max_players), dtype=torch.bool)
    winner = torch.zeros((len(batch), max_players), dtype=torch.float32)
    auxiliary = {
        target: torch.zeros((len(batch), max_players), dtype=torch.float32)
        for target in batch[0].auxiliary
    }
    tournament_ids: list[str] = []
    tournament_names: list[str] = []
    player_ids: list[list[str]] = []
    player_names: list[list[str]] = []

    for batch_index, group in enumerate(batch):
        player_count = group.sequences.shape[0]
        sequence[batch_index, :player_count] = torch.from_numpy(group.sequences.astype(np.float32))
        history_length[batch_index, :player_count] = torch.from_numpy(group.history_lengths.astype(np.int64))
        static[batch_index, :player_count] = torch.from_numpy(group.static.astype(np.float32))
        valid_mask[batch_index, :player_count] = True
        winner[batch_index, :player_count] = torch.from_numpy(group.winner.astype(np.float32))
        for target, values in group.auxiliary.items():
            auxiliary[target][batch_index, :player_count] = torch.from_numpy(values.astype(np.float32))
        tournament_ids.append(group.tournament_id)
        tournament_names.append(group.tournament_name)
        player_ids.append(group.player_ids.tolist())
        player_names.append(group.player_names.tolist())

    return TournamentBatch(
        sequence=sequence,
        history_length=history_length,
        static=static,
        valid_mask=valid_mask,
        winner=winner,
        auxiliary=auxiliary,
        tournament_id=tournament_ids,
        tournament_name=tournament_names,
        player_id=player_ids,
        player_name=player_names,
    )


def _masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~valid_mask, -1e9)
    target_index = torch.argmax(targets, dim=1)
    return nn.functional.cross_entropy(masked_logits, target_index)


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    losses = criterion(logits, targets)
    losses = losses * valid_mask.float()
    return losses.sum() / valid_mask.float().sum().clamp_min(1.0)


def _train_epoch(
    model: PGATournamentRanker,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    auxiliary_criterions: dict[str, nn.Module],
    auxiliary_weights: dict[str, float],
    device: torch.device,
    label_smoothing: float,
    clip_grad_norm: float,
) -> float:
    model.train()
    running_loss = 0.0
    group_count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        sequence = batch.sequence.to(device)
        history_length = batch.history_length.to(device)
        static = batch.static.to(device)
        valid_mask = batch.valid_mask.to(device)
        winner = batch.winner.to(device)
        outputs = model(sequence, history_length, static)
        loss = _masked_cross_entropy(outputs["winner"], winner, valid_mask)
        for target, criterion in auxiliary_criterions.items():
            truth = _label_smooth(batch.auxiliary[target].to(device), label_smoothing)
            loss = loss + auxiliary_weights[target] * _masked_bce(outputs[target], truth, valid_mask, criterion)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        running_loss += float(loss.detach().cpu())
        group_count += len(batch.tournament_id)
    return running_loss / max(group_count, 1)


def _predict(
    model: PGATournamentRanker,
    loader: DataLoader,
    *,
    device: torch.device,
    auxiliary_targets: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            sequence = batch.sequence.to(device)
            history_length = batch.history_length.to(device)
            static = batch.static.to(device)
            valid_mask = batch.valid_mask.to(device)
            outputs = model(sequence, history_length, static)
            winner_logits = outputs["winner"].masked_fill(~valid_mask, -1e9)
            winner_probs = torch.softmax(winner_logits, dim=1).detach().cpu().numpy()
            aux_probs = {
                target: torch.sigmoid(outputs[target]).detach().cpu().numpy()
                for target in auxiliary_targets
            }
            valid_mask_np = batch.valid_mask.numpy()
            winner_truth = batch.winner.numpy()
            for batch_index, tournament_id in enumerate(batch.tournament_id):
                player_mask = valid_mask_np[batch_index]
                for player_index, is_valid in enumerate(player_mask):
                    if not is_valid:
                        continue
                    row = {
                        "tournament_id": tournament_id,
                        "tournament_name": batch.tournament_name[batch_index],
                        "player_id": batch.player_id[batch_index][player_index],
                        "player_name": batch.player_name[batch_index][player_index],
                        "winner_probability": float(winner_probs[batch_index, player_index]),
                        "won": int(winner_truth[batch_index, player_index]),
                    }
                    for target in auxiliary_targets:
                        row[f"{target}_probability"] = float(aux_probs[target][batch_index, player_index])
                        row[target] = int(batch.auxiliary[target][batch_index, player_index].item())
                    rows.append(row)
    return pd.DataFrame(rows)


def train_tournament_ranker(args: argparse.Namespace) -> dict[str, Any]:
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

    sequence_columns = neural_sequence_feature_columns(dataset)
    static_columns = [
        column
        for column in neural_static_feature_columns(dataset)
        if column not in _LABEL_COLUMNS and column not in _IDENTIFIER_COLUMNS and column not in _LEAKY_COLUMNS
    ]
    if not sequence_columns:
        raise ValueError("No usable sequence columns found for PGA tournament ranker")
    if not static_columns:
        raise ValueError("No usable static feature columns found for PGA tournament ranker")

    sequences_raw, history_lengths_raw, static_raw, labels_all, meta = _build_examples(
        dataset,
        sequence_columns=sequence_columns,
        static_columns=static_columns,
        targets=["won", *AUX_TARGETS],
        sequence_length=args.sequence_length,
        min_history=args.min_history,
    )
    for target, values in labels_all.items():
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
    y_train = {target: values[train_mask].astype(np.float32) for target, values in labels_all.items()}
    y_val = {target: values[val_mask].astype(np.float32) for target, values in labels_all.items()}

    train_groups = _build_tournament_groups(
        meta.loc[train_mask, ["tournament_id", "tournament_name", "player_id", "player_name"]].reset_index(drop=True),
        X_train_seq,
        X_train_lengths,
        X_train_static,
        y_train,
        auxiliary_targets=AUX_TARGETS,
    )
    val_groups = _build_tournament_groups(
        meta.loc[val_mask, ["tournament_id", "tournament_name", "player_id", "player_name"]].reset_index(drop=True),
        X_val_seq,
        X_val_lengths,
        X_val_static,
        y_val,
        auxiliary_targets=AUX_TARGETS,
    )
    if not train_groups or not val_groups:
        raise ValueError("Tournament grouping produced no usable train/validation groups.")

    train_loader = DataLoader(TournamentDataset(train_groups), batch_size=args.tournament_batch_size, shuffle=True, collate_fn=_collate_tournaments)
    val_loader = DataLoader(TournamentDataset(val_groups), batch_size=args.tournament_batch_size, shuffle=False, collate_fn=_collate_tournaments)

    model = PGATournamentRanker(
        sequence_feature_count=len(sequence_columns),
        static_feature_count=len(static_columns),
        lstm_units=args.lstm_units,
        static_width=args.static_width,
        dense_width=args.dense_width,
        dropout=args.dropout,
        auxiliary_targets=AUX_TARGETS,
    ).to(device)

    aux_positive_weights = {target: _positive_weight(y_train[target]) for target in AUX_TARGETS}
    auxiliary_criterions = {
        target: nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([aux_positive_weights[target]], device=device),
            reduction="none",
        )
        for target in AUX_TARGETS
    }
    auxiliary_weights = {"top_10": 0.8, "made_cut": 0.4}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_state: dict[str, Any] | None = None
    best_top_pick = -float("inf")
    best_epoch = -1
    history_rows: list[dict[str, Any]] = []
    patience = 6
    patience_left = patience

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            auxiliary_criterions=auxiliary_criterions,
            auxiliary_weights=auxiliary_weights,
            device=device,
            label_smoothing=args.label_smoothing,
            clip_grad_norm=args.clip_grad_norm,
        )
        predictions = _predict(model, val_loader, device=device, auxiliary_targets=AUX_TARGETS)
        winner_event_log_loss = -np.log(
            predictions.loc[predictions["won"] == 1, "winner_probability"].clip(lower=1e-9)
        ).mean()
        top_pick_hit_rate = float(
            predictions.sort_values(["tournament_id", "winner_probability"], ascending=[True, False])
            .groupby("tournament_id")
            .head(1)["won"]
            .mean()
        )
        winner_auc = roc_auc_score(predictions["won"], predictions["winner_probability"])
        scheduler.step(winner_event_log_loss)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "winner_auc": float(winner_auc),
                "winner_event_log_loss": float(winner_event_log_loss),
                "winner_top_pick_hit_rate": top_pick_hit_rate,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if args.verbose:
            logger.info(
                "epoch=%s train_loss=%.5f top_pick=%.4f winner_auc=%.4f lr=%.6f",
                epoch,
                train_loss,
                top_pick_hit_rate,
                winner_auc,
                optimizer.param_groups[0]["lr"],
            )
        if top_pick_hit_rate > best_top_pick:
            best_top_pick = top_pick_hit_rate
            best_epoch = epoch
            best_state = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Tournament ranker training finished without a valid checkpoint.")

    model.load_state_dict(best_state["model"])
    predictions = _predict(model, val_loader, device=device, auxiliary_targets=AUX_TARGETS)
    winner_auc = roc_auc_score(predictions["won"], predictions["winner_probability"])
    winner_log_loss = float(log_loss(predictions["won"], predictions["winner_probability"], labels=[0, 1]))
    winner_brier = float(brier_score_loss(predictions["won"], predictions["winner_probability"]))
    winner_event_log_loss = float(-np.log(predictions.loc[predictions["won"] == 1, "winner_probability"].clip(lower=1e-9)).mean())
    top_pick = (
        predictions.sort_values(["tournament_id", "winner_probability"], ascending=[True, False])
        .groupby("tournament_id")
        .head(1)
    )
    top_3 = (
        predictions.sort_values(["tournament_id", "winner_probability"], ascending=[True, False])
        .groupby("tournament_id")
        .head(3)
        .groupby("tournament_id")["won"]
        .max()
        .mean()
    )

    artifact_dir = Path(args.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_feature_columns": sequence_columns,
            "static_feature_columns": static_columns,
            "sequence_length": args.sequence_length,
            "lstm_units": args.lstm_units,
            "static_width": args.static_width,
            "dense_width": args.dense_width,
            "dropout": args.dropout,
            "auxiliary_targets": AUX_TARGETS,
        },
        artifact_dir / "model.pt",
    )
    joblib.dump(seq_imputer, artifact_dir / "sequence_imputer.joblib")
    joblib.dump(seq_scaler, artifact_dir / "sequence_scaler.joblib")
    joblib.dump(static_imputer, artifact_dir / "static_imputer.joblib")
    joblib.dump(static_scaler, artifact_dir / "static_scaler.joblib")
    with (artifact_dir / "static_feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(static_columns, handle, indent=2)
        handle.write("\n")
    with (artifact_dir / "sequence_feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(sequence_columns, handle, indent=2)
        handle.write("\n")
    pd.DataFrame(history_rows).to_csv(artifact_dir / "training_history.csv", index=False)
    predictions.to_csv(artifact_dir / "validation_predictions.csv", index=False)

    per_target_metrics = {
        "won": {
            "roc_auc": float(winner_auc),
            "log_loss": winner_log_loss,
            "brier_score": winner_brier,
            "event_log_loss": winner_event_log_loss,
            "top_pick_hit_rate": float(top_pick["won"].mean()) if not top_pick.empty else 0.0,
            "top_3_hit_rate": float(top_3),
        }
    }
    for target in AUX_TARGETS:
        per_target_metrics[target] = {
            "roc_auc": float(roc_auc_score(predictions[target], predictions[f"{target}_probability"]))
            if len(np.unique(predictions[target])) > 1
            else None,
            "log_loss": float(log_loss(predictions[target], predictions[f"{target}_probability"], labels=[0, 1])),
            "brier_score": float(brier_score_loss(predictions[target], predictions[f"{target}_probability"])),
            "positive_weight": aux_positive_weights[target],
        }

    metrics = {
        "dataset_path": str(dataset_path),
        "artifact_dir": str(artifact_dir),
        "validation_prediction_path": str(artifact_dir / "validation_predictions.csv"),
        "feature_snapshot_mode": dataset["feature_snapshot_mode"].iloc[0] if "feature_snapshot_mode" in dataset.columns else "unknown",
        "leakage_warning": bool(dataset["feature_leakage_warning"].fillna(False).astype(bool).any()) if "feature_leakage_warning" in dataset.columns else False,
        "device": str(device),
        "sequence_length": int(args.sequence_length),
        "sequence_feature_count": len(sequence_columns),
        "static_feature_count": len(static_columns),
        "train_tournaments": int(len(train_groups)),
        "validation_tournaments": int(len(val_groups)),
        "train_rows": int(sum(group.sequences.shape[0] for group in train_groups)),
        "validation_rows": int(sum(group.sequences.shape[0] for group in val_groups)),
        "tournament_batch_size": int(args.tournament_batch_size),
        "best_epoch": int(best_epoch),
        "per_target": per_target_metrics,
        "notes": [
            "Winner head is trained with tournament-level softmax ranking loss rather than independent BCE.",
            "Auxiliary heads retain top_10 and made_cut supervision to stabilize shared golf representations.",
            "This path is designed to improve top-pick ranking quality on Apple Silicon MPS hardware.",
        ],
    }
    write_json(artifact_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    metrics = train_tournament_ranker(args)
    print("=" * 72)
    print("PGA TOURNAMENT RANKER TRAINING COMPLETE")
    print("=" * 72)
    print(f"Device:                 {metrics['device']}")
    print(f"Sequence length:        {metrics['sequence_length']}")
    print(f"Sequence features:      {metrics['sequence_feature_count']}")
    print(f"Static features:        {metrics['static_feature_count']}")
    print(f"Train tournaments:      {metrics['train_tournaments']}")
    print(f"Validation tournaments: {metrics['validation_tournaments']}")
    print(f"Winner ROC AUC:         {metrics['per_target']['won']['roc_auc']}")
    print(f"Winner log loss:        {metrics['per_target']['won']['log_loss']:.6f}")
    print(f"Winner event log loss:  {metrics['per_target']['won']['event_log_loss']:.6f}")
    print(f"Winner top-pick hit:    {metrics['per_target']['won']['top_pick_hit_rate']:.2%}")
    print(f"Winner top-3 hit:       {metrics['per_target']['won']['top_3_hit_rate']:.2%}")
    print(f"Artifact dir:           {metrics['artifact_dir']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
