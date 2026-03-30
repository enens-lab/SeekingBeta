"""Roster-availability and transaction feature helpers for MLB."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def classify_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return transactions

    frame = transactions.copy()
    frame["effective_date"] = pd.to_datetime(
        frame.get("effective_date", frame.get("date")),
        errors="coerce",
    ).fillna(pd.to_datetime(frame.get("date"), errors="coerce"))
    frame["description_text"] = frame.get("description", "").map(_normalize_text)
    frame["type_desc_text"] = frame.get("type_desc", "").map(_normalize_text)

    activation_keywords = ("activated", "reinstated", "returned from injured list")
    addition_keywords = ("placed on", "transferred to")
    roster_add_keywords = ("recalled", "selected", "claimed", "acquired")
    roster_loss_keywords = ("optioned", "assigned", "designated", "waived", "released")

    frame["is_il_activation"] = frame["description_text"].apply(
        lambda text: "injured list" in text and any(keyword in text for keyword in activation_keywords)
    )
    frame["is_il_addition"] = frame["description_text"].apply(
        lambda text: "injured list" in text
        and any(keyword in text for keyword in addition_keywords)
        and not any(keyword in text for keyword in activation_keywords)
    )
    frame["is_rehab_assignment"] = frame["description_text"].str.contains("rehab assignment", regex=False)
    frame["is_roster_addition"] = frame["type_desc_text"].apply(
        lambda text: any(keyword in text for keyword in roster_add_keywords)
    )
    frame["is_roster_loss"] = frame["type_desc_text"].apply(
        lambda text: any(keyword in text for keyword in roster_loss_keywords)
    )
    frame["is_status_change"] = frame["type_desc_text"].eq("status change")
    return frame


def build_transaction_feature_frame(games: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    if games.empty or transactions.empty:
        return pd.DataFrame()

    tx = classify_transactions(transactions)
    tx = tx.dropna(subset=["effective_date"]).copy()
    tx["team_id"] = pd.to_numeric(tx["team_id"], errors="coerce")

    lookbacks = (7, 14, 30)
    team_game_rows: list[dict[str, Any]] = []
    for game in games.itertuples(index=False):
        for side, team_id, team_name in (
            ("away", getattr(game, "away_team_id"), getattr(game, "away_team_name")),
            ("home", getattr(game, "home_team_id"), getattr(game, "home_team_name")),
        ):
            team_game_rows.append(
                {
                    "game_pk": game.game_pk,
                    "official_date": pd.to_datetime(game.official_date, errors="coerce"),
                    "team_side": side,
                    "team_id": team_id,
                    "team_name": team_name,
                }
            )

    base = pd.DataFrame(team_game_rows)
    output_rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        team_transactions = tx.loc[tx["team_id"] == row.team_id]
        record: dict[str, Any] = {
            "game_pk": row.game_pk,
            "team_side": row.team_side,
            "team_id": row.team_id,
            "team_name": row.team_name,
        }
        if team_transactions.empty or pd.isna(row.official_date):
            for window in lookbacks:
                record[f"availability_transactions_last_{window}"] = 0
                record[f"availability_il_additions_last_{window}"] = 0
                record[f"availability_il_activations_last_{window}"] = 0
                record[f"availability_roster_additions_last_{window}"] = 0
                record[f"availability_roster_losses_last_{window}"] = 0
                record[f"availability_rehab_assignments_last_{window}"] = 0
            output_rows.append(record)
            continue

        for window in lookbacks:
            start_date = row.official_date - pd.Timedelta(days=window)
            recent = team_transactions.loc[
                (team_transactions["effective_date"] < row.official_date)
                & (team_transactions["effective_date"] >= start_date)
            ]
            record[f"availability_transactions_last_{window}"] = int(len(recent))
            record[f"availability_il_additions_last_{window}"] = int(recent["is_il_addition"].sum())
            record[f"availability_il_activations_last_{window}"] = int(recent["is_il_activation"].sum())
            record[f"availability_roster_additions_last_{window}"] = int(recent["is_roster_addition"].sum())
            record[f"availability_roster_losses_last_{window}"] = int(recent["is_roster_loss"].sum())
            record[f"availability_rehab_assignments_last_{window}"] = int(recent["is_rehab_assignment"].sum())
        output_rows.append(record)

    return pd.DataFrame(output_rows)


def merge_transaction_features(games: pd.DataFrame, transaction_features: pd.DataFrame) -> pd.DataFrame:
    if transaction_features.empty:
        return games

    away = transaction_features.loc[transaction_features["team_side"] == "away"].copy()
    away = away.rename(
        columns={column: f"away_{column}" for column in away.columns if column not in {"game_pk", "team_side", "team_id", "team_name"}}
    )
    away = away.rename(columns={"team_id": "away_team_id", "team_name": "away_team_name"})

    home = transaction_features.loc[transaction_features["team_side"] == "home"].copy()
    home = home.rename(
        columns={column: f"home_{column}" for column in home.columns if column not in {"game_pk", "team_side", "team_id", "team_name"}}
    )
    home = home.rename(columns={"team_id": "home_team_id", "team_name": "home_team_name"})

    merged = games.merge(away.drop(columns=["team_side"]), on=["game_pk", "away_team_id", "away_team_name"], how="left")
    merged = merged.merge(home.drop(columns=["team_side"]), on=["game_pk", "home_team_id", "home_team_name"], how="left")
    return merged
