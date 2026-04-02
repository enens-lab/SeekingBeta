import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.wta.train_tournament_ranker_torch import WTATournamentRanker
from sports.player_media import resolve_player_image


DIV_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = DIV_ROOT / "artifacts" / "wta_tournament_ranker_torch"
DATASET_PATH = DIV_ROOT / "data" / "sports" / "wta" / "normalized" / "wta_training_dataset_latest.csv"
FRONTEND_DATA_DIR = PROJ_ROOT / "pythia_prophecy" / "frontend" / "src" / "data"
UPCOMING_PER_TOUR = {"ATP": 12, "WTA": 12}


def _append_stat(stats: list[dict[str, str]], label: str, value: str | None) -> None:
    if value:
        stats.append({"label": label, "value": value})


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_tennis_profile(row) -> dict:
    stats: list[dict[str, str]] = []
    elo = _safe_float(getattr(row, "elo", None))
    surf_elo = _safe_float(getattr(row, "surf_elo", None))
    serve_won = _safe_float(getattr(row, "serve_won", None))
    return_won = _safe_float(getattr(row, "return_won", None))
    age = _safe_float(getattr(row, "age", None))
    height = _safe_float(getattr(row, "height", None))
    tour = str(getattr(row, "tour", "Tennis"))
    surface = str(getattr(row, "surface", "Unknown"))

    _append_stat(stats, "Elo", f"{elo:.0f}" if elo is not None else None)
    _append_stat(stats, "Surface Elo", f"{surf_elo:.0f}" if surf_elo is not None else None)
    _append_stat(stats, "Serve Won", f"{serve_won * 100:.1f}%" if serve_won is not None else None)
    _append_stat(stats, "Return Won", f"{return_won * 100:.1f}%" if return_won is not None else None)
    if len(stats) < 4:
        _append_stat(stats, "Age", f"{age:.1f}" if age is not None else None)
    if len(stats) < 4:
        _append_stat(stats, "Height", f"{height:.0f} cm" if height is not None else None)

    subtitle_parts = [tour, surface]
    if age is not None:
        subtitle_parts.append(f"Age {age:.1f}")

    return {
        "imageUrl": resolve_player_image(
            name=str(row.player_name),
            sport="tennis",
            tour=tour,
            player_id=getattr(row, "player_id", None),
            player_key=getattr(row, "player_key", None),
        ),
        "subtitle": " | ".join(subtitle_parts),
        "country": None,
        "stats": stats[:4],
    }


def _load_model():
    with open(ARTIFACT_DIR / "feature_columns.json", "r") as f:
        static_columns = json.load(f)

    imputer = joblib.load(ARTIFACT_DIR / "imputer.joblib")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
    model = WTATournamentRanker(
        static_feature_count=len(static_columns),
        static_width=256,
        dense_width=128,
        dropout=0.3,
    )
    model.load_state_dict(torch.load(ARTIFACT_DIR / "model.pt", map_location="cpu"))
    model.eval()
    return model, imputer, scaler, static_columns


def _predict_group(model, imputer, scaler, static_columns, group: pd.DataFrame):
    x_raw = group[static_columns]
    x = scaler.transform(imputer.transform(x_raw))
    x_tensor = torch.from_numpy(x).float().unsqueeze(0)

    with torch.no_grad():
        logits = model(x_tensor)["winner"]
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    sorted_idx = np.argsort(-probs)
    return probs, sorted_idx


def _build_backtests(df: pd.DataFrame) -> list[dict]:
    model, imputer, scaler, static_columns = _load_model()

    df = df.sort_values(["date", "tour", "tournament_name"])
    tourneys = df["tournament_id"].unique()
    val_ids = set(tourneys[int(len(tourneys) * 0.8):])
    val_df = df[df["tournament_id"].isin(val_ids)].copy()

    backtests = []
    for _, group in val_df.groupby("tournament_id", sort=False):
        if group.empty:
            continue
        try:
            probs, sorted_idx = _predict_group(model, imputer, scaler, static_columns, group)
        except Exception:
            continue

        actual_winner_row = group[group["won_tournament"] == 1]
        if actual_winner_row.empty:
            continue
        actual_winner = str(actual_winner_row.iloc[0]["player_name"])
        top_winner = group.iloc[sorted_idx[0]]

        predicted_top_3 = [str(group.iloc[sorted_idx[i]]["player_name"]) for i in range(min(3, len(sorted_idx)))]
        predicted_top_5 = [str(group.iloc[sorted_idx[i]]["player_name"]) for i in range(min(5, len(sorted_idx)))]
        if actual_winner == str(top_winner["player_name"]):
            hit_status = "Top Pick"
        elif actual_winner in predicted_top_3:
            hit_status = "Top 3"
        elif actual_winner in predicted_top_5:
            hit_status = "Top 5"
        else:
            hit_status = "Miss"

        full_field = []
        for idx in sorted_idx:
            row = group.iloc[idx]
            full_field.append(
                {
                    "rank": len(full_field) + 1,
                    "playerName": str(row["player_name"]),
                    "winProbability": float(probs[idx] * 100),
                    "actualWinner": bool(row["won_tournament"] == 1),
                    "profile": _build_tennis_profile(row),
                }
            )

        backtests.append(
            {
                "year": int(group["date"].iloc[0] // 10000),
                "tournament": str(group["tournament_name"].iloc[0]),
                "tour": str(group["tour"].iloc[0]),
                "surface": str(group["surface"].iloc[0]) if "surface" in group.columns else "Unknown",
                "predictedWinner": str(top_winner["player_name"]),
                "predictedTop3": predicted_top_3,
                "predictedTop5": predicted_top_5,
                "actualWinner": actual_winner,
                "hitStatus": hit_status,
                "prob": float(probs[sorted_idx[0]]),
                "fullField": full_field,
                "latestDate": int(group["date"].max()),
                "tournamentId": str(group["tournament_id"].iloc[0]),
            }
        )

    return sorted(
        backtests,
        key=lambda row: (-row["year"], row["tour"], row["tournament"]),
    )


def _build_upcoming(backtests: list[dict]) -> list[dict]:
    today = datetime.now()
    today_key = today.year * 10000 + today.month * 100 + today.day
    per_tour_counts = {tour: 0 for tour in UPCOMING_PER_TOUR}
    upcoming = []
    seen = set()

    sorted_backtests = sorted(
        backtests,
        key=lambda row: ((row.get("latestDate", 0) or 0) % 10000, row["tour"], row["tournament"]),
    )

    for bt in sorted_backtests:
        latest_date = int(bt.get("latestDate", 0) or 0)
        if latest_date <= 0:
            continue

        synthetic_event_date = today.year * 10000 + (latest_date % 10000)
        if synthetic_event_date < today_key:
            continue

        tournament_key = (bt["tour"], bt["tournament"])
        if tournament_key in seen:
            continue
        if per_tour_counts.get(bt["tour"], 0) >= UPCOMING_PER_TOUR.get(bt["tour"], 10):
            continue

        seen.add(tournament_key)
        per_tour_counts[bt["tour"]] = per_tour_counts.get(bt["tour"], 0) + 1
        upcoming.append(
            {
                "id": f"{bt['tour'].lower()}-{bt['tournament'].replace(' ', '-').lower()}",
                "name": f"{today.year} {bt['tournament']}",
                "tour": bt["tour"],
                "course": bt.get("surface", "Unknown Surface"),
                "scheduledDate": synthetic_event_date,
                "predictions": bt["fullField"],
            }
        )

    return sorted(upcoming, key=lambda row: (row["scheduledDate"], row["tour"], row["name"]))


def export_wta_frontend_data():
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    backtests = _build_backtests(df)
    upcoming = _build_upcoming(backtests)

    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (FRONTEND_DATA_DIR / "wta_historical_backtests.json").write_text(json.dumps(backtests, indent=2))
    (FRONTEND_DATA_DIR / "wta_upcoming_tournaments.json").write_text(json.dumps(upcoming, indent=2))

    backtest_tours = pd.Series([row["tour"] for row in backtests]).value_counts().to_dict()
    upcoming_tours = pd.Series([row["tour"] for row in upcoming]).value_counts().to_dict()
    print(f"Exported {len(backtests)} tennis backtests by tour: {backtest_tours}")
    print(f"Exported {len(upcoming)} tennis upcoming events by tour: {upcoming_tours}")


if __name__ == "__main__":
    export_wta_frontend_data()
