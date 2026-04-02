import json
import sys
from pathlib import Path

import pandas as pd


DIV_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

from sports.player_media import resolve_player_image

META_PATH = DIV_ROOT / "data" / "sports" / "pga" / "normalized" / "golf_training_dataset_latest.csv"
FRONTEND_DATA_DIR = PROJ_ROOT / "pythia_prophecy" / "frontend" / "src" / "data"

PREDICTION_SOURCES = [
    {
        "path": DIV_ROOT / "artifacts" / "pga_neural_multitask_torch" / "validation_predictions.csv",
        "probability_columns": [
            "won_normalized_probability",
            "won_probability",
            "winner_probability",
        ],
        "top_10_column": "top_10_probability",
        "made_cut_column": "made_cut_probability",
        "tour_hint": "PGA",
        "source_priority": 0,
    },
    {
        "path": DIV_ROOT / "artifacts" / "pga_tournament_ranker_torch" / "validation_predictions.csv",
        "probability_columns": ["winner_probability"],
        "top_10_column": "top_10_probability",
        "made_cut_column": "made_cut_probability",
        "tour_hint": "LPGA",
        "source_priority": 1,
    },
    {
        "path": DIV_ROOT / "artifacts" / "pga_neural" / "won" / "validation_predictions.csv",
        "probability_columns": ["normalized_win_probability", "raw_probability"],
        "top_10_column": None,
        "made_cut_column": None,
        "tour_hint": "PGA",
        "source_priority": 2,
    },
]

TOUR_PRIORITY = {"PGA": 0, "LPGA": 1}
UPCOMING_PER_TOUR = {"PGA": 18, "LPGA": 18}


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_stat(stats: list[dict[str, str]], label: str, value: str | None) -> None:
    if value:
        stats.append({"label": label, "value": value})


def _build_golf_profile(row) -> dict:
    country = row.feature_country_name if pd.notna(getattr(row, "feature_country_name", None)) else getattr(row, "country", None)
    stats: list[dict[str, str]] = []

    owgr_rank = _safe_float(getattr(row, "owgr__rank", None))
    sg_total = _safe_float(getattr(row, "sg_total__avg", None))
    sg_approach = _safe_float(getattr(row, "sg_approach__avg", None))
    sg_putting = _safe_float(getattr(row, "sg_putting__avg", None))
    top10_prob = _safe_float(getattr(row, "top_10_probability", None))
    made_cut_prob = _safe_float(getattr(row, "made_cut_probability", None))

    _append_stat(stats, "OWGR", f"#{int(round(owgr_rank))}" if owgr_rank is not None else None)
    _append_stat(stats, "SG Total", f"{sg_total:+.2f}" if sg_total is not None else None)
    _append_stat(stats, "Approach", f"{sg_approach:+.2f}" if sg_approach is not None else None)
    if len(stats) < 4:
        _append_stat(stats, "Putting", f"{sg_putting:+.2f}" if sg_putting is not None else None)
    if len(stats) < 4:
        _append_stat(stats, "Top 10", f"{top10_prob * 100:.1f}%" if top10_prob is not None else None)
    if len(stats) < 4:
        _append_stat(stats, "Made Cut", f"{made_cut_prob * 100:.1f}%" if made_cut_prob is not None else None)

    subtitle_parts = [part for part in [country, getattr(row, "tour", None)] if part and not pd.isna(part)]
    return {
        "imageUrl": resolve_player_image(
            name=str(row.player_name),
            sport="golf",
            tour=str(getattr(row, "tour", "PGA")),
            player_id=getattr(row, "player_id", None),
        ),
        "subtitle": " | ".join(str(part) for part in subtitle_parts) if subtitle_parts else str(getattr(row, "tour", "Golf")),
        "country": str(country) if country and not pd.isna(country) else None,
        "stats": stats[:4],
    }


def _load_meta() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    meta = pd.read_csv(META_PATH, low_memory=False)

    tournament_meta = (
        meta[
            [
                "tournament_id",
                "tournament_name",
                "season_year",
                "display_date",
                "course_name",
                "course_state_code",
                "tour",
            ]
        ]
        .dropna(subset=["tournament_id"])
        .sort_values(["tournament_id", "season_year", "display_date"])
        .drop_duplicates(subset=["tournament_id"], keep="last")
    )

    player_outcomes = (
        meta[
            [
                "tournament_id",
                "player_name",
                "won",
                "player_id",
                "country",
                "feature_country_name",
                "owgr__rank",
                "sg_total__avg",
                "sg_approach__avg",
                "sg_putting__avg",
            ]
        ]
        .dropna(subset=["tournament_id", "player_name"])
        .drop_duplicates(subset=["tournament_id", "player_name"], keep="last")
    )

    winners = (
        meta.loc[meta["won"] == 1, ["tournament_id", "player_name"]]
        .dropna(subset=["tournament_id", "player_name"])
        .drop_duplicates(subset=["tournament_id"], keep="last")
        .set_index("tournament_id")["player_name"]
        .to_dict()
    )

    return tournament_meta, player_outcomes, winners


def _choose_probability_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Missing probability column. Tried: {candidates}")


def _load_predictions(tournament_meta: pd.DataFrame, player_outcomes: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for config in PREDICTION_SOURCES:
        path = config["path"]
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if df.empty or "tournament_id" not in df.columns or "player_name" not in df.columns:
            continue

        probability_column = _choose_probability_column(df, config["probability_columns"])

        renamed = df.copy()
        renamed["winner_probability"] = renamed[probability_column]
        renamed["top_10_probability"] = (
            renamed[config["top_10_column"]]
            if config["top_10_column"] and config["top_10_column"] in renamed.columns
            else pd.NA
        )
        renamed["made_cut_probability"] = (
            renamed[config["made_cut_column"]]
            if config["made_cut_column"] and config["made_cut_column"] in renamed.columns
            else pd.NA
        )
        renamed["source_priority"] = config["source_priority"]
        renamed["tour_hint"] = config["tour_hint"]

        merged = renamed.merge(
            tournament_meta,
            on="tournament_id",
            how="left",
            suffixes=("", "_meta"),
        )
        merged = merged.merge(
            player_outcomes,
            on=["tournament_id", "player_name"],
            how="left",
            suffixes=("", "_label"),
        )

        merged["tour"] = merged["tour"].fillna(merged["tour_hint"])
        merged["tournament_name"] = merged["tournament_name"].fillna(merged.get("tournament_name_meta"))
        merged["won"] = merged["won"].fillna(0).astype(int)

        frames.append(
            merged[
                [
                    "tournament_id",
                    "tournament_name",
                    "player_name",
                    "winner_probability",
                    "top_10_probability",
                    "made_cut_probability",
                    "won",
                    "tour",
                    "season_year",
                    "display_date",
                    "course_name",
                    "course_state_code",
                    "player_id",
                    "country",
                    "feature_country_name",
                    "owgr__rank",
                    "sg_total__avg",
                    "sg_approach__avg",
                    "sg_putting__avg",
                    "source_priority",
                ]
            ]
        )

    if not frames:
        raise FileNotFoundError("No usable golf prediction artifacts were found.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["tournament_id", "player_name", "winner_probability"])
    combined = combined.sort_values(
        ["source_priority", "tournament_id", "player_name", "winner_probability"],
        ascending=[True, True, True, False],
    )
    combined = combined.drop_duplicates(
        subset=["tournament_id", "player_name"],
        keep="first",
    )
    combined["tournament_name"] = combined["tournament_name"].fillna("Unknown Tournament")
    combined["tour"] = combined["tour"].fillna("PGA")

    return combined


def _sort_priority(event_name: str) -> int:
    name = event_name.lower()
    if "masters" in name:
        return 1
    if "pga championship" in name:
        return 2
    if "u.s. open" in name or "us open" in name:
        return 3
    if "open championship" in name or name == "the open":
        return 4
    if "players championship" in name:
        return 5
    if "evian" in name:
        return 6
    if "women's pga" in name or "womens pga" in name:
        return 7
    if "chevron" in name:
        return 8
    if "aig women's open" in name:
        return 9
    return 20


def _build_backtests(df: pd.DataFrame, winners: dict[str, str]) -> list[dict]:
    backtests: list[dict] = []

    for tournament_id, group in df.groupby("tournament_id", sort=False):
        top_preds = group.sort_values("winner_probability", ascending=False).reset_index(drop=True)
        if top_preds.empty:
            continue

        actual_winner = winners.get(tournament_id)
        if not actual_winner:
            winner_rows = top_preds[top_preds["won"] == 1]
            actual_winner = winner_rows.iloc[0]["player_name"] if not winner_rows.empty else "Unknown"

        predicted_top_3 = top_preds.head(3)["player_name"].tolist()
        predicted_top_5 = top_preds.head(5)["player_name"].tolist()
        predicted_winner = top_preds.iloc[0]["player_name"]

        if actual_winner == predicted_winner:
            hit_status = "Top Pick"
        elif actual_winner in predicted_top_3:
            hit_status = "Top 3"
        elif actual_winner in predicted_top_5:
            hit_status = "Top 5"
        else:
            hit_status = "Miss"

        full_field = []
        for rank, row in enumerate(top_preds.itertuples(index=False), start=1):
            full_field.append(
                {
                    "rank": rank,
                    "playerName": row.player_name,
                    "winProbability": float(row.winner_probability * 100),
                    "actualWinner": bool(row.player_name == actual_winner),
                    "profile": _build_golf_profile(row),
                }
            )

        backtests.append(
            {
                "year": int(top_preds.iloc[0]["season_year"]) if pd.notna(top_preds.iloc[0]["season_year"]) else 2025,
                "tournament": str(top_preds.iloc[0]["tournament_name"]),
                "tour": str(top_preds.iloc[0]["tour"]),
                "predictedWinner": predicted_winner,
                "predictedTop3": predicted_top_3,
                "predictedTop5": predicted_top_5,
                "actualWinner": actual_winner,
                "hitStatus": hit_status,
                "prob": float(top_preds.iloc[0]["winner_probability"]),
                "fullField": full_field,
            }
        )

    return sorted(
        backtests,
        key=lambda row: (
            -row["year"],
            TOUR_PRIORITY.get(row["tour"], 99),
            _sort_priority(row["tournament"]),
            row["tournament"],
        ),
    )


def _build_upcoming(df: pd.DataFrame) -> list[dict]:
    upcoming: list[dict] = []

    event_frames = []
    for (tour, tournament_name), group in df.groupby(["tour", "tournament_name"], sort=False):
        latest_year = group["season_year"].max()
        latest_event = (
            group[group["season_year"] == latest_year]
            .sort_values("winner_probability", ascending=False)
            .reset_index(drop=True)
        )
        if latest_event.empty:
            continue
        event_frames.append((tour, tournament_name, latest_event))

    event_frames.sort(
        key=lambda item: (
            TOUR_PRIORITY.get(item[0], 99),
            _sort_priority(item[1]),
            item[1],
        )
    )

    per_tour_counts = {tour: 0 for tour in TOUR_PRIORITY}
    for tour, tournament_name, latest_event in event_frames:
        max_for_tour = UPCOMING_PER_TOUR.get(tour, 12)
        if per_tour_counts.get(tour, 0) >= max_for_tour:
            continue

        predictions = [
            {
                "rank": rank,
                "playerName": row.player_name,
                "winProbability": float(row.winner_probability * 100),
                "profile": _build_golf_profile(row),
            }
            for rank, row in enumerate(latest_event.itertuples(index=False), start=1)
        ]

        course = latest_event.iloc[0]["course_name"] if pd.notna(latest_event.iloc[0]["course_name"]) else "TBD Course"
        state = latest_event.iloc[0]["course_state_code"] if pd.notna(latest_event.iloc[0]["course_state_code"]) else ""

        upcoming.append(
            {
                "id": f"{tour.lower()}-{str(tournament_name).replace(' ', '-').lower()}",
                "name": f"2026 {tournament_name}",
                "original_name": str(tournament_name),
                "tour": str(tour),
                "course": f"{course}, {state}".strip(", "),
                "predictions": predictions,
            }
        )
        per_tour_counts[tour] = per_tour_counts.get(tour, 0) + 1

    return upcoming


def main() -> None:
    tournament_meta, player_outcomes, winners = _load_meta()
    predictions = _load_predictions(tournament_meta, player_outcomes)

    backtests = _build_backtests(predictions, winners)
    upcoming = _build_upcoming(predictions)

    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (FRONTEND_DATA_DIR / "historical_backtests.json").write_text(json.dumps(backtests, indent=2))
    (FRONTEND_DATA_DIR / "upcoming_tournaments.json").write_text(json.dumps(upcoming, indent=2))

    backtest_tours = pd.Series([row["tour"] for row in backtests]).value_counts().to_dict()
    upcoming_tours = pd.Series([row["tour"] for row in upcoming]).value_counts().to_dict()
    print(f"Generated {len(backtests)} golf backtests by tour: {backtest_tours}")
    print(f"Generated {len(upcoming)} golf upcoming events by tour: {upcoming_tours}")


if __name__ == "__main__":
    main()
