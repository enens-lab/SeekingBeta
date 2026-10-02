"""Golf (PGA + LPGA) boards for the web/iOS/Android clients.

Historical boards are the multitask model's held-out (chronological validation)
predictions for events with exactly one known winner; each names the best-OWGR
player as the baseline. Top-10 and made-cut probabilities are calibrated walk-
forward (sports/pga/calibration.py) because the model was trained with
pos_weight and over-stated them about four-fold; LPGA top-10/made-cut heads were
trained on all-zero labels and are hidden. Upcoming boards carry the previous
edition's field, labelled ``fieldBasis: "previous_edition"``, and are dropped
when they cannot be dated. Also writes golf_model_metrics.json.
"""

from __future__ import annotations  # py3.9-compatible PEP 604 annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


DIV_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

from sports.pga.calibration import calibrate_board, calibration_report
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
        "calibrate": True,
    },
    {
        "path": DIV_ROOT / "artifacts" / "pga_tournament_ranker_torch" / "validation_predictions.csv",
        "probability_columns": ["winner_probability"],
        "top_10_column": "top_10_probability",
        "made_cut_column": "made_cut_probability",
        "tour_hint": "LPGA",
        "source_priority": 1,
        # Trained on all-zero top-10 / made-cut labels (LPGA tables carry none).
        "hide_aux_heads": True,
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
UPCOMING_LOOKAHEAD_DAYS = 120
ESPN_GOLF_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/{tour}/scoreboard"
MODEL_VERSION = "pga-multitask-torch+wf-calibration-v1"


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


def _clamp_score(value: float | None, low: float, high: float, *, inverse: bool = False) -> float | None:
    if value is None or high <= low:
        return None
    clipped = min(max(value, low), high)
    ratio = (clipped - low) / (high - low)
    if inverse:
        ratio = 1.0 - ratio
    return round(ratio * 100.0, 1)


def _build_golf_radar(row) -> list[dict[str, float]]:
    metrics = [
        ("World Rank", _clamp_score(_safe_float(getattr(row, "owgr__rank", None)), 1.0, 200.0, inverse=True)),
        ("SG Total", _clamp_score(_safe_float(getattr(row, "sg_total__avg", None)), -1.5, 3.5)),
        ("Tee to Green", _clamp_score(_safe_float(getattr(row, "sg_tee_to_green__avg", None)), -1.2, 2.8)),
        ("Approach", _clamp_score(_safe_float(getattr(row, "sg_approach__avg", None)), -1.2, 2.4)),
        ("Putting", _clamp_score(_safe_float(getattr(row, "sg_putting__avg", None)), -1.8, 2.0)),
        ("Birdie Rate", _clamp_score(_safe_float(getattr(row, "birdie_or_better_pct__value", None)), 10.0, 30.0)),
    ]
    return [{"label": label, "value": value or 0.0} for label, value in metrics if value is not None]


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


META_COLUMNS = {
    "tournament_id",
    "tournament_name",
    "season_year",
    "display_date",
    "event_start_date",
    "course_name",
    "course_state_code",
    "tour",
    "player_name",
    "player_id",
    "won",
    "top_10",
    "made_cut",
    "event_field_size",
    "country",
    "feature_country_name",
    "owgr__rank",
    "sg_total__avg",
    "sg_approach__avg",
    "sg_putting__avg",
    "sg_tee_to_green__avg",
    "birdie_or_better_pct__value",
}


def _binary(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}
    )
    return mapped.astype(float)


def _load_meta() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], pd.DataFrame]:
    meta = pd.read_csv(META_PATH, usecols=lambda column: column in META_COLUMNS, low_memory=False)
    for column in ("won", "top_10", "made_cut"):
        if column in meta.columns:
            meta[column] = _binary(meta[column])

    tournament_meta = (
        meta[
            [
                "tournament_id",
                "tournament_name",
                "season_year",
                "display_date",
                # Real ISO event date. `display_date` is a bare human string
                # ("Sep 30 - Oct 3") with no year, so it cannot key a date.
                "event_start_date",
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
                "sg_tee_to_green__avg",
                "birdie_or_better_pct__value",
            ]
        ]
        .dropna(subset=["tournament_id", "player_name"])
        .drop_duplicates(subset=["tournament_id", "player_name"], keep="last")
    )

    # Only events with exactly one recorded winner can be graded. Events with no
    # winner in the table (not yet played / missing results) used to ship as
    # "Miss" against an "Unknown" winner.
    winner_rows = meta.loc[meta["won"] == 1, ["tournament_id", "player_name"]].dropna()
    winner_counts = winner_rows.groupby("tournament_id")["player_name"].nunique()
    single = winner_counts[winner_counts == 1].index
    winners = (
        winner_rows[winner_rows["tournament_id"].isin(single)]
        .drop_duplicates(subset=["tournament_id"], keep="last")
        .set_index("tournament_id")["player_name"]
        .to_dict()
    )

    labels = meta[
        ["tournament_id", "tournament_name", "player_id", "event_start_date", "won", "top_10", "made_cut", "tour", "event_field_size"]
    ].copy()
    labels["player_id"] = pd.to_numeric(labels["player_id"], errors="coerce")
    labels = labels.dropna(subset=["tournament_id", "player_id"])

    return tournament_meta, player_outcomes, winners, labels


def _choose_probability_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Missing probability column. Tried: {candidates}")


def _calibrate_source(df: pd.DataFrame, probability_column: str, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Walk-forward calibrate the top-10 / made-cut heads (sports/pga/calibration.py)."""
    frame = df.copy()
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    inputs = frame[["tournament_id", "tournament_name", "player_id"]].copy()
    inputs["winner_probability"] = frame[probability_column].astype(float)
    inputs["top_10_probability"] = frame["top_10_probability"].astype(float)
    inputs["made_cut_probability"] = frame["made_cut_probability"].astype(float)
    pga_labels = labels[labels["tour"].astype(str).str.upper() == "PGA"]
    calibrated, log = calibrate_board(inputs, pga_labels)
    report = calibration_report(calibrated, pga_labels)
    frame["top_10_probability"] = calibrated["top_10_probability"].to_numpy()
    frame["made_cut_probability"] = calibrated["made_cut_probability"].to_numpy()
    return frame, report, log


def _load_predictions(
    tournament_meta: pd.DataFrame,
    player_outcomes: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    frames: list[pd.DataFrame] = []
    calibration: dict = {}

    for config in PREDICTION_SOURCES:
        path = config["path"]
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if df.empty or "tournament_id" not in df.columns or "player_name" not in df.columns:
            continue

        probability_column = _choose_probability_column(df, config["probability_columns"])
        top_10_column = config["top_10_column"]
        made_cut_column = config["made_cut_column"]
        if config.get("calibrate") and top_10_column in df.columns and made_cut_column in df.columns:
            df = df.rename(columns={top_10_column: "top_10_probability", made_cut_column: "made_cut_probability"})
            df, report, log = _calibrate_source(df, probability_column, labels)
            calibration = {"source": str(path.relative_to(DIV_ROOT)), "report": report, "log": log}
            top_10_column, made_cut_column = "top_10_probability", "made_cut_probability"

        renamed = df.copy()
        renamed["winner_probability"] = renamed[probability_column]
        show_aux = not config.get("hide_aux_heads")
        renamed["top_10_probability"] = (
            renamed[top_10_column].astype(float) if show_aux and top_10_column and top_10_column in renamed.columns else float("nan")
        )
        renamed["made_cut_probability"] = (
            renamed[made_cut_column].astype(float) if show_aux and made_cut_column and made_cut_column in renamed.columns else float("nan")
        )
        renamed["source_priority"] = config["source_priority"]
        renamed["tour_hint"] = config["tour_hint"]
        renamed = renamed.drop(columns=[column for column in ("won",) if column in renamed.columns])

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
                    "event_start_date",
                    "course_name",
                    "course_state_code",
                    "player_id",
                    "country",
                    "feature_country_name",
                    "owgr__rank",
                    "sg_total__avg",
                    "sg_approach__avg",
                    "sg_putting__avg",
                    "sg_tee_to_green__avg",
                    "birdie_or_better_pct__value",
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

    return combined, calibration


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


def _event_date_key(value) -> int | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.strftime("%Y%m%d"))


def _build_backtests(df: pd.DataFrame, winners: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Graded boards for held-out events with exactly one known winner, plus a
    scoring row per board (model top pick vs the best-OWGR player)."""
    backtests: list[dict] = []
    scoring: list[dict] = []

    for tournament_id, group in df.groupby("tournament_id", sort=False):
        top_preds = group.sort_values("winner_probability", ascending=False).reset_index(drop=True)
        if top_preds.empty:
            continue
        actual_winner = winners.get(tournament_id)
        if not actual_winner or actual_winner not in set(top_preds["player_name"]):
            # No graded result for this event (or the winner is not in the board's
            # field): it cannot be scored, so it is not a historical board.
            continue

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
                    "radarMetrics": _build_golf_radar(row),
                }
            )

        event_date_key = _event_date_key(top_preds.iloc[0]["event_start_date"]) if "event_start_date" in top_preds.columns else None
        owgr = pd.to_numeric(top_preds["owgr__rank"], errors="coerce")
        favourite = str(top_preds.at[int(owgr.idxmin()), "player_name"]) if owgr.notna().sum() > len(top_preds) / 2 else None
        tour = str(top_preds.iloc[0]["tour"])
        year = int(top_preds.iloc[0]["season_year"]) if pd.notna(top_preds.iloc[0]["season_year"]) else (event_date_key // 10000 if event_date_key else 2025)

        backtests.append(
            {
                "year": year,
                "tournament": str(top_preds.iloc[0]["tournament_name"]),
                "tour": tour,
                "predictedWinner": predicted_winner,
                "predictedTop3": predicted_top_3,
                "predictedTop5": predicted_top_5,
                "actualWinner": actual_winner,
                "hitStatus": hit_status,
                "prob": float(top_preds.iloc[0]["winner_probability"]),
                # Lets date-windowed surfaces (Daily Brief, weekly Receipts, ledger)
                # see golf results at all. None stays tolerated downstream.
                "scheduledDate": event_date_key,
                "latestDate": event_date_key,
                "fullField": full_field,
                "fieldSize": int(len(top_preds)),
                "rankingFavorite": favourite,
                "rankingFavoriteWon": bool(favourite is not None and favourite == actual_winner),
                "modelVersion": MODEL_VERSION,
            }
        )
        scoring.append(
            {
                "tour": tour,
                "season": year,
                "top_pick": predicted_winner == actual_winner,
                "favourite_known": favourite is not None,
                "favourite_hit": favourite is not None and favourite == actual_winner,
            }
        )

    backtests.sort(
        key=lambda row: (
            -row["year"],
            TOUR_PRIORITY.get(row["tour"], 99),
            _sort_priority(row["tournament"]),
            row["tournament"],
        ),
    )
    return backtests, scoring


# ----------------------------------------------------------------- upcoming


_GOLF_NAME_STOPWORDS = {
    "the", "championship", "championships", "open", "classic", "invitational", "presented", "by", "tournament",
    "pga", "lpga", "tour", "at", "of", "and", "in", "golf", "international", "cup", "s",
}


def _canonical_golf_name(value: str) -> str:
    text = str(value or "").lower().replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _golf_tokens(value: str) -> set[str]:
    return {token for token in _canonical_golf_name(value).split() if token not in _GOLF_NAME_STOPWORDS and len(token) > 1}


def _fetch_espn_golf_calendar(tours: tuple[str, ...] = ("PGA", "LPGA")) -> dict[str, list[dict]]:
    """This season's schedule per tour from ESPN's scoreboard calendar (one request
    per tour; the BFF already reads the same endpoint). Never raises."""
    if os.getenv("GOLF_ESPN_CALENDAR", "1").strip().lower() in {"0", "false", "no", "off"}:
        return {}
    calendar: dict[str, list[dict]] = {}
    for tour in tours:
        try:
            response = requests.get(ESPN_GOLF_SCOREBOARD.format(tour=tour.lower()), timeout=20)
            response.raise_for_status()
            entries = []
            for league in response.json().get("leagues") or []:
                for item in league.get("calendar") or []:
                    start = _event_date_key(str(item.get("startDate") or "")[:10])
                    end = _event_date_key(str(item.get("endDate") or "")[:10]) or start
                    if item.get("label") and start:
                        entries.append({"name": str(item["label"]), "startKey": start, "endKey": end})
            calendar[tour] = entries
        except Exception as exc:  # pragma: no cover - network
            print(f"ESPN {tour} golf calendar unavailable: {exc}")
    return calendar


def _match_calendar(name: str, entries: list[dict]) -> dict | None:
    canonical = _canonical_golf_name(name)
    for entry in entries:
        if _canonical_golf_name(entry["name"]) == canonical:
            return entry
    ours = _golf_tokens(name)
    best, best_score = None, 0.0
    for entry in entries:
        theirs = _golf_tokens(entry["name"])
        if not ours or not theirs:
            continue
        score = len(ours & theirs) / len(ours | theirs)
        if score > best_score:
            best, best_score = entry, score
    return best if best_score >= 0.5 else None


def _projected_date(previous_start: int | None, today: datetime) -> int | None:
    if not previous_start:
        return None
    try:
        projected = datetime(today.year, (previous_start // 100) % 100, previous_start % 100)
    except ValueError:
        return None
    return int(projected.strftime("%Y%m%d"))


def _build_upcoming(df: pd.DataFrame, *, calendar: dict[str, list[dict]] | None = None, today: datetime | None = None) -> list[dict]:
    """Next events on each tour, shown with the field and probabilities of their
    previous edition (the model cannot be re-run on a field that is not public
    yet), dated from this season's ESPN calendar or, failing that, last year's
    date moved to this year. Boards that cannot be dated are dropped."""
    today = today or datetime.now(timezone.utc)
    today_key = int(today.strftime("%Y%m%d"))
    grace_key = int((today - timedelta(days=2)).strftime("%Y%m%d"))
    horizon_key = int((today + timedelta(days=UPCOMING_LOOKAHEAD_DAYS)).strftime("%Y%m%d"))
    calendar = calendar if calendar is not None else _fetch_espn_golf_calendar()

    candidates = []
    for (tour, tournament_name), group in df.groupby(["tour", "tournament_name"], sort=False):
        latest_year = group["season_year"].max()
        latest_event = (
            group[group["season_year"] == latest_year] if pd.notna(latest_year) else group
        ).sort_values("winner_probability", ascending=False).reset_index(drop=True)
        if latest_event.empty:
            continue
        previous_start = None
        if "event_start_date" in latest_event.columns and latest_event["event_start_date"].notna().any():
            previous_start = _event_date_key(latest_event["event_start_date"].dropna().iloc[0])

        season_calendar = calendar.get(str(tour).upper(), [])
        entry = _match_calendar(str(tournament_name), season_calendar)
        if entry is not None:
            start, end, date_source = int(entry["startKey"]), int(entry["endKey"]), "espn_calendar"
        elif season_calendar:
            continue  # not on this season's schedule (renamed, moved or dropped)
        else:
            start = _projected_date(previous_start, today)
            end = start
            date_source = "previous_edition_date"
            if start is not None:
                end = int((datetime.strptime(str(start), "%Y%m%d") + timedelta(days=3)).strftime("%Y%m%d"))
        if start is None:
            continue  # cannot be dated: never show an undated "upcoming" board
        if previous_start and start <= previous_start:
            continue  # the calendar entry IS the edition we already have: not upcoming
        if end < grace_key or start > horizon_key:
            continue
        candidates.append((start, end, date_source, str(tour), str(tournament_name), latest_event, latest_year))

    candidates.sort(key=lambda item: (item[0], TOUR_PRIORITY.get(item[3], 99), _sort_priority(item[4]), item[4]))
    upcoming: list[dict] = []
    per_tour_counts = {tour: 0 for tour in TOUR_PRIORITY}
    for start, end, date_source, tour, tournament_name, latest_event, latest_year in candidates:
        if per_tour_counts.get(tour, 0) >= UPCOMING_PER_TOUR.get(tour, 12):
            continue
        predictions = [
            {
                "rank": rank,
                "playerName": row.player_name,
                "winProbability": float(row.winner_probability * 100),
                "profile": _build_golf_profile(row),
                "radarMetrics": _build_golf_radar(row),
            }
            for rank, row in enumerate(latest_event.itertuples(index=False), start=1)
        ]
        course = latest_event.iloc[0]["course_name"] if pd.notna(latest_event.iloc[0]["course_name"]) else "TBD Course"
        state = latest_event.iloc[0]["course_state_code"] if pd.notna(latest_event.iloc[0]["course_state_code"]) else ""
        field_season = int(latest_year) if pd.notna(latest_year) else None
        upcoming.append(
            {
                "id": f"{tour.lower()}-{str(tournament_name).replace(' ', '-').lower()}",
                "name": f"{start // 10000} {tournament_name}",
                "original_name": str(tournament_name),
                "tour": str(tour),
                "course": f"{course}, {state}".strip(", "),
                "scheduledDate": int(start),
                "latestDate": int(end),
                "predictedWinner": predictions[0]["playerName"] if predictions else None,
                "predictions": predictions,
                "predictionSource": "previous_edition_model_run",
                "fieldBasis": "previous_edition",
                "fieldNote": (
                    f"Field and probabilities are from the {field_season} edition; this year's field is not modelled yet."
                    if field_season
                    else "Field and probabilities are from the previous edition; this year's field is not modelled yet."
                ),
                "fieldSeason": field_season,
                "dateSource": date_source,
                "modelVersion": MODEL_VERSION,
            }
        )
        per_tour_counts[tour] = per_tour_counts.get(tour, 0) + 1

    return upcoming


# ----------------------------------------------------------------- metrics


def _season_rows(scoring: list[dict]) -> list[dict]:
    if not scoring:
        return []
    frame = pd.DataFrame(scoring)
    rows = []
    for (tour, season), group in frame.groupby(["tour", "season"]):
        known = group[group["favourite_known"]]
        rows.append(
            {
                "tour": tour,
                "season": int(season),
                "events": int(len(group)),
                "topPickHits": int(group["top_pick"].sum()),
                "topPickRate": round(float(group["top_pick"].mean()), 4),
                "owgrFavoriteEvents": int(len(known)),
                "owgrFavoriteHits": int(known["favourite_hit"].sum()),
                "owgrFavoriteRate": round(float(known["favourite_hit"].mean()), 4) if len(known) else None,
            }
        )
    return rows


def _build_metrics(calibration: dict, scoring: list[dict]) -> dict:
    report = calibration.get("report", {})
    return {
        "sport": "golf",
        "modelVersion": MODEL_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "heldOutEvents": report,
        "calibration": {
            "method": (
                "Per event, logit(p) * slope + shift. Top-10 probabilities sum to the expected number of "
                "top-10 finishers (ties included, from earlier events, held to 9.5-11); made-cut probabilities "
                "average the previous edition's cut rate. Slopes are fitted only on events that finished earlier. "
                "P(win) <= P(top 10) <= P(made cut) on every row."
            ),
            "source": calibration.get("source"),
            "events": (calibration.get("log") or {}).get("events", [])[-10:],
        },
        "bySeason": _season_rows(scoring),
        "notes": [
            "Held-out events are the model's chronological validation split; its training epoch was chosen on that split.",
            "LPGA top-10 and made-cut probabilities are hidden: the LPGA tables carry no top-10 or cut labels.",
            "Upcoming boards show the previous edition's field (fieldBasis = previous_edition).",
        ],
    }


def main() -> None:
    tournament_meta, player_outcomes, winners, labels = _load_meta()
    predictions, calibration = _load_predictions(tournament_meta, player_outcomes, labels)

    backtests, scoring = _build_backtests(predictions, winners)
    upcoming = _build_upcoming(predictions)
    metrics = _build_metrics(calibration, scoring)

    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (FRONTEND_DATA_DIR / "historical_backtests.json").write_text(json.dumps(backtests, indent=2))
    (FRONTEND_DATA_DIR / "upcoming_tournaments.json").write_text(json.dumps(upcoming, indent=2))
    (FRONTEND_DATA_DIR / "golf_model_metrics.json").write_text(json.dumps(metrics, indent=2))

    backtest_tours = pd.Series([row["tour"] for row in backtests]).value_counts().to_dict()
    upcoming_tours = pd.Series([row["tour"] for row in upcoming]).value_counts().to_dict()
    print(f"Generated {len(backtests)} golf backtests by tour: {backtest_tours}")
    print(f"Generated {len(upcoming)} golf upcoming events by tour: {upcoming_tours}")
    report = calibration.get("report") or {}
    if report.get("events"):
        top10, cut = report["top10"], report["madeCut"]
        print(
            f"Calibrated on {report['events']} held-out PGA events: top-10 Brier {top10['brier']} "
            f"(raw {top10['brierRaw']}, constant {top10['brierConstant']}), sums {top10['sumPerEventMin']}-{top10['sumPerEventMax']}; "
            f"made-cut Brier {cut['brier']} (raw {cut['brierRaw']}, constant {cut['brierConstant']}), "
            f"mean {cut['meanPredicted']} vs actual {cut['actualRate']}; rows with P(cut) < P(top 10): {report['rowsMadeCutBelowTop10']}"
        )
    for row in metrics["bySeason"]:
        print(f"  {row['tour']} {row['season']}: top pick {row['topPickHits']}/{row['events']} vs OWGR favourite {row['owgrFavoriteHits']}/{row['owgrFavoriteEvents']}")


if __name__ == "__main__":
    main()
