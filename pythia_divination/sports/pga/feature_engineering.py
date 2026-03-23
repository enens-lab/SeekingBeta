"""Golf-specific feature engineering for PGA tournament prediction datasets."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
_PLAYER_HISTORY_WINDOWS: tuple[int, ...] = (3, 5, 10)
_COURSE_HISTORY_WINDOWS: tuple[int, ...] = (2, 4)
_NUMERIC_RESULT_COLUMNS = (
    "won",
    "top_5",
    "top_10",
    "made_cut",
    "withdrawn",
    "position_numeric",
    "total_score_sort",
    "total_strokes",
)
_SHORT_MONTH_TO_NUMBER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _NON_ALNUM_PATTERN.sub("_", text).strip("_")
    return text or "unknown"


def build_course_key(frame: pd.DataFrame) -> pd.Series:
    course_id = frame.get("course_id", pd.Series(index=frame.index, dtype=object)).fillna("")
    course_name = frame.get("course_name", pd.Series(index=frame.index, dtype=object)).fillna("")
    if "course_name_live" in frame.columns:
        course_name = course_name.where(course_name.astype(str).str.len() > 0, frame["course_name_live"].fillna(""))
    course_city = frame.get("course_city", pd.Series(index=frame.index, dtype=object)).fillna("")
    course_state = frame.get("course_state_code", pd.Series(index=frame.index, dtype=object)).fillna("")
    course_country = frame.get("course_country", pd.Series(index=frame.index, dtype=object)).fillna("")
    key = (
        course_id.map(_slugify)
        + "__"
        + course_name.map(_slugify)
        + "__"
        + course_city.map(_slugify)
        + "__"
        + course_state.map(_slugify)
        + "__"
        + course_country.map(_slugify)
    )
    return key


def add_schedule_identifiers(schedule_df: pd.DataFrame) -> pd.DataFrame:
    frame = schedule_df.copy()
    frame["course_key"] = build_course_key(frame)
    frame["event_sort_key"] = (
        pd.to_numeric(frame["season"], errors="coerce").fillna(0).astype(int) * 1000
        + pd.to_numeric(frame["season_event_index"], errors="coerce").fillna(0).astype(int)
    )
    return frame


def add_round_and_result_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    round_columns = [
        column for column in ("round_1_score", "round_2_score", "round_3_score", "round_4_score")
        if column in enriched.columns
    ]
    if round_columns:
        round_scores = enriched[round_columns].apply(pd.to_numeric, errors="coerce")
        enriched["rounds_completed"] = round_scores.notna().sum(axis=1)
    else:
        enriched["rounds_completed"] = 0

    enriched["total_strokes"] = pd.to_numeric(enriched.get("total_strokes"), errors="coerce")
    enriched["total_score_sort"] = pd.to_numeric(enriched.get("total_score_sort"), errors="coerce")
    enriched["position_numeric"] = pd.to_numeric(enriched.get("position_numeric"), errors="coerce")

    rounds = enriched["rounds_completed"].replace(0, np.nan)
    enriched["strokes_per_round"] = enriched["total_strokes"] / rounds
    enriched["score_to_par"] = enriched["total_score_sort"]
    enriched["made_cut"] = enriched.get("made_cut", False).fillna(False).astype(bool)
    enriched["top_10"] = enriched.get("top_10", False).fillna(False).astype(bool)
    enriched["top_5"] = enriched.get("top_5", False).fillna(False).astype(bool)
    enriched["won"] = enriched.get("won", False).fillna(False).astype(bool)
    enriched["withdrawn"] = enriched.get("withdrawn", False).fillna(False).astype(bool)
    return enriched


def add_event_level_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    event_stats = (
        enriched.groupby("tournament_id")
        .agg(
            event_field_size=("player_id", "nunique"),
            event_cut_rate=("made_cut", "mean"),
            event_top10_rate=("top_10", "mean"),
            event_win_rate=("won", "mean"),
            event_avg_position=("position_numeric", "mean"),
            event_avg_score_to_par=("score_to_par", "mean"),
            event_avg_total_strokes=("total_strokes", "mean"),
            event_avg_strokes_per_round=("strokes_per_round", "mean"),
            event_best_score_to_par=("score_to_par", "min"),
            event_score_vs_field_std=("score_to_par", "std"),
            event_spr_vs_field_std=("strokes_per_round", "std"),
        )
        .reset_index()
    )
    enriched = enriched.merge(event_stats, on="tournament_id", how="left")
    enriched["score_to_par_vs_field"] = enriched["score_to_par"] - enriched["event_avg_score_to_par"]
    enriched["strokes_per_round_vs_field"] = (
        enriched["strokes_per_round"] - enriched["event_avg_strokes_per_round"]
    )
    enriched["position_pct_in_field"] = (
        enriched["position_numeric"] / enriched["event_field_size"].replace(0, np.nan)
    )
    return enriched


def _rolling_shift_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def _rolling_shift_min(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).min()


def _rolling_shift_max(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).max()


def _rolling_shift_std(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=2).std()


def _expanding_shift_std(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=2).std()


def _parse_display_start_date(display_date: Any, season_year: Any) -> pd.Timestamp | pd.NaT:
    text = str(display_date or "").strip()
    if not text:
        return pd.NaT
    match = re.search(r"([A-Za-z]{3,9})\s+(\d{1,2})", text)
    if not match:
        return pd.NaT
    month_number = _SHORT_MONTH_TO_NUMBER.get(match.group(1)[:3].lower())
    if month_number is None:
        return pd.NaT
    try:
        year = int(float(season_year))
        day = int(match.group(2))
    except (TypeError, ValueError):
        return pd.NaT
    return pd.to_datetime(f"{year:04d}-{month_number:02d}-{day:02d}", errors="coerce")


def _conditional_prior_mean(
    frame: pd.DataFrame,
    *,
    group_column: str,
    metric: str,
    mask: pd.Series,
) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    grouped_indices = frame.groupby(group_column, sort=False).groups
    for _, index in grouped_indices.items():
        subset = frame.loc[index]
        metric_values = pd.to_numeric(subset[metric], errors="coerce")
        mask_values = pd.to_numeric(mask.loc[index], errors="coerce").fillna(0.0)
        weighted = (metric_values * mask_values).shift(1).expanding().sum()
        counts = mask_values.shift(1).expanding().sum()
        result.loc[index] = weighted / counts.replace(0, np.nan)
    return result


def _prior_binary_streak(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype(int).tolist()
    result: list[int] = []
    streak = 0
    for value in values:
        result.append(streak)
        if value == 1:
            streak += 1
        else:
            streak = 0
    return pd.Series(result, index=series.index, dtype=float)


def _exp_weighted_prior_mean(values: pd.Series, dates: pd.Series, *, half_life_days: float) -> pd.Series:
    results: list[float] = []
    history_values: list[float] = []
    history_dates: list[pd.Timestamp] = []
    for current_value, current_date in zip(values.tolist(), dates.tolist()):
        current_ts = pd.to_datetime(current_date, errors="coerce")
        if not history_values or pd.isna(current_ts):
            results.append(np.nan)
        else:
            valid_pairs = [
                (float(value), pd.to_datetime(date_value, errors="coerce"))
                for value, date_value in zip(history_values, history_dates)
                if pd.notna(value) and pd.notna(date_value)
            ]
            if not valid_pairs:
                results.append(np.nan)
            else:
                deltas = np.asarray(
                    [max((current_ts - date_value).days, 0) for _, date_value in valid_pairs],
                    dtype=float,
                )
                weights = np.exp(-np.log(2.0) * deltas / float(half_life_days))
                metric_values = np.asarray([value for value, _ in valid_pairs], dtype=float)
                results.append(float(np.average(metric_values, weights=weights)) if weights.sum() > 0 else np.nan)
        history_values.append(pd.to_numeric(current_value, errors="coerce"))
        history_dates.append(current_ts)
    return pd.Series(results, index=values.index, dtype=float)


def _prior_event_count_in_window(dates: pd.Series, *, window_days: int) -> pd.Series:
    timestamps = pd.to_datetime(dates, errors="coerce").tolist()
    results: list[float] = []
    history: list[pd.Timestamp] = []
    for current_date in timestamps:
        if pd.isna(current_date):
            results.append(np.nan)
        else:
            count = sum(
                1
                for previous_date in history
                if pd.notna(previous_date) and 0 <= (current_date - previous_date).days <= window_days
            )
            results.append(float(count))
        history.append(current_date)
    return pd.Series(results, index=dates.index, dtype=float)


def add_player_recent_form_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy().sort_values(["player_id", "event_sort_key", "tournament_id"]).reset_index(drop=True)
    grouped = enriched.groupby("player_id", group_keys=False)

    enriched["player_events_played_before"] = grouped.cumcount()
    enriched["player_prev_position"] = grouped["position_numeric"].shift(1)
    enriched["player_prev_score_to_par_vs_field"] = grouped["score_to_par_vs_field"].shift(1)
    enriched["player_prev_strokes_per_round_vs_field"] = grouped["strokes_per_round_vs_field"].shift(1)
    enriched["player_prev_made_cut"] = (
        pd.to_numeric(grouped["made_cut"].shift(1), errors="coerce").fillna(0).astype(int)
    )
    enriched["player_prev_top_10"] = (
        pd.to_numeric(grouped["top_10"].shift(1), errors="coerce").fillna(0).astype(int)
    )
    enriched["player_prev_win"] = pd.to_numeric(grouped["won"].shift(1), errors="coerce").fillna(0).astype(int)

    cumulative_sources = {
        "won": "player_career_win_rate",
        "top_10": "player_career_top10_rate",
        "made_cut": "player_career_made_cut_rate",
        "score_to_par_vs_field": "player_career_score_vs_field_mean",
        "strokes_per_round_vs_field": "player_career_spr_vs_field_mean",
        "position_numeric": "player_career_finish_mean",
    }
    for source, target in cumulative_sources.items():
        series = pd.to_numeric(enriched[source], errors="coerce")
        cumulative_sum = grouped[source].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).expanding().mean())
        enriched[target] = cumulative_sum

    window_metrics = (
        "won",
        "top_5",
        "top_10",
        "made_cut",
        "position_numeric",
        "score_to_par_vs_field",
        "strokes_per_round_vs_field",
        "event_field_size",
        "event_cut_rate",
        "event_avg_score_to_par",
    )
    for metric in window_metrics:
        numeric = pd.to_numeric(enriched[metric], errors="coerce")
        for window in _PLAYER_HISTORY_WINDOWS:
            enriched[f"player_{metric}_mean_last_{window}"] = grouped[metric].transform(
                lambda s, w=window: _rolling_shift_mean(pd.to_numeric(s, errors="coerce"), w)
            )
        if metric == "position_numeric":
            for window in _PLAYER_HISTORY_WINDOWS:
                enriched[f"player_{metric}_best_last_{window}"] = grouped[metric].transform(
                    lambda s, w=window: _rolling_shift_min(pd.to_numeric(s, errors="coerce"), w)
                )

    enriched["player_finish_momentum_last_3_vs_10"] = (
        enriched["player_position_numeric_mean_last_3"]
        - enriched["player_position_numeric_mean_last_10"]
    )
    enriched["player_top10_momentum_last_3_vs_10"] = (
        enriched["player_top_10_mean_last_3"]
        - enriched["player_top_10_mean_last_10"]
    )
    enriched["player_cut_momentum_last_3_vs_10"] = (
        enriched["player_made_cut_mean_last_3"]
        - enriched["player_made_cut_mean_last_10"]
    )
    return enriched


def add_player_variability_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy().sort_values(["player_id", "event_sort_key", "tournament_id"]).reset_index(drop=True)
    grouped = enriched.groupby("player_id", group_keys=False)
    for metric in ("position_numeric", "score_to_par_vs_field", "strokes_per_round_vs_field", "made_cut", "top_10"):
        for window in (5, 10):
            enriched[f"player_{metric}_std_last_{window}"] = grouped[metric].transform(
                lambda s, w=window: _rolling_shift_std(pd.to_numeric(s, errors="coerce"), w)
            )
    enriched["player_finish_range_last_5"] = (
        grouped["position_numeric"].transform(lambda s: _rolling_shift_max(pd.to_numeric(s, errors="coerce"), 5))
        - grouped["position_numeric"].transform(lambda s: _rolling_shift_min(pd.to_numeric(s, errors="coerce"), 5))
    )
    enriched["player_finish_range_last_10"] = (
        grouped["position_numeric"].transform(lambda s: _rolling_shift_max(pd.to_numeric(s, errors="coerce"), 10))
        - grouped["position_numeric"].transform(lambda s: _rolling_shift_min(pd.to_numeric(s, errors="coerce"), 10))
    )
    return enriched


def add_player_season_form_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy().sort_values(["player_id", "season_year", "event_sort_key", "tournament_id"]).reset_index(drop=True)
    grouped = enriched.groupby(["player_id", "season_year"], group_keys=False)
    enriched["player_season_events_before"] = grouped.cumcount()
    for metric in ("won", "top_10", "made_cut", "position_numeric", "score_to_par_vs_field", "strokes_per_round_vs_field"):
        enriched[f"player_season_{metric}_prior_mean"] = grouped[metric].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).expanding().mean()
        )
    if "season_event_index" in enriched.columns:
        season_event_index = pd.to_numeric(enriched["season_event_index"], errors="coerce")
        enriched["player_season_share_of_schedule_before"] = (
            enriched["player_season_events_before"] / season_event_index.replace(0, np.nan)
        )
    return enriched


def add_player_timing_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy().sort_values(["player_id", "event_sort_key", "tournament_id"]).reset_index(drop=True)
    if "event_start_date" not in enriched.columns:
        return enriched
    grouped = enriched.groupby("player_id", group_keys=False)
    for metric in ("score_to_par_vs_field", "strokes_per_round_vs_field", "top_10", "made_cut", "won"):
        if metric not in enriched.columns:
            continue
        enriched[f"player_timing_{metric}_ewm_90d"] = grouped.apply(
            lambda group, m=metric: _exp_weighted_prior_mean(
                pd.to_numeric(group[m], errors="coerce"),
                group["event_start_date"],
                half_life_days=90.0,
            )
        ).reset_index(level=0, drop=True)
        enriched[f"player_timing_{metric}_ewm_180d"] = grouped.apply(
            lambda group, m=metric: _exp_weighted_prior_mean(
                pd.to_numeric(group[m], errors="coerce"),
                group["event_start_date"],
                half_life_days=180.0,
            )
        ).reset_index(level=0, drop=True)
    enriched["player_timing_events_last_90d"] = grouped["event_start_date"].apply(
        lambda dates: _prior_event_count_in_window(dates, window_days=90)
    ).reset_index(level=0, drop=True)
    enriched["player_timing_events_last_180d"] = grouped["event_start_date"].apply(
        lambda dates: _prior_event_count_in_window(dates, window_days=180)
    ).reset_index(level=0, drop=True)
    if "player_score_to_par_vs_field_mean_last_5" in enriched.columns and "player_timing_score_to_par_vs_field_ewm_90d" in enriched.columns:
        enriched["player_timing_score_vs_field_ewm90_vs_last5"] = (
            pd.to_numeric(enriched["player_timing_score_to_par_vs_field_ewm_90d"], errors="coerce")
            - pd.to_numeric(enriched["player_score_to_par_vs_field_mean_last_5"], errors="coerce")
        )
    if "player_top_10_mean_last_5" in enriched.columns and "player_timing_top_10_ewm_90d" in enriched.columns:
        enriched["player_timing_top10_ewm90_vs_last5"] = (
            pd.to_numeric(enriched["player_timing_top_10_ewm_90d"], errors="coerce")
            - pd.to_numeric(enriched["player_top_10_mean_last_5"], errors="coerce")
        )
    return enriched


def add_course_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy().sort_values(
        ["player_id", "course_key", "event_sort_key", "tournament_id"]
    ).reset_index(drop=True)
    grouped = enriched.groupby(["player_id", "course_key"], group_keys=False)

    enriched["course_history_events_before"] = grouped.cumcount()
    metrics = (
        "won",
        "top_10",
        "made_cut",
        "position_numeric",
        "score_to_par_vs_field",
        "strokes_per_round_vs_field",
    )
    for metric in metrics:
        for window in _COURSE_HISTORY_WINDOWS:
            enriched[f"course_history_{metric}_mean_last_{window}"] = grouped[metric].transform(
                lambda s, w=window: _rolling_shift_mean(pd.to_numeric(s, errors="coerce"), w)
            )

    enriched["course_history_finish_delta_vs_recent_5"] = (
        enriched["course_history_position_numeric_mean_last_4"]
        - enriched["player_position_numeric_mean_last_5"]
    )
    enriched["course_history_score_delta_vs_recent_5"] = (
        enriched["course_history_score_to_par_vs_field_mean_last_4"]
        - enriched["player_score_to_par_vs_field_mean_last_5"]
    )
    enriched["course_history_top10_delta_vs_recent_5"] = (
        enriched["course_history_top_10_mean_last_4"]
        - enriched["player_top_10_mean_last_5"]
    )
    enriched["course_history_cut_delta_vs_recent_5"] = (
        enriched["course_history_made_cut_mean_last_4"]
        - enriched["player_made_cut_mean_last_5"]
    )
    return enriched


def add_course_profile_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    course_events = (
        enriched[
            [
                "tournament_id",
                "course_key",
                "event_sort_key",
                "event_field_size",
                "event_cut_rate",
                "event_avg_score_to_par",
                "event_avg_strokes_per_round",
                "event_best_score_to_par",
                "event_score_vs_field_std",
                "event_spr_vs_field_std",
            ]
        ]
        .drop_duplicates()
        .sort_values(["course_key", "event_sort_key", "tournament_id"])
        .reset_index(drop=True)
    )
    grouped = course_events.groupby("course_key", group_keys=False)
    course_events["course_profile_prior_events"] = grouped.cumcount()
    for metric in (
        "event_field_size",
        "event_cut_rate",
        "event_avg_score_to_par",
        "event_avg_strokes_per_round",
        "event_best_score_to_par",
        "event_score_vs_field_std",
        "event_spr_vs_field_std",
    ):
        course_events[f"course_profile_{metric}_mean_prior"] = grouped[metric].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).expanding().mean()
        )
        course_events[f"course_profile_{metric}_std_prior"] = grouped[metric].transform(
            lambda s: _expanding_shift_std(pd.to_numeric(s, errors="coerce"))
        )
    merge_columns = [column for column in course_events.columns if column.startswith("course_profile_")]
    enriched = enriched.merge(
        course_events[["tournament_id"] + merge_columns],
        on="tournament_id",
        how="left",
    )
    return enriched


def add_field_recent_form_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    aggregations = {
        "player_career_win_rate": "mean",
        "player_career_top10_rate": "mean",
        "player_career_made_cut_rate": "mean",
        "player_position_numeric_mean_last_5": "mean",
        "player_score_to_par_vs_field_mean_last_5": "mean",
    }
    available = {k: v for k, v in aggregations.items() if k in enriched.columns}
    if not available:
        return enriched
    grouped = enriched.groupby("tournament_id").agg(available).reset_index()
    grouped = grouped.rename(columns={column: f"field_{column}" for column in grouped.columns if column != "tournament_id"})
    enriched = enriched.merge(grouped, on="tournament_id", how="left")

    if "player_career_top10_rate" in enriched.columns:
        enriched["player_career_top10_pct_in_field"] = enriched.groupby("tournament_id")[
            "player_career_top10_rate"
        ].rank(pct=True, ascending=False)
    if "player_position_numeric_mean_last_5" in enriched.columns:
        enriched["player_recent_finish_pct_in_field"] = enriched.groupby("tournament_id")[
            "player_position_numeric_mean_last_5"
        ].rank(pct=True, ascending=True)
    return enriched


def add_player_context_form_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy().sort_values(["player_id", "event_sort_key", "tournament_id"]).reset_index(drop=True)
    event_difficulty = pd.to_numeric(enriched.get("event_avg_score_to_par"), errors="coerce")
    difficulty_threshold = float(event_difficulty.median(skipna=True)) if event_difficulty.notna().any() else 0.0
    context_masks = {
        "signature": pd.to_numeric(enriched.get("is_signature_like_event"), errors="coerce").fillna(0),
        "us": pd.to_numeric(enriched.get("is_us_event"), errors="coerce").fillna(0),
        "hard_course": event_difficulty.ge(difficulty_threshold).astype(int),
        "easy_course": event_difficulty.lt(difficulty_threshold).astype(int),
    }
    for context_name, mask in context_masks.items():
        for metric in ("top_10", "made_cut", "won", "score_to_par_vs_field"):
            enriched[f"player_context_{context_name}_{metric}_prior_mean"] = _conditional_prior_mean(
                enriched,
                group_column="player_id",
                metric=metric,
                mask=mask,
            )
    return enriched


def add_course_fit_proxy_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    difficulty = pd.to_numeric(enriched.get("course_profile_event_avg_score_to_par_mean_prior"), errors="coerce")
    cut_toughness = 1.0 - pd.to_numeric(enriched.get("course_profile_event_cut_rate_mean_prior"), errors="coerce")
    if "snapshot_ball_striking_index" in enriched.columns:
        enriched["course_fit_ball_striking_difficulty"] = (
            pd.to_numeric(enriched["snapshot_ball_striking_index"], errors="coerce") * difficulty
        )
    if "snapshot_short_game_index" in enriched.columns:
        enriched["course_fit_short_game_toughness"] = (
            pd.to_numeric(enriched["snapshot_short_game_index"], errors="coerce") * cut_toughness
        )
    if "snapshot_scoring_index" in enriched.columns:
        enriched["course_fit_easy_scoring_proxy"] = (
            pd.to_numeric(enriched["snapshot_scoring_index"], errors="coerce") * (-difficulty)
        )
    if "player_context_hard_course_score_to_par_vs_field_prior_mean" in enriched.columns:
        enriched["course_fit_hard_course_history_proxy"] = (
            pd.to_numeric(enriched["player_context_hard_course_score_to_par_vs_field_prior_mean"], errors="coerce")
            * difficulty
        )
    if "player_context_easy_course_score_to_par_vs_field_prior_mean" in enriched.columns:
        enriched["course_fit_easy_course_history_proxy"] = (
            pd.to_numeric(enriched["player_context_easy_course_score_to_par_vs_field_prior_mean"], errors="coerce")
            * (-difficulty)
        )
    if "is_signature_like_event" in enriched.columns and "player_context_signature_top_10_prior_mean" in enriched.columns:
        enriched["course_fit_signature_pressure_proxy"] = (
            pd.to_numeric(enriched["is_signature_like_event"], errors="coerce")
            * pd.to_numeric(enriched["player_context_signature_top_10_prior_mean"], errors="coerce")
        )
    if "is_us_event" in enriched.columns and "player_context_us_top_10_prior_mean" in enriched.columns:
        enriched["course_fit_us_event_proxy"] = (
            pd.to_numeric(enriched["is_us_event"], errors="coerce")
            * pd.to_numeric(enriched["player_context_us_top_10_prior_mean"], errors="coerce")
        )
    if "player_score_to_par_vs_field_std_last_10" in enriched.columns:
        enriched["course_fit_volatility_on_difficult_courses"] = (
            pd.to_numeric(enriched["player_score_to_par_vs_field_std_last_10"], errors="coerce")
            * difficulty
        )
    if "player_top_10_mean_last_10" in enriched.columns:
        enriched["course_fit_recent_upside_on_easy_courses"] = (
            pd.to_numeric(enriched["player_top_10_mean_last_10"], errors="coerce")
            * (-difficulty)
        )
    if "course_profile_event_score_vs_field_std_mean_prior" in enriched.columns and "snapshot_ball_striking_index" in enriched.columns:
        enriched["course_fit_ball_striking_on_volatile_courses"] = (
            pd.to_numeric(enriched["course_profile_event_score_vs_field_std_mean_prior"], errors="coerce")
            * pd.to_numeric(enriched["snapshot_ball_striking_index"], errors="coerce")
        )
    if "driving_distance__avg" in enriched.columns:
        # Distance advantage often scales with course difficulty/length
        enriched["course_fit_distance_difficulty_proxy"] = (
            pd.to_numeric(enriched["driving_distance__avg"], errors="coerce") * difficulty
        )
    if "driving_accuracy_pct__value" in enriched.columns:
        # Accuracy advantage often scales with course difficulty/volatility
        volatility = pd.to_numeric(enriched.get("course_profile_event_score_vs_field_std_mean_prior"), errors="coerce")
        if volatility is not None:
            enriched["course_fit_accuracy_volatility_proxy"] = (
                pd.to_numeric(enriched["driving_accuracy_pct__value"], errors="coerce") * volatility
            )
    return enriched


def add_field_relative_skill_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    metric_directions = {
        "sg_total__avg": "higher",
        "snapshot_ball_striking_index": "higher",
        "snapshot_short_game_index": "higher",
        "snapshot_scoring_index": "higher",
        "driving_distance__avg": "higher",
        "driving_accuracy_pct__value": "higher",
        "scoring_average__avg": "lower",
        "owgr__rank": "lower",
    }
    for metric, direction in metric_directions.items():
        if metric not in enriched.columns:
            continue
        numeric = pd.to_numeric(enriched[metric], errors="coerce")
        field_mean = numeric.groupby(enriched["tournament_id"]).transform("mean")
        field_std = numeric.groupby(enriched["tournament_id"]).transform("std")
        diff = numeric - field_mean if direction == "higher" else field_mean - numeric
        enriched[f"field_relative_{metric}_advantage"] = diff
        enriched[f"field_relative_{metric}_zscore"] = diff / field_std.replace(0, np.nan)
    return enriched


def add_schedule_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "display_date" in enriched.columns and "season_year" in enriched.columns:
        enriched["event_start_date"] = [
            _parse_display_start_date(display_date, season_year)
            for display_date, season_year in zip(enriched["display_date"], enriched["season_year"])
        ]
        event_start = pd.to_datetime(enriched["event_start_date"], errors="coerce")
        enriched["event_day_of_year"] = event_start.dt.dayofyear
        enriched["event_week_of_year"] = event_start.dt.isocalendar().week.astype(float)
        enriched["event_day_of_year_sin"] = np.sin(2 * np.pi * enriched["event_day_of_year"] / 366.0)
        enriched["event_day_of_year_cos"] = np.cos(2 * np.pi * enriched["event_day_of_year"] / 366.0)
    if "month_number" in enriched.columns:
        month = pd.to_numeric(enriched["month_number"], errors="coerce")
        enriched["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        enriched["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    numeric_columns = (
        "purse_value",
        "champion_earnings_value",
        "fedex_points_value",
        "season_event_index",
        "season_event_total",
        "season_progress_pct",
        "purse_rank_pct_in_season",
        "fedex_points_rank_pct_in_season",
        "is_us_event",
        "is_signature_like_event",
    )
    for column in numeric_columns:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    if "champion_earnings_value" in enriched.columns and "purse_value" in enriched.columns:
        enriched["champion_earnings_share"] = (
            enriched["champion_earnings_value"] / enriched["purse_value"].replace(0, np.nan)
        )
    if "season_event_index" in enriched.columns and "season_event_total" in enriched.columns:
        enriched["season_remaining_events_estimate"] = (
            enriched["season_event_total"] - enriched["season_event_index"]
        )
    if "event_start_date" in enriched.columns:
        ordered = enriched.sort_values(["player_id", "event_sort_key", "tournament_id"]).reset_index(drop=True)
        event_start = pd.to_datetime(ordered["event_start_date"], errors="coerce")
        days_since = event_start.groupby(ordered["player_id"]).diff().dt.days
        ordered["player_days_since_last_event"] = days_since
        ordered["player_weeks_since_last_event"] = days_since / 7.0
        enriched = ordered
    return enriched


def add_player_profile_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    event_start = pd.to_datetime(
        enriched.get("event_start_date", pd.Series(index=enriched.index, dtype=object)),
        errors="coerce",
    )
    born_date = pd.to_datetime(
        enriched.get("profile_born_date", pd.Series(index=enriched.index, dtype=object)),
        errors="coerce",
    )
    current_age = pd.to_numeric(
        enriched.get("profile_age_current", pd.Series(index=enriched.index, dtype=float)),
        errors="coerce",
    )
    turned_pro_year = pd.to_numeric(
        enriched.get("profile_turned_pro_year", pd.Series(index=enriched.index, dtype=float)),
        errors="coerce",
    )
    season_year = pd.to_numeric(enriched.get("season_year", pd.Series(index=enriched.index, dtype=float)), errors="coerce")
    height_inches = pd.to_numeric(
        enriched.get("profile_height_inches", pd.Series(index=enriched.index, dtype=float)),
        errors="coerce",
    )

    age_at_event = (event_start - born_date).dt.days / 365.25
    enriched["profile_age_at_event"] = age_at_event.where(age_at_event.notna(), current_age)
    enriched["profile_age_squared"] = enriched["profile_age_at_event"] ** 2
    enriched["profile_born_month"] = born_date.dt.month
    enriched["profile_born_month_sin"] = np.sin(2 * np.pi * enriched["profile_born_month"] / 12.0)
    enriched["profile_born_month_cos"] = np.cos(2 * np.pi * enriched["profile_born_month"] / 12.0)
    enriched["profile_years_pro_at_event"] = season_year - turned_pro_year
    enriched["profile_years_pro_squared"] = enriched["profile_years_pro_at_event"] ** 2
    enriched["profile_age_years_pro_interaction"] = (
        enriched["profile_age_at_event"] * enriched["profile_years_pro_at_event"]
    )
    enriched["profile_height_inches"] = height_inches
    enriched["profile_height_zscore_proxy"] = (
        (height_inches - height_inches.median(skipna=True))
        / height_inches.std(skipna=True)
        if height_inches.notna().sum() > 1 and float(height_inches.std(skipna=True) or 0) > 0
        else np.nan
    )
    enriched["profile_college_known"] = (
        enriched.get("profile_college", pd.Series(index=enriched.index, dtype=object))
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
        .astype(int)
    )
    enriched["profile_birthplace_known"] = (
        enriched.get("profile_birthplace", pd.Series(index=enriched.index, dtype=object))
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
        .astype(int)
    )
    enriched["profile_age_under_25"] = enriched["profile_age_at_event"].lt(25).astype(int)
    enriched["profile_age_25_to_30"] = enriched["profile_age_at_event"].between(25, 30, inclusive="both").astype(int)
    enriched["profile_age_31_to_35"] = enriched["profile_age_at_event"].between(31, 35, inclusive="both").astype(int)
    enriched["profile_age_over_35"] = enriched["profile_age_at_event"].gt(35).astype(int)
    enriched["profile_years_pro_under_3"] = enriched["profile_years_pro_at_event"].lt(3).astype(int)
    enriched["profile_years_pro_3_to_8"] = enriched["profile_years_pro_at_event"].between(3, 8, inclusive="both").astype(int)
    enriched["profile_years_pro_over_8"] = enriched["profile_years_pro_at_event"].gt(8).astype(int)
    return enriched


def add_player_availability_proxy_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy().sort_values(["player_id", "event_sort_key", "tournament_id"]).reset_index(drop=True)
    grouped = enriched.groupby("player_id", group_keys=False)

    withdrawn_numeric = pd.to_numeric(enriched.get("withdrawn"), errors="coerce").fillna(0)
    made_cut_numeric = pd.to_numeric(enriched.get("made_cut"), errors="coerce").fillna(0)
    rounds_completed = pd.to_numeric(enriched.get("rounds_completed"), errors="coerce")
    short_event_numeric = rounds_completed.lt(4).fillna(False).astype(int)
    missed_cut_numeric = (1 - made_cut_numeric).clip(lower=0, upper=1)

    for window in (3, 5, 10, 20):
        enriched[f"player_availability_withdrawal_rate_last_{window}"] = grouped["withdrawn"].transform(
            lambda s, w=window: _rolling_shift_mean(pd.to_numeric(s, errors="coerce"), w)
        )
        enriched[f"player_availability_short_event_rate_last_{window}"] = grouped["rounds_completed"].transform(
            lambda s, w=window: _rolling_shift_mean(pd.to_numeric(s, errors="coerce").lt(4).astype(int), w)
        )
        enriched[f"player_availability_missed_cut_rate_last_{window}"] = grouped["made_cut"].transform(
            lambda s, w=window: _rolling_shift_mean(1 - pd.to_numeric(s, errors="coerce").fillna(0), w)
        )

    enriched["player_availability_prior_withdrawal_streak"] = grouped["withdrawn"].transform(_prior_binary_streak).fillna(0).astype(int)
    enriched["player_availability_prior_missed_cut_streak"] = grouped["made_cut"].transform(
        lambda s: _prior_binary_streak(1 - pd.to_numeric(s, errors="coerce").fillna(0))
    ).fillna(0).astype(int)
    enriched["player_availability_prev_short_event"] = short_event_numeric.groupby(enriched["player_id"]).shift(1).fillna(0).astype(int)
    enriched["player_availability_prev_withdrawal"] = withdrawn_numeric.groupby(enriched["player_id"]).shift(1).fillna(0).astype(int)
    enriched["player_availability_prev_missed_cut"] = missed_cut_numeric.groupby(enriched["player_id"]).shift(1).fillna(0).astype(int)
    days_since = pd.to_numeric(enriched.get("player_days_since_last_event"), errors="coerce")
    enriched["player_availability_long_layoff_28d"] = days_since.ge(28).astype(int)
    enriched["player_availability_long_layoff_56d"] = days_since.ge(56).astype(int)
    enriched["player_availability_long_layoff_84d"] = days_since.ge(84).astype(int)
    enriched["player_availability_returning_from_withdrawal"] = (
        enriched["player_availability_prev_withdrawal"] * enriched["player_availability_long_layoff_28d"]
    )
    enriched["player_availability_layoff_withdrawal_risk"] = (
        pd.to_numeric(enriched["player_availability_withdrawal_rate_last_10"], errors="coerce")
        * days_since
    )
    return enriched


def _mean_available(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    numeric = frame[available].apply(pd.to_numeric, errors="coerce")
    return numeric.mean(axis=1)


def add_snapshot_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["snapshot_ball_striking_index"] = _mean_available(
        enriched,
        (
            "sg_tee_to_green__avg",
            "sg_off_the_tee__avg",
            "sg_approach__avg",
            "good_drive_pct__value",
            "gir_pct__value",
        ),
    )
    enriched["snapshot_short_game_index"] = _mean_available(
        enriched,
        (
            "sg_around_green__avg",
            "sg_putting__avg",
            "scrambling_pct__value",
            "sand_save_pct__value",
            "one_putt_pct__value",
            "three_putt_avoidance__value",
        ),
    )
    enriched["snapshot_scoring_index"] = _mean_available(
        enriched,
        (
            "birdie_average__avg",
            "birdie_or_better_pct__value",
            "birdie_or_better_conversion_pct__value",
            "par_5_birdie_or_better__value",
            "total_birdies__total",
            "total_eagles__total",
        ),
    )
    if "driving_distance__avg" in enriched.columns and "driving_accuracy_pct__value" in enriched.columns:
        enriched["snapshot_distance_accuracy_blend"] = (
            pd.to_numeric(enriched["driving_distance__avg"], errors="coerce")
            * pd.to_numeric(enriched["driving_accuracy_pct__value"], errors="coerce")
            / 100.0
        )
    if "par_3_performance__par_3_avg" in enriched.columns and "par_5_performance__par_5_avg" in enriched.columns:
        enriched["snapshot_par_scoring_spread"] = (
            pd.to_numeric(enriched["par_5_performance__par_5_avg"], errors="coerce")
            - pd.to_numeric(enriched["par_3_performance__par_3_avg"], errors="coerce")
        )
    if "owgr__rank" in enriched.columns and "sg_total__avg" in enriched.columns:
        enriched["snapshot_world_rank_vs_sg_total"] = (
            pd.to_numeric(enriched["sg_total__avg"], errors="coerce")
            - pd.to_numeric(enriched["owgr__rank"], errors="coerce") / 100.0
        )
    return enriched


def add_hometown_advantage_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    
    if "profile_birthplace" in enriched.columns and "course_state_code" in enriched.columns:
        birthplace = enriched["profile_birthplace"].fillna("").astype(str).str.lower()
        course_state = enriched["course_state_code"].fillna("").astype(str).str.lower()
        
        state_match = [
            1 if s and (f", {s}" in b or b.endswith(f" {s}")) else 0
            for b, s in zip(birthplace, course_state)
        ]
        enriched["course_fit_home_state"] = state_match

    if "profile_country" in enriched.columns and "course_country" in enriched.columns:
        p_country = enriched["profile_country"].fillna("").astype(str).str.lower().replace("united states", "united states of america")
        c_country = enriched["course_country"].fillna("").astype(str).str.lower()
        
        country_match = [
            1 if p == c and c != "" else 0
            for p, c in zip(p_country, c_country)
        ]
        enriched["course_fit_home_country"] = country_match
        
    return enriched


def engineer_training_features(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    frame = add_schedule_context_features(frame)
    frame = add_player_profile_features(frame)
    frame = add_snapshot_derived_features(frame)
    frame = add_round_and_result_features(frame)
    frame = add_event_level_features(frame)
    frame = add_player_availability_proxy_features(frame)
    frame = add_player_recent_form_features(frame)
    frame = add_player_variability_features(frame)
    frame = add_player_season_form_features(frame)
    frame = add_player_timing_features(frame)
    frame = add_course_history_features(frame)
    frame = add_course_profile_features(frame)
    frame = add_player_context_form_features(frame)
    frame = add_field_recent_form_features(frame)
    frame = add_course_fit_proxy_features(frame)
    frame = add_hometown_advantage_features(frame)
    frame = add_field_relative_skill_features(frame)
    return frame


def neural_static_feature_columns(frame: pd.DataFrame) -> list[str]:
    allowed_prefixes = (
        "player_career_",
        "player_prev_",
        "player_events_played_before",
        "player_position_numeric_",
        "player_score_to_par_vs_field_",
        "player_strokes_per_round_vs_field_",
        "player_top_10_",
        "player_top_5_",
        "player_made_cut_",
        "player_won_",
        "player_finish_momentum_",
        "player_top10_momentum_",
        "player_cut_momentum_",
        "player_availability_",
        "profile_",
        "player_season_",
        "player_timing_",
        "player_context_",
        "course_history_",
        "course_profile_",
        "course_fit_",
        "field_player_",
        "field_relative_",
        "field_recent_",
    )
    safe_snapshot_columns = {
        "sg_total__avg",
        "sg_tee_to_green__avg",
        "sg_off_the_tee__avg",
        "sg_approach__avg",
        "sg_around_green__avg",
        "sg_putting__avg",
        "scoring_average__avg",
        "driving_distance__avg",
        "driving_accuracy_pct__value",
        "good_drive_pct__value",
        "gir_pct__value",
        "proximity_to_hole__avg",
        "scrambling_pct__value",
        "sand_save_pct__value",
        "putts_per_round__avg",
        "one_putt_pct__value",
        "three_putt_avoidance__value",
        "birdie_or_better_conversion_pct__value",
        "top_10_finishes__1st",
        "victory_leaders__victories",
        "owgr__rank",
        "all_around_ranking__rank",
        "par_3_performance__par_3_avg",
        "par_4_performance__par_4_avg",
        "par_5_performance__par_5_avg",
        "par_3_birdie_or_better__value",
        "par_4_birdie_or_better__value",
        "par_5_birdie_or_better__value",
        "birdie_average__avg",
        "birdie_or_better_pct__value",
        "putting_average__avg",
        "overall_putting_average__avg",
        "putting_inside_10ft__made",
        "putting_5_to_10ft__value",
        "putting_10_to_15ft__made",
        "putting_15_to_20ft__made",
        "putting_20_to_25ft__made",
        "fairway_proximity__avg",
        "arg_proximity__avg_dtp",
        "arg_proximity_10_20__avg_dtp",
        "arg_proximity_20_30__avg_dtp",
        "arg_proximity_under_10__avg_dtp",
        "arg_proximity_sand__avg_dtp",
        "arg_proximity_rough__avg_dtp",
        "scrambling_from_rough__value",
        "scrambling_from_sand__value",
        "scrambling_over_30__value",
        "scrambling_avg_distance__avg_dtp",
        "round_1_scoring_average__avg",
        "round_2_scoring_average__avg",
        "round_3_scoring_average__avg",
        "round_4_scoring_average__avg",
        "approach_125_150_fairway__avg",
        "approach_150_175_fairway__avg",
        "approach_175_200_fairway__avg",
        "birdie_better_125_150__value",
        "birdie_better_150_175__value",
        "birdie_better_175_200__value",
        "birdie_better_200_plus__value",
        "birdie_better_under_125__value",
        "total_eagles__total",
        "total_birdies__total",
        "par_4_eagle_leaders__total",
        "par_5_eagle_leaders__total",
        "driving_distance_all_drives__all_drives",
    }
    explicit_columns = {
        "month_number",
        "month_sin",
        "month_cos",
        "event_day_of_year",
        "event_week_of_year",
        "event_day_of_year_sin",
        "event_day_of_year_cos",
        "season_event_index",
        "season_event_total",
        "season_progress_pct",
        "purse_value",
        "champion_earnings_value",
        "fedex_points_value",
        "purse_rank_pct_in_season",
        "fedex_points_rank_pct_in_season",
        "champion_earnings_share",
        "season_remaining_events_estimate",
        "is_us_event",
        "is_signature_like_event",
        "snapshot_ball_striking_index",
        "snapshot_short_game_index",
        "snapshot_scoring_index",
        "snapshot_distance_accuracy_blend",
        "snapshot_par_scoring_spread",
        "snapshot_world_rank_vs_sg_total",
        "course_profile_prior_events",
        "owgr_rank_pct_in_field",
        "sg_total_pct_in_field",
        "scoring_avg_pct_in_field",
        "player_career_top10_pct_in_field",
        "player_recent_finish_pct_in_field",
        "player_days_since_last_event",
        "player_weeks_since_last_event",
    }
    result: list[str] = []
    for column in frame.columns:
        if (
            column in explicit_columns
            or column.startswith(allowed_prefixes)
            or column in safe_snapshot_columns
        ):
            if pd.api.types.is_numeric_dtype(frame[column]) or frame[column].dtype == bool:
                result.append(column)
    return sorted(set(result))


def neural_sequence_feature_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "position_numeric",
        "score_to_par_vs_field",
        "strokes_per_round_vs_field",
        "made_cut",
        "top_10",
        "top_5",
        "won",
        "event_field_size",
        "event_cut_rate",
        "event_avg_score_to_par",
        "course_profile_event_cut_rate_mean_prior",
        "course_profile_event_avg_score_to_par_mean_prior",
    ]
    return [column for column in candidates if column in frame.columns]
