"""Checks for sports/table_dtypes.coerce_table_dtypes.

Run (cwd = pythia_divination):  python tests/test_table_dtypes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sports.table_dtypes import coerce_table_dtypes, describe_dtypes  # noqa: E402


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": [2026001, "2026_002", None],                   # mixed -> str
        "season": ["2026", "2026", "2025"],                      # text ints -> int64
        "week": ["1", "2", None],                                # text with a null -> float64
        "official_date": ["2026-09-13", "2026-09-20T13:00:00Z", None],  # mixed formats -> datetime
        "away_probable_pitcher_id": pd.Series([None, None, None], dtype=object),  # all-null object -> float64
        "home_probable_pitcher_id": ["660271", 543037, np.nan],  # mixed numeric-like -> float64
        "gsis_id": ["00-0039150", "00-0032950", None],           # non-numeric ids stay as-is
        "player_id": pd.Series([660271, 543037, 111111], dtype="int64"),  # already numeric: untouched
        "season_display": ["2026", "2026", "2025"],              # display text: untouched
        "spread_line": ["-3.5", "2.5", None],                    # generic text: untouched
        "team": ["CAR", "ATL", "GB"],
        "won": [1.0, 0.0, np.nan],
        "temp": [72, 65, 58],
    })


def test_key_columns_become_numpy_ints_or_floats():
    out = coerce_table_dtypes(_frame(), table="t", log=False)
    assert out["season"].dtype == "int64" and out["season"].tolist() == [2026, 2026, 2025]
    assert out["week"].dtype == "float64" and out["week"].tolist()[:2] == [1.0, 2.0] and np.isnan(out["week"].iloc[2])


def test_dates_become_naive_datetimes_with_mixed_formats():
    out = coerce_table_dtypes(_frame(), log=False)
    assert pd.api.types.is_datetime64_any_dtype(out["official_date"])
    assert out["official_date"].dt.tz is None
    assert out["official_date"].iloc[0].strftime("%Y-%m-%d") == "2026-09-13"
    assert out["official_date"].iloc[1].strftime("%Y-%m-%d %H:%M") == "2026-09-20 13:00"
    assert pd.isna(out["official_date"].iloc[2])


def test_object_id_columns_get_one_mergeable_dtype():
    out = coerce_table_dtypes(_frame(), log=False)
    assert out["away_probable_pitcher_id"].dtype == "float64"  # was all-null object
    assert out["home_probable_pitcher_id"].dtype == "float64"
    assert out["home_probable_pitcher_id"].tolist()[:2] == [660271.0, 543037.0]
    assert out["gsis_id"].tolist() == ["00-0039150", "00-0032950", None]
    assert out["game_id"].tolist() == ["2026001", "2026_002", None]
    # the merge that broke MLB for two months: object-vs-int64 keys now merge
    right = pd.DataFrame({"home_probable_pitcher_id": pd.Series([660271], dtype="int64"), "era": [3.1]})
    merged = out[["home_probable_pitcher_id"]].merge(right, on="home_probable_pitcher_id", how="left")
    assert merged["era"].notna().sum() == 1


def test_already_typed_and_display_columns_are_untouched():
    src = _frame()
    out = coerce_table_dtypes(src, log=False)
    # merge_asof needs exact dtype equality: a numeric id must NEVER be re-typed
    assert out["player_id"].dtype == "int64" and out["player_id"].tolist() == src["player_id"].tolist()
    assert out["season_display"].tolist() == ["2026", "2026", "2025"]
    assert out["spread_line"].tolist() == ["-3.5", "2.5", None]
    assert out["team"].tolist() == ["CAR", "ATL", "GB"]
    assert out["won"].dtype == "float64" and out["temp"].dtype == "int64"


def test_empty_and_idempotent():
    empty = pd.DataFrame()
    assert coerce_table_dtypes(empty, log=False) is empty
    once = coerce_table_dtypes(_frame(), log=False)
    twice = coerce_table_dtypes(once, log=False)
    assert describe_dtypes(once) == describe_dtypes(twice)
    pd.testing.assert_frame_equal(once, twice)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("test_table_dtypes: all checks passed")
