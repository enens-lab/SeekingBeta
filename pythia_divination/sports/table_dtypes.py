"""Column dtype normalization for the sports tables at load time.

Why: the exporters and dataset builders join tables on ids, seasons, weeks and
dates that arrive with whatever dtype the last writer happened to produce. An
all-null pitcher-id column parses as `object`, a season as a string in one file
and an integer in the next, a date as text. pandas then either refuses the merge
("You are trying to merge on object and int64 columns", which took the MLB
upcoming boards down for two months) or silently matches nothing. Fixing the
dtype once, where the table is read, covers every downstream join.

Rules are deliberately conservative. Only columns that are currently `object`
(text / mixed / all-null) are touched; a column that is already numeric or
datetime is never changed, because merge_asof requires EXACT dtype equality on
both sides and the MLB export uses it (turning an int64 id into float64 on one
side broke it in the dry run). No pandas extension dtypes (Int64 / Float64) are
introduced either -- they leak into feature frames as object arrays.

  * season / week / year (object)              -> int64, or float64 when nulls exist
  * official_date / gameday / game_date / date -> datetime64[ns], naive (mixed formats)
  * game_id (object)                           -> str
  * other *_id (object) whose non-null values all parse as numbers
                                               -> int64, or float64 when nulls exist
  * *_display / *_name / *_key / *_code / *_label / *_text are never touched
`coerce_table_dtypes` returns a new frame and logs one line per table listing
what changed, so a schema drift is visible in the export log.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_INT_KEY_COLUMNS = {"season", "week", "year"}
_DATE_COLUMNS = {"official_date", "gameday", "game_date", "date"}
_STRING_ID_COLUMNS = {"game_id"}
_NEVER_SUFFIXES = ("_display", "_name", "_key", "_code", "_label", "_text")


def _numeric_like(values: pd.Series) -> bool:
    """True when every non-null value parses as a number (empty -> True)."""
    values = values.dropna()
    if values.empty:
        return True
    if pd.api.types.is_bool_dtype(values):
        return False
    converted = pd.to_numeric(values.astype(str).str.strip(), errors="coerce")
    return bool(converted.notna().all())


def _to_numeric_key(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and np.allclose(numeric % 1, 0):
        return numeric.astype("int64")
    return numeric.astype("float64")


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    try:
        converted = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
    except (TypeError, ValueError):  # pandas < 2.0 has no format="mixed"
        converted = pd.to_datetime(series, errors="coerce", utc=True)
    return converted.dt.tz_localize(None)


def coerce_table_dtypes(frame: pd.DataFrame, *, table: str = "", log: bool = True) -> pd.DataFrame:
    """Normalize key/id/date dtypes of `object` columns; returns a new frame."""
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    changes: dict[str, str] = {}
    for column in list(out.columns):
        name = str(column)
        lower = name.lower()
        series = out[column]
        if series.dtype != object or lower.endswith(_NEVER_SUFFIXES):
            continue  # already typed (numeric/datetime/bool) or a display field: leave it
        try:
            if lower in _INT_KEY_COLUMNS:
                if _numeric_like(series):
                    out[column] = _to_numeric_key(series)
                    changes[name] = str(out[column].dtype)
            elif lower in _DATE_COLUMNS:
                converted = _to_naive_datetime(series)
                if converted.notna().sum() >= series.notna().sum() * 0.9:
                    out[column] = converted
                    changes[name] = "datetime64"
            elif lower in _STRING_ID_COLUMNS:
                if series.dropna().map(type).nunique() > 1 or not series.dropna().map(lambda v: isinstance(v, str)).all():
                    out[column] = series.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
                    changes[name] = "str"
            elif lower.endswith("_id"):
                if _numeric_like(series):
                    out[column] = _to_numeric_key(series)
                    changes[name] = str(out[column].dtype)
        except Exception as exc:  # never let a coercion take the export down
            logger.debug("dtype coercion skipped for %s.%s: %s", table, name, exc)
    if log and changes:
        summary = ", ".join(f"{k}->{v}" for k, v in list(changes.items())[:12])
        more = f" (+{len(changes) - 12} more)" if len(changes) > 12 else ""
        logger.info("dtype coercion %s: %d columns normalized: %s%s", table or "<table>", len(changes), summary, more)
    return out


def describe_dtypes(frame: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    """Small helper for logs/tests: {column: dtype} for the given (or all) columns."""
    cols = columns or list(frame.columns)
    return {str(c): str(frame[c].dtype) for c in cols if c in frame.columns}
