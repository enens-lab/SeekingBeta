"""Storage helpers for raw and normalized PGA Tour data snapshots."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PGA_DATA_ROOT = ROOT / "data" / "sports" / "pga"


@dataclass(frozen=True)
class PGAIngestionPaths:
    """Concrete file-system paths for a single ingestion run."""

    root: Path
    raw_dir: Path
    normalized_dir: Path
    snapshot_tag: str


def build_ingestion_paths(
    *,
    base_dir: Path | str | None = None,
    snapshot_tag: str | None = None,
) -> PGAIngestionPaths:
    chosen_base = Path(base_dir) if base_dir else DEFAULT_PGA_DATA_ROOT
    resolved_tag = snapshot_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = chosen_base / "raw" / resolved_tag
    normalized_dir = chosen_base / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    return PGAIngestionPaths(root=chosen_base, raw_dir=raw_dir, normalized_dir=normalized_dir, snapshot_tag=resolved_tag)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, Path)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=_json_default)
        handle.write("\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_parquet(path: Path, frame: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return True
    except Exception as exc:  # pragma: no cover - depends on optional parquet engine
        logger.warning("Skipping parquet write for %s: %s", path, exc)
        return False


def write_partitioned_parquet(
    root_dir: Path,
    frame: pd.DataFrame,
    *,
    partition_column: str,
    filename: str = "data.parquet",
) -> list[str]:
    root_dir.mkdir(parents=True, exist_ok=True)
    if partition_column not in frame.columns or frame.empty:
        return []

    written_paths: list[str] = []
    grouped = frame.groupby(partition_column, dropna=False, sort=True)
    for partition_value, partition_frame in grouped:
        partition_slug = "unknown" if pd.isna(partition_value) else str(partition_value)
        target = root_dir / f"{partition_column}={partition_slug}" / filename
        if write_parquet(target, partition_frame.reset_index(drop=True)):
            written_paths.append(str(target))
    return written_paths


def read_preferred_table(*paths: Path) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".csv":
                return pd.read_csv(path, low_memory=False)
    raise FileNotFoundError(f"No table found in candidates: {[str(path) for path in paths]}")
