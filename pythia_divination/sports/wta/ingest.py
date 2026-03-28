"""Ingest historical ATP/WTA match data with optional local fallbacks."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd
import requests

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

DEFAULT_WTA_DATA_ROOT = PACKAGE_ROOT / "data" / "sports" / "wta"

logger = logging.getLogger(__name__)

BASE_URL_WTA = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"
BASE_URL_ATP = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _download_file(url: str, dest: Path) -> bool:
    """Download a file from a URL to a destination path."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        return True
    except Exception as exc:
        logger.error("Failed to download %s: %s", url, exc)
        return False


def _looks_like_html(path: Path) -> bool:
    try:
        head = path.read_text(errors="ignore")[:256].lower()
    except Exception:
        return False
    return "<html" in head or "<!doctype html" in head


def _column_index(cell_reference: str) -> int:
    letters = "".join(char for char in cell_reference if char.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.iter(f"{XML_NS}t"))

    value_node = cell.find(f"{XML_NS}v")
    if value_node is None:
        return ""

    raw_value = value_node.text or ""
    if cell_type == "s":
        return shared_strings[int(raw_value)]
    return raw_value


def _read_xlsx_frame(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for string_item in shared_root.findall(f"{XML_NS}si"):
                shared_strings.append("".join(text.text or "" for text in string_item.iter(f"{XML_NS}t")))

        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in rels_root.findall(f"{PKG_REL_NS}Relationship")
        }

        sheets = workbook_root.find(f"{XML_NS}sheets")
        first_sheet = sheets.findall(f"{XML_NS}sheet")[0]
        target = rel_map[first_sheet.attrib[f"{REL_NS}id"]]
        sheet_root = ET.fromstring(archive.read(f"xl/{target}"))
        sheet_rows = sheet_root.find(f"{XML_NS}sheetData").findall(f"{XML_NS}row")

        parsed_rows: list[list[str]] = []
        for row in sheet_rows:
            cells: dict[int, str] = {}
            max_index = -1
            for cell in row.findall(f"{XML_NS}c"):
                cell_reference = cell.attrib.get("r", "")
                cell_index = _column_index(cell_reference) if cell_reference else max_index + 1
                cells[cell_index] = _xlsx_cell_value(cell, shared_strings)
                max_index = max(max_index, cell_index)
            parsed_rows.append([cells.get(index, "") for index in range(max_index + 1)])

    if not parsed_rows:
        return pd.DataFrame()

    header = parsed_rows[0]
    body = parsed_rows[1:]
    return pd.DataFrame(body, columns=header)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def _excel_serial_to_yyyymmdd(value: Any) -> int | None:
    if value in {None, ""} or pd.isna(value):
        return None
    try:
        timestamp = pd.Timestamp("1899-12-30") + pd.to_timedelta(int(float(value)), unit="D")
    except (TypeError, ValueError, OverflowError):
        return None
    return int(timestamp.strftime("%Y%m%d"))


def _synthetic_player_id(tour: str, player_name: str) -> str:
    return f"fallback-{tour}-{_slugify(player_name)}"


def _build_score(row: pd.Series) -> str:
    sets: list[str] = []
    for set_number in range(1, 6):
        winner_games = str(row.get(f"W{set_number}", "")).strip()
        loser_games = str(row.get(f"L{set_number}", "")).strip()
        if winner_games and loser_games and winner_games.lower() != "nan" and loser_games.lower() != "nan":
            sets.append(f"{winner_games}-{loser_games}")
    if sets:
        return " ".join(sets)
    return str(row.get("Comment", "")).strip()


def _normalize_fallback_matches(frame: pd.DataFrame, *, year: int, tour: str, source: Path) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    level_column = "Series" if tour == "atp" else "Tier"
    if level_column not in frame.columns:
        raise ValueError(f"Fallback file {source} is missing required column {level_column}")

    normalized = pd.DataFrame()
    normalized["tourney_name"] = frame.get("Tournament", "")
    normalized["surface"] = frame.get("Surface", "")
    normalized["draw_size"] = pd.NA
    normalized["tourney_level"] = frame.get(level_column, "")
    court_series = frame.get("Court", "")
    normalized["indoor"] = pd.Series(court_series).fillna("").astype(str).map(
        lambda value: "I" if "indoor" in value.lower() else "O" if "outdoor" in value.lower() else pd.NA,
    )
    normalized["tourney_date"] = frame.get("Date", "").map(_excel_serial_to_yyyymmdd)
    normalized["winner_id"] = frame.get("Winner", "").fillna("").astype(str).map(lambda value: _synthetic_player_id(tour, value))
    normalized["winner_name"] = frame.get("Winner", "")
    normalized["loser_id"] = frame.get("Loser", "").fillna("").astype(str).map(lambda value: _synthetic_player_id(tour, value))
    normalized["loser_name"] = frame.get("Loser", "")
    normalized["winner_rank"] = pd.to_numeric(frame.get("WRank", pd.Series(dtype=float)), errors="coerce")
    normalized["loser_rank"] = pd.to_numeric(frame.get("LRank", pd.Series(dtype=float)), errors="coerce")
    normalized["winner_rank_points"] = pd.to_numeric(frame.get("WPts", pd.Series(dtype=float)), errors="coerce")
    normalized["loser_rank_points"] = pd.to_numeric(frame.get("LPts", pd.Series(dtype=float)), errors="coerce")
    normalized["score"] = frame.apply(_build_score, axis=1)
    normalized["best_of"] = pd.to_numeric(frame.get("Best of", pd.Series(dtype=float)), errors="coerce")
    normalized["round"] = frame.get("Round", "")
    normalized["minutes"] = pd.NA
    normalized["tour"] = tour.upper()
    normalized["data_source"] = f"fallback:{source.name}"

    for column in [
        "winner_seed",
        "winner_entry",
        "winner_hand",
        "winner_ht",
        "winner_ioc",
        "winner_age",
        "loser_seed",
        "loser_entry",
        "loser_hand",
        "loser_ht",
        "loser_ioc",
        "loser_age",
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_SvGms",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_SvGms",
        "l_bpSaved",
        "l_bpFaced",
    ]:
        normalized[column] = pd.NA

    grouping_columns = ["tourney_name", "tourney_date", "surface"]
    normalized["tourney_id"] = normalized.apply(
        lambda row: f"{year}-fallback-{_slugify(str(row['tourney_name']))}-{row['tourney_date']}",
        axis=1,
    )
    normalized["match_num"] = normalized.groupby(grouping_columns, dropna=False).cumcount() + 1

    preferred_order = [
        "tourney_id",
        "tourney_name",
        "surface",
        "draw_size",
        "tourney_level",
        "indoor",
        "tourney_date",
        "match_num",
        "winner_id",
        "winner_seed",
        "winner_entry",
        "winner_name",
        "winner_hand",
        "winner_ht",
        "winner_ioc",
        "winner_age",
        "winner_rank",
        "winner_rank_points",
        "loser_id",
        "loser_seed",
        "loser_entry",
        "loser_name",
        "loser_hand",
        "loser_ht",
        "loser_ioc",
        "loser_age",
        "loser_rank",
        "loser_rank_points",
        "score",
        "best_of",
        "round",
        "minutes",
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_SvGms",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_SvGms",
        "l_bpSaved",
        "l_bpFaced",
        "tour",
        "data_source",
    ]
    return normalized[preferred_order]


def _fallback_candidates(year: int, tour: str) -> list[Path]:
    candidates = [
        REPO_ROOT / f"{year}_{tour}.xlsx",
        REPO_ROOT / f"{year}_{tour}.csv",
        DEFAULT_WTA_DATA_ROOT / "fallback" / f"{year}_{tour}.xlsx",
        DEFAULT_WTA_DATA_ROOT / "fallback" / f"{year}_{tour}.csv",
    ]
    return [candidate for candidate in candidates if candidate.exists()]


def _load_fallback_matches(year: int, tour: str) -> pd.DataFrame:
    for candidate in _fallback_candidates(year, tour):
        try:
            logger.info("Trying local fallback for %s %s from %s", tour.upper(), year, candidate)
            if candidate.suffix.lower() == ".xlsx":
                frame = _read_xlsx_frame(candidate)
                return _normalize_fallback_matches(frame, year=year, tour=tour, source=candidate)
            if candidate.suffix.lower() == ".csv":
                if _looks_like_html(candidate):
                    logger.warning("Skipping fallback %s because it is HTML, not CSV data.", candidate)
                    continue
                frame = pd.read_csv(candidate, low_memory=False)
                if {"winner_id", "loser_id", "tourney_id"}.issubset(frame.columns):
                    frame["tour"] = tour.upper()
                    frame["data_source"] = f"fallback:{candidate.name}"
                    return frame
                logger.warning("Skipping fallback %s because it does not match the expected tennis schema.", candidate)
        except Exception as exc:
            logger.warning("Failed to load fallback %s: %s", candidate, exc)
    return pd.DataFrame()


def ingest_matches(years: list[int], tour: str = "wta", force: bool = False) -> pd.DataFrame:
    """Download and merge match data for the requested years and tour."""
    all_matches: list[pd.DataFrame] = []
    base_url = BASE_URL_ATP if tour == "atp" else BASE_URL_WTA

    for year in years:
        filename = f"{tour}_matches_{year}.csv"
        url = f"{base_url}/{filename}"
        dest = DEFAULT_WTA_DATA_ROOT / "raw" / filename
        frame = pd.DataFrame()

        if not dest.exists() or force:
            logger.info("Downloading %s match data for year %s...", tour.upper(), year)
            _download_file(url, dest)
        else:
            logger.info("Using cached %s match data for year %s.", tour.upper(), year)

        if dest.exists() and not _looks_like_html(dest):
            try:
                frame = pd.read_csv(dest, low_memory=False)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", dest, exc)

        if frame.empty:
            frame = _load_fallback_matches(year, tour)
            if not frame.empty:
                dest.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(dest, index=False)
                logger.info("Saved fallback %s data for %s to %s", tour.upper(), year, dest)

        if frame.empty:
            logger.warning("No usable %s data found for year %s", tour.upper(), year)
            continue

        frame["tour"] = tour.upper()
        all_matches.append(frame)

    if not all_matches:
        return pd.DataFrame()

    return pd.concat(all_matches, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ATP/WTA data from Jeff Sackmann's repos with local fallbacks.")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--force", action="store_true", help="Force re-download of data.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    years = list(range(args.start_year, args.end_year + 1))
    wta_matches = ingest_matches(years, tour="wta", force=args.force)
    atp_matches = ingest_matches(years, tour="atp", force=args.force)

    combined = pd.concat([wta_matches, atp_matches], ignore_index=True)
    logger.info("Combined %s match rows across ATP and WTA.", len(combined))

    for column in combined.columns:
        if combined[column].dtype == "object":
            combined[column] = combined[column].astype(str).replace("nan", "")

    normalized_dir = DEFAULT_WTA_DATA_ROOT / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    csv_output_path = normalized_dir / "tennis_matches_combined.csv"
    combined.to_csv(csv_output_path, index=False)
    logger.info("Saved combined tennis matches to %s", csv_output_path)

    parquet_output_path = normalized_dir / "tennis_matches_combined.parquet"
    try:
        combined.to_parquet(parquet_output_path, index=False)
        logger.info("Saved combined tennis matches to %s", parquet_output_path)
    except ImportError:
        logger.warning("Parquet engine unavailable; CSV output will be used instead of %s", parquet_output_path)

    for tour, url_base in [("wta", BASE_URL_WTA), ("atp", BASE_URL_ATP)]:
        filename = f"{tour}_players.csv"
        url = f"{url_base}/{filename}"
        dest = DEFAULT_WTA_DATA_ROOT / "raw" / filename
        if not dest.exists() or args.force:
            _download_file(url, dest)


if __name__ == "__main__":
    main()
