"""Shared stock universe loading and categorization utilities."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_UNIVERSE_CSV_PATH = ROOT / "data" / "universe.csv"
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")

# Curated sector buckets used for UI-friendly grouping.
SECTOR_BUCKETS: dict[str, list[str]] = {
    "Technology": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "AMD",
        "INTC",
        "AVGO",
        "QCOM",
        "TXN",
        "MU",
        "AMAT",
        "LRCX",
        "CRM",
        "ORCL",
        "ADBE",
        "NOW",
        "INTU",
        "IBM",
        "CSCO",
        "HPQ",
        "DELL",
    ],
    "Financials": [
        "JPM",
        "BAC",
        "WFC",
        "C",
        "GS",
        "MS",
        "USB",
        "PNC",
        "BRK-B",
        "AIG",
        "MET",
        "PRU",
        "BLK",
        "SCHW",
        "AXP",
        "V",
        "MA",
        "PYPL",
    ],
    "Healthcare": [
        "JNJ",
        "PFE",
        "MRK",
        "ABBV",
        "LLY",
        "BMY",
        "AMGN",
        "GILD",
        "REGN",
        "MRNA",
        "UNH",
        "CVS",
        "CI",
        "MDT",
        "ABT",
    ],
    "Consumer": [
        "HD",
        "LOW",
        "NKE",
        "SBUX",
        "MCD",
        "TGT",
        "TJX",
        "BKNG",
        "MAR",
        "F",
        "GM",
        "LULU",
        "WMT",
        "COST",
        "PG",
        "KO",
        "PEP",
        "PM",
        "MO",
        "CL",
    ],
    "Energy": [
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "PXD",
        "OXY",
        "KMI",
    ],
    "Industrials": [
        "CAT",
        "BA",
        "UPS",
        "FDX",
        "HON",
        "RTX",
        "LMT",
        "GE",
        "DE",
        "UNP",
    ],
    "ETFs": [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "IVV",
        "MDY",
        "XLF",
        "XLK",
        "XLV",
        "XLE",
        "XLI",
        "XLY",
        "XLP",
        "XLU",
        "XLB",
        "XLRE",
        "XLC",
        "BND",
        "TLT",
        "IEF",
        "LQD",
        "HYG",
        "GLD",
        "SLV",
        "USO",
        "UNG",
        "EFA",
        "EEM",
        "VWO",
        "FXI",
        "EWJ",
    ],
}


def normalize_ticker(value: str) -> str | None:
    ticker = value.strip().upper()
    if not ticker:
        return None
    if not TICKER_PATTERN.match(ticker):
        return None
    return ticker


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for raw in values:
        ticker = normalize_ticker(raw)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        deduped.append(ticker)
    return deduped


def read_universe_csv(path: Path) -> list[str]:
    if not path.exists():
        return []

    tickers: list[str] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() in {"ticker", "symbol", "a"}:
                continue
            maybe_ticker = normalize_ticker(row[0])
            if maybe_ticker:
                tickers.append(maybe_ticker)
    return _dedupe_keep_order(tickers)


def load_universe(base_universe: Iterable[str] | None = None, csv_path: str | None = None) -> list[str]:
    base = _dedupe_keep_order(base_universe or [])
    source_path = csv_path or os.getenv("UNIVERSE_CSV_PATH")
    csv_universe_path = Path(source_path) if source_path else DEFAULT_UNIVERSE_CSV_PATH

    expanded = read_universe_csv(csv_universe_path)

    # Keep curated symbols at the front for free/basic tiers, then append expanded list.
    merged = _dedupe_keep_order([*base, *expanded])
    return merged or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def categorize_stocks(stocks: list[str]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    stock_set = set(stocks)

    for category, symbols in SECTOR_BUCKETS.items():
        matched = [symbol for symbol in symbols if symbol in stock_set]
        if matched:
            categories[category] = matched

    categorized: set[str] = set()
    for symbols in categories.values():
        categorized.update(symbols)

    other = [symbol for symbol in stocks if symbol not in categorized]
    if not other:
        return categories

    # For a very large universe, split uncategorized symbols by first letter
    # so the UI is still navigable.
    if len(other) > 200:
        buckets: dict[str, list[str]] = {}
        for symbol in other:
            lead = symbol[0] if symbol else "#"
            key = lead if lead.isalpha() else "#"
            buckets.setdefault(key, []).append(symbol)

        for key in sorted(buckets):
            categories[f"Other ({key})"] = buckets[key]
        return categories

    categories["Other"] = other
    return categories
