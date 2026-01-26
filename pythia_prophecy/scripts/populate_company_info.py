#!/usr/bin/env python3
"""
Populate company info and news from yfinance.
Run from the pythia_prophecy directory: python scripts/populate_company_info.py

Options:
    --ticker AAPL       Fetch a single ticker
    --stale-only        Only re-fetch companies with stale data (>24h old)
    --no-news           Skip fetching news articles
"""
import sys
import argparse
from pathlib import Path

# Add parent to path so we can import api modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import init_database, get_stale_companies
from api.stock_info import fetch_and_store_stock_info, fetch_and_store_batch


def load_universe() -> list[str]:
    """Load the stock universe from pythia_divination config."""
    try:
        import yaml
        config_path = Path(__file__).parent.parent.parent / "pythia_divination" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            universe = config.get("universe", [])
            if universe:
                return universe
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Could not load divination config: {e}")

    # Fallback universe
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "UNH",
        "HD", "PG", "MA", "XOM", "BAC", "JNJ", "COST", "ABBV", "WMT", "KO",
        "MRK", "PEP", "CVX", "LLY", "AMD", "AVGO", "ADBE", "CRM", "ORCL", "INTC",
        "SPY", "QQQ", "IWM", "DIA", "GLD",
    ]


def main():
    parser = argparse.ArgumentParser(description="Populate company info from yfinance")
    parser.add_argument("--ticker", type=str, help="Fetch a single ticker")
    parser.add_argument("--stale-only", action="store_true", help="Only re-fetch stale data (>24h)")
    parser.add_argument("--no-news", action="store_true", help="Skip news articles")
    args = parser.parse_args()

    init_database()
    include_news = not args.no_news

    if args.ticker:
        # Single ticker mode
        ticker = args.ticker.upper()
        print(f"Fetching info for {ticker}...")
        result = fetch_and_store_stock_info(ticker, include_news=include_news)
        _print_result(result)
    elif args.stale_only:
        # Stale-only mode
        stale = get_stale_companies(hours=24)
        if not stale:
            print("No stale companies found. All data is fresh.")
            return
        print(f"Found {len(stale)} stale companies to refresh.")
        results = fetch_and_store_batch(stale, include_news=include_news)
        _print_summary(results)
    else:
        # Full universe mode
        universe = load_universe()
        print(f"Fetching info for {len(universe)} tickers...")
        print(f"News: {'yes' if include_news else 'no'}")
        print("-" * 50)
        results = fetch_and_store_batch(universe, include_news=include_news)
        _print_summary(results)


def _print_result(result: dict):
    """Print a single result."""
    ticker = result["ticker"]
    if result.get("error"):
        print(f"  [{ticker}] ERROR: {result['error']}")
    else:
        name = result.get("name", "Unknown")
        asset_type = result.get("asset_type", "?")
        news_count = result.get("news_count", 0)
        print(f"  [{ticker}] {name} ({asset_type}) - info stored, {news_count} new articles")
        if result.get("news_error"):
            print(f"    News warning: {result['news_error']}")


def _print_summary(results: list[dict]):
    """Print batch summary."""
    print("-" * 50)
    success = [r for r in results if r.get("info_stored")]
    errors = [r for r in results if r.get("error")]
    total_news = sum(r.get("news_count", 0) for r in results)

    print(f"Results: {len(success)} success, {len(errors)} errors, {total_news} new articles")

    if errors:
        print("\nErrors:")
        for r in errors:
            print(f"  [{r['ticker']}] {r['error']}")


if __name__ == "__main__":
    main()
