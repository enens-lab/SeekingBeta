"""
Stock info fetcher service.
Fetches company fundamentals and news from yfinance and stores in the database.
"""
from __future__ import annotations

import time
import json
from datetime import datetime
from typing import Optional

import yfinance as yf

from .database import upsert_company, upsert_company_info, add_company_news
from .logging_config import get_logger

logger = get_logger("stock_info")


# Known ETF tickers for classification
ETF_TICKERS = {
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", "MDY",
    "XLF", "XLK", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU",
    "XLB", "XLRE", "XLC", "BND", "TLT", "IEF", "LQD", "HYG",
    "GLD", "SLV", "USO", "UNG", "EFA", "EEM", "VWO", "FXI", "EWJ",
}

# ADR indicators
ADR_EXCHANGES = {"PNK", "OTC"}


def determine_asset_type(ticker: str, info: dict) -> str:
    """Classify a ticker as stock, etf, or adr."""
    if ticker.upper() in ETF_TICKERS:
        return "etf"
    quote_type = info.get("quoteType", "").upper()
    if quote_type == "ETF":
        return "etf"
    exchange = info.get("exchange", "")
    if exchange in ADR_EXCHANGES:
        return "adr"
    if info.get("country") and info.get("country") != "United States":
        if exchange not in {"NMS", "NYQ", "NGM", "NCM", "ASE"}:
            return "adr"
    return "stock"


def fetch_and_store_stock_info(ticker: str, include_news: bool = True) -> dict:
    """
    Fetch company info and news from yfinance and store in the database.

    Args:
        ticker: Stock ticker symbol
        include_news: Whether to also fetch and store news articles

    Returns:
        dict with keys: ticker, name, asset_type, info_stored, news_count
    """
    ticker = ticker.upper()
    result = {"ticker": ticker, "name": None, "asset_type": None, "info_stored": False, "news_count": 0}
    logger.debug(f"Fetching stock info for {ticker}")

    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
    except Exception as e:
        logger.error(f"Failed to fetch info for {ticker}: {e}")
        result["error"] = f"Failed to fetch info: {e}"
        return result

    if not info or info.get("trailingPegRatio") is None and info.get("shortName") is None:
        # yfinance returns an empty-ish dict for invalid tickers
        if not info.get("shortName") and not info.get("longName"):
            logger.warning(f"No data returned from yfinance for {ticker}")
            result["error"] = "No data returned from yfinance"
            return result

    # Determine asset type and name
    asset_type = determine_asset_type(ticker, info)
    name = info.get("longName") or info.get("shortName")
    result["name"] = name
    result["asset_type"] = asset_type

    # Upsert company record
    upsert_company(ticker, name=name, asset_type=asset_type)

    # Build info dict for storage
    company_info = {
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "description": info.get("longBusinessSummary"),
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "employees": info.get("fullTimeEmployees"),
        "website": info.get("website"),
        "country": info.get("country"),
        "state": info.get("state"),
        "city": info.get("city"),
        "exchange": info.get("exchange"),
        "currency": info.get("currency"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "avg_volume": info.get("averageVolume"),
        "raw_info": info,
    }

    upsert_company_info(ticker, company_info)
    result["info_stored"] = True
    logger.info(f"Stored company info for {ticker}: {name} ({asset_type})")

    # Fetch and store news
    if include_news:
        try:
            news = yf_ticker.news
            if news:
                articles = []
                for item in news:
                    # yfinance 1.0 returns news in a new structure with content nested
                    content = item.get("content", {})

                    # Handle both old and new yfinance formats
                    if content:
                        # New yfinance 1.0 format
                        published_at = content.get("pubDate")  # Already ISO format

                        thumbnail_url = None
                        thumbnail = content.get("thumbnail")
                        if thumbnail:
                            resolutions = thumbnail.get("resolutions", [])
                            if resolutions:
                                thumbnail_url = resolutions[-1].get("url")

                        provider = content.get("provider", {})
                        canonical_url = content.get("canonicalUrl", {})

                        articles.append({
                            "article_id": item.get("id"),
                            "title": content.get("title"),
                            "publisher": provider.get("displayName"),
                            "link": canonical_url.get("url"),
                            "published_at": published_at,
                            "article_type": content.get("contentType"),
                            "thumbnail_url": thumbnail_url,
                            "related_tickers": None,  # Not available in new format
                        })
                    else:
                        # Old yfinance format (fallback)
                        published_ts = item.get("providerPublishTime")
                        published_at = None
                        if published_ts:
                            published_at = datetime.utcfromtimestamp(published_ts).isoformat()

                        thumbnail_url = None
                        if item.get("thumbnail"):
                            resolutions = item["thumbnail"].get("resolutions", [])
                            if resolutions:
                                thumbnail_url = resolutions[-1].get("url")

                        articles.append({
                            "article_id": item.get("uuid"),
                            "title": item.get("title"),
                            "publisher": item.get("publisher"),
                            "link": item.get("link"),
                            "published_at": published_at,
                            "article_type": item.get("type"),
                            "thumbnail_url": thumbnail_url,
                            "related_tickers": item.get("relatedTickers"),
                        })
                result["news_count"] = add_company_news(ticker, articles)
                if result["news_count"] > 0:
                    logger.debug(f"Stored {result['news_count']} news articles for {ticker}")
        except Exception as e:
            logger.warning(f"Failed to fetch news for {ticker}: {e}")
            result["news_error"] = str(e)

    return result


def fetch_and_store_batch(
    tickers: list[str],
    include_news: bool = True,
    delay: float = 0.5,
    batch_size: int = 10,
    batch_delay: float = 2.0,
) -> list[dict]:
    """
    Fetch and store info for multiple tickers with rate limiting.

    Args:
        tickers: List of ticker symbols
        include_news: Whether to fetch news for each ticker
        delay: Seconds between individual calls
        batch_size: Number of tickers per batch
        batch_delay: Seconds between batches

    Returns:
        List of result dicts from fetch_and_store_stock_info
    """
    logger.info(f"Starting batch fetch for {len(tickers)} tickers")
    results = []
    success_count = 0
    error_count = 0

    for i, ticker in enumerate(tickers):
        result = fetch_and_store_stock_info(ticker, include_news=include_news)
        results.append(result)

        if result.get("info_stored"):
            success_count += 1
        if result.get("error"):
            error_count += 1

        # Rate limiting
        if i < len(tickers) - 1:
            time.sleep(delay)
            if (i + 1) % batch_size == 0:
                logger.debug(f"Batch progress: {i + 1}/{len(tickers)} tickers processed")
                time.sleep(batch_delay)

    logger.info(f"Batch fetch complete: {success_count} succeeded, {error_count} failed out of {len(tickers)}")
    return results
