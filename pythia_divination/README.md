# Pythia Divination - Prediction Engine

FastAPI service for model inference and market-data synchronization.

## Current architecture (production)

- LSTM endpoints are **DB-first + cache-aware**.
- Market OHLCV data is synchronized in background batches into Postgres (`market_ohlcv_daily`).
- Homepage and ticker predictions can run in **cache-only mode** to avoid request-path provider latency.
- Supported providers: `alpaca`, `stooq`, `yahoo` (runtime-configurable via `DATA_SOURCE`).

## Core endpoints

- `GET /healthz`
- `GET /predict/lstm_5d/{ticker}`
- `GET /predict/lstm_jackpot/{ticker}`
- `GET /predict/homepage`
- `POST /predict/cache/warm?force_refresh=true`
- `GET /predict/cache/status`
- `POST /ops/market-data/sync?limit=250`
- `GET /ops/market-data/status`

## Important env flags

```env
# Request-path behavior
HOMEPAGE_CACHE_ONLY=true
LSTM_SERVE_CACHE_ONLY=true
LSTM_DB_MARKET_DATA_ENABLED=true
LSTM_ALLOW_LIVE_DATA_FETCH=false

# Cache TTLs
LSTM_RESPONSE_CACHE_TTL_SECONDS=259200
LSTM_STALE_CACHE_TTL_SECONDS=1209600

# Market-data sync
MARKET_DATA_SYNC_ENABLED=true
MARKET_DATA_SYNC_SOURCE=alpaca
MARKET_DATA_SYNC_INTERVAL_SECONDS=86400
MARKET_DATA_SYNC_BATCH_SIZE=250
MARKET_DATA_SYNC_MAX_AGE_DAYS=1
MARKET_DATA_SYNC_LOOKBACK_DAYS=520
MARKET_DATA_SYNC_TICKERS=
```

## Quickstart (local)

```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.service:app --reload --port 8000
```

## Notes

- `MARKET_DATA_SYNC_ENABLED=true` starts a background sync loop on service startup.
- Retired/delisted symbols may return no bars from providers and are counted as per-ticker sync failures.
- For first-time bootstrap, run:
  - `POST /ops/market-data/sync`
  - `POST /predict/cache/warm`
