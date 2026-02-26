# Homepage Prediction Cache Runbook

Goal: keep homepage predictions available even when upstream market-data providers are rate-limited.

## Architecture

- Cache-first LSTM serving for homepage tickers:
  - `/predict/lstm_5d/{ticker}`
  - `/predict/lstm_jackpot/{ticker}`
- New batch endpoint for frontend:
  - `/predict/homepage`
- Single-flight refresh lock per `(data_source, model, ticker)` to prevent request stampedes.
- In-memory raw OHLCV cache (shared across both LSTM models).
- Stale fallback for both raw OHLCV and prediction payloads.
- Background cache warmer loop with startup warm + periodic refresh.
- Snapshot persistence:
  - loads on startup from `HOME_CACHE_SNAPSHOT_PATH`
  - saves after warm cycles and on shutdown

## Default settings

- `HOME_CACHE_MODELS=lstm_5d,lstm_jackpot`
- `HOME_CACHE_TICKERS=AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META`
- `LSTM_RESPONSE_CACHE_TTL_SECONDS=900` (fresh)
- `LSTM_STALE_CACHE_TTL_SECONDS=3600` (stale responses)
- `LSTM_RAW_DATA_CACHE_TTL_SECONDS=600` (fresh OHLCV)
- `LSTM_RAW_DATA_STALE_TTL_SECONDS=86400` (stale OHLCV fallback)
- `HOME_CACHE_WARM_INTERVAL_SECONDS=900`
- `HOME_CACHE_WARM_ENABLED=true`
- `HOME_CACHE_WARM_ON_STARTUP=true`

## Operational endpoints

- Homepage batch:
  - `GET /predict/homepage`
- Cache status:
  - `GET /predict/cache/status`
- Force warm:
  - `POST /predict/cache/warm?force_refresh=true`

## Deploy checklist

1. Rebuild/recreate services:
   - `docker compose up -d --build --force-recreate divination-api prophecy-api frontend`
2. Confirm API health:
   - `curl -s http://localhost:8000/healthz`
3. Warm cache once:
   - `curl -s -X POST "http://localhost:8000/predict/cache/warm?force_refresh=true"`
4. Verify batch endpoint:
   - `curl -s http://localhost:8000/predict/homepage`
5. Verify cache status:
   - `curl -s http://localhost:8000/predict/cache/status`

## Incident triage

If cards fail:

1. Check direct provider error in divination logs:
   - `docker compose logs --tail=200 divination-api`
2. Check cache status endpoint:
   - if cache has entries, stale fallback should still serve.
3. If cache is empty and provider is rate-limited:
   - reduce warm interval aggressiveness.
   - avoid manual high-frequency curls.
   - wait for provider reset or switch data provider / egress IP.
