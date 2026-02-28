# Prediction DB-First Runbook

## Goal
- Serve homepage and dashboard predictions from cache/DB first.
- Remove live market-data dependency from request path.
- Perform provider calls only from your EC2 host on a controlled schedule.

## Env Settings
Set these in `/home/ec2-user/seekingbeta/.env`:

```env
# Request-path behavior
HOMEPAGE_CACHE_ONLY=true
LSTM_SERVE_CACHE_ONLY=true
LSTM_DB_MARKET_DATA_ENABLED=true
LSTM_ALLOW_LIVE_DATA_FETCH=true

# Cache retention
LSTM_RESPONSE_CACHE_TTL_SECONDS=259200
LSTM_STALE_CACHE_TTL_SECONDS=1209600

# Daily market-data sync (provider calls happen only on EC2)
MARKET_DATA_SYNC_ENABLED=true
MARKET_DATA_SYNC_SOURCE=stooq
MARKET_DATA_SYNC_INTERVAL_SECONDS=86400
MARKET_DATA_SYNC_BATCH_SIZE=250
MARKET_DATA_SYNC_MAX_AGE_DAYS=1
MARKET_DATA_SYNC_LOOKBACK_DAYS=520
# Optional: comma-separated subset; blank means full universe
MARKET_DATA_SYNC_TICKERS=

# Prophecy LSTM proxy behavior
DIVINATION_LSTM_TIMEOUT_SECONDS=8
LSTM_PROXY_DISABLE_LOCAL_FALLBACK=true
```

## Deploy
```bash
cd /home/ec2-user/seekingbeta
docker compose up -d --build --force-recreate divination-api prophecy-api frontend
```

## Initial Warm + Sync
```bash
cd /home/ec2-user/seekingbeta

# Sync daily OHLCV snapshot batch into Postgres
curl -sS -X POST "http://localhost:8000/ops/market-data/sync?limit=250" | python -m json.tool

# Warm prediction cache for homepage models/tickers
curl -sS -X POST "http://localhost:8000/predict/cache/warm?force_refresh=true" | python -m json.tool

# Verify cache and serving mode
curl -sS "http://localhost:8000/predict/cache/status" | python -m json.tool | sed -n '1,180p'
curl -skI "https://seekingbeta.ai/predict/homepage"
```

## Health Checks
```bash
cd /home/ec2-user/seekingbeta
docker compose ps
curl -sS http://localhost:8000/ops/market-data/status | python -m json.tool
curl -sS http://localhost:8000/predict/cache/status | python -m json.tool | sed -n '1,180p'
docker compose logs --tail=200 divination-api prophecy-api frontend | egrep -i "market-data sync|cache warm|stooq|timeout|503|504"
```

## Notes
- `HOMEPAGE_CACHE_ONLY=true` prevents homepage endpoint from triggering live fetches.
- `LSTM_SERVE_CACHE_ONLY=true` prevents per-ticker LSTM endpoints from blocking on provider fetches.
- If cache is empty for a ticker/model, endpoint returns quickly instead of waiting ~60s+.
- Keep `LSTM_ALLOW_LIVE_DATA_FETCH=true` during warm/sync bootstrap. After cache is stable, you can set it to `false` for strict DB-only serving.
