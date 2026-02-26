# Prediction Serving Architecture (Scale Plan)

## Objective

Serve homepage and model predictions for high concurrency without overloading market-data providers.

## Current implementation (in repo)

- Batch homepage API: `GET /predict/homepage`
- Cache-first serving for LSTM homepage models (`lstm_5d`, `lstm_jackpot`)
- Single-flight lock per `(source, model, ticker)` to prevent duplicate refresh calls
- Background warming loop (startup + periodic interval)
- Stale fallback for:
  - prediction payload cache
  - raw OHLCV cache
- Cache snapshot persistence at `HOME_CACHE_SNAPSHOT_PATH`

## Recommended production defaults

- Warm interval: `900s`
- Fresh prediction TTL: `900s`
- Stale prediction TTL: `3600s`
- Fresh OHLCV TTL: `600s`
- Stale OHLCV TTL: `86400s`
- Scope cache to homepage tickers unless you add Redis + queue workers

## Capacity notes

- Homepage cache scope (7 tickers x 2 models) is small and stable.
- Main protection is eliminating per-user upstream fetches.
- With this flow, user traffic reads cache; only warmer/refresh path hits providers.

## Next phase for larger scale

1. Replace in-memory cache with Redis:
   - shared across containers
   - survives app restarts
   - supports distributed locking
2. Move refresh to worker queue:
   - Celery/RQ/Arq
   - scheduled jobs per ticker/model
3. Persist prediction history to Postgres:
   - audit trail
   - charting and analytics
4. Add provider abstraction with health scoring:
   - route reads to best-available provider
   - circuit-breakers for degraded providers
5. Add API rate limits at edge:
   - prevent abusive traffic from triggering stampedes

## SLO guardrails

- `p95` homepage prediction API latency < 250ms from cache
- cache hit ratio for homepage endpoint > 95%
- upstream provider calls bounded by warm schedule, not user volume
- stale serves allowed during provider incidents, with alerting
