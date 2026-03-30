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

## Sports expansion: PGA data collection

We now have an initial PGA Tour ingestion + dataset scaffold under:

- `sports/pga/client.py`
- `sports/pga/ingest.py`
- `sports/pga/storage.py`
- `sports/pga/ingest_history.py`
- `sports/pga/ingest_profiles.py`
- `sports/pga/build_training_dataset.py`
- `sports/pga/train_baseline.py`
- `sports/pga/train_neural_model.py`
- `sports/pga/train_multitask_neural_model.py`
- `sports/pga/train_multitask_torch.py`

It collects PGA Tour stat detail tables from `https://www.pgatour.com/stats` and stores both:

- raw Next.js snapshots for reproducibility
- normalized player-stat CSVs for future model feature engineering
- tournament history / leaderboard labels
- stable public player-profile features (age/birth year, turned pro, birthplace, college, height, equipment sponsor, socials)
- joined training datasets for first-pass supervised modeling
- multi-task neural model scaffolding for `won`, `top_10`, and `made_cut`

Quick smoke test:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python -m sports.pga.ingest --limit 3 -v
python -m sports.pga.ingest_history --season 2026 --limit 3 -v
python -m sports.pga.ingest_profiles --player-id 46046 --player-name "Scottie Scheffler" -v
python -m sports.pga.build_training_dataset --season-start 2024 --season-end 2026 -v
python -m sports.pga.train_baseline --target won -v
python -m sports.pga.train_neural_model --target won --sequence-length 8 -v
python -m sports.pga.train_multitask_neural_model --targets won top_10 made_cut --sequence-length 10 -v
python -m sports.pga.train_multitask_torch --targets won top_10 made_cut --sequence-length 12 --device auto -v
```

Apple-Silicon PyTorch workflow:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
bash scripts/setup_pga_torch_env.sh
.venv-torch/bin/python -m sports.pga.benchmark_torch_env --device auto
bash scripts/train_pga_multitask_torch.sh
```

Notes:

- `.venv-torch` is the working Apple-GPU path today and uses PyTorch MPS.
- `.venv-metal` is experimental only; the current TensorFlow Metal plugin path is not stable enough to rely on for day-to-day PGA training.
- `scripts/train_pga_multitask_torch.sh` rebuilds the dataset and launches the multi-task PGA model with defaults tuned for larger local Apple-Silicon runs.

## Sports expansion: MLB data collection

We now have an MLB matchup pipeline under:

- `sports/mlb/client.py`
- `sports/mlb/ingest_history.py`
- `sports/mlb/roster_features.py`
- `sports/mlb/collect_statcast.py`
- `sports/mlb/statcast_enrichment.py`
- `sports/mlb/build_training_dataset.py`
- `sports/mlb/train_baseline.py`
- `sports/mlb/train_torch.py`

It currently supports:

- official MLB regular-season schedule + game-detail backfills
- rolling team and starting-pitcher features
- lineup-strength features from saved boxscores
- bullpen-quality features from saved boxscores
- optional Statcast daily feature enrichment
- durable CSV/parquet storage for training datasets and feature tables

Quick smoke test:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python -m sports.mlb.ingest_history --season 2025 --include-game-details -v
python -m sports.mlb.build_training_dataset --season-start 2020 --season-end 2025 -v
python -m sports.mlb.train_baseline --model hist_gradient_boosting -v
```

Multi-season Statcast backfill:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
bash scripts/backfill_mlb_statcast.sh
```

Apple-Silicon MLB torch workflow:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
bash scripts/setup_pga_torch_env.sh
bash scripts/train_mlb_torch.sh
```

Output lands under:

```text
pythia_divination/data/sports/pga/raw/<snapshot_tag>/
pythia_divination/data/sports/pga/normalized/
pythia_divination/artifacts/pga_baseline/<model_type>/<target>/
pythia_divination/artifacts/pga_neural/<target>/
pythia_divination/artifacts/pga_neural_multitask/
pythia_divination/artifacts/pga_neural_multitask_torch/
```
