# CLAUDE.md - Pythia Divination

Personal trading engine containing ML strategies, algorithmic trading strategies, automated trade execution for the owner, and backtesting capabilities. This component handles all trading-related functionality including strategy development, model training, and trade automation.

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Database
python -m storage.scripts.migrate

# Train models
python -m models.train --all

# Run server
python main.py --reload --port 8000
```

## Architecture

```
pythia_divination/
├── api/
│   ├── service.py      # Main FastAPI app, all endpoints
│   ├── schemas.py      # Pydantic request/response models
│   ├── auth.py         # JWT authentication
│   ├── tiers.py        # User tier configuration
│   └── analytics.py    # Model analytics router
├── models/
│   ├── registry.py     # Model registration system
│   ├── base.py         # Base model class
│   ├── train.py        # Training script
│   ├── gradient_boosting.py
│   ├── linear_regression.py
│   ├── random_forest.py
│   └── lstm.py
├── features/
│   ├── technical.py    # Feature engineering (10 features)
│   └── sequences.py    # LSTM sequence creation
├── storage/
│   ├── postgres.py     # Async PostgreSQL layer
│   ├── migrations/     # SQL migration files
│   └── scripts/        # Migration runner
├── data/
│   ├── fetch.py        # OHLCV data fetching
│   └── providers/      # Data source adapters (Alpaca)
├── artifacts/          # Trained model files
│   └── {model}/{task}/ # model.joblib, scaler.joblib, metrics.json
├── static/             # Dashboard UI
│   ├── index.html      # Main dashboard
│   ├── analyze.html    # Stock analysis page
│   ├── css/
│   └── js/
└── config/
    ├── settings.py     # Settings loader
    └── config.yaml     # Universe + configuration
```

## Key Endpoints

### Authentication
- `POST /auth/login` - Login with username/password, returns JWT
- `GET /auth/me` - Get current user info

### Predictions
- `GET /predict/{ticker}` - Single model prediction
- `POST /predict/multi` - Multi-model predictions with consensus
- `GET /models` - List available models

### Analysis (requires auth)
- `GET /api/universe` - Get available stocks (tier-filtered)
- `GET /api/analyze/models` - Get models available to user
- `POST /api/analyze` - Run analysis on selected stocks
- `GET /api/user/features` - Get user's tier features and limits

### Paper Trading
- `POST /paper/run` - Execute trading strategy
- `GET /paper/positions` - Current positions
- `GET /paper/metrics` - Historical performance

## User Tiers

| Tier | Daily Requests | Models | Max Stocks |
|------|----------------|--------|------------|
| free | 10 | gradient_boosting, linear_regression | 5 |
| pro | 100 | All 4 models | 20 |
| enterprise | Unlimited | All + custom | 150 |

## Test Accounts

| Username | Password | Tier |
|----------|----------|------|
| free_user | free123 | free |
| pro_user | pro123 | pro |
| enterprise_user | enterprise123 | enterprise |

## ML Models

All models support both classification (buy/hold/sell) and regression (return prediction):

1. **Gradient Boosting** - Best overall classifier (AUC ~0.51)
2. **Linear Regression** - Fastest training, baseline
3. **Random Forest** - Ensemble method
4. **LSTM** - Neural network, requires sequences (best AUC ~0.54)

## Features (10 total)

- `ret_1d`, `ret_5d` - Momentum
- `vol_10` - Volatility
- `sma_10`, `sma_20`, `sma_gap` - Moving averages
- `rsi_14` - Relative Strength Index
- `macd`, `macd_sig` - MACD indicators
- `vol_z` - Volume z-score

## Database Tables

- `users` - User accounts with tier
- `rate_limits` - Daily API usage tracking
- `trades` - Order history
- `positions_snapshots` - Portfolio snapshots
- `run_metrics` - Performance metrics

## Environment Variables

Required in `.env`:
- `DATABASE_URL` - PostgreSQL connection string
- `ALPACA_KEY_ID` - Alpaca API key
- `ALPACA_SECRET_KEY` - Alpaca secret
- `JWT_SECRET_KEY` - JWT signing key (optional, has default)
