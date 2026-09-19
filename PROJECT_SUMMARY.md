# Pythia - Automated Trading Platform

## Overview

Pythia is a personal automated trading platform with two distinct components: a core trading engine for the owner's automated trading activities, and a user-facing subscription service.

## Components

### pythia_divination (Trading Engine)
**Status**: Active Development

Personal trading engine containing ML strategies, algorithmic trading strategies, automated trade execution, and backtesting capabilities:
- **ML Models**: Gradient Boosting, Linear Regression, Random Forest, LSTM
- **Tasks**: Classification (buy/hold/sell signals) and Regression (return prediction)
- **Data**: 150+ stocks/ETFs via Alpaca Markets API
- **Features**: 10 technical indicators (momentum, volatility, moving averages, RSI, MACD)

Key capabilities:
- ML-powered trading strategies
- Algorithmic trading strategy development
- Automated trade execution
- Backtesting framework
- Paper trading integration
- Multi-model predictions with consensus voting
- Interactive analysis dashboard

**Tech Stack**: FastAPI, PostgreSQL, scikit-learn, PyTorch, asyncpg

### pythia_prophecy (User Platform)
**Status**: Active Development

Frontend and Backend-for-Frontend (BFF) serving external users:
- **Purpose**: Marketing, user authentication, credential management, and user-specific information display
- **Tiers**: Free ($0), Basic ($19/mo), Pro ($49/mo)

Key capabilities:
- Marketing landing pages
- User login flow and authentication
- User credential management
- User-specific dashboard and information
- Subscription tier management
- React-based responsive UI
- Integration with pythia_divination for predictions

**Tech Stack**: FastAPI, React, Vite, SQLite, TailwindCSS

## Recent Work

### 2026-01-19
- Retrained all ML models (8 model/task combinations)
- Added user authentication system to pythia_divination
- Created three test accounts (free/pro/enterprise tiers)
- Built new /analyze page with stock selection, model selection, and tier-based limits
- Implemented rate limiting per user tier
- Set up project documentation structure

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      End Users                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              pythia_prophecy (port 8001)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐   │
│  │   Auth   │  │  Tiers   │  │   React Frontend (:3000) │   │
│  │  (JWT)   │  │ (SQLite) │  │                          │   │
│  └──────────┘  └──────────┘  └──────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼───────────────────────────────────┐
│              pythia_divination (port 8000)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐   │
│  │   Auth   │  │ Models   │  │   PostgreSQL             │   │
│  │  (JWT)   │  │ (4 types)│  │   (trades, metrics)      │   │
│  └──────────┘  └──────────┘  └──────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                 Alpaca Markets API                           │
│            (OHLCV data, paper trading)                       │
└─────────────────────────────────────────────────────────────┘
```

**Production topology (since 2026-09-19):** the diagram above is the local-dev
shape (`docker compose --profile ml up`). In production `pythia_divination` does
NOT run as a service. Its work is done in batch on the RunPod serverless worker
(`runpod/sports/`): the daily `stock_predictions` LSTM sweep and the sports
exports, both published as JSON to S3. `pythia_prophecy` serves `/predict/*`,
`/api/analyze`, the Daily Brief and the sports boards from those files
(`api/stock_predictions_store.py`, bind-mounted `data/predictions/` and
`frontend/src/data/`), and fetches `/api/market/history` from Yahoo itself. The
web box is therefore nginx + prophecy-api + postgres (~1 GB RSS), sized for a
2 GB Lightsail (`LIGHTSAIL_MIGRATION.md`).

## Quick Start

```bash
# Clone and setup
git clone <repo> && cd pythia

# Start databases
docker-compose up -d postgres

# Setup pythia_divination
cd pythia_divination
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m storage.scripts.migrate
python -m models.train --all  # Train all models
python main.py --reload       # Start ML API

# Setup pythia_prophecy (separate terminal)
cd pythia_prophecy
make install
make install-frontend
make run-dev                  # Start user API
cd frontend && npm run dev    # Start React frontend
```

## Configuration

Each subproject has its own `.env` file. See `.env.example` files for required variables.

Key environment variables:
- `DATABASE_URL` - PostgreSQL connection
- `ALPACA_KEY_ID` / `ALPACA_SECRET_KEY` - Market data
- `JWT_SECRET_KEY` - Authentication

## Development Workflow

1. Make changes in appropriate subproject
2. Test locally with `python main.py --reload`
3. Commit changes with descriptive message
4. Update daily log in `.claude/logs/`
5. Update this summary for significant changes

## Roadmap

### pythia_divination - Backtest Metrics
- [ ] Win/loss count - Track number of winning and losing trades
- [ ] Overall gains/loss - Calculate total P&L across all trades
- [ ] Risk metrics - Volatility, Value at Risk (VaR), standard deviation of returns
- [ ] Drawdown tracking - Maximum drawdown, average drawdown, drawdown duration
- [ ] Risk-adjusted returns - Sharpe ratio, Sortino ratio, Calmar ratio

### General
- [ ] Real-time predictions via WebSocket
- [ ] Model ensembling with weighted voting
- [ ] Backtesting dashboard
- [ ] Payment integration (Stripe)
- [ ] Mobile app
