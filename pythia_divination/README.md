# Pythia Divination - Trading Engine

Personal trading engine for ML strategies, algorithmic trading, automated trade execution, and backtesting.

Key capabilities:
- ML-powered trading strategies (Gradient Boosting, Linear Regression, Random Forest, LSTM)
- Algorithmic trading strategy development
- Automated trade execution
- Backtesting framework
- Paper trading integration
- Feature engineering with technical indicators

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train baseline model and save artifacts
python -m models.train

# Run API
uvicorn api.service:app --reload --port 8000
```

Key endpoints:
- `GET /predict/{ticker}`
- `POST /portfolio/allocate`
- `POST /simulate/dry-run`
- `POST /data/ingest`

Project layout:

```
pythia_divination/
├─ api/             # FastAPI app, auth, tiers, schemas
├─ data/            # Data fetching & providers (Alpaca)
├─ features/        # Feature engineering (technical indicators)
├─ models/          # ML models (GB, LR, RF, LSTM)
├─ storage/         # PostgreSQL layer & migrations
├─ artifacts/       # Trained model files
├─ static/          # Dashboard UI
├─ config/          # Settings & configuration
└─ requirements.txt
```
