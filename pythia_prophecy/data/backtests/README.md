Place model-specific backtest artifacts here for the public performance API.

## Current default files

- `production_trade_log.csv` for `model=lstm_5d`
- `production_metrics.json` for `model=lstm_5d` summary overrides
- `jackpot_trade_log.csv` for `model=lstm_jackpot`
- `jackpot_metrics.json` for `model=lstm_jackpot` summary overrides

## API usage

- `GET /api/performance/track-record?model=lstm_5d`
- `GET /api/performance/track-record?model=lstm_jackpot`
- `GET /api/performance/curve?model=lstm_5d`
- `GET /api/performance/curve?model=lstm_jackpot`

## Supported trade log columns

Minimum:
- `date`
- `return_pct` or `return` or `actual_return`

Optional:
- `balance` / `equity` / `portfolio_value`
- `days_held`
- `regime` / `market_regime`

## Environment overrides

Global:
- `BACKTEST_RESULTS_PATH=/app/data/backtests/your_file.csv`
- `BACKTEST_METRICS_PATH=/app/data/backtests/your_metrics.json`

Model-specific:
- `BACKTEST_RESULTS_PATH_LSTM_5D=...`
- `BACKTEST_RESULTS_PATH_LSTM_JACKPOT=...`
- `BACKTEST_METRICS_PATH_LSTM_5D=...`
- `BACKTEST_METRICS_PATH_LSTM_JACKPOT=...`

Cost setting:
- `BACKTEST_TRANSACTION_COST_BPS=10` (default: 10 bps one-way; round-trip applied to non-net return fields)
