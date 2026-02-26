Place backtest CSV files here for the public track-record API.

Supported default filenames:
- `jackpot_trade_log.csv` (trade ledger with `date`, `balance`, and `return`)
- `trade_summary_prob_strategy.csv` (from `run_backtest_v4.py`)
- `trade_summary.csv`
- `backtest_results.csv`
- `results.csv`

You can also override the path with:
- `BACKTEST_RESULTS_PATH=/app/data/backtests/your_file.csv`

Optional:
- `BACKTEST_TRANSACTION_COST_BPS=10` (default: 10 bps one-way; API applies round-trip cost)
