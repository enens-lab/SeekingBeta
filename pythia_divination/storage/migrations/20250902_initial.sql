-- 20250902_initial.sql

-- trades: broker order intents / results
CREATE TABLE IF NOT EXISTS trades (
  id SERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL CHECK (side IN ('buy','sell')),
  qty INTEGER NOT NULL,
  price DOUBLE PRECISION,
  status TEXT,
  provider_order_id TEXT,
  reason TEXT,
  raw_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);

-- positions snapshots: point-in-time holdings
CREATE TABLE IF NOT EXISTS positions_snapshots (
  ts TIMESTAMPTZ NOT NULL,
  symbol TEXT NOT NULL,
  qty INTEGER NOT NULL,
  avg_price DOUBLE PRECISION,
  PRIMARY KEY (ts, symbol)
);

-- run_metrics: daily/periodic rollups for dashboarding
CREATE TABLE IF NOT EXISTS run_metrics (
  ts TIMESTAMPTZ PRIMARY KEY,
  pnl DOUBLE PRECISION,
  equity DOUBLE PRECISION,
  cash DOUBLE PRECISION,
  win_rate DOUBLE PRECISION,
  turnover DOUBLE PRECISION,
  max_drawdown DOUBLE PRECISION,
  notes TEXT
);
