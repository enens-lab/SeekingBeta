-- 20260214_training_samples.sql

-- Store raw training samples (one row per timestamp per ticker)
CREATE TABLE IF NOT EXISTS training_samples (
  ts TIMESTAMPTZ NOT NULL,
  ticker TEXT NOT NULL,
  data JSONB NOT NULL,
  PRIMARY KEY (ts, ticker)
);

CREATE INDEX IF NOT EXISTS idx_training_samples_ticker ON training_samples(ticker);
