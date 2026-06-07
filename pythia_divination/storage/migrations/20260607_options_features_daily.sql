-- Daily options-feature archive for lstm_quant.
-- One row per symbol per trade day; rebuilds a ThetaData-like options history
-- from live Schwab (real greeks) / yfinance (Black-Scholes) reconstructions.

CREATE TABLE IF NOT EXISTS options_features_daily (
    symbol TEXT NOT NULL,
    trade_date DATE NOT NULL,
    source TEXT NOT NULL DEFAULT 'unknown',          -- 'schwab' | 'yfinance'
    options_live BOOLEAN NOT NULL DEFAULT FALSE,      -- features actually reconstructed (vs zero)
    underlying_price DOUBLE PRECISION NULL,
    atm_iv DOUBLE PRECISION NULL,
    iv_skew_25d DOUBLE PRECISION NULL,
    iv_term_slope DOUBLE PRECISION NULL,
    pc_oi_ratio DOUBLE PRECISION NULL,
    pc_volume_ratio DOUBLE PRECISION NULL,
    opt_volume_ratio DOUBLE PRECISION NULL,
    gex DOUBLE PRECISION NULL,
    net_delta_oi DOUBLE PRECISION NULL,
    total_oi DOUBLE PRECISION NULL,
    n_contracts DOUBLE PRECISION NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_options_features_daily_date
    ON options_features_daily(trade_date DESC);
