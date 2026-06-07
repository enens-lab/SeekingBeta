"""Options-enriched quant LSTM (`lstm_quant`) — ported from TradeBot.

A 48-feature, 60-timestep torch LSTM (Conv1D -> LSTM256 -> MHA -> LSTM128 ->
Dense) predicting P(>2% in 5 trading days). Served in pythia alongside the keras
lstm_5d/lstm_jackpot models. The 10 options features are reconstructed live from
yfinance/Schwab (ThetaData being retired) via the shared extractor.
"""
