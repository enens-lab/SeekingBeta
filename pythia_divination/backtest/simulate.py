
import numpy as np, pandas as pd

# TODO: Add win/loss count - track number of winning vs losing trades
# TODO: Add overall gains/loss - total P&L calculation across all trades
# TODO: Add risk metrics - volatility, VaR, standard deviation of returns
# TODO: Add drawdown tracking - max drawdown, avg drawdown, drawdown duration
# TODO: Add risk-adjusted returns - Sharpe ratio, Sortino ratio, Calmar ratio

def equal_weight_topk(prob_up: pd.DataFrame, close: pd.DataFrame, budget: float, k_top=5, threshold=0.55):
    pred = prob_up.shift(1)  # trade next day on yesterday's signal
    picks = pred.apply(lambda row: row[row>=threshold].nlargest(k_top).index.tolist(), axis=1)
    weights = pred.copy()*0.0
    for i, cols in enumerate(picks):
        if len(cols):
            weights.iloc[i][cols] = 1.0/len(cols)
    rets = close.pct_change().fillna(0.0)
    strat = (weights * rets).sum(axis=1)
    equity = (1+strat).cumprod()
    total_return = equity.iloc[-1]-1
    peak = equity.cummax()
    drawdown = (equity/peak - 1).min()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = equity.iloc[-1]**(1/years) - 1 if years>0 else 0
    # TODO: Expand return dict to include:
    #   - win_count, loss_count (number of winning/losing trades)
    #   - total_pnl (overall gains/loss in dollar terms)
    #   - volatility, var_95 (risk metrics)
    #   - avg_drawdown, drawdown_duration (extended drawdown tracking)
    #   - sharpe_ratio, sortino_ratio, calmar_ratio (risk-adjusted returns)
    return {"cagr": float(cagr), "max_drawdown": float(drawdown), "total_return": float(total_return)}
