"""Country medal-count model for the Olympics (Summer or Winter).

Gradient-boosted regression on lagged medal counts (the previous three Games)
plus a host-nation indicator. Trained on all historical editions of a given
season and rolled forward to project the next Games' medal table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Host NOC by year (best-effort match to dataset NOC codes).
SUMMER_HOSTS: dict[int, str] = {
    1896: "GRE", 1900: "FRA", 1904: "USA", 1908: "GBR", 1912: "SWE", 1920: "BEL",
    1924: "FRA", 1928: "NED", 1932: "USA", 1936: "GER", 1948: "GBR", 1952: "FIN",
    1956: "AUS", 1960: "ITA", 1964: "JPN", 1968: "MEX", 1972: "FRG", 1976: "CAN",
    1980: "URS", 1984: "USA", 1988: "KOR", 1992: "ESP", 1996: "USA", 2000: "AUS",
    2004: "GRE", 2008: "CHN", 2012: "GBR", 2016: "BRA",
}
WINTER_HOSTS: dict[int, str] = {
    1924: "FRA", 1928: "SUI", 1932: "USA", 1936: "GER", 1948: "SUI", 1952: "NOR",
    1956: "ITA", 1960: "USA", 1964: "AUT", 1968: "FRA", 1972: "JPN", 1976: "AUT",
    1980: "USA", 1984: "YUG", 1988: "CAN", 1992: "FRA", 1994: "NOR", 1998: "JPN",
    2002: "USA", 2006: "ITA", 2010: "CAN", 2014: "RUS",
}

FEATURES = ["prev1", "prev2", "prev3", "avg3", "host"]


def build_panel(medal_table: pd.DataFrame, hosts: dict[int, str] | None = None) -> pd.DataFrame:
    hosts = hosts if hosts is not None else SUMMER_HOSTS
    """Build a (year x noc) panel with lagged medal features over editions."""
    editions = sorted(medal_table["year"].unique())
    edition_index = {y: i for i, y in enumerate(editions)}
    nocs = sorted(medal_table["noc"].unique())

    pivot = (
        medal_table.pivot_table(index="noc", columns="year", values="total", aggfunc="sum", fill_value=0)
        .reindex(index=nocs, columns=editions, fill_value=0)
    )

    rows = []
    for noc in nocs:
        series = pivot.loc[noc]
        for y in editions:
            i = edition_index[y]
            prev1 = float(series[editions[i - 1]]) if i >= 1 else 0.0
            prev2 = float(series[editions[i - 2]]) if i >= 2 else 0.0
            prev3 = float(series[editions[i - 3]]) if i >= 3 else 0.0
            # only model countries with some recent presence
            if i < 1 or (prev1 == 0 and prev2 == 0 and prev3 == 0):
                continue
            rows.append(
                {
                    "year": y, "noc": noc,
                    "prev1": prev1, "prev2": prev2, "prev3": prev3,
                    "avg3": (prev1 + prev2 + prev3) / 3.0,
                    "host": 1.0 if hosts.get(y) == noc else 0.0,
                    "total": float(series[y]),
                }
            )
    return pd.DataFrame(rows)


def train_model(panel: pd.DataFrame, exclude_year: int | None = None) -> GradientBoostingRegressor:
    train = panel if exclude_year is None else panel[panel["year"] != exclude_year]
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=7)
    model.fit(train[FEATURES], train["total"])
    return model


def _feature_row_for_projection(pivot_row: pd.Series, editions: list[int], host: bool) -> dict[str, float]:
    last3 = [float(pivot_row[editions[-k]]) for k in range(1, 4) if len(editions) >= k]
    while len(last3) < 3:
        last3.append(0.0)
    prev1, prev2, prev3 = last3[0], last3[1], last3[2]
    return {
        "prev1": prev1, "prev2": prev2, "prev3": prev3,
        "avg3": (prev1 + prev2 + prev3) / 3.0,
        "host": 1.0 if host else 0.0,
    }


def project_next_games(medal_table: pd.DataFrame, host_noc: str, hosts: dict[int, str] | None = None) -> pd.DataFrame:
    """Project the next Games' medal totals per NOC. Returns noc, projected."""
    panel = build_panel(medal_table, hosts)
    model = train_model(panel)
    editions = sorted(medal_table["year"].unique())
    pivot = (
        medal_table.pivot_table(index="noc", columns="year", values="total", aggfunc="sum", fill_value=0)
        .reindex(columns=editions, fill_value=0)
    )

    out = []
    for noc, row in pivot.iterrows():
        feats = _feature_row_for_projection(row, editions, host=(noc == host_noc))
        if feats["prev1"] == 0 and feats["prev2"] == 0 and feats["prev3"] == 0:
            continue  # not recently active
        pred = float(model.predict(pd.DataFrame([feats])[FEATURES])[0])
        out.append({"noc": noc, "projected": max(0.0, pred)})
    return pd.DataFrame(out).sort_values("projected", ascending=False).reset_index(drop=True)


def backtest_last_games(medal_table: pd.DataFrame, hosts: dict[int, str] | None = None) -> dict:
    """Hold out the most recent edition, predict it, compare to actual."""
    editions = sorted(medal_table["year"].unique())
    holdout = editions[-1]
    panel = build_panel(medal_table, hosts)
    model = train_model(panel, exclude_year=holdout)

    test = panel[panel["year"] == holdout].copy()
    if test.empty:
        return {"year": holdout, "rows": []}
    test["predicted"] = np.maximum(0.0, model.predict(test[FEATURES]))
    test = test.sort_values("predicted", ascending=False)
    rows = [
        {"noc": r.noc, "predicted": float(r.predicted), "actual": float(r.total)}
        for r in test.itertuples(index=False)
    ]
    return {"year": int(holdout), "rows": rows}
