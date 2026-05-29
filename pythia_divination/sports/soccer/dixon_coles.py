"""Dixon-Coles bivariate-Poisson model for soccer 1X2 prediction.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market."

Each team has an attack strength and a defense strength; the home side gets a
multiplicative home advantage. Goals are Poisson, with a low-score correlation
correction (rho) that fixes the independent-Poisson under-estimation of draws.
Recent matches are up-weighted with exponential time decay.

    lambda_home = exp(home_adv + attack[home] - defense[away])
    lambda_away = exp(           attack[away] - defense[home])

The fitted score matrix is collapsed to P(home win) / P(draw) / P(away win).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def _dc_tau(home_goals: np.ndarray, away_goals: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles low-score dependence adjustment tau(x, y)."""
    tau = np.ones_like(lam, dtype=float)
    h0 = home_goals == 0
    h1 = home_goals == 1
    a0 = away_goals == 0
    a1 = away_goals == 1
    tau = np.where(h0 & a0, 1.0 - lam * mu * rho, tau)
    tau = np.where(h0 & a1, 1.0 + lam * rho, tau)
    tau = np.where(h1 & a0, 1.0 + mu * rho, tau)
    tau = np.where(h1 & a1, 1.0 - rho, tau)
    return tau


class DixonColesModel:
    def __init__(self) -> None:
        self.teams: list[str] = []
        self._index: dict[str, int] = {}
        self.attack: np.ndarray = np.zeros(0)
        self.defense: np.ndarray = np.zeros(0)
        self.home_adv: float = 0.25
        self.rho: float = 0.0
        self.fitted: bool = False

    # --- fitting -------------------------------------------------------------
    def fit(
        self,
        home_teams: Iterable[str],
        away_teams: Iterable[str],
        home_goals: Iterable[int],
        away_goals: Iterable[int],
        weights: Iterable[float] | None = None,
        l2: float = 1e-3,
    ) -> "DixonColesModel":
        home = np.asarray(list(home_teams), dtype=object)
        away = np.asarray(list(away_teams), dtype=object)
        hg = np.asarray(list(home_goals), dtype=float)
        ag = np.asarray(list(away_goals), dtype=float)
        if not (len(home) == len(away) == len(hg) == len(ag)):
            raise ValueError("home/away/goals arrays must be the same length")
        if len(home) == 0:
            raise ValueError("cannot fit Dixon-Coles on an empty match set")

        w = np.ones(len(home)) if weights is None else np.asarray(list(weights), dtype=float)

        self.teams = sorted(set(home.tolist()) | set(away.tolist()))
        self._index = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        hi = np.array([self._index[t] for t in home])
        ai = np.array([self._index[t] for t in away])

        # param vector: [attack(n), defense(n), home_adv, rho]
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [0.0]])
        bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 2.0), (-0.2, 0.2)]

        def neg_log_likelihood(params: np.ndarray) -> float:
            attack = params[:n]
            defense = params[n : 2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]
            lam = np.exp(home_adv + attack[hi] - defense[ai])
            mu = np.exp(attack[ai] - defense[hi])
            tau = _dc_tau(hg, ag, lam, mu, rho)
            tau = np.clip(tau, 1e-10, None)
            ll = np.log(tau) + poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
            penalty = l2 * (np.sum(attack ** 2) + np.sum(defense ** 2))
            return -float(np.sum(w * ll)) + penalty

        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )
        params = result.x
        attack = params[:n]
        # Identifiability: center attack at 0, push the level into home_adv-free
        # scoring by folding the mean into defense symmetrically.
        mean_attack = float(np.mean(attack))
        self.attack = attack - mean_attack
        self.defense = params[n : 2 * n] + mean_attack
        self.home_adv = float(params[2 * n])
        self.rho = float(params[2 * n + 1])
        self.fitted = True
        return self

    # --- prediction ----------------------------------------------------------
    def _rates(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        ah = self.attack[self._index[home]] if home in self._index else 0.0
        dh = self.defense[self._index[home]] if home in self._index else 0.0
        aa = self.attack[self._index[away]] if away in self._index else 0.0
        da = self.defense[self._index[away]] if away in self._index else 0.0
        # On neutral ground (e.g. a World Cup venue) neither side gets the home
        # edge; split it so it cancels rather than favouring the nominal "home".
        home_adv = 0.0 if neutral else self.home_adv
        lam = float(np.exp(home_adv + ah - da))
        mu = float(np.exp(aa - dh))
        return lam, mu

    def predict_match(self, home: str, away: str, max_goals: int = 10, neutral: bool = False) -> dict[str, float]:
        """Return 1X2 probabilities + expected goals for a fixture."""
        if not self.fitted:
            raise RuntimeError("model is not fitted")
        lam, mu = self._rates(home, away, neutral=neutral)
        goals = np.arange(0, max_goals + 1)
        home_pmf = poisson.pmf(goals, lam)
        away_pmf = poisson.pmf(goals, mu)
        matrix = np.outer(home_pmf, away_pmf)

        # Apply DC correction to the four low-score cells.
        for x in (0, 1):
            for y in (0, 1):
                matrix[x, y] *= _dc_tau(
                    np.array([x]), np.array([y]), np.array([lam]), np.array([mu]), self.rho
                )[0]
        total = matrix.sum()
        if total <= 0:
            return {"homeWin": 1 / 3, "draw": 1 / 3, "awayWin": 1 / 3, "lambdaHome": lam, "lambdaAway": mu}
        matrix /= total

        home_win = float(np.tril(matrix, -1).sum())  # home_goals > away_goals
        away_win = float(np.triu(matrix, 1).sum())   # away_goals > home_goals
        draw = float(np.trace(matrix))
        return {
            "homeWin": home_win,
            "draw": draw,
            "awayWin": away_win,
            "lambdaHome": lam,
            "lambdaAway": mu,
        }

    def knows_team(self, name: str) -> bool:
        return name in self._index

    # --- serialization -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "teams": self.teams,
            "attack": self.attack.tolist(),
            "defense": self.defense.tolist(),
            "home_adv": self.home_adv,
            "rho": self.rho,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DixonColesModel":
        model = cls()
        model.teams = list(data["teams"])
        model._index = {t: i for i, t in enumerate(model.teams)}
        model.attack = np.asarray(data["attack"], dtype=float)
        model.defense = np.asarray(data["defense"], dtype=float)
        model.home_adv = float(data["home_adv"])
        model.rho = float(data["rho"])
        model.fitted = bool(data.get("fitted", True))
        return model
