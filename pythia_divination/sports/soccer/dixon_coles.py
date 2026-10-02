"""Dixon-Coles bivariate-Poisson model for soccer scores.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market."

Each team has an attack and a defense rating; the league has one goal level
(intercept), one home advantage and a low-score correlation rho. Goals are
Poisson with the Dixon-Coles correction on the 0-0 / 0-1 / 1-0 / 1-1 cells.
Recent matches are up-weighted with exponential time decay (the caller passes
the weights).

    log lambda_home = intercept + home_adv * (not neutral) + attack[home] - defense[away]
    log lambda_away = intercept +                            attack[away] - defense[home]

Attack and defense are shrunk toward 0 (= league average) with an L2 penalty and
re-centred after the fit, so the intercept carries the league's goal level and a
team the fit has never seen is exactly league-average (attack = defense = 0).

History of the 2026-10 fix: the previous version had no intercept and, after the
fit, subtracted the mean attack from attack but ADDED it to defense. That scaled
every expected-goals rate by exp(-2 * mean_attack), a factor that depended on
where the optimizer stopped: walk-forward it predicted 2.55 goals per match vs
2.84 actual and its over/under 2.5 log-loss (0.734) was worse than the base rate
(0.686). Unknown teams were not league-average either. See
sports/soccer/model_params.json for the tuned settings and measured acceptance.

The full score matrix (``score_matrix``) is what the markets are priced from
(sports/markets.py score_matrix_*): 1X2, totals at any line, BTTS, Asian
handicap including quarter lines, draw no bet, double chance, correct score.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

DEFAULT_MAX_GOALS = 10
RHO_BOUNDS = (-0.2, 0.2)


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


def _tau_and_grads(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float):
    """tau (clipped) and d log tau / d log lam, d log mu, d rho for the analytic gradient."""
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau = np.ones_like(lam)
    tau = np.where(m00, 1.0 - lam * mu * rho, tau)
    tau = np.where(m01, 1.0 + lam * rho, tau)
    tau = np.where(m10, 1.0 + mu * rho, tau)
    tau = np.where(m11, 1.0 - rho, tau)
    clipped = tau < 1e-10
    tau_c = np.clip(tau, 1e-10, None)
    zero = np.zeros_like(lam)
    dl = np.where(m00, -lam * mu * rho / tau_c, np.where(m01, lam * rho / tau_c, zero))
    dm = np.where(m00, -lam * mu * rho / tau_c, np.where(m10, mu * rho / tau_c, zero))
    dr = np.where(m00, -lam * mu / tau_c,
                  np.where(m01, lam / tau_c, np.where(m10, mu / tau_c, np.where(m11, -1.0 / tau_c, zero))))
    dl = np.where(clipped, 0.0, dl)
    dm = np.where(clipped, 0.0, dm)
    dr = np.where(clipped, 0.0, dr)
    return tau_c, dl, dm, dr


def score_matrices(lam: np.ndarray, mu: np.ndarray, rho: np.ndarray | float,
                   max_goals: int = DEFAULT_MAX_GOALS) -> np.ndarray:
    """Vectorised normalised score matrices, shape (n, max_goals+1, max_goals+1):
    M[k, i, j] = P(home scores i, away scores j) for match k."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    rho = np.broadcast_to(np.asarray(rho, dtype=float), lam.shape)
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g[None, :], lam[:, None])
    pa = poisson.pmf(g[None, :], mu[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    m[:, 0, 0] *= 1.0 - lam * mu * rho
    m[:, 0, 1] *= 1.0 + lam * rho
    m[:, 1, 0] *= 1.0 + mu * rho
    m[:, 1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)
    total = m.sum(axis=(1, 2), keepdims=True)
    total[total <= 0] = 1.0
    return m / total


class DixonColesModel:
    def __init__(self) -> None:
        self.teams: list[str] = []
        self._index: dict[str, int] = {}
        self.attack: np.ndarray = np.zeros(0)
        self.defense: np.ndarray = np.zeros(0)
        self.intercept: float = 0.0
        self.home_adv: float = 0.25
        self.rho: float = 0.0
        self.fitted: bool = False
        self.converged: bool = False

    # --- fitting -------------------------------------------------------------
    def fit(
        self,
        home_teams: Iterable[str],
        away_teams: Iterable[str],
        home_goals: Iterable[int],
        away_goals: Iterable[int],
        weights: Iterable[float] | None = None,
        l2: float = 1e-3,
        neutral: Iterable[bool] | None = None,
        warm_start: Optional["DixonColesModel"] = None,
    ) -> "DixonColesModel":
        """Weighted maximum likelihood with an L2 penalty on attack/defense.

        ``neutral`` marks matches on neutral ground (no home advantage), so a model
        that is later asked for a neutral-venue prediction was fitted the same way.
        ``warm_start`` seeds the optimizer from a previous fit (walk-forward speed-up)."""
        home = np.asarray(list(home_teams), dtype=object)
        away = np.asarray(list(away_teams), dtype=object)
        hg = np.asarray(list(home_goals), dtype=float)
        ag = np.asarray(list(away_goals), dtype=float)
        if not (len(home) == len(away) == len(hg) == len(ag)):
            raise ValueError("home/away/goals arrays must be the same length")
        if len(home) == 0:
            raise ValueError("cannot fit Dixon-Coles on an empty match set")

        w = np.ones(len(home)) if weights is None else np.asarray(list(weights), dtype=float)
        if neutral is None:
            home_flag = np.ones(len(home))
        else:
            home_flag = 1.0 - np.asarray(list(neutral), dtype=float)

        self.teams = sorted(set(home.tolist()) | set(away.tolist()))
        self._index = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        hi = np.array([self._index[t] for t in home])
        ai = np.array([self._index[t] for t in away])
        log_fact = gammaln(hg + 1) + gammaln(ag + 1)

        # param vector: [attack(n), defense(n), home_adv, rho, intercept]
        def neg_log_likelihood(params: np.ndarray) -> tuple[float, np.ndarray]:
            attack = params[:n]
            defense = params[n: 2 * n]
            h, rho, c = params[2 * n], params[2 * n + 1], params[2 * n + 2]
            log_lam = c + h * home_flag + attack[hi] - defense[ai]
            log_mu = c + attack[ai] - defense[hi]
            lam = np.exp(log_lam)
            mu = np.exp(log_mu)
            tau, dl, dm, dr = _tau_and_grads(hg, ag, lam, mu, rho)
            ll = np.log(tau) + hg * log_lam - lam + ag * log_mu - mu - log_fact
            nll = -float(np.sum(w * ll)) + l2 * float(np.sum(attack ** 2) + np.sum(defense ** 2))
            gl = w * (hg - lam + dl)   # d ll / d log_lam
            gm = w * (ag - mu + dm)    # d ll / d log_mu
            grad = np.empty_like(params)
            grad[:n] = -(np.bincount(hi, gl, n) + np.bincount(ai, gm, n)) + 2.0 * l2 * attack
            grad[n: 2 * n] = (np.bincount(ai, gl, n) + np.bincount(hi, gm, n)) + 2.0 * l2 * defense
            grad[2 * n] = -float(np.sum(gl * home_flag))
            grad[2 * n + 1] = -float(np.sum(w * dr))
            grad[2 * n + 2] = -float(np.sum(gl) + np.sum(gm))
            return nll, grad

        x0 = np.concatenate([np.zeros(2 * n), [0.25, 0.0, 0.3]])
        if warm_start is not None and warm_start.fitted:
            for t, i in self._index.items():
                j = warm_start._index.get(t)
                if j is not None:
                    x0[i] = warm_start.attack[j]
                    x0[n + i] = warm_start.defense[j]
            x0[2 * n: 2 * n + 3] = [warm_start.home_adv, warm_start.rho, warm_start.intercept]
        bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 2.0), RHO_BOUNDS, (-2.0, 2.0)]
        result = minimize(
            neg_log_likelihood,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7},
        )
        params = result.x
        attack = params[:n]
        defense = params[n: 2 * n]
        intercept = float(params[2 * n + 2])
        # Identifiability: shifting every attack by k (or every defense by k) and the
        # intercept by -k (+k) leaves every rate unchanged. Centre both at 0 and fold
        # the shifts into the intercept: exact reparameterisation, rates unchanged,
        # and an unseen team (0, 0) is league-average.
        mean_attack = float(np.mean(attack))
        mean_defense = float(np.mean(defense))
        self.attack = attack - mean_attack
        self.defense = defense - mean_defense
        self.intercept = intercept + mean_attack - mean_defense
        self.home_adv = float(params[2 * n])
        self.rho = float(params[2 * n + 1])
        self.fitted = True
        self.converged = bool(result.success)
        return self

    # --- prediction ----------------------------------------------------------
    def _rates(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        """Expected goals (home, away). Teams the fit has not seen are league-average."""
        ah = self.attack[self._index[home]] if home in self._index else 0.0
        dh = self.defense[self._index[home]] if home in self._index else 0.0
        aa = self.attack[self._index[away]] if away in self._index else 0.0
        da = self.defense[self._index[away]] if away in self._index else 0.0
        home_adv = 0.0 if neutral else self.home_adv
        lam = float(np.exp(self.intercept + home_adv + ah - da))
        mu = float(np.exp(self.intercept + aa - dh))
        return lam, mu

    def rates(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        if not self.fitted:
            raise RuntimeError("model is not fitted")
        return self._rates(home, away, neutral=neutral)

    def score_matrix(self, home: str, away: str, max_goals: int = DEFAULT_MAX_GOALS, neutral: bool = False) -> np.ndarray:
        """Normalised P[home_goals=i, away_goals=j], shape (max_goals+1, max_goals+1)."""
        lam, mu = self.rates(home, away, neutral=neutral)
        return score_matrices(np.array([lam]), np.array([mu]), self.rho, max_goals)[0]

    def predict_match(self, home: str, away: str, max_goals: int = DEFAULT_MAX_GOALS, neutral: bool = False) -> dict[str, float]:
        """Return 1X2 probabilities + expected goals for a fixture."""
        if not self.fitted:
            raise RuntimeError("model is not fitted")
        lam, mu = self._rates(home, away, neutral=neutral)
        matrix = score_matrices(np.array([lam]), np.array([mu]), self.rho, max_goals)[0]
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
            "intercept": self.intercept,
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
        model.intercept = float(data.get("intercept", 0.0))
        model.home_adv = float(data["home_adv"])
        model.rho = float(data["rho"])
        model.fitted = bool(data.get("fitted", True))
        return model
