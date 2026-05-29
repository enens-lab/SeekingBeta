"""2026 FIFA World Cup: fixtures, an international Dixon-Coles model, and a
Monte Carlo tournament simulator.

Format: 48 teams, 12 groups (A-L) of 4, top 2 + 8 best third-placed teams ->
Round of 32 -> knockout to the final (104 matches, Jun 11 - Jul 19 2026).

The simulator drives two product surfaces:
  * group-advancement probabilities (P reach knockout)
  * title odds (P win the World Cup) -> a ranked "field" board

Knockout note: the exact 2026 R32 seeding table (which third-place team lands
in which slot) is intricate; this simulator uses random re-pairing among the
32 qualifiers each round. That captures team-strength -> title odds well while
slightly understating the easier-bracket edge of finishing top of a group.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import requests

from .constants import DEFAULT_TIMEOUT_SECONDS, DEFAULT_USER_AGENT, canonical_national_name
from .dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)

WORLD_CUP_2026_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
TOUR_NAME = "FIFA World Cup"


# --- fixtures / groups -------------------------------------------------------
def load_world_cup_groups(url: str = WORLD_CUP_2026_URL) -> dict[str, list[str]]:
    """Return {group_label: [canonical national names]} for the group stage."""
    resp = requests.get(url, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=DEFAULT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()

    groups: dict[str, set[str]] = {}
    for match in data.get("matches", []) or []:
        group = match.get("group")
        if not group:
            continue
        for key in ("team1", "team2"):
            team = match.get(key)
            # team may be a string or {"name": ...}
            if isinstance(team, dict):
                team = team.get("name") or team.get("code")
            if not isinstance(team, str):
                continue
            # skip knockout placeholders ("Winner Group A", "1A", etc.)
            low = team.lower()
            if "winner" in low or "runner" in low or team.strip()[:1].isdigit():
                continue
            groups.setdefault(group, set()).add(canonical_national_name(team))

    # keep only real 4-team groups
    return {g: sorted(t) for g, t in sorted(groups.items()) if len(t) >= 3}


# --- international model ------------------------------------------------------
def build_international_model(
    results, since_year: int = 2015, min_matches: int = 25, decay_per_day: float = 0.0010
) -> DixonColesModel:
    """Fit Dixon-Coles on recent internationals, filtered to established teams."""
    df = results[results["date"].dt.year >= since_year].copy()
    counts = df["home"].value_counts().add(df["away"].value_counts(), fill_value=0)
    established = set(counts[counts >= min_matches].index)
    df = df[df["home"].isin(established) & df["away"].isin(established)]
    if df.empty:
        raise ValueError("no international matches after filtering")

    latest = df["date"].max()
    weights = np.exp(-decay_per_day * (latest - df["date"]).dt.days.clip(lower=0).to_numpy(dtype=float))
    return DixonColesModel().fit(
        df["home"], df["away"], df["home_goals"].astype(int), df["away_goals"].astype(int),
        weights=weights, l2=5e-3,
    )


# --- Monte Carlo simulator ---------------------------------------------------
def _win_prob_matrix(model: DixonColesModel, teams: list[str]) -> np.ndarray:
    """p[i, j] = P(team i beats team j) on neutral ground (draw split 50/50)."""
    n = len(teams)
    p = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pred = model.predict_match(teams[i], teams[j], neutral=True)
            p[i, j] = pred["homeWin"] + 0.5 * pred["draw"]
    return p


def simulate_tournament(
    model: DixonColesModel, groups: dict[str, list[str]], n_sims: int = 8000, seed: int = 12345
) -> dict[str, dict[str, float]]:
    """Monte Carlo the group stage + knockout. Returns per-team probabilities:
    {team: {advance, reach_final, win_title}}.
    """
    rng = np.random.default_rng(seed)
    teams: list[str] = []
    for members in groups.values():
        teams.extend(members)
    idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # neutral expected goals for every group pairing, pre-sampled across sims
    group_pairs: list[tuple[str, str]] = []
    for members in groups.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                group_pairs.append((members[a], members[b]))

    points = np.zeros((n_teams, n_sims))
    gd = np.zeros((n_teams, n_sims))
    gf = np.zeros((n_teams, n_sims))
    for home, away in group_pairs:
        lam, mu = model._rates(home, away, neutral=True)  # neutral group match
        hg = rng.poisson(lam, n_sims)
        ag = rng.poisson(mu, n_sims)
        hi, ai = idx[home], idx[away]
        home_win = hg > ag
        away_win = ag > hg
        draw = hg == ag
        points[hi] += np.where(home_win, 3, np.where(draw, 1, 0))
        points[ai] += np.where(away_win, 3, np.where(draw, 1, 0))
        gd[hi] += hg - ag
        gd[ai] += ag - hg
        gf[hi] += hg
        gf[ai] += ag

    # ranking score: points dominate, then GD, then GF (lexicographic via scale)
    score = points * 1e6 + (gd + 100) * 1e3 + gf

    advance = np.zeros(n_teams)
    reach_final = np.zeros(n_teams)
    win_title = np.zeros(n_teams)

    win_p = _win_prob_matrix(model, teams)
    group_members_idx = [[idx[t] for t in members] for members in groups.values()]

    for s in range(n_sims):
        third_place: list[tuple[float, int]] = []
        qualifiers: list[int] = []
        for members in group_members_idx:
            ranked = sorted(members, key=lambda m: score[m, s], reverse=True)
            qualifiers.extend(ranked[:2])  # top two advance
            if len(ranked) >= 3:
                third_place.append((score[ranked[2], s], ranked[2]))
        # 8 best third-placed teams
        third_place.sort(key=lambda x: x[0], reverse=True)
        qualifiers.extend(m for _, m in third_place[:8])

        for m in qualifiers:
            advance[m] += 1

        # knockout: random single-elim re-pairing each round (neutral)
        bracket = qualifiers[:]
        rng.shuffle(bracket)
        # pad to a power of two with byes if needed (32 is already 2^5)
        while len(bracket) > 1:
            winners: list[int] = []
            for k in range(0, len(bracket) - 1, 2):
                a, b = bracket[k], bracket[k + 1]
                winners.append(a if rng.random() < win_p[a, b] else b)
            if len(bracket) % 2 == 1:  # odd -> last team gets a bye
                winners.append(bracket[-1])
            if len(bracket) == 2:
                win_title[winners[0]] += 1
            if len(bracket) <= 4:  # this round produced finalists
                if len(bracket) in (3, 4):
                    for w in winners:
                        reach_final[w] += 1
            bracket = winners

    return {
        teams[i]: {
            "advance": advance[i] / n_sims,
            "reach_final": reach_final[i] / n_sims,
            "win_title": win_title[i] / n_sims,
        }
        for i in range(n_teams)
    }
