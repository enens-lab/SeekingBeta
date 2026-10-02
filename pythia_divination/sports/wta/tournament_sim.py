"""Tournament-winner probabilities from match-level Elo by seeded knockout Monte Carlo.

The real draw is usually unknown when a board is published, so each simulation
draws a bracket the way the tours do: the field is padded to the next power of
two with byes, the top ~quarter are seeded (1 and 2 in opposite halves, 3-4 in
the two remaining quarters, 5-8 in the remaining eighths, ...), byes go to the
top seeds, everyone else is placed at random. Matches are then played out with
the Elo win probability. Round-robin finals are approximated as knockouts (they
are under 16 players, so they never enter a published hit rate).
"""

from __future__ import annotations

import zlib
from functools import lru_cache
from typing import Sequence

import numpy as np


def elo_win_matrix(ratings: Sequence[float]) -> np.ndarray:
    """P[i, j] = P(player i beats player j) from Elo ratings."""
    r = np.asarray(ratings, dtype=float)
    return 1.0 / (1.0 + 10.0 ** ((r[None, :] - r[:, None]) / 400.0))


def bracket_size(field_size: int) -> int:
    size = 1
    while size < field_size:
        size *= 2
    return max(size, 2)


def seed_count(field_size: int, seed_fraction: float = 0.25) -> int:
    size = bracket_size(field_size)
    return int(min(field_size, max(2, round(size * seed_fraction))))


@lru_cache(maxsize=64)
def seed_slot_tiers(size: int, seeds: int) -> tuple[tuple[int, ...], ...]:
    """Bracket slots available to each seeding tier: seed 1, seed 2, seeds 3-4, 5-8, ...

    Seeds sit at the outer edge of their section (top edge in the top half,
    bottom edge in the bottom half), so a bye always lands next to a seed.
    """
    tiers: list[tuple[int, ...]] = [(0,)]
    if seeds >= 2 and size >= 2:
        tiers.append((size - 1,))
    occupied = {slot for tier in tiers for slot in tier}
    placed = len(occupied)
    level = 2
    while placed < seeds:
        sections = 2 ** level
        width = size // sections
        if width < 1:
            break
        slots: list[int] = []
        for section in range(sections):
            lo, hi = section * width, (section + 1) * width
            if any(lo <= slot < hi for slot in occupied):
                continue
            slots.append(lo if section < sections // 2 else hi - 1)
        if not slots:
            break
        tiers.append(tuple(slots))
        occupied.update(slots)
        placed += len(slots)
        level += 1
    return tuple(tiers)


def event_rng(key: str) -> np.random.Generator:
    """Deterministic per-event RNG so a board does not wobble between exports."""
    return np.random.default_rng(zlib.crc32(str(key).encode("utf-8")))


def simulate_knockout(
    ratings: Sequence[float],
    *,
    seed_order: Sequence[int] | None = None,
    simulations: int = 4000,
    seed_fraction: float = 0.25,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """P(win the event) for each player, from ``simulations`` random seeded draws.

    ``seed_order`` lists player indices from top seed down (defaults to rating
    order). Returns an array summing to 1 over the field.
    """
    strengths = np.asarray(ratings, dtype=float)
    n = len(strengths)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    rng = rng or np.random.default_rng(0)
    order = np.asarray(seed_order if seed_order is not None else np.argsort(-strengths, kind="mergesort"), dtype=int)
    size = bracket_size(n)
    seeds = seed_count(n, seed_fraction)
    tiers = seed_slot_tiers(size, seeds)
    sims = int(simulations)
    rows = np.arange(sims)[:, None]

    slots = np.full((sims, size), -2, dtype=int)
    seed_slots = np.empty((sims, seeds), dtype=int)
    cursor = 0
    for tier in tiers:
        if cursor >= seeds:
            break
        tier_slots = np.asarray(tier, dtype=int)
        take = min(len(tier_slots), seeds - cursor)
        pick = np.argsort(rng.random((sims, len(tier_slots))), axis=1)[:, :take]
        chosen = tier_slots[pick]
        players = order[cursor:cursor + take]
        slots[rows, chosen] = players[None, :]
        seed_slots[:, cursor:cursor + take] = chosen
        cursor += take

    byes = size - n
    seeded_byes = min(byes, seeds)
    if seeded_byes:
        slots[rows, seed_slots[:, :seeded_byes] ^ 1] = -1
    extra_byes = byes - seeded_byes
    if extra_byes > 0:
        empty_pairs = (slots[:, 0::2] == -2) & (slots[:, 1::2] == -2)
        keys = np.where(empty_pairs, rng.random(empty_pairs.shape), -1.0)
        pairs = np.argsort(-keys, axis=1)[:, :extra_byes]
        slots[rows, pairs * 2] = -1

    unseeded = order[seeds:]
    if len(unseeded):
        shuffled = unseeded[np.argsort(rng.random((sims, len(unseeded))), axis=1)]
        free = slots == -2
        slots[free] = shuffled.ravel()

    bye = n
    matrix = np.zeros((n + 1, n + 1))
    matrix[:n, :n] = elo_win_matrix(strengths)
    matrix[:n, bye] = 1.0
    matrix[bye, :n] = 0.0
    matrix[bye, bye] = 0.5
    current = np.where(slots < 0, bye, slots)
    while current.shape[1] > 1:
        a, b = current[:, 0::2], current[:, 1::2]
        p = matrix[a, b]
        current = np.where(rng.random(p.shape) < p, a, b)
    champions = current[:, 0]
    counts = np.bincount(champions, minlength=n + 1)[:n]
    return counts / float(sims)
