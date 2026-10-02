"""Checks for the WTA results source built from ESPN's scoreboard (sports/wta/espn_results.py)
and the identity / baseline changes that came with it.

Run (cwd = pythia_divination):  python tests/test_tennis_espn_results.py
No network: every ESPN payload here is synthetic.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sports.wta import elo as E  # noqa: E402
from sports.wta import espn_results as R  # noqa: E402
from sports.wta.live_fields import parse_scoreboard_events  # noqa: E402

PARAMS = E.load_params()


# --------------------------------------------------------------------------- fixtures


def _athlete(name, pid, seed=None, winner=False, sets=()):
    competitor = {"id": str(pid), "winner": winner, "athlete": {"displayName": name},
                  "linescores": [{"value": float(v), "winner": w} for v, w in sets]}
    if seed:
        competitor["curatedRank"] = {"current": seed}
    return competitor


def _comp(rnd, date_key, a, b, *, status="STATUS_FINAL", completed=True, cid="1"):
    day = f"{str(date_key)[:4]}-{str(date_key)[4:6]}-{str(date_key)[6:]}T10:00Z"
    return {"id": cid, "date": day, "round": {"displayName": rnd},
            "status": {"type": {"name": status, "completed": completed, "state": "post" if completed else "pre"}},
            "competitors": [a, b]}


def _event(event_id, name, date_key, comps, gender="Women's"):
    day = f"{str(date_key)[:4]}-{str(date_key)[4:6]}-{str(date_key)[6:]}T00:00Z"
    return {"id": event_id, "name": name, "date": day, "venue": {"displayName": "Somewhere"},
            "groupings": [{"grouping": {"displayName": f"{gender} Singles"}, "competitions": comps}]}


def _knockout(event_id, name, start, players, *, seeds=None, round_dates=None):
    """Complete 2^k knockout where the earlier-listed player always wins
    (round r is played on ``round_dates[r]``, default start + r)."""
    seeds = seeds or {}
    comps, cid = [], 0
    alive = list(players)
    labels = {2: "Final", 4: "Semifinal", 8: "Quarterfinal"}
    total_rounds = int(np.log2(len(players)))
    round_dates = list(round_dates or [start + r for r in range(total_rounds)])
    round_number = 1
    while len(alive) > 1:
        day = round_dates[round_number - 1]
        label = labels.get(len(alive), f"Round {round_number}")
        nxt = []
        for i in range(0, len(alive), 2):
            a, b = alive[i], alive[i + 1]
            cid += 1
            comps.append(_comp(label, day, _athlete(a, 1000 + players.index(a), seeds.get(a), True, [(6, True), (6, True)]),
                               _athlete(b, 1000 + players.index(b), seeds.get(b), False, [(3, False), (4, False)]), cid=str(cid)))
            nxt.append(a)
        alive = nxt
        round_number += 1
    assert round_number - 1 == total_rounds
    return _event(event_id, name, start, comps)


def _history_rows(tid, name, date_key, players, ids, *, level="WTA250", surface="Hard", short=False):
    rows = []
    alive = list(zip(players, ids))
    num = 0
    while len(alive) > 1:
        nxt = []
        for i in range(0, len(alive), 2):
            (wn, wi), (ln, li) = alive[i], alive[i + 1]
            num += 1
            rows.append({"tourney_id": tid, "tourney_name": name, "tourney_level": level, "surface": surface,
                         "draw_size": len(players), "tourney_date": date_key, "match_num": num,
                         "winner_id": wi, "winner_name": wn, "loser_id": li, "loser_name": ln, "tour": "WTA"})
            nxt.append(alive[i])
        alive = nxt
    return rows


# --------------------------------------------------------------------------- parsing


def test_round_codes():
    slam = R.round_codes(["Round 1", "Round 2", "Round 3", "Round 4", "Quarterfinal", "Semifinal", "Final"])
    assert slam["Round 1"] == "R128" and slam["Round 4"] == "R16" and slam["Final"] == "F"
    small = R.round_codes(["Round 1", "Round 2", "Quarterfinal", "Semifinal", "Final", "Round Robin"])
    assert small["Round 1"] == "R32" and small["Round 2"] == "R16" and small["Round Robin"] == "RR"
    assert "Qualifying Final" not in R.round_codes(["Qualifying Final"])


def test_parse_event_matches():
    comps = [
        _comp("Qualifying 1st Round", 20260101, _athlete("Q A", 1, winner=True), _athlete("Q B", 2)),
        _comp("Round 1", 20260102, _athlete("Ana  One", 3, seed=1, winner=True, sets=[(6, True), (7, True)]),
              _athlete("Bea Two", 4, sets=[(4, False), (6, False)])),
        _comp("Round 1", 20260102, _athlete("Cara Three", 5), _athlete("Dee Four", 6, winner=True), status="STATUS_WALKOVER"),
        _comp("Round 1", 20260102, _athlete("Eve Five", 7, winner=True, sets=[(6, True), (2, False)]),
              _athlete("Fay Six", 8, sets=[(1, False), (3, True)]), status="STATUS_RETIRED"),
        _comp("Round 1", 20260103, _athlete("Gia Seven", 9), _athlete("Hal Eight", 10), completed=False),
        _comp("Final", 20260104, _athlete("Ana One", 3), _athlete("Dee Four", 6)),  # no winner flag
    ]
    event = _event("1-2026", "Test Open presented by X", 20260101, comps)
    event["groupings"].append({"grouping": {"displayName": "Men's Singles"},
                               "competitions": [_comp("Round 1", 20260102, _athlete("Man A", 11, winner=True), _athlete("Man B", 12))]})
    event["groupings"].append({"grouping": {"displayName": "Women's Doubles"},
                               "competitions": [_comp("Round 1", 20260102, _athlete("Dbl A", 13, winner=True), _athlete("Dbl B", 14))]})
    matches = R.parse_event_matches(event, "WTA")
    assert [m["winner_name"] for m in matches] == ["Ana One", "Dee Four", "Eve Five"], matches
    assert all(m["round"] == "R32" or m["round"] == "R16" or m["round"] == "R64" for m in matches)
    assert matches[0]["winner_seed"] == 1 and np.isnan(matches[0]["loser_seed"])
    assert matches[0]["score"] == "6-4 7-6" and matches[1]["score"] == "W/O" and matches[2]["score"].endswith("RET")
    assert E.is_walkover(matches[1]["score"]) and not E.is_walkover(matches[2]["score"])
    assert R.parse_event_matches(event, "ATP")[0]["winner_name"] == "Man A"


def test_live_draw_seeds():
    payload = {"events": [_event("2-2026", "Seeded Open", 20261005, [
        _comp("Round 1", 20261005, _athlete("Top Seed", 1, seed=1), _athlete("No Seed", 2), completed=False),
        _comp("Round 1", 20261005, _athlete("Second Seed", 3, seed=2), _athlete("Other", 4), completed=False),
    ])]}
    event = parse_scoreboard_events(payload, "WTA")[0]
    assert event["seeds"] == {"Top Seed": 1, "Second Seed": 2}


# --------------------------------------------------------------------------- identity


def test_player_resolver_tiers():
    prior = pd.DataFrame([
        {"tour": "WTA", "tourney_date": 20240105, "winner_id": "1", "winner_name": "Elena Rybakina", "loser_id": "2", "loser_name": "Shuai Zhang"},
        {"tour": "WTA", "tourney_date": 20240105, "winner_id": "3", "winner_name": "Caty Mcnally", "loser_id": "4", "loser_name": "Old Timer"},
        {"tour": "WTA", "tourney_date": 20250105, "winner_id": "5", "winner_name": "Rakotomanga Rajaonah T.", "loser_id": "3", "loser_name": "McNally C."},
        {"tour": "WTA", "tourney_date": 20250106, "winner_id": "6", "winner_name": "Maria Sakkari", "loser_id": "7", "loser_name": "Mirra Andreeva"},
        {"tour": "WTA", "tourney_date": 20250107, "winner_id": "8", "winner_name": "Erika Andreeva", "loser_id": "9", "loser_name": "Lin Zhu"},
        {"tour": "WTA", "tourney_date": 20200107, "winner_id": "10", "winner_name": "Retired Player", "loser_id": "9", "loser_name": "Lin Zhu"},
        {"tour": "ATP", "tourney_date": 20250107, "winner_id": "11", "winner_name": "Some Man", "loser_id": "12", "loser_name": "Other Man"},
    ])
    r = R.PlayerResolver(prior, "WTA")
    assert r.resolve("Elena  Rybakina") == ("1", "Elena Rybakina")
    assert r.resolve("Zhang Shuai") == ("2", "Shuai Zhang")  # surname-first spelling
    assert r.resolve("Catherine McNally") == ("3", "Caty Mcnally")  # same surname, shared first-name prefix
    assert r.resolve("Tiantsoa Rakotomanga Rajaonah") == ("5", "Tiantsoa Rakotomanga Rajaonah")  # multi-word surname
    assert r.resolve("Mira Andreeva") == ("7", "Mirra Andreeva")
    assert r.resolve("Maria Andreeva") is None  # "mar" matches neither Mirra nor Erika: newcomer
    assert r.resolve("Retiring Player") is None  # only candidate inactive for 2+ seasons
    assert r.resolve("Some Man") is None  # other tour
    assert r.resolve("Brand New") is None


def test_repair_takes_a_dominant_namesake():
    rows = []
    for i in range(12):  # Leylah: a regular; Lya: one match
        rows.append({"tourney_id": f"2024-{i}", "tourney_name": "E", "tourney_level": "I", "tourney_date": 20240105 + i, "match_num": 1,
                     "round": "R32", "surface": "Hard", "winner_id": "220367", "winner_name": "Leylah Fernandez",
                     "loser_id": str(500 + i), "loser_name": f"Opp {i}", "score": "6-1 6-1", "tour": "WTA"})
    rows.append({"tourney_id": "2023-x", "tourney_name": "E", "tourney_level": "I", "tourney_date": 20230918, "match_num": 1, "round": "R32",
                 "surface": "Hard", "winner_id": "248660", "winner_name": "Lya Fernandez", "loser_id": "600", "loser_name": "Opp X",
                 "score": "6-1 6-1", "tour": "WTA"})
    rows.append({"tourney_id": "2025-1", "tourney_name": "E", "tourney_level": "WTA250", "tourney_date": 20250301, "match_num": 1,
                 "round": "", "surface": "Hard", "winner_id": "200420", "winner_name": "Fernandez L.A.", "loser_id": "700",
                 "loser_name": "Somebody Else", "score": "", "tour": "WTA"})
    keyed = E.assign_player_keys(pd.DataFrame(rows))
    assert keyed.iloc[-1]["winner_key"] == "WTA:220367", keyed.iloc[-1]["winner_key"]


# --------------------------------------------------------------------------- tour level + season


_LETTERS = "abcdefghijklmnopqrstuvwxyz"
PLAYERS_W = [f"Fi{_LETTERS[i // 26]}{_LETTERS[i % 26]}na La{_LETTERS[i % 26]}{_LETTERS[i // 26]}st" for i in range(64)]
IDS_W = [str(10_000 + i) for i in range(64)]


def _prior():
    rows = []
    # 2025: a 64-player slam (players 0-63) and, the week before, a 16-player warm-up
    # whose field is a subset of the slam's: the crosswalk must not swap them.
    rows += _history_rows("2025-SLAM", "Grass Slam", 20250630, PLAYERS_W, IDS_W, level="G", surface="Grass")
    rows += _history_rows("2025-WARM", "Warm-up Open", 20250622, PLAYERS_W[:16], IDS_W[:16], surface="Grass")
    return pd.DataFrame(rows)


def test_crosswalk_and_season_frame():
    prior = _prior()
    editions = R.prior_editions(prior, "WTA", 2025)
    assert set(editions["tourney_name"]) == {"Grass Slam", "Warm-up Open"}
    espn_2025 = [
        _knockout("188-2025", "Wimbledon-like", 20250630, PLAYERS_W),
        _knockout("636-2025", "Warm-up Open powered by Sponsor", 20250621, PLAYERS_W[:16]),
        _knockout("999-2025", "Some 125", 20250622, [f"Ot{_LETTERS[i]}her Pe{_LETTERS[i]}rson" for i in range(16)]),
    ]
    crosswalk = R.tour_level_crosswalk(R.event_profiles(espn_2025, "WTA", 2025), editions)
    assert crosswalk["188"]["tourney_name"] == "Grass Slam" and crosswalk["188"]["tourney_level"] == "G"
    assert crosswalk["636"]["tourney_name"] == "Warm-up Open"
    assert "999" not in crosswalk

    espn_2026 = [
        _knockout("188-2026", "Wimbledon-like 2026", 20260629, PLAYERS_W, seeds={PLAYERS_W[0]: 1, PLAYERS_W[1]: 2}),
        _knockout("999-2026", "Some 125", 20260622, [f"Ot{_LETTERS[i]}her Pe{_LETTERS[i]}rson" for i in range(16)]),
        # starts 29 Dec 2025, final in January: belongs to 2026
        _knockout("636-2026", "Warm-up Open", 20251229, PLAYERS_W[:4], round_dates=[20251229, 20260102]),
        # an edition entirely inside 2025 belongs to the 2025 season, not this one
        _knockout("636-2025b", "Warm-up Open", 20251201, PLAYERS_W[:4]),
    ]
    profiles = R.event_profiles(espn_2026, "WTA", 2026)
    frame = R.build_season_frame(profiles, crosswalk, prior, tour="WTA", year=2026)
    assert set(frame["tourney_name"]) == {"Grass Slam", "Warm-up Open"}
    slam = frame[frame["tourney_name"] == "Grass Slam"]
    assert len(slam) == 63 and set(slam["tourney_id"]) == {"2026-ESPN-188"} and set(slam["surface"]) == {"Grass"}
    assert slam["tourney_date"].nunique() == 1 and int(slam["tourney_date"].iloc[0]) == 20260629
    assert slam["winner_id"].astype(str).str.isdigit().all()  # every player resolved to her id
    assert slam.loc[slam["round"] == "F", "winner_name"].iloc[0] == PLAYERS_W[0]
    assert slam.loc[slam["winner_name"] == PLAYERS_W[0], "winner_seed"].iloc[0] == 1
    warm = frame[frame["tourney_name"] == "Warm-up Open"]
    assert int(warm["tourney_date"].min()) == 20251229  # the season of its final, dated by its first match
    assert list(frame.columns) == R.OUTPUT_COLUMNS

    # Round trip: the new rows key onto the existing ids and become one event with a champion.
    from sports.wta.feature_engineering import build_player_event_features, process_match_history

    combined = pd.concat([prior, frame], ignore_index=True)
    history = process_match_history(combined, PARAMS)
    assert set(history.loc[history["tourney_id"] == "2026-ESPN-188", "winner_key"]) <= {f"WTA:{i}" for i in IDS_W}
    events = build_player_event_features(history)
    slam_event = events[events["tournament_id"] == "WTA:2026-ESPN-188"]
    assert len(slam_event) == 64 and slam_event["has_champion"].all()
    assert slam_event.loc[slam_event["won_tournament"] == 1, "player_name"].tolist() == [PLAYERS_W[0]]
    assert slam_event["seed"].notna().sum() == 2


def test_season_matches_paths():
    prior = _prior()
    calls = []

    def failing_fetch(tour, year, today=None):
        calls.append(year)
        return None

    assert R.season_matches("WTA", 2026, prior, fetch=failing_fetch) is None and calls == [2025]

    # An earlier ESPN-sourced season carries espn_event_id: no second season is fetched.
    espn_rows = prior[prior["tourney_id"] == "2025-SLAM"].copy()
    espn_rows["espn_event_id"] = "188-2025"
    espn_rows["espn_event_name"] = "Wimbledon-like"
    calls.clear()

    def fetch_2026(tour, year, today=None):
        calls.append(year)
        return [_knockout("188-2026", "Wimbledon-like", 20260629, PLAYERS_W)]

    frame = R.season_matches("WTA", 2026, pd.concat([prior[prior["tourney_id"] != "2025-SLAM"], espn_rows]), fetch=fetch_2026)
    assert calls == [2026] and len(frame) == 63 and set(frame["tourney_name"]) == {"Grass Slam"}


def test_ingest_writes_espn_season_only_on_success():
    from sports.wta import espn_results
    from sports.wta import ingest as I

    class Response:
        status_code, text, content = 404, "404", b"404"

        def raise_for_status(self):
            raise RuntimeError(404)

    original = (I.requests.get, I.DEFAULT_WTA_DATA_ROOT, espn_results.season_matches)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        raw.mkdir()
        _prior().to_csv(raw / "wta_matches_2025.csv", index=False)
        I.requests.get = lambda url, timeout=30: Response()
        I.DEFAULT_WTA_DATA_ROOT = Path(tmp)
        try:
            espn_results.season_matches = lambda tour, year, prior, **kw: None  # ESPN down
            frame = I.ingest_matches([2025, 2026], tour="wta", force=True)
            assert not (raw / "wta_matches_2026.csv").exists() and frame["tourney_date"].max() < 20260101

            built = pd.DataFrame([{**{c: np.nan for c in R.OUTPUT_COLUMNS}, "tourney_id": "2026-ESPN-1", "tourney_name": "X",
                                   "tourney_date": 20260105, "winner_id": "10000", "winner_name": PLAYERS_W[0],
                                   "loser_id": "10001", "loser_name": PLAYERS_W[1], "tour": "WTA", "data_source": R.SOURCE_LABEL}])
            seen = {}

            def fake(tour, year, prior, **kw):
                seen["prior_years"] = sorted(set(pd.to_numeric(prior["tourney_date"]) // 10000))
                return built

            espn_results.season_matches = fake
            frame = I.ingest_matches([2025, 2026], tour="wta", force=True)
            assert (raw / "wta_matches_2026.csv").exists() and seen["prior_years"] == [2025]
            assert frame["tourney_date"].max() == 20260105

            espn_results.season_matches = lambda tour, year, prior, **kw: None  # down again: last file kept
            I.ingest_matches([2025, 2026], tour="wta", force=True)
            assert "2026-ESPN-1" in (raw / "wta_matches_2026.csv").read_text()
        finally:
            I.requests.get, I.DEFAULT_WTA_DATA_ROOT, espn_results.season_matches = original


# --------------------------------------------------------------------------- boards


def test_ranking_favourite_falls_back_to_top_seed():
    from scripts.export_wta_frontend_data import ranking_favourite

    ranked = pd.DataFrame({"player_name": ["A", "B", "C"], "rank": [12.0, 3.0, np.nan], "seed": [1.0, np.nan, np.nan]})
    assert ranking_favourite(ranked) == ("B", "ranking")
    seeded = pd.DataFrame({"player_name": ["A", "B", "C"], "rank": [np.nan] * 3, "seed": [2.0, 1.0, np.nan]})
    assert ranking_favourite(seeded) == ("B", "top_seed")
    assert ranking_favourite(pd.DataFrame({"player_name": ["A"], "rank": [np.nan], "seed": [np.nan]})) == (None, None)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("test_tennis_espn_results: all checks passed")
