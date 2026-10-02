"""Checks for the tennis Elo, identity, event hygiene, simulation and boards.

Run (cwd = pythia_divination):  python tests/test_tennis_elo.py
Real-data checks run only when data/sports/wta/normalized holds the combined matches.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sports.wta import elo as E  # noqa: E402
from sports.wta.feature_engineering import (  # noqa: E402
    add_rolling_features,
    build_player_event_features,
    event_champions,
    process_match_history,
)
from sports.wta.live_fields import match_espn_event, parse_scoreboard_events  # noqa: E402
from sports.wta.tournament_sim import seed_slot_tiers, simulate_knockout  # noqa: E402

PARAMS = E.load_params()
DATA = Path(__file__).resolve().parents[1] / "data" / "sports" / "wta" / "normalized"


def _match(tid, date_key, w, wid, l, lid, *, tour="ATP", rnd="R32", num=1, level="A", name="Event", score="6-4 6-4", surface="Hard"):
    return {
        "tourney_id": tid, "tourney_name": name, "tourney_level": level, "tourney_date": date_key, "match_num": num,
        "round": rnd, "surface": surface, "winner_id": wid, "winner_name": w, "loser_id": lid, "loser_name": l,
        "score": score, "tour": tour, "winner_rank": np.nan, "loser_rank": np.nan,
    }


def test_identity_survives_id_scheme_change():
    rows = [
        _match("2024-1", 20240105, "Jannik Sinner", "206173", "Carlos Alcaraz", "207989"),
        _match("2025-1", 20250105, "Jannik Sinner", "S0AG", "Carlos Alcaraz", "A0E2"),
        _match("2025-1", 20250106, "Joao Fonseca", "F0FV", "Jannik Sinner", "S0AG", rnd="R16"),
    ]
    keyed = E.assign_player_keys(pd.DataFrame(rows))
    assert keyed.loc[0, "winner_key"] == keyed.loc[1, "winner_key"] == "ATP:206173"
    assert keyed.loc[1, "loser_key"] == "ATP:207989"
    assert keyed.loc[2, "winner_key"] == "ATP:name:joaofonseca"  # never had a legacy id
    # ambiguous name (two legacy ids) never merges into either history
    rows.append(_match("2023-9", 20230101, "Jannik Sinner", "999999", "X Y", "1"))
    keyed = E.assign_player_keys(pd.DataFrame(rows))
    assert keyed.loc[1, "winner_key"] == "ATP:name:janniksinner"


def test_short_name_repair():
    rows = [
        _match("2024-1", 20240105, "Emma Navarro", "1", "Coco Gauff", "2", tour="WTA"),
        _match("2024-2", 20240205, "Yafan Wang", "3", "Yuxuan Wang", "4", tour="WTA"),
        _match("2024-3", 20240305, "Yue Yuan", "5", "Yuan Yue", "6", tour="WTA"),
        _match("2025-1", 20250105, "Navarro E.", "999", "Gauff C.", "2", tour="WTA"),
        _match("2025-2", 20250205, "Wang Y.", "777", "Gauff C.", "2", tour="WTA"),
    ]
    keyed = E.assign_player_keys(pd.DataFrame(rows))
    assert keyed.loc[3, "winner_key"] == "WTA:1", keyed.loc[3, "winner_key"]  # empty converter id -> real history
    assert keyed.loc[3, "loser_key"] == "WTA:2"  # consistent id kept
    assert keyed.loc[4, "winner_key"] == "WTA:777"  # two Wang Y. candidates: left alone
    names = E.display_names(keyed)
    assert names["WTA:1"] == "Emma Navarro"
    assert names["WTA:777"] == "Y. Wang"


def test_k_decays_and_blend():
    state = E.EloState(params=PARAMS)
    assert abs(state.k(0) - PARAMS["k_numerator"] / PARAMS["k_offset"] ** PARAMS["k_shape"]) < 1e-9
    assert state.k(0) > state.k(10) > state.k(100) > 0
    state.update("a", "b", "Clay")
    overall, surface, blend = state.ratings("a", "Clay")
    w = PARAMS["overall_weight"]
    assert abs(blend - (w * overall + (1 - w) * surface)) < 1e-9
    assert state.ratings("a", "Grass")[1] == PARAMS["initial_rating"]  # surface ratings are separate


def test_walk_forward_no_leak():
    rng = np.random.default_rng(3)
    players = [f"P{i}" for i in range(12)]
    rows = []
    for i in range(300):
        a, b = rng.choice(players, 2, replace=False)
        rows.append(_match(f"T{i // 10}", 20200101 + i, a, str(100 + players.index(a)), b, str(100 + players.index(b)), num=i))
    base = E.prepare_matches(pd.DataFrame(rows), PARAMS)
    values, _, _ = E.run_elo(base, PARAMS)
    altered = pd.DataFrame(rows)
    for i in range(200, 300):  # flip every later result
        r = altered.loc[i]
        altered.loc[i, ["winner_id", "winner_name", "loser_id", "loser_name"]] = [r.loser_id, r.loser_name, r.winner_id, r.winner_name]
    values2, _, _ = E.run_elo(E.prepare_matches(altered, PARAMS), PARAMS)
    assert np.allclose(values["p_winner"].values[:200], values2["p_winner"].values[:200])
    # match 200's prediction cannot depend on match 200's own result
    assert abs(values["p_winner"].values[200] - (1 - values2["p_winner"].values[200])) < 1e-12


def test_walkovers_do_not_move_ratings_and_mask():
    rows = [
        _match("T1", 20240101, "A", "1", "B", "2", score="W/O"),
        _match("T1", 20240102, "A", "1", "B", "2", rnd="R16"),
        _match("D1", 20240103, "A", "1", "B", "2", level="D", name="Davis Cup WG1"),
    ]
    prep = E.prepare_matches(pd.DataFrame(rows), PARAMS)
    values, state, _ = E.run_elo(prep, PARAMS)
    assert values["p_winner"].iloc[1] == 0.5  # the walkover did not count
    assert state.matches["ATP:1"] == 2  # real match + Davis Cup rubber
    mask = E.evaluation_mask(prep, 20240101, 20250101)
    assert mask.tolist() == [False, True, False]
    assert prep["is_team_event"].tolist() == [False, False, True]


def test_champion_from_final_not_match_number():
    rows = [
        _match("T1", 20240101, "A", "1", "B", "2", rnd="SF", num=1),
        _match("T1", 20240101, "C", "3", "D", "4", rnd="SF", num=2),
        _match("T1", 20240101, "C", "3", "A", "1", rnd="F", num=3),
        _match("T1", 20240101, "B", "2", "D", "4", rnd="BR", num=9),  # bronze match numbered last
        # no round labels: the one unbeaten player is champion
        _match("T2", 20240201, "A", "1", "B", "2", rnd=""),
        _match("T2", 20240202, "A", "1", "C", "3", rnd=""),
    ]
    history = process_match_history(pd.DataFrame(rows), PARAMS)
    champions = event_champions(history)
    assert champions["ATP:T1"] == "ATP:3"
    assert champions["ATP:T2"] == "ATP:1"


def test_team_events_never_become_boards():
    rows = [
        _match("DC", 20240101, "A", "1", "B", "2", level="D", name="Davis Cup Finals RR: ESP vs SRB", rnd="RR"),
        _match("LC", 20240102, "A", "1", "C", "3", level="A", name="Laver Cup", rnd="RR"),
        _match("UC", 20240103, "A", "1", "D", "4", level="A", name="United Cup", rnd="RR"),
        _match("T1", 20240104, "A", "1", "B", "2", rnd="F"),
    ]
    events = build_player_event_features(process_match_history(pd.DataFrame(rows), PARAMS))
    assert set(events["tournament_id"]) == {"ATP:T1"}
    assert events.loc[events["player_key"] == "ATP:1", "won_tournament"].iloc[0] == 1
    assert (events["field_size"] == 2).all()


def test_seeded_simulation():
    assert seed_slot_tiers(32, 8) == ((0,), (31,), (8, 23), (4, 12, 19, 27))
    tiers = seed_slot_tiers(128, 32)
    slots = [slot for tier in tiers for slot in tier]
    assert len(slots) == len(set(slots)) == 32
    # every quarter holds exactly one of seeds 1-4, every eighth one of seeds 1-8
    assert sorted(slot // 32 for slot in slots[:4]) == [0, 1, 2, 3]
    assert sorted(slot // 16 for slot in slots[:8]) == list(range(8))
    for n in (5, 16, 17, 28, 56, 96, 128):
        ratings = np.linspace(1500, 1800, n)
        ratings[-1] += 250  # a clear favourite, so MC noise cannot reorder the top
        p = simulate_knockout(ratings, rng=np.random.default_rng(1), simulations=2000)
        assert abs(p.sum() - 1) < 1e-9 and (p >= 0).all()
        assert p[-1] == p.max()
    flat = simulate_knockout([1500.0] * 32, rng=np.random.default_rng(2), simulations=4000)
    assert flat.max() - flat.min() < 0.03
    # two equal top seeds can only meet in the final: each wins exactly half the time vs a weak field
    p = simulate_knockout([3000.0, 3000.0] + [1000.0] * 14, rng=np.random.default_rng(4), simulations=4000)
    assert abs(p[0] + p[1] - 1) < 1e-9 and 0.45 < p[0] < 0.55


def test_espn_parsing_and_matching():
    def comp(rnd, a, b, state="pre"):
        return {"round": {"displayName": rnd}, "status": {"type": {"state": state}},
                "competitors": [{"athlete": {"displayName": a}}, {"athlete": {"displayName": b}}]}
    payload = {"events": [{
        "id": "315-2026", "name": "Rolex Shanghai Masters", "date": "2026-10-07T04:00Z", "endDate": "2026-10-19T03:59Z",
        "venue": {"displayName": "Shanghai, China PR"},
        "groupings": [
            {"grouping": {"displayName": "Men's Singles"}, "competitions": [
                comp("Qualifying 1st Round", "Q One", "Q Two", "post"),
                comp("Round 1", "Jannik Sinner", "TBD"),
                comp("Round 1", "Carlos Alcaraz", "Aruzhan  Sagandykova"),
            ]},
            {"grouping": {"displayName": "Men's Doubles"}, "competitions": [comp("Round 1", "D One", "D Two")]},
        ],
    }]}
    events = parse_scoreboard_events(payload, "ATP")
    assert len(events) == 1
    ev = events[0]
    assert ev["entrants"] == ["Jannik Sinner", "Carlos Alcaraz", "Aruzhan Sagandykova"]
    assert ev["openSlots"] == 1 and ev["startKey"] == 20261007 and not ev["started"]
    assert match_espn_event("ATP", "Shanghai Masters", 20261005, events)["espnId"] == "315-2026"
    assert match_espn_event("WTA", "Shanghai Masters", 20261005, events) is None
    assert match_espn_event("ATP", "Shanghai Masters", 20261101, events) is None  # too far apart
    assert match_espn_event("ATP", "Beijing", 20261005, events) is None


def test_entrant_resolver():
    from scripts.export_wta_frontend_data import EntrantResolver

    history = pd.DataFrame(
        {
            "tour": ["ATP", "WTA", "WTA", "WTA"],
            "winner_key": ["ATP:1", "WTA:2", "WTA:3", "WTA:5"],
            "winner_name": ["Zhizhen Zhang", "Tagger L.", "Emma Navarro", "Emma Raducanu"],
            "loser_key": ["ATP:9", "WTA:4", "WTA:4", "WTA:6"],
            "loser_name": ["Some One", "Coco Gauff", "Coco Gauff", "Elena Rybakina"],
        }
    )
    resolver = EntrantResolver({}, history)
    assert resolver.resolve("ATP", "Zhang Zhizhen") == "ATP:1"  # surname-first spelling
    assert resolver.resolve("WTA", "Lilli Tagger") == "WTA:2"  # only known as "Tagger L."
    assert resolver.resolve("WTA", "Coco  Gauff") == "WTA:4"
    assert resolver.resolve("ATP", "Coco Gauff") is None  # other tour
    assert resolver.resolve("WTA", "Erin Navarro") is None  # first name known and different
    assert resolver.resolve("WTA", "Emily Nobody") is None


def test_ingest_current_season_fallback():
    import tempfile

    from sports.wta import ingest as I

    class Response:
        def __init__(self, status, text):
            self.status_code, self.text, self.content = status, text, text.encode()

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    csv_text = "tourney_id,tourney_name,tourney_date,winner_name,loser_name,winner_id,loser_id\n2026-1,X,20260105,A B,C D,A0B1,C0D1\n"
    calls = []

    def fake_get(url, timeout=30):
        calls.append(url)
        if "githubusercontent" in url:
            return Response(404, "404: Not Found")
        if url.endswith("2026.csv"):
            return Response(200, csv_text)
        return Response(200, "<html>Attention Required</html>")

    original_get, original_root = I.requests.get, I.DEFAULT_WTA_DATA_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        I.requests.get = fake_get
        I.DEFAULT_WTA_DATA_ROOT = Path(tmp)
        try:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            (raw / "atp_matches_2024.csv").write_text(csv_text.replace("2026", "2024"))
            (raw / "atp_matches_2025.csv").write_text(csv_text.replace("2026", "2025"))
            frame = I.ingest_matches([2024, 2025, 2026], tour="atp", force=True)
            assert sorted(frame["tourney_date"].astype(int)) == [20240105, 20250105, 20260105]
            assert "2025-1" in (raw / "atp_matches_2025.csv").read_text()  # HTML reply did not overwrite it
            assert not any("tennismylife" in url and "2024" in url for url in calls)  # pre-2025 seasons left alone
            I.ingest_matches([2026], tour="wta", force=True)
            assert not any("tennismylife" in url for url in calls if "wta" in url)
        finally:
            I.requests.get, I.DEFAULT_WTA_DATA_ROOT = original_get, original_root


def _real_tables():
    if not ((DATA / "tennis_matches_combined.parquet").exists() or (DATA / "tennis_matches_combined.csv").exists()):
        print("  (no local tennis data: real-data checks skipped)")
        return None
    from scripts.export_wta_frontend_data import load_tennis_tables

    return load_tennis_tables(PARAMS)


def test_real_data_acceptance():
    tables = _real_tables()
    if tables is None:
        return
    from scripts.export_wta_frontend_data import build_backtests, build_upcoming
    from sports.wta.evaluate_elo import evaluate

    report = evaluate(tables["matches"], PARAMS, start=20230101, end=20260101)
    model = report["all"]["elo"]
    atp25 = report["by_tour_year"]["ATP-2025"]["elo"]
    print("  2023-25 n=%d acc=%.4f ll=%.4f | ATP 2025 ll=%.4f" % (model["n"], model["accuracy"], model["log_loss"], atp25["log_loss"]))
    assert model["accuracy"] >= 0.641 and model["log_loss"] <= 0.6295 and atp25["log_loss"] <= 0.632

    boards, scoring = build_backtests(tables["events"], PARAMS)
    assert boards and len(boards) == len(scoring)
    assert min(board["fieldSize"] for board in boards) >= 16
    for board in boards:
        name = board["tournament"].lower()
        assert not any(team in name for team in ("davis cup", "bjk cup", "billie jean", "united cup", "laver cup", "atp cup"))
        winners = [p for p in board["fullField"] if p["actualWinner"]]
        assert len(winners) == 1 and winners[0]["playerName"] == board["actualWinner"]
        assert board["hitStatus"] in {"Top Pick", "Top 3", "Top 5", "Miss"}
        assert (board["hitStatus"] == "Top Pick") == (board["predictedWinner"] == board["actualWinner"])
        assert abs(sum(p["winProbability"] for p in board["fullField"]) - 100) < 0.5

    upcoming = build_upcoming(tables, PARAMS, today=date(2026, 10, 2), espn_events=[])
    assert upcoming
    for board in upcoming:
        assert board["fieldBasis"] in {"previous_edition", "observed_entrants"}
        assert board["scheduledDate"] and board["latestDate"] >= 20261002
        assert all("actualWinner" not in p for p in board["predictions"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("test_tennis_elo: all checks passed")
