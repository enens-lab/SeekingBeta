from __future__ import annotations

from pathlib import Path

import pandas as pd

from sports.pga.build_training_dataset import _load_local_label_table
from sports.pga.ingest_history import _partition_result_frames
from sports.wta.feature_engineering import build_player_event_features, process_match_history


def _minimal_tennis_match(
    *,
    tour: str,
    tournament_id: str,
    winner_id: str,
    loser_id: str,
    match_num: int = 1,
) -> dict[str, object]:
    return {
        "tour": tour,
        "tourney_id": tournament_id,
        "tourney_name": "Australian Open",
        "surface": "Hard",
        "tourney_level": "G",
        "tourney_date": 20200101,
        "match_num": match_num,
        "winner_id": winner_id,
        "winner_name": f"{tour} Winner",
        "winner_age": 25,
        "winner_ht": 185,
        "loser_id": loser_id,
        "loser_name": f"{tour} Loser",
        "loser_age": 24,
        "loser_ht": 183,
        "w_svpt": 100,
        "w_1stWon": 40,
        "w_2ndWon": 20,
        "l_svpt": 90,
        "l_1stWon": 35,
        "l_2ndWon": 15,
    }


def test_tennis_canonical_tournament_ids_are_namespaced_by_tour() -> None:
    matches = pd.DataFrame(
        [
            _minimal_tennis_match(tour="ATP", tournament_id="2020-580", winner_id="A1", loser_id="A2"),
            _minimal_tennis_match(tour="WTA", tournament_id="2020-580", winner_id="W1", loser_id="W2"),
        ]
    )

    history = process_match_history(matches)
    player_events = build_player_event_features(history)

    assert sorted(player_events["tournament_id"].unique().tolist()) == ["ATP:2020-580", "WTA:2020-580"]
    wins_by_tournament = player_events.groupby("tournament_id")["won_tournament"].sum().to_dict()
    assert wins_by_tournament == {"ATP:2020-580": 1, "WTA:2020-580": 1}


def test_tennis_alphanumeric_player_ids_survive_processing() -> None:
    matches = pd.DataFrame(
        [
            _minimal_tennis_match(tour="ATP", tournament_id="2025-341", winner_id="CD85", loser_id="AB12"),
        ]
    )

    history = process_match_history(matches)
    player_events = build_player_event_features(history)

    assert not history.empty
    assert not player_events.empty
    assert set(player_events["player_key"]) == {"ATP:AB12", "ATP:CD85"}


def test_pga_partition_result_frames_splits_team_events() -> None:
    frame = pd.DataFrame(
        [
            {"tournament_id": "R1", "player_id": "p1", "team_event": False},
            {"tournament_id": "R2", "player_id": "p2", "team_event": True},
            {"tournament_id": "R2", "player_id": "p3", "team_event": True},
        ]
    )

    combined, individual, team = _partition_result_frames(frame)

    assert len(combined) == 3
    assert len(individual) == 1
    assert len(team) == 2
    assert individual["tournament_id"].unique().tolist() == ["R1"]
    assert team["tournament_id"].unique().tolist() == ["R2"]


def test_pga_label_loader_prefers_individual_tables(tmp_path: Path) -> None:
    all_labels = pd.DataFrame(
        [
            {"tournament_id": "R1", "player_id": "p1", "team_event": False},
            {"tournament_id": "R2", "player_id": "p2", "team_event": True},
        ]
    )
    individual_labels = pd.DataFrame(
        [
            {"tournament_id": "R1", "player_id": "p1", "team_event": False},
        ]
    )

    all_labels.to_csv(tmp_path / "season_tournament_labels_2025_latest.csv", index=False)
    individual_labels.to_csv(tmp_path / "season_tournament_labels_individual_2025_latest.csv", index=False)

    loaded, stem = _load_local_label_table(tmp_path, 2025)

    assert stem == "season_tournament_labels_individual_2025_latest"
    assert loaded["tournament_id"].tolist() == ["R1"]
    assert loaded["team_event"].tolist() == [False]
