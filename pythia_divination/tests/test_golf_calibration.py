"""Checks for golf top-10 / made-cut calibration and the golf board exporter.

Run (cwd = pythia_divination):  python tests/test_golf_calibration.py
Real-data checks run only when the golf dataset and multitask artifacts exist.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sports.pga import calibration as C  # noqa: E402


def _synthetic(seed: int = 0, events: int = 8, field: int = 120):
    rng = np.random.default_rng(seed)
    preds, labels = [], []
    for e in range(events):
        tid = f"R2025{e:03d}"
        skill = rng.normal(size=field)
        order = np.argsort(-(skill + rng.normal(scale=1.5, size=field)))
        position = np.empty(field, dtype=int)
        position[order] = np.arange(1, field + 1)
        cut = position <= 65
        top10 = position <= 10
        won = position == 1
        # inflated heads, like a pos_weight-trained model
        p10 = 1 / (1 + np.exp(-(skill * 0.9 - 0.2)))
        pmc = 1 / (1 + np.exp(-(skill * 0.8 + 0.1)))
        pw = np.exp(skill * 1.2)
        pw /= pw.sum()
        start = f"2025-{1 + e:02d}-10"
        for i in range(field):
            preds.append({"tournament_id": tid, "tournament_name": f"Event {e % 4}", "player_id": i,
                          "winner_probability": pw[i], "top_10_probability": p10[i], "made_cut_probability": pmc[i]})
            labels.append({"tournament_id": tid, "tournament_name": f"Event {e % 4}", "player_id": i, "event_start_date": start,
                           "won": float(won[i]), "top_10": float(top10[i]), "made_cut": float(cut[i]), "tour": "PGA",
                           "event_field_size": field})
    return pd.DataFrame(preds), pd.DataFrame(labels)


def test_shift_to_sum():
    logits = np.linspace(-3, 3, 50)
    for target in (1.0, 10.0, 49.5):
        assert abs(C.shift_to_sum(logits, target).sum() - target) < 1e-6
    p = C.shift_to_sum(logits, 10.0)
    assert np.all(np.diff(p) > 0)  # ordering preserved


def test_calibration_targets_and_ordering():
    preds, labels = _synthetic()
    out, log = C.calibrate_board(preds, labels)
    sums = out.groupby("tournament_id")["top_10_probability"].sum()
    assert sums.between(9.5 - 1e-6, 11.0 + 1e-6).all(), sums
    assert (out["made_cut_probability"] >= out["top_10_probability"] - 1e-12).all()
    assert (out["top_10_probability"] >= out["winner_probability"] - 1e-12).all()
    assert log["events"][0]["top10_target"] == C.TOP10_DEFAULT_TARGET  # nothing earlier to learn from
    # the first edition of "Event 0" has no previous edition; the second uses the first's cut rate
    fifth = log["events"][4]
    assert abs(fifth["cut_rate_target"] - 65 / 120) < 1e-4  # logged to 4 decimals
    report = C.calibration_report(out, labels)
    assert report["top10"]["brier"] < report["top10"]["brierRaw"]
    assert report["rowsMadeCutBelowTop10"] == 0


def test_calibration_is_walk_forward():
    preds, labels = _synthetic(seed=1)
    out, _ = C.calibrate_board(preds, labels)
    changed = labels.copy()
    late = changed["tournament_id"] >= "R2025004"
    changed.loc[late, "top_10"] = 1 - changed.loc[late, "top_10"]
    changed.loc[late, "made_cut"] = 1 - changed.loc[late, "made_cut"]
    out2, _ = C.calibrate_board(preds, changed)
    early = out["tournament_id"] <= "R2025004"  # event 4 itself may not see its own outcome
    for column in ("top_10_probability", "made_cut_probability"):
        assert np.allclose(out.loc[early, column], out2.loc[early, column]), column


def test_upcoming_boards_are_dated_and_labelled():
    from scripts import export_frontend_data as G

    def rows(tid, name, season, start, tour="PGA"):
        return [{"tournament_id": tid, "tournament_name": name, "player_name": f"P{i}", "winner_probability": 0.1 * (10 - i),
                 "top_10_probability": 0.5, "made_cut_probability": 0.9, "won": 0, "tour": tour, "season_year": season,
                 "display_date": None, "event_start_date": start, "course_name": "C", "course_state_code": "GA",
                 "player_id": i, "country": None, "feature_country_name": None, "owgr__rank": i + 1, "sg_total__avg": 0.0,
                 "sg_approach__avg": 0.0, "sg_putting__avg": 0.0, "sg_tee_to_green__avg": 0.0,
                 "birdie_or_better_pct__value": 20.0, "source_priority": 0} for i in range(10)]

    df = pd.DataFrame(
        rows("R2025493", "The RSM Classic", 2025, "2025-11-20")
        + rows("R2024527", "ZOZO CHAMPIONSHIP", 2024, "2024-10-24")
        + rows("LPGA-1", "TOTO Japan Classic", 2025, None, tour="LPGA")
        + rows("LPGA-2", "Undated LPGA Event", 2025, None, tour="LPGA")
    )
    today = datetime(2026, 10, 2)
    calendar = {
        "PGA": [{"name": "The RSM Classic", "startKey": 20261119, "endKey": 20261122}],
        "LPGA": [{"name": "TOTO Japan Classic", "startKey": 20261105, "endKey": 20261108}],
    }
    boards = G._build_upcoming(df, calendar=calendar, today=today)
    names = sorted(board["name"] for board in boards)
    assert names == ["2026 TOTO Japan Classic", "2026 The RSM Classic"], names  # ZOZO not on this season's calendar
    for board in boards:
        assert board["fieldBasis"] == "previous_edition" and board["scheduledDate"] and board["fieldNote"]
    # calendar unavailable: PGA falls back to last year's date; undated LPGA boards are dropped
    boards = G._build_upcoming(df, calendar={}, today=today)
    assert sorted(board["name"] for board in boards) == ["2026 The RSM Classic", "2026 ZOZO CHAMPIONSHIP"]
    assert all(board["dateSource"] == "previous_edition_date" for board in boards)


def test_backtests_need_a_known_winner():
    from scripts import export_frontend_data as G

    base = {"tournament_name": "E", "top_10_probability": 0.2, "made_cut_probability": 0.6, "tour": "PGA", "season_year": 2025,
            "display_date": None, "event_start_date": "2025-05-01", "course_name": None, "course_state_code": None, "country": None,
            "feature_country_name": None, "sg_total__avg": 0.0, "sg_approach__avg": 0.0, "sg_putting__avg": 0.0,
            "sg_tee_to_green__avg": 0.0, "birdie_or_better_pct__value": 20.0, "source_priority": 0}
    df = pd.DataFrame([
        {**base, "tournament_id": "A", "player_name": "X", "winner_probability": 0.6, "won": 1, "player_id": 1, "owgr__rank": 5},
        {**base, "tournament_id": "A", "player_name": "Y", "winner_probability": 0.4, "won": 0, "player_id": 2, "owgr__rank": 1},
        {**base, "tournament_id": "B", "player_name": "X", "winner_probability": 0.7, "won": 0, "player_id": 1, "owgr__rank": 5},
        {**base, "tournament_id": "B", "player_name": "Y", "winner_probability": 0.3, "won": 0, "player_id": 2, "owgr__rank": 1},
    ])
    boards, scoring = G._build_backtests(df, {"A": "X"})
    assert [board["tournament"] for board in boards] == ["E"] and len(scoring) == 1
    assert boards[0]["hitStatus"] == "Top Pick" and boards[0]["rankingFavorite"] == "Y" and not boards[0]["rankingFavoriteWon"]


def test_real_data_acceptance():
    artifact = ROOT / "artifacts" / "pga_neural_multitask_torch" / "validation_predictions.csv"
    dataset = ROOT / "data" / "sports" / "pga" / "normalized" / "golf_training_dataset_latest.csv"
    if not artifact.exists() or not dataset.exists():
        print("  (no local golf data: real-data checks skipped)")
        return
    from sports.pga.recompute_metrics import load_labels, top_pick_summary

    labels = load_labels(dataset)
    preds = pd.read_csv(artifact)
    preds["player_id"] = pd.to_numeric(preds["player_id"], errors="coerce")
    inputs = preds[["tournament_id", "tournament_name", "player_id"]].copy()
    inputs["winner_probability"] = preds["won_normalized_probability"]
    inputs["top_10_probability"] = preds["top_10_probability"]
    inputs["made_cut_probability"] = preds["made_cut_probability"]
    out, _ = C.calibrate_board(inputs, labels)
    report = C.calibration_report(out, labels)
    print("  held-out events=%d top-10 Brier %.4f (const %.4f) sums %.2f-%.2f; made-cut mean %.4f vs %.4f"
          % (report["events"], report["top10"]["brier"], report["top10"]["brierConstant"], report["top10"]["sumPerEventMin"],
             report["top10"]["sumPerEventMax"], report["madeCut"]["meanPredicted"], report["madeCut"]["actualRate"]))
    assert report["top10"]["brier"] < 0.0897
    sums = out.groupby("tournament_id")["top_10_probability"].sum()
    assert sums.between(9.0, 11.0 + 1e-9).all()
    assert (out["made_cut_probability"] >= out["top_10_probability"] - 1e-12).all()
    assert abs(report["madeCut"]["meanPredicted"] - report["madeCut"]["actualRate"]) <= 0.02
    summary = top_pick_summary(preds, labels, "won_normalized_probability")
    assert summary["top_pick_events"] == 51 and summary["top_pick_hits"] == 7


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("test_golf_calibration: all checks passed")
