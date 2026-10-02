# Basketball Modeling Foundation

This package is the SeekingBeta.AI basketball pipeline for NBA and WNBA.

## Served model (2026-10-02, plan items P1-4 / P1-12)

`scripts/export_basketball_frontend_data.py` serves `honest_model.py`, refit at every bake:

- **Win probability:** Elo with margin of victory (`elo.py`; K=16 NBA / 30 WNBA, home
  advantage 60, 0.75 season carry-over) as a fixed log-odds offset, plus an L1 logistic
  regression over same-season box-score form, rotation/availability, true rest and
  back-to-back from local tip dates (`results.py`, capped at 4 days) and a postseason
  flag. A Platt map fitted only on earlier walk-forward predictions, shrunk toward the
  identity, calibrates it. The torch MLP / HGB artifacts are no longer read.
  **WNBA serves calibrated Elo-MOV** (`leagues.wnba.win_model` in `model_params.json`):
  the L1 model's paired Brier CI against Elo included 0 there, and the review rule
  prefers Elo in that case. NBA keeps the L1 model (CI below 0).
- **No pick:** a probability within 0.005 of 0.5 (shown as 50% / 50%) is no pick: no
  history board, no accuracy credit, no moneyline lean (`honest_model.NO_PICK_BAND`).
- **Missing-season guard:** `honest_model.data_coverage` checks every season from
  `margin_warmup_from_season` through the current one (count >= 60% of the median of
  complete seasons, box scores for >= 60% of finals). A failing league keeps its
  previous history / model record and publishes no upcoming boards.
- **Spread / total / team totals (model only, `market: null`):** opponent-adjusted
  offense/defense ratings (`ratings.py`) blended 50/50 with an Elo+rest ridge for the
  margin; normal distributions with a rolling out-of-sample sigma. Play-in and playoff
  boards carry spread + moneyline only (postseason totals ran 8.6 points under the model).
- **History:** a 14-day walk-forward (`recordBasis: "simulated"`); every game is graded by
  a model fitted only on earlier dates. NBA 2022-24 / WNBA 2024 tuned the parameters, so
  they are predicted (to seed the calibrator and sigma) but never published.
- **Inputs:** results, Elo and rest come from the normalized tables plus the live CDN
  schedule and freshly fetched boxscores. `backfill_history.py` appends archived seasons,
  play-in, playoff and cup games without dropping existing rows.
- **Parameters:** `model_params.json` (committed). **Evidence:** `walk_forward_metrics.json`,
  regenerated with `python -m sports.basketball.evaluate_walk_forward --market-file <closing lines> --out ...`.
- **Leakage guard:** the schedule/boxscore `wins`/`losses` columns are POST-game records
  (opening night shows 1-0 / 0-1). They and anything derived from them, such as
  `matchup_diff_wins`, never enter the model.

## What it does
- pulls the current live season schedule from the official NBA/WNBA schedule feeds
- fetches completed-game boxscores from the official league live-data CDN
- builds rolling team-form features for home-win prediction
- builds player game logs and rotation continuity features from boxscores
- trains a first baseline classifier for current-season basketball games

## Current limitation
The official schedule feed that is easiest to use here exposes the **current season** directly.
That means this first version is honest but narrow:
- NBA current season can be trained right now because completed regular-season games exist
- WNBA current season schedule is available, but training will not work until regular-season games have finished
- older season backfill still needs a second official archive path

## Commands

Collect current-season history:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python3 -m sports.basketball.ingest_history --league nba --include-game-details -v
python3 -m sports.basketball.ingest_history --league wnba --include-game-details -v
```

Build datasets:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python3 -m sports.basketball.build_training_dataset --league nba -v
python3 -m sports.basketball.build_training_dataset --league all -v
```

The dataset builder also writes:
- `player_game_logs_<league>_latest.csv`
- `rotation_game_logs_<league>_latest.csv`

Train a first baseline:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python3 -m sports.basketball.train_baseline --league nba --model hist_gradient_boosting -v
```

Train the PyTorch model:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python3 -m sports.basketball.train_torch --league nba --device auto -v
python3 -m sports.basketball.train_torch --league wnba --device auto -v
```
