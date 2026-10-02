# MLB prediction modeling notes

This package is the starting point for SeekingBeta.AI's MLB prediction pipeline.

## What the app serves (since 2026-10)

`pregame_model.py` + `markets_mlb.py`, used by `scripts/export_mlb_frontend_data.py`:

- **Win probability**: logistic regression on Elo (K=4, home edge 24, margin-of-victory
  multiplier, 1/3 regression between seasons, postseason included), decayed + shrunk
  starter FIP-like / K-BB% / innings / experience, park factor, temperature and wind.
- **Runs**: Poisson GLM per team (offense vs the opposing starter, park, weather, Elo
  edge) with a negative-binomial dispersion (r about 3.65), two teams independent. This
  prices the model-view **run line** (favourite -1.5), **game total** and **team totals**
  (`markets[]`, basis "model", no market line, never an edge).
- **Monthly walk-forward**: both models are refit at each month start on every game
  before it. The in-app history (2024 -> yesterday) is these out-of-sample predictions,
  and upcoming boards use the same procedure.
- Fixed constants, the current fit and the evaluation live in `model_params.json`
  (regenerate with `python scripts/fit_mlb_pregame_model.py`).

Honest record (regular season, walk-forward): 2024 57.3% (Brier 0.2427), 2025 55.7%
(0.2423), 2026 56.2% (0.2437). Always-home: 52.2% / 54.3% / 52.9%. 2025 betting
favourite (ESPN BET close): 56.4% on the same games, Brier 0.2419 -- the market is
still better than the model.

The HistGradientBoosting baseline (`train_baseline.py`) used to be served at "68.8%":
that came from post-game boxscore fields (`team_box["bullpen"]` = relievers who did
NOT pitch; `team_box["battingOrder"]` = end-of-game order). `roster_features.py` now
builds the true starting nine (battingOrder codes ending in 00) and a pregame bullpen
(relievers who pitched in the previous 14 days); `scripts/mlb_leak_tripwire.py` checks
that bullpen features cannot beat always-home by more than 2 points.

## Recommended v1 target

Start with a **pregame home-team win probability model**.

Why this should be first:

- It maps cleanly to one row per game.
- We can train it from official MLB game outcomes without needing a proprietary feed.
- It gives us one prediction board per matchup, which matches the current product direction.
- It is the right foundation for later extensions:
  - live in-game win probability
  - run line / total models
  - starting pitcher props
  - hitter props

## Official data sources we can use now

### MLB Stats API

We can rely on the official MLB Stats API for:

- schedules and game labels
- teams and venue metadata
- player bio / handedness / age / position
- team and player stat splits
- probable pitchers
- game weather and venue field dimensions
- live feed / boxscore / play-by-play
- context metrics
- per-event win probability history

The initial client in this folder covers the core schedule/game/team/player endpoints.

### Baseball Savant / Statcast

We should treat Baseball Savant as the second-layer feature source once the core game model is working.
It is valuable for:

- exit velocity
- launch angle
- xwOBA/xERA-style contact quality
- pitch movement / whiff / chase metrics
- pitcher arsenal quality
- hitter damage profile

That should come after the official schedule/game ingestion, not before it.

## Feature plan

### Game-level matchup features

- home team ID / away team ID
- venue / park / turf / roof
- weather condition / temperature / wind
- date, weekday, travel/rest proxies
- season stage / month

### Team rolling form

- last 7 / 14 / 30 day offense
- last 7 / 14 / 30 day pitching
- recent run differential
- home/away splits
- vs-handedness splits

### Starting pitcher features

- handedness
- rolling ERA/FIP-like proxies from official stats
- strikeout and walk rates
- innings workload / recent rest
- opponent split history

### Bullpen features

- pregame bullpen: relievers who pitched for the team in the previous 14 days
  (never the boxscore `bullpen` list -- that is the relievers who did NOT pitch)
- rolling reliever performance from prior appearances
- bullpen rest, workload, strikeout, WHIP, and ERA-like aggregates
- best-arm and weakest-link style summary features

### Lineup quality

- true starting nine from saved MLB boxscores (player battingOrder codes ending in 00,
  never the end-of-game `battingOrder` list)
- rolling hitter form from prior games
- top/middle/bottom of order strength splits
- lineup-wide OBP/SLG/OPS-like aggregates

### Park and weather context

- venue dimensions from official venue field info
- roof type / turf type
- game-time weather

## Model roadmap

### v1

- binary classifier for home-team win probability
- strong baseline: gradient boosting / random forest / logistic stack

### v2

- grouped ranking or dual-head model for:
  - home win probability
  - run differential bucket

### v3

- live model using:
  - inning state
  - outs
  - base occupancy
  - score differential
  - leverage
  - official win-probability history as a training target

## Storage pattern

Follow the PGA pattern:

- raw snapshots under `data/sports/mlb/raw/<snapshot_tag>/`
- normalized CSV/parquet under `data/sports/mlb/normalized/`
- training datasets stored as durable parquet or CSV manifests
- web crawling / API refresh only for:
  - backfills
  - appending new training data
  - scheduled refreshes

Training runs should read from stored normalized tables, not hit the web every time.

## Current files

- `client.py`
  - official MLB Stats API wrapper and flatteners
- `ingest_history.py`
  - starting CLI for schedule and completed-game summaries
- `roster_features.py`
  - lineup roster extraction, batter logs, bullpen roster extraction, and reliever logs
- `collect_statcast.py`
  - chunked Baseball Savant / Statcast export caching
- `build_training_dataset.py`
  - combined matchup dataset builder with lineup, bullpen, and Statcast features
- `train_baseline.py`
  - tabular baseline models (`hist_gradient_boosting`, `random_forest`, `extra_trees`)
- `train_torch.py`
  - Apple-Silicon-friendly PyTorch home-win model
- `pregame_model.py`
  - the served honest model (results/starter loaders, Elo + starter features, walk-forward)
- `markets_mlb.py`
  - model-view run line / total / team-total `markets[]` entries
- `model_params.json`
  - constants, current fit and walk-forward evaluation of the served model

## Useful commands

Build the matchup dataset:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python -m sports.mlb.build_training_dataset --season-start 2020 --season-end 2025
```

Backfill multi-season Statcast:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
bash scripts/backfill_mlb_statcast.sh
```

Refit + evaluate the served model (writes model_params.json):

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python scripts/fit_mlb_pregame_model.py
python scripts/mlb_leak_tripwire.py
```

Train the (research-only) tabular baseline:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python -m sports.mlb.train_baseline --model hist_gradient_boosting
```

Train the torch model on Apple Silicon:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
bash scripts/setup_pga_torch_env.sh
bash scripts/train_mlb_torch.sh
```
