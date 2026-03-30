# MLB prediction modeling notes

This package is the starting point for SeekingBeta.AI's MLB prediction pipeline.

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

- available bullpen list from saved MLB boxscores
- rolling reliever performance from prior appearances
- bullpen rest, workload, strikeout, WHIP, and ERA-like aggregates
- best-arm and weakest-link style summary features

### Lineup quality

- starting lineup slots from saved MLB boxscores
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

Train the best current tabular baseline:

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
