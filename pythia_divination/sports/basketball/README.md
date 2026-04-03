# Basketball Modeling Foundation

This package is the first SeekingBeta.AI basketball pipeline for NBA and WNBA.

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
