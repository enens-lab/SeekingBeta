# Football Modeling Foundation

This package is the first SeekingBeta.AI Football pipeline for NFL matchup prediction.

## What it does
- downloads historical NFL schedules from nflverse release assets
- downloads weekly team stats, player stats, and weekly rosters
- builds rolling pregame team and quarterback features
- builds weekly roster-summary availability features
- trains a first home-win baseline model

## Recommended v1 target

Start with a **pregame home-team win probability model**.

Why this is the right first Football board:
- one row per matchup
- aligns naturally with the product's team-vs-team board layout
- can be trained from historical schedules and weekly stats without proprietary live feeds

## Current feature set
- rolling team form
- rolling offensive and defensive weekly stats
- rest and schedule context
- surface / roof / weather context
- divisional-game context
- quarterback rolling form
- roster depth and experience summaries

## Useful commands

Backfill Football history:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python3 -m sports.football.ingest_history --season-start 2020 --season-end 2025 -v
```

Build the training dataset:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python3 -m sports.football.build_training_dataset --season-start 2020 --season-end 2025 --refresh-source-data -v
```

Train the baseline model:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python3 -m sports.football.train_baseline --rebuild-dataset --season-start 2020 --season-end 2025 --model hist_gradient_boosting -v
```

## Honest limitations
- This v1 does not have play-by-play or EPA-by-situation features yet.
- It does not model betting markets directly; lines are only context features.
- It does not yet produce live Football boards in the product.
- Next upgrades should be:
  - play-by-play drive efficiency features
  - injury report / probable inactive features
  - player skill-position matchup features
  - torch inference once the tabular baseline is stable
