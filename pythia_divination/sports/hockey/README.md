# Hockey Modeling Foundation

This package is the first SeekingBeta.AI Hockey pipeline for NHL matchup
prediction, backed by the official `api-web.nhle.com` API.

## What it does
- downloads per-team season schedules and rosters
- downloads completed regular-season boxscores in parallel
- downloads season-to-date club stats for skater and goalie aggregates
- builds rolling pregame team-form features over 3-, 5-, and 10-game windows
- builds rolling starter-goalie proxy features (save %, rest days, prior win %)
- trains a first home-win baseline (sklearn) and a torch MLP

## Recommended v1 target

Start with a **pregame home-team win probability model**.

Why this is the right first Hockey board:
- one row per matchup
- aligns naturally with the product's team-vs-team board layout
- can be trained from public NHL boxscores without proprietary tracking data

## Current feature set
- rolling team form (wins, goals for/against, shot differential, shooting %, save %)
- rolling hits / blocked shots / PIM / power-play goals / giveaways / takeaways
- rolling starter-goalie form (save %, GAA, TOI, rest days)
- same-site rolling win % (home or away streaks)
- games-played-prior + season win % expanding mean
- matchup differentials between home and away rolling features

## Useful commands

Backfill Hockey history:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python3 -m sports.hockey.ingest_history --season-start 2024 --season-end 2025 --include-game-details -v
```

Build the training dataset:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python3 -m sports.hockey.build_training_dataset --season-start 2024 --season-end 2025 --refresh-source-data -v
```

Train the baseline model:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python3 -m sports.hockey.train_baseline --rebuild-dataset --season-start 2024 --season-end 2025 --model hist_gradient_boosting -v
```

Train the torch model:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python3 -m sports.hockey.train_torch --rebuild-dataset --season-start 2024 --season-end 2025 -v
```

## Honest limitations
- This v1 does not have on-ice shift / line-combo features.
- It does not pull tracking data (xG, high-danger chances, zone time).
- Goalie features only model the starter; backup-on-relief is not handled.
- It does not yet produce live Hockey boards in the product. The `/api/sports/hockey/boards` route and the iOS / web wiring are not part of this package.
- `branding.py` is a TODO skeleton; NHL team colors and logo URLs still need to land before boards can render.
- Next upgrades should be:
  - play-by-play shift and faceoff-location features
  - injury report / scratched skater features
  - goalie matchup features against the opposing team
  - power-play and penalty-kill efficiency rolling splits
  - torch inference once the tabular baseline is stable
