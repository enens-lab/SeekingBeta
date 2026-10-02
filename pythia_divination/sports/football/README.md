# Football (NFL) Modeling and Markets

NFL boards built from nflverse release assets (schedules with lines, weekly team and
player stats, weekly rosters).

## What the board shows

- **Headline win probability** (`predictions[].winProbability`, `predictedWinner`): the
  de-vigged nflverse moneyline whenever a fresh line is posted (`basis: "market"`).
  When no line is posted the board says so (`lineStatus: "not_posted"`) and the headline
  is the model view (`basis: "model"`), never silently.
- **Model view**: a `moneyline` market with `basis: "model"` (ridge margin model,
  Platt-calibrated on prior-season out-of-sample predictions). Its fair spread appears
  only as `modelLine` on the spread markets. It is not a pick and shows no edge.
- **Markets centered on the posted line** (`sports/football/markets_nfl.py`): spread at
  the posted line (both sides) plus an alt ladder for the favourite (line +/-3, +/-7),
  total at the posted line plus an alt-over ladder, and team totals at x.5 around
  (total +/- spread) / 2. Lines and prices: nflverse (CC-BY-4.0), `spread_line > 0` means
  the HOME team is favoured, so the home line is `-spread_line`.
- Spread probabilities come from a margin pmf with key-number weights (NFL margins land
  on 3 and 7 far more often than a normal says), re-centered so the posted line matches
  the de-vigged spread price. Push probabilities and whole-number alt lines are only
  published while the walk-forward push gate passes (`ladder.push_validated`).

## Honest numbers (season walk-forward 2022-2025, n=1,084, ties excluded)

| | Accuracy | Brier | Log-loss |
|---|---|---|---|
| Served headline (de-vigged moneyline) | 67.75% | 0.2105 | 0.6082 |
| Model view (ridge + Platt) | 63.75% | 0.2212 | 0.6326 |
| Old HGB recipe (440 features) | 63.75% | 0.2704 | 0.895 |

Alt-ladder Brier: spread 0.1945, total 0.2022. Full detail, per season, calibration
slope/intercept CIs and the push gate: `metrics.json`.

## Files

- `model_view.py`: allowlist, walk-forward team ratings, ridge + Platt, calibration test
- `markets_nfl.py`: de-vig, key-number margin pmf, ladders, board markets, summary buckets
- `model_params.json`: committed serving params (no pickles, no gitignored artifacts)
- `metrics.json`: committed walk-forward metrics with the market beside every number
- `train_baseline.py`: trains/evaluates the above and rewrites both JSON files
- `train_torch.py`: research comparison only (not served; walk-forward Brier 0.2271)

## Commands

```bash
cd pythia_divination
python -m sports.football.ingest_history --season-start 2020 --season-end 2026
python -m sports.football.build_training_dataset --season-start 2020 --season-end 2026
python -m sports.football.train_baseline          # rewrites model_params.json + metrics.json
python scripts/export_football_frontend_data.py   # boards + pick log + graded record
python tests/test_football_markets.py
```

Retrain (`train_baseline`) once a season is complete; within a season the ratings and
rolling form update from the daily ingest, and the params stay fixed so the current
season stays out of sample.

## Training hygiene

- `overtime`, scores, `is_tie` are in `_LEAKY_COLUMNS`; features come from an explicit
  allowlist (no betting lines, no game-time weather, no game-week roster statuses).
- Ties are not a home win or a home loss: no binary label, half a win in team form.
- Every reported number is a season walk-forward; calibrations use prior seasons only.

## Market record

Each daily bake appends the published markets to `data/sports/market_picks/football/`
(append-only, `sports/market_log.py`). History is graded only from that log, against the
line shown before kickoff: `football_market_history.json` and
`football_market_summary.json`. The backtest boards are a separate, labelled simulation
(`recordBasis: "simulated"`).
