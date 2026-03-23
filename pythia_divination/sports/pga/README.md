# PGA Tour Data Ingestion

This module collects PGA Tour player stat tables from the public stats site and stores them in a format that is usable for feature engineering and, later, tournament-level win probability models.

## Why this module exists

The existing `pythia_divination` training pipeline is stock-specific. PGA data needs its own ingestion layer because:

- the source is tournament/player statistics rather than OHLCV candles
- feature columns are derived from stat detail tables, not technical indicators
- we need raw snapshots for reproducibility before training any deep neural network

## Current source strategy

We use the public PGA Tour stats pages:

- index/catalog: `https://www.pgatour.com/stats`
- stat detail pages: `https://www.pgatour.com/stats/detail/{statId}`

The detail pages expose structured `__NEXT_DATA__` payloads. That gives us full player-level stat rows without scraping rendered table cells.

## CLI usage

From `/Users/huyngo/Downloads/pythia/pythia_divination`:

```bash
python -m sports.pga.ingest --limit 3 -v
python -m sports.pga.ingest --year 2026
python -m sports.pga.ingest --stat-id 02675 --stat-id 02568
python -m sports.pga.ingest_history --season 2026 --limit 5 -v
python -m sports.pga.ingest_profiles --player-id 46046 --player-name "Scottie Scheffler" -v
python -m sports.pga.build_training_dataset --season-start 2024 --season-end 2026 --snapshot-mode prior_season -v
python -m sports.pga.train_baseline --target won -v
python -m sports.pga.train_neural_model --target won --sequence-length 8 -v
python -m sports.pga.train_multitask_neural_model --targets won top_10 made_cut --sequence-length 10 -v
python -m sports.pga.train_multitask_torch --targets won top_10 made_cut --sequence-length 12 --device auto -v
bash scripts/setup_pga_torch_env.sh
.venv-torch/bin/python -m sports.pga.benchmark_torch_env --device auto
bash scripts/train_pga_multitask_torch.sh
```

## Output layout

Outputs are written under:

```text
pythia_divination/data/sports/pga/
├── raw/<snapshot_tag>/
│   ├── stats_index.json
│   ├── stat_<slug>_<stat_id>.json
│   └── manifest.json
└── normalized/
    ├── stat_catalog_latest.csv
    ├── current_leaders_latest.csv
    ├── stat_detail_<slug>_latest.csv
    ├── player_feature_snapshot_latest.csv
    └── manifest_latest.json
```

## What this enables next

- durable historical PGA stat collection
- player-level season feature snapshots for modeling
- future tournament labels and event-level feature joins
- a sports-specific model training pipeline alongside the existing stock models

## Tournament results / labels

`sports.pga.ingest_history` collects:

- season schedule metadata
- completed tournament leaderboard snapshots
- player-level tournament results with derived labels:
  - `won`
  - `top_5`
  - `top_10`
  - `made_cut`
  - `withdrawn`

Example:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python -m sports.pga.ingest_history --season 2026 --limit 3 -v
```

Important output files:

- `schedule_<season>_latest.csv`
- `leaderboard_<tournament_id>_latest.csv`
- `season_tournament_results_<season>_latest.csv`
- `season_tournament_labels_<season>_latest.csv`

## First training dataset

`sports.pga.build_training_dataset` produces:

- `pga_training_dataset_latest.csv`
- `pga_training_dataset_manifest_latest.json`

The current dataset joins:

- prior-season public PGA stat snapshots by default (`--snapshot-mode prior_season`)
- stable player-profile metadata from public PGA player pages
- completed tournament labels across one or more seasons
- player recent-form features from prior tournaments only
- player season-to-date form features from prior tournaments only
- player availability proxies such as withdrawal rate, short-event rate, missed-cut streaks, and layoff flags
- player context features such as signature-event history, U.S.-event history, and hard/easy-course history
- player course-history features from prior events at the same course
- course-profile features derived from prior years at the same course
- course-fit proxy features that combine public player skill profiles with historical course difficulty proxies
- field-strength proxies derived from prior player form
- schedule/context features such as purse, FedExCup points, event progression, season timing, and rest spacing

This is much closer to a real sports training table than the first pass. The current default avoids the biggest leakage issue by using prior-season public stat priors rather than same-season end-state stats. We still do not have true tournament-week historical stat snapshots, so the remaining path to a fully clean as-of dataset is to store time-versioned stat snapshots throughout the season or ingest a licensed shot-level feed.

We also do not pretend to have unavailable public features such as slope conditions or full hole-by-hole ShotLink context. Instead, the current pipeline uses defensible public-data proxies:

- course difficulty from prior tournament scoring at the same course
- cut-rate and winning-score history at the same course
- player par-type scoring splits
- putting-distance and around-the-green splits
- approach-distance family stats
- player age / turned-pro / height / college / equipment-brand features from the public player profile pages
- player availability proxies instead of unstructured injury claims
- player rest / season timing / event pressure context

## First baseline model

`sports.pga.train_baseline` now trains a tabular baseline model for one target. The default is a random forest because this is a small, highly tabular tournament-ranking problem, not a sequential price-series problem.

Targets currently supported:

- `won`
- `top_5`
- `top_10`
- `made_cut`

Artifacts are written under:

```text
pythia_divination/artifacts/pga_baseline/<model_type>/<target>/
```

with:

- `model.joblib`
- `feature_columns.json`
- `metrics.json`
- `validation_predictions.csv`
- `feature_importances.csv` (for tree-based models)

Example:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python -m sports.pga.train_baseline --target won -v
python -m sports.pga.train_baseline --target top_10 -v
```

## First neural model

`sports.pga.train_neural_model` builds a two-branch network:

- sequence branch: previous tournaments for the same player
- static branch: current tournament context plus prior-form/course-history aggregates

The neural model now uses:

- a sequence branch built only from prior completed tournaments
- a static branch built from prior-form, season-to-date, course-history, course-fit, and curated public stat priors

The neural path is still exploratory. The cleaned random-forest baseline is currently the more trustworthy benchmark, while the neural model needs more work on auxiliary targets and calibration before it should be treated as the production choice.

## Multi-task neural model

`sports.pga.train_multitask_neural_model` is the current path for scaling into a more serious golf neural network:

- one shared sequence tower for prior tournament history
- one shared static tower for profile, form, course-fit, and stat-prior features
- separate output heads for:
  - `won`
  - `top_10`
  - `made_cut`

This lets the network learn richer structure than a single sparse win label alone, and it gives a better path to using more seasons and more local hardware without forcing the model to optimize only the rarest target.

Recommended larger local run on Apple Silicon:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
python -m sports.pga.build_training_dataset --season-start 2021 --season-end 2026 --snapshot-mode prior_season --profile-workers 12 -v
python -m sports.pga.train_multitask_neural_model \
  --targets won top_10 made_cut \
  --sequence-length 12 \
  --batch-size 256 \
  --epochs 60 \
  --lstm-units 128 \
  --static-width 256 \
  --dense-width 256 \
  --cache-dataset \
  --mixed-precision \
  -v
```

## PyTorch Apple Silicon path

If `tensorflow-metal` is unstable on your machine, `sports.pga.train_multitask_torch` provides a separate multi-task path that targets Apple MPS through PyTorch:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
bash scripts/setup_pga_torch_env.sh
.venv-torch/bin/python -m sports.pga.benchmark_torch_env --device auto --batch-size 512 --steps 25
python -m sports.pga.train_multitask_torch \
  --targets won top_10 made_cut \
  --sequence-length 12 \
  --batch-size 512 \
  --epochs 30 \
  --lstm-units 192 \
  --static-width 384 \
  --dense-width 384 \
  --device auto \
  -v
```

For repeatable larger local runs on your Mac, use the wrapper script instead:

```bash
cd /Users/huyngo/Downloads/pythia/pythia_divination
SEASON_START=2021 \
SEASON_END=2026 \
BATCH_SIZE=512 \
EPOCHS=30 \
LSTM_UNITS=192 \
STATIC_WIDTH=384 \
DENSE_WIDTH=384 \
DEVICE=auto \
bash scripts/train_pga_multitask_torch.sh
```

This path uses the dedicated `.venv-torch` environment plus Apple MPS when available. The separate `.venv-metal` environment is still experimental because the current TensorFlow Metal plugin path is unstable on this machine.

Artifacts are written under:

```text
pythia_divination/artifacts/pga_neural/<target>/
pythia_divination/artifacts/pga_neural_multitask_torch/
```

with:

- `model.keras`
- `sequence_feature_columns.json`
- `static_feature_columns.json`
- `sequence_scaler.joblib`
- `static_imputer.joblib`
- `static_scaler.joblib`
- `metrics.json`
- `validation_predictions.csv`
