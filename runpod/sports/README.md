# RunPod sports-export worker

Offloads the heavy per-sport board exports off the EC2 box (the jobs that OOM'd the
7.6 GB instance) onto RunPod serverless. Follows the QD.TEK worker pattern:
`handler(job)` → dispatch on `input.type` → `runpod.serverless.start(...)`.

```
EC2 cron (refresh_sports_runpod.sh)                 RunPod worker (handler.py)
  POST /run {type:sports_export, sports:[...]}  ─────►  sync inputs from S3
        ▲                                                run export_<sport>.py
        │ poll /status                                   push board JSON → S3
  aws s3 sync boards ◄──────────────────────────────────┘
  rebuild prophecy + nginx reload
```

**Data exchange = the `pythia-ml-artifacts` S3 bucket** (your existing convention):
- inputs (gitignored on EC2): normalized data `s3://…/sports-data/<sport>/`, model
  artifacts `s3://…/artifacts/artifacts/`
- outputs: board JSON `s3://…/frontend-boards/<file>.json`

The worker image is **code-only**; data + artifacts sync at runtime, so the image
stays small and the boards always reflect current data without an image rebuild.

---

## One-time setup (YOU — needs RunPod/AWS console + secrets)

1. **Worker AWS creds** — the read-write S3 user from `ops/iam-artifact-uploader-policy.json`
   (`pythia-artifact-uploader`). The worker needs `s3:GetObject` + `s3:PutObject` on
   `pythia-ml-artifacts/*`. (If not created yet, see `ops/README-iam-artifacts.md`.)

2. **Seed the bucket from EC2** (so the worker has inputs to pull):
   ```bash
   cd /home/ec2-user/seekingbeta
   export AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… AWS_REGION=us-east-1
   ops/upload_sports_data_to_s3.sh mlb        # start with MLB; later: no arg = all sports
   ```

3. **Build & publish the image** — pick ONE:
   - **Docker Hub / registry** (build anywhere with Docker, from the repo root):
     ```bash
     docker build -f runpod/sports/Dockerfile -t <youruser>/pythia-sports-runpod:latest .
     docker push <youruser>/pythia-sports-runpod:latest
     ```
   - **RunPod GitHub build (chosen):** RunPod console → Serverless → New Endpoint →
     "Import Git Repository". Authorize RunPod's GitHub app on the **private** repo
     `enens-lab/SeekingBeta`, then set:
       - Branch: `feature/sports-prediction-market` (where the worker lives today;
         switch to `lstm_v2`/main once merged)
       - Dockerfile path: `runpod/sports/Dockerfile`
       - Build context: repo root (default) — the Dockerfile COPYs `pythia_divination/`
         from the root, so do NOT set the context to `runpod/sports`.
     RunPod clones + builds on each push; the repo is code-only (data/artifacts
     gitignored) so the build stays lean.

4. **Create the RunPod serverless endpoint** from that image:
   - **Compute — NO GPU is used** (exports are pandas/sklearn = CPU + RAM bound; the
     constraint is system RAM ≥ 8 GB, not VRAM):
       - If RunPod offers a **CPU endpoint**, pick it with a flavor of **≥ 8 GB RAM**
         (16 GB comfortable — the MLB *upcoming* phase OOM'd at 3 GB). Cheapest + correct.
       - If the flow **forces a GPU**, pick the cheapest 16 GB-class card JUST for its
         bundled CPU+RAM — **RTX A4000 / RTX 4000 Ada / A5000** — and multi-select a few
         for availability. Confirm RAM ≥ 8 GB. Never A100/H100 (paying for an unused GPU).
         (Save real GPU spend for the future training worker, which actually uses CUDA.)
   - **Active (min) workers 0** (pay-per-run), **Max workers 1–2**, **Idle timeout 5–10 s**.
   - **Execution timeout ≥ 1800 s** — a full multi-sport run is ~10–15 min; the default is
     often too low and would kill the job mid-run.
   - **Container disk ≥ 10 GB** (room for synced data + artifacts), or attach a
     **network volume** mounted somewhere stable to cache `data/`+`artifacts/`
     across runs (faster cold starts).
   - **Environment variables** on the endpoint:
     ```
     S3_BUCKET=pythia-ml-artifacts
     AWS_REGION=us-east-1
     AWS_ACCESS_KEY_ID=<uploader key>
     AWS_SECRET_ACCESS_KEY=<uploader secret>
     # optional overrides (defaults shown):
     # SPORTS_DATA_S3_PREFIX=sports-data
     # BOARDS_S3_PREFIX=frontend-boards
     # ARTIFACTS_S3_URI=s3://pythia-ml-artifacts/artifacts/artifacts/
     # SPORTS_EXPORT_TIMEOUT_SEC=1800
     ```

5. **Give me the endpoint ID + a RunPod API key** → I add to the EC2 `.env`
   (never committed):
   ```
   RUNPOD_API_KEY=<key>
   RUNPOD_SPORTS_ENDPOINT_ID=<endpoint id>
   ```

---

## Smoke test (after the endpoint is up)

```bash
RUNPOD_URL="https://api.runpod.ai/v2/$RUNPOD_SPORTS_ENDPOINT_ID"
# health
curl -s -X POST "$RUNPOD_URL/runsync" -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" -d '{"input":{"type":"health"}}'
# one real export (async; poll /status)
curl -s -X POST "$RUNPOD_URL/run" -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" -d '{"input":{"type":"sports_export","sport":"mlb"}}'
```

Or end-to-end from the EC2 box (submit → wait → pull boards → rebuild prophecy):
```bash
ops/refresh_sports_runpod.sh '["mlb"]'
```

---

## Cutover (once MLB is proven)

- Seed all sports: `ops/upload_sports_data_to_s3.sh` (no args).
- Repoint the crons from `ops/refresh_sports_cron.sh` (on-box export) to
  `ops/refresh_sports_runpod.sh '["all"]'`. Keep the on-box script as a fallback.
- The on-box export + its 3 GB memory cap stay as a safety net but no longer run
  in the hot path.

## Job contract

```jsonc
{"input": {"type": "sports_export", "sport": "mlb"}}            // one sport
{"input": {"type": "sports_export", "sports": ["mlb","golf"]}} // several
{"input": {"type": "sports_export", "sports": ["all"]}}        // every sport
{"input": {"type": "stock_predictions", "universe": "core",     // daily LSTM sweep (below)
           "tickers": ["AAPL", "..."], "models": ["lstm_5d","lstm_jackpot","lstm_quant"],
           "homepage_tickers": ["AAPL", "..."], "limit": 0}}
{"input": {"type": "health"}}
```
Supported sports: `mlb, soccer, golf, basketball, football, olympics, tennis`
(tennis runs ingest + build before its export, mirroring `refresh_sports_cron.sh`).

## `stock_predictions` — the daily LSTM sweep (since 2026-09-19)

This is what replaced the always-on `divination-api` container on the web box.
`pythia_divination/scripts/export_stock_predictions.py` runs the torch options
LSTMs (`artifacts/artifacts/{lstm_5d,lstm_jackpot}/torch/` — **those bundles must
exist in S3**, the worker refuses to run without them) over:

- `universe: core` (default) = `config.yaml`'s curated ~170 names + `homepage_tickers`
  + the `tickers` list the box passes (Free/Basic tier lists + every user watchlist,
  collected by `ops/refresh_stock_predictions_runpod.sh`), ~200 tickers, **~3-5 min**
  with 4 workers; the cost is dominated by Yahoo I/O (~0.5 s bars + ~2.5 s option
  chain per ticker), not compute (~10 ms per ticker for both models).
- `universe: full` = all ~6,700 names in `data/universe.csv`, which is **not in the
  image** (sync it first) and is **hours of Yahoo I/O** (5-7 h sequential). Do not
  schedule it daily; the sweep's `--deadline-seconds` (1500 s default, under the
  1800 s endpoint timeout) will truncate it and publish a partial file.

Output: `frontend-boards/stock_predictions_latest.json` (schema in
`pythia_prophecy/api/stock_predictions_store.py`). The box pulls it, validates it
(`ops/validate_stock_predictions.py`) and drops it into the bind-mounted
`pythia_prophecy/data/predictions/`; prophecy-api serves `/predict/*`,
`/api/analyze` and the Daily Brief from it. No Postgres and no Alpaca on the worker:
bars come from Yahoo (`DATA_SOURCE=yahoo`); add `ALPACA_KEY_ID`/`ALPACA_SECRET_KEY`
+ `DATA_SOURCE=alpaca` to the endpoint env to reproduce the box's old feed exactly.

Cron on the box (weekdays after the close, apart from the 21:30 options archive):
```
15 22 * * 1-5 cd /home/ec2-user/seekingbeta && ops/refresh_stock_predictions_runpod.sh >> logs/refresh_stock_predictions.log 2>&1
```
