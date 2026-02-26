# Stooq Prediction Incident Runbook

This runbook is for the recurring issue where LSTM predictions fail in production with:
- `Failed to fetch OHLCV (Yahoo) ...`
- `Failed to fetch OHLCV (Stooq) ...`
- `Failed to fetch OHLCV for <ticker> after retries and fallbacks`

## 1) Non-negotiable deployment rules

1. Keep exactly one `DATA_SOURCE=` entry in `.env`.
2. After changing `.env`, do not use `docker compose restart`.
3. Always recreate containers after env changes:
   - `docker compose up -d --build --force-recreate divination-api prophecy-api frontend`
4. Verify runtime config from inside the container:
   - `docker compose exec -T divination-api sh -lc 'echo "$DATA_SOURCE"'`
   - `docker compose exec -T divination-api python -c "from config.settings import settings; print(settings.data_source)"`

## 2) Fast diagnosis sequence

Run in order:

```bash
cd /home/ec2-user/seekingbeta
docker compose ps
curl -i http://localhost:8000/healthz
curl -i http://localhost:8000/predict/lstm_5d/AAPL
curl -i http://localhost:8000/predict/lstm_jackpot/AAPL
curl -sk -i https://seekingbeta.ai/predict/lstm_5d/AAPL
curl -sk -i https://seekingbeta.ai/predict/lstm_jackpot/AAPL
docker compose logs --tail=300 divination-api
```

## 3) What the errors usually mean

- `No artifacts found for lstm_*`: artifacts are missing or not mounted correctly.
- `Failed to fetch OHLCV (Yahoo)...JSONDecodeError`: Yahoo API response is invalid/blocked.
- `Failed to fetch OHLCV (Stooq)...`: Stooq historical CSV endpoint unavailable/challenged/rate-limited.
- `stooq rate limit exceeded for this source IP`: daily hit limit for current egress IP.
- `Not enough data after feature engineering (need 60 samples, got 0)`: upstream returned quote-like or too-short history (single-row data is not usable for LSTM).
- `502` at public URL with `200` from `:8000`: nginx upstream/proxy mismatch.

## 4) Artifact verification

```bash
ls -la pythia_divination/artifacts/lstm_5d/classifier
ls -la pythia_divination/artifacts/lstm_jackpot/classifier
```

If missing:

```bash
aws s3 sync s3://pythia-ml-artifacts/artifacts/artifacts/ pythia_divination/artifacts/ --region us-east-1
docker compose up -d --build --force-recreate divination-api
```

## 5) Stooq-specific verification (inside container)

```bash
docker compose exec -T divination-api python - <<'PY'
from data.fetch import _stooq_http, fetch_ohlcv_for_lstm

for t in ["AAPL","MSFT","NVDA"]:
    try:
        df = _stooq_http(t, None)
        print("stooq raw ok:", t, len(df), float(df["Close"].iloc[-1]))
    except Exception as e:
        print("stooq raw err:", t, repr(e))

for src in ["stooq","auto"]:
    try:
        df = fetch_ohlcv_for_lstm("AAPL", data_source=src)
        print("lstm fetch ok:", src, len(df), float(df["Close"].iloc[-1]))
    except Exception as e:
        print("lstm fetch err:", src, repr(e))
PY
```

## 5.1) Cache visibility during incident

```bash
curl -s http://localhost:8000/predict/cache/status
curl -s http://localhost:8000/predict/homepage
```

If cache has entries, homepage should still return stale predictions even while live provider calls fail.

## 6) Known-good health criteria

- `http://localhost:8000/healthz` returns `200`.
- `http://localhost:8000/predict/lstm_5d/AAPL` returns `200`.
- `http://localhost:8000/predict/lstm_jackpot/AAPL` returns `200`.
- `https://seekingbeta.ai/predict/lstm_5d/AAPL` returns `200`.
- `https://seekingbeta.ai/predict/lstm_jackpot/AAPL` returns `200`.

If any of these fail, capture logs and stop making additional env changes until root cause is identified.
