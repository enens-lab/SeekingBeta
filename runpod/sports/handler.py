"""RunPod serverless worker — Pythia sports-board exports.

Offloads the heavy per-sport board exports (the jobs that OOM'd the 7.6 GB EC2
box on 2026-06-19/20) to RunPod. The worker:

  1. syncs inputs from S3 — the normalized sport data + model artifacts that are
     gitignored on EC2 (so they live in S3, not in the image),
  2. runs the SAME export scripts the EC2 cron runs (pythia_divination/scripts/
     export_*_frontend_data.py), unmodified, from the same /work/pythia_divination
     working dir so all the relative-path resolution matches the box,
  3. pushes the resulting board JSON back to S3 (frontend-boards/ prefix).

The EC2 side (ops/refresh_sports_runpod.sh) then triggers this worker, waits,
`aws s3 sync`s the boards down, and rebuilds prophecy. Net effect: the multi-GB
export memory runs on RunPod, never on the box.

Job contract (RunPod `input`):
  {"input": {"type": "sports_export", "sport": "mlb"}}                  # one sport
  {"input": {"type": "sports_export", "sports": ["mlb", "golf"]}}       # several
  {"input": {"type": "sports_export", "sports": ["all"]}}               # every sport
  {"input": {"type": "health"}}                                         # readiness

Returns: {"ok": bool, "results": [{"sport","ok","exit_code","boards":[...],"log_tail"}]}

Layout in the image (see Dockerfile): code at /work/pythia_divination, output dir
/work/pythia_prophecy/frontend/src/data (so the export scripts' PROJ_ROOT =
parents[2] resolves to /work, matching the EC2 repo layout). data/ + artifacts/
are emptied at build time and re-synced from S3 here at runtime.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import time
from pathlib import Path

import runpod

REPO = Path(os.getenv("PYTHIA_REPO", "/work"))
DIV = REPO / "pythia_divination"
OUTPUT_DIR = REPO / "pythia_prophecy" / "frontend" / "src" / "data"

S3_BUCKET = os.getenv("S3_BUCKET", "pythia-ml-artifacts")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SPORTS_DATA_PREFIX = os.getenv("SPORTS_DATA_S3_PREFIX", "sports-data").strip("/")
ARTIFACTS_S3_URI = os.getenv("ARTIFACTS_S3_URI", f"s3://{S3_BUCKET}/artifacts/artifacts/")
BOARDS_PREFIX = os.getenv("BOARDS_S3_PREFIX", "frontend-boards").strip("/")
RUN_TIMEOUT_SEC = int(os.getenv("SPORTS_EXPORT_TIMEOUT_SEC", "1800"))
SYNC_TIMEOUT_SEC = int(os.getenv("S3_SYNC_TIMEOUT_SEC", "1800"))

# Per-sport: which data/sports/<dir> subtree(s) to sync from S3, and the command
# chain to run (cwd = DIV). Mirrors ops/refresh_sports_cron.sh exactly — tennis
# needs an ingest + build step before its export; the rest are export-only.
SPORTS = {
    "mlb":        {"data": ["mlb"],        "cmds": [["python", "-u", "scripts/export_mlb_frontend_data.py"]]},
    "soccer":     {"data": ["soccer"],     "cmds": [["python", "-u", "scripts/export_soccer_frontend_data.py"]]},
    "golf":       {"data": ["pga"],        "cmds": [["python", "-u", "scripts/export_frontend_data.py"]]},
    "basketball": {"data": ["basketball"], "cmds": [["python", "-u", "scripts/export_basketball_frontend_data.py"]]},
    "football":   {"data": ["football"],   "cmds": [["python", "-u", "scripts/export_football_frontend_data.py"]]},
    "olympics":   {"data": ["olympics"],   "cmds": [["python", "-u", "scripts/export_olympics_frontend_data.py"]]},
    "tennis":     {"data": ["wta"],        "cmds": [
        ["python", "-m", "sports.wta.ingest", "--start-year", "2020", "--end-year", str(dt.datetime.utcnow().year), "--force"],
        ["python", "-m", "sports.wta.build_training_dataset"],
        ["python", "-u", "scripts/export_wta_frontend_data.py"],
    ]},
}

_ARTIFACTS_READY = False


def _run(cmd, cwd, timeout):
    print(f"[worker] $ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    try:
        p = subprocess.run(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.output or "") if isinstance(exc.output, str) else ""
        print(f"[worker] TIMEOUT after {timeout}s", flush=True)
        return 124, out + f"\n[worker] TIMEOUT after {timeout}s"
    out = p.stdout or ""
    print(out[-4000:], flush=True)
    return p.returncode, out


def _aws_sync(src, dest, timeout=SYNC_TIMEOUT_SEC):
    Path(dest).mkdir(parents=True, exist_ok=True)
    return _run(["aws", "s3", "sync", src, dest, "--region", AWS_REGION, "--only-show-errors"], REPO, timeout)


def _sync_artifacts():
    global _ARTIFACTS_READY
    if _ARTIFACTS_READY:
        return
    rc, _ = _aws_sync(ARTIFACTS_S3_URI, str(DIV / "artifacts") + "/")
    if rc != 0:
        raise RuntimeError(f"artifact sync failed (rc={rc}) from {ARTIFACTS_S3_URI}")
    _ARTIFACTS_READY = True


def _sync_sport_data(sport):
    for sub in SPORTS[sport]["data"]:
        src = f"s3://{S3_BUCKET}/{SPORTS_DATA_PREFIX}/{sub}/"
        rc, _ = _aws_sync(src, str(DIV / "data" / "sports" / sub) + "/")
        if rc != 0:
            raise RuntimeError(f"data sync failed for '{sub}' (rc={rc}) from {src}")


def _upload_boards(since_ts):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    uploaded = []
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        # Only files this export run actually (re)wrote — robust to naming.
        if f.stat().st_mtime >= since_ts - 1:
            rc, _ = _run(
                ["aws", "s3", "cp", str(f), f"s3://{S3_BUCKET}/{BOARDS_PREFIX}/{f.name}",
                 "--region", AWS_REGION, "--only-show-errors"],
                REPO, 900,
            )
            if rc != 0:
                raise RuntimeError(f"board upload failed for {f.name} (rc={rc})")
            uploaded.append(f.name)
    return uploaded


def _export_one(sport):
    if sport not in SPORTS:
        return {"sport": sport, "ok": False, "error": f"unknown sport '{sport}'", "known": sorted(SPORTS)}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _sync_artifacts()
        _sync_sport_data(sport)
    except Exception as exc:  # noqa: BLE001
        return {"sport": sport, "ok": False, "error": f"input sync failed: {exc}"}

    start = time.time()
    last_rc, last_log = 0, ""
    for cmd in SPORTS[sport]["cmds"]:
        last_rc, last_log = _run(cmd, DIV, RUN_TIMEOUT_SEC)
        if last_rc != 0:
            break  # stop the chain, but still upload any boards already written

    # Upload whatever boards this run produced, even on partial failure. Each board is
    # written atomically at the end of its phase, so e.g. if the MLB *upcoming* phase
    # OOMs (exit -9), the *historical* board from phase 1 is already complete on disk —
    # publish it rather than losing the whole run.
    try:
        boards = _upload_boards(start)
    except Exception as exc:  # noqa: BLE001
        return {"sport": sport, "ok": False, "exit_code": last_rc, "boards": [], "error": f"board upload failed: {exc}", "log_tail": last_log[-1500:]}
    return {"sport": sport, "ok": last_rc == 0, "exit_code": last_rc, "boards": boards, "log_tail": last_log[-1500:]}


def handler(job):
    payload = job.get("input") or {}
    if not isinstance(payload, dict):
        raise ValueError("job.input must be an object")
    jtype = str(payload.get("type") or "sports_export").lower()

    if jtype == "health":
        rc, out = _run(["aws", "--version"], REPO, 30)
        return {"ok": rc == 0, "aws_cli": out.strip()[:120], "sports": sorted(SPORTS), "bucket": S3_BUCKET}

    if jtype == "sports_export":
        sports = payload.get("sports")
        if not sports:
            one = payload.get("sport")
            sports = [one] if one else ["mlb"]
        if isinstance(sports, str):
            sports = [sports]
        if [s.lower() for s in sports] == ["all"]:
            sports = list(SPORTS)
        results = [_export_one(str(s).lower()) for s in sports]
        return {"ok": all(r.get("ok") for r in results), "results": results}

    raise ValueError(f"Unsupported type: {jtype!r} (expected sports_export | health)")


runpod.serverless.start({"handler": handler})
