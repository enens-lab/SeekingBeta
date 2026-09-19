#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Season-aware DAILY sports refresh (RunPod).
# ==============================================================================
# Daily-cadence sports (MLB, soccer, tennis) need their *upcoming* boards
# refreshed every day — the BFF runtime feed only serves not-yet-started games,
# so a weekly-only export leaves `upcoming` empty within a day or two. But running
# them year-round wastes RunPod runs on off-season sports (empty boards).
#
# This wrapper computes which sports are IN-SEASON today and refreshes only those
# (skipping RunPod entirely if none are). A sport auto-rejoins when its window
# opens — no yearly crontab edits. Tune the SEASONS map as schedules shift.
#
#   ops/refresh_sports_daily.sh            # refresh today's in-season daily sports
#   ops/refresh_sports_daily.sh --dry-run  # print the selection, trigger nothing
# ==============================================================================
REPO="${SEEKINGBETA_REPO:-/home/ec2-user/seekingbeta}"
cd "$REPO"

ACTIVE=$(python3 - <<'PY'
import datetime, json
md = (lambda d: (d.month, d.day))(datetime.datetime.now(datetime.timezone.utc).date())
# sport -> ((start_m, start_d), (end_m, end_d)); end < start means the window wraps the new year.
SEASONS = {
    "mlb":    ((3, 20), (11, 5)),    # opening day -> end of postseason
    "soccer": ((1, 1),  (12, 31)),   # leagues + tournaments, effectively year-round
    "tennis": ((1, 1),  (11, 25)),   # Australian Open -> Tour Finals (coarse; tournaments cluster)
    # basketball: WNBA May-Oct + NBA mid-Oct-June overlap into a year-round window.
    #   (The NBA CDN 403 that excluded it was fixed with browser headers, commit 2d71c29.)
    "basketball": ((1, 1), (12, 31)),
    # football: NFL week 1 -> Super Bowl. Games are weekly but the runtime feed only
    #   shows not-yet-started games, so the slate must be re-exported daily in season.
    "football": ((9, 1), (2, 15)),
    # golf weekly; olympics static -> the weekly cron handles them.
}
def in_season(s, e):
    return (s <= md <= e) if s <= e else (md >= s or md <= e)
print(json.dumps([sp for sp, (s, e) in SEASONS.items() if in_season(s, e)]))
PY
)

COUNT=$(printf '%s' "$ACTIVE" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)))')

if [ "${1:-}" = "--dry-run" ]; then
  echo "[daily] $(date -u +%FT%TZ) in-season today: $ACTIVE ($COUNT sport(s)) — dry run, nothing triggered"
  exit 0
fi

if [ "$COUNT" -eq 0 ]; then
  echo "[daily] $(date -u +%FT%TZ) no daily sports in season; skipping RunPod"
  exit 0
fi

echo "[daily] $(date -u +%FT%TZ) in-season: $ACTIVE"
exec ops/refresh_sports_runpod.sh "$ACTIVE"
