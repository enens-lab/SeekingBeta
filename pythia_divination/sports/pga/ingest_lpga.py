"""Ingest LPGA tournament results from ESPN."""

import requests
import pandas as pd
import json
import logging
from pathlib import Path
import time

# Ensure repo root is on sys.path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.pga.storage import DEFAULT_PGA_DATA_ROOT

DEFAULT_LPGA_DATA_ROOT = DEFAULT_PGA_DATA_ROOT / "lpga"

logger = logging.getLogger(__name__)

def fetch_espn_schedule(year: int):
    url = f"https://site.api.espn.com/apis/site/v2/sports/golf/lpga/scoreboard?dates={year}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_espn_leaderboard(event_id: str):
    url = f"https://site.api.espn.com/apis/site/v2/sports/golf/leaderboard?league=lpga&event={event_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def ingest_lpga_results(years: list[int]):
    all_results = []
    
    for year in years:
        logger.info(f"Fetching LPGA schedule for {year}...")
        try:
            schedule = fetch_espn_schedule(year)
            events = schedule.get("leagues", [{}])[0].get("calendar", [])
            
            for cal_item in events:
                event_id = cal_item.get("id")
                event_name = cal_item.get("label")
                logger.info(f"  Fetching leaderboard for {event_name} ({event_id})...")
                
                try:
                    lb_data = fetch_espn_leaderboard(event_id)
                    event_meta = lb_data.get("events", [{}])[0]
                    competitions = event_meta.get("competitions", [{}])[0]
                    competitors = competitions.get("competitors", [])
                    
                    for comp in competitors:
                        athlete = comp.get("athlete", {})
                        status = comp.get("status", {})
                        
                        pos = status.get("position", {}).get("displayName")
                        pos_numeric = status.get("position", {}).get("id") # ESPN often has numeric ID
                        try:
                            pos_val = int(pos_numeric) if pos_numeric and str(pos_numeric).isdigit() else None
                        except:
                            pos_val = None
                            
                        score = comp.get("score", {}).get("value")
                        
                        all_results.append({
                            "season_year": year,
                            "tournament_id": f"LPGA-{event_id}",
                            "tournament_name": event_name,
                            "player_id": str(athlete.get("id")),
                            "player_name": athlete.get("displayName"),
                            "country": athlete.get("flag", {}).get("alt"),
                            "position": pos,
                            "position_numeric": pos_val,
                            "total_score": score,
                            "won": bool(pos_val == 1),
                            "tour": "LPGA"
                        })
                    time.sleep(0.5) # rate limit
                except Exception as e:
                    logger.error(f"Failed to fetch event {event_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch schedule for {year}: {e}")
            
    df = pd.DataFrame(all_results)
    
    # Calculate form features
    df = df.sort_values(["player_id", "season_year", "tournament_id"]) # simple sort
    grouped = df.groupby("player_id")
    df["player_won_mean_last_5"] = grouped["won"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["player_top_10_mean_last_5"] = grouped["position_numeric"].transform(
        lambda x: (x.shift(1) <= 10).rolling(5, min_periods=1).mean()
    )
    
    output_path = DEFAULT_PGA_DATA_ROOT / "normalized" / "lpga_results_combined.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} LPGA result rows to {output_path}")
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ingest_lpga_results([2020, 2021, 2022, 2023, 2024, 2025])
