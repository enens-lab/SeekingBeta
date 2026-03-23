import pandas as pd
import json
from pathlib import Path
import torch
import joblib
import numpy as np

# Ensure repo root is on sys.path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.wta.train_tournament_ranker_torch import WTATournamentRanker

def export_wta_frontend_data():
    div_root = Path(__file__).resolve().parents[1]
    artifact_dir = div_root / "artifacts" / "wta_tournament_ranker_torch"
    dataset_path = div_root / "data" / "sports" / "wta" / "normalized" / "wta_training_dataset_latest.csv"
    
    df = pd.read_csv(dataset_path)
    imputer = joblib.load(artifact_dir / "imputer.joblib")
    scaler = joblib.load(artifact_dir / "scaler.joblib")
    
    # Load model
    with open(artifact_dir / "feature_columns.json", "r") as f:
        static_columns = json.load(f)
        
    model = WTATournamentRanker(
        static_feature_count=len(static_columns),
        static_width=256,
        dense_width=128,
        dropout=0.3
    )
    model.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu"))
    model.eval()
    
    # 1. Historical Backtests
    # Get recent tournaments from validation (last 20%)
    tourneys = df["tournament_id"].unique()
    val_ids = tourneys[int(len(tourneys) * 0.8):]
    val_df = df[df["tournament_id"].isin(val_ids)].copy()
    
    backtests = []
    for t_id, group in val_df.groupby("tournament_id", sort=False):
        try:
            X_raw = group[static_columns]
            X = scaler.transform(imputer.transform(X_raw))
            X_tensor = torch.from_numpy(X).float().unsqueeze(0) # batch of 1
            
            with torch.no_grad():
                logits = model(X_tensor)["winner"]
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
            
            sorted_idx = np.argsort(-probs)
            top_winner = group.iloc[sorted_idx[0]]
            
            actual_winner_row = group[group["won_tournament"] == 1]
            if actual_winner_row.empty: continue
            actual_winner = actual_winner_row.iloc[0]["player_name"]
            
            # Form field info
            full_field = []
            for idx in sorted_idx:
                p_row = group.iloc[idx]
                full_field.append({
                    "rank": len(full_field) + 1,
                    "playerName": str(p_row["player_name"]),
                    "winProbability": float(probs[idx] * 100),
                    "actualWinner": bool(p_row["won_tournament"] == 1)
                })
            
            backtests.append({
                "year": int(group["date"].iloc[0] // 10000),
                "tournament": str(group["tournament_name"].iloc[0]),
                "tour": str(group["tour"].iloc[0]), # Added tour field
                "predictedWinner": str(top_winner["player_name"]),
                "predictedTop3": [str(group.iloc[sorted_idx[i]]["player_name"]) for i in range(min(3, len(sorted_idx)))],
                "predictedTop5": [str(group.iloc[sorted_idx[i]]["player_name"]) for i in range(min(5, len(sorted_idx)))],
                "actualWinner": str(actual_winner),
                "hitStatus": "Top Pick" if top_winner["player_name"] == actual_winner else ("Top 3" if actual_winner in [group.iloc[sorted_idx[i]]["player_name"] for i in range(min(3, len(sorted_idx)))] else ("Top 5" if actual_winner in [group.iloc[sorted_idx[i]]["player_name"] for i in range(min(5, len(sorted_idx)))] else "Miss")),
                "prob": float(probs[sorted_idx[0]]),
                "fullField": full_field
            })
        except Exception:
            continue
            
    # Sort by date desc
    backtests = sorted(backtests, key=lambda x: (-x["year"], x["tournament"]))
    
    proj_root = Path(__file__).resolve().parents[2]
    with open(proj_root / "pythia_prophecy/frontend/src/data/wta_historical_backtests.json", "w") as f:
        json.dump(backtests, f, indent=2)
        
    # 2. Upcoming (Mocking 2026 with recent 2024 versions)
    # We will just take the latest unique tournaments and label them 2026
    upcoming = []
    seen_tourneys = set()
    for bt in backtests:
        t_key = (bt["tournament"], bt["tour"])
        if t_key in seen_tourneys: continue
        seen_tourneys.add(t_key)
        
        up = bt.copy()
        up["name"] = f"2026 {bt['tournament']}"
        up["id"] = f"{bt['tour'].lower()}-{bt['tournament'].replace(' ', '-').lower()}"
        up["course"] = "Hard Court" if "Open" in bt["tournament"] else "Clay Court"
        up["predictions"] = bt["fullField"]
        upcoming.append(up)
        if len(upcoming) >= 20: break
        
    with open(proj_root / "pythia_prophecy/frontend/src/data/wta_upcoming_tournaments.json", "w") as f:
        json.dump(upcoming, f, indent=2)
        
    print(f"Exported {len(backtests)} WTA backtests and {len(upcoming)} upcoming.")

if __name__ == "__main__":
    export_wta_frontend_data()
