import pandas as pd
from pathlib import Path
import numpy as np

def convert_wta_2025():
    div_root = Path("pythia_divination")
    xlsx_path = Path("2025_wta.xlsx")
    players_path = div_root / "data/sports/wta/raw/wta_players.csv"
    
    if not xlsx_path.exists() or not players_path.exists():
        print("Missing required files for WTA 2025 conversion.")
        return
        
    df = pd.read_excel(xlsx_path)
    players = pd.read_csv(players_path, low_memory=False)
    players["full_name"] = players["name_first"] + " " + players["name_last"]
    
    # Simple name-to-ID mapping
    player_map = dict(zip(players["full_name"], players["player_id"]))
    
    # Manual fixes for common name mismatches if needed
    # player_map["Name In XLSX"] = ID
    
    rows = []
    print(f"Total rows in XLSX: {len(df)}")
    for i, row in df.iterrows():
        try:
            w_name = str(row["Winner"]).strip()
            l_name = str(row["Loser"]).strip()
            
            def find_id(name):
                # Format: "Hibino N."
                parts = name.replace(".", " ").split()
                if not parts: return 0
                last = parts[0]
                initial = parts[-1][0] if len(parts) > 1 else ""
                
                matches = players[players["name_last"].str.lower() == last.lower()]
                if i < 5:
                    print(f"Looking for {name} -> last={last}, initial={initial}. Matches: {len(matches)}")
                
                if matches.empty:
                    matches = players[players["name_last"].str.contains(last, na=False, case=False)]
                
                if len(matches) == 1:
                    return matches.iloc[0]["player_id"]
                elif len(matches) > 1 and initial:
                    sub = matches[matches["name_first"].str.startswith(initial, na=False)]
                    if not sub.empty:
                        return sub.iloc[0]["player_id"]
                return 0

            w_id = find_id(w_name)
            l_id = find_id(l_name)
            if i < 5:
                print(f"Row {i}: {w_name}({w_id}) vs {l_name}({l_id})")
            
            if w_id == 0 or l_id == 0: continue
            
            # Format date to YYYYMMDD
            t_date = pd.to_datetime(row["Date"]).strftime("%Y%m%d")
            
            tourney_level = str(row.get("Tier", row.get("Series", "WTA")))
            
            rows.append({
                "tourney_id": f"2025-W-{row['Tournament'][:10]}",
                "tourney_name": row["Tournament"],
                "surface": row["Surface"],
                "draw_size": 32, # default
                "tourney_level": tourney_level,
                "tourney_date": int(t_date),
                "match_num": 1, 
                "winner_id": w_id,
                "winner_name": w_name,
                "loser_id": l_id,
                "loser_name": l_name,
                "winner_rank": row["WRank"],
                "loser_rank": row["LRank"],
                "tour": "WTA",
                # Technical stats are missing in tennis-data, we'll use NaN
                # Our model will imputer them with medians
                "w_svpt": np.nan, "w_1stIn": np.nan, "w_1stWon": np.nan, "w_2ndWon": np.nan, "w_ace": np.nan, "w_df": np.nan,
                "l_svpt": np.nan, "l_1stIn": np.nan, "l_1stWon": np.nan, "l_2ndWon": np.nan, "l_ace": np.nan, "l_df": np.nan,
            })
        except Exception:
            continue
            
    out_df = pd.DataFrame(rows)
    out_path = div_root / "data/sports/wta/raw/wta_matches_2025.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Converted {len(out_df)} WTA 2025 matches.")

if __name__ == "__main__":
    convert_wta_2025()
