import pandas as pd
from pathlib import Path

def combine_golf_tours():
    div_root = Path("pythia_divination")
    pga_path = div_root / "data/sports/pga/normalized/pga_training_dataset_latest.csv"
    lpga_path = div_root / "data/sports/pga/normalized/lpga_results_combined.csv"
    
    if not pga_path.exists():
        print("PGA dataset not found.")
        return
        
    pga = pd.read_csv(pga_path, low_memory=False)
    pga["tour"] = "PGA"
    
    if lpga_path.exists():
        lpga = pd.read_csv(lpga_path, low_memory=False)
        lpga["tour"] = "LPGA"
        
        # Align columns
        # PGA has 800+ columns, LPGA has ~15.
        # We will keep all PGA columns and fill LPGA missing ones with NaN.
        combined = pd.concat([pga, lpga], ignore_index=True, sort=False)
        
        # Save combined
        out_path = div_root / "data/sports/pga/normalized/golf_training_dataset_latest.csv"
        combined.to_csv(out_path, index=False)
        print(f"Combined {len(pga)} PGA and {len(lpga)} LPGA rows.")
        print(f"Saved to {out_path}")
    else:
        print("LPGA results not found.")

if __name__ == "__main__":
    combine_golf_tours()
