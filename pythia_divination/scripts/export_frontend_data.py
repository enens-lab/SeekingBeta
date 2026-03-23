import pandas as pd
import json
from pathlib import Path

# Load data
preds = pd.read_csv('pythia_divination/artifacts/pga_tournament_ranker_torch/validation_predictions.csv')
meta = pd.read_csv('pythia_divination/data/sports/pga/normalized/pga_training_dataset_latest.csv', low_memory=False)
meta = meta[['tournament_id', 'tournament_name', 'season_year', 'display_date', 'course_name', 'course_state_code']].drop_duplicates(subset=['tournament_id'])

# Merge
df = preds.merge(meta, on='tournament_id', how='left')

# 1. Generate Historical Backtests (2024-2025 validation set)
# We will show the top predicted player vs the actual winner.
backtests = []
for t_id, group in df.groupby('tournament_id'):
    try:
        t_name = group['tournament_name_x'].iloc[0] if 'tournament_name_x' in group.columns else group['tournament_name'].iloc[0]
        year = group['season_year'].iloc[0]
        
        # Sort by prediction
        top_pred = group.sort_values('winner_probability', ascending=False).iloc[0]
        pred_winner = top_pred['player_name']
        
        # Actual winner
        actual_winners = group[group['won'] == 1]
        if not actual_winners.empty:
            actual_winner = actual_winners.iloc[0]['player_name']
        else:
            actual_winner = "Unknown"
            
        hit = (pred_winner == actual_winner)
        
        # To keep the list clean, let's only include events where the top-pick probability was > 5% or it was a major
        if top_pred['winner_probability'] > 0.05 or 'Masters' in t_name or 'Open' in t_name or 'Championship' in t_name:
             backtests.append({
                 'year': int(year) if pd.notna(year) else 2024,
                 'tournament': t_name,
                 'predictedWinner': pred_winner,
                 'actualWinner': actual_winner,
                 'hit': bool(hit),
                 'prob': float(top_pred['winner_probability'])
             })
    except Exception as e:
        pass

# Sort backtests by year desc, then hit desc
backtests = sorted(backtests, key=lambda x: (-x['year'], -x['prob']))

with open('pythia_prophecy/frontend/src/data/historical_backtests.json', 'w') as f:
    json.dump(backtests[:50], f, indent=2) # Keep top 50 for the UI


# 2. Generate Upcoming Tournaments (Mocking 2026 using 2025 data for UI demonstration)
upcoming_events = [
    'Masters Tournament',
    'PGA Championship',
    'U.S. Open',
    'The Open Championship',
    'THE PLAYERS Championship',
    'Arnold Palmer Invitational',
    'The Genesis Invitational'
]

upcoming_data = []

for event_name in upcoming_events:
    # Find the most recent instance of this event in the predictions
    matches = df[df['tournament_name_x'].str.contains(event_name, na=False, case=False)]
    if not matches.empty:
        # Get the latest year
        latest_year = matches['season_year'].max()
        latest_event = matches[matches['season_year'] == latest_year].copy()
        
        # Get top 15 predictions
        top_15 = latest_event.sort_values('winner_probability', ascending=False).head(15)
        
        predictions = []
        for rank, (_, row) in enumerate(top_15.iterrows(), 1):
            predictions.append({
                'rank': rank,
                'playerName': row['player_name'],
                'winProbability': float(row['winner_probability'] * 100)
            })
            
        course = latest_event['course_name'].iloc[0]
        state = latest_event['course_state_code'].iloc[0]
        if pd.isna(course): course = "TBD Course"
        if pd.isna(state): state = ""
            
        upcoming_data.append({
            'id': event_name.replace(' ', '-').lower(),
            'name': f"2026 {event_name}",
            'course': f"{course}, {state}".strip(", "),
            'predictions': predictions
        })

with open('pythia_prophecy/frontend/src/data/upcoming_tournaments.json', 'w') as f:
    json.dump(upcoming_data, f, indent=2)

print(f"Generated {len(backtests)} backtests and {len(upcoming_data)} upcoming tournaments.")
