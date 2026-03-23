import pandas as pd
import json
from pathlib import Path

# Load data
preds = pd.read_csv('pythia_divination/artifacts/pga_tournament_ranker_torch/validation_predictions.csv')
meta = pd.read_csv('pythia_divination/data/sports/pga/normalized/golf_training_dataset_latest.csv', low_memory=False)
meta = meta[['tournament_id', 'tournament_name', 'season_year', 'display_date', 'course_name', 'course_state_code', 'tour']].drop_duplicates(subset=['tournament_id'])

# Merge
df = preds.merge(meta, on='tournament_id', how='left')

# 1. Generate Historical Backtests (2024-2025 validation set)
# We will evaluate Top 1, Top 3, and Top 5 predictions against the actual winner.
backtests = []
for t_id, group in df.groupby('tournament_id'):
    try:
        t_name = group['tournament_name_x'].iloc[0] if 'tournament_name_x' in group.columns else group['tournament_name'].iloc[0]
        year = group['season_year'].iloc[0]
        tour = group['tour'].iloc[0]
        
        # Sort by prediction
        top_preds = group.sort_values('winner_probability', ascending=False)
        
        pred_top_1 = top_preds.iloc[0]['player_name']
        pred_top_3 = top_preds.head(3)['player_name'].tolist()
        pred_top_5 = top_preds.head(5)['player_name'].tolist()
        
        # Actual winner
        actual_winners = group[group['won'] == 1]
        if not actual_winners.empty:
            actual_winner = actual_winners.iloc[0]['player_name']
        else:
            actual_winner = "Unknown"
            
        hit_top_1 = actual_winner == pred_top_1
        hit_top_3 = actual_winner in pred_top_3
        hit_top_5 = actual_winner in pred_top_5
        
        # Determine the best hit category to display
        hit_status = "Miss"
        if hit_top_1:
            hit_status = "Top Pick"
        elif hit_top_3:
            hit_status = "Top 3"
        elif hit_top_5:
            hit_status = "Top 5"
            
        # Store full field for detail view
        full_field = []
        for rank, (_, row) in enumerate(top_preds.iterrows(), 1):
            full_field.append({
                'rank': rank,
                'playerName': row['player_name'],
                'winProbability': float(row['winner_probability'] * 100),
                'actualWinner': row['won'] == 1
            })
        
        backtests.append({
            'year': int(year) if pd.notna(year) else 2024,
            'tournament': t_name,
            'tour': tour,
            'predictedWinner': pred_top_1,
            'predictedTop3': pred_top_3,
            'predictedTop5': pred_top_5,
            'actualWinner': actual_winner,
            'hitStatus': hit_status,
            'prob': float(top_preds.iloc[0]['winner_probability']),
            'fullField': full_field
        })
    except Exception as e:
        pass

# Sort backtests by year desc, then hit desc
backtests = sorted(backtests, key=lambda x: (-x['year'], -x['prob']))

with open('pythia_prophecy/frontend/src/data/historical_backtests.json', 'w') as f:
    json.dump(backtests, f, indent=2) # Include all backtests


# 2. Generate Upcoming Tournaments (Mocking 2026 using the latest instance of ALL unique tournaments)
upcoming_data = []

# Get all unique tournament names from the dataset to build the 2026 schedule
unique_tournaments = df[['tournament_name_x', 'tour']].drop_duplicates().values

for event_name, tour in unique_tournaments:
    if pd.isna(event_name): continue
    # Find the most recent instance of this event in the predictions
    matches = df[(df['tournament_name_x'] == event_name) & (df['tour'] == tour)]
    if not matches.empty:
        # Get the latest year
        latest_year = matches['season_year'].max()
        latest_event = matches[matches['season_year'] == latest_year].copy()
        
        # Get ALL predictions (not just top 15) to support "View All"
        all_preds = latest_event.sort_values('winner_probability', ascending=False)
        
        predictions = []
        for rank, (_, row) in enumerate(all_preds.iterrows(), 1):
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
            'id': f"{tour.lower()}-{event_name.replace(' ', '-').lower()}",
            'name': f"2026 {event_name}",
            'original_name': event_name,
            'tour': tour,
            'course': f"{course}, {state}".strip(", "),
            'predictions': predictions
        })

# Sort so Majors are near the top, followed by others
def sort_priority(event):
    name = event['name'].lower()
    if 'masters' in name: return 1
    if 'pga championship' in name: return 2
    if 'u.s. open' in name: return 3
    if 'open championship' in name: return 4
    if 'players championship' in name: return 5
    return 10

upcoming_data = sorted(upcoming_data, key=lambda x: (sort_priority(x), x['name']))

with open('pythia_prophecy/frontend/src/data/upcoming_tournaments.json', 'w') as f:
    json.dump(upcoming_data, f, indent=2)

print(f"Generated {len(backtests)} backtests and {len(upcoming_data)} upcoming tournaments.")
