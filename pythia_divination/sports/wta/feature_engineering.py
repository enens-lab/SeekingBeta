"""Feature engineering for WTA (Tennis) prediction models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class EloConfig:
    k_factor: float = 32
    initial_elo: float = 1500.0

def calculate_elo_updates(
    winner_elo: float,
    loser_elo: float,
    k: float = 32
) -> tuple[float, float]:
    """Calculate the new Elo ratings after a match."""
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 - expected_winner
    
    new_winner_elo = winner_elo + k * (1 - expected_winner)
    new_loser_elo = loser_elo + k * (0 - expected_loser)
    
    return new_winner_elo, new_loser_elo

def process_match_history(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate through matches chronologically to calculate rolling stats and Elo.
    Returns a dataframe of match results with pre-match features for both players.
    """
    # Sort by date and match number
    df = matches.sort_values(["tourney_date", "match_num"]).copy()
    
    # Initialize Elo
    player_elos: dict[int, float] = {}
    surface_elos: dict[str, dict[int, float]] = {
        "Hard": {}, "Clay": {}, "Grass": {}, "Carpet": {}
    }
    
    rows = []
    
    for idx, row in df.iterrows():
        w_id = int(row["winner_id"])
        l_id = int(row["loser_id"])
        surface = row["surface"]
        
        # Get current elos
        w_elo = player_elos.get(w_id, 1500.0)
        l_elo = player_elos.get(l_id, 1500.0)
        
        w_surf_elo = surface_elos.get(surface, {}).get(w_id, 1500.0)
        l_surf_elo = surface_elos.get(surface, {}).get(l_id, 1500.0)
        
        # Store pre-match data
        match_feat = row.to_dict()
        match_feat["w_pre_match_elo"] = w_elo
        match_feat["l_pre_match_elo"] = l_elo
        match_feat["w_pre_match_surf_elo"] = w_surf_elo
        match_feat["l_pre_match_surf_elo"] = l_surf_elo
        
        rows.append(match_feat)
        
        # Update ratings
        new_w_elo, new_l_elo = calculate_elo_updates(w_elo, l_elo)
        player_elos[w_id] = new_w_elo
        player_elos[l_id] = new_l_elo
        
        if surface in surface_elos:
            new_w_surf, new_l_surf = calculate_elo_updates(w_surf_elo, l_surf_elo, k=16) # Lower K for surface
            surface_elos[surface][w_id] = new_w_surf
            surface_elos[surface][l_id] = new_l_surf
            
    return pd.DataFrame(rows)

def build_player_event_features(match_history: pd.DataFrame) -> pd.DataFrame:
    """
    Transform match-level data into player-event-level features.
    For each tournament, we want one row per player representing their state 
    BEFORE the tournament began.
    """
    df = match_history.copy()
    
    # For each player-tournament, get the state from their first match of the week
    winners = df[["tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date", 
                  "winner_id", "winner_name", "w_pre_match_elo", "w_pre_match_surf_elo"]].copy()
    winners.columns = ["tournament_id", "tournament_name", "surface", "level", "date", 
                       "player_id", "player_name", "elo", "surf_elo"]
    
    losers = df[["tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date", 
                 "loser_id", "loser_name", "l_pre_match_elo", "l_pre_match_surf_elo"]].copy()
    losers.columns = ["tournament_id", "tournament_name", "surface", "level", "date", 
                      "player_id", "player_name", "elo", "surf_elo"]
    
    player_events = pd.concat([winners, losers]).drop_duplicates(subset=["tournament_id", "player_id"], keep="first")
    
    # Calculate who actually won the tournament
    # The winner of the final (last match_num in the tournament) is the champion
    finals = df.sort_values(["tourney_id", "match_num"]).groupby("tourney_id").tail(1)
    champions = dict(zip(finals["tourney_id"], finals["winner_id"]))
    
    player_events["won_tournament"] = player_events.apply(
        lambda x: 1 if champions.get(x["tournament_id"]) == x["player_id"] else 0, axis=1
    )
    
    return player_events

def add_rolling_features(player_events: pd.DataFrame) -> pd.DataFrame:
    """Add momentum and form features."""
    df = player_events.sort_values(["player_id", "date"]).copy()
    grouped = df.groupby("player_id")
    
    # Rolling tournament win rate
    df["player_rolling_win_rate_5"] = grouped["won_tournament"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    
    # Elo momentum
    df["player_elo_diff_5"] = grouped["elo"].transform(lambda x: x - x.shift(5))
    
    return df
