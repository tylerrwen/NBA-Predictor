

import time
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Try to import XGBoost for better accuracy
try:
    import xgboost as xgb
    USE_XGBOOST = True
except ImportError:
    USE_XGBOOST = False

# import scrapers from your modules (must exist in same folder)
from scrapingStats import scrapingStats
from scrapingStandings import get_team_standing

TEAMS_ALL = [
    "ATL","BOS","BRK","CHO","CHI","CLE","DAL","DEN","DET","GSW",
    "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
    "OKC","ORL","PHI","PHO","POR","SAC","SAS","TOR","UTA","WAS"
]

# Team name mapping for common abbreviations (scraped data may use different abbreviations)
TEAM_NAME_MAP = {
    "PHX": "PHO",  # Phoenix Suns - scraped data uses PHO
    "BKN": "BRK",  # Brooklyn Nets - scraped data uses BRK
    "CHA": "CHO",  # Charlotte Hornets - scraped data uses CHO
    "CHO": "CHO",  # Explicitly allow CHO input
}

# Factors to surface in the UI with "better" direction and display settings
STAT_FACTORS_CONFIG = [
    {"key": "pt_diff", "label": "Point Differential", "better": "higher", "decimals": 1, "multiplier": 1},
    {"key": "pts_for_avg", "label": "Points For", "better": "higher", "decimals": 1, "multiplier": 1},
    {"key": "fg_pct_avg", "label": "Field Goal %", "better": "higher", "decimals": 1, "multiplier": 100},
    {"key": "fg3_pct_avg", "label": "3PT %", "better": "higher", "decimals": 1, "multiplier": 100},
    {"key": "trb_avg", "label": "Rebounds", "better": "higher", "decimals": 1, "multiplier": 1},
    {"key": "ast_avg", "label": "Assists", "better": "higher", "decimals": 1, "multiplier": 1},
    {"key": "tov_avg", "label": "Turnovers", "better": "lower", "decimals": 1, "multiplier": 1},
    {"key": "opp_fg_pct_avg", "label": "Opponent FG %", "better": "lower", "decimals": 1, "multiplier": 100},
]

SEASON = 2026  # change to 2025 if needed
MIN_SEASON = 2000  # for fallback search
REQUEST_DELAY = 1.0  # seconds between requests to be polite
NUM_SEASONS = 1  # Number of seasons to use for training (current + previous seasons)
RECENT_FORM_WINDOW = 5  # Number of recent games to use for form calculation

# Model persistence paths (saved in NBAPredictor/saved_models folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # Go up one level to NBAPredictor folder
MODEL_DIR = os.path.join(PARENT_DIR, "saved_models")
MODEL_FILE = os.path.join(MODEL_DIR, "nba_predictor_model.pkl")
DATA_CACHE_FILE = os.path.join(MODEL_DIR, "games_data_cache.pkl")
TEAMS_CACHE_FILE = os.path.join(MODEL_DIR, "teams_data_cache.pkl")


def to_num(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s == ".":
        return np.nan
    try:
        return float(s.replace(",", ""))
    except:
        # fallback: keep digits, dot, minus
        cleaned = "".join(ch for ch in s if (ch.isdigit() or ch in ".-"))
        try:
            return float(cleaned) if cleaned not in ("", ".", "-") else np.nan
        except:
            return np.nan


def save_model_and_data(model, feature_cols, games_df, teams_df, season):
    """Save the trained model and data to disk for future use."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_data = {
        "model": model,
        "feature_cols": feature_cols,
        "season": season,
        "model_version": 2,  # Version 2 includes recent form features
        "num_seasons": NUM_SEASONS,
        "recent_form_window": RECENT_FORM_WINDOW
    }
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)
    
    with open(DATA_CACHE_FILE, "wb") as f:
        pickle.dump(games_df, f)
    
    with open(TEAMS_CACHE_FILE, "wb") as f:
        pickle.dump(teams_df, f)
    
    # Model and data persisted for reuse


def load_model_and_data():
    """Load saved model and data from disk if they exist. Returns (model, feature_cols, games_df, teams_df, season)."""
    if not os.path.exists(MODEL_FILE):
        return None, None, None, None, None
    
    try:
        with open(MODEL_FILE, "rb") as f:
            model_data = pickle.load(f)
        
        with open(DATA_CACHE_FILE, "rb") as f:
            games_df = pickle.load(f)
        
        with open(TEAMS_CACHE_FILE, "rb") as f:
            teams_df = pickle.load(f)
        
        season = model_data.get("season", None)
        return model_data["model"], model_data["feature_cols"], games_df, teams_df, season
    except Exception:
        return None, None, None, None, None


def load_existing_games_data():
    """Load existing games data from cache if it exists. Returns games_df or None."""
    if not os.path.exists(DATA_CACHE_FILE):
        return None
    
    try:
        with open(DATA_CACHE_FILE, "rb") as f:
            games_df = pickle.load(f)
        return games_df
    except Exception:
        return None


def merge_games_data(existing_df, new_df):
    """
    Merge new games with existing games, avoiding duplicates.
    Returns the combined dataframe with duplicates removed.
    """
    if existing_df is None or existing_df.empty:
        return new_df.copy() if new_df is not None and not new_df.empty else pd.DataFrame()
    
    if new_df is None or new_df.empty:
        return existing_df.copy()
    
    # Ensure date column is datetime for both
    for df in [existing_df, new_df]:
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
            except:
                pass
    
    # Combine the dataframes
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates based on date, home_team, and away_team
    # Keep the most recent entry if there are duplicates
    combined = combined.sort_values('date', ascending=False).drop_duplicates(
        subset=['date', 'home_team', 'away_team'],
        keep='first'
    ).sort_values('date', ascending=True).reset_index(drop=True)
    
    return combined


def find_working_season(team, start_season, min_season=MIN_SEASON):
    """
    Try start_season down to min_season until scrapingStats returns non-empty list.
    Returns the first season that yields gamelogs (or None).
    """
    for s in range(start_season, min_season - 1, -1):
        time.sleep(REQUEST_DELAY)
        try:
            games = scrapingStats(team, s)
        except Exception:
            games = []
        if games:
            return s
    return None


def build_games_dataframe(teams, season):
    all_games = []
    seen_pairs = set()  # dedupe by (date, home, away)

    for team in teams:
        # polite delay
        time.sleep(REQUEST_DELAY)
        try:
            games = scrapingStats(team, season)
        except Exception:
            games = []

        if not games:
            continue

        for g in games:
            # identify keys commonly present
            team_pts = g.get("team_game_score") or g.get("pts") or g.get("team_pts") or g.get("team_game_score_home")
            opp_pts = g.get("opp_team_game_score") or g.get("opp_pts") or g.get("opp_team_game_score")
            opp = g.get("opp_name_abbr") or g.get("opp_name") or g.get("opp")
            date = g.get("date") or g.get("date_game") or g.get("game_date")

            # home flag from your scraped dict (boolean) or game_location
            home_flag = g.get("home")
            if home_flag is None:
                loc = g.get("game_location", "")
                home_flag = (loc != "@")

            if not opp or date is None:
                continue

            # canonicalize
            team_upper = team.upper()
            opp_upper = opp.upper() if isinstance(opp, str) else opp

            # Extract additional stats
            if home_flag:
                home_team = team_upper
                away_team = opp_upper
                home_points = to_num(team_pts)
                away_points = to_num(opp_pts)
                # Team stats (home team)
                home_fg_pct = to_num(g.get("fg_pct"))
                home_fg3_pct = to_num(g.get("fg3_pct"))
                home_ft_pct = to_num(g.get("ft_pct"))
                home_trb = to_num(g.get("trb"))
                home_ast = to_num(g.get("ast"))
                home_stl = to_num(g.get("stl"))
                home_blk = to_num(g.get("blk"))
                home_tov = to_num(g.get("tov"))
                home_orb = to_num(g.get("orb"))
                home_drb = to_num(g.get("drb"))
                # Opponent stats (away team)
                away_fg_pct = to_num(g.get("opp_fg_pct"))
                away_fg3_pct = to_num(g.get("opp_fg3_pct"))
                away_ft_pct = to_num(g.get("opp_ft_pct"))
                away_trb = to_num(g.get("opp_trb"))
                away_ast = to_num(g.get("opp_ast"))
                away_stl = to_num(g.get("opp_stl"))
                away_blk = to_num(g.get("opp_blk"))
                away_tov = to_num(g.get("opp_tov"))
                away_orb = to_num(g.get("opp_orb"))
                away_drb = to_num(g.get("opp_drb"))
            else:
                home_team = opp_upper
                away_team = team_upper
                home_points = to_num(opp_pts)
                away_points = to_num(team_pts)
                # Opponent stats (home team)
                home_fg_pct = to_num(g.get("opp_fg_pct"))
                home_fg3_pct = to_num(g.get("opp_fg3_pct"))
                home_ft_pct = to_num(g.get("opp_ft_pct"))
                home_trb = to_num(g.get("opp_trb"))
                home_ast = to_num(g.get("opp_ast"))
                home_stl = to_num(g.get("opp_stl"))
                home_blk = to_num(g.get("opp_blk"))
                home_tov = to_num(g.get("opp_tov"))
                home_orb = to_num(g.get("opp_orb"))
                home_drb = to_num(g.get("opp_drb"))
                # Team stats (away team)
                away_fg_pct = to_num(g.get("fg_pct"))
                away_fg3_pct = to_num(g.get("fg3_pct"))
                away_ft_pct = to_num(g.get("ft_pct"))
                away_trb = to_num(g.get("trb"))
                away_ast = to_num(g.get("ast"))
                away_stl = to_num(g.get("stl"))
                away_blk = to_num(g.get("blk"))
                away_tov = to_num(g.get("tov"))
                away_orb = to_num(g.get("orb"))
                away_drb = to_num(g.get("drb"))

            if np.isnan(home_points) or np.isnan(away_points):
                continue

            key = (str(date).strip(), home_team, away_team)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            all_games.append({
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "home_pts": home_points,
                "away_pts": away_points,
                "home_fg_pct": home_fg_pct,
                "home_fg3_pct": home_fg3_pct,
                "home_ft_pct": home_ft_pct,
                "home_trb": home_trb,
                "home_ast": home_ast,
                "home_stl": home_stl,
                "home_blk": home_blk,
                "home_tov": home_tov,
                "home_orb": home_orb,
                "home_drb": home_drb,
                "away_fg_pct": away_fg_pct,
                "away_fg3_pct": away_fg3_pct,
                "away_ft_pct": away_ft_pct,
                "away_trb": away_trb,
                "away_ast": away_ast,
                "away_stl": away_stl,
                "away_blk": away_blk,
                "away_tov": away_tov,
                "away_orb": away_orb,
                "away_drb": away_drb
            })

    if not all_games:
        return pd.DataFrame(columns=["date","home_team","away_team","home_pts","away_pts","home_win"])

    df = pd.DataFrame(all_games)
    # parse date if possible
    try:
        df["date"] = pd.to_datetime(df["date"])
    except:
        pass
    df["home_win"] = (df["home_pts"] > df["away_pts"]).astype(int)
    df = df.drop_duplicates(subset=["date","home_team","away_team"]).reset_index(drop=True)
    return df


def build_multi_season_dataframe(teams, start_season, num_seasons=NUM_SEASONS):
    """Build dataframe from single season using scrapingStats.py directly."""
    # Just use the current season - scrapingStats.py works for all teams
    games_df = build_games_dataframe(teams, start_season)
    if not games_df.empty:
        games_df['season'] = start_season
    return games_df


def compute_recent_form(games_df, team, game_date, window=RECENT_FORM_WINDOW):
    """Compute recent form stats for a team up to a specific date."""
    # Get all games for this team before the current game
    team_games = games_df[
        ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
        (games_df['date'] < game_date)
    ].sort_values('date').tail(window)
    
    if len(team_games) < window:
        # Not enough games, return None (will use season averages)
        return None
    
    wins = 0
    pt_diff_sum = 0
    pts_for_sum = 0
    pts_against_sum = 0
    
    for _, game in team_games.iterrows():
        if game['home_team'] == team:
            wins += game['home_win']
            pt_diff = game['home_pts'] - game['away_pts']
            pts_for_sum += game['home_pts']
            pts_against_sum += game['away_pts']
        else:
            wins += (1 - game['home_win'])
            pt_diff = game['away_pts'] - game['home_pts']
            pts_for_sum += game['away_pts']
            pts_against_sum += game['home_pts']
        pt_diff_sum += pt_diff
    
    return {
        'recent_win_pct': wins / window,
        'recent_pt_diff': pt_diff_sum / window,
        'recent_pts_for': pts_for_sum / window,
        'recent_pts_against': pts_against_sum / window
    }


def compute_team_aggregates(games_df):
    if games_df.empty:
        return pd.DataFrame()

    # Aggregate home game stats
    home = games_df.groupby("home_team").agg(
        home_games=("home_pts","count"),
        home_pts_total=("home_pts","sum"),
        home_pts_avg=("home_pts","mean"),
        home_opp_pts_avg=("away_pts","mean"),
        home_fg_pct_avg=("home_fg_pct","mean"),
        home_fg3_pct_avg=("home_fg3_pct","mean"),
        home_ft_pct_avg=("home_ft_pct","mean"),
        home_trb_avg=("home_trb","mean"),
        home_ast_avg=("home_ast","mean"),
        home_stl_avg=("home_stl","mean"),
        home_blk_avg=("home_blk","mean"),
        home_tov_avg=("home_tov","mean"),
        home_orb_avg=("home_orb","mean"),
        home_drb_avg=("home_drb","mean"),
        # Opponent stats when team is home
        home_opp_fg_pct_avg=("away_fg_pct","mean"),
        home_opp_fg3_pct_avg=("away_fg3_pct","mean"),
        home_opp_ft_pct_avg=("away_ft_pct","mean"),
        home_opp_trb_avg=("away_trb","mean"),
        home_opp_ast_avg=("away_ast","mean"),
        home_opp_stl_avg=("away_stl","mean"),
        home_opp_blk_avg=("away_blk","mean"),
        home_opp_tov_avg=("away_tov","mean")
    ).rename_axis("team").reset_index()

    # Aggregate away game stats
    away = games_df.groupby("away_team").agg(
        away_games=("away_pts","count"),
        away_pts_total=("away_pts","sum"),
        away_pts_avg=("away_pts","mean"),
        away_opp_pts_avg=("home_pts","mean"),
        away_fg_pct_avg=("away_fg_pct","mean"),
        away_fg3_pct_avg=("away_fg3_pct","mean"),
        away_ft_pct_avg=("away_ft_pct","mean"),
        away_trb_avg=("away_trb","mean"),
        away_ast_avg=("away_ast","mean"),
        away_stl_avg=("away_stl","mean"),
        away_blk_avg=("away_blk","mean"),
        away_tov_avg=("away_tov","mean"),
        away_orb_avg=("away_orb","mean"),
        away_drb_avg=("away_drb","mean"),
        # Opponent stats when team is away
        away_opp_fg_pct_avg=("home_fg_pct","mean"),
        away_opp_fg3_pct_avg=("home_fg3_pct","mean"),
        away_opp_ft_pct_avg=("home_ft_pct","mean"),
        away_opp_trb_avg=("home_trb","mean"),
        away_opp_ast_avg=("home_ast","mean"),
        away_opp_stl_avg=("home_stl","mean"),
        away_opp_blk_avg=("home_blk","mean"),
        away_opp_tov_avg=("home_tov","mean")
    ).rename_axis("team").reset_index()

    teams = pd.merge(home, away, how="outer", on="team").fillna(0)
    teams["games_played"] = teams["home_games"] + teams["away_games"]
    
    # Weighted averages for overall team stats
    teams["pts_for_avg"] = (teams["home_pts_total"] + teams["away_pts_total"]) / teams["games_played"].replace(0, np.nan)
    teams["pts_against_avg"] = ((teams["home_opp_pts_avg"] * teams["home_games"]) + (teams["away_opp_pts_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["pt_diff"] = teams["pts_for_avg"] - teams["pts_against_avg"]
    
    # Weighted averages for shooting percentages
    teams["fg_pct_avg"] = ((teams["home_fg_pct_avg"] * teams["home_games"]) + (teams["away_fg_pct_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["fg3_pct_avg"] = ((teams["home_fg3_pct_avg"] * teams["home_games"]) + (teams["away_fg3_pct_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["ft_pct_avg"] = ((teams["home_ft_pct_avg"] * teams["home_games"]) + (teams["away_ft_pct_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    
    # Weighted averages for other stats
    teams["trb_avg"] = ((teams["home_trb_avg"] * teams["home_games"]) + (teams["away_trb_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["ast_avg"] = ((teams["home_ast_avg"] * teams["home_games"]) + (teams["away_ast_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["stl_avg"] = ((teams["home_stl_avg"] * teams["home_games"]) + (teams["away_stl_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["blk_avg"] = ((teams["home_blk_avg"] * teams["home_games"]) + (teams["away_blk_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["tov_avg"] = ((teams["home_tov_avg"] * teams["home_games"]) + (teams["away_tov_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["orb_avg"] = ((teams["home_orb_avg"] * teams["home_games"]) + (teams["away_orb_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["drb_avg"] = ((teams["home_drb_avg"] * teams["home_games"]) + (teams["away_drb_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    
    # Opponent stats (defensive metrics)
    teams["opp_fg_pct_avg"] = ((teams["home_opp_fg_pct_avg"] * teams["home_games"]) + (teams["away_opp_fg_pct_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["opp_fg3_pct_avg"] = ((teams["home_opp_fg3_pct_avg"] * teams["home_games"]) + (teams["away_opp_fg3_pct_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["opp_ft_pct_avg"] = ((teams["home_opp_ft_pct_avg"] * teams["home_games"]) + (teams["away_opp_ft_pct_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["opp_trb_avg"] = ((teams["home_opp_trb_avg"] * teams["home_games"]) + (teams["away_opp_trb_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["opp_ast_avg"] = ((teams["home_opp_ast_avg"] * teams["home_games"]) + (teams["away_opp_ast_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["opp_stl_avg"] = ((teams["home_opp_stl_avg"] * teams["home_games"]) + (teams["away_opp_stl_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["opp_blk_avg"] = ((teams["home_opp_blk_avg"] * teams["home_games"]) + (teams["away_opp_blk_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["opp_tov_avg"] = ((teams["home_opp_tov_avg"] * teams["home_games"]) + (teams["away_opp_tov_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    
    teams = teams.fillna(0).set_index("team")
    return teams


def build_feature_matrix_and_train(games_df, teams_df):
    rows = []
    labels = []
    for _, r in games_df.iterrows():
        h = r["home_team"]
        a = r["away_team"]
        game_date = r.get("date")
        
        if h not in teams_df.index or a not in teams_df.index:
            continue
        hstats = teams_df.loc[h]
        astats = teams_df.loc[a]
        
        # Get recent form if date is available
        home_recent = None
        away_recent = None
        if game_date is not None and pd.notna(game_date):
            home_recent = compute_recent_form(games_df, h, game_date, RECENT_FORM_WINDOW)
            away_recent = compute_recent_form(games_df, a, game_date, RECENT_FORM_WINDOW)
        
        feat = {
            # Basic scoring stats
            "home_pts_for_avg": hstats["pts_for_avg"],
            "home_pts_against_avg": hstats["pts_against_avg"],
            "home_pt_diff": hstats["pt_diff"],
            "away_pts_for_avg": astats["pts_for_avg"],
            "away_pts_against_avg": astats["pts_against_avg"],
            "away_pt_diff": astats["pt_diff"],
            # Shooting percentages
            "home_fg_pct": hstats["fg_pct_avg"],
            "home_fg3_pct": hstats["fg3_pct_avg"],
            "home_ft_pct": hstats["ft_pct_avg"],
            "away_fg_pct": astats["fg_pct_avg"],
            "away_fg3_pct": astats["fg3_pct_avg"],
            "away_ft_pct": astats["ft_pct_avg"],
            # Rebounding
            "home_trb": hstats["trb_avg"],
            "home_orb": hstats["orb_avg"],
            "home_drb": hstats["drb_avg"],
            "away_trb": astats["trb_avg"],
            "away_orb": astats["orb_avg"],
            "away_drb": astats["drb_avg"],
            # Playmaking and defense
            "home_ast": hstats["ast_avg"],
            "home_stl": hstats["stl_avg"],
            "home_blk": hstats["blk_avg"],
            "home_tov": hstats["tov_avg"],
            "away_ast": astats["ast_avg"],
            "away_stl": astats["stl_avg"],
            "away_blk": astats["blk_avg"],
            "away_tov": astats["tov_avg"],
            # Defensive metrics (opponent stats)
            "home_opp_fg_pct": hstats["opp_fg_pct_avg"],
            "home_opp_fg3_pct": hstats["opp_fg3_pct_avg"],
            "home_opp_ast": hstats["opp_ast_avg"],
            "home_opp_tov": hstats["opp_tov_avg"],
            "away_opp_fg_pct": astats["opp_fg_pct_avg"],
            "away_opp_fg3_pct": astats["opp_fg3_pct_avg"],
            "away_opp_ast": astats["opp_ast_avg"],
            "away_opp_tov": astats["opp_tov_avg"],
            # Relative/differential features (often more predictive)
            "pt_diff_diff": hstats["pt_diff"] - astats["pt_diff"],
            "fg_pct_diff": hstats["fg_pct_avg"] - astats["fg_pct_avg"],
            "fg3_pct_diff": hstats["fg3_pct_avg"] - astats["fg3_pct_avg"],
            "trb_diff": hstats["trb_avg"] - astats["trb_avg"],
            "ast_diff": hstats["ast_avg"] - astats["ast_avg"],
            "stl_diff": hstats["stl_avg"] - astats["stl_avg"],
            "blk_diff": hstats["blk_avg"] - astats["blk_avg"],
            "tov_diff": astats["tov_avg"] - hstats["tov_avg"],  # Negative is good (fewer TOs)
            "opp_fg_pct_diff": astats["opp_fg_pct_avg"] - hstats["opp_fg_pct_avg"],  # Lower opp FG% is better
            # Recent form features (if available)
            "home_recent_win_pct": home_recent["recent_win_pct"] if home_recent else 0.5,
            "home_recent_pt_diff": home_recent["recent_pt_diff"] if home_recent else hstats["pt_diff"],
            "away_recent_win_pct": away_recent["recent_win_pct"] if away_recent else 0.5,
            "away_recent_pt_diff": away_recent["recent_pt_diff"] if away_recent else astats["pt_diff"],
            "recent_form_diff": (home_recent["recent_win_pct"] if home_recent else 0.5) - (away_recent["recent_win_pct"] if away_recent else 0.5),
            # Home court advantage
            "home_flag": 1
        }
        rows.append(feat)
        labels.append(r["home_win"])

    if not rows:
        raise ValueError("No training data available after filtering. Check scraped data.")

    X = pd.DataFrame(rows).fillna(0)
    y = pd.Series(labels)
    
    # Try stratified split, but fall back to regular split if it fails (e.g., too few samples)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    except ValueError:
        # If stratification fails (e.g., not enough samples in each class), use regular split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Use XGBoost if available, otherwise fall back to Random Forest
    if USE_XGBOOST:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    else:
        model = RandomForestClassifier(
            n_estimators=300,           # More trees for better accuracy
            max_depth=15,               # Prevent overfitting
            min_samples_split=10,       # More samples required to split
            min_samples_leaf=5,         # More samples in leaf nodes
            max_features='sqrt',        # Use sqrt of features (good default)
            class_weight='balanced',    # Handle class imbalance
            random_state=42,
            n_jobs=-1                   # Use all CPU cores
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, preds)
    
    # Cross-validation for more robust accuracy estimate
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    
    return model, X.columns.tolist()

#

def display_prediction_report(*args, **kwargs):
    """Previously used for console output; now intentionally silent."""
    return


def predict_winner(model, feature_cols, teams_df, teamA, teamB, home_team, games_df=None, injuries_home=None, injuries_away=None):
    """
    Optionally accepts injuries_home and injuries_away (list of dicts with 'player', 'description').
    Auto-assigns importance: 3 if 'starter', 'star', 'all-star' in description, else 1.
    Each 'importance' point reduces team win prob by 0.02.
    """
    teamA = teamA.upper(); teamB = teamB.upper(); home_team = home_team.upper()
    if home_team not in (teamA, teamB):
        raise ValueError("home_team must be either teamA or teamB")
    # Map team names to match scraped data abbreviations
    teamA_mapped = TEAM_NAME_MAP.get(teamA, teamA)
    teamB_mapped = TEAM_NAME_MAP.get(teamB, teamB)
    home_team_mapped = TEAM_NAME_MAP.get(home_team, home_team)

    def resolve_team(abbr: str) -> str:
        """
        Resolve team abbreviation against teams_df with common aliases.
        Ensures Charlotte works whether caller uses CHA or CHO.
        """
        candidates = [
            abbr,
            TEAM_NAME_MAP.get(abbr, abbr),
        ]
        if abbr in ("CHO", "CHA"):
            candidates.extend(["CHO", "CHA"])
        if abbr in ("PHX", "PHO"):
            candidates.extend(["PHX", "PHO"])
        if abbr in ("BKN", "BRK"):
            candidates.extend(["BKN", "BRK"])
        for cand in candidates:
            if cand in teams_df.index:
                return cand
        return candidates[-1]  # fall back to last candidate
    home = resolve_team(home_team_mapped)
    away = resolve_team(teamB_mapped if home == teamA_mapped else teamA_mapped)
    # Check which teams are missing and provide helpful error message
    missing_teams = []
    if home not in teams_df.index:
        missing_teams.append(f"{home_team} (mapped to {home})")
    if away not in teams_df.index:
        original_away = teamB if home == teamA_mapped else teamA
        missing_teams.append(f"{original_away} (mapped to {away})")
    if missing_teams:
        available = list(teams_df.index)
        raise ValueError(
            f"Teams missing aggregate stats: {', '.join(missing_teams)}. "
            f"Available teams: {', '.join(sorted(available))}")
    h = teams_df.loc[home]; a = teams_df.loc[away]
    # Get recent form if games_df is available
    home_recent = None
    away_recent = None
    if games_df is not None and not games_df.empty:
        # Use most recent date in games_df as prediction date
        if 'date' in games_df.columns:
            prediction_date = games_df['date'].max()
            if pd.notna(prediction_date):
                home_recent = compute_recent_form(games_df, home, prediction_date, RECENT_FORM_WINDOW)
                away_recent = compute_recent_form(games_df, away, prediction_date, RECENT_FORM_WINDOW)
    feat = {
        # Basic scoring stats
        "home_pts_for_avg": h["pts_for_avg"],
        "home_pts_against_avg": h["pts_against_avg"],
        "home_pt_diff": h["pt_diff"],
        "away_pts_for_avg": a["pts_for_avg"],
        "away_pts_against_avg": a["pts_against_avg"],
        "away_pt_diff": a["pt_diff"],
        # Shooting percentages
        "home_fg_pct": h["fg_pct_avg"],
        "home_fg3_pct": h["fg3_pct_avg"],
        "home_ft_pct": h["ft_pct_avg"],
        "away_fg_pct": a["fg_pct_avg"],
        "away_fg3_pct": a["fg3_pct_avg"],
        "away_ft_pct": a["ft_pct_avg"],
        # Rebounding
        "home_trb": h["trb_avg"],
        "home_orb": h["orb_avg"],
        "home_drb": h["drb_avg"],
        "away_trb": a["trb_avg"],
        "away_orb": a["orb_avg"],
        "away_drb": a["drb_avg"],
        # Playmaking and defense
        "home_ast": h["ast_avg"],
        "home_stl": h["stl_avg"],
        "home_blk": h["blk_avg"],
        "home_tov": h["tov_avg"],
        "away_ast": a["ast_avg"],
        "away_stl": a["stl_avg"],
        "away_blk": a["blk_avg"],
        "away_tov": a["tov_avg"],
        # Defensive metrics (opponent stats)
        "home_opp_fg_pct": h["opp_fg_pct_avg"],
        "home_opp_fg3_pct": h["opp_fg3_pct_avg"],
        "home_opp_ast": h["opp_ast_avg"],
        "home_opp_tov": h["opp_tov_avg"],
        "away_opp_fg_pct": a["opp_fg_pct_avg"],
        "away_opp_fg3_pct": a["opp_fg3_pct_avg"],
        "away_opp_ast": a["opp_ast_avg"],
        "away_opp_tov": a["opp_tov_avg"],
        # Relative/differential features (often more predictive)
        "pt_diff_diff": h["pt_diff"] - a["pt_diff"],
        "fg_pct_diff": h["fg_pct_avg"] - a["fg_pct_avg"],
        "fg3_pct_diff": h["fg3_pct_avg"] - a["fg3_pct_avg"],
        "trb_diff": h["trb_avg"] - a["trb_avg"],
        "ast_diff": h["ast_avg"] - a["ast_avg"],
        "stl_diff": h["stl_avg"] - a["stl_avg"],
        "blk_diff": h["blk_avg"] - a["blk_avg"],
        "tov_diff": a["tov_avg"] - h["tov_avg"],  # Negative is good (fewer TOs)
        "opp_fg_pct_diff": a["opp_fg_pct_avg"] - h["opp_fg_pct_avg"],  # Lower opp FG% is better
        # Recent form features (if available)
        "home_recent_win_pct": home_recent["recent_win_pct"] if home_recent else 0.5,
        "home_recent_pt_diff": home_recent["recent_pt_diff"] if home_recent else h["pt_diff"],
        "away_recent_win_pct": away_recent["recent_win_pct"] if away_recent else 0.5,
        "away_recent_pt_diff": away_recent["recent_pt_diff"] if away_recent else a["pt_diff"],
        "recent_form_diff": (home_recent["recent_win_pct"] if home_recent else 0.5) - (away_recent["recent_win_pct"] if away_recent else 0.5),
        # Home court advantage
        "home_flag": 1
    }
    # Ensure all required features are present (handle old models with fewer features)
    feat_dict = {col: feat.get(col, 0) for col in feature_cols}
    X_row = pd.DataFrame([feat_dict])[feature_cols].fillna(0)
    prob_home = model.predict_proba(X_row)[0][1]
    prob_away = 1 - prob_home
    # --- Injury adjustment logic ---
    total_importance_home = 0
    total_importance_away = 0
    injury_note_home = []
    injury_note_away = []
    if injuries_home:
        for inj in injuries_home:
            desc = (inj.get("description") or "").lower()
            importance = 3 if any(word in desc for word in ["star", "starter", "all-star"]) else 1
            total_importance_home += importance
            injury_note_home.append(f"{inj.get('player', '')} (level {importance})")
    if injuries_away:
        for inj in injuries_away:
            desc = (inj.get("description") or "").lower()
            importance = 3 if any(word in desc for word in ["star", "starter", "all-star"]) else 1
            total_importance_away += importance
            injury_note_away.append(f"{inj.get('player', '')} (level {importance})")
    # Each importance point = 2% (0.02) win prob, capped to [0,1].
    adjustment = 0.02
    prob_home_adj = prob_home - (total_importance_home * adjustment) + (total_importance_away * adjustment)
    prob_home_adj = max(0.0, min(1.0, prob_home_adj))
    prob_away_adj = 1 - prob_home_adj
    predicted = home if prob_home_adj >= 0.5 else away
    display_prediction_report(home, away, prob_home_adj, prob_away_adj, predicted, dict(h), dict(a), injury_note_home, injury_note_away)
    stat_factors = build_stat_factors(home, away, h, a)
    return {"home_team": home, "away_team": away, "home_win_prob": round(prob_home_adj,3), "away_win_prob": round(prob_away_adj,3), "predicted_winner": predicted,
            "injuries_home": injury_note_home, "injuries_away": injury_note_away, "adjustment_detail": { "raw_model_prob_home": round(prob_home,3), "raw_model_prob_away": round(prob_away,3), "inj_importance_home": total_importance_home, "inj_importance_away": total_importance_away, "adj_delta_percent": (total_importance_away - total_importance_home)*adjustment*100 },
            "stat_factors": stat_factors}


def build_stat_factors(home_label, away_label, home_stats, away_stats, top_n=6):
    """
    Build a short list of the biggest stat differences that influenced the model.
    Returns list sorted by magnitude of advantage.
    """
    factors = []
    for cfg in STAT_FACTORS_CONFIG:
        hv = float(home_stats.get(cfg["key"], 0) or 0)
        av = float(away_stats.get(cfg["key"], 0) or 0)
        advantage = (hv - av) if cfg["better"] == "higher" else (av - hv)
        better_side = "home" if advantage > 0 else ("away" if advantage < 0 else "even")
        factors.append({
            "label": cfg["label"],
            "key": cfg["key"],
            "home_value": hv,
            "away_value": av,
            "better": better_side,
            "decimals": cfg["decimals"],
            "multiplier": cfg["multiplier"],
            "direction": cfg["better"],
            "advantage": abs(advantage)
        })
    factors = sorted(factors, key=lambda x: x["advantage"], reverse=True)
    return factors[:top_n]


def main(teams=None, season=SEASON, force_retrain=False):
    # Use all teams by default for more training data
    if teams is None:
        teams = TEAMS_ALL
    
    # Try to load saved model and data
    model = None
    feature_cols = None
    games_df = None
    teams_df = None
    existing_games_df = None
    
    if not force_retrain:
        model, feature_cols, games_df, teams_df, saved_season = load_model_and_data()
        if games_df is not None and not games_df.empty:
            existing_games_df = games_df.copy()
        
        if model is not None:
            # Check model version and compatibility
            try:
                with open(MODEL_FILE, "rb") as f:
                    model_data = pickle.load(f)
                model_version = model_data.get("model_version", 1)  # Old models are version 1
                saved_num_seasons = model_data.get("num_seasons", 1)
                
                # Check if model needs retraining
                needs_retrain = False
                if saved_season != season:
                    needs_retrain = True
                elif model_version < 2:
                    needs_retrain = True
                elif saved_num_seasons != NUM_SEASONS:
                    needs_retrain = True
                
                if needs_retrain:
                    model = None  # Force retrain
            except Exception:
                model = None  # Force retrain to be safe
    
    # Always scrape new data to check for new games (incremental learning)
    new_games_df = build_games_dataframe(teams, season)
    
    if new_games_df.empty:
        if existing_games_df is not None and not existing_games_df.empty:
            games_df = existing_games_df
            if model is None:
                return
        else:
            return
    else:
        # Merge new games with existing games (incremental learning)
        games_df = merge_games_data(existing_games_df, new_games_df)
        
        # Check if we have new games or need to retrain
        has_new_games = existing_games_df is None or len(games_df) > len(existing_games_df)
        needs_retrain = model is None or force_retrain or has_new_games
        
        if needs_retrain:
            teams_df = compute_team_aggregates(games_df)
            model, feature_cols = build_feature_matrix_and_train(games_df, teams_df)
            
            # Save the newly trained model with accumulated data
            save_model_and_data(model, feature_cols, games_df, teams_df, season)
        else:
            # Use existing model but update teams_df with latest aggregates
            teams_df = compute_team_aggregates(games_df)
    return {"games_df": games_df, "teams_df": teams_df, "model": model, "feature_cols": feature_cols}

if __name__ == "__main__":
    out = main()
