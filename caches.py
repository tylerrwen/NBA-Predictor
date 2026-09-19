"""
Disk-backed caches that let the web app serve predictions without ever
scraping or training inside a request. Everything here is written by
build_cache.py (the offline prep job) and only read by app.py.
"""
import os
import json
import pickle
import hashlib
from datetime import datetime

from nbaPredictor import MODEL_DIR

INJURIES_FILE = os.path.join(MODEL_DIR, "injuries_cache.json")
PLAYERS_FILE = os.path.join(MODEL_DIR, "players_cache.json")
SEASONS_DIR = os.path.join(MODEL_DIR, "seasons")
TEAMSEASON_DIR = os.path.join(MODEL_DIR, "team_seasons")


# --- Injuries -------------------------------------------------------------

def save_injuries(injuries_by_team, season):
    """Persist {team_abbr: [injury dicts]} plus metadata as JSON."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "season": season,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "teams": injuries_by_team,
    }
    with open(INJURIES_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_injuries():
    """Return (teams_dict, season, generated_at). Empty/None if not built yet."""
    if not os.path.exists(INJURIES_FILE):
        return {}, None, None
    try:
        with open(INJURIES_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("teams", {}), payload.get("season"), payload.get("generated_at")
    except (OSError, ValueError):
        return {}, None, None


# --- Key players -----------------------------------------------------------

def save_players(players_by_team, season):
    """Persist {team_abbr: [player dicts]} plus metadata as JSON."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "season": season,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "teams": players_by_team,
    }
    with open(PLAYERS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_players():
    """Return (teams_dict, season, generated_at). Empty/None if not built yet."""
    if not os.path.exists(PLAYERS_FILE):
        return {}, None, None
    try:
        with open(PLAYERS_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("teams", {}), payload.get("season"), payload.get("generated_at")
    except (OSError, ValueError):
        return {}, None, None


# --- Per-team season stat rows (for the historic predictor) ----------------

def _team_season_path(season, team):
    return os.path.join(TEAMSEASON_DIR, f"{season}_{team.upper()}.json")


def save_team_season(season, team, stats):
    os.makedirs(TEAMSEASON_DIR, exist_ok=True)
    with open(_team_season_path(season, team), "w", encoding="utf-8") as f:
        json.dump(stats, f)


def load_team_season(season, team):
    path = _team_season_path(season, team)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# --- Per-season / per-team model resources (for the historic predictor) ----

def _season_key(season, teams):
    teams_part = "-".join(sorted(t.upper() for t in teams))
    if len(teams_part) > 60:  # keep filenames sane for large team sets
        teams_part = hashlib.md5(teams_part.encode()).hexdigest()
    return f"season_{season}_{teams_part}.pkl"


def season_cache_path(season, teams):
    return os.path.join(SEASONS_DIR, _season_key(season, teams))


def save_season_resources(season, teams, resources):
    """resources is the (model, feature_cols, games_df, teams_df) tuple."""
    os.makedirs(SEASONS_DIR, exist_ok=True)
    with open(season_cache_path(season, teams), "wb") as f:
        pickle.dump(resources, f)


def load_season_resources(season, teams):
    """Return the cached (model, feature_cols, games_df, teams_df) tuple, or None."""
    path = season_cache_path(season, teams)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, ModuleNotFoundError):
        return None
