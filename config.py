"""
Single source of truth for season and team constants shared across the web app
(app.py), the model/training code (nbaPredictor.py), and the offline cache
builder (build_cache.py). Keeping them here prevents the three modules from
drifting -- e.g. a season bump applied in one file but forgotten in another.
"""

# Current season = the calendar year the season ends in (2026 == the 2025-26 season).
SEASON = 2026

# UI/app team abbreviations (what the web form submits).
UI_TEAMS = [
    "ATL", "BOS", "BKN", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

# basketball-reference gamelog abbreviations (a few teams differ from the UI ones).
BR_TEAMS = [
    "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

# UI abbreviation -> basketball-reference abbreviation (only the ones that differ).
BR_ABBR_MAP = {"BKN": "BRK", "PHX": "PHO"}
