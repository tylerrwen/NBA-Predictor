"""
Offline data-prep for the NBA predictor.

This is the ONLY place that scrapes basketball-reference. Run it on a schedule
(e.g. daily during the season); the Flask app then serves everything from the
caches this writes and never touches the network in a request.

Examples:
    python build_cache.py                       # current model + injuries (default)
    python build_cache.py --current --seasons 2026,2025,2024   # retrain on specific seasons
    python build_cache.py --historic --seasons 2024,2023 --teams LAL,BOS
"""
import argparse
import time

from nbaPredictor import train_current_model, compute_team_season_stats, SEASON
from scrapingRoster import scrape_injuries
from scrapingPlayers import scrape_key_players
from caches import save_injuries, save_players, save_team_season

# UI abbreviations (must match app.TEAMS) -> basketball-reference abbreviations
BR_ABBR_MAP = {"BKN": "BRK", "PHX": "PHO"}

APP_TEAMS = [
    "ATL", "BOS", "BKN", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]


def build_current_model(seasons=None):
    seasons = seasons or [SEASON - 2, SEASON - 1, SEASON]
    print(f"[current] training model on seasons {seasons} ...")
    metrics = train_current_model(seasons=seasons)
    print("  metrics:", metrics)
    return metrics


def build_injuries(season=SEASON):
    print(f"[injuries] scraping injuries for {len(APP_TEAMS)} teams, season {season} ...")
    by_team = {}
    for abbr in APP_TEAMS:
        br = BR_ABBR_MAP.get(abbr, abbr)
        try:
            by_team[abbr] = scrape_injuries(br, season)
        except Exception as e:  # keep going; one bad team shouldn't sink the run
            print(f"  {abbr}: failed ({e})")
            by_team[abbr] = []
    save_injuries(by_team, season)
    total = sum(len(v) for v in by_team.values())
    print(f"  cached {total} injuries across {len(by_team)} teams")


def build_players(season=SEASON):
    print(f"[players] scraping key players for {len(APP_TEAMS)} teams, season {season} ...")
    by_team = {}
    for abbr in APP_TEAMS:
        br = BR_ABBR_MAP.get(abbr, abbr)
        try:
            by_team[abbr] = scrape_key_players(br, season)
        except Exception as e:  # one bad team shouldn't sink the run
            print(f"  {abbr}: failed ({e})")
            by_team[abbr] = []
        time.sleep(2.5)  # be polite to basketball-reference
    save_players(by_team, season)
    total = sum(len(v) for v in by_team.values())
    print(f"  cached {total} players across {len(by_team)} teams")


def build_historic(seasons, teams):
    """Pre-warm the historic team-season stat cache so those queries are instant.
    Historic also builds these on demand, so this is optional."""
    for season in seasons:
        for team in teams:
            br = BR_ABBR_MAP.get(team, team)
            print(f"[historic] {team} {season} ...")
            try:
                stats = compute_team_season_stats(br, season)
            except Exception as e:
                print(f"  failed ({e})")
                continue
            if not stats:
                print("  no data, skipped")
                continue
            save_team_season(season, br, stats)
            print("  cached")
            time.sleep(2.5)


def main():
    ap = argparse.ArgumentParser(description="Build offline caches for the NBA predictor web app.")
    ap.add_argument("--current", action="store_true", help="rebuild the current-season model")
    ap.add_argument("--injuries", action="store_true", help="rebuild the injuries cache")
    ap.add_argument("--players", action="store_true", help="rebuild the key-players cache")
    ap.add_argument("--historic", action="store_true", help="rebuild historic season/team resources")
    ap.add_argument("--seasons", default="", help="comma-separated seasons for --current/--historic (e.g. 2024,2023)")
    ap.add_argument("--teams", default="", help="comma-separated teams for --historic (default: all)")
    args = ap.parse_args()

    # Default action when no flags: current model + injuries + players.
    if not (args.current or args.injuries or args.players or args.historic):
        args.current = args.injuries = args.players = True

    if args.current:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()] or None
        build_current_model(seasons=seasons)
    if args.injuries:
        build_injuries()
    if args.players:
        build_players()
    if args.historic:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()] or [SEASON - 1]
        teams = [t.strip().upper() for t in args.teams.split(",") if t.strip()] or APP_TEAMS
        build_historic(seasons, teams)


if __name__ == "__main__":
    main()
