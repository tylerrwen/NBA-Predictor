import pandas as pd
from flask import Flask, render_template, request
from nbaPredictor import load_model_and_data, predict_winner, compute_team_season_stats
from caches import load_injuries, load_players, load_team_season, save_team_season

# basketball-reference uses different gamelog abbreviations for a few teams.
BR_ABBR_MAP = {"BKN": "BRK", "PHX": "PHO"}

app = Flask(__name__)

TEAMS = [
    "ATL", "BOS", "BKN", "CHO", "CHI", "CLE", "DAL", "DEN",
    "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
    "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
    "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
]

TEAM_NAME_MAP = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHO": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

TEAM_OPTIONS = [{"abbr": abbr, "name": TEAM_NAME_MAP.get(abbr, abbr)} for abbr in TEAMS]

SEASON = 2026
SEASONS = list(range(2000, SEASON + 1))[::-1]  # 2026 down to 2000
HISTORIC_SEASONS = [s for s in SEASONS if s < SEASON]  # remove current 2025-2026 from historic options

model, feature_cols, games_df, teams_df, _ = load_model_and_data()

# Injuries and key players are scraped offline by build_cache.py and served from disk.
INJURIES_BY_TEAM, INJURIES_SEASON, INJURIES_UPDATED = load_injuries()
PLAYERS_BY_TEAM, PLAYERS_SEASON, PLAYERS_UPDATED = load_players()


def annotate_injuries(injuries, players):
    """Tag each injured player with their impact tier (from the key-player cache)
    so the prediction weights a star's absence more than a benchwarmer's."""
    tier_by_name = {p.get("player"): p.get("tier") for p in players}
    for inj in injuries:
        inj["impact_tier"] = tier_by_name.get(inj.get("player"))
    return injuries

_TEAM_SEASON_MEM = {}  # (season, br_abbr) -> stats dict, in-process cache


def get_team_season_stats(season: int, app_abbr: str):
    """
    Return (br_abbr, stats_dict) for a team in a given season. Served from disk
    cache; scraped on demand the first time a season/team is requested, then
    cached so repeat queries are instant. The live predictor never calls this.
    """
    br = BR_ABBR_MAP.get(app_abbr, app_abbr)
    key = (season, br)
    if key in _TEAM_SEASON_MEM:
        return br, _TEAM_SEASON_MEM[key]
    stats = load_team_season(season, br)
    if stats is None:
        stats = compute_team_season_stats(br, season)
        if not stats:
            raise LookupError(f"No data available for {app_abbr} in {season - 1}-{season}.")
        save_team_season(season, br, stats)
    _TEAM_SEASON_MEM[key] = stats
    return br, stats

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    injuries_home = []
    injuries_away = []
    selected_home = None
    selected_away = None
    error = None
    notable_home = []
    notable_away = []
    adj_detail = None
    stat_factors = None
    players_home = []
    players_away = []

    if request.method == "POST":
        home = request.form.get("home")
        away = request.form.get("away")
        selected_home = home
        selected_away = away
        if not home or not away:
            error = "Both teams must be selected."
        elif home == away:
            error = "Teams must be different."
        else:
            players_home = PLAYERS_BY_TEAM.get(home, [])
            players_away = PLAYERS_BY_TEAM.get(away, [])
            injuries_home = annotate_injuries(INJURIES_BY_TEAM.get(home, []), players_home)
            injuries_away = annotate_injuries(INJURIES_BY_TEAM.get(away, []), players_away)
            try:
                prediction = predict_winner(
                    model, feature_cols, teams_df, home, away, home, games_df,
                    injuries_home=injuries_home, injuries_away=injuries_away
                )
                # winner label for template: home or away team string
                if prediction["predicted_winner"] == home:
                    winner = home
                    confidence = round(prediction["home_win_prob"] * 100, 1)
                else:
                    winner = away
                    confidence = round(prediction["away_win_prob"] * 100, 1)
                notable_home = [inj for inj in prediction.get("injuries_home", []) if "level 3" in inj]
                notable_away = [inj for inj in prediction.get("injuries_away", []) if "level 3" in inj]
                adj_detail = prediction.get("adjustment_detail", {})
                stat_factors = prediction.get("stat_factors", [])
                result = dict(
                    winner=winner,
                    confidence=confidence,
                    home=home,
                    away=away,
                )
            except Exception as e:
                error = "Prediction model error: " + str(e)
    return render_template(
        "index.html",
        teams=TEAM_OPTIONS,
        result=result,
        injuries_home=injuries_home,
        injuries_away=injuries_away,
        notable_home=notable_home,
        notable_away=notable_away,
        adj_detail=adj_detail,
        stat_factors=stat_factors,
        players_home=players_home,
        players_away=players_away,
        selected_home=selected_home,
        selected_away=selected_away,
        injuries_updated=INJURIES_UPDATED,
        error=error
    )

@app.route("/historic", methods=["GET", "POST"])
def historic():
    """
    Historic selector page. Uses the same model pipeline (predict_winner) to provide
    a probability, but note this is limited by the currently loaded model/data.
    """
    result = None
    selected_a = None
    selected_b = None
    season_a = HISTORIC_SEASONS[0]
    season_b = HISTORIC_SEASONS[0]
    error = None
    stat_factors = None

    seasons_options = [{"value": s, "label": f"{s-1}-{s}"} for s in HISTORIC_SEASONS]

    if request.method == "POST":
        team_a = request.form.get("team_a")
        team_b = request.form.get("team_b")
        season_a = int(request.form.get("season_a") or SEASONS[0])
        season_b = int(request.form.get("season_b") or SEASONS[0])
        selected_a = team_a
        selected_b = team_b

        if not team_a or not team_b:
            error = "Please select both teams."
        elif team_a == team_b and season_a == season_b:
            error = "Pick different season/team combinations."
        else:
            try:
                # Each team's season stat row, then score with the current model.
                _, stats_a = get_team_season_stats(season_a, team_a)
                _, stats_b = get_team_season_stats(season_b, team_b)
                # Distinct index labels so the same team in two seasons still works.
                combined = pd.DataFrame([stats_a, stats_b], index=["TMA", "TMB"])

                # Predict both ways for a neutral court, then average.
                pred_home = predict_winner(
                    model, feature_cols, combined, "TMA", "TMB", home_team="TMA",
                    games_df=None, injuries_home=None, injuries_away=None,
                )
                pred_away = predict_winner(
                    model, feature_cols, combined, "TMB", "TMA", home_team="TMB",
                    games_df=None, injuries_home=None, injuries_away=None,
                )
                home_prob_a = float(pred_home.get("home_win_prob", 0))
                home_prob_b = float(pred_away.get("home_win_prob", 0))

                # Neutral combine: average probabilities
                prob_a = (home_prob_a + (1 - home_prob_b)) / 2
                prob_b = 1 - prob_a

                prob_a = round(prob_a * 100, 1)
                prob_b = round(prob_b * 100, 1)
                winner = team_a if prob_a >= prob_b else team_b

                stat_factors = pred_home.get("stat_factors", [])
                result = {
                    "winner": winner,
                    "team_a": team_a,
                    "team_b": team_b,
                    "season_a": season_a,
                    "season_b": season_b,
                    "prob_a": prob_a,
                    "prob_b": prob_b,
                }
            except Exception as e:
                error = f"Prediction error: {e}"

    return render_template(
        "historic.html",
        teams=TEAM_OPTIONS,
        seasons=seasons_options,
        result=result,
        selected_a=selected_a,
        selected_b=selected_b,
        season_a=season_a,
        season_b=season_b,
        stat_factors=stat_factors,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)
