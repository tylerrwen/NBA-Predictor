from flask import Flask, render_template, request
from nbaPredictor import load_model_and_data, predict_winner, main as train_season_model
from scrapingRoster import scrape_injuries

app = Flask(__name__)

TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN",
    "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
    "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
    "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
]

TEAM_NAME_MAP = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
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

model, feature_cols, games_df, teams_df, _ = load_model_and_data()

# Simple in-memory cache for season/team combinations
# Key: (season, tuple(sorted(team_abbrs))) -> (model, feature_cols, games_df, teams_df)
SEASON_CACHE = {(SEASON, tuple(sorted(TEAMS))): (model, feature_cols, games_df, teams_df)}

def get_season_resources(season: int, team_abbrs):
    """
    Load or train model/data for a specific season and a specific subset of teams.
    We only scrape/train the teams requested to avoid scraping the entire season.
    """
    team_list = sorted(set(team_abbrs or TEAMS))
    cache_key = (season, tuple(team_list))
    if cache_key in SEASON_CACHE:
        return SEASON_CACHE[cache_key]
    # Train or load for this season with only requested teams
    out = train_season_model(teams=team_list, season=season, force_retrain=True)
    mdl = out.get("model")
    fcols = out.get("feature_cols")
    gdf = out.get("games_df")
    tdf = out.get("teams_df")
    SEASON_CACHE[cache_key] = (mdl, fcols, gdf, tdf)
    return mdl, fcols, gdf, tdf

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
            injuries_home = scrape_injuries(home, SEASON)
            injuries_away = scrape_injuries(away, SEASON)
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
        selected_home=selected_home,
        selected_away=selected_away,
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
    season_a = SEASONS[0]
    season_b = SEASONS[0]
    error = None

    seasons_options = [{"value": s, "label": f"{s-1}-{s}"} for s in SEASONS]

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
                # Load/train season-specific resources for only the teams in that season
                model_a, fcols_a, gdf_a, tdf_a = get_season_resources(season_a, [team_a])
                model_b, fcols_b, gdf_b, tdf_b = get_season_resources(season_b, [team_b])

                # Predict twice (neutral): Team A home in season A model, Team B home in season B model
                pred_home = predict_winner(
                    model_a, fcols_a, tdf_a, team_a, team_b, home_team=team_a, games_df=gdf_a,
                    injuries_home=None, injuries_away=None
                )
                home_prob_a = float(pred_home.get("home_win_prob", 0))

                pred_away_home = predict_winner(
                    model_b, fcols_b, tdf_b, team_b, team_a, home_team=team_b, games_df=gdf_b,
                    injuries_home=None, injuries_away=None
                )
                home_prob_b = float(pred_away_home.get("home_win_prob", 0))

                # Neutral combine: average probabilities
                prob_a = (home_prob_a + (1 - home_prob_b)) / 2
                prob_b = 1 - prob_a

                prob_a = round(prob_a * 100, 1)
                prob_b = round(prob_b * 100, 1)
                winner = team_a if prob_a >= prob_b else team_b

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
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)
    import os
    print("TEMPLATE FOLDER LOCATION:", os.path.abspath("templates"))

