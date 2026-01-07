from flask import Flask, render_template, request
from nbaPredictor import load_model_and_data, predict_winner
from scrapingRoster import scrape_injuries

app = Flask(__name__)

TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN",
    "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
    "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
    "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
]

SEASON = 2026
SEASONS = list(range(2000, SEASON + 1))[::-1]  # 2026 down to 2000

model, feature_cols, games_df, teams_df, _ = load_model_and_data()

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
        teams=TEAMS,
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
                # Reuse the existing model pipeline (no cross-season retrain here).
                pred = predict_winner(
                    model, feature_cols, teams_df, team_a, team_b, home_team=team_a, games_df=games_df,
                    injuries_home=None, injuries_away=None
                )
                winner = pred.get("predicted_winner", team_a)
                home_prob = float(pred.get("home_win_prob", 0))
                prob_a = round(home_prob * 100, 1)          # team_a is home
                prob_b = round((1 - home_prob) * 100, 1)    # team_b is away
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
        teams=TEAMS,
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

