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

if __name__ == "__main__":
    app.run(debug=True)
    import os
    print("TEMPLATE FOLDER LOCATION:", os.path.abspath("templates"))

