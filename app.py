from flask import Flask, render_template, request
from nbaPredictor import predict_matchup
from scrapingRoster import scrape_injuries

app = Flask(__name__)

TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN",
    "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
    "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
    "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
]

SEASON = 2026

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    injuries_home = []
    injuries_away = []
    selected_home = None
    selected_away = None
    error = None

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
            result = predict_matchup(home, away)
            injuries_home = scrape_injuries(home, SEASON)
            injuries_away = scrape_injuries(away, SEASON)

    return render_template(
        "index.html",
        teams=TEAMS,
        result=result,
        injuries_home=injuries_home,
        injuries_away=injuries_away,
        selected_home=selected_home,
        selected_away=selected_away,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
