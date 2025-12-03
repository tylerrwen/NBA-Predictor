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

    if request.method == "POST":
        home = request.form["home"]
        away = request.form["away"]

        # Run prediction
        result = predict_matchup(home, away)

        # Scrape injuries
        injuries_home = scrape_injuries(home, SEASON)
        injuries_away = scrape_injuries(away, SEASON)

    return render_template(
        "index.html",
        teams=TEAMS,
        result=result,
        injuries_home=injuries_home,
        injuries_away=injuries_away
    )

if __name__ == "__main__":
    app.run(debug=True)
