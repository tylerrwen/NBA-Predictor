import pandas as pd

URL = "https://www.basketball-reference.com/leagues/NBA_2025_standings.html"


def get_standings():
    tables = pd.read_html(URL)

    east = tables[0]
    west = tables[1]

    east["Conference"] = "East"
    west["Conference"] = "West"

    standings = pd.concat([east, west], ignore_index=True)

    # Detect team column
    possible_cols = ["Team", "Teams", "Eastern Conference", "Western Conference"]
    team_col = None

    for col in standings.columns:
        if col in possible_cols:
            team_col = col
            break

    if team_col is None:
        for col in standings.columns:
            if standings[col].dtype == object:
                team_col = col
                break

    if team_col is None:
        raise ValueError("Team column not found.")

    standings["team_clean"] = standings[team_col].fillna("").astype(str).str.lower()

    return standings, team_col



def get_team_standing(team_name: str):
    standings, _ = get_standings()

    team_name = team_name.lower()
    match = standings[standings["team_clean"].str.contains(team_name)]

    if match.empty:
        return None

    return match.iloc[0]  # return one row



# -------------------------------------------------------
#   MATCHUP PREDICTOR WITH HOME/AWAY ADVANTAGE
# -------------------------------------------------------

def predict_matchup(team1: str, team2: str, home_team: str):
    """
    home_team must be either team1 or team2.
    """
    t1 = get_team_standing(team1)
    t2 = get_team_standing(team2)

    if t1 is None:
        return f"Team '{team1}' not found."
    if t2 is None:
        return f"Team '{team2}' not found."

    # Extract stats
    srs1 = float(t1["SRS"])
    srs2 = float(t2["SRS"])

    off1 = float(t1["PS/G"])
    def1 = float(t1["PA/G"])

    off2 = float(t2["PS/G"])
    def2 = float(t2["PA/G"])

    # Base margin from stats & SRS
    margin = (off1 - def2) - (off2 - def1) + (srs1 - srs2)

    # -------------------------------------------------------
    #                HOME-COURT ADVANTAGE
    # -------------------------------------------------------
    HOME_ADVANTAGE = 3.0  # ≈ historically accurate

    home_team = home_team.lower()

    if home_team in t1["team_clean"]:
        margin += HOME_ADVANTAGE
    elif home_team in t2["team_clean"]:
        margin -= HOME_ADVANTAGE
    else:
        return "Error: home_team must match one of the two teams."

    # Convert to win probability via logistic curve
    win_prob = 1 / (1 + 10 ** (-margin / 7))

    return {
        "team1": t1["team_clean"],
        "team2": t2["team_clean"],
        "home_team": home_team,
        "team1_win_prob": round(win_prob, 3),
        "team2_win_prob": round(1 - win_prob, 3),
        "predicted_margin": round(margin, 2)
    }


