import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

TEAMS = [
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW",
    "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
    "OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS"
]
SEASON = 2026
USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; NBA-Predictor/1.0)"}

def to_num(x):
    """Convert scraped value (string or None) to float or NaN."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s == ".":
        return np.nan
    # Some csk numeric attributes might be integers stored as strings; handle commas
    try:
        return float(s.replace(",", ""))
    except:
        # Try stripping non-digit characters
        try:
            cleaned = "".join(ch for ch in s if (ch.isdigit() or ch in ".-"))
            return float(cleaned) if cleaned not in ("", ".", "-") else np.nan
        except:
            return np.nan

def scrape_team_stats(team, season):
    """
    Scrape a team's gamelog page and return a list of dicts per game.
    Robust to different table IDs and tables inside HTML comments.
    Returns [] if no usable gamelog found.
    """
    url = f"https://www.basketball-reference.com/teams/{team}/{season}/gamelog/"
    print(f"Requesting: {url}")
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=20)
    except Exception as e:
        print(f"  ⚠ Request failed for {team} {season}: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")

    # Candidate table ids (cover many historical and new layouts)
    candidate_ids = [
        "team_game_log_reg", "team_game_log", "tgl_basic", "team_and_opponent",
        "team_game_log_reg", "team_game_log_playoffs", "team_game_log_all"
    ]

    # 1) Try to find by these ids in the visible DOM
    table = None
    for tid in candidate_ids:
        table = soup.find("table", {"id": tid})
        if table:
            break

    # 2) If not found, search inside HTML comments (BBRef often hides tables there)
    if table is None:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            # Quick check to reduce parsing
            if "<table" not in c:
                continue
            try:
                csoup = BeautifulSoup(c, "html.parser")
            except Exception:
                continue
            for tid in candidate_ids:
                table = csoup.find("table", {"id": tid})
                if table:
                    break
            if table:
                break

    # 3) If still not found, fallback: find any table with 'pts' and 'opp_pts' data-stat headers
    if table is None:
        for t in soup.find_all("table"):
            thead = t.find("thead")
            if not thead:
                continue
            headers = [th.get("data-stat") or th.get_text(strip=True) for th in thead.find_all("th")]
            if any(h and "pts" in h for h in headers) and any(h and "opp" in str(h).lower() for h in headers):
                table = t
                break
        # try comments too
        if table is None:
            for c in comments:
                try:
                    csoup = BeautifulSoup(c, "html.parser")
                except Exception:
                    continue
                for t in csoup.find_all("table"):
                    thead = t.find("thead")
                    if not thead:
                        continue
                    headers = [th.get("data-stat") or th.get_text(strip=True) for th in thead.find_all("th")]
                    if any(h and "pts" in h for h in headers) and any(h and "opp" in str(h).lower() for h in headers):
                        table = t
                        break
                if table:
                    break

    if table is None:
        print(f"  ⚠ No game log found for {team} {season}. Skipping.")
        return []

    tbody = table.find("tbody")
    if tbody is None:
        print(f"  ⚠ Found table but no tbody for {team} {season}. Skipping.")
        return []

    games = []
    for row in tbody.find_all("tr"):
        # skip in-table header rows
        if row.get("class") and "thead" in row.get("class"):
            continue

        # sometimes rows are separators with no <td>
        tds = row.find_all("td")
        if not tds:
            continue

        game_data = {}
        for td in tds:
            stat = td.get("data-stat")
            if not stat:
                continue
            # numeric values often in 'csk'
            val = td.get("csk") if td.get("csk") is not None else td.get_text(strip=True)
            game_data[stat] = val

        # Determine home/away from game_location or presence of '@'
        loc = game_data.get("game_location", "")
        # Some pages set empty string for home; '@' for away.
        is_home = True if loc == "" or loc is None else (loc != "@")
        game_data["home"] = is_home

        games.append(game_data)

    return games

def build_games_dataframe(teams, season):
    """
    For each team in teams, scrape gamelogs and build a deduplicated game-level DataFrame:
    columns: date, home_team, away_team, home_pts, away_pts, home_win
    """
    all_games = []
    for team in teams:
        games = scrape_team_stats(team, season)
        if not games:
            continue

        for g in games:
            # common keys for scores may be: 'pts', 'team_game_score', 'team_game_score' etc.
            team_pts = g.get("pts") or g.get("team_game_score") or g.get("team_pts") or g.get("team_game_score_home")
            opp_pts = g.get("opp_pts") or g.get("opp_team_game_score") or g.get("opp_team_game_score")
            # opponent code might be opp_name_abbr or opp_name
            opp = g.get("opp_name_abbr") or g.get("opp_name") or g.get("opp")
            date = g.get("date_game") or g.get("date") or g.get("game_date")

            home_flag = g.get("home", True)
            # canonicalize team & opponent strings
            home_team = team if home_flag else (opp if opp else None)
            away_team = (opp if home_flag else team) if opp else (None if not home_flag else None)

            # convert to numeric
            home_points = to_num(team_pts) if home_flag else to_num(opp_pts)
            away_points = to_num(opp_pts) if home_flag else to_num(team_pts)

            # skip rows without numeric scores or without opponent
            if home_team is None or away_team is None:
                continue
            if np.isnan(home_points) or np.isnan(away_points):
                continue

            all_games.append({
                "date": date,
                "home_team": home_team.upper() if isinstance(home_team, str) else home_team,
                "away_team": away_team.upper() if isinstance(away_team, str) else away_team,
                "home_pts": home_points,
                "away_pts": away_points
            })

    if not all_games:
        return pd.DataFrame(columns=["date","home_team","away_team","home_pts","away_pts","home_win"])

    df = pd.DataFrame(all_games)
    # dedupe (a game might appear twice)
    df = df.drop_duplicates(subset=["date","home_team","away_team"])
    # compute label
    df["home_win"] = (df["home_pts"] > df["away_pts"]).astype(int)
    # try parse date
    try:
        df["date"] = pd.to_datetime(df["date"])
    except:
        pass
    df = df.reset_index(drop=True)
    return df

def compute_team_aggregates(games_df):
    """
    Returns DataFrame indexed by team abbreviation with columns:
    pts_for_avg, pts_against_avg, pt_diff, games_played
    """
    # aggregate home perspective
    home = games_df.groupby("home_team").agg(
        home_games=("home_pts", "count"),
        home_pts_total=("home_pts", "sum"),
        home_pts_avg=("home_pts", "mean"),
        home_opp_pts_avg=("away_pts", "mean")
    ).rename_axis("team")
    # aggregate away perspective
    away = games_df.groupby("away_team").agg(
        away_games=("away_pts", "count"),
        away_pts_total=("away_pts", "sum"),
        away_pts_avg=("away_pts", "mean"),
        away_opp_pts_avg=("home_pts", "mean")
    ).rename_axis("team")

    teams = pd.merge(home.reset_index(), away.reset_index(), how="outer", left_on="team", right_on="team").fillna(0)
    teams["games_played"] = teams["home_games"] + teams["away_games"]
    teams["pts_for_avg"] = (teams["home_pts_total"] + teams["away_pts_total"]) / teams["games_played"].replace(0, np.nan)
    teams["pts_against_avg"] = ((teams["home_opp_pts_avg"] * teams["home_games"]) + (teams["away_opp_pts_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["pt_diff"] = teams["pts_for_avg"] - teams["pts_against_avg"]
    teams = teams.fillna(0).set_index("team")
    return teams

def build_feature_matrix_and_train(games_df, teams_df):
    """
    Builds X, y from games and teams aggregates and trains RandomForest.
    Returns trained model and feature column order.
    """
    # create features per game
    rows = []
    labels = []
    for _, r in games_df.iterrows():
        h = r["home_team"]
        a = r["away_team"]
        hstats = teams_df.loc[h] if h in teams_df.index else None
        astats = teams_df.loc[a] if a in teams_df.index else None

        # fallbacks
        if hstats is None or astats is None:
            # skip games where either team is missing aggregates
            continue

        feat = {
            "home_pts_for_avg": hstats["pts_for_avg"],
            "home_pts_against_avg": hstats["pts_against_avg"],
            "home_pt_diff": hstats["pt_diff"],
            "away_pts_for_avg": astats["pts_for_avg"],
            "away_pts_against_avg": astats["pts_against_avg"],
            "away_pt_diff": astats["pt_diff"],
            "home_flag": 1
        }
        rows.append(feat)
        labels.append(r["home_win"])

    if not rows:
        raise ValueError("No training rows available. Check scraped games and aggregates.")

    X = pd.DataFrame(rows).fillna(0)
    y = pd.Series(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model trained — test accuracy: {acc:.3f} ({len(X_train)} train, {len(X_test)} test)")
    return model, X.columns.tolist()

def display_prediction_report(home_team, away_team, home_prob, away_prob, winner, home_stats, away_stats):
    print("\n" + "="*60)
    print("                 NBA GAME PREDICTION REPORT")
    print("="*60 + "\n")

    print(f"Matchup: {home_team} (Home)  vs  {away_team} (Away)\n")

    print("-"*60)
    print(f"🏆  Predicted Winner: {winner}")
    print("-"*60 + "\n")

    print("📈 Win Probabilities:")
    print(f"  • {home_team}: {home_prob*100:.2f}%")
    print(f"  • {away_team}: {away_prob*100:.2f}%\n")

    print("-"*60)
    print("📊 Team Season Averages\n")

    print(f"{home_team}:")
    print(f"  • Points Scored:      {home_stats['pts_for_avg']:.1f}")
    print(f"  • Points Allowed:     {home_stats['pts_against_avg']:.1f}")
    print(f"  • Point Differential: {home_stats['pt_diff']:+.1f}\n")

    print(f"{away_team}:")
    print(f"  • Points Scored:      {away_stats['pts_for_avg']:.1f}")
    print(f"  • Points Allowed:     {away_stats['pts_against_avg']:.1f}")
    print(f"  • Point Differential: {away_stats['pt_diff']:+.1f}")

    print("\n" + "="*60 + "\n")

def predict_winner(model, feature_cols, teams_df, teamA, teamB, home_team):
    teamA = teamA.upper(); teamB = teamB.upper(); home_team = home_team.upper()
    if home_team not in (teamA, teamB):
        raise ValueError("home_team must be either teamA or teamB")

    home = home_team
    away = teamB if home == teamA else teamA

    if home not in teams_df.index or away not in teams_df.index:
        return {"error": "One or both teams missing aggregate stats; ensure both were scraped."}

    h = teams_df.loc[home]
    a = teams_df.loc[away]

    feat = {
        "home_pts_for_avg": h["pts_for_avg"],
        "home_pts_against_avg": h["pts_against_avg"],
        "home_pt_diff": h["pt_diff"],
        "away_pts_for_avg": a["pts_for_avg"],
        "away_pts_against_avg": a["pts_against_avg"],
        "away_pt_diff": a["pt_diff"],
        "home_flag": 1
    }

    # ensure column order
    X_row = pd.DataFrame([feat])[feature_cols].fillna(0)
    prob_home = model.predict_proba(X_row)[0][1]
    prob_away = 1 - prob_home
    predicted = home if prob_home >= 0.5 else away

    # pretty print
    display_prediction_report(home, away, prob_home, prob_away, predicted, dict(h), dict(a))

    return {
        "home_team": home,
        "away_team": away,
        "home_win_prob": round(prob_home, 3),
        "away_win_prob": round(prob_away, 3),
        "predicted_winner": predicted
    }

def main(teams=TEAMS, season=SEASON):
    print("Starting NBA Predictor pipeline...")
    games_df = build_games_dataframe(teams, season)
    if games_df.empty:
        print("No games were scraped for any team. Exiting.")
        return

    print(f"Scraped {len(games_df)} unique games from {season} season data.")
    teams_df = compute_team_aggregates(games_df)
    print("\nTeam aggregates (sample):")
    print(teams_df.head())

    model, feature_cols = build_feature_matrix_and_train(games_df, teams_df)

    # Example prediction using first two available scraped teams
    available_teams = list(teams_df.index)
    if len(available_teams) >= 2:
        a = available_teams[0]; b = available_teams[1]; home = a
        print("\nExample prediction:")
        predict_winner(model, feature_cols, teams_df, a, b, home)
    else:
        print("Not enough teams with aggregates to run example prediction.")

    # Return objects for interactive usage or testing
    return {
        "games_df": games_df,
        "teams_df": teams_df,
        "model": model,
        "feature_cols": feature_cols
    }

if __name__ == "__main__":
    out = main()
