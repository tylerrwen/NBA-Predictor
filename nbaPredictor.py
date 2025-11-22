

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import scrapers from your modules (must exist in same folder)
from scrapingStats import scrape_team_stats
from scrapingStandings import get_team_standing

TEAMS_ALL = [
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW",
    "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
    "OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS"
]

SEASON = 2026  # change to 2025 if needed
MIN_SEASON = 2000  # for fallback search
REQUEST_DELAY = 1.0  # seconds between requests to be polite


def to_num(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s == ".":
        return np.nan
    try:
        return float(s.replace(",", ""))
    except:
        # fallback: keep digits, dot, minus
        cleaned = "".join(ch for ch in s if (ch.isdigit() or ch in ".-"))
        try:
            return float(cleaned) if cleaned not in ("", ".", "-") else np.nan
        except:
            return np.nan


def find_working_season(team, start_season, min_season=MIN_SEASON):
    """
    Try start_season down to min_season until scrape_team_stats returns non-empty list.
    Returns the first season that yields gamelogs (or None).
    """
    for s in range(start_season, min_season - 1, -1):
        time.sleep(REQUEST_DELAY)
        try:
            games = scrape_team_stats(team, s)
        except Exception:
            games = []
        if games:
            return s
    return None


def build_games_dataframe(teams, season):
    all_games = []
    seen_pairs = set()  # dedupe by (date, home, away)

    for team in teams:
        # polite delay
        time.sleep(REQUEST_DELAY)
        try:
            games = scrape_team_stats(team, season)
        except Exception as e:
            print(f"  ⚠ Scraper error for {team} {season}: {e}")
            # try to find an earlier season for this team automatically
            fallback = find_working_season(team, season-1)
            if fallback:
                print(f"    → Found fallback season {fallback} for {team}, scraping that.")
                time.sleep(REQUEST_DELAY)
                try:
                    games = scrape_team_stats(team, fallback)
                except Exception as e2:
                    print(f"    ⚠ fallback scrape failed for {team}: {e2}")
                    games = []
            else:
                games = []

        if not games:
            print(f"  ⚠ No game log found for {team} {season}. Skipping.")
            continue

        for g in games:
            # identify keys commonly present
            team_pts = g.get("team_game_score") or g.get("pts") or g.get("team_pts") or g.get("team_game_score_home")
            opp_pts = g.get("opp_team_game_score") or g.get("opp_pts") or g.get("opp_team_game_score")
            opp = g.get("opp_name_abbr") or g.get("opp_name") or g.get("opp")
            date = g.get("date") or g.get("date_game") or g.get("game_date")

            # home flag from your scraped dict (boolean) or game_location
            home_flag = g.get("home")
            if home_flag is None:
                loc = g.get("game_location", "")
                home_flag = (loc != "@")

            if not opp or date is None:
                continue

            # canonicalize
            team_upper = team.upper()
            opp_upper = opp.upper() if isinstance(opp, str) else opp

            if home_flag:
                home_team = team_upper
                away_team = opp_upper
                home_points = to_num(team_pts)
                away_points = to_num(opp_pts)
            else:
                home_team = opp_upper
                away_team = team_upper
                home_points = to_num(opp_pts)
                away_points = to_num(team_pts)

            if np.isnan(home_points) or np.isnan(away_points):
                continue

            key = (str(date).strip(), home_team, away_team)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            all_games.append({
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "home_pts": home_points,
                "away_pts": away_points
            })

    if not all_games:
        return pd.DataFrame(columns=["date","home_team","away_team","home_pts","away_pts","home_win"])

    df = pd.DataFrame(all_games)
    # parse date if possible
    try:
        df["date"] = pd.to_datetime(df["date"])
    except:
        pass
    df["home_win"] = (df["home_pts"] > df["away_pts"]).astype(int)
    df = df.drop_duplicates(subset=["date","home_team","away_team"]).reset_index(drop=True)
    return df


def compute_team_aggregates(games_df):
    if games_df.empty:
        return pd.DataFrame()

    home = games_df.groupby("home_team").agg(
        home_games=("home_pts","count"),
        home_pts_total=("home_pts","sum"),
        home_pts_avg=("home_pts","mean"),
        home_opp_pts_avg=("away_pts","mean")
    ).rename_axis("team").reset_index()

    away = games_df.groupby("away_team").agg(
        away_games=("away_pts","count"),
        away_pts_total=("away_pts","sum"),
        away_pts_avg=("away_pts","mean"),
        away_opp_pts_avg=("home_pts","mean")
    ).rename_axis("team").reset_index()

    teams = pd.merge(home, away, how="outer", on="team").fillna(0)
    teams["games_played"] = teams["home_games"] + teams["away_games"]
    teams["pts_for_avg"] = (teams["home_pts_total"] + teams["away_pts_total"]) / teams["games_played"].replace(0, np.nan)
    teams["pts_against_avg"] = ((teams["home_opp_pts_avg"] * teams["home_games"]) + (teams["away_opp_pts_avg"] * teams["away_games"])) / teams["games_played"].replace(0, np.nan)
    teams["pt_diff"] = teams["pts_for_avg"] - teams["pts_against_avg"]
    teams = teams.fillna(0).set_index("team")
    return teams


def build_feature_matrix_and_train(games_df, teams_df):
    rows = []
    labels = []
    for _, r in games_df.iterrows():
        h = r["home_team"]
        a = r["away_team"]
        if h not in teams_df.index or a not in teams_df.index:
            continue
        hstats = teams_df.loc[h]
        astats = teams_df.loc[a]
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
        raise ValueError("No training data available after filtering. Check scraped data.")

    X = pd.DataFrame(rows).fillna(0)
    y = pd.Series(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
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
        raise ValueError("One or both teams missing aggregate stats.")
    h = teams_df.loc[home]; a = teams_df.loc[away]
    feat = {
        "home_pts_for_avg": h["pts_for_avg"],
        "home_pts_against_avg": h["pts_against_avg"],
        "home_pt_diff": h["pt_diff"],
        "away_pts_for_avg": a["pts_for_avg"],
        "away_pts_against_avg": a["pts_against_avg"],
        "away_pt_diff": a["pt_diff"],
        "home_flag": 1
    }
    X_row = pd.DataFrame([feat])[feature_cols].fillna(0)
    prob_home = model.predict_proba(X_row)[0][1]
    prob_away = 1 - prob_home
    predicted = home if prob_home >= 0.5 else away
    display_prediction_report(home, away, prob_home, prob_away, predicted, dict(h), dict(a))
    return {"home_team": home, "away_team": away, "home_win_prob": round(prob_home,3), "away_win_prob": round(prob_away,3), "predicted_winner": predicted}


def main(teams=TEAMS_ALL, season=SEASON):
    print("Starting NBA Predictor pipeline...")
    games_df = build_games_dataframe(teams, season)
    if games_df.empty:
        print("No games were scraped for any team. Exiting.")
        return
    print(f"Scraped {len(games_df)} unique games from season {season}.")
    teams_df = compute_team_aggregates(games_df)
    print("\nTeam aggregates (sample):")
    print(teams_df.head())
    model, feature_cols = build_feature_matrix_and_train(games_df, teams_df)
    # Example prediction: take first two teams with aggregates
    available = list(teams_df.index)
    if len(available) >= 2:
        a = available[0]; b = available[1]; home = a
        print("\nExample prediction:")
        predict_winner(model, feature_cols, teams_df, a, b, home)
    else:
        print("Not enough teams with aggregates to produce an example prediction.")
    return {"games_df": games_df, "teams_df": teams_df, "model": model, "feature_cols": feature_cols}

if __name__ == "__main__":
    out = main()
