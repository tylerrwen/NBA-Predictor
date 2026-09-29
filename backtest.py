"""
Backtest harness for measuring the train/serve skew (review item #5).

The model is TRAINED on point-in-time features (team aggregates + Elo/rest
recomputed strictly from games before each game). It is SERVED from a static
`teams_df` snapshot that predict_winner treats as current strength -- a cache
built once and refreshed only periodically. This script quantifies how much that
difference costs by replaying a season's late games both ways against the actual
results:

  * point-in-time : fresh aggregates + per-game Elo/rest (the "as trained" path)
  * serve-style   : static season-to-boundary aggregates via predict_winner
                    (the "as served" path -- a frozen cache)

The gap between the two sets of metrics is the skew. Both paths use the same
trained model and the same [0.03, 0.97] production clamp, so the ONLY difference
is how the features are constructed.

Usage:
    python backtest.py                      # default: SEASON-1, last 25% of games
    python backtest.py --season 2025 --holdout 0.3
    python backtest.py --season 2025 --scrape   # scrape instead of using the cache
"""
import argparse
import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import nbaPredictor as nba
from config import SEASON, BR_TEAMS

CLIP_LO, CLIP_HI = 0.03, 0.97  # match the production guardrail in predict_winner


def _clip(p):
    return min(CLIP_HI, max(CLIP_LO, float(p)))


def _load_season_games(season, use_cached):
    """Return a games dataframe for one season, from the training cache if present."""
    if use_cached and os.path.exists(nba.TRAINING_CACHE_FILE):
        with open(nba.TRAINING_CACHE_FILE, "rb") as f:
            g = pickle.load(f)
        if "season" in g.columns and season in set(g["season"]):
            return g[g["season"] == season].copy()
        print(f"  season {season} not in training cache; falling back to scrape")
    print(f"  scraping season {season} from basketball-reference (slow, network)...")
    df = nba.build_games_dataframe(BR_TEAMS, season)
    if not df.empty:
        df["season"] = season
    return df


def _point_in_time_prob(model, feature_cols, season_df, row, agg_cache):
    """Predict the way the model was trained: aggregates + Elo/rest from games
    strictly before this game's date."""
    game_date = row["date"]
    if game_date not in agg_cache:
        agg_cache[game_date] = nba.compute_team_aggregates(season_df[season_df["date"] < game_date])
    agg = agg_cache[game_date]
    h, a = row["home_team"], row["away_team"]
    if h not in agg.index or a not in agg.index:
        return None
    home_recent = nba.compute_recent_form(season_df, h, game_date, nba.RECENT_FORM_WINDOW)
    away_recent = nba.compute_recent_form(season_df, a, game_date, nba.RECENT_FORM_WINDOW)
    extras = nba._elo_rest_extras(
        row["home_elo_pre"], row["away_elo_pre"],
        row["home_rest"], row["away_rest"], row["home_b2b"], row["away_b2b"],
    )
    feat = nba.build_feature_dict(agg.loc[h], agg.loc[a], home_recent, away_recent, extras)
    X = pd.DataFrame([{c: feat.get(c, 0) for c in feature_cols}])[feature_cols].fillna(0)
    return float(model.predict_proba(X)[0][1])


def _serve_prob(model, feature_cols, serve_teams, pre_games, row):
    """Predict the way the app serves: a static teams_df snapshot frozen at the
    holdout boundary, scored through predict_winner (raw, no injuries)."""
    h, a = row["home_team"], row["away_team"]
    try:
        res = nba.predict_winner(
            model, feature_cols, serve_teams, h, a, home_team=h, games_df=pre_games,
        )
    except Exception:
        return None
    return res["adjustment_detail"]["raw_model_prob_home"]


def _train_excluding(season):
    """Train a fresh model on every cached season EXCEPT `season`, so the backtest
    of `season` is genuinely out-of-sample (removes in-sample optimism)."""
    if not os.path.exists(nba.TRAINING_CACHE_FILE):
        raise SystemExit("--retrain needs the training cache. Run: python build_cache.py --current")
    with open(nba.TRAINING_CACHE_FILE, "rb") as f:
        all_games = pickle.load(f)
    train = all_games[all_games["season"] != season].copy()
    if train.empty:
        raise SystemExit(f"No other seasons in the cache to train on (only {season}).")
    train, _ = nba.compute_elo_and_rest(train)
    print(f"  retraining out-of-sample on seasons {sorted(train['season'].unique())} ...")
    model, feature_cols, metrics = nba.build_feature_matrix_and_train(train)
    print(f"  held-out test acc during (re)training: {metrics.get('test_accuracy')}")
    return model, feature_cols


def backtest_season(season, holdout_frac=0.25, use_cached=True, retrain=False):
    if retrain:
        model, feature_cols = _train_excluding(season)
    else:
        model, feature_cols, _, _, _ = nba.load_model_and_data()
        if model is None:
            raise SystemExit("No trained model on disk. Run: python build_cache.py --current")

    games = _load_season_games(season, use_cached)
    if games is None or games.empty:
        raise SystemExit(f"No games available for season {season}.")

    games = games.sort_values("date").reset_index(drop=True)
    games, _ = nba.compute_elo_and_rest(games)  # adds *_elo_pre, *_rest, *_b2b

    n = len(games)
    split = int(n * (1 - holdout_frac))
    pre, holdout = games.iloc[:split], games.iloc[split:]
    if holdout.empty:
        raise SystemExit("Holdout is empty; lower --holdout or use a fuller season.")

    # Static serve-time resources, frozen at the boundary (mimics the cached teams_df).
    serve_teams = nba.compute_team_aggregates(pre)
    _, boundary_elo = nba.compute_elo_and_rest(pre.copy())
    serve_teams["elo"] = pd.Series(boundary_elo).reindex(serve_teams.index).fillna(nba.ELO_BASE)

    y, p_pit, p_serve = [], [], []
    agg_cache = {}
    for _, row in holdout.iterrows():
        pit = _point_in_time_prob(model, feature_cols, games, row, agg_cache)
        srv = _serve_prob(model, feature_cols, serve_teams, pre, row)
        if pit is None or srv is None:
            continue
        y.append(int(row["home_win"]))
        p_pit.append(_clip(pit))
        p_serve.append(_clip(srv))

    if not y:
        raise SystemExit("No scorable holdout games (teams missing prior data).")

    def metrics(probs):
        preds = [1 if p >= 0.5 else 0 for p in probs]
        return accuracy_score(y, preds), log_loss(y, probs, labels=[0, 1])

    acc_pit, ll_pit = metrics(p_pit)
    acc_srv, ll_srv = metrics(p_serve)
    base_rate = sum(y) / len(y)
    home_baseline = max(base_rate, 1 - base_rate)

    return {
        "season": season,
        "n_games": n,
        "n_scored": len(y),
        "home_win_rate": round(base_rate, 4),
        "home_baseline_accuracy": round(home_baseline, 4),
        "point_in_time": {"accuracy": round(acc_pit, 4), "log_loss": round(ll_pit, 4)},
        "serve_style": {"accuracy": round(acc_srv, 4), "log_loss": round(ll_srv, 4)},
        "skew": {
            "accuracy_drop": round(acc_pit - acc_srv, 4),
            "log_loss_increase": round(ll_srv - ll_pit, 4),
        },
    }


def print_report(r):
    print(f"\nBacktest — season {r['season']} ({r['n_scored']}/{r['n_games']} games scored)")
    print(f"  home win rate: {r['home_win_rate']:.1%}   home-picks baseline acc: {r['home_baseline_accuracy']:.4f}")
    print(f"  {'method':<16}{'accuracy':>10}{'log_loss':>10}")
    print(f"  {'point-in-time':<16}{r['point_in_time']['accuracy']:>10}{r['point_in_time']['log_loss']:>10}")
    print(f"  {'serve-style':<16}{r['serve_style']['accuracy']:>10}{r['serve_style']['log_loss']:>10}")
    print(f"  {'-> skew':<16}{r['skew']['accuracy_drop']:>+10}{r['skew']['log_loss_increase']:>+10}")
    print("  (positive skew numbers mean the serve path is worse than point-in-time)\n")


def main():
    ap = argparse.ArgumentParser(description="Measure train/serve skew for the NBA predictor.")
    ap.add_argument("--season", type=int, default=SEASON - 1,
                    help="season to backtest (default: previous completed season)")
    ap.add_argument("--holdout", type=float, default=0.25,
                    help="fraction of late-season games to evaluate (default: 0.25)")
    ap.add_argument("--scrape", action="store_true",
                    help="scrape the season instead of using the training cache")
    ap.add_argument("--retrain", action="store_true",
                    help="retrain excluding the backtest season for an out-of-sample (trustworthy) measurement")
    args = ap.parse_args()
    print_report(backtest_season(args.season, args.holdout,
                                 use_cached=not args.scrape, retrain=args.retrain))


if __name__ == "__main__":
    main()
