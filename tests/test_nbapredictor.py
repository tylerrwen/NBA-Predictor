"""
Unit tests for the pure (no-network, no-model) helpers in the NBA predictor.

These cover data cleaning, Elo math, feature construction, recent-form, the
stat-factor summary, and the injury-tier heuristic. Run with:  pytest
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the package importable when pytest is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nbaPredictor as nba
from scrapingPlayers import _tier


# --- to_num ---------------------------------------------------------------

def test_to_num_passthrough_numbers():
    assert nba.to_num(5) == 5.0
    assert nba.to_num(3.5) == 3.5


def test_to_num_strips_commas():
    assert nba.to_num("1,234") == 1234.0


def test_to_num_extracts_from_dirty_string():
    assert nba.to_num("$45.5") == 45.5
    assert nba.to_num("12abc") == 12.0


@pytest.mark.parametrize("bad", [None, "", ".", "-", "abc"])
def test_to_num_returns_nan_for_junk(bad):
    assert np.isnan(nba.to_num(bad))


# --- Elo math -------------------------------------------------------------

def test_elo_expected_equal_ratings_favors_home():
    # Equal ratings but home-court advantage -> home prob > 0.5.
    assert nba._elo_expected(1500, 1500) > 0.5


def test_elo_expected_neutral_court_is_even():
    assert nba._elo_expected(1500, 1500, home_adv=0) == pytest.approx(0.5)


def test_elo_expected_stronger_team_more_likely():
    strong = nba._elo_expected(1700, 1500)
    weak = nba._elo_expected(1300, 1500)
    assert strong > weak


def test_elo_rest_extras_caps_rest_at_five():
    extras = nba._elo_rest_extras(1500, 1500, home_rest=10, away_rest=2, home_b2b=0, away_b2b=0)
    assert extras["home_rest"] == 5.0          # 10 days capped to 5
    assert extras["away_rest"] == 2.0
    assert extras["rest_diff"] == pytest.approx(3.0)
    assert extras["elo_diff"] == 0.0


# --- build_feature_dict ---------------------------------------------------

def _stats(**overrides):
    """A minimal aggregate-stats dict with every key build_feature_dict reads."""
    base = {k: 0.0 for k in (
        "pts_for_avg", "pts_against_avg", "pt_diff", "fg_pct_avg", "fg3_pct_avg",
        "ft_pct_avg", "trb_avg", "orb_avg", "drb_avg", "ast_avg", "stl_avg",
        "blk_avg", "tov_avg", "opp_fg_pct_avg", "opp_fg3_pct_avg", "opp_ast_avg",
        "opp_tov_avg",
    )}
    base.update(overrides)
    return base


def test_build_feature_dict_home_flag_always_one():
    feat = nba.build_feature_dict(_stats(), _stats(), None, None)
    assert feat["home_flag"] == 1


def test_build_feature_dict_recent_form_fallback():
    # With no recent-form data, win pct defaults to 0.5 and pt_diff falls back
    # to the season pt_diff.
    feat = nba.build_feature_dict(_stats(pt_diff=4.0), _stats(pt_diff=-1.0), None, None)
    assert feat["home_recent_win_pct"] == 0.5
    assert feat["home_recent_pt_diff"] == 4.0
    assert feat["away_recent_pt_diff"] == -1.0
    assert feat["recent_form_diff"] == 0.0


def test_build_feature_dict_tov_diff_sign():
    # tov_diff is (away - home): fewer home turnovers should be positive for home.
    feat = nba.build_feature_dict(_stats(tov_avg=10.0), _stats(tov_avg=15.0), None, None)
    assert feat["tov_diff"] == pytest.approx(5.0)


# --- compute_recent_form --------------------------------------------------

def _five_game_df(team="LAL"):
    """Five games for `team`, all wins by 10, as home team."""
    rows = []
    for i in range(5):
        rows.append({
            "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
            "home_team": team,
            "away_team": "BOS",
            "home_pts": 110.0,
            "away_pts": 100.0,
            "home_win": 1,
        })
    return pd.DataFrame(rows)


def test_compute_recent_form_all_wins():
    df = _five_game_df()
    form = nba.compute_recent_form(df, "LAL", pd.Timestamp("2025-02-01"), window=5)
    assert form is not None
    assert form["recent_win_pct"] == 1.0
    assert form["recent_pt_diff"] == pytest.approx(10.0)
    assert form["recent_pts_for"] == pytest.approx(110.0)


def test_compute_recent_form_insufficient_games_returns_none():
    df = _five_game_df().head(3)
    assert nba.compute_recent_form(df, "LAL", pd.Timestamp("2025-02-01"), window=5) is None


# --- build_stat_factors ---------------------------------------------------

def test_build_stat_factors_direction_for_lower_is_better():
    # Turnovers: lower is better, so the team with fewer TOs is the "better" side.
    home = _stats(tov_avg=10.0)
    away = _stats(tov_avg=14.0)
    factors = nba.build_stat_factors("LAL", "BOS", home, away)
    tov = next(f for f in factors if f["key"] == "tov_avg")
    assert tov["better"] == "home"
    assert tov["advantage"] == pytest.approx(4.0)


def test_build_stat_factors_sorted_and_limited():
    home = _stats(pt_diff=8.0, fg_pct_avg=0.5)
    away = _stats(pt_diff=1.0, fg_pct_avg=0.49)
    factors = nba.build_stat_factors("LAL", "BOS", home, away, top_n=3)
    assert len(factors) == 3
    advantages = [f["advantage"] for f in factors]
    assert advantages == sorted(advantages, reverse=True)


# --- merge_games_data -----------------------------------------------------

def test_merge_games_data_dedupes_on_date_and_teams():
    existing = pd.DataFrame([
        {"date": "2025-01-01", "home_team": "LAL", "away_team": "BOS", "home_pts": 100, "away_pts": 90},
    ])
    new = pd.DataFrame([
        # duplicate of the existing game + one genuinely new game
        {"date": "2025-01-01", "home_team": "LAL", "away_team": "BOS", "home_pts": 100, "away_pts": 90},
        {"date": "2025-01-03", "home_team": "GSW", "away_team": "DEN", "home_pts": 120, "away_pts": 118},
    ])
    merged = nba.merge_games_data(existing, new)
    assert len(merged) == 2


def test_merge_games_data_handles_empty_existing():
    new = pd.DataFrame([
        {"date": "2025-01-03", "home_team": "GSW", "away_team": "DEN", "home_pts": 120, "away_pts": 118},
    ])
    merged = nba.merge_games_data(None, new)
    assert len(merged) == 1


# --- injury tier heuristic (scrapingPlayers._tier) ------------------------

@pytest.mark.parametrize("ws,mpg,expected", [
    (10.0, 30, 3),   # elite by win shares
    (4.0, 30, 2),    # starter by win shares
    (1.0, 30, 1),    # role player by win shares
    (None, 30, 2),   # no WS, heavy minutes -> starter
    (None, 10, 1),   # no WS, light minutes -> role
])
def test_tier_thresholds(ws, mpg, expected):
    assert _tier(ws, mpg) == expected
