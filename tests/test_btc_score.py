import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock streamlit before importing btc_tab (no streamlit installed in test env)
from unittest.mock import MagicMock, patch
sys.modules["streamlit"] = MagicMock()

from btc_tab import compute_btc_score, _halving_cycle_score


def make_df(rsi_val, close, bb_lower, bb_upper):
    """Helper : crée un DataFrame minimal pour compute_btc_score."""
    n = 60
    df = pd.DataFrame({
        "Close":    [close]    * n,
        "RSI":      [rsi_val]  * n,
        "BB_lower": [bb_lower] * n,
        "BB_upper": [bb_upper] * n,
        "BB_mid":   [(bb_lower + bb_upper) / 2] * n,
        "SMA20":    [close]    * n,
        "SMA50":    [close]    * n,
    })
    return df


def test_extreme_fear_low_rsi_gives_high_score():
    """Extreme Fear + RSI oversold + prix sous bande basse → score élevé (≥60)."""
    df = make_df(rsi_val=25, close=80_000, bb_lower=85_000, bb_upper=100_000)
    score, label, _ = compute_btc_score(df, fear_greed=15)
    assert score >= 60, f"Score attendu ≥60, obtenu {score}"


def test_extreme_greed_high_rsi_gives_low_score():
    """Extreme Greed + RSI surachat + prix au-dessus bande haute → score faible (≤40)."""
    df = make_df(rsi_val=78, close=100_000, bb_lower=80_000, bb_upper=95_000)
    score, label, _ = compute_btc_score(df, fear_greed=85)
    assert score <= 40, f"Score attendu ≤40, obtenu {score}"


def test_neutral_conditions_give_neutral_score():
    """Conditions neutres → pas de signal fort dans un sens ou l'autre."""
    df = make_df(rsi_val=50, close=90_000, bb_lower=85_000, bb_upper=95_000)
    score, label, _ = compute_btc_score(df, fear_greed=50)
    assert score < 60, f"Score attendu <60 pour conditions neutres, obtenu {score}"
    assert label not in ("ACHAT FORT", "ÉVITER"), f"Label inattendu: {label}"


def test_label_strong_buy():
    """Score ≥75 → label 'ACHAT FORT'."""
    df = make_df(rsi_val=20, close=75_000, bb_lower=85_000, bb_upper=100_000)
    score, label, _ = compute_btc_score(df, fear_greed=10)
    assert label == "ACHAT FORT", f"Label attendu 'ACHAT FORT', obtenu '{label}' (score={score})"


def test_label_avoid():
    """Score ≤24 → label 'ÉVITER'."""
    df = make_df(rsi_val=82, close=105_000, bb_lower=80_000, bb_upper=100_000)
    score, label, _ = compute_btc_score(df, fear_greed=90)
    assert label == "ÉVITER", f"Label attendu 'ÉVITER', obtenu '{label}' (score={score})"


def test_score_clamped_0_100():
    """Score toujours entre 0 et 100."""
    df = make_df(rsi_val=5, close=50_000, bb_lower=90_000, bb_upper=100_000)
    score, _, _ = compute_btc_score(df, fear_greed=5)
    assert 0 <= score <= 100

    df2 = make_df(rsi_val=95, close=120_000, bb_lower=80_000, bb_upper=100_000)
    score2, _, _ = compute_btc_score(df2, fear_greed=95)
    assert 0 <= score2 <= 100


def test_missing_fear_greed_still_works():
    """Si fear_greed=None, la fonction ne plante pas et retourne un score valide."""
    df = make_df(rsi_val=50, close=90_000, bb_lower=85_000, bb_upper=95_000)
    score, label, reasons = compute_btc_score(df, fear_greed=None)
    assert 0 <= score <= 100
    assert isinstance(label, str)
    assert isinstance(reasons, list)


def test_halving_cycle_returns_tuple():
    """_halving_cycle_score retourne un tuple (int, str)."""
    pts, msg = _halving_cycle_score()
    assert isinstance(pts, int)
    assert isinstance(msg, str)
    assert 0 <= pts <= 20
