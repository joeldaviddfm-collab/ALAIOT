"""Walk-forward (purging/embargo + test ciego) y pruebas anti-suerte/anti-overfitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.validation import robustness as R
from src.validation.walkforward import blind_split, wf_splits


def test_wf_splits_embargo_purging():
    dates = pd.bdate_range("2020-01-01", periods=400)
    splits = wf_splits(dates, train_days=100, embargo_days=10, test_days=30)
    assert len(splits) > 0
    for tr, te in splits:
        # purging: el train termina `embargo_days` antes del test
        assert tr.max() < te.min() - 0 or True  # posiciones, no fechas
        # en posiciones: tr.max() + embargo <= te.min()
        assert tr.max() + 10 <= te.min()
        assert len(te) == 30
    # los test son contiguos y no se solapan
    te_all = np.concatenate([te for _, te in splits])
    assert len(te_all) == len(np.unique(te_all))


def test_blind_split_reserves_tail():
    dates = pd.bdate_range("2020-01-01", periods=300)
    dev, blind = blind_split(dates, 60)
    assert len(blind) == 60
    assert dev.max() < blind.min()
    assert len(dev) + len(blind) == 300


def test_permutation_pvalue_low_for_good_strategy():
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.02, 300)  # media positiva clara
    p = R.permutation_pvalue(r, observed_sharpe=3.0, n=200, seed=1)
    assert p < 0.05
    p2 = R.permutation_pvalue(rng.normal(0, 0.02, 300), observed_sharpe=0.0, n=200, seed=1)
    assert p2 > 0.05


def test_random_strategies_pvalue(small_market):
    prices = small_market.prices
    dates = sorted(prices["SPY"].index)
    sig_points = [(dates[100 + i], "SPY" if i % 2 == 0 else "BTC-USD") for i in range(30)]
    p = R.random_strategies_pvalue(prices, sig_points, observed_sharpe=8.0, n=100, seed=2)
    assert p <= 0.15  # un edge fuerte apenas aparece entre estrategias aleatorias
    p2 = R.random_strategies_pvalue(prices, sig_points, observed_sharpe=-8.0, n=100, seed=2)
    assert p2 > 0.5


def test_monte_carlo_outputs():
    rng = np.random.default_rng(3)
    r = rng.normal(0.005, 0.02, 200)
    mc = R.monte_carlo_block_bootstrap(r, block=5, n_sims=200, ruin_threshold=0.5, seed=3)
    assert 0.0 <= mc["p_ruin"] <= 1.0
    assert mc["ret_p05"] <= mc["ret_p50"] <= mc["ret_p95"]
    assert mc["n_sims"] == 200


def test_historical_stress_windows():
    idx = pd.bdate_range("2020-01-01", periods=300)
    equity = pd.Series(
        100000 * np.exp(np.cumsum(np.where(idx < pd.Timestamp("2020-03-15"), -0.005, 0.002))), index=idx
    )
    out = R.historical_stress(equity, [{"name": "COVID 2020", "start": "2020-02-15", "end": "2020-04-30"}])
    assert "COVID 2020" in out and out["COVID 2020"] < 0


def test_leakage_auc_returns_float(cfg, small_market):
    from src.training import build_training_set

    X, y, _ = build_training_set(cfg, small_market)
    cols = [c for c in X.columns if c not in ("date", "asset", "close_ref", "atr14")]
    auc = R.leakage_auc(X[cols], y, seed=cfg.seed, n_splits=3)
    assert 0.0 <= auc <= 1.0
