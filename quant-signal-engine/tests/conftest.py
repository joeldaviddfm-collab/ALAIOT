"""Fixtures compartidos: config rápida con datos sintéticos pequeños."""

from __future__ import annotations

import pandas as pd
import pytest
from src.config import Config
from src.data.pipeline import MarketData
from src.data.synthetic import synthetic_global, synthetic_ohlcv

FAST_OVERRIDES = {
    "data": {"start": "2022-01-01"},
    "model": {
        "walk_forward": {"train_days": 120, "embargo_days": 11, "test_days": 30, "blind_days": 60},
        "lgbm": {"n_estimators": 40, "max_depth": 3, "num_leaves": 7, "min_child_samples": 10},
        "threshold": 0.55,
        "meta_min": 0.5,
    },
    "robustness": {"n_permutations": 50, "n_random_strategies": 50, "mc_sims": 100},
}


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """En tests no esperamos los backoffs reales: un solo intento, sin dormir."""
    import src.data.base as base_mod

    def _no_wait(fn, *, attempts=3, base_wait=2.0, max_wait=30.0, what="call"):
        try:
            return fn()
        except Exception:
            return None

    monkeypatch.setattr(base_mod, "with_retry", _no_wait)


@pytest.fixture()
def cfg(tmp_path, monkeypatch) -> Config:
    return Config(overrides=FAST_OVERRIDES)


@pytest.fixture()
def small_market(cfg) -> MarketData:
    assets = ["BTC-USD", "SPY"]
    prices = {a: synthetic_ohlcv(a, "2022-01-01", "2023-06-30", cfg.seed) for a in assets}
    return MarketData(
        prices=prices, global_df=synthetic_global("2022-01-01", "2023-06-30", cfg.seed), data_mode="synthetic"
    )


@pytest.fixture()
def tiny_df() -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-01", periods=120)
    import numpy as np

    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(idx))))
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, len(idx))),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e6 * (1 + rng.random(len(idx))),
        },
        index=idx,
    )
    df.index.name = "date"
    return df
