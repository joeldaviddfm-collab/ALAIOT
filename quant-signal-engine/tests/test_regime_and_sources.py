"""Régimen (GMM expansivo sin look-ahead) y fuentes mockeadas (parsers)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.data.prices import normalize_ohlcv
from src.data.sentiment import lexicon_sentiment
from src.data.synthetic import synthetic_ohlcv
from src.features.regime import RegimeDetector


def test_regime_detector_no_nan_and_expanding(tiny_df):
    det = RegimeDetector(n_states=3, refit_days=30, min_train=60, seed=0)
    out = det.fit_predict(tiny_df)
    assert len(out) == len(tiny_df)
    assert out["regime_id"].isin([0, 1, 2]).all()
    assert out["regime_prob"].between(0, 1).all()
    # antes de min_train: régimen por defecto (1) y prob uniforme
    assert (out["regime_id"].iloc[:60] == 1).all()


def test_regime_no_lookahead(tiny_df):
    det = RegimeDetector(n_states=3, refit_days=30, min_train=60, seed=0)
    full = det.fit_predict(tiny_df)
    t = tiny_df.index[90]
    part = det.fit_predict(tiny_df.loc[:t])
    # el régimen en t calculado con datos <= t debe coincidir con el del dataset completo
    assert full.loc[t, "regime_id"] == part.loc[t, "regime_id"]


def test_normalize_ohlcv_handles_multiindex():
    idx = pd.bdate_range("2024-01-01", periods=3)
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["SPY"]])
    df = pd.DataFrame(np.arange(15, dtype=float).reshape(3, 5) + 100, index=idx, columns=cols)
    out = normalize_ohlcv(df, "test")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.name == "date"


def test_lexicon_sentiment():
    assert lexicon_sentiment("Profits surge to record high, upgrade to buy") > 0
    assert lexicon_sentiment("Market crash, losses and fear dump") < 0
    assert lexicon_sentiment("the cat sleeps") == 0.0


def test_synthetic_deterministic():
    a = synthetic_ohlcv("BTC-USD", "2023-01-01", "2023-03-01", seed=7)
    b = synthetic_ohlcv("BTC-USD", "2023-01-01", "2023-03-01", seed=7)
    pd.testing.assert_frame_equal(a, b)
    c = synthetic_ohlcv("BTC-USD", "2023-01-01", "2023-03-01", seed=8)
    assert not np.allclose(a["close"], c["close"])
