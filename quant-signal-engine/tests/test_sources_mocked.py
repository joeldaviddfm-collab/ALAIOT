"""Fuentes de datos con mocks (sin red): parsers y lógica de fallback/caché de cada fuente."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import src.data.derivatives as der_mod
import src.data.macro as macro_mod
import src.data.onchain as onc_mod
import src.data.prices as prices_mod
import src.data.sentiment as sent_mod
from src.data.base import DataUnavailableError
from src.data.cache import DiskCache
from src.data.pipeline import load_market_data
from src.utils.alerts import AlertBus


def _fake_yf_df(asset="SPY", scale=1.0):
    # fechas recientes (hasta hoy) para que la validación de "stale" no rechace el mock
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=30)
    close = 100 * scale * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 30)))
    df = pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6}, index=idx
    )
    df.columns = pd.MultiIndex.from_product([df.columns, [asset]])
    return df


def test_get_prices_yfinance_and_cache(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr("yfinance.download", lambda *a, **k: _fake_yf_df())
    cache = DiskCache(tmp_path / "cache")
    alerts = AlertBus(tmp_path)
    df = prices_mod.get_prices("SPY", "2023-01-01", None, cache, alerts, crypto_list=[])
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 30
    # segunda llamada sale del cache
    df2 = prices_mod.get_prices(
        "SPY", "2023-01-01", None, cache, alerts, crypto_list=[], cache_max_age_days=10
    )
    pd.testing.assert_frame_equal(df, df2, check_freq=False)


class _FakeCCXT:
    name = "ccxt:binance"

    def __init__(self):
        pass

    def fetch(self, asset, start, end):
        return prices_mod.normalize_ohlcv(_fake_yf_df("BTC-USD", scale=1.0), self.name)


def test_get_prices_crypto_fallback_and_cross_block(cfg, monkeypatch, tmp_path):
    # yfinance devuelve precios 50% más altos que ccxt => el cruce debe bloquear el activo
    monkeypatch.setattr(prices_mod, "CCXTSource", _FakeCCXT)
    monkeypatch.setattr("yfinance.download", lambda tickers, *a, **k: _fake_yf_df(tickers, scale=1.5))
    cache = DiskCache(tmp_path / "cache")
    alerts = AlertBus(tmp_path)
    with pytest.raises(DataUnavailableError):
        prices_mod.get_prices(
            "BTC-USD", "2023-01-01", None, cache, alerts, crypto_list=["BTC-USD"], cross_tol=0.02
        )


def test_get_prices_crypto_cross_ok(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(prices_mod, "CCXTSource", _FakeCCXT)
    monkeypatch.setattr("yfinance.download", lambda tickers, *a, **k: _fake_yf_df(tickers, scale=1.001))
    cache = DiskCache(tmp_path / "cache")
    alerts = AlertBus(tmp_path)
    df = prices_mod.get_prices(
        "BTC-USD", "2023-01-01", None, cache, alerts, crypto_list=["BTC-USD"], cross_tol=0.02
    )
    assert len(df) == 30 and (df["close"] > 0).all()


def test_get_prices_all_fail(cfg, monkeypatch, tmp_path):
    def boom(self, a, s, e):
        raise ConnectionError("down")

    monkeypatch.setattr(prices_mod.YFinanceSource, "fetch", boom)
    cache = DiskCache(tmp_path / "cache")
    alerts = AlertBus(tmp_path)
    with pytest.raises(DataUnavailableError):
        prices_mod.get_prices("SPY", "2023-01-01", None, cache, alerts, crypto_list=[])


def test_macro_fred_and_yf_fallback(cfg, monkeypatch, tmp_path):
    class FakeFred:
        def __init__(self, api_key=None):
            pass

        def get_series(self, ticker, observation_start=None):
            idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=30)
            return pd.Series(np.linspace(4.0, 4.5, 30), index=idx)

    monkeypatch.setattr("fredapi.Fred", FakeFred)
    monkeypatch.setenv("FRED_API_KEY", "fake-key-for-test")
    monkeypatch.setattr("yfinance.download", lambda tickers, *a, **k: _fake_yf_df(tickers))
    cache = DiskCache(tmp_path / "cache")
    alerts = AlertBus(tmp_path)
    df = macro_mod.get_macro("2023-01-01", None, cache, alerts)
    assert "dgs10" in df.columns and "vix" in df.columns and "cpi_yoy" in df.columns


def test_macro_no_key_uses_yfinance(cfg, monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr("yfinance.download", lambda tickers, *a, **k: _fake_yf_df(tickers))
    cache = DiskCache(tmp_path / "cache")
    alerts = AlertBus(tmp_path)
    df = macro_mod.get_macro("2023-01-01", None, cache, alerts)
    assert "vix" in df.columns and "dxy" in df.columns


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.ok = True
        self.status_code = 200
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_sentiment_fear_greed_and_news(cfg, monkeypatch, tmp_path):
    fng = {
        "data": [
            {"timestamp": str(int(pd.Timestamp("2023-11-14").timestamp())), "value": "55"},
            {"timestamp": str(int(pd.Timestamp("2023-11-15").timestamp())), "value": "60"},
        ]
    }
    monkeypatch.setattr(sent_mod.requests, "get", lambda *a, **k: _FakeResp(fng))
    trends = pd.Series([50.0, 55.0], index=pd.DatetimeIndex(["2023-11-14", "2023-11-15"]))
    news = pd.Series([0.2, -0.1], index=pd.DatetimeIndex(["2023-11-14", "2023-11-15"]))
    monkeypatch.setattr(sent_mod, "google_trends", lambda alerts, keyword="bitcoin": trends)
    monkeypatch.setattr(sent_mod, "news_sentiment", lambda alerts: news)
    cache = DiskCache(tmp_path / "cache")
    alerts = AlertBus(tmp_path)
    df = sent_mod.get_sentiment("2023-11-01", None, cache, alerts)
    assert {"fear_greed", "trends_btc", "news_sent"}.issubset(df.columns)
    # todo caído => DataFrame vacío + alerta, sin excepción
    monkeypatch.setattr(sent_mod, "fear_greed", lambda alerts: None)
    monkeypatch.setattr(sent_mod, "google_trends", lambda alerts, keyword="bitcoin": None)
    monkeypatch.setattr(sent_mod, "news_sentiment", lambda alerts: None)
    df2 = sent_mod.get_sentiment("2023-11-01", None, DiskCache(tmp_path / "c2"), alerts)
    assert isinstance(df2, pd.DataFrame) and df2.empty


def test_derivatives_parsing(cfg, monkeypatch, tmp_path):
    def fake_get(url, params=None, timeout=None, headers=None):
        if "fundingRate" in url:
            return _FakeResp([{"fundingTime": 1700000000000, "fundingRate": "0.0001"}])
        if "openInterestHist" in url:
            return _FakeResp([{"timestamp": 1700000000000, "sumOpenInterestValue": "1e10"}])
        return _FakeResp([{"timestamp": 1700000000000, "longShortRatio": "1.2"}])

    monkeypatch.setattr(der_mod.requests, "get", fake_get)
    cache = DiskCache(tmp_path / "cache")
    alerts = AlertBus(tmp_path)
    df = der_mod.get_derivatives("2023-11-01", None, cache, alerts)
    assert {"funding_btc", "oi_btc", "ls_ratio_btc"}.issubset(df.columns)


def test_onchain_parsing(cfg, monkeypatch, tmp_path):
    def fake_get(url, params=None, timeout=None, headers=None):
        if "coingecko" in url:
            return _FakeResp(
                {
                    "prices": [[1700000000000, 30000.0]],
                    "market_caps": [[1700000000000, 6e11]],
                    "total_volumes": [[1700000000000, 1e10]],
                }
            )
        return _FakeResp({"values": [{"x": 1700000000, "y": 3e8}]})

    monkeypatch.setattr(onc_mod.requests, "get", fake_get)
    monkeypatch.delenv("GLASSNODE_API_KEY", raising=False)
    cache = DiskCache(tmp_path / "cache")
    alerts = AlertBus(tmp_path)
    df = onc_mod.get_onchain("2023-11-01", None, cache, alerts)
    assert "btc_price_cg" in df.columns and "hash_rate" in df.columns


def test_load_market_data_offline_synthetic(cfg):
    alerts = AlertBus(cfg.reports_dir)
    market = load_market_data(cfg, ["BTC-USD", "SPY"], offline=True, alerts=alerts)
    assert market.data_mode == "synthetic"
    assert set(market.prices) == {"BTC-USD", "SPY"}
    assert not market.global_df.empty
