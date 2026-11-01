"""TEST OBLIGATORIO: métricas contra casos conocidos (valores calculados a mano)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.validation.metrics import deflated_sharpe, equity_metrics, pbo_cscv


def test_equity_metrics_known_case():
    # equity con retornos diarios +1%/-1% alternados
    rets = np.array([0.01, -0.01] * 50)
    equity = pd.Series(100000 * np.cumprod(1 + rets))
    m = equity_metrics(equity)
    assert m["n_days"] == 100
    assert abs(m["total_return"] - (equity.iloc[-1] / equity.iloc[0] - 1)) < 1e-12
    # serie simétrica: sharpe cercano a 0
    assert abs(m["sharpe"]) < 0.2
    # max drawdown debe coincidir exactamente con el cálculo directo sobre la curva
    expected_dd = float((equity / equity.cummax() - 1).min())
    assert m["max_drawdown"] == expected_dd
    assert -0.05 < m["max_drawdown"] < 0


def test_equity_metrics_uptrend():
    rets = np.full(100, 0.001)
    equity = pd.Series(100000 * np.cumprod(1 + rets))
    m = equity_metrics(equity)
    assert m["sharpe"] > 5  # tendencia diaria constante => sharpe alto (std≈0)
    assert m["max_drawdown"] == 0.0
    assert m["total_return"] > 0.09


def test_trade_metrics_known_case():
    from src.backtester.engine import Trade

    t = pd.Timestamp("2024-01-01")
    trades = [
        Trade("A", t, t, 100, t, 110, 1, 1, 90, 130, 10.0, 1.0, 0.1, "tp", 0.8),
        Trade("A", t, t, 100, t, 95, 1, 1, 90, 130, -5.0, -0.5, 0.1, "stop", 0.8),
        Trade("A", t, t, 100, t, 112, 1, 1, 90, 130, 12.0, 1.2, 0.1, "tp", 0.8),
    ]
    equity = pd.Series([100000, 100010, 100005, 100017], index=pd.bdate_range("2024-01-01", periods=4))
    m = equity_metrics(equity, trades)
    assert m["n_trades"] == 3
    assert abs(m["win_rate"] - 2 / 3) < 1e-9
    assert abs(m["profit_factor"] - (10 + 12) / 5) < 1e-9
    assert abs(m["expectancy"] - (10 - 5 + 12) / 3) < 1e-9


def test_deflated_sharpe_null_hypothesis():
    dsr, p = deflated_sharpe(0.0, skew=0.0, kurt=3.0, T=500, n_trials=1)
    assert dsr == 0.0 and abs(p - 0.5) < 1e-6
    # sharpe moderado, pocas configs => p bajo
    _, p2 = deflated_sharpe(0.3, skew=0.0, kurt=3.0, T=500, n_trials=1)
    assert p2 < 0.05
    # muchas configuraciones probadas => p sube (corrección por data snooping)
    _, p3 = deflated_sharpe(0.3, skew=0.0, kurt=3.0, T=500, n_trials=1000)
    assert p3 > p2


def test_pbo_cscv_bounds():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(1)
    good = pd.Series(0.001 + rng.normal(0, 0.005, 400), index=idx)  # estrategia con edge
    noise = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    pbo = pbo_cscv({"good": good, "noise": noise}, n_segments=8)
    assert 0.0 <= pbo <= 1.0
