"""TEST OBLIGATORIO: consistencia backtest/paper — el mismo motor y la misma lógica de
"ejecución en la apertura de la barra siguiente" se usan en ambos modos."""

from __future__ import annotations

import pandas as pd
from src.backtester.engine import next_trading_date, run_backtest


def test_entry_is_next_bar_open(cfg, small_market):
    prices = small_market.prices
    asset = "SPY"
    df = prices[asset]
    sig_date = df.index[100]
    exec_date = next_trading_date(df.index, sig_date)
    assert exec_date == df.index[101]
    sig = pd.DataFrame(
        [
            {
                "date": sig_date,
                "asset": asset,
                "direction": 1,
                "prob": 0.8,
                "stop": float(df["close"].iloc[100]) - 5.0,
                "tp": float(df["close"].iloc[100]) + 8.0,
            }
        ]
    )
    res = run_backtest(prices, sig, cfg)
    assert len(res.trades) == 1
    trade = res.trades[0]
    # la entrada debe ser la apertura de la barra siguiente, no el close de la señal
    assert trade.entry_date == exec_date
    assert (
        abs(trade.entry_price - float(df.loc[exec_date, "open"]) * (1 + 0.0003)) < 1e-6
        or trade.entry_price <= float(df.loc[exec_date, "open"]) * 1.001
    )


def test_same_signals_same_trades(cfg, small_market):
    """Correr el motor dos veces con las mismas señales produce exactamente los mismos trades."""
    prices = small_market.prices
    sig = pd.DataFrame(
        [
            {
                "date": prices["SPY"].index[100],
                "asset": "SPY",
                "direction": 1,
                "prob": 0.8,
                "stop": 90.0,
                "tp": 120.0,
            },
            {
                "date": prices["BTC-USD"].index[100],
                "asset": "BTC-USD",
                "direction": -1,
                "prob": 0.75,
                "stop": 200.0,
                "tp": 50.0,
            },
        ]
    )
    r1 = run_backtest(prices, sig, cfg)
    r2 = run_backtest(prices, sig, cfg)
    assert len(r1.trades) == len(r2.trades)
    for a, b in zip(r1.trades, r2.trades, strict=False):
        assert a.entry_price == b.entry_price and a.exit_price == b.exit_price and a.pnl == b.pnl
    pd.testing.assert_series_equal(r1.equity, r2.equity)


def test_costs_reduce_pnl(cfg, small_market):
    prices = small_market.prices
    sig = pd.DataFrame(
        [
            {
                "date": prices["SPY"].index[100],
                "asset": "SPY",
                "direction": 1,
                "prob": 0.8,
                "stop": 90.0,
                "tp": 120.0,
            }
        ]
    )
    r0 = run_backtest(prices, sig, cfg, cost_mult=1.0)
    r3 = run_backtest(prices, sig, cfg, cost_mult=3.0)
    assert r3.trades[0].pnl <= r0.trades[0].pnl + 1e-9
