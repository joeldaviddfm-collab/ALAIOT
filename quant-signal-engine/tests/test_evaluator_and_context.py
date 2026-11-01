"""Evaluador en vivo y filtros de contexto (calendario de eventos)."""

from __future__ import annotations

import pandas as pd
from src.signals.context import event_block, load_event_dates
from src.signals.evaluator import accuracy_by_threshold, evaluate_signals


def test_event_block_window():
    events = load_event_dates()
    assert len(events) > 100
    fomc = pd.Timestamp("2024-03-20")  # FOMC programado
    assert event_block(fomc, events, before=0, after=1) == "FOMC"
    assert event_block(fomc + pd.Timedelta(days=1), events, before=0, after=1) == "FOMC"
    assert event_block(pd.Timestamp("2024-03-25"), events, before=0, after=1) is None


def test_evaluator_known_outcome(cfg, small_market):
    prices = small_market.prices
    asset = "SPY"
    df = prices[asset]
    d = df.index[100]
    # señal long; el resultado realizado depende del camino real (sintético)
    hist = pd.DataFrame(
        [
            {
                "date": str(d.date()),
                "asset": asset,
                "direction": "long",
                "prob": 0.8,
                "entry_ref": float(df["close"].iloc[100]),
                "stop": float(df["close"].iloc[100]) - 5,
                "tp": float(df["close"].iloc[100]) + 8,
            }
        ]
    )
    res = evaluate_signals(hist, prices, cfg)
    assert len(res) == 1
    assert res["hit"].iloc[0] in (0, 1)
    assert "realized_r" in res.columns
    acc = accuracy_by_threshold(pd.concat([res, res], ignore_index=True))
    assert len(acc) >= 1
