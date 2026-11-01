"""TEST OBLIGATORIO: datos corruptos. El validador debe detectar velas inválidas,
duplicados, no-monotonía, precios <= 0, datos stale; y el cruce entre fuentes bloquea
discrepancias grandes."""

from __future__ import annotations

import pandas as pd
from src.data.validator import cross_source_check, validate_ohlcv


def test_detects_high_low_inversion(tiny_df):
    bad = tiny_df.copy()
    bad.loc[bad.index[10], "high"] = bad.loc[bad.index[10], "low"] - 1.0
    rep = validate_ohlcv(bad)
    assert not rep.ok
    assert any("inválidas" in i for i in rep.issues)


def test_detects_negative_price(tiny_df):
    bad = tiny_df.copy()
    bad.loc[bad.index[5], "close"] = -10.0
    rep = validate_ohlcv(bad)
    assert not rep.ok


def test_detects_duplicates_and_non_monotonic(tiny_df):
    bad = pd.concat([tiny_df, tiny_df.iloc[[3]]]).sort_index()
    rep = validate_ohlcv(bad)
    assert not rep.ok and any("duplicados" in i for i in rep.issues)
    bad2 = tiny_df.iloc[::-1]
    rep2 = validate_ohlcv(bad2)
    assert not rep2.ok and any("monotónicos" in i for i in rep2.issues)


def test_detects_stale(tiny_df):
    old = tiny_df.copy()
    old.index = old.index - pd.Timedelta(days=365)
    rep = validate_ohlcv(old, max_stale_bdays=5, today=pd.Timestamp("2026-11-01"))
    assert rep.stale and not rep.ok


def test_detects_jump_warning(tiny_df):
    bad = tiny_df.copy()
    bad.loc[bad.index[50], "close"] = bad.loc[bad.index[49], "close"] * 5.0
    rep = validate_ohlcv(bad, jump_sigma=6.0)
    assert any("saltos anómalos" in w for w in rep.warnings)


def test_cross_source_check_blocks_discrepancy(tiny_df):
    other = tiny_df.copy()
    other["close"] = other["close"] * 1.10  # 10% de diferencia
    ok, diff = cross_source_check(tiny_df, other, tol=0.02)
    assert not ok and diff > 0.02
    ok2, diff2 = cross_source_check(tiny_df, tiny_df.copy(), tol=0.02)
    assert ok2 and diff2 == 0.0


def test_empty_df_invalid():
    rep = validate_ohlcv(pd.DataFrame())
    assert not rep.ok and rep.stale
