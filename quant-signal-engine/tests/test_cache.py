"""Cache local: roundtrip put/get y caducidad por max_age."""

from __future__ import annotations

import os
import time

import pandas as pd
from src.data.cache import DiskCache


def test_cache_roundtrip(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    df.index.name = "date"
    cache.put(("prices", "equity", "SPY"), df)
    got = cache.get(("prices", "equity", "SPY"))
    pd.testing.assert_frame_equal(got, df, check_freq=False)


def test_cache_expiry(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    df = pd.DataFrame({"close": [1.0]}, index=pd.bdate_range("2024-01-01", periods=1))
    df.index.name = "date"
    cache.put(("k",), df)
    assert cache.get(("k",), max_age_days=10) is not None
    # simular archivo viejo
    p = tmp_path / "cache" / "k.csv"
    old = time.time() - 3 * 86400
    os.utime(p, (old, old))
    assert cache.get(("k",), max_age_days=1) is None
    assert cache.age_days(("k",)) is not None
