"""Cache local en disco (data/cache, gitignored) para no repetir descargas."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd


class DiskCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: tuple[str, ...]) -> Path:
        return self.root.joinpath("__".join(key)).with_suffix(".csv")

    def get(self, key: tuple[str, ...], max_age_days: float | None = None) -> pd.DataFrame | None:
        path = self._path(key)
        if not path.exists():
            return None
        if max_age_days is not None:
            age_days = (time.time() - path.stat().st_mtime) / 86400.0
            if age_days > max_age_days:
                return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.DatetimeIndex(df.index)
            df.index.name = "date"
            return df
        except Exception:
            return None

    def put(self, key: tuple[str, ...], df: pd.DataFrame) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_csv(tmp)
        tmp.replace(path)  # escritura atómica

    def age_days(self, key: tuple[str, ...]) -> float | None:
        path = self._path(key)
        if not path.exists():
            return None
        return (dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 86400.0
