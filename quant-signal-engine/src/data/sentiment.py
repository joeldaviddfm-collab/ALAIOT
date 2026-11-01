"""Sentimiento: Crypto Fear & Greed Index (alternative.me), Google Trends (pytrends)
y titulares RSS puntuados con FinBERT (si está instalado) o lexicón financiero propio.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests
import structlog

from src.data.base import safe_call
from src.data.cache import DiskCache
from src.utils.alerts import AlertBus

log = structlog.get_logger("qse.sentiment")

NEWS_FEEDS = [
    "https://finance.yahoo.com/rss/topstories",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
]

# Lexicón financiero mínimo (fallback cuando transformers/FinBERT no está instalado)
_POS = {
    "beat",
    "beats",
    "surge",
    "surges",
    "rally",
    "gain",
    "gains",
    "up",
    "upgrade",
    "upgraded",
    "record",
    "profit",
    "profits",
    "growth",
    "strong",
    "bullish",
    "outperform",
    "buy",
    "adoption",
    "approve",
    "approved",
    "etf",
    "inflow",
    "inflows",
    "boost",
    "soar",
    "soars",
    "rise",
    "rises",
}
_NEG = {
    "miss",
    "misses",
    "crash",
    "plunge",
    "plunges",
    "drop",
    "drops",
    "fall",
    "falls",
    "down",
    "downgrade",
    "downgraded",
    "loss",
    "losses",
    "weak",
    "bearish",
    "sell",
    "fraud",
    "hack",
    "ban",
    "banned",
    "lawsuit",
    "default",
    "bankruptcy",
    "outflow",
    "outflows",
    "fear",
    "dump",
    "dumps",
    "liquidation",
    "liquidations",
    "recession",
    "hike",
    "crisis",
}

_finbert_pipe = None
_finbert_tried = False


def finbert_pipeline():
    """Carga perezosa de FinBERT (ProsusAI/finbert). Devuelve None si no está disponible."""
    global _finbert_pipe, _finbert_tried
    if _finbert_tried:
        return _finbert_pipe
    _finbert_tried = True
    try:
        from transformers import pipeline

        _finbert_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", top_k=None)
        log.info("finbert_loaded")
    except Exception as exc:  # noqa: BLE001 - fallback a lexicón
        log.warning("finbert_unavailable", error=repr(exc), fallback="lexicon")
        _finbert_pipe = None
    return _finbert_pipe


def lexicon_sentiment(text: str) -> float:
    words = [w.strip(".,:;!?()[]\"'").lower() for w in text.split()]
    pos = sum(1 for w in words if w in _POS)
    neg = sum(1 for w in words if w in _NEG)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def score_text(text: str) -> float:
    pipe = finbert_pipeline()
    if pipe is None:
        return lexicon_sentiment(text)
    try:
        res = pipe(text[:512])
        scores = {r["label"].lower(): r["score"] for r in res[0]}
        return scores.get("positive", 0.0) - scores.get("negative", 0.0)
    except Exception:  # noqa: BLE001
        return lexicon_sentiment(text)


def fear_greed(alerts: AlertBus) -> pd.Series | None:
    def _fetch() -> pd.Series:
        params: dict[str, str | int] = {"limit": 0, "format": "json"}
        r = requests.get("https://api.alternative.me/fng/", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()["data"]
        s = pd.Series(
            {
                pd.Timestamp(dt.datetime.utcfromtimestamp(int(d["timestamp"])).date()): float(d["value"])
                for d in data
            }
        )
        s.index = pd.DatetimeIndex(s.index)
        return s.sort_index()

    s = safe_call("fear_greed", _fetch, alerts)
    return s


def google_trends(alerts: AlertBus, keyword: str = "bitcoin") -> pd.Series | None:
    def _fetch() -> pd.Series:
        from pytrends.request import TrendReq

        pytrend = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pytrend.build_payload([keyword], timeframe="today 5-y")
        df = pytrend.interest_over_time()
        if df.empty:
            raise ValueError("pytrends vacío")
        s = df[keyword]
        s.index = pd.DatetimeIndex(s.index).normalize()
        return s.resample("B").ffill()

    return safe_call("pytrends", _fetch, alerts)


def news_sentiment(alerts: AlertBus) -> pd.Series | None:
    import feedparser

    def _fetch() -> pd.Series:
        scores: list[tuple[pd.Timestamp, float]] = []
        for url in NEWS_FEEDS:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                title = entry.get("title", "")
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub is None:
                    continue
                ts = pd.Timestamp(dt.datetime(*pub[:6]).date())
                scores.append((ts, score_text(title)))
        if not scores:
            raise ValueError("sin noticias")
        s = pd.Series([v for _, v in scores], index=pd.DatetimeIndex([t for t, _ in scores]))
        return s.groupby(s.index).mean().sort_index().resample("B").mean()

    return safe_call("news_rss", _fetch, alerts)


def get_sentiment(start: str, end: str | None, cache: DiskCache, alerts: AlertBus) -> pd.DataFrame:
    key = ("sentiment", str(start), str(end))
    cached = cache.get(key, max_age_days=1.0)
    if cached is not None and not cached.empty:
        return cached
    frames: dict[str, pd.Series] = {}
    fg = fear_greed(alerts)
    if fg is not None:
        frames["fear_greed"] = fg
    tr = google_trends(alerts)
    if tr is not None:
        frames["trends_btc"] = tr
    news = news_sentiment(alerts)
    if news is not None:
        frames["news_sent"] = news
    if not frames:
        alerts.add("ERROR", "Sentimiento no disponible en ninguna fuente")
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index.name = "date"
    df = df.resample("B").ffill()
    cache.put(key, df)
    log.info("sentiment_ok", cols=list(df.columns), rows=len(df))
    return df
