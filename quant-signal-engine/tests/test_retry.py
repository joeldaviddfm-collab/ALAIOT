"""with_retry: reintenta con backoff y, al agotar intentos, devuelve None (fail-safe)."""

from __future__ import annotations

from src.utils.retry import with_retry


def test_retry_eventually_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("fallo temporal")
        return "ok"

    monkeypatch.setattr("src.utils.retry.time.sleep", lambda s: None)
    assert with_retry(flaky, attempts=3, what="t") == "ok"
    assert calls["n"] == 3


def test_retry_exhausted_returns_none(monkeypatch):
    monkeypatch.setattr("src.utils.retry.time.sleep", lambda s: None)

    def always_fail():
        raise ConnectionError("caído")

    assert with_retry(always_fail, attempts=2, what="t") is None
