"""Notificaciones multi-canal: mensaje con señales + precisión honesta, fail-safe, sin claves en código."""

from __future__ import annotations

import json
import re

import pandas as pd
from src.signals.evaluator import write_live_accuracy_json
from src.utils.alerts import AlertBus
from src.utils.notify import (
    DISCLAIMER,
    build_signals_message,
    load_live_accuracy,
    notify_signals_full,
    send_discord,
)


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-11-01",
                "asset": "BTC-USD",
                "direction": "long",
                "prob": 0.83,
                "meta_prob": 0.71,
                "agreement": 1.0,
                "entry_ref": 100.0,
                "stop": 96.0,
                "tp": 112.0,
                "size": 12.5,
                "top1": "rsi_14",
                "top2": "adx_14",
                "top3": "vix_z",
                "reason": "ok",
                "data_mode": "real",
            },
            {
                "date": "2026-11-01",
                "asset": "SPY",
                "direction": "nada",
                "prob": 0.41,
                "meta_prob": 0.0,
                "agreement": 0.0,
                "entry_ref": 600.0,
                "stop": 0.0,
                "tp": 0.0,
                "size": 0.0,
                "top1": "",
                "top2": "",
                "top3": "",
                "reason": "umbral no alcanzado",
                "data_mode": "real",
            },
        ]
    )


def _acc() -> dict:
    return {
        "n_evaluated": 25,
        "accuracy": 0.72,
        "brier": 0.19,
        "mean_prob": 0.78,
        "data_mode": "real",
        "by_threshold": [{"bucket": "0.70-0.75", "n": 10, "accuracy": 0.7}],
        "recent_trades": [
            {
                "date": "2026-10-30",
                "asset": "BTC-USD",
                "direction": "long",
                "prob": 0.83,
                "realized_r": 1.5,
                "hit": 1,
            },
            {
                "date": "2026-10-31",
                "asset": "SPY",
                "direction": "short",
                "prob": 0.71,
                "realized_r": -0.4,
                "hit": 0,
            },
        ],
    }


def test_message_contains_signals_accuracy_and_disclaimer(cfg):
    alerts = AlertBus(cfg.reports_dir)
    msg = build_signals_message(cfg, _signals(), _acc(), alerts)
    assert "BTC-USD" in msg and "LONG" in msg and "0.83" in msg
    assert "72.0%" in msg and "25" in msg and "Brier" in msg
    assert "✅" in msg and "❌" in msg and "R=+1.50" in msg
    assert "🥇" in msg and "🟢" in msg  # rank 1 + banda verde (prob 0.83)
    assert "NADA" in msg
    assert DISCLAIMER in msg
    assert "asesoría financiera" in msg


def test_message_without_accuracy_is_honest(cfg):
    alerts = AlertBus(cfg.reports_dir)
    msg = build_signals_message(cfg, _signals(), None, alerts)
    assert "sin señales evaluables" in msg.lower()


def test_discord_noop_without_env(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert send_discord("DISCORD_WEBHOOK_URL", "hola") is False


def test_discord_posts_payload(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/x/y")
    captured = {}

    class Resp:
        ok = True
        status_code = 204
        text = ""

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return Resp()

    monkeypatch.setattr("requests.post", fake_post)
    assert send_discord("DISCORD_WEBHOOK_URL", "señales BTC-USD LONG 0.83") is True
    assert captured["url"].endswith("/x/y")
    assert "BTC-USD" in captured["json"]["content"]


def test_discord_exception_is_fail_safe(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/x/y")

    def boom(url, json=None, timeout=None):
        raise ConnectionError("sin red")

    monkeypatch.setattr("requests.post", boom)
    assert send_discord("DISCORD_WEBHOOK_URL", "hola") is False  # no lanza


def test_notify_signals_full_no_env_returns_false_and_never_raises(cfg, monkeypatch):
    for v in (
        "TELEGRAM_TOKEN",
        "TELEGRAM_CHAT_ID",
        "DISCORD_WEBHOOK_URL",
        "SMTP_HOST",
        "EMAIL_FROM",
        "EMAIL_TO",
    ):
        monkeypatch.delenv(v, raising=False)
    alerts = AlertBus(cfg.reports_dir)
    res = notify_signals_full(cfg, alerts, _signals(), _acc())
    assert set(res) == {"telegram", "discord", "email"}
    assert all(v is False for v in res.values())


def test_notify_signals_full_telegram_mock(cfg, monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("EMAIL_FROM", raising=False)
    monkeypatch.delenv("EMAIL_TO", raising=False)
    sent = {}

    def fake_send(token_env, chat_env, text, timeout=10.0):
        sent["text"] = text
        return True

    monkeypatch.setattr("src.utils.notify.send_telegram", fake_send)
    alerts = AlertBus(cfg.reports_dir)
    res = notify_signals_full(cfg, alerts, _signals(), _acc())
    assert res["telegram"] is True
    assert "BTC-USD" in sent["text"] and "72.0%" in sent["text"] and DISCLAIMER in sent["text"]


def test_write_live_accuracy_json_and_load(cfg, tmp_path):
    results = pd.DataFrame(
        {
            "date": ["2026-10-01", "2026-10-02"],
            "asset": ["BTC-USD", "SPY"],
            "direction": ["long", "short"],
            "prob": [0.8, 0.6],
            "realized_r": [1.5, -0.5],
            "hit": [1, 0],
        }
    )
    acc = pd.DataFrame({"bucket": ["0.70-0.75", ">=0.90"], "n": [1, 1], "accuracy": [0.0, 1.0]})
    p = write_live_accuracy_json(cfg, results, acc, "real")
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["n_evaluated"] == 2 and payload["accuracy"] == 0.5
    assert 0 <= payload["brier"] <= 1
    assert load_live_accuracy(cfg.reports_dir) == payload
    # vacío => n_evaluated 0
    p2 = write_live_accuracy_json(cfg, pd.DataFrame(columns=results.columns), pd.DataFrame(), "synthetic")
    assert json.loads(p2.read_text(encoding="utf-8"))["n_evaluated"] == 0


def test_no_secrets_in_code():
    """Ninguna clave real hardcodeada en el código fuente."""
    import pathlib

    for f in pathlib.Path("src").rglob("*.py"):
        t = f.read_text(encoding="utf-8")
        assert "api.telegram.org/bot" not in t or "bot{token}" in t
        assert not re.search(r"[0-9]{9,}:[A-Za-z0-9_-]{30,}", t), f"posible token telegram en {f}"
