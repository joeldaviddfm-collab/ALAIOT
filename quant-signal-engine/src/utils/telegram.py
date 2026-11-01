"""Notificaciones opcionales por Telegram. Sin token/chat id en env => no-op."""

from __future__ import annotations

import os

import requests
import structlog

from src.utils.alerts import AlertBus

log = structlog.get_logger("qse.telegram")


def send_telegram(token_env: str, chat_id_env: str, text: str, timeout: float = 10.0) -> bool:
    token = os.getenv(token_env, "").strip()
    chat_id = os.getenv(chat_id_env, "").strip()
    if not token or not chat_id:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
            timeout=timeout,
        )
        ok = resp.ok
        if not ok:
            log.warning("telegram_failed", status=resp.status_code, body=resp.text[:200])
        return ok
    except Exception as exc:  # noqa: BLE001 - la notificación nunca debe romper el pipeline
        log.warning("telegram_exception", error=repr(exc))
        return False


def notify_signals(cfg, alerts: AlertBus, signals_summary: str) -> bool:
    token_env = cfg.get("alerts.telegram_token_env", "TELEGRAM_TOKEN")
    chat_env = cfg.get("alerts.telegram_chat_id_env", "TELEGRAM_CHAT_ID")
    text = f"*quant-signal-engine (PAPER)*\n{signals_summary}\nAlertas: {alerts.summary()}"
    return send_telegram(token_env, chat_env, text)
