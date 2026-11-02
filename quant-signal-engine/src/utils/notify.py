"""Notificaciones multi-canal de señales (Telegram / Discord / email).

TODO se configura con variables de entorno (ver .env.example); sin ellas es no-op.
Fail-safe total: un fallo de red o de credenciales NUNCA rompe el pipeline de señales
(se loguea como alerta y se devuelve False).

El mensaje incluye SIEMPRE la precisión en vivo (paper) medida contra resultados reales,
para que la acertabilidad se vea al lado de cada señal de forma honesta.
"""

from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import requests
import structlog

from src.utils.alerts import AlertBus
from src.utils.telegram import send_telegram

log = structlog.get_logger("qse.notify")

DISCLAIMER = "Esto no es asesoría financiera ni garantiza rentabilidad. Solo PAPER TRADING."


def load_live_accuracy(reports_dir: Path) -> dict | None:
    """Carga reports/live_accuracy.json (generado por src.signals.evaluator)."""
    p = Path(reports_dir) / "live_accuracy.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("live_accuracy_unreadable", error=repr(exc))
        return None


def build_signals_message(cfg, signals: pd.DataFrame, accuracy: dict | None, alerts: AlertBus) -> str:
    """Construye el mensaje de señales: lista de señales + precisión en vivo honesta + alertas."""
    lines = ["⚡ quant-signal-engine — señales del día (PAPER TRADING)"]
    if not signals.empty:
        date = str(signals["date"].iloc[0])
        mode = str(signals["data_mode"].iloc[0]) if "data_mode" in signals.columns else "real"
        lines.append(f"📅 {date} | datos: {mode}{' ⚠️ SINTÉTICOS' if mode == 'synthetic' else ''}")
        act = signals[signals["direction"] != "nada"].copy()
        if act.empty:
            lines.append("🚫 Hoy no hay señales (fail-safe NADA: ninguna llegó al umbral de emisión).")
        else:
            act["_score"] = act["meta_prob"].astype(float)
            act = act.sort_values("_score", ascending=False)  # mejor primero (top 1, 2, 3...)
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        dots = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}
        for rk, (_, s) in enumerate(act.iterrows(), start=1):
            mark = medals.get(rk, f"{rk}.")
            prob = float(s.get("prob", 0))
            band = str(s.get("band", "")) or (
                "verde" if prob >= 0.70 else ("amarillo" if prob >= 0.50 else "rojo")
            )
            dot = dots.get(band, "⚪")
            top = ", ".join(
                str(s.get(c, ""))
                for c in ("top1", "top2", "top3")
                if c in s and str(s.get(c, "")) not in ("", "nan")
            )
            lines.append(
                f"{mark}{dot} {s['asset']} → {str(s['direction']).upper()} | prob {prob:.2f} "
                f"meta {float(s.get('meta_prob', 0)):.2f} | entry {s.get('entry_ref')} SL {s.get('stop')} "
                f"TP {s.get('tp')} size {s.get('size')}" + (f" | top: {top}" if top else "")
            )
        lines.append(f"Emitidas: {len(act)}/{len(signals)} (umbral de emisión 0.86; el resto NADA)")
    # precisión en vivo, honesta, junto a las señales
    lines.append("📊 Precisión en vivo (paper, vs resultados reales):")
    if accuracy and int(accuracy.get("n_evaluated", 0)) > 0:
        lines.append(
            f"   • Global: {float(accuracy['accuracy']):.1%} aciertos sobre {accuracy['n_evaluated']} señales "
            f"evaluadas (Brier {float(accuracy.get('brier', float('nan'))):.3f}, prob media {float(accuracy.get('mean_prob', 0)):.2f})"
        )
        for row in accuracy.get("by_threshold", [])[:6]:
            lines.append(f"   • prob {row['bucket']}: {float(row['accuracy']):.0%} (n={row['n']})")
    else:
        lines.append("   • Aún sin señales evaluables (se mide al cerrar cada operación paper).")
    for tr in (accuracy or {}).get("recent_trades", [])[-5:]:
        mark = "✅" if int(tr.get("hit", 0)) == 1 else "❌"
        lines.append(
            f"   {mark} {tr.get('date')} {tr.get('asset')} {str(tr.get('direction')).upper()} "
            f"p={float(tr.get('prob', 0)):.2f} → R={float(tr.get('realized_r', 0)):+.2f}"
        )
    lines.append(f"🚨 Alertas: {alerts.summary()}")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def send_discord(webhook_env: str, text: str, timeout: float = 10.0) -> bool:
    url = os.getenv(webhook_env, "").strip()
    if not url:
        return False
    try:
        resp = requests.post(url, json={"content": text[:1900]}, timeout=timeout)
        ok = resp.ok
        if not ok:
            log.warning("discord_failed", status=resp.status_code, body=resp.text[:200])
        return ok
    except Exception as exc:  # noqa: BLE001 - fail-safe
        log.warning("discord_exception", error=repr(exc))
        return False


def send_email(cfg, text: str, timeout: float = 15.0) -> bool:
    host = os.getenv(cfg.get("alerts.smtp_host_env", "SMTP_HOST"), "").strip()
    user = os.getenv(cfg.get("alerts.smtp_user_env", "SMTP_USER"), "").strip()
    pw = os.getenv(cfg.get("alerts.smtp_pass_env", "SMTP_PASS"), "").strip()
    frm = os.getenv(cfg.get("alerts.email_from_env", "EMAIL_FROM"), "").strip()
    to = os.getenv(cfg.get("alerts.email_to_env", "EMAIL_TO"), "").strip()
    if not (host and frm and to):
        return False
    try:
        port = int(os.getenv(cfg.get("alerts.smtp_port_env", "SMTP_PORT"), "587"))
    except ValueError:
        port = 587
    try:
        msg = EmailMessage()
        msg["Subject"] = "⚡ quant-signal-engine — señales del día (PAPER)"
        msg["From"] = frm
        msg["To"] = to
        msg.set_content(text)
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=timeout) as srv:
                if user and pw:
                    srv.login(user, pw)
                srv.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=timeout) as srv:
                srv.starttls()
                if user and pw:
                    srv.login(user, pw)
                srv.send_message(msg)
        return True
    except Exception as exc:  # noqa: BLE001 - fail-safe
        log.warning("email_exception", error=repr(exc))
        return False


def notify_signals_full(
    cfg, alerts: AlertBus, signals: pd.DataFrame, accuracy: dict | None = None
) -> dict[str, bool]:
    """Envía el mensaje por todos los canales configurados. Devuelve {canal: ok}. Nunca lanza."""
    results: dict[str, bool] = {}
    try:
        if accuracy is None:
            accuracy = load_live_accuracy(cfg.reports_dir)
        text = build_signals_message(cfg, signals, accuracy, alerts)
        raw = os.getenv(cfg.get("alerts.notify_channels_env", "QSE_NOTIFY_CHANNELS"), "").strip().lower()
        want = {c.strip() for c in raw.split(",") if c.strip()} if raw else {"telegram", "discord", "email"}
        if "telegram" in want:
            results["telegram"] = send_telegram(
                cfg.get("alerts.telegram_token_env", "TELEGRAM_TOKEN"),
                cfg.get("alerts.telegram_chat_id_env", "TELEGRAM_CHAT_ID"),
                text,
            )
        if "discord" in want:
            results["discord"] = send_discord(
                cfg.get("alerts.discord_webhook_env", "DISCORD_WEBHOOK_URL"), text
            )
        if "email" in want:
            results["email"] = send_email(cfg, text)
        log.info("notify_done", channels=results, n_chars=len(text))
    except Exception as exc:  # noqa: BLE001 - fail-safe absoluto
        log.warning("notify_exception", error=repr(exc))
        alerts.add("WARNING", f"Fallo al notificar (fail-safe, las señales ya quedaron en CSV): {exc!r}")
    return results
