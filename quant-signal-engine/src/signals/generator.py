"""Generador de señales diarias (MODO PAPER ÚNICAMENTE).

Fail-safe: por defecto NO emite señal. Solo emite si pasan TODAS las validaciones:
datos válidos, modelo entrenado, filtros de contexto, gestión de riesgo y kill switch.
Ante cualquier duda: direction = "nada" + alerta explícita.

Salidas:
- signals/latest.csv  (una fila por activo para la fecha de la señal)
- signals/history.csv (registro acumulado, idempotente por (date, asset))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from src.backtester.portfolio import RiskManager
from src.data.validator import validate_ohlcv
from src.features import build_feature_matrix
from src.features.technical import atr
from src.models.pipeline import SignalModel
from src.signals.context import context_block, load_event_dates
from src.utils.alerts import AlertBus

log = structlog.get_logger("qse.signals")

SIGNAL_COLUMNS = [
    "date",
    "asset",
    "direction",
    "prob",
    "meta_prob",
    "agreement",
    "entry_ref",
    "stop",
    "tp",
    "size",
    "top1",
    "top2",
    "top3",
    "reason",
    "data_mode",
]


def _paper_equity_from_history(cfg) -> tuple[pd.Series, list]:
    """Reconstruye la curva de equity paper desde history.csv (misma lógica que el engine)."""
    import os

    hist_path = cfg.signals_dir / "history.csv"
    equity0 = float(cfg.get("risk.initial_equity", 100000.0))
    if not os.path.exists(hist_path):
        return pd.Series([equity0]), []
    hist = pd.read_csv(hist_path)
    trades = []
    eq = [equity0]
    for _, r in hist.iterrows():
        if str(r.get("direction", "nada")) in ("long", "short") and pd.notna(r.get("realized_pnl")):
            trades.append(type("T", (), {"pnl": float(r["realized_pnl"])})())
            eq.append(eq[-1] + float(r["realized_pnl"]))
    return pd.Series(eq), trades


def generate_signals(
    cfg,
    market,
    model: SignalModel | None,
    alerts: AlertBus,
    *,
    asof_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    assets = list(market.prices.keys())
    rm = RiskManager(cfg)
    events = load_event_dates()
    data_mode = market.data_mode

    # --- kill switch global: ninguna señal ---
    if rm.kill_switch_active():
        alerts.add("CRITICAL", "KILL SWITCH activo: TODAS las señales son NADA")
        for asset in assets + [a for a in market.blocked if a not in assets]:
            rows.append(_nada_row(asset, asof_date, "kill switch activo", data_mode))
        return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)

    # --- circuit breakers sobre el paper trading acumulado ---
    eq, paper_trades = _paper_equity_from_history(cfg)
    allowed, risk_reason = rm.allow_new_signals(eq, paper_trades)
    if not allowed:
        alerts.add("CRITICAL", f"Gestión de riesgo bloquea señales: {risk_reason}")

    X, _groups = build_feature_matrix(market.prices, market.global_df, cfg)
    if X.empty:
        # sin precios válidos no hay features: todo NADA (fail-safe)
        for asset in sorted(set(market.prices) | set(market.blocked)):
            reason = market.blocked.get(asset, "sin datos de precios")
            alerts.add("CRITICAL", f"{asset}: señal NADA ({reason})")
            rows.append(_nada_row(asset, asof_date, f"sin datos: {reason}", data_mode))
        return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
    X = X.sort_values(["asset", "date"]).reset_index(drop=True)

    for asset in sorted(set(assets) | set(market.blocked)):
        if asset in market.blocked:
            alerts.add("CRITICAL", f"{asset}: señal NADA ({market.blocked[asset]})")
            rows.append(
                _nada_row(asset, asof_date, f"datos no disponibles: {market.blocked[asset]}", data_mode)
            )
            continue
        df = market.prices[asset]
        rep = validate_ohlcv(df, max_stale_bdays=int(cfg.get("data.max_stale_bdays", 5)))
        for w in rep.warnings:
            alerts.add("WARNING", f"{asset}: {w}")
        if not rep.ok:
            alerts.add("CRITICAL", f"{asset}: señal NADA por validación fallida: {rep.issues}")
            rows.append(
                _nada_row(asset, asof_date, f"validación fallida: {'; '.join(rep.issues)}", data_mode)
            )
            continue
        if not allowed:
            rows.append(
                _nada_row(asset, asof_date, f"bloqueado por gestión de riesgo: {risk_reason}", data_mode)
            )
            continue
        sub = X[X["asset"] == asset]
        if asof_date is not None:
            sub = sub[sub["date"] <= asof_date]
        if sub.empty:
            rows.append(_nada_row(asset, asof_date, "sin features para la fecha", data_mode))
            continue
        row = sub.iloc[-1]
        date = pd.Timestamp(row["date"])
        if model is None or not model.trained:
            alerts.add("CRITICAL", f"{asset}: modelo no entrenado => señal NADA")
            rows.append(_nada_row(asset, date, "modelo no disponible", data_mode))
            continue
        # --- filtros de contexto ---
        block = context_block(row, cfg, events, date)
        if block is not None:
            alerts.add("WARNING", f"{asset}: contexto bloquea señal ({block})")
            rows.append(_nada_row(asset, date, f"filtro de contexto: {block}", data_mode))
            continue
        # --- filtro de régimen malo ---
        regime_id = row.get("regime_id")
        if pd.notna(regime_id) and model.regime_perf:
            wr = model.regime_perf.get(int(regime_id))
            if wr is not None and wr < float(cfg.get("context_filters.bad_regime_winrate_min", 0.45)):
                alerts.add(
                    "WARNING", f"{asset}: régimen {int(regime_id)} con winrate histórico {wr:.2f} < umbral"
                )
                rows.append(_nada_row(asset, date, f"régimen desfavorable (winrate {wr:.2f})", data_mode))
                continue
        # --- predicción ---
        pred = model.predict(row.to_frame().T)[["direction", "prob", "meta_prob", "agreement"]].iloc[0]
        direction = int(pred["direction"])
        if direction == 0:
            rows.append(
                _nada_row(
                    asset,
                    date,
                    f"umbral no alcanzado (p={pred['prob']:.2f}, meta={pred['meta_prob']:.2f}, acuerdo={pred['agreement']:.2f})",
                    data_mode,
                )
            )
            continue
        # --- niveles de stop/TP con RR mínima ---
        atr_v = float(atr(df, 14).iloc[-1])
        entry_ref = float(df["close"].iloc[-1])
        sl_mult = float(cfg.get("model.triple_barrier.atr_mult_sl", 2.0))
        tp_mult = float(cfg.get("model.triple_barrier.atr_mult_tp", 3.0))
        min_rr = float(cfg.get("risk.min_rr", 1.5))
        dist = sl_mult * atr_v
        rr_needed = max(tp_mult, min_rr * sl_mult)
        if direction == 1:
            stop = entry_ref - dist
            tp = entry_ref + rr_needed * atr_v
        else:
            stop = entry_ref + dist
            tp = entry_ref - rr_needed * atr_v
        if stop <= 0 or tp <= 0:
            rows.append(_nada_row(asset, date, "stop/TP inválidos", data_mode))
            continue
        size = float(
            max(0.0, (eq.iloc[-1] * float(cfg.get("risk.risk_per_trade", 0.01))) / abs(entry_ref - stop))
        )
        top3 = model.top_features(row.to_frame().T, direction, k=3)
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "asset": asset,
                "direction": "long" if direction == 1 else "short",
                "prob": round(float(pred["prob"]), 4),
                "meta_prob": round(float(pred["meta_prob"]), 4),
                "agreement": round(float(pred["agreement"]), 4),
                "entry_ref": round(entry_ref, 6),
                "stop": round(stop, 6),
                "tp": round(tp, 6),
                "size": round(size, 8),
                "top1": top3[0] if len(top3) > 0 else "",
                "top2": top3[1] if len(top3) > 1 else "",
                "top3": top3[2] if len(top3) > 2 else "",
                "reason": "señal válida (paper)",
                "data_mode": data_mode,
            }
        )
        log.info(
            "signal",
            asset=asset,
            direction=rows[-1]["direction"],
            prob=rows[-1]["prob"],
            date=str(date.date()),
        )
    out = pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
    return out


def _nada_row(asset: str, date, reason: str, data_mode: str) -> dict:
    d = (
        pd.Timestamp(date).strftime("%Y-%m-%d")
        if date is not None
        else pd.Timestamp.now().strftime("%Y-%m-%d")
    )
    return {
        "date": d,
        "asset": asset,
        "direction": "nada",
        "prob": 0.0,
        "meta_prob": 0.0,
        "agreement": 0.0,
        "entry_ref": np.nan,
        "stop": np.nan,
        "tp": np.nan,
        "size": 0.0,
        "top1": "",
        "top2": "",
        "top3": "",
        "reason": reason,
        "data_mode": data_mode,
    }


def write_signals(cfg, latest: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Escribe latest.csv (overwrite) e history.csv (append idempotente por date+asset)."""
    cfg.signals_dir.mkdir(parents=True, exist_ok=True)
    latest_path = cfg.signals_dir / "latest.csv"
    hist_path = cfg.signals_dir / "history.csv"
    latest.to_csv(latest_path, index=False)
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        combined = pd.concat([hist, latest], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "asset"], keep="last")
        combined = combined.sort_values(["date", "asset"]).reset_index(drop=True)
    else:
        combined = latest.copy()
    combined.to_csv(hist_path, index=False)
    log.info("signals_written", latest=str(latest_path), history=str(hist_path), rows=len(combined))
    return latest_path, combined
