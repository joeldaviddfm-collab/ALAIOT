"""Gestión de riesgo de portfolio: circuit breakers (drawdown diario/semanal/total),
límite de pérdidas consecutivas y kill switch. En modo paper, si se supera el límite
de drawdown el sistema DEJA de emitir señales y alerta.
"""

from __future__ import annotations

import pandas as pd


class RiskManager:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def kill_switch_active(self) -> bool:
        return bool(self.cfg.kill_switch)

    def drawdown_state(self, equity: pd.Series) -> dict:
        equity = pd.Series(equity).astype(float).dropna()
        if len(equity) < 2:
            return {"max_dd": 0.0, "daily_dd": 0.0, "weekly_dd": 0.0, "breached": False, "reason": ""}
        dd = float((equity / equity.cummax() - 1).min())
        rets = equity.pct_change().dropna()
        daily_dd = float(rets.min()) if len(rets) else 0.0
        weekly = equity.resample("W").last().pct_change().dropna()
        weekly_dd = float(weekly.min()) if len(weekly) else 0.0
        max_dd = float(self.cfg.get("risk.max_drawdown", 0.15))
        daily_max = float(self.cfg.get("risk.max_drawdown_daily", 0.05))
        weekly_max = float(self.cfg.get("risk.max_drawdown_weekly", 0.10))
        breached = dd <= -max_dd or daily_dd <= -daily_max or weekly_dd <= -weekly_max
        reason = ""
        if dd <= -max_dd:
            reason = f"drawdown total {dd:.1%} supera límite {max_dd:.0%}"
        elif daily_dd <= -daily_max:
            reason = f"drawdown diario {daily_dd:.1%} supera {daily_max:.0%}"
        elif weekly_dd <= -weekly_max:
            reason = f"drawdown semanal {weekly_dd:.1%} supera {weekly_max:.0%}"
        return {
            "max_dd": dd,
            "daily_dd": daily_dd,
            "weekly_dd": weekly_dd,
            "breached": breached,
            "reason": reason,
        }

    def consecutive_losses(self, trades: list) -> int:
        n = 0
        for t in reversed(trades):
            if t.pnl < 0:
                n += 1
            else:
                break
        return n

    def allow_new_signals(self, equity: pd.Series, trades: list) -> tuple[bool, str]:
        """Devuelve (permitido, razón). Fail-safe: por defecto NO opera si algo falla."""
        if self.kill_switch_active():
            return False, "KILL SWITCH activo (archivo KILL_SWITCH o QSE_KILL_SWITCH)"
        st = self.drawdown_state(equity)
        if st["breached"]:
            return False, f"circuit breaker: {st['reason']}"
        max_losses = int(self.cfg.get("risk.max_consecutive_losses", 4))
        cl = self.consecutive_losses(trades)
        if cl >= max_losses:
            return False, f"{cl} pérdidas consecutivas (límite {max_losses}): pausa y revisión"
        return True, "ok"
