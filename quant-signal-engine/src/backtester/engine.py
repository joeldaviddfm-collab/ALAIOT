"""Motor de backtest realista:
- la señal generada al cierre de t se ejecuta en la APERTURA de la barra siguiente;
- comisiones + spread + slippage configurables (config.yaml), multiplicables para stress;
- salidas intradía por stop/TP usando high/low de la vela, o al vencimiento (max_h);
- dimensionamiento por ATR (riesgo 1% por operación) con Kelly fraccional (tope 1/4)
  y tope duro de nocional; correlation y máximo de posiciones;
- sin apalancamiento por defecto.

El MISMO motor se usa en backtest y en paper trading (consistencia garantizada).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger("qse.engine")


@dataclass
class Trade:
    asset: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    side: int
    size_units: float
    stop: float
    tp: float
    pnl: float
    r_multiple: float
    costs: float
    exit_reason: str
    prob: float


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    metrics: dict
    signals: pd.DataFrame
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _costs_bps(cfg, cost_mult: float) -> dict[str, float]:
    c = cfg.get("costs", {})
    return {
        "commission": float(c.get("commission_bps", 5.0)) * cost_mult,
        "spread": float(c.get("spread_bps", 2.0)) * cost_mult,
        "slippage": float(c.get("slippage_bps", 3.0)) * cost_mult,
    }


def size_position(
    equity: float,
    entry: float,
    stop: float,
    cfg,
    *,
    win_rate: float | None = None,
    payoff: float | None = None,
) -> float:
    """Tamaño en unidades: riesgo 1% del capital por operación (ATR), con Kelly
    fraccional (máx. 1/4) y tope duro de nocional. Sin apalancamiento."""
    risk = float(cfg.get("risk.risk_per_trade", 0.01))
    hard_cap = float(cfg.get("risk.hard_risk_cap", risk))
    kelly_frac = float(cfg.get("risk.kelly_fraction", 0.25))
    max_pos = float(cfg.get("risk.max_position_pct", 0.25))
    dist = abs(entry - stop)
    if dist <= 0 or entry <= 0:
        return 0.0
    units_risk = equity * risk / dist
    units = units_risk
    if win_rate is not None and payoff is not None and payoff > 0 and 0 < win_rate < 1:
        kelly = win_rate - (1 - win_rate) / payoff
        f = max(0.0, kelly) * kelly_frac
        f = min(f, hard_cap)
        units_kelly = equity * f / entry
        units = min(units, units_kelly)
    units = min(units, equity * max_pos / entry)
    return float(max(0.0, units))


def next_trading_date(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    after = index[index > date]
    return after[0] if len(after) else None


def run_backtest(
    prices: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    cfg,
    *,
    cost_mult: float = 1.0,
    initial_equity: float | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> BacktestResult:
    """signals: DataFrame con columnas [date (cierre que genera la señal), asset,
    direction (+1/-1; 0 = ignorar), prob, stop, tp]. La entrada es la apertura de la
    siguiente barra disponible de ese activo."""
    risk_cfg = cfg.get("risk", {})
    initial_equity = float(initial_equity or risk_cfg.get("initial_equity", 100000.0))
    max_h = int(cfg.get("model.triple_barrier.max_horizon_days", 10))
    max_positions = int(risk_cfg.get("max_positions", 5))
    max_corr = float(risk_cfg.get("max_correlation", 0.70))
    max_per_day = int(risk_cfg.get("max_signals_per_day", 3))
    costs = _costs_bps(cfg, cost_mult)

    sig = signals.copy()
    sig["date"] = pd.DatetimeIndex(sig["date"])
    sig = sig[sig["direction"] != 0].copy()
    # precio de ejecución: apertura de la siguiente barra
    exec_map = {}
    for asset, df in prices.items():
        idx = df.index
        for d in sig.loc[sig["asset"] == asset, "date"].unique():
            nd = next_trading_date(idx, pd.Timestamp(d))
            if nd is not None:
                exec_map[(pd.Timestamp(d), asset)] = nd
    sig["exec_date"] = [exec_map.get((d, a)) for d, a in zip(sig["date"], sig["asset"], strict=False)]
    sig = sig.dropna(subset=["exec_date"])
    sig["exec_date"] = pd.DatetimeIndex(sig["exec_date"])

    all_dates = sorted({d for df in prices.values() for d in df.index})
    if start is not None:
        all_dates = [d for d in all_dates if d >= start]
    if end is not None:
        all_dates = [d for d in all_dates if d <= end]
    if not all_dates:
        eq = pd.Series([initial_equity], index=pd.DatetimeIndex([pd.Timestamp("2000-01-01")]))
        return BacktestResult(equity=eq, trades=[], metrics={"n_trades": 0}, signals=sig)

    equity_curve: dict[pd.Timestamp, float] = {}
    equity = initial_equity
    open_pos: dict[str, Trade] = {}
    trades: list[Trade] = []
    # win rate/payoff histórico para Kelly fraccional (sobre trades ya cerrados)
    closed: list[Trade] = []

    def corr_with_open(asset: str, date: pd.Timestamp) -> float:
        if not open_pos:
            return 0.0
        rets_new = prices[asset]["close"].pct_change()
        worst = 0.0
        for other in open_pos:
            rets_o = prices[other]["close"].pct_change()
            common = rets_new.index.intersection(rets_o.index)
            w = common[common <= date][-60:]
            if len(w) < 20:
                continue
            c = float(np.corrcoef(rets_new.loc[w], rets_o.loc[w])[0, 1])
            if np.isfinite(c):
                worst = max(worst, abs(c))
        return worst

    for t in all_dates:
        # 1) salidas intradía de posiciones abiertas
        for asset in list(open_pos):
            p = open_pos[asset]
            df = prices[asset]
            if t not in df.index:
                continue
            bar = df.loc[t]
            exit_price: float | None = None
            reason: str = ""
            if p.side == 1:
                if bar["low"] <= p.stop:
                    exit_price, reason = p.stop * (1 - costs["slippage"] / 1e4), "stop"
                elif bar["high"] >= p.tp:
                    exit_price, reason = p.tp * (1 - costs["slippage"] / 1e4), "tp"
            else:
                if bar["high"] >= p.stop:
                    exit_price, reason = p.stop * (1 + costs["slippage"] / 1e4), "stop"
                elif bar["low"] <= p.tp:
                    exit_price, reason = p.tp * (1 + costs["slippage"] / 1e4), "tp"
            if exit_price is None and (t - p.entry_date).days >= max_h:
                exit_price, reason = float(bar["close"]), "time"
            if exit_price is not None:
                notional_exit = exit_price * p.size_units
                cost = notional_exit * costs["commission"] / 1e4
                pnl = p.side * (exit_price - p.entry_price) * p.size_units - cost
                trade = Trade(
                    asset=asset,
                    signal_date=p.signal_date,
                    entry_date=p.entry_date,
                    entry_price=p.entry_price,
                    exit_date=t,
                    exit_price=float(exit_price),
                    side=p.side,
                    size_units=p.size_units,
                    stop=p.stop,
                    tp=p.tp,
                    pnl=float(pnl),
                    r_multiple=float(pnl / (equity * 0.01) if equity else 0.0),
                    costs=float(cost),
                    exit_reason=reason or "unknown",
                    prob=p.prob,
                )
                trades.append(trade)
                closed.append(trade)
                equity += pnl
                del open_pos[asset]
        # 2) entradas de señales cuyo exec_date es hoy (apertura de la barra)
        todays = sig[sig["exec_date"] == t]
        if len(todays) > max_per_day:
            todays = todays.sort_values("prob", ascending=False).head(max_per_day)
        for _, s in todays.iterrows():
            asset = s["asset"]
            if asset in open_pos or asset not in prices:
                continue
            if len(open_pos) >= max_positions:
                break
            if corr_with_open(asset, t) > max_corr:
                log.info("skip_correlation", asset=asset, date=str(t.date()))
                continue
            df = prices[asset]
            if t not in df.index:
                continue
            entry = float(df.loc[t, "open"])
            entry_adj = entry * (1 + s["direction"] * costs["slippage"] / 1e4)  # slippage en contra
            cost_in = entry_adj * costs["commission"] / 1e4 + entry_adj * costs["spread"] / 1e4
            direction = int(s["direction"])
            stop = float(s["stop"])
            tp = float(s["tp"])
            win_rate = payoff = None
            if len(closed) >= 20:
                pnls = np.array([c.pnl for c in closed])
                win_rate = float((pnls > 0).mean())
                wins = pnls[pnls > 0]
                losses = pnls[pnls < 0]
                payoff = (
                    float(wins.mean() / abs(losses.mean()))
                    if len(wins) and len(losses) and losses.mean() != 0
                    else None
                )
            units = size_position(equity, entry_adj, stop, cfg, win_rate=win_rate, payoff=payoff)
            if units <= 0:
                continue
            cost_total = cost_in * units
            pos = Trade(
                asset=asset,
                signal_date=s["date"],
                entry_date=t,
                entry_price=entry_adj,
                exit_date=t,
                exit_price=np.nan,
                side=direction,
                size_units=units,
                stop=stop,
                tp=tp,
                pnl=-cost_total,
                r_multiple=0.0,
                costs=cost_total,
                exit_reason="open",
                prob=float(s["prob"]),
            )
            open_pos[asset] = pos
            equity -= cost_total
        equity_curve[t] = equity

    # cerrar posiciones abiertas al final al último close
    for asset, p in list(open_pos.items()):
        df = prices[asset]
        last = df.index[-1]
        if last not in df.index:
            continue
        exit_price = float(df.loc[last, "close"])
        cost = exit_price * p.size_units * costs["commission"] / 1e4
        pnl = p.side * (exit_price - p.entry_price) * p.size_units - cost
        trades.append(
            Trade(
                asset=asset,
                signal_date=p.signal_date,
                entry_date=p.entry_date,
                entry_price=p.entry_price,
                exit_date=last,
                exit_price=exit_price,
                side=p.side,
                size_units=p.size_units,
                stop=p.stop,
                tp=p.tp,
                pnl=float(pnl),
                r_multiple=float(pnl / (initial_equity * 0.01)),
                costs=float(cost),
                exit_reason="end",
                prob=p.prob,
            )
        )

    eq = pd.Series(equity_curve, dtype=float)
    eq.index.name = "date"
    from src.validation.metrics import equity_metrics

    metrics = equity_metrics(eq, trades)
    metrics["cost_mult"] = cost_mult
    daily_ret = eq.pct_change().dropna()
    return BacktestResult(equity=eq, trades=trades, metrics=metrics, signals=sig, daily_returns=daily_ret)
