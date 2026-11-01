"""Backtest walk-forward con purging/embargo + reporte completo + pruebas de robustez.

Uso:
    python -m src.backtest --walk-forward --report
    python -m src.backtest --walk-forward --report --offline --quick   # demo/CI con datos sintéticos

Genera: report.md, go_no_go.md, reports/ (gráficos y métricas).
Nunca usa train/test aleatorio; el test ciego final es intocable.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd

from src.backtester.baseline import run_baseline, run_buy_hold
from src.backtester.engine import run_backtest
from src.config import ROOT, Config
from src.data.pipeline import load_market_data
from src.models.pipeline import SignalModel
from src.reporting.report import write_full_report
from src.signals.evaluator import accuracy_by_threshold, evaluate_signals
from src.training import build_training_set, signals_from_predictions
from src.utils.alerts import AlertBus
from src.utils.logging import setup_logging
from src.utils.seeds import set_seeds
from src.validation import robustness as R
from src.validation.metrics import deflated_sharpe, equity_metrics, pbo_cscv
from src.validation.walkforward import blind_split, wf_splits

QUICK_OVERRIDES = {
    "data": {"start": "2020-01-01"},
    "model": {
        "walk_forward": {"train_days": 252, "embargo_days": 21, "test_days": 63, "blind_days": 126},
        "lgbm": {"n_estimators": 150},
    },
    "robustness": {"n_permutations": 200, "n_random_strategies": 200, "mc_sims": 500},
}

AUX_COLS = ("date", "asset", "close_ref", "atr14")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="src.backtest", description="Backtest walk-forward + reporte")
    p.add_argument(
        "--walk-forward", action="store_true", help="activa validación walk-forward (siempre activa)"
    )
    p.add_argument("--report", action="store_true", help="genera report.md, go_no_go.md y gráficos")
    p.add_argument("--offline", action="store_true", help="datos sintéticos (solo demo/test)")
    p.add_argument("--quick", action="store_true", help="configuración rápida para CI/demo")
    p.add_argument("--assets", nargs="+", default=None)
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def _engine(prices, sig, cfg, cost_mult=1.0, start=None, end=None):
    return run_backtest(prices, sig, cfg, cost_mult=cost_mult, start=start, end=end)


def _slice_equity(eq: pd.Series, start, end) -> pd.Series:
    return eq.loc[str(pd.Timestamp(start).date()) : str(pd.Timestamp(end).date())]  # type: ignore[misc]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = QUICK_OVERRIDES if args.quick else None
    cfg = Config(args.config, overrides=overrides)
    log = setup_logging(ROOT / "logs" / "backtest.log")
    set_seeds(cfg.seed)
    alerts = AlertBus(cfg.reports_dir)
    assets = args.assets or list(cfg.get("assets.default", []))
    log.info("backtest_start", offline=args.offline, quick=args.quick, assets=assets)

    market = load_market_data(cfg, assets, offline=args.offline, alerts=alerts)
    if not market.prices:
        print("ERROR: sin datos de precios; no se puede hacer backtest.", flush=True)
        return 1
    X, y, groups = build_training_set(cfg, market)
    feat_cols = [c for c in X.columns if c not in AUX_COLS]

    wf = cfg.get("model.walk_forward", {})
    dates = pd.DatetimeIndex(sorted(X["date"].unique()))
    dev_pos, blind_pos = blind_split(dates, int(wf.get("blind_days", 252)))
    dev_dates = pd.DatetimeIndex(dates[dev_pos])
    blind_dates = pd.DatetimeIndex(dates[blind_pos])
    mask_dev = X["date"].isin(set(dev_dates))
    mask_blind = X["date"].isin(set(blind_dates))
    X_dev, y_dev = X[mask_dev].reset_index(drop=True), y[mask_dev]
    X_blind = X[mask_blind].reset_index(drop=True)
    log.info(
        "splits",
        n_dev=len(X_dev),
        n_blind=len(X_blind),
        blind_start=str(blind_dates[0].date()),
        blind_end=str(blind_dates[-1].date()),
    )

    n_configs = 1  # el modelo final; se incrementa con cada variante probada

    # ---------------- walk-forward (validación OOS) ----------------
    folds = wf_splits(
        dev_dates,
        train_days=int(wf.get("train_days", 504)),
        embargo_days=int(wf.get("embargo_days", 21)),
        test_days=int(wf.get("test_days", 63)),
    )
    thr = float(cfg.get("model.threshold", 0.70))
    val_full, val_primary, val_ens = [], [], []
    for tr_pos, te_pos in folds:
        tr_set = set(dev_dates[tr_pos])
        te_set = set(dev_dates[te_pos])
        tr_mask = X_dev["date"].isin(tr_set)
        te_mask = X_dev["date"].isin(te_set)
        if tr_mask.sum() < 100 or te_mask.sum() == 0:
            continue
        model = SignalModel(cfg)
        model.fit(X_dev[tr_mask][list(AUX_COLS) + feat_cols], y_dev[tr_mask], fast_calibration=True)
        n_configs += 1
        preds = model.predict(X_dev[te_mask][list(AUX_COLS) + feat_cols])
        sig = signals_from_predictions(X_dev[te_mask], preds, cfg)
        val_full.append(sig)
        sp = sig.copy()
        sp["direction"] = np.where(
            (sig["prob_pos"] >= thr) & (sig["prob_pos"] > sig["prob_neg"]),
            1,
            np.where((sig["prob_neg"] >= thr) & (sig["prob_neg"] > sig["prob_pos"]), -1, 0),
        )
        val_primary.append(sp)
        se = sig.copy()
        se["direction"] = np.where(
            (sig["prob_ens_pos"] >= thr) & (sig["prob_ens_pos"] > sig["prob_ens_neg"]),
            1,
            np.where((sig["prob_ens_neg"] >= thr) & (sig["prob_ens_neg"] > sig["prob_pos"]), -1, 0),
        )
        val_ens.append(se)
    sig_val = (
        pd.concat(val_full, ignore_index=True)
        if val_full
        else pd.DataFrame(columns=["date", "asset", "direction", "prob", "stop", "tp"])
    )
    sig_val_p = pd.concat(val_primary, ignore_index=True) if val_primary else sig_val.iloc[0:0]
    sig_val_e = pd.concat(val_ens, ignore_index=True) if val_ens else sig_val.iloc[0:0]
    log.info("wf_done", folds=len(folds), val_signals=int((sig_val["direction"] != 0).sum()))

    dev_start, dev_end = dev_dates[0], dev_dates[-1]
    bl_start, bl_end = blind_dates[0], blind_dates[-1]
    res_val = _engine(market.prices, sig_val, cfg, start=dev_start, end=dev_end)
    res_val_p = _engine(market.prices, sig_val_p, cfg, start=dev_start, end=dev_end)
    res_val_e = _engine(market.prices, sig_val_e, cfg, start=dev_start, end=dev_end)

    # ---------------- modelo final sobre dev; test ciego intocable ----------------
    model = SignalModel(cfg)
    model.fit(X_dev[list(AUX_COLS) + feat_cols], y_dev, fast_calibration=False)
    preds_blind = model.predict(X_blind[list(AUX_COLS) + feat_cols])
    sig_blind = signals_from_predictions(X_blind, preds_blind, cfg)
    res_blind = _engine(market.prices, sig_blind, cfg, start=bl_start, end=bl_end)
    # train (in-sample, solo referencia)
    preds_train = model.predict(X_dev[list(AUX_COLS) + feat_cols])
    sig_train = signals_from_predictions(X_dev, preds_train, cfg)
    res_train = _engine(market.prices, sig_train, cfg, start=dev_start, end=dev_end)

    # ---------------- baseline y buy & hold ----------------
    eq_base_full = run_baseline(market.prices, cfg)
    eq_bh_full = run_buy_hold(market.prices)
    base_blind = equity_metrics(_slice_equity(eq_base_full, bl_start, bl_end))
    bh_blind = equity_metrics(_slice_equity(eq_bh_full, bl_start, bl_end))

    # ---------------- robustez ----------------
    rob = cfg.get("robustness", {})
    trades = res_blind.trades
    rs = np.array([t.r_multiple for t in trades], dtype=float)
    sharpe_b = float(res_blind.metrics.get("sharpe", 0.0))
    perm_p = R.permutation_pvalue(rs, sharpe_b, n=int(rob.get("n_permutations", 1000)), seed=cfg.seed)
    sig_points = [(pd.Timestamp(t.signal_date), t.asset) for t in trades]
    rand_p = R.random_strategies_pvalue(
        market.prices, sig_points, sharpe_b, n=int(rob.get("n_random_strategies", 1000)), seed=cfg.seed
    )
    mc = R.monte_carlo_block_bootstrap(
        rs,
        block=int(rob.get("mc_block", 5)),
        n_sims=int(rob.get("mc_sims", 2000)),
        ruin_threshold=float(rob.get("ruin_threshold", 0.5)),
        seed=cfg.seed,
    )
    n_days_blind = int(res_blind.metrics.get("n_days", 1))
    dsr, deflated_p = deflated_sharpe(
        sharpe_b,
        float(res_blind.metrics.get("skew", 0.0)),
        float(res_blind.metrics.get("kurt", 0.0)),
        max(2, n_days_blind),
        n_trials=max(1, n_configs),
    )
    pbo = pbo_cscv(
        {
            "full": res_val.daily_returns,
            "primary_only": res_val_p.daily_returns,
            "ensemble": res_val_e.daily_returns,
        }
    )
    leak_auc = R.leakage_auc(X_dev[feat_cols], y_dev, seed=cfg.seed)

    # stress de costos x2 / x3 sobre el test ciego
    res_x2 = _engine(market.prices, sig_blind, cfg, cost_mult=2.0, start=bl_start, end=bl_end)
    res_x3 = _engine(market.prices, sig_blind, cfg, cost_mult=3.0, start=bl_start, end=bl_end)
    stress = R.historical_stress(res_blind.equity, list(rob.get("stress_windows", [])))

    # ablation study
    ablation: dict[str, float] = {"__full__": sharpe_b}
    for grp in sorted(set(groups.values())):
        drop = [c for c, g in groups.items() if g == grp and c in feat_cols]
        if not drop:
            continue
        keep = [c for c in feat_cols if c not in drop]
        m = SignalModel(cfg)
        m.fit(X_dev[list(AUX_COLS) + keep], y_dev, fast_calibration=True)
        n_configs += 1
        pb = m.predict(X_blind[list(AUX_COLS) + keep])
        sb = signals_from_predictions(X_blind, pb, cfg)
        rb = _engine(market.prices, sb, cfg, start=bl_start, end=bl_end)
        ablation[grp] = float(rb.metrics.get("sharpe", 0.0))

    # sensibilidad de hiperparámetros ±20%
    sens: dict[str, list] = {}
    base_params = dict(cfg.get("model.lgbm", {}))
    for param in ("max_depth", "num_leaves", "n_estimators"):
        base = float(base_params.get(param, 4))
        sens[param] = []
        for f in (0.8, 1.2):
            ov: dict = copy.deepcopy(overrides) if overrides else {}
            ov.setdefault("model", {}).setdefault("lgbm", {})[param] = int(round(base * f))
            cfg2 = Config(args.config, overrides=ov)
            m = SignalModel(cfg2)
            m.fit(X_dev[list(AUX_COLS) + feat_cols], y_dev, fast_calibration=True)
            n_configs += 1
            pb = m.predict(X_blind[list(AUX_COLS) + feat_cols])
            sb = signals_from_predictions(X_blind, pb, cfg)
            rb = _engine(market.prices, sb, cfg, start=bl_start, end=bl_end)
            sens[param].append((f"{base * f:.2f}", float(rb.metrics.get("sharpe", 0.0))))

    # validación cruzada entre activos: entrena sin cripto, prueba en cripto (ciego)
    cross_metrics: dict = {}
    tr_nocrypto = (
        X_dev["is_crypto"].to_numpy() == 0 if "is_crypto" in X_dev.columns else np.ones(len(X_dev), bool)
    )
    if tr_nocrypto.sum() > 100 and "is_crypto" in X_blind.columns and (X_blind["is_crypto"] == 1).sum() > 0:
        m = SignalModel(cfg)
        m.fit(X_dev[tr_nocrypto][list(AUX_COLS) + feat_cols], y_dev[tr_nocrypto], fast_calibration=True)
        n_configs += 1
        bl_crypto = X_blind["is_crypto"].to_numpy() == 1
        pb = m.predict(X_blind[bl_crypto][list(AUX_COLS) + feat_cols])
        sb = signals_from_predictions(X_blind[bl_crypto], pb, cfg)
        rb = _engine(market.prices, sb, cfg, start=bl_start, end=bl_end)
        cross_metrics = rb.metrics

    # ---------------- precisión por umbral / por activo / por régimen (ciego) ----------------
    sig_eval = sig_blind[sig_blind["direction"] != 0].copy()
    if len(sig_eval):
        sig_eval["direction"] = np.where(sig_eval["direction"] == 1, "long", "short")
        sig_eval = sig_eval.rename(columns={"close_ref": "entry_ref"})
        results = evaluate_signals(sig_eval, market.prices, cfg)
        acc_tbl = accuracy_by_threshold(results)
        per_asset = (
            results.groupby("asset")
            .agg(n=("hit", "size"), accuracy=("hit", "mean"), avg_r=("realized_r", "mean"))
            .reset_index()
            if len(results)
            else pd.DataFrame()
        )
        per_regime = (
            results.merge(sig_eval[["date", "asset", "regime_id"]], on=["date", "asset"])
            .groupby("regime_id")
            .agg(n=("hit", "size"), accuracy=("hit", "mean"))
            .reset_index()
            if len(results) and "regime_id" in sig_eval.columns
            else pd.DataFrame()
        )
    else:
        results, acc_tbl, per_asset, per_regime = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    beats_bh = bool(
        sharpe_b > float(bh_blind.get("sharpe", 0.0))
        and abs(float(res_blind.metrics.get("max_drawdown", 0)))
        <= abs(float(bh_blind.get("max_drawdown", -1)))
    )

    bundle = {
        "data_mode": market.data_mode,
        "train": res_train.metrics,
        "val": res_val.metrics,
        "blind": res_blind.metrics,
        "baseline": base_blind,
        "buy_hold": bh_blind,
        "equity_val": res_val.equity,
        "equity_blind": res_blind.equity,
        "equity_base": _slice_equity(eq_base_full, dev_start, bl_end),
        "equity_bh": _slice_equity(eq_bh_full, dev_start, bl_end),
        "accuracy_by_threshold": acc_tbl,
        "per_asset": per_asset,
        "per_regime": per_regime,
        "paper": {"days": 0, "within_ci": False},  # se actualiza con paper trading real acumulado
        "extra": {
            "n_configs": n_configs,
            "perm_p": perm_p,
            "random_p": rand_p,
            "n_random": int(rob.get("n_random_strategies", 1000)),
            "dsr": dsr,
            "deflated_p": deflated_p,
            "pbo": pbo,
            "mc": mc,
            "leakage_auc": leak_auc,
            "exp_x2": res_x2.metrics.get("expectancy"),
            "sharpe_x2": res_x2.metrics.get("sharpe"),
            "exp_x3": res_x3.metrics.get("expectancy"),
            "sharpe_x3": res_x3.metrics.get("sharpe"),
            "expectancy_costs_x2": res_x2.metrics.get("expectancy"),
            "stress": stress,
            "ablation": ablation,
            "sensitivity": sens,
            "cross_asset": cross_metrics,
            "beats_bh_risk_adjusted": beats_bh,
            "n_folds": len(folds),
            "n_val_signals": int((sig_val["direction"] != 0).sum()),
            "n_blind_signals": int((sig_blind["direction"] != 0).sum()),
        },
        "metrics_json": {
            "data_mode": market.data_mode,
            "train": res_train.metrics,
            "val": res_val.metrics,
            "blind": res_blind.metrics,
            "baseline_blind": base_blind,
            "buy_hold_blind": bh_blind,
            "robustness": {
                "perm_p": perm_p,
                "random_p": rand_p,
                "dsr": dsr,
                "deflated_p": deflated_p,
                "pbo": pbo,
                "mc": mc,
                "leakage_auc": leak_auc,
                "n_configs": n_configs,
            },
        },
    }

    summary = write_full_report(cfg, bundle) if args.report else {"go_ok": None, "criteria": []}
    # baseline para regresión en CI
    (cfg.reports_dir / "baseline_metrics.json").write_text(
        __import__("json").dumps(
            {
                "sharpe_blind": sharpe_b,
                "max_dd_blind": res_blind.metrics.get("max_drawdown"),
                "n_trades_blind": res_blind.metrics.get("n_trades"),
                "win_rate_blind": res_blind.metrics.get("win_rate"),
                "profit_factor_blind": res_blind.metrics.get("profit_factor"),
                "data_mode": market.data_mode,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    b = res_blind.metrics
    print("\n=== RESUMEN (test ciego, out-of-sample) ===")
    print(
        f"Operaciones: {b.get('n_trades', 0)} | Win rate: {b.get('win_rate', 0):.2%} | Profit factor: {b.get('profit_factor', 0):.2f}"
    )
    print(
        f"Sharpe: {b.get('sharpe', 0):.2f} (deflated p={deflated_p:.3f}) | MaxDD: {b.get('max_drawdown', 0):.2%} | PBO: {pbo:.2f}"
    )
    print(
        f"p-value vs permutación: {perm_p:.4f} | vs {bundle['extra']['n_random']} estrategias aleatorias: {rand_p:.4f} | leakage AUC: {leak_auc:.3f}"
    )
    print(
        f"Monte Carlo: p_ruina={mc.get('p_ruin', 1):.3f}, ret IC90%=[{mc.get('ret_p05', 0):.2%}, {mc.get('ret_p95', 0):.2%}]"
    )
    print(
        f"Costos x2 -> Sharpe {res_x2.metrics.get('sharpe', 0):.2f}, expectancy {res_x2.metrics.get('expectancy', 0):.4f} | x3 -> Sharpe {res_x3.metrics.get('sharpe', 0):.2f}"
    )
    print(
        f"Supera a buy&hold ajustado por riesgo: {'Sí' if beats_bh else 'No'} | Configuraciones probadas: {n_configs}"
    )
    print(
        f"Modo de datos: {market.data_mode}"
        + (" (SINTÉTICO — demo/CI)" if market.data_mode == "synthetic" else "")
    )
    if args.report:
        verdict = "APTO" if summary.get("go_ok") else "NO APTO PARA CAPITAL REAL"
        print(f"\nVeredicto GO/NO-GO: {verdict} (ver go_no_go.md y report.md)")
        print("Esto no es asesoría financiera ni garantiza rentabilidad.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
