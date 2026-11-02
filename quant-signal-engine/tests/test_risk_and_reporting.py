"""RiskManager (circuit breakers, pérdidas consecutivas), reportes (report.md/go_no_go.md)
y Telegram (sin token => no-op)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.backtester.portfolio import RiskManager
from src.reporting.report import evaluate_go_no_go, write_full_report, write_go_no_go
from src.utils.telegram import send_telegram


def _trade(pnl):
    return type("T", (), {"pnl": pnl})()


def test_drawdown_circuit_breaker(cfg):
    rm = RiskManager(cfg)
    equity = pd.Series([100000, 99000, 84000, 83000], index=pd.bdate_range("2024-01-01", periods=4))
    st = rm.drawdown_state(equity)
    assert st["breached"] and st["max_dd"] <= -0.15
    ok, reason = rm.allow_new_signals(equity, [])
    assert not ok and "drawdown" in reason


def test_consecutive_losses_pause(cfg):
    rm = RiskManager(cfg)
    equity = pd.Series([100000, 99500, 99000], index=pd.bdate_range("2024-01-01", periods=3))
    trades = [_trade(-100)] * 5
    ok, reason = rm.allow_new_signals(equity, trades)
    assert not ok and "consecutivas" in reason


def test_risk_ok(cfg):
    rm = RiskManager(cfg)
    equity = pd.Series(
        100000 * np.cumprod(1 + np.full(30, 0.001)), index=pd.bdate_range("2024-01-01", periods=30)
    )
    ok, reason = rm.allow_new_signals(equity, [_trade(100), _trade(-50)])
    assert ok


def test_go_no_go_verdict(cfg, tmp_path):
    blind = {
        "n_trades": 150,
        "win_rate": 0.6,
        "profit_factor": 1.8,
        "sharpe": 1.5,
        "max_drawdown": -0.08,
        "expectancy": 0.004,
    }
    extra = {"deflated_p": 0.01, "pbo": 0.2, "expectancy_costs_x2": 0.001, "beats_bh_risk_adjusted": True}
    criteria = evaluate_go_no_go(blind, {"days": 120, "within_ci": True}, extra, cfg)
    path = tmp_path / "go_no_go.md"
    ok = write_go_no_go(cfg, criteria, path)
    text = path.read_text()
    assert "| Criterio |" in text
    assert ("APTO" in text) == ok
    # si falla un criterio, el veredicto es NO APTO
    blind_bad = dict(blind, sharpe=0.1)
    criteria2 = evaluate_go_no_go(blind_bad, {"days": 0, "within_ci": False}, extra, cfg)
    ok2 = write_go_no_go(cfg, criteria2, path)
    assert not ok2 and "NO APTO" in path.read_text()


def test_write_full_report(cfg, tmp_path):
    bundle = {
        "data_mode": "synthetic",
        "train": {
            "n_trades": 10,
            "win_rate": 0.5,
            "sharpe": 0.5,
            "max_drawdown": -0.05,
            "profit_factor": 1.1,
            "expectancy": 0.001,
            "sortino": 0.6,
            "calmar": 0.2,
            "cagr": 0.05,
            "volatility": 0.1,
        },
        "val": {
            "n_trades": 20,
            "win_rate": 0.55,
            "sharpe": 0.8,
            "max_drawdown": -0.07,
            "profit_factor": 1.4,
            "expectancy": 0.002,
        },
        "blind": {
            "n_trades": 30,
            "win_rate": 0.6,
            "sharpe": 1.2,
            "max_drawdown": -0.09,
            "profit_factor": 1.6,
            "expectancy": 0.003,
            "sortino": 1.4,
            "calmar": 1.0,
            "cagr": 0.12,
            "volatility": 0.12,
            "skew": 0.1,
            "kurt": 1.0,
        },
        "baseline": {"sharpe": 0.4, "max_drawdown": -0.2},
        "buy_hold": {"sharpe": 0.6, "max_drawdown": -0.25},
        "equity_val": pd.Series(
            100000 * np.cumprod(1 + np.full(50, 0.001)), index=pd.bdate_range("2024-01-01", periods=50)
        ),
        "equity_blind": pd.Series(
            105000 * np.cumprod(1 + np.full(40, 0.001)), index=pd.bdate_range("2024-03-01", periods=40)
        ),
        "equity_base": pd.Series(
            100000 * np.cumprod(1 + np.full(90, 0.0005)), index=pd.bdate_range("2024-01-01", periods=90)
        ),
        "equity_bh": pd.Series(
            100000 * np.cumprod(1 + np.full(90, 0.0004)), index=pd.bdate_range("2024-01-01", periods=90)
        ),
        "accuracy_by_threshold": pd.DataFrame(
            {"bucket": ["0.70-0.75", ">=0.90"], "n": [10, 5], "accuracy": [0.6, 0.8]}
        ),
        "per_asset": pd.DataFrame({"asset": ["BTC-USD", "SPY"], "n": [15, 15], "accuracy": [0.6, 0.6]}),
        "per_regime": pd.DataFrame({"regime_id": [0, 1, 2], "n": [5, 10, 15], "accuracy": [0.4, 0.6, 0.7]}),
        "paper": {"days": 0, "within_ci": False},
        "extra": {
            "n_configs": 12,
            "perm_p": 0.01,
            "random_p": 0.02,
            "n_random": 1000,
            "dsr": 2.5,
            "deflated_p": 0.01,
            "pbo": 0.2,
            "mc": {
                "p_ruin": 0.001,
                "ret_p05": -0.1,
                "ret_p50": 0.2,
                "ret_p95": 0.6,
                "maxdd_p95": -0.2,
                "n_sims": 2000,
            },
            "leakage_auc": 0.51,
            "exp_x2": 0.001,
            "sharpe_x2": 0.9,
            "exp_x3": -0.001,
            "sharpe_x3": 0.5,
            "expectancy_costs_x2": 0.001,
            "stress": {"COVID 2020": -0.12},
            "ablation": {"__full__": 1.2, "momentum": 1.1, "macro": 1.25},
            "sensitivity": {"max_depth": [("3", 1.1), ("5", 1.3)]},
            "cross_asset": {"n_trades": 10, "sharpe": 0.5},
            "beats_bh_risk_adjusted": True,
            "n_folds": 8,
            "n_val_signals": 40,
            "n_blind_signals": 30,
        },
        "metrics_json": {"blind": {"sharpe": 1.2}},
    }
    out = write_full_report(cfg, bundle)
    assert (cfg.root / "report.md").exists()
    assert (cfg.root / "go_no_go.md").exists()
    assert (cfg.reports_dir / "metrics.json").exists()
    charts = out["charts"]
    assert charts and all(p.endswith(".png") for p in charts)
    text = (cfg.root / "report.md").read_text()
    assert "Cómo podría fallar este sistema" in text
    assert "no es asesoría financiera" in text


def test_telegram_noop_without_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert send_telegram("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "hola") is False


def test_telegram_with_mock(monkeypatch):
    import src.utils.telegram as tg

    monkeypatch.setenv("TELEGRAM_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
    called = {}

    class R:
        ok = True

    def fake_post(url, json=None, timeout=None):
        called["url"] = url
        return R()

    monkeypatch.setattr(tg.requests, "post", fake_post)
    assert send_telegram("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "hola") is True
    assert "botx" in called["url"]


def test_check_regression_skip_cuando_cero_operaciones(tmp_path, monkeypatch):
    """Si la corrida actual no emitió operaciones OOS (fail-safe), no es regresión."""
    import scripts.check_regression as cr

    monkeypatch.setattr(cr, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "metrics.json").write_text(
        '{"data_mode": "synthetic", "blind": {"n_trades": 0, "sharpe": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}}'
    )
    (tmp_path / "reports" / "baseline_metrics.json").write_text(
        '{"data_mode": "synthetic", "blind": {"n_trades": 5, "sharpe": 1.2, "profit_factor": 1.5, "max_drawdown": -0.05}}'
    )
    assert cr.main() == 0
