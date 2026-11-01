"""Kill switch global: archivo KILL_SWITCH o env QSE_KILL_SWITCH => todas NADA."""

from __future__ import annotations

from src.signals.generator import generate_signals
from src.utils.alerts import AlertBus


def test_kill_switch_env_blocks_everything(cfg, small_market, monkeypatch):
    monkeypatch.setenv("QSE_KILL_SWITCH", "1")
    alerts = AlertBus(cfg.reports_dir)
    sig = generate_signals(cfg, small_market, None, alerts)
    assert (sig["direction"] == "nada").all()
    assert alerts.has_critical()
    assert any("KILL SWITCH" in a.msg for a in alerts.items)


def test_kill_switch_file_blocks_everything(cfg, small_market, monkeypatch, tmp_path):
    ks = cfg.root / "KILL_SWITCH"
    ks.write_text("1")
    try:
        alerts = AlertBus(cfg.reports_dir)
        sig = generate_signals(cfg, small_market, None, alerts)
        assert (sig["direction"] == "nada").all()
    finally:
        ks.unlink()
