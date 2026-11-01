"""TEST OBLIGATORIO: APIs caídas (mocks). Si todas las fuentes fallan, el sistema no
rompe: marca activos bloqueados, registra alertas CRITICAL y emite señal = NADA."""

from __future__ import annotations

import pandas as pd
import src.data.prices as prices_mod
from src.data.pipeline import load_market_data
from src.signals.generator import generate_signals
from src.utils.alerts import AlertBus


def _boom(*args, **kwargs):
    raise ConnectionError("API caída (simulada)")


def test_all_apis_down_fail_safe(cfg, monkeypatch):
    monkeypatch.setattr(prices_mod.YFinanceSource, "fetch", _boom)
    monkeypatch.setattr(prices_mod.CCXTSource, "fetch", _boom)
    # parchear en el namespace del pipeline (que es quien llama a estas funciones)
    import src.data.pipeline as pipe_mod

    def _down(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(pipe_mod, "get_macro", _down)
    monkeypatch.setattr(pipe_mod, "get_sentiment", _down)
    monkeypatch.setattr(pipe_mod, "get_derivatives", _down)
    monkeypatch.setattr(pipe_mod, "get_onchain", _down)

    alerts = AlertBus(cfg.reports_dir)
    # load_market_data no debe lanzar: los activos quedan bloqueados
    market = load_market_data(cfg, ["BTC-USD", "SPY"], offline=False, alerts=alerts)
    assert market.blocked, "debería haber activos bloqueados"
    assert market.prices == {}
    assert alerts.has_critical()

    # generar señales sin modelo y con todo caído => todo NADA, sin excepción
    signals = generate_signals(cfg, market, None, alerts)
    assert (signals["direction"] == "nada").all()
    assert len(signals) >= 2


def test_run_paper_with_apis_down_does_not_crash(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(prices_mod.YFinanceSource, "fetch", _boom)
    monkeypatch.setattr(prices_mod.CCXTSource, "fetch", _boom)
    from src import run as run_mod

    rc = run_mod.main(["--mode", "paper", "--assets", "BTC-USD", "SPY"])
    assert rc == 0  # fail-safe: no rompe; emite NADA
    latest = pd.read_csv(cfg.signals_dir / "latest.csv")
    assert (latest["direction"] == "nada").all()
