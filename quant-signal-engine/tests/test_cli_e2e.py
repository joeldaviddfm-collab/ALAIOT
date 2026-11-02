"""Tests end-to-end de los CLI: src.run (paper, offline) y src.backtest (offline, quick)
con datos sintéticos y config pequeña."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

VENV_PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def tiny_config(tmp_path) -> Path:
    base = yaml.safe_load((ROOT / "config.yaml").read_text())
    base["data"]["start"] = "2023-01-01"
    base["model"]["walk_forward"] = {"train_days": 100, "embargo_days": 11, "test_days": 30, "blind_days": 45}
    base["model"]["lgbm"].update(
        {"n_estimators": 30, "max_depth": 3, "num_leaves": 7, "min_child_samples": 10}
    )
    base["model"]["threshold"] = 0.55
    base["model"]["meta_min"] = 0.5
    base["robustness"].update({"n_permutations": 30, "n_random_strategies": 30, "mc_sims": 100})
    base["assets"]["default"] = ["BTC-USD", "SPY"]
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(base))
    return p


def test_run_paper_offline(tiny_config):
    # limpiar señales previas para probar idempotencia desde cero
    for f in ("history.csv", "latest.csv"):
        p = ROOT / "signals" / f
        if p.exists():
            p.unlink()
    r = subprocess.run(
        [VENV_PY, "-m", "src.run", "--mode", "paper", "--offline", "--retrain", "--config", str(tiny_config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    latest = pd.read_csv(ROOT / "signals" / "latest.csv")
    assert set(latest.columns) >= {
        "date",
        "asset",
        "direction",
        "prob",
        "entry_ref",
        "stop",
        "tp",
        "size",
        "top1",
        "top2",
        "top3",
    }
    assert set(latest["direction"]).issubset({"long", "short", "nada"})
    assert {"rank", "band"} <= set(latest.columns)  # ranking top1/2/3 + banda de color
    hist = pd.read_csv(ROOT / "signals" / "history.csv")
    assert len(hist) == len(latest)
    # correr de nuevo el mismo día no duplica
    r2 = subprocess.run(
        [VENV_PY, "-m", "src.run", "--mode", "paper", "--offline", "--config", str(tiny_config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r2.returncode == 0
    hist2 = pd.read_csv(ROOT / "signals" / "history.csv")
    assert len(hist2) == len(hist)
    # funnel de filtros escrito tras la corrida
    funnel = json.loads((ROOT / "reports" / "filter_funnel.json").read_text())
    assert funnel["candidatos"] >= 2
    assert funnel["emitidas"] == int((latest["direction"] != "nada").sum())
    assert set(funnel["stages"]) >= {"candidatos", "umbral_no_alcanzado", "emitidas"}


def test_resolve_assets_universe_preset():
    from src.config import Config
    from src.run import parse_args, resolve_assets

    cfg = Config()
    args = parse_args(["--mode", "paper", "--universe", "preset"])
    assets = resolve_assets(cfg, args)
    assert len(assets) >= 20 and "BTC-USD" in assets and "SPY" in assets and "EURUSD=X" in assets
    args2 = parse_args(["--mode", "paper", "--assets", "AAPL", "TSLA"])
    assert resolve_assets(cfg, args2) == ["AAPL", "TSLA"]


def test_backtest_walkforward_report_offline(tiny_config):
    r = subprocess.run(
        [
            VENV_PY,
            "-m",
            "src.backtest",
            "--walk-forward",
            "--report",
            "--offline",
            "--config",
            str(tiny_config),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    assert (ROOT / "report.md").exists()
    assert (ROOT / "go_no_go.md").exists()
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    assert "blind" in metrics and "val" in metrics and "robustness" in metrics
    assert "sharpe" in metrics["blind"]
    txt = (ROOT / "go_no_go.md").read_text()
    assert "NO APTO" in txt or "APTO" in txt
    # gráficos generados
    assert (ROOT / "reports" / "equity_curve.png").exists()


def test_evaluator_cli(tiny_config):
    r = subprocess.run(
        [VENV_PY, "-m", "src.signals.evaluator", "--offline"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
