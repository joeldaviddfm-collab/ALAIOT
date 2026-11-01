"""Dashboard estático: build_dashboard genera un HTML auto-contenido con los marcadores clave."""

from __future__ import annotations

import json

import pandas as pd
from src.reporting.dashboard import build_dashboard, md_to_html


def test_md_to_html_tables_and_headings():
    md = "# Título\n\n| A | B |\n|---|---|\n| 1 | ✅ |\n\n- item uno\n- item dos\n\n> cita\n\ntexto con **negrita** y `codigo`\n"
    out = md_to_html(md)
    assert "<h2>Título</h2>" in out
    assert "<table>" in out and "<th>A</th>" in out and "✅" in out
    assert "<ul>" in out and "<li>item uno</li>" in out
    assert "<blockquote>" in out
    assert "<b>negrita</b>" in out and "<code>codigo</code>" in out


def test_build_dashboard_self_contained(cfg):
    # artefactos mínimos
    cfg.signals_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2026-10-30",
                "asset": "BTC-USD",
                "direction": "long",
                "prob": 0.81,
                "meta_prob": 0.7,
                "agreement": 1.0,
                "entry_ref": 100.0,
                "stop": 96.0,
                "tp": 112.0,
                "size": 10.0,
                "top1": "rsi_14",
                "top2": "adx_14",
                "top3": "vix_z",
                "reason": "señal válida (paper)",
                "data_mode": "synthetic",
            }
        ]
    ).to_csv(cfg.signals_dir / "latest.csv", index=False)
    (cfg.reports_dir / "metrics.json").write_text(
        json.dumps(
            {
                "data_mode": "synthetic",
                "blind": {
                    "n_trades": 120,
                    "win_rate": 0.72,
                    "sharpe": 1.4,
                    "profit_factor": 1.8,
                    "max_drawdown": -0.08,
                },
                "val": {"n_trades": 50},
                "robustness": {
                    "pbo": 0.2,
                    "deflated_p": 0.01,
                    "leakage_auc": 0.5,
                    "n_configs": 10,
                    "mc": {"p_ruin": 0.001},
                },
            }
        ),
        encoding="utf-8",
    )
    (cfg.root / "go_no_go.md").write_text(
        "# Compuerta GO / NO-GO\n\n**Veredicto: NO APTO PARA CAPITAL REAL**\n\n"
        "| Criterio | Umbral mínimo | Medido | ¿Cumple? |\n|---|---|---|---|\n"
        "| Profit factor | > 1.3 | 1.8 | ✅ |\n| Sharpe OOS | > 1.0 | 0.2 | ❌ |\n",
        encoding="utf-8",
    )
    (cfg.root / "report.md").write_text(
        "# Informe\n\n## 1. Resultados\n\n| M | V |\n|---|---|\n| Sharpe | 1.2 |\n", encoding="utf-8"
    )
    (cfg.reports_dir / "alerts.log").write_text(
        "[2026-11-01T10:00:00+00:00] WARNING: demo\n", encoding="utf-8"
    )

    path = build_dashboard(cfg)
    text = path.read_text(encoding="utf-8")
    assert path.name == "dashboard.html"
    assert "<!DOCTYPE html>" in text and "</html>" in text
    # auto-contenido: sin referencias externas (no http/https src, no CDN)
    assert 'src="http' not in text and 'href="http' not in text.replace('href="https://github.com', "")
    assert "NO APTO" in text and "BTC-USD" in text and "dir-long" in text
    assert "Profit factor" in text and "✅" in text and "❌" in text
    import re

    srcs = re.findall(r"<img src='([^']+)'", text)
    # auto-contenido: toda imagen debe ir incrustada en base64, nunca externa
    assert all(s.startswith("data:image/png;base64,") for s in srcs)
    assert "<style>" in text  # CSS inline
    assert "no es asesoría financiera" in text.lower()
