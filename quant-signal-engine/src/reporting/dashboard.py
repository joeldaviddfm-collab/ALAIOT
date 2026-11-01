"""Dashboard estático auto-contenido: reports/dashboard.html.

Se VE como una web (tema oscuro, KPIs, gráficos, tablas) pero NO es una web:
es un único archivo HTML con todo incrustado (CSS inline + imágenes en base64),
sin servidor, sin CDN y sin dependencias externas => funciona abriéndolo con
doble clic (file://) o sirviéndolo con cualquier servidor estático.
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

import pandas as pd

CSS = """
:root{--bg:#0b0e14;--panel:#121826;--panel2:#0f1420;--line:#1e2a44;--txt:#dbe4ff;--dim:#8b98b8;
--acc:#4f8cff;--ok:#2ecc71;--bad:#ff5c5c;--warn:#f1c40f;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.5 -apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
header{position:sticky;top:0;z-index:10;background:rgba(11,14,20,.92);backdrop-filter:blur(6px);
border-bottom:1px solid var(--line);padding:14px 22px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header h1{font-size:18px;margin:0;letter-spacing:.3px}
header .sub{color:var(--dim);font-size:12.5px;width:100%}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;border:1px solid var(--line);background:var(--panel)}
.badge.ok{color:var(--ok);border-color:var(--ok)} .badge.bad{color:var(--bad);border-color:var(--bad)}
.badge.warn{color:var(--warn);border-color:var(--warn)} .badge.acc{color:var(--acc);border-color:var(--acc)}
main{max-width:1180px;margin:0 auto;padding:22px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.card .k{color:var(--dim);font-size:11.5px;text-transform:uppercase;letter-spacing:.6px}
.card .v{font:600 22px/1.2 var(--mono);margin-top:4px}
.card .v.ok{color:var(--ok)} .card .v.bad{color:var(--bad)} .card .v.warn{color:var(--warn)}
section{margin-top:26px}
h2{font-size:16px;margin:0 0 12px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h2 .n{color:var(--acc);font-family:var(--mono);margin-right:8px}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
th{background:var(--panel2);text-align:left;padding:8px 10px;color:var(--dim);font-size:11.5px;text-transform:uppercase;letter-spacing:.5px}
td{padding:7px 10px;border-top:1px solid var(--line);font-family:var(--mono);font-size:12.5px}
tr:hover td{background:#16203a}
.dir-long{color:var(--ok);font-weight:700}.dir-short{color:var(--bad);font-weight:700}.dir-nada{color:var(--dim)}
.charts{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.charts img{width:100%;border:1px solid var(--line);border-radius:12px;background:#fff}
pre{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;overflow:auto;font:12px var(--mono);color:var(--dim);max-height:280px}
details{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin-top:12px}
summary{cursor:pointer;font-weight:600;color:var(--acc)}
.report h1{font-size:20px}.report h2{font-size:16px}.report h3{font-size:14px;color:var(--acc)}
.report blockquote{border-left:3px solid var(--warn);margin:10px 0;padding:6px 12px;background:var(--panel);border-radius:0 10px 10px 0}
.report table{margin:10px 0}.report code{background:var(--panel2);padding:1px 5px;border-radius:6px;font-family:var(--mono)}
footer{color:var(--dim);font-size:12px;text-align:center;padding:26px 0 40px;border-top:1px solid var(--line);margin-top:34px}
.two{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(360px,1fr))}
"""


def _esc(s) -> str:
    return html.escape("" if s is None else str(s))


def _img_b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _df_table(df: pd.DataFrame, *, cls: str = "") -> str:
    if df is None or df.empty:
        return "<p style='color:var(--dim)'>Sin datos.</p>"
    head = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            cell = _esc(v)
            if c == "direction":
                d = str(v).lower()
                cell = f"<span class='dir-{d}'>{_esc(v)}</span>"
            elif c in ("¿Cumple?", "ok") and str(v) in ("✅", "❌", "True", "False"):
                cell = (
                    f"<span class='{'dir-long' if str(v) in ('✅','True') else 'dir-short'}'>{_esc(v)}</span>"
                )
            tds.append(f"<td>{cell}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _inline(text: str) -> str:
    text = _esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" style="color:var(--acc)">\1</a>', text)
    return text


def md_to_html(md: str) -> str:
    """Conversor mínimo markdown→HTML (títulos, tablas, listas, citas, negritas)."""
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    list_open = False

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            out.append("</ul>")
            list_open = False

    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            close_list()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            if len(rows) >= 2:
                header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
                body = (
                    rows[2:]
                    if len(rows) > 2 and set(rows[1].replace("|", "").strip()) <= set("-: ")
                    else rows[1:]
                )
                out.append(
                    "<table><thead><tr>"
                    + "".join(f"<th>{_inline(h)}</th>" for h in header)
                    + "</tr></thead><tbody>"
                )
                for r in body:
                    cells = [c.strip() for c in r.strip().strip("|").split("|")]
                    out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
                out.append("</tbody></table>")
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            close_list()
            level = min(len(m.group(1)) + 1, 4)
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if line.startswith("- "):
            if not list_open:
                out.append("<ul>")
                list_open = True
            out.append(f"<li>{_inline(line[2:])}</li>")
            i += 1
            continue
        if line.startswith("> "):
            close_list()
            out.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
            i += 1
            continue
        if line.strip() == "":
            close_list()
            i += 1
            continue
        close_list()
        out.append(f"<p>{_inline(line)}</p>")
        i += 1
    close_list()
    return "\n".join(out)


def _parse_go_no_go(path: Path) -> tuple[str, pd.DataFrame]:
    """Extrae veredicto y tabla de criterios de go_no_go.md."""
    if not path.exists():
        return "SIN DATOS", pd.DataFrame()
    text = path.read_text(encoding="utf-8")
    m = re.search(r"\*\*Veredicto:\s*(.+?)\*\*", text)
    verdict = m.group(1) if m else "SIN DATOS"
    rows = []
    for line in text.splitlines():
        if (
            line.startswith("|")
            and "Criterio" not in line
            and not set(line.replace("|", "").strip()) <= set("-: ")
        ):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 4:
                rows.append(cells)
    return verdict, pd.DataFrame(
        rows, columns=["Criterio", "Umbral", "Medido", "¿Cumple?"]
    ) if rows else pd.DataFrame()


def _kpi(label: str, value, cls: str = "") -> str:
    return (
        f"<div class='card'><div class='k'>{_esc(label)}</div><div class='v {cls}'>{_esc(value)}</div></div>"
    )


def _fmt(x, pct=False):
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if xf != xf or xf in (float("inf"), float("-inf")):
        return "inf" if xf > 0 else "n/a"
    return f"{xf:.2%}" if pct else f"{xf:.3f}"


def build_dashboard(cfg) -> Path:
    """Genera reports/dashboard.html auto-contenido a partir de signals/ y reports/."""
    root: Path = cfg.root
    reports: Path = cfg.reports_dir
    signals_dir: Path = cfg.signals_dir
    out_path = reports / "dashboard.html"

    metrics = {}
    mpath = reports / "metrics.json"
    if mpath.exists():
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
    blind = metrics.get("blind", {}) or {}
    rob = metrics.get("robustness", {}) or {}
    mc = rob.get("mc", {}) or {}
    data_mode = metrics.get("data_mode", "real")

    verdict, gonogo_df = _parse_go_no_go(root / "go_no_go.md")
    vcls = "ok" if "APTO" in verdict and "NO APTO" not in verdict else "bad"
    mode_cls = "warn" if data_mode == "synthetic" else "acc"

    # KPIs
    kpis = "".join(
        [
            _kpi("Veredicto GO/NO-GO", verdict, vcls),
            _kpi(
                "Modo de datos",
                data_mode.upper() + (" (SINTÉTICO)" if data_mode == "synthetic" else ""),
                mode_cls,
            ),
            _kpi(
                "Sharpe OOS (ciego)",
                _fmt(blind.get("sharpe")),
                "ok" if (blind.get("sharpe") or 0) > 1 else "bad",
            ),
            _kpi(
                "Win rate OOS",
                _fmt(blind.get("win_rate"), pct=True),
                "ok" if (blind.get("win_rate") or 0) >= 0.7 else "warn",
            ),
            _kpi("Profit factor", _fmt(blind.get("profit_factor"))),
            _kpi(
                "Max drawdown",
                _fmt(blind.get("max_drawdown"), pct=True),
                "ok" if abs(blind.get("max_drawdown") or 0) < 0.15 else "bad",
            ),
            _kpi(
                "Operaciones OOS",
                blind.get("n_trades", 0),
                "ok" if (blind.get("n_trades") or 0) >= 100 else "bad",
            ),
            _kpi("PBO (CSCV)", _fmt(rob.get("pbo")), "ok" if (rob.get("pbo") or 1) < 0.5 else "bad"),
            _kpi(
                "p-value Deflated Sharpe",
                _fmt(rob.get("deflated_p")),
                "ok" if (rob.get("deflated_p") or 1) < 0.05 else "bad",
            ),
            _kpi(
                "Prob. de ruina (MC)",
                _fmt(mc.get("p_ruin"), pct=True),
                "ok" if (mc.get("p_ruin") or 1) < 0.01 else "bad",
            ),
            _kpi("Leakage AUC (≈0.5 esperado)", _fmt(rob.get("leakage_auc"))),
            _kpi("Configs probadas", rob.get("n_configs", 1)),
        ]
    )

    # señales de hoy
    latest = pd.DataFrame()
    lpath = signals_dir / "latest.csv"
    if lpath.exists():
        latest = pd.read_csv(lpath)
    n_sig = int((latest["direction"] != "nada").sum()) if not latest.empty else 0

    # gráficos
    charts_html = ""
    for name, caption in (
        ("equity_curve.png", "Curva de equity"),
        ("drawdown.png", "Drawdown"),
        ("accuracy_by_threshold.png", "Precisión por umbral de confianza"),
    ):
        p = reports / name
        if p.exists():
            charts_html += f"<figure style='margin:0'><img src='{_img_b64(p)}' alt='{caption}'><figcaption style='color:var(--dim);font-size:12px;margin-top:6px'>{caption}</figcaption></figure>"

    # alertas recientes
    alerts_html = "<p style='color:var(--dim)'>Sin alertas registradas.</p>"
    alog = reports / "alerts.log"
    if alog.exists():
        tail = alog.read_text(encoding="utf-8").splitlines()[-40:]
        alerts_html = "<pre>" + _esc("\n".join(tail)) + "</pre>"

    # informe completo (report.md → HTML)
    report_html = "<p style='color:var(--dim)'>Ejecuta <code>python -m src.backtest --walk-forward --report</code> para generar el informe.</p>"
    rpath = root / "report.md"
    if rpath.exists():
        report_html = f"<div class='report'>{md_to_html(rpath.read_text(encoding='utf-8'))}</div>"

    # precisión en vivo por umbral (si existe)
    acc_live = reports / "live_accuracy_by_threshold.csv"
    acc_live_html = ""
    if acc_live.exists():
        acc_live_html = "<h2><span class='n'>§</span>Precisión en vivo (paper) por umbral</h2>" + _df_table(
            pd.read_csv(acc_live)
        )

    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC")
    doc = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>quant-signal-engine — dashboard (PAPER)</title>
<style>{CSS}</style></head>
<body>
<header>
  <h1>⚡ quant-signal-engine</h1>
  <span class="badge {vcls}">Veredicto: {_esc(verdict)}</span>
  <span class="badge {mode_cls}">Datos: {_esc(data_mode)}{' ⚠️ sintéticos' if data_mode=='synthetic' else ''}</span>
  <span class="badge acc">MODO PAPER TRADING</span>
  <span class="badge">Señales activas hoy: {n_sig}</span>
  <span class="badge">Generado: {now}</span>
  <div class="sub">Dashboard estático auto-contenido (un solo archivo HTML, sin servidor). Esto no es asesoría financiera ni garantiza rentabilidad.</div>
</header>
<main>
<section><h2><span class="n">01</span>KPIs out-of-sample (test ciego)</h2><div class="grid">{kpis}</div></section>
<section><h2><span class="n">02</span>Señales de hoy (signals/latest.csv)</h2>{_df_table(latest)}</section>
<section><h2><span class="n">03</span>Compuerta GO / NO-GO</h2>{_df_table(gonogo_df)}</section>
<section><h2><span class="n">04</span>Gráficos</h2><div class="charts">{charts_html or '<p style="color:var(--dim)">Sin gráficos todavía.</p>'}</div></section>
{acc_live_html}
<section><h2><span class="n">05</span>Alertas recientes (reports/alerts.log)</h2>{alerts_html}</section>
<section><h2><span class="n">06</span>Informe completo (report.md)</h2>{report_html}</section>
</main>
<footer>quant-signal-engine · solo PAPER TRADING · esto no es asesoría financiera ni garantiza rentabilidad · archivo estático generado automáticamente</footer>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path
