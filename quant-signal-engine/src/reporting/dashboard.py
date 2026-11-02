"""Dashboard estático auto-contenido: reports/dashboard.html.

Se VE como una web profesional (tema oscuro, banner de veredicto, KPIs, tablas,
gráficos) pero NO es una web: es un único archivo HTML con todo incrustado
(CSS inline + imágenes en base64), sin servidor, sin CDN y sin dependencias
externas => funciona abriéndolo con doble clic (file://) o sirviéndolo con
cualquier servidor estático (python -m http.server, Netlify, GitHub Pages...).
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

import pandas as pd

CSS = """
:root{--bg:#070b12;--bg2:#0b101b;--panel:#101828;--panel2:#0d1420;--line:#1f2c47;--txt:#e6edf7;--dim:#8fa3c4;
--acc:#5b8cff;--acc2:#22d3ee;--ok:#34d399;--bad:#f87171;--warn:#fbbf24;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:radial-gradient(1200px 500px at 70% -10%,#12203a 0%,var(--bg) 55%);color:var(--txt);
font:15px/1.55 -apple-system,'Segoe UI',Inter,Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
header{position:sticky;top:0;z-index:20;background:rgba(7,11,18,.85);backdrop-filter:blur(10px);
border-bottom:1px solid var(--line);padding:0}
.hbar{max-width:1200px;margin:0 auto;padding:14px 22px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.logo{display:flex;align-items:center;gap:10px;font-weight:800;font-size:18px;letter-spacing:.4px}
.logo .bolt{display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:9px;
background:linear-gradient(135deg,var(--acc),var(--acc2));color:#04101f;font-size:17px;box-shadow:0 4px 14px rgba(91,140,255,.35)}
.logo small{display:block;font-weight:600;font-size:10.5px;color:var(--dim);letter-spacing:1.6px;text-transform:uppercase}
.spacer{flex:1}
.badge{display:inline-block;padding:4px 11px;border-radius:999px;font-size:12px;font-weight:700;border:1px solid var(--line);background:var(--panel);white-space:nowrap}
.badge.ok{color:var(--ok);border-color:rgba(52,211,153,.5);background:rgba(52,211,153,.08)}
.badge.bad{color:var(--bad);border-color:rgba(248,113,113,.5);background:rgba(248,113,113,.08)}
.badge.warn{color:var(--warn);border-color:rgba(251,191,36,.5);background:rgba(251,191,36,.08)}
.badge.acc{color:var(--acc);border-color:rgba(91,140,255,.5);background:rgba(91,140,255,.08)}
.badge.dim{color:var(--dim)}
.banner{border-top:1px solid var(--line);background:linear-gradient(90deg,rgba(248,113,113,.14),rgba(248,113,113,.05))}
.banner.go{background:linear-gradient(90deg,rgba(52,211,153,.16),rgba(52,211,153,.05))}
.banner .inner{max-width:1200px;margin:0 auto;padding:12px 22px;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:baseline}
.banner .verdict{font-size:20px;font-weight:800;letter-spacing:.3px}
.banner.bad .verdict{color:var(--bad)} .banner.go .verdict{color:var(--ok)}
.banner .meta{color:var(--dim);font-size:12.5px}
main{max-width:1200px;margin:0 auto;padding:24px 22px}
section{margin-top:30px}
h2{font-size:15px;margin:0 0 14px;display:flex;align-items:center;gap:10px;letter-spacing:.4px;text-transform:uppercase}
h2 .n{display:inline-flex;align-items:center;justify-content:center;min-width:26px;height:26px;padding:0 7px;border-radius:8px;
background:linear-gradient(135deg,var(--acc),var(--acc2));color:#04101f;font:700 12.5px var(--mono)}
h2 .hint{margin-left:auto;font-size:11.5px;color:var(--dim);text-transform:none;letter-spacing:0;font-weight:500}
.grid{display:grid;gap:13px;grid-template-columns:repeat(auto-fit,minmax(165px,1fr))}
.card{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:14px;
padding:14px 16px;transition:transform .15s,border-color .15s}
.card:hover{transform:translateY(-2px);border-color:#2c3d63}
.card .k{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.9px;font-weight:700}
.card .v{font:600 22px/1.25 var(--mono);margin-top:5px}
.card .s{font-size:11px;color:var(--dim);margin-top:2px}
.v.ok{color:var(--ok)} .v.bad{color:var(--bad)} .v.warn{color:var(--warn)} .v.acc{color:var(--acc)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:14px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13px}
thead th{position:sticky;top:0;background:#0e1626;text-align:left;padding:9px 12px;color:var(--dim);font-size:10.5px;
text-transform:uppercase;letter-spacing:.7px;white-space:nowrap;border-bottom:1px solid var(--line)}
tbody td{padding:8px 12px;border-top:1px solid var(--line);font-family:var(--mono);font-size:12.5px;white-space:nowrap}
tbody tr:hover td{background:#15203a}
td.num,th.num{text-align:right}
.dir-long{color:var(--ok);font-weight:700}.dir-short{color:var(--bad);font-weight:700}.dir-nada{color:var(--dim)}
.band-verde{color:var(--ok);font-weight:700}.band-amarillo{color:var(--warn);font-weight:700}.band-rojo{color:var(--bad);font-weight:700}
.medal{font-size:18px}
.chip{display:inline-block;padding:3px 9px;margin:2px 4px 2px 0;border-radius:8px;font-size:11.5px;border:1px solid var(--line);background:var(--panel2)}
.chip.bad{color:var(--bad);border-color:rgba(248,113,113,.4)}
.chip.ok{color:var(--ok);border-color:rgba(52,211,153,.4)}
.two{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(380px,1fr))}
.panel{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:16px;padding:18px}
.panel h3{margin:0 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:1px;color:var(--dim)}
.bignum{font:700 40px/1 var(--mono)}
.bignum.ok{color:var(--ok)} .bignum.bad{color:var(--bad)} .bignum.warn{color:var(--warn)}
.sub{color:var(--dim);font-size:12.5px;margin-top:6px}
.charts{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(360px,1fr))}
.charts figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px}
.charts img{width:100%;border-radius:9px;background:#fff;display:block}
.charts figcaption{color:var(--dim);font-size:12px;margin-top:8px}
pre{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;overflow:auto;
font:12px var(--mono);color:var(--dim);max-height:300px}
.report h1{font-size:21px}.report h2{font-size:16px;text-transform:none;letter-spacing:0}
.report h3{font-size:14px;color:var(--acc)}
.report blockquote{border-left:3px solid var(--warn);margin:10px 0;padding:8px 14px;background:var(--panel);border-radius:0 10px 10px 0}
.report table{margin:10px 0;border:1px solid var(--line);border-radius:10px}
.report code{background:var(--panel2);padding:1px 6px;border-radius:6px;font-family:var(--mono)}
.report p{margin:8px 0}
footer{color:var(--dim);font-size:12px;text-align:center;padding:28px 20px 44px;border-top:1px solid var(--line);margin-top:40px;line-height:1.7}
@media print{header{position:static}.card:hover{transform:none}}
.alrow{display:flex;gap:10px;align-items:baseline;padding:7px 12px;border-top:1px solid var(--line);font-size:12.5px}
.alrow:first-child{border-top:none}
.alrow:hover{background:#15203a}
.alrow .ts{color:var(--dim);font:11px var(--mono);white-space:nowrap}
.lvl{display:inline-block;min-width:74px;text-align:center;padding:2px 8px;border-radius:7px;font:700 10.5px var(--mono);letter-spacing:.5px}
.lvl-CRITICAL,.lvl-ERROR{color:var(--bad);background:rgba(248,113,113,.12);border:1px solid rgba(248,113,113,.45)}
.lvl-WARNING{color:var(--warn);background:rgba(251,191,36,.10);border:1px solid rgba(251,191,36,.4)}
.lvl-INFO{color:var(--acc);background:rgba(91,140,255,.10);border:1px solid rgba(91,140,255,.35)}
.lvl-OK{color:var(--ok);background:rgba(52,211,153,.10);border:1px solid rgba(52,211,153,.4)}
.alrow .msg{font-family:var(--mono);font-size:12px;word-break:break-word}
.alrow .cnt{color:var(--dim);font:700 11px var(--mono);white-space:nowrap}
.bar{height:9px;border-radius:6px;background:var(--panel2);border:1px solid var(--line);overflow:hidden;margin-top:6px}
.bar>i{display:block;height:100%;border-radius:6px;background:linear-gradient(90deg,var(--acc),var(--acc2))}
.prow{display:flex;justify-content:space-between;gap:10px;font-size:12.5px;margin-top:12px}
.prow .lab{color:var(--dim)} .prow .val{font:700 12.5px var(--mono)}
"""


def _esc(s) -> str:
    return html.escape("" if s is None else str(s))


def _img_b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _df_table(df: pd.DataFrame, *, num_cols: tuple[str, ...] = ()) -> str:
    if df is None or df.empty:
        return "<p class='sub'>Sin datos todavía.</p>"
    head = "".join(f"<th class='{'num' if c in num_cols else ''}'>{_esc(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            cell = _esc(v)
            if (
                c == "prob"
                and "band" in df.columns
                and str(r.get("band", "")) in ("verde", "amarillo", "rojo")
            ):
                cell = f"<span class='band-{r['band']}'>{_esc(v)}</span>"
            elif c == "direction":
                d = str(v).lower()
                cell = f"<span class='dir-{d}'>{_esc(v)}</span>"
            elif c in ("¿Cumple?", "ok", "resultado") and str(v) in ("✅", "❌", "True", "False"):
                good = str(v) in ("✅", "True")
                cell = f"<span class='{'dir-long' if good else 'dir-short'}'>{_esc(v)}</span>"
            tds.append(f"<td class='{'num' if c in num_cols else ''}'>{cell}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f"<div class='tablewrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


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
    df = pd.DataFrame(rows, columns=["Criterio", "Umbral", "Medido", "¿Cumple?"]) if rows else pd.DataFrame()
    return verdict, df


def _kpi(label: str, value, cls: str = "", sub: str = "") -> str:
    return (
        f"<div class='card'><div class='k'>{_esc(label)}</div>"
        f"<div class='v {cls}'>{_esc(value)}</div>"
        + (f"<div class='s'>{_esc(sub)}</div>" if sub else "")
        + "</div>"
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


def _live_accuracy_panel(acc: dict | None) -> str:
    """Panel de acertabilidad en vivo (paper) medida contra resultados reales."""
    if not acc or int(acc.get("n_evaluated", 0)) <= 0:
        return (
            "<div class='panel'><h3>Precisión en vivo (paper)</h3>"
            "<p class='sub'>Aún no hay señales evaluables: la acertabilidad se calcula contra los resultados "
            "reales de cada operación paper (<code>python -m src.signals.evaluator</code>). "
            "Aquí se mostrará SIEMPRE la cifra honesta, sin redondeos favorables.</p></div>"
        )
    a = float(acc["accuracy"])
    cls = "ok" if a >= 0.70 else ("warn" if a >= 0.55 else "bad")
    thr_rows = "".join(
        f"<span class='chip'>{_esc(r['bucket'])}: <b>{float(r['accuracy']):.0%}</b> (n={r['n']})</span>"
        for r in acc.get("by_threshold", [])
    )
    return (
        f"<div class='panel'><h3>Precisión en vivo (paper) — medida contra resultados reales</h3>"
        f"<div class='bignum {cls}'>{a:.1%}</div>"
        f"<div class='sub'>aciertos sobre <b>{acc['n_evaluated']}</b> señales evaluadas · "
        f"Brier {float(acc.get('brier', 0)):.3f} · prob media {float(acc.get('mean_prob', 0)):.2f} · "
        f"datos: {acc.get('data_mode', 'real')}</div>"
        f"<div style='margin-top:10px'>{thr_rows}</div></div>"
    )


def _go_checklist(gonogo_df: pd.DataFrame) -> str:
    """Chips con lo que FALTA para que el veredicto sea APTO (honesto, sin forzar)."""
    if gonogo_df.empty:
        return ""
    failing = gonogo_df[gonogo_df["¿Cumple?"].astype(str).isin(("❌", "False"))]
    if failing.empty:
        return "<div class='panel'><h3>Requisitos GO</h3><span class='chip ok'>✅ Todos los criterios cumplidos</span></div>"
    chips = "".join(
        f"<span class='chip bad'>❌ {_esc(r['Criterio'])}: medido {_esc(r['Medido'])} (mín {_esc(r['Umbral'])})</span>"
        for _, r in failing.iterrows()
    )
    return (
        "<div class='panel'><h3>Qué falta para ser APTO PARA CAPITAL REAL</h3>"
        "<div>" + chips + "</div>"
        "<p class='sub'>El veredicto lo decide la compuerta con datos out-of-sample REALES (test ciego + ≥90 días "
        "de paper). No se fuerza: si los números no llegan, seguirá diciendo NO APTO.</p></div>"
    )


def _alerts_panel(reports: Path) -> str:
    """Panel de alertas con chips de nivel; deduplica repetidas consecutivas (muestra ×N)."""
    alog = reports / "alerts.log"
    if not alog.exists():
        return "<p class='sub'>Sin alertas registradas.</p>"
    rx = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<lvl>[A-Z]+):\s*(?P<msg>.*)$")
    parsed = []
    for line in alog.read_text(encoding="utf-8").splitlines():
        m = rx.match(line.strip())
        if m:
            parsed.append((m.group("ts"), m.group("lvl"), m.group("msg").strip()))
        elif line.strip():
            parsed.append(("", "INFO", line.strip()))
    if not parsed:
        return "<p class='sub'>Sin alertas registradas.</p>"
    # deduplica consecutivas iguales -> cuenta
    groups: list[tuple[str, str, str, int]] = []
    for ts, lvl, msg in parsed:
        if groups and groups[-1][1] == lvl and groups[-1][2] == msg:
            groups[-1] = (ts, lvl, msg, groups[-1][3] + 1)
        else:
            groups.append((ts, lvl, msg, 1))
    groups = groups[-25:]  # últimas 25
    rows = []
    for ts, lvl, msg, cnt in reversed(groups):  # más reciente primero
        short_ts = ts[5:16].replace("T", " ") if ts else ""
        cnt_html = f"<span class='cnt'>×{cnt}</span>" if cnt > 1 else ""
        rows.append(
            f"<div class='alrow'><span class='ts'>{_esc(short_ts)}</span>"
            f"<span class='lvl lvl-{html.escape(lvl)}'>{_esc(lvl)}</span>"
            f"<span class='msg'>{_esc(msg)}</span>{cnt_html}</div>"
        )
    return "<div class='tablewrap'>" + "".join(rows) + "</div>"


def _progress_panel(cfg, signals_dir: Path, blind: dict) -> str:
    """Progreso REAL hacia los umbrales GO: días de paper y operaciones OOS medidas."""
    min_days = int(cfg.get("go_no_go.paper_min_days", 90))
    min_trades = int(cfg.get("go_no_go.min_trades_oos", 100))
    days = 0
    h = signals_dir / "history.csv"
    if h.exists():
        try:
            days = int(pd.read_csv(h)["date"].nunique())
        except Exception:  # noqa: BLE001
            days = 0
    trades = int(blind.get("n_trades") or 0)
    pd_pct = min(100.0, 100.0 * days / min_days) if min_days else 100.0
    tr_pct = min(100.0, 100.0 * trades / min_trades) if min_trades else 100.0
    return (
        "<div class='panel' style='margin-top:14px'><h3>Progreso real hacia APTO (sin trucar)</h3>"
        f"<div class='prow'><span class='lab'>📅 Días de paper trading acumulados</span>"
        f"<span class='val'>{days} / {min_days}</span></div>"
        f"<div class='bar'><i style='width:{pd_pct:.1f}%'></i></div>"
        f"<div class='prow'><span class='lab'>📈 Operaciones out-of-sample (test ciego)</span>"
        f"<span class='val'>{trades} / {min_trades}</span></div>"
        f"<div class='bar'><i style='width:{tr_pct:.1f}%'></i></div>"
        "<p class='sub'>Estos números se cuentan de verdad (history.csv y test ciego). La compuerta pasa a "
        "APTO solo cuando TODOS los criterios se cumplen con datos reales; no se editan a mano.</p></div>"
    )


FUNNEL_LABELS = {
    "candidatos": "🧺 Candidatos (universo)",
    "kill_switch": "⛔ Kill switch activo",
    "bloqueo_riesgo": "🛡️ Bloqueo gestión de riesgo (DD/correlación)",
    "datos_invalidos": "🩺 Datos inválidos / no disponibles",
    "sin_features": "🧮 Sin features para la fecha",
    "modelo_no_disponible": "🤖 Modelo no disponible",
    "filtro_contexto": "📅 Filtro de contexto (FOMC/CPI/NFP…)",
    "regimen_desfavorable": "📉 Régimen desfavorable",
    "umbral_no_alcanzado": "🎯 Umbral prob/meta no alcanzado",
    "sl_tp_invalidos": "📏 SL/TP inválidos (RR)",
    "emitidas": "✅ SEÑALES EMITIDAS (paper)",
}


def _funnel_panel(fj: dict | None) -> str:
    """Panel del funnel de filtros: cuántos candidatos sobreviven cada etapa."""
    if not fj or not fj.get("stages"):
        return ""
    stages = fj["stages"]
    mx = max([v for v in stages.values()] + [1])
    rows = ""
    for k, lab in FUNNEL_LABELS.items():
        v = int(stages.get(k, 0))
        w = 100.0 * v / mx
        emph = " style='color:var(--ok);font-weight:700'" if k == "emitidas" else ""
        rows += (
            f"<div class='prow'><span class='lab'{emph}>{lab}</span>"
            f"<span class='val'{emph}>{v}</span></div>"
            f"<div class='bar'><i style='width:{w:.1f}%'></i></div>"
        )
    return (
        f"<div class='panel' style='margin-top:14px'><h3>Funnel de filtros — {fj.get('date', '')} "
        f"(datos {fj.get('data_mode', '')})</h3>{rows}"
        "<p class='sub'>Cada etapa descarta lo que no cumple; ante la duda, NADA. "
        "Solo lo que pasa TODOS los filtros se emite como señal.</p></div>"
    )


def _trades_table(acc: dict | None) -> str:
    """Tabla de resultados por operación (paper): ✅/❌ de cada señal evaluada."""
    if not acc or not acc.get("recent_trades"):
        return ""
    df = pd.DataFrame(acc["recent_trades"])
    if df.empty:
        return ""
    df = df.copy()
    df["resultado"] = df["hit"].map({1: "✅", 0: "❌"})
    df = df[["date", "asset", "direction", "prob", "realized_r", "resultado"]]
    return (
        "<h3 style='color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px'>"
        "Resultados por operación (cada señal evaluada vs resultado real)</h3>"
        + _df_table(df, num_cols=("prob", "realized_r"))
    )


def _top_panel(latest: pd.DataFrame) -> str:
    """Top señales emitidas, mejor primero (rank 1,2,3...) con medalla y color de confianza."""
    if latest is None or latest.empty or "rank" not in latest.columns:
        return ""
    em = latest[latest["direction"].isin(["long", "short"])].copy()
    em["_r"] = pd.to_numeric(em["rank"], errors="coerce")
    em = em.sort_values("_r").head(3)
    if em.empty:
        return ""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    cards = ""
    for _, s in em.iterrows():
        rk = int(pd.to_numeric(s["rank"], errors="coerce") or 0)
        band = str(s.get("band", "rojo"))
        border = {"verde": "var(--ok)", "amarillo": "var(--warn)", "rojo": "var(--bad)"}.get(
            band, "var(--bad)"
        )
        cards += (
            f"<div class='card' style='border-left:4px solid {border}'>"
            f"<span class='medal'>{medals.get(rk, rk)}</span> <b>{_esc(s['asset'])}</b> "
            f"<span class='dir-{str(s['direction']).lower()}'>{str(s['direction']).upper()}</span> "
            f"<span class='band-{band}'>prob {float(s['prob']):.2f}</span> meta {float(s.get('meta_prob', 0)):.2f} "
            f"| SL {s.get('stop')} TP {s.get('tp')} size {s.get('size')}</div>"
        )
    return (
        "<div class='panel' style='margin-bottom:14px'><h3>🥇 Top señales de hoy (mejor primero; umbral de emisión 0.86)</h3>"
        + cards
        + "</div>"
    )


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
    is_go = "APTO" in verdict and "NO APTO" not in verdict
    mode_cls = "warn" if data_mode == "synthetic" else "acc"

    funnel_json = None
    fj = reports / "filter_funnel.json"
    if fj.exists():
        try:
            funnel_json = json.loads(fj.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            funnel_json = None

    acc_json = None
    la = reports / "live_accuracy.json"
    if la.exists():
        try:
            acc_json = json.loads(la.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            acc_json = None

    # KPIs
    kpis = "".join(
        [
            _kpi(
                "Sharpe OOS (test ciego)",
                _fmt(blind.get("sharpe")),
                "ok" if (blind.get("sharpe") or 0) > 1 else "bad",
                "mínimo 1.0",
            ),
            _kpi(
                "Win rate OOS",
                _fmt(blind.get("win_rate"), pct=True),
                "ok" if (blind.get("win_rate") or 0) >= 0.70 else "warn",
                "objetivo ≥70%",
            ),
            _kpi(
                "Profit factor",
                _fmt(blind.get("profit_factor")),
                "ok" if (blind.get("profit_factor") or 0) > 1.3 else "bad",
                "mínimo 1.3",
            ),
            _kpi(
                "Max drawdown",
                _fmt(blind.get("max_drawdown"), pct=True),
                "ok" if abs(blind.get("max_drawdown") or 0) < 0.15 else "bad",
                "límite 15%",
            ),
            _kpi(
                "Operaciones OOS",
                blind.get("n_trades", 0),
                "ok" if (blind.get("n_trades") or 0) >= 100 else "bad",
                "mínimo 100",
            ),
            _kpi(
                "PBO (CSCV)", _fmt(rob.get("pbo")), "ok" if (rob.get("pbo") or 1) < 0.5 else "bad", "máx 0.5"
            ),
            _kpi(
                "p-val Deflated Sharpe",
                _fmt(rob.get("deflated_p")),
                "ok" if (rob.get("deflated_p") or 1) < 0.05 else "bad",
                "máx 0.05",
            ),
            _kpi(
                "Prob. ruina (MC)",
                _fmt(mc.get("p_ruin"), pct=True),
                "ok" if (mc.get("p_ruin") or 1) < 0.01 else "bad",
                "máx 1%",
            ),
            _kpi(
                "Leakage AUC",
                _fmt(rob.get("leakage_auc")),
                "ok" if abs((rob.get("leakage_auc") or 0.5) - 0.5) < 0.05 else "warn",
                "≈0.5 = sin fuga",
            ),
            _kpi("Configs probadas", rob.get("n_configs", 1), "", "penalizadas en DSR"),
        ]
    )

    # señales de hoy
    latest = pd.DataFrame()
    lpath = signals_dir / "latest.csv"
    if lpath.exists():
        latest = pd.read_csv(lpath)
    n_sig = int((latest["direction"] != "nada").sum()) if not latest.empty else 0
    sig_cols = [
        c
        for c in (
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
        )
        if c in latest.columns
    ]
    sig_table = (
        _df_table(
            latest[sig_cols] if sig_cols else latest,
            num_cols=("prob", "meta_prob", "agreement", "entry_ref", "stop", "tp", "size"),
        )
        if not latest.empty
        else "<p class='sub'>Ejecuta <code>python -m src.run --mode paper</code> para generar señales.</p>"
    )

    # gráficos
    charts_html = ""
    for name, caption in (
        ("equity_curve.png", "Curva de equity (walk-forward OOS)"),
        ("drawdown.png", "Drawdown"),
        ("accuracy_by_threshold.png", "Precisión por umbral de confianza"),
    ):
        p = reports / name
        if p.exists():
            charts_html += f"<figure><img src='{_img_b64(p)}' alt='{caption}'><figcaption>{caption}</figcaption></figure>"

    # alertas recientes (panel con chips de nivel y deduplicado)
    alerts_html = _alerts_panel(reports)

    # informe completo (report.md → HTML)
    report_html = "<p class='sub'>Ejecuta <code>python -m src.backtest --walk-forward --report</code> para generar el informe.</p>"
    rpath = root / "report.md"
    if rpath.exists():
        report_html = f"<div class='report'>{md_to_html(rpath.read_text(encoding='utf-8'))}</div>"

    # precisión por umbral del backtest (si existe)
    acc_bt = reports / "live_accuracy_by_threshold.csv"
    acc_bt_html = ""
    if acc_bt.exists():
        acc_bt_html = (
            "<h2><span class='n'>05</span>Precisión por umbral (backtest OOS)<span class='hint'>reports/live_accuracy_by_threshold.csv</span></h2>"
            + _df_table(pd.read_csv(acc_bt))
        )

    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>quant-signal-engine — Dashboard (PAPER TRADING)</title>
<style>{CSS}</style></head>
<body>
<header>
  <div class="hbar">
    <div class="logo"><span class="bolt">⚡</span><span>quant-signal-engine<small>señales diarias · solo paper trading</small></span></div>
    <div class="spacer"></div>
    <span class="badge acc">MODO PAPER</span>
    <span class="badge {mode_cls}">DATOS: {_esc(data_mode.upper())}{' ⚠ sintéticos' if data_mode == 'synthetic' else ''}</span>
    <span class="badge dim">señales hoy: {n_sig}</span>
    <span class="badge dim">{now}</span>
  </div>
  <div class="banner {'go' if is_go else 'bad'}"><div class="inner">
    <span class="verdict">Veredicto GO/NO-GO: {_esc(verdict)}</span>
    <span class="meta">decidido por la compuerta con datos out-of-sample · test ciego intocable · generado {now}</span>
  </div></div>
</header>
<main>
<section><h2><span class="n">01</span>KPIs out-of-sample (test ciego)<span class="hint">umbrales mínimos entre paréntesis</span></h2>
<div class="grid">{kpis}</div></section>

<section><h2><span class="n">02</span>Señales de hoy + acertabilidad en vivo<span class="hint">signals/latest.csv · precisión medida contra resultados reales</span></h2>
<div class="two">
  <div>{_top_panel(latest)}<h3 style="color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px">Señales (PAPER)</h3>{sig_table}</div>
  <div>{_live_accuracy_panel(acc_json)}{_trades_table(acc_json)}{_funnel_panel(funnel_json)}</div>
</div></section>

<section><h2><span class="n">03</span>Compuerta GO / NO-GO</h2>
<div class="two">
  <div>{_df_table(gonogo_df)}{_progress_panel(cfg, signals_dir, blind)}</div>
  {_go_checklist(gonogo_df)}
</div></section>

<section><h2><span class="n">04</span>Gráficos</h2>
<div class="charts">{charts_html or "<p class='sub'>Sin gráficos todavía: ejecuta el backtest con --report.</p>"}</div></section>

{acc_bt_html}

<section><h2><span class="n">06</span>Alertas recientes<span class="hint">reports/alerts.log</span></h2>{alerts_html}</section>

<section><h2><span class="n">07</span>Informe completo<span class="hint">report.md</span></h2>{report_html}</section>
</main>
<footer>⚡ quant-signal-engine · SOLO PAPER TRADING (no envía órdenes reales)<br>
Esto no es asesoría financiera ni garantiza rentabilidad.<br>
Dashboard estático auto-contenido generado automáticamente — ábrelo con doble clic, sin servidor.</footer>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path
