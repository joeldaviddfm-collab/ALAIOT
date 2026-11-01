"""Genera report.md (métricas completas por split), go_no_go.md (compuerta) y gráficos.

TODO se reporta con honestidad: intervalos de confianza, nº de configuraciones probadas,
y si el resultado no es robusto se dice en la primera página.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISCLAIMER = "Esto no es asesoría financiera ni garantiza rentabilidad."


def _fmt(x, pct=False, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "inf" if np.isinf(x) else "n/a"
    return f"{x:.2%}" if pct else f"{x:.{nd}f}"


def _metrics_table(m: dict) -> str:
    if not m:
        return "_sin datos_\n"
    rows = [
        ("Operaciones", m.get("n_trades", 0)),
        ("Win rate", _fmt(m.get("win_rate"), pct=True)),
        ("Expectativa por op.", _fmt(m.get("expectancy"))),
        ("Profit factor", _fmt(m.get("profit_factor"))),
        ("Avg R-multiple", _fmt(m.get("avg_r_multiple"))),
        ("Sharpe", _fmt(m.get("sharpe"))),
        ("Sortino", _fmt(m.get("sortino"))),
        ("Calmar", _fmt(m.get("calmar"))),
        ("Max drawdown", _fmt(m.get("max_drawdown"), pct=True)),
        ("CAGR", _fmt(m.get("cagr"), pct=True)),
        ("Volatilidad anual", _fmt(m.get("volatility"), pct=True)),
    ]
    return "| Métrica | Valor |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows) + "\n"


def evaluate_go_no_go(blind: dict, paper: dict | None, extra: dict, cfg) -> list[dict]:
    g = cfg.get("go_no_go", {})
    dsr_p = extra.get("deflated_p", 1.0)
    pbo = extra.get("pbo", 1.0)
    n_trades = int(blind.get("n_trades", 0))
    sharpe = float(blind.get("sharpe", 0.0))
    pf = float(blind.get("profit_factor", 0.0)) if np.isfinite(blind.get("profit_factor", 0)) else 999.0
    maxdd = float(blind.get("max_drawdown", 0.0))
    exp_x2 = extra.get("expectancy_costs_x2")
    beats_bh = extra.get("beats_bh_risk_adjusted", False)
    paper_days = (paper or {}).get("days", 0)
    paper_ok = (paper or {}).get("within_ci", False)
    criteria = [
        {
            "criterio": "Expectativa por operación (costos x2)",
            "umbral": f"> {g.get('expectancy_x2_min', 0)}",
            "medido": _fmt(exp_x2),
            "ok": bool(exp_x2 is not None and exp_x2 > g.get("expectancy_x2_min", 0)),
        },
        {
            "criterio": "Profit factor",
            "umbral": f"> {g.get('profit_factor_min', 1.3)}",
            "medido": _fmt(pf),
            "ok": bool(pf > g.get("profit_factor_min", 1.3)),
        },
        {
            "criterio": "Sharpe out-of-sample (test ciego)",
            "umbral": f"> {g.get('sharpe_oos_min', 1.0)}",
            "medido": _fmt(sharpe),
            "ok": bool(sharpe > g.get("sharpe_oos_min", 1.0)),
        },
        {
            "criterio": "Deflated Sharpe p-value vs azar",
            "umbral": f"< {g.get('p_value_max', 0.05)}",
            "medido": _fmt(dsr_p),
            "ok": bool(dsr_p < g.get("p_value_max", 0.05)),
        },
        {
            "criterio": "PBO (CSCV)",
            "umbral": f"< {g.get('pbo_max', 0.5)}",
            "medido": _fmt(pbo),
            "ok": bool(pbo < g.get("pbo_max", 0.5)),
        },
        {
            "criterio": "Max drawdown",
            "umbral": f"< {g.get('max_dd_max', 0.15):.0%} (en valor absoluto)",
            "medido": _fmt(maxdd, pct=True),
            "ok": bool(abs(maxdd) < g.get("max_dd_max", 0.15)),
        },
        {
            "criterio": "Nº mínimo de operaciones OOS",
            "umbral": f">= {g.get('min_trades_oos', 100)}",
            "medido": n_trades,
            "ok": n_trades >= g.get("min_trades_oos", 100),
        },
        {
            "criterio": "Supera a buy & hold ajustada por riesgo",
            "umbral": "Sí",
            "medido": "Sí" if beats_bh else "No",
            "ok": bool(beats_bh),
        },
        {
            "criterio": "Paper trading >= 90 días dentro de IC del backtest",
            "umbral": f">= {g.get('paper_min_days', 90)} días",
            "medido": f"{paper_days} días, dentro IC: {paper_ok}",
            "ok": bool(paper_days >= g.get("paper_min_days", 90) and paper_ok),
        },
    ]
    return criteria


def write_go_no_go(cfg, criteria: list[dict], path: Path) -> bool:
    all_ok = all(c["ok"] for c in criteria)
    verdict = "APTO" if all_ok else "NO APTO PARA CAPITAL REAL"
    lines = [
        "# Compuerta GO / NO-GO",
        "",
        f"**Veredicto: {verdict}**",
        "",
        "Medido en el test ciego (y, cuando hay, en paper trading). Si CUALQUIER criterio falla, el sistema no es apto.",
        "",
        "| Criterio | Umbral mínimo | Medido | ¿Cumple? |",
        "|---|---|---|---|",
    ]
    for c in criteria:
        lines.append(f"| {c['criterio']} | {c['umbral']} | {c['medido']} | {'✅' if c['ok'] else '❌'} |")
    lines += ["", f"_{DISCLAIMER}_", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return all_ok


def _charts(reports_dir: Path, bundle: dict) -> list[str]:
    paths = []
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
    # 1) curvas de equity
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, eq in (
        ("estrategia (OOS)", bundle.get("equity_val")),
        ("baseline", bundle.get("equity_base")),
        ("buy & hold", bundle.get("equity_bh")),
    ):
        if eq is not None and len(eq):
            ax.plot(pd.Series(eq).index, pd.Series(eq).values, label=name, lw=1.4)
    if bundle.get("equity_blind") is not None and len(bundle["equity_blind"]):
        ax.plot(
            pd.Series(bundle["equity_blind"]).index,
            pd.Series(bundle["equity_blind"]).values,
            label="estrategia (test ciego)",
            lw=1.8,
        )
    ax.set_title("Curva de equity (paper/backtest)")
    ax.legend()
    ax.grid(alpha=0.3)
    p = reports_dir / "equity_curve.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))
    # 2) drawdown
    eq = bundle.get("equity_blind")
    if eq is None:
        eq = bundle.get("equity_val")
    if eq is not None and len(eq):
        s = pd.Series(eq)
        dd = s / s.cummax() - 1
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.fill_between(dd.index, dd.values, 0, alpha=0.5)
        ax.set_title("Drawdown (test ciego)")
        ax.grid(alpha=0.3)
        p = reports_dir / "drawdown.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(p))
    # 3) precisión por umbral
    acc = bundle.get("accuracy_by_threshold")
    if acc is not None and len(acc):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(acc["bucket"].astype(str), acc["accuracy"], color="steelblue")
        ax.axhline(0.5, color="red", ls="--", lw=1, label="azar (0.5)")
        ax.set_ylim(0, 1)
        ax.set_title("Precisión por umbral de confianza (señales con prob >= umbral)")
        ax.legend()
        ax.grid(alpha=0.3)
        p = reports_dir / "accuracy_by_threshold.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(p))
    return paths


def write_full_report(cfg, bundle: dict) -> dict:
    """Escribe report.md, go_no_go.md, metrics.json y gráficos en reports/."""
    reports_dir = cfg.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    blind = bundle.get("blind", {})
    val = bundle.get("val", {})
    train = bundle.get("train", {})
    base = bundle.get("baseline", {})
    bh = bundle.get("buy_hold", {})
    extra = bundle.get("extra", {})
    criteria = evaluate_go_no_go(blind, bundle.get("paper"), extra, cfg)
    go_ok = write_go_no_go(cfg, criteria, cfg.root / "go_no_go.md")
    charts = _charts(reports_dir, bundle)

    lines = [
        "# Informe de resultados — quant-signal-engine",
        "",
        f"> **{DISCLAIMER}** Solo modo PAPER TRADING; no se envían órdenes reales.",
        "",
    ]
    if not go_ok:
        lines += [
            "## ⚠️ VEREDICTO: NO ROBUSTO / NO APTO",
            "",
            "El sistema **no cumple** uno o más criterios de la compuerta GO/NO-GO (ver go_no_go.md). "
            "Los resultados de abajo se reportan sin maquillar.",
            "",
        ]
    else:
        lines += ["## Veredicto: cumple la compuerta GO/NO-GO (ver go_no_go.md)", ""]
    lines += [
        f"- Modo de datos de este informe: **{bundle.get('data_mode', 'real')}**"
        + (
            " (⚠️ SINTÉTICO: demo/CI, no datos de mercado reales)"
            if bundle.get("data_mode") == "synthetic"
            else ""
        ),
        f"- Fecha de generación: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Configuraciones/modelos probados en total: **{extra.get('n_configs', 1)}** (se usa en el Deflated Sharpe)",
        f"- Semilla fija: {cfg.seed}",
        "",
        "## 1. Resultados por split (walk-forward con embargo; test ciego intocable)",
        "",
        "### Train (in-sample, solo referencia)",
        "",
        _metrics_table(train),
        "### Validación walk-forward (out-of-sample)",
        "",
        _metrics_table(val),
        "### Test ciego (out-of-sample final)",
        "",
        _metrics_table(blind),
        "### Comparación: baseline y buy & hold (mismo periodo OOS)",
        "",
        _metrics_table(base),
        _metrics_table(bh),
        "",
        "## 2. Robustez (anti-suerte / anti-overfitting)",
        "",
        f"- p-value vs permutación de operaciones: **{extra.get('perm_p', 'n/a')}**",
        f"- p-value vs {extra.get('n_random', 1000)} estrategias aleatorias (misma frecuencia): **{extra.get('random_p', 'n/a')}**",
        f"- Deflated Sharpe: **{extra.get('dsr', 'n/a')}** (p = {extra.get('deflated_p', 'n/a')})",
        f"- PBO (CSCV): **{extra.get('pbo', 'n/a')}**",
        f"- Monte Carlo (bootstrap bloques, {extra.get('mc', {}).get('n_sims', 0)} sims): retorno p50 = {extra.get('mc', {}).get('ret_p50', 'n/a')}, "
        f"IC 90% = [{extra.get('mc', {}).get('ret_p05', 'n/a')}, {extra.get('mc', {}).get('ret_p95', 'n/a')}], "
        f"maxDD p95 = {extra.get('mc', {}).get('maxdd_p95', 'n/a')}, **prob. de ruina = {extra.get('mc', {}).get('p_ruin', 'n/a')}**",
        f"- Test de fuga (etiquetas barajadas, AUC esperado ~0.5): **{extra.get('leakage_auc', 'n/a')}**",
        f"- Stress de costos x2: expectativa = {extra.get('exp_x2', 'n/a')}, Sharpe = {extra.get('sharpe_x2', 'n/a')} | "
        f"x3: expectativa = {extra.get('exp_x3', 'n/a')}, Sharpe = {extra.get('sharpe_x3', 'n/a')}",
        "",
        "### Stress histórico (max drawdown en cada ventana)",
        "",
        "| Ventana | Max DD |",
        "|---|---|",
    ]
    for name, dd in extra.get("stress", {}).items():
        lines.append(f"| {name} | {_fmt(dd, pct=True)} |")
    lines += [
        "",
        "### Ablation study (Sharpe OOS quitando cada grupo de features)",
        "",
        "| Grupo quitado | Sharpe OOS | Δ vs completo |",
        "|---|---|---|",
    ]
    full_sharpe = extra.get("ablation", {}).get("__full__", float("nan"))
    for grp, sr in extra.get("ablation", {}).items():
        if grp == "__full__":
            continue
        delta = sr - full_sharpe if np.isfinite(sr) and np.isfinite(full_sharpe) else float("nan")
        lines.append(f"| {grp} | {_fmt(sr)} | {_fmt(delta)} |")
    lines += [
        "",
        "### Sensibilidad de hiperparámetros (±20%)",
        "",
        "| Parámetro | Valor | Sharpe OOS |",
        "|---|---|---|",
    ]
    for param, vals in extra.get("sensitivity", {}).items():
        for v, sr in vals:
            lines.append(f"| {param} | {v} | {_fmt(sr)} |")
    lines += [
        "",
        "### Validación cruzada entre activos (entrena sin cripto, prueba en cripto)",
        "",
        _metrics_table(extra.get("cross_asset", {})),
        "",
        "## 3. Precisión por umbral de confianza (señales emitidas)",
        "",
    ]
    acc = bundle.get("accuracy_by_threshold")
    if acc is not None and len(acc):
        lines.append(acc.to_markdown(index=False))
    else:
        lines.append("_Sin señales suficientes para evaluar por umbral._")
    lines += [
        "",
        "## 4. Resultados por activo (test ciego)",
        "",
    ]
    per_asset = bundle.get("per_asset")
    if per_asset is not None and len(per_asset):
        lines.append(per_asset.to_markdown(index=False))
    else:
        lines.append("_Sin desglose por activo._")
    lines += [
        "",
        "## 5. Resultados por régimen de mercado (test ciego)",
        "",
    ]
    per_regime = bundle.get("per_regime")
    if per_regime is not None and len(per_regime):
        lines.append(per_regime.to_markdown(index=False))
    else:
        lines.append("_Sin desglose por régimen._")
    lines += [
        "",
        "## 6. Cómo podría fallar este sistema (escenarios)",
        "",
        "1. **Régimen nuevo no visto**: un crash sistémico distinto a los del histórico (ej. crisis de deuda soberana) puede invalidar los patrones aprendidos.",
        "2. **Costos reales mayores a los modelados**: comisiones, spread en horas de baja liquidez o slippage en gaps nocturnos pueden erosionar la expectativa (ver stress x2/x3).",
        "3. **Data snooping acumulado**: aunque se usa walk-forward + embargo + test ciego, haber mirado resultados intermedios al diseñar el sistema infla el rendimiento aparente (por eso se reporta Deflated Sharpe y PBO).",
        "4. **Fallo de fuentes de datos**: si una API cae y el fallback también, el activo queda en NADA; si fallan todas, no hay señales (fail-safe), pero el sistema queda ciego.",
        "5. **Look-ahead residual**: rezagos de publicación mal estimados (ej. revisiones de datos macro) podrían filtrar información futura; mitigado con publication_lags conservadores y test de fuga.",
        "6. **Overfitting del meta-modelo**: si el primario mejora en validación pero el meta filtra todo, no hay señales; si el meta es permisivo, puede dejar pasar señales malas.",
        "7. **Criptos en bear market extremo**: drawdowns > 60% (2018, 2022) pueden superar el circuit breaker del 15% y pausar el sistema durante meses.",
        "8. **Eventos de cola (flash crash, halts de mercado)**: el slippage real puede ser muy superior al modelado; los stops pueden ejecutarse muy por debajo.",
        "9. **Cambios regulatorios** (prohibiciones, impuestos) que alteren la estructura del mercado de un día para otro.",
        "10. **Deriva de concepto (concept drift)**: la relación entre features y resultados cambia con el tiempo; sin reentrenamiento frecuente el modelo degrada (mitigado con PSI/KS y champion/challenger).",
        "11. **Falta de diversificación real**: correlaciones que suben a ~1 en crisis reducen el beneficio de tener varias posiciones.",
        "12. **Errores operativos**: doble ejecución, señales duplicadas o datos corruptos; mitigados con idempotencia y validación, pero no imposibles.",
        "",
        "## 7. Limitaciones",
        "",
        "- Los datos gratuitos tienen latencia y a veces calidad limitada (revisiones, huecos, ajustes de splits/dividendos).",
        "- El backtest asume ejecución al open siguiente con slippage fijo; la realidad puede diferir (gap, liquidez).",
        "- No se modela latencia de la señal ni rechazos de órdenes (no hay órdenes reales en paper).",
        "- El calendario económico es una aproximación (FOMC programados, NFP 1er viernes, CPI estimado).",
        "- FinBERT es opcional; sin él se usa un lexicón propio (menos potente).",
        "",
        f"_{DISCLAIMER}_",
        "",
    ]
    (cfg.root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (reports_dir / "metrics.json").write_text(
        json.dumps(bundle.get("metrics_json", {}), indent=2, default=str), encoding="utf-8"
    )
    log = __import__("structlog").get_logger("qse.report")
    log.info("report_written", charts=charts, go=go_ok)
    return {"go_ok": go_ok, "criteria": criteria, "charts": charts}
