# Informe de resultados — quant-signal-engine

> **Esto no es asesoría financiera ni garantiza rentabilidad.** Solo modo PAPER TRADING; no se envían órdenes reales.

## ⚠️ VEREDICTO: NO ROBUSTO / NO APTO

El sistema **no cumple** uno o más criterios de la compuerta GO/NO-GO (ver go_no_go.md). Los resultados de abajo se reportan sin maquillar.

- Modo de datos de este informe: **synthetic** (⚠️ SINTÉTICO: demo/CI, no datos de mercado reales)
- Fecha de generación: 2026-11-02 01:11 UTC
- Configuraciones/modelos probados en total: **12** (se usa en el Deflated Sharpe)
- Semilla fija: 42

## 1. Resultados por split (walk-forward con embargo; test ciego intocable)

### Train (in-sample, solo referencia)

| Métrica | Valor |
|---|---|
| Operaciones | 10 |
| Win rate | 50.00% |
| Expectativa por op. | 0.0010 |
| Profit factor | 1.1000 |
| Avg R-multiple | n/a |
| Sharpe | 0.5000 |
| Sortino | 0.6000 |
| Calmar | 0.2000 |
| Max drawdown | -5.00% |
| CAGR | 5.00% |
| Volatilidad anual | 10.00% |

### Validación walk-forward (out-of-sample)

| Métrica | Valor |
|---|---|
| Operaciones | 20 |
| Win rate | 55.00% |
| Expectativa por op. | 0.0020 |
| Profit factor | 1.4000 |
| Avg R-multiple | n/a |
| Sharpe | 0.8000 |
| Sortino | n/a |
| Calmar | n/a |
| Max drawdown | -7.00% |
| CAGR | n/a |
| Volatilidad anual | n/a |

### Test ciego (out-of-sample final)

| Métrica | Valor |
|---|---|
| Operaciones | 30 |
| Win rate | 60.00% |
| Expectativa por op. | 0.0030 |
| Profit factor | 1.6000 |
| Avg R-multiple | n/a |
| Sharpe | 1.2000 |
| Sortino | 1.4000 |
| Calmar | 1.0000 |
| Max drawdown | -9.00% |
| CAGR | 12.00% |
| Volatilidad anual | 12.00% |

### Comparación: baseline y buy & hold (mismo periodo OOS)

| Métrica | Valor |
|---|---|
| Operaciones | 0 |
| Win rate | n/a |
| Expectativa por op. | n/a |
| Profit factor | n/a |
| Avg R-multiple | n/a |
| Sharpe | 0.4000 |
| Sortino | n/a |
| Calmar | n/a |
| Max drawdown | -20.00% |
| CAGR | n/a |
| Volatilidad anual | n/a |

| Métrica | Valor |
|---|---|
| Operaciones | 0 |
| Win rate | n/a |
| Expectativa por op. | n/a |
| Profit factor | n/a |
| Avg R-multiple | n/a |
| Sharpe | 0.6000 |
| Sortino | n/a |
| Calmar | n/a |
| Max drawdown | -25.00% |
| CAGR | n/a |
| Volatilidad anual | n/a |


## 2. Robustez (anti-suerte / anti-overfitting)

- p-value vs permutación de operaciones: **0.01**
- p-value vs 1000 estrategias aleatorias (misma frecuencia): **0.02**
- Deflated Sharpe: **2.5** (p = 0.01)
- PBO (CSCV): **0.2**
- Monte Carlo (bootstrap bloques, 2000 sims): retorno p50 = 0.2, IC 90% = [-0.1, 0.6], maxDD p95 = -0.2, **prob. de ruina = 0.001**
- Test de fuga (etiquetas barajadas, AUC esperado ~0.5): **0.51**
- Stress de costos x2: expectativa = 0.001, Sharpe = 0.9 | x3: expectativa = -0.001, Sharpe = 0.5

### Stress histórico (max drawdown en cada ventana)

| Ventana | Max DD |
|---|---|
| COVID 2020 | -12.00% |

### Ablation study (Sharpe OOS quitando cada grupo de features)

| Grupo quitado | Sharpe OOS | Δ vs completo |
|---|---|---|
| momentum | 1.1000 | -0.1000 |
| macro | 1.2500 | 0.0500 |

### Sensibilidad de hiperparámetros (±20%)

| Parámetro | Valor | Sharpe OOS |
|---|---|---|
| max_depth | 3 | 1.1000 |
| max_depth | 5 | 1.3000 |

### Validación cruzada entre activos (entrena sin cripto, prueba en cripto)

| Métrica | Valor |
|---|---|
| Operaciones | 10 |
| Win rate | n/a |
| Expectativa por op. | n/a |
| Profit factor | n/a |
| Avg R-multiple | n/a |
| Sharpe | 0.5000 |
| Sortino | n/a |
| Calmar | n/a |
| Max drawdown | n/a |
| CAGR | n/a |
| Volatilidad anual | n/a |


## 3. Precisión por umbral de confianza (señales emitidas)

| bucket    |   n |   accuracy |
|:----------|----:|-----------:|
| 0.70-0.75 |  10 |        0.6 |
| >=0.90    |   5 |        0.8 |

## 4. Resultados por activo (test ciego)

| asset   |   n |   accuracy |
|:--------|----:|-----------:|
| BTC-USD |  15 |        0.6 |
| SPY     |  15 |        0.6 |

## 5. Resultados por régimen de mercado (test ciego)

|   regime_id |   n |   accuracy |
|------------:|----:|-----------:|
|           0 |   5 |        0.4 |
|           1 |  10 |        0.6 |
|           2 |  15 |        0.7 |

## 6. Cómo podría fallar este sistema (escenarios)

1. **Régimen nuevo no visto**: un crash sistémico distinto a los del histórico (ej. crisis de deuda soberana) puede invalidar los patrones aprendidos.
2. **Costos reales mayores a los modelados**: comisiones, spread en horas de baja liquidez o slippage en gaps nocturnos pueden erosionar la expectativa (ver stress x2/x3).
3. **Data snooping acumulado**: aunque se usa walk-forward + embargo + test ciego, haber mirado resultados intermedios al diseñar el sistema infla el rendimiento aparente (por eso se reporta Deflated Sharpe y PBO).
4. **Fallo de fuentes de datos**: si una API cae y el fallback también, el activo queda en NADA; si fallan todas, no hay señales (fail-safe), pero el sistema queda ciego.
5. **Look-ahead residual**: rezagos de publicación mal estimados (ej. revisiones de datos macro) podrían filtrar información futura; mitigado con publication_lags conservadores y test de fuga.
6. **Overfitting del meta-modelo**: si el primario mejora en validación pero el meta filtra todo, no hay señales; si el meta es permisivo, puede dejar pasar señales malas.
7. **Criptos en bear market extremo**: drawdowns > 60% (2018, 2022) pueden superar el circuit breaker del 15% y pausar el sistema durante meses.
8. **Eventos de cola (flash crash, halts de mercado)**: el slippage real puede ser muy superior al modelado; los stops pueden ejecutarse muy por debajo.
9. **Cambios regulatorios** (prohibiciones, impuestos) que alteren la estructura del mercado de un día para otro.
10. **Deriva de concepto (concept drift)**: la relación entre features y resultados cambia con el tiempo; sin reentrenamiento frecuente el modelo degrada (mitigado con PSI/KS y champion/challenger).
11. **Falta de diversificación real**: correlaciones que suben a ~1 en crisis reducen el beneficio de tener varias posiciones.
12. **Errores operativos**: doble ejecución, señales duplicadas o datos corruptos; mitigados con idempotencia y validación, pero no imposibles.

## 7. Limitaciones

- Los datos gratuitos tienen latencia y a veces calidad limitada (revisiones, huecos, ajustes de splits/dividendos).
- El backtest asume ejecución al open siguiente con slippage fijo; la realidad puede diferir (gap, liquidez).
- No se modela latencia de la señal ni rechazos de órdenes (no hay órdenes reales en paper).
- El calendario económico es una aproximación (FOMC programados, NFP 1er viernes, CPI estimado).
- FinBERT es opcional; sin él se usa un lexicón propio (menos potente).

_Esto no es asesoría financiera ni garantiza rentabilidad._
