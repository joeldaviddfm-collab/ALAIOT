# Informe de resultados — quant-signal-engine

> **Esto no es asesoría financiera ni garantiza rentabilidad.** Solo modo PAPER TRADING; no se envían órdenes reales.

## ⚠️ VEREDICTO: NO ROBUSTO / NO APTO

El sistema **no cumple** uno o más criterios de la compuerta GO/NO-GO (ver go_no_go.md). Los resultados de abajo se reportan sin maquillar.

- Modo de datos de este informe: **synthetic** (⚠️ SINTÉTICO: demo/CI, no datos de mercado reales)
- Fecha de generación: 2026-11-01 16:22 UTC
- Configuraciones/modelos probados en total: **43** (se usa en el Deflated Sharpe)
- Semilla fija: 42

## 1. Resultados por split (walk-forward con embargo; test ciego intocable)

### Train (in-sample, solo referencia)

| Métrica | Valor |
|---|---|
| Operaciones | 0 |
| Win rate | n/a |
| Expectativa por op. | n/a |
| Profit factor | n/a |
| Avg R-multiple | n/a |
| Sharpe | 0.0000 |
| Sortino | 0.0000 |
| Calmar | inf |
| Max drawdown | 0.00% |
| CAGR | 0.00% |
| Volatilidad anual | 0.00% |

### Validación walk-forward (out-of-sample)

| Métrica | Valor |
|---|---|
| Operaciones | 185 |
| Win rate | 47.57% |
| Expectativa por op. | 2.1117 |
| Profit factor | 1.0402 |
| Avg R-multiple | 0.0026 |
| Sharpe | 0.0056 |
| Sortino | 0.0022 |
| Calmar | -0.0005 |
| Max drawdown | -5.33% |
| CAGR | -0.00% |
| Volatilidad anual | 1.46% |

### Test ciego (out-of-sample final)

| Métrica | Valor |
|---|---|
| Operaciones | 0 |
| Win rate | n/a |
| Expectativa por op. | n/a |
| Profit factor | n/a |
| Avg R-multiple | n/a |
| Sharpe | 0.0000 |
| Sortino | 0.0000 |
| Calmar | inf |
| Max drawdown | 0.00% |
| CAGR | 0.00% |
| Volatilidad anual | 0.00% |

### Comparación: baseline y buy & hold (mismo periodo OOS)

| Métrica | Valor |
|---|---|
| Operaciones | 0 |
| Win rate | n/a |
| Expectativa por op. | n/a |
| Profit factor | n/a |
| Avg R-multiple | n/a |
| Sharpe | 0.4140 |
| Sortino | 0.4206 |
| Calmar | 0.3859 |
| Max drawdown | -3.65% |
| CAGR | 1.41% |
| Volatilidad anual | 3.52% |

| Métrica | Valor |
|---|---|
| Operaciones | 0 |
| Win rate | n/a |
| Expectativa por op. | n/a |
| Profit factor | n/a |
| Avg R-multiple | n/a |
| Sharpe | 1.2332 |
| Sortino | 1.3639 |
| Calmar | 2.0457 |
| Max drawdown | -11.15% |
| CAGR | 22.81% |
| Volatilidad anual | 17.96% |


## 2. Robustez (anti-suerte / anti-overfitting)

- p-value vs permutación de operaciones: **1.0**
- p-value vs 1000 estrategias aleatorias (misma frecuencia): **1.0**
- Deflated Sharpe: **0.0** (p = 0.9999999701976776)
- PBO (CSCV): **0.4**
- Monte Carlo (bootstrap bloques, 0 sims): retorno p50 = 0.0, IC 90% = [0.0, 0.0], maxDD p95 = 0.0, **prob. de ruina = 1.0**
- Test de fuga (etiquetas barajadas, AUC esperado ~0.5): **0.49789326924184085**
- Stress de costos x2: expectativa = None, Sharpe = 0.0 | x3: expectativa = None, Sharpe = 0.0

### Stress histórico (max drawdown en cada ventana)

| Ventana | Max DD |
|---|---|
| GFC 2008 | 0.00% |
| COVID 2020 | 0.00% |
| Subida de tasas 2022 | 0.00% |
| Cripto 2018 | 0.00% |
| Cripto 2022 (Luna/FTX) | 0.00% |
| Flash crash 2010 | 0.00% |

### Ablation study (Sharpe OOS quitando cada grupo de features)

| Grupo quitado | Sharpe OOS | Δ vs completo |
|---|---|---|
| asset_meta | 0.0000 | 0.0000 |
| derivatives | 0.0000 | 0.0000 |
| macro | 0.0000 | 0.0000 |
| momentum | 0.0000 | 0.0000 |
| onchain | 0.0000 | 0.0000 |
| regime | 0.0000 | 0.0000 |
| sentiment | 0.0000 | 0.0000 |
| structure_seasonal | 0.0000 | 0.0000 |
| trend | 0.0000 | 0.0000 |
| volatility | 0.0000 | 0.0000 |
| volume | 0.0000 | 0.0000 |

### Sensibilidad de hiperparámetros (±20%)

| Parámetro | Valor | Sharpe OOS |
|---|---|---|
| max_depth | 3.20 | 0.7309 |
| max_depth | 4.80 | 0.0000 |
| num_leaves | 24.80 | 0.0000 |
| num_leaves | 37.20 | 0.0000 |
| n_estimators | 320.00 | 0.0000 |
| n_estimators | 480.00 | 0.0000 |

### Validación cruzada entre activos (entrena sin cripto, prueba en cripto)

| Métrica | Valor |
|---|---|
| Operaciones | 0 |
| Win rate | n/a |
| Expectativa por op. | n/a |
| Profit factor | n/a |
| Avg R-multiple | n/a |
| Sharpe | 0.0000 |
| Sortino | 0.0000 |
| Calmar | inf |
| Max drawdown | 0.00% |
| CAGR | 0.00% |
| Volatilidad anual | 0.00% |


## 3. Precisión por umbral de confianza (señales emitidas)

_Sin señales suficientes para evaluar por umbral._

## 4. Resultados por activo (test ciego)

_Sin desglose por activo._

## 5. Resultados por régimen de mercado (test ciego)

_Sin desglose por régimen._

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
