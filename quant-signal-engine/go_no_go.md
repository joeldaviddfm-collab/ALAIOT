# Compuerta GO / NO-GO

**Veredicto: NO APTO PARA CAPITAL REAL**

Medido en el test ciego (y, cuando hay, en paper trading). Si CUALQUIER criterio falla, el sistema no es apto.

| Criterio | Umbral mínimo | Medido | ¿Cumple? |
|---|---|---|---|
| Expectativa por operación (costos x2) | > 0.0 | 0.0010 | ✅ |
| Profit factor | > 1.3 | 1.6000 | ✅ |
| Sharpe out-of-sample (test ciego) | > 1.0 | 1.2000 | ✅ |
| Deflated Sharpe p-value vs azar | < 0.05 | 0.0100 | ✅ |
| PBO (CSCV) | < 0.5 | 0.2000 | ✅ |
| Max drawdown | < 15% (en valor absoluto) | -9.00% | ✅ |
| Nº mínimo de operaciones OOS | >= 100 | 30 | ❌ |
| Supera a buy & hold ajustada por riesgo | Sí | Sí | ✅ |
| Paper trading >= 90 días dentro de IC del backtest | >= 90 días | 0 días, dentro IC: False | ❌ |

_Esto no es asesoría financiera ni garantiza rentabilidad._
