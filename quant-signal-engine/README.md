# quant-signal-engine

Motor de **señales de trading diarias** construido como un sistema cuantitativo completo:
múltiples fuentes de datos gratuitas, features sin look-ahead, etiquetado **triple barrier**,
**meta-labeling**, calibración de probabilidades, validación **walk-forward con purging y
embargo**, test ciego intocable, gestión de riesgo estricta y un bloque de robustez
anti-overfitting. **Funciona SOLO en modo PAPER TRADING: no envía ninguna orden real.**

> **Esto no es asesoría financiera ni garantiza rentabilidad.** Las señales son para
> estudio y paper trading. Ninguna precisión está prometida; los resultados out-of-sample
> se reportan en `report.md` y `go_no_go.md` con honestidad (incluido si no se alcanza la meta).

---

## 1. Ejecución rápida

```bash
pip install -r requirements.txt
python -m src.run --assets BTC-USD ETH-USD SPY QQQ EURUSD=X --mode paper
python -m src.backtest --walk-forward --report
pytest
```

Variantes útiles:

```bash
python -m src.run --mode paper --offline     # demo local con datos SINTÉTICOS (sin internet)
python -m src.run --mode paper --retrain     # reentrena el modelo
python -m src.backtest --walk-forward --report --offline --quick   # backtest rápido (CI)
python -m src.signals.evaluator              # precisión en vivo de señales pasadas
python scripts/healthcheck.py                # healthcheck (datos frescos, señales recientes)
```

El modo `live` **no existe**: `--mode` solo acepta `paper`.

## 2. Arquitectura

```
src/
  run.py               # CLI principal (paper): descarga datos, genera señales, escribe CSVs
  backtest.py          # CLI de backtest walk-forward + reporte + robustez
  config.py            # config.yaml + overrides + kill switch
  training.py          # dataset (features + etiquetas triple barrier alineadas por fecha/activo)
  data/                # fuentes con fallback + cache + validación (fail-safe)
    prices.py          #   yfinance (acciones/ETF/índices/FX) y ccxt (cripto), validación cruzada
    macro.py           #   FRED (con FRED_API_KEY) + fallback yfinance (^VIX, DX-Y.NYB, ^TNX, ^FVX)
    sentiment.py       #   Fear&Greed (alternative.me), Google Trends (pytrends), noticias RSS
                       #     puntuadas con FinBERT (opcional) o lexicón propio (fallback)
    derivatives.py     #   Binance público: funding rate, open interest, long/short ratio
    onchain.py         #   CoinGecko, blockchain.info, Glassnode free tier (si hay key)
    validator.py       #   pydantic + chequeos: duplicados, high<low, precios<=0, stale, saltos, volumen
    cache.py           #   cache local en data/cache (gitignored)
    synthetic.py       #   datos sintéticos SOLO para --offline/tests (nunca en producción)
    pipeline.py        #   orquesta todo; activo sin datos válidos => bloqueado (señal NADA)
  features/
    technical.py       #   SMA/EMA, ADX, RSI, MACD, ROC, ATR, Bollinger, vol realizada, OBV, VWAP,
                       #     estructura de mercado (HH/LL), estacionalidad (todo rolling, sin futuro)
    regime.py          #   régimen (alcista/bajista/lateral) con Gaussian Mixture en ventana expansiva
    alignment.py       #   join as-of con rezagos de publicación (point-in-time, anti look-ahead)
  labeling/triple_barrier.py  # TP/SL/tiempo máx. con ATR (López de Prado)
  models/
    primary.py         #   LightGBM (o XGBoost) multiclase {-1,0,+1}; SHAP nativo (pred_contrib)
    calibrate.py       #   isotónica one-vs-rest o Platt
    ensemble.py        #   LightGBM + logística + random forest (soft vote + acuerdo mínimo)
    meta.py            #   meta-labeling: ¿tomar la señal del primario?
    pipeline.py        #   SignalModel: OOF walk-forward interno para calibrar sin leakage
  validation/
    walkforward.py     #   splits walk-forward con embargo (purging) + test ciego final
    metrics.py         #   Sharpe/Sortino/Calmar/MDD/profit factor/expectativa, Deflated Sharpe, PBO (CSCV)
    robustness.py      #   permutación, estrategias aleatorias, Monte Carlo por bloques, fuga (leakage), stress histórico
  backtester/
    engine.py          #   backtest realista: ejecución en apertura de la barra siguiente, comisiones+spread+slippage,
                       #     salidas intradía por stop/TP, sizing por ATR (1% riesgo), Kelly fraccional (tope 1/4),
                       #     límites de correlación/posición. MISMO código para backtest y paper.
    baseline.py        #   baseline SMA50/200 + vol objetivo, y buy & hold
    portfolio.py       #   RiskManager: circuit breakers (DD diario/semanal/total), pérdidas consecutivas, kill switch
  signals/
    generator.py       #   señales diarias: latest.csv + history.csv (idempotente), SHAP top-3
    evaluator.py       #   precisión en vivo por umbral de confianza
    context.py         #   filtros: calendario económico (FOMC/CPI/NFP), VIX extremo, ATR anómalo, iliquidez, régimen malo
  reporting/report.py  #   report.md, go_no_go.md, gráficos
```

## 3. Fuentes de datos (todas gratuitas y legales, con fallback)

| Fuente | Qué aporta | Sin API key | Fallback |
|---|---|---|---|
| yfinance | OHLCV acciones/ETF/índices/FX | ✅ | — |
| ccxt (Binance) | OHLCV cripto | ✅ | yfinance |
| FRED | tasas, inflación, desempleo, curva | ❌ (key gratis opcional) | yfinance (^VIX, DX-Y.NYB, ^TNX, ^FVX) |
| alternative.me | Fear & Greed Index | ✅ | señal NADA + alerta |
| pytrends | Google Trends | ✅ | — |
| RSS (Yahoo/CNBC/MarketWatch) | titulares + sentimiento (FinBERT opcional) | ✅ | lexicón propio |
| Binance fapi público | funding, OI, long/short | ✅ | — |
| CoinGecko / blockchain.info | on-chain (mcap, volumen, hash rate, nº txs) | ✅ | Glassnode free tier si hay key |

Todas las fuentes pasan por reintentos con backoff exponencial, timeouts, cache local en
`data/` (gitignored) y **validación cruzada** (si dos fuentes difieren > 2% en el close, el
activo queda bloqueado). **Política fail-safe: si algo falla, la señal es `nada` + alerta;
nunca rellenamos con suposiciones.**

## 4. Anti look-ahead

- Las **features** solo usan datos `<= t` (todo rolling; test `test_no_lookahead.py` verifica
  que un spike en `t+1` no cambia las features en `t`).
- Las **variables externas** se alinean con `merge_asof` aplicando **rezagos de publicación**
  (`publication_lags` en config.yaml: macro mensual +30d, trends +7d, etc.).
- Las **etiquetas** triple barrier sí miran el futuro (miden el resultado), pero el embargo
  del walk-forward (≥ horizonte de etiqueta) las purga del entrenamiento.
- Test de **fuga** (`test_leakage.py`): etiquetas barajadas => AUC ≈ 0.5.

## 5. Modelado

1. **Baseline**: tendencia SMA50/200 + volatilidad objetivo (referencia a batir).
2. **Primario**: LightGBM/XGBoost multiclase con etiquetado triple barrier basado en ATR.
3. **Meta-labeling**: segundo modelo decide si tomar la señal (P(resultado > 0)).
4. **Calibración** isotónica (o Platt) sobre predicciones OOF; señal solo si
   `prob_calibrada >= 0.70` (configurable) **y** meta ≥ 0.55 **y** acuerdo del ensamble ≥ 2/3.
   Se reporta precisión y nº de señales por umbral (`reports/live_accuracy_by_threshold.csv`).
5. **Ensamble**: LightGBM + regresión logística + random forest.
6. **Validación**: walk-forward con purging/embargo + test ciego final intocable
   (nunca split aleatorio).
7. **Deflated Sharpe** con el nº real de configuraciones probadas, y **PBO (CSCV)**.

## 6. Gestión de riesgo (paper)

- Sin apalancamiento. Riesgo máx. por operación: **1%** del capital (tope duro).
- Tamaño por **ATR** (stop = close ∓ 2·ATR) con **Kelly fraccional (máx. 1/4)** y tope de nocional.
- **Stop loss obligatorio** en cada señal y **RR mínima 1:1.5** (el TP se ajusta si hace falta).
- **Circuit breaker**: si el drawdown (diario/semanal/total) supera los límites (total 15%),
  el sistema **deja de emitir señales y alerta**.
- Máx. de posiciones simultáneas, correlación máxima entre posiciones, máx. señales/día y
  pausa tras 4 pérdidas consecutivas.
- **Kill switch**: archivo `KILL_SWITCH` en la raíz o `QSE_KILL_SWITCH=1` => todas las señales NADA.

## 7. Salidas

- `signals/latest.csv`: `date, asset, direction (long/short/nada), prob, meta_prob, agreement,
  entry_ref, stop, tp, size, top1, top2, top3 (SHAP), reason, data_mode`
- `signals/history.csv`: registro acumulado (idempotente por `date+asset`).
- `python -m src.signals.evaluator`: compara señales pasadas contra resultados reales y
  calcula la **precisión en vivo** global y por umbral de confianza
  (`reports/live_accuracy.csv`, `live_accuracy_by_threshold.csv`, `live_accuracy.json`).
- `report.md`, `go_no_go.md`, `reports/` (curva de equity, drawdown, precisión por umbral,
  métricas por activo y por régimen, `metrics.json`, `alerts.log`).
- **`reports/dashboard.html`**: dashboard **profesional y auto-contenido** (un solo archivo HTML
  con CSS y gráficos incrustados en base64, sin servidor ni dependencias externas). Se abre con
  doble clic en el navegador (`file://`) o sirviéndolo estáticamente (Netlify/Pages/http.server):
  banner de veredicto GO/NO-GO, KPIs out-of-sample con sus umbrales, **señales de hoy al lado de
  la acertabilidad en vivo medida contra resultados reales**, checklist de qué falta para ser APTO,
  gráficos, alertas e informe completo. Se regenera solo tras cada backtest y en el workflow diario
  (`python scripts/build_dashboard.py`).

## 7b. Notificaciones de señales (Telegram / Discord / email)

El propio motor te manda las señales del día **junto con la precisión en vivo honesta** por el canal
que configures en `.env` (ver `.env.example`); en GitHub Actions se leen de los secrets del repo:

| Canal | Variables de entorno |
|---|---|
| Telegram | `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` |
| Discord | `DISCORD_WEBHOOK_URL` |
| Email (SMTP) | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `EMAIL_FROM`, `EMAIL_TO` |
| Elegir canales | `QSE_NOTIFY_CHANNELS=telegram,discord` (vacío = todos los configurados) |

Sin ninguna variable configurada es no-op. Un fallo de red/credenciales **nunca** rompe el pipeline
(fail-safe: queda en `alerts.log`). El mensaje incluye cada señal (dirección, prob, meta, entry/SL/TP,
tamaño, top-3 SHAP), cuántas salieron `nada`, la **precisión en vivo acumulada** (global, por umbral y
Brier) y el disclaimer. En `daily_signals.yml` cada corrida diaria envía la notificación
automáticamente.

### Telegram paso a paso (recomendado)

1. En Telegram busca **@BotFather** → `/newbot` → dale un nombre → copia el **token** (`123456:ABC...`).
2. Busca **@userinfobot** → `/start` → copia tu **chat id** (tu número). *Alternativa*: escríbele
   `/start` a tu bot nuevo y usa el id que te dé @userinfobot.
3. Local: copia `.env.example` a `.env` y rellena `TELEGRAM_TOKEN=` y `TELEGRAM_CHAT_ID=`.
4. Prueba inmediata: `python scripts/test_notify.py` → debe llegar un mensaje de prueba a tu chat.
5. En GitHub (para el workflow diario): repo → **Settings → Secrets and variables → Actions →
   New repository secret** con `TELEGRAM_TOKEN` y `TELEGRAM_CHAT_ID`. Listo: cada día tras el cierre
   (21:40 UTC días hábiles, 00:40 UTC cripto) recibirás las señales + la precisión en vivo.
6. Opcional: `QSE_NOTIFY_CHANNELS=telegram` para usar solo Telegram.

El token del bot es un secreto: nunca lo subas a GitHub (`.env` está gitignored y `gitleaks.yml` vigila
que no se filtre). Si se filtra, revócalo con @BotFather (`/revoke`).

## 8. Automatización (GitHub Actions)

- `daily_signals.yml`: cada día tras el cierre (21:40 UTC días hábiles + 00:40 UTC cripto)
  descarga datos, genera señales (paper), evalúa precisión en vivo, hace commit de los CSVs
  y, si falla el healthcheck, **abre un Issue automático**. Notificación opcional por Telegram
  con `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID` como secrets.
- `ci.yml`: en cada push/PR — ruff, mypy, pytest con cobertura, backtest de regresión
  (offline, rápido) y comparación de métricas contra el baseline (falla si empeoran > umbral).
- `gitleaks.yml`: escaneo de secretos en cada push/PR.

## 9. Seguridad y claves

- **Ninguna API key en el código.** Todo por variables de entorno (ver `.env.example`;
  copia a `.env`, que está gitignored y protegido por gitleaks).
- `FRED_API_KEY` y `GLASSNODE_API_KEY` son opcionales (hay fallbacks).

## 10. Reproducibilidad y calidad

- Dependencias **fijadas** (`requirements.txt`) + `Dockerfile` (python:3.11-slim).
- Semillas fijas (`seed: 42`); reintentos con backoff; ejecución **idempotente**
  (correr dos veces el mismo día no duplica señales — test incluido).
- Logging estructurado (structlog), sin `except: pass`, mypy + ruff + pre-commit.
- Tests obligatorios: **look-ahead bias, leakage, idempotencia, datos corruptos, APIs caídas
  (mocks), métricas contra casos conocidos, kill switch, cache, consistencia backtest/paper**.

## 11. Limitaciones y riesgos

- Los datos gratuitos tienen latencia y calidad limitada (huecos, revisiones, ajustes).
- El backtest asume ejecución al open siguiente con slippage fijo; la realidad puede diferir
  (gaps, liquidez, halts). Los stops pueden ejecutarse peor que el nivel teórico.
- No hay órdenes reales en paper: no se modelan rechazos ni latencia de ejecución.
- El calendario económico es una aproximación (FOMC programados, NFP primer viernes, CPI ~día 12).
- FinBERT requiere `pip install -r requirements-ml.txt`; sin él se usa un lexicón propio.
- El modelo puede degradarse con el tiempo (concept drift); se mitiga con reentrenamiento
  programado y champion/challenger, pero no se elimina.
- Ver `report.md` §6 para la lista completa de "cómo podría fallar este sistema".
- **Riesgo de pérdida total del capital** si se usara con dinero real; por eso este repo
  solo opera en paper.

## 12. Resultados

Los resultados out-of-sample (sin maquillar), la compuerta GO/NO-GO y los intervalos de
confianza están en [`report.md`](report.md) y [`go_no_go.md`](go_no_go.md). La meta era
≥ 70-80% de aciertos en señales de alta confianza **out-of-sample**; si no llega, el
informe lo declara explícitamente.

---

*Esto no es asesoría financiera ni garantiza rentabilidad.*
