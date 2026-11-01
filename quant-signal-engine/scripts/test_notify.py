"""Envía un mensaje de PRUEBA por los canales configurados en .env (Telegram/Discord/email).

Uso:  python scripts/test_notify.py
Útil para verificar credenciales antes de esperar al workflow diario.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.config import ROOT, Config  # noqa: E402
from src.utils.alerts import AlertBus  # noqa: E402
from src.utils.notify import load_live_accuracy, notify_signals_full  # noqa: E402


def main() -> int:
    load_dotenv(ROOT / ".env")
    cfg = Config()
    alerts = AlertBus(cfg.reports_dir)
    test_signals = pd.DataFrame(
        [
            {
                "date": str(pd.Timestamp.now().date()),
                "asset": "BTC-USD",
                "direction": "long",
                "prob": 0.83,
                "meta_prob": 0.71,
                "agreement": 1.0,
                "entry_ref": 0.0,
                "stop": 0.0,
                "tp": 0.0,
                "size": 0.0,
                "top1": "rsi_14",
                "top2": "adx_14",
                "top3": "vix_z",
                "reason": "MENSAJE DE PRUEBA (sin señales reales)",
                "data_mode": "test",
            }
        ]
    )
    res = notify_signals_full(cfg, alerts, test_signals, load_live_accuracy(cfg.reports_dir))
    print("Resultado por canal:", res)
    if any(res.values()):
        print("✅ Llegó por al menos un canal. Revisa Telegram/Discord/correo.")
        return 0
    print(
        "❌ No llegó por ningún canal. Revisa:\n"
        "  1) .env con TELEGRAM_TOKEN y TELEGRAM_CHAT_ID correctos\n"
        "  2) que le hayas enviado /start a tu bot en Telegram\n"
        "  3) QSE_NOTIFY_CHANNELS (si pusiste 'telegram', solo probará Telegram)\n"
        "  4) conexión a internet (api.telegram.org)"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
