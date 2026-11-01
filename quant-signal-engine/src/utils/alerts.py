"""Bus de alertas: acumula alertas, las escribe en reports/alerts.log y las imprime.

Niveles: INFO, WARNING, ERROR, CRITICAL. CRITICAL implica señal = NADA (fail-safe).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import structlog

log = structlog.get_logger("qse.alerts")


@dataclass
class Alert:
    ts: str
    level: str
    msg: str


@dataclass
class AlertBus:
    reports_dir: Path | None = None
    items: list[Alert] = field(default_factory=list)

    def add(self, level: str, msg: str) -> Alert:
        alert = Alert(ts=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), level=level, msg=msg)
        self.items.append(alert)
        line = f"[{alert.ts}] {alert.level}: {alert.msg}"
        if alert.level == "CRITICAL" or alert.level == "ERROR":
            log.error("alert", msg=msg)
        elif alert.level == "WARNING":
            log.warning("alert", msg=msg)
        else:
            log.info("alert", msg=msg)
        if self.reports_dir is not None:
            path = self.reports_dir / "alerts.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return alert

    def has_critical(self) -> bool:
        return any(a.level == "CRITICAL" for a in self.items)

    def summary(self) -> str:
        if not self.items:
            return "Sin alertas."
        counts: dict[str, int] = {}
        for a in self.items:
            counts[a.level] = counts.get(a.level, 0) + 1
        return "; ".join(f"{k}={v}" for k, v in sorted(counts.items()))
