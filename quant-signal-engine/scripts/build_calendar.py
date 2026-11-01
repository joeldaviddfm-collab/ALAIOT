"""Genera calendars/economic.csv: FOMC (fechas públicos programados), NFP (1er viernes
de mes) y CPI estimado (día 12, aproximación documentada) para 2018-2026.

Es una aproximación con fines de filtro de contexto; no pretende ser un calendario
económico oficial completo.
"""

from __future__ import annotations

import calendar
import datetime as dt
import sys
from pathlib import Path

FOMC_2018 = [
    "2018-01-31",
    "2018-03-21",
    "2018-05-02",
    "2018-06-13",
    "2018-08-01",
    "2018-09-26",
    "2018-11-08",
    "2018-12-19",
]
FOMC_2019 = [
    "2019-01-30",
    "2019-03-20",
    "2019-05-01",
    "2019-06-19",
    "2019-07-31",
    "2019-09-18",
    "2019-10-30",
    "2019-12-11",
]
FOMC_2020 = [
    "2020-01-29",
    "2020-03-03",
    "2020-03-15",
    "2020-04-29",
    "2020-06-10",
    "2020-07-29",
    "2020-09-16",
    "2020-11-05",
    "2020-12-16",
]
FOMC_2021 = [
    "2021-01-27",
    "2021-03-17",
    "2021-04-28",
    "2021-06-16",
    "2021-07-28",
    "2021-09-22",
    "2021-11-03",
    "2021-12-15",
]
FOMC_2022 = [
    "2022-01-26",
    "2022-03-16",
    "2022-05-04",
    "2022-06-15",
    "2022-07-27",
    "2022-09-21",
    "2022-11-02",
    "2022-12-14",
]
FOMC_2023 = [
    "2023-02-01",
    "2023-03-22",
    "2023-05-03",
    "2023-06-14",
    "2023-07-26",
    "2023-09-20",
    "2023-11-01",
    "2023-12-13",
]
FOMC_2024 = [
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
]
FOMC_2025 = [
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
]
FOMC_2026 = [
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-09",
]
ALL_FOMC = (
    FOMC_2018 + FOMC_2019 + FOMC_2020 + FOMC_2021 + FOMC_2022 + FOMC_2023 + FOMC_2024 + FOMC_2025 + FOMC_2026
)


def first_friday(year: int, month: int) -> dt.date:
    cal = calendar.Calendar()
    for day in cal.itermonthdates(year, month):
        if day.month == month and day.weekday() == 4:
            return day
    raise ValueError("no friday")


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "calendars" / "economic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in ALL_FOMC:
        rows.append((d, "FOMC", "high"))
    for year in range(2018, 2027):
        for month in range(1, 13):
            rows.append((first_friday(year, month).isoformat(), "NFP", "high"))
            rows.append((f"{year}-{month:02d}-12", "CPI (estimado)", "high"))
    rows = sorted(set(rows))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("date,event,impact\n")
        for r in rows:
            fh.write(",".join(r) + "\n")
    print(f"OK {out} ({len(rows)} eventos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
