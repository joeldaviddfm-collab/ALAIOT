"""Motor de backtest realista y gestión de riesgo. Mismo código para backtest y paper."""

from src.backtester.baseline import run_baseline, run_buy_hold
from src.backtester.engine import BacktestResult, Trade, run_backtest
from src.backtester.portfolio import RiskManager

__all__ = ["BacktestResult", "Trade", "run_backtest", "run_baseline", "run_buy_hold", "RiskManager"]
