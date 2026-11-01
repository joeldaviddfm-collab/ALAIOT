"""Fuentes de datos gratuitas con fallback y validación fail-safe."""

from src.data.base import DataUnavailableError, safe_call
from src.data.cache import DiskCache
from src.data.pipeline import MarketData, load_market_data
from src.data.validator import ValidationReport, cross_source_check, validate_ohlcv

__all__ = [
    "DataUnavailableError",
    "safe_call",
    "DiskCache",
    "ValidationReport",
    "cross_source_check",
    "validate_ohlcv",
    "MarketData",
    "load_market_data",
]
