"""Generación de señales (paper), evaluación en vivo e idempotencia."""

from src.signals.evaluator import evaluate_signals
from src.signals.generator import generate_signals, write_signals

__all__ = ["generate_signals", "write_signals", "evaluate_signals"]
