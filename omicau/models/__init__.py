"""Leakage-safe cross-validated fusion benchmarks (classical + neural) and XAI."""

from __future__ import annotations

from omicau.models.base import CVResult, make_cv_splitter, score_predictions
from omicau.models.classical import run_classical_benchmarks
from omicau.models.neural import run_neural_benchmark

__all__ = [
    "CVResult",
    "make_cv_splitter",
    "score_predictions",
    "run_classical_benchmarks",
    "run_neural_benchmark",
]
