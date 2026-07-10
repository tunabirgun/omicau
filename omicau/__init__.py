"""omicau -- Omics Audit.

A reproducible, leakage-safe, platform-agnostic CLI for auditing multi-omic
datasets: flexible ingestion and alignment, cryptographic provenance, batch and
missingness-bias diagnostics, leakage-safe classical and neural fusion
benchmarks, feature attribution, and dual clinical/research reporting.

The core package is fully self-contained: it runs with no internet access, no
LLM connection, and no orchestration framework. Optional tiers (LLM
interpretation, remote data hubs) degrade gracefully when their dependencies or
network are absent.
"""

from __future__ import annotations

__version__ = "0.1.1"
__author__ = "Tuna Birgun"

# Lightweight, dependency-free symbols are re-exported for convenience. Heavy
# submodules (models, reporting) are imported lazily by callers to keep
# ``import omicau`` cheap and side-effect free.
from omicau.config import OmicauConfig  # noqa: E402

__all__ = ["OmicauConfig", "__version__", "__author__"]
