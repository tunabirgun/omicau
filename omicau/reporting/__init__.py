"""Dual reporting: interactive HTML dashboard and multi-format documentation."""

from __future__ import annotations

from omicau.reporting.reporter import build_report
from omicau.reporting.docs_generator import build_documentation

__all__ = ["build_report", "build_documentation"]
