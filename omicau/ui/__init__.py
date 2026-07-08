"""Optional local web UI for omicau (localhost only).

This tier is opt-in (`pip install omicau[ui]`) and never required: omicau is a
CLI tool first. The UI is a thin, single-user, local-first shell around the same
``run_audit`` library entry point — it binds to 127.0.0.1 with a one-time token,
uploads nothing off the machine, and embeds the exact HTML dashboard the CLI
produces.
"""

from __future__ import annotations

from omicau.ui.server import launch

__all__ = ["launch"]
