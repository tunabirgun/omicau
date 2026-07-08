"""Token-gated API routes for the local UI.

Registered onto the FastAPI app by ``server.create_app``. Phase 1 provides the
per-session workspace; the upload / inspect / validate / run routes are added by
the wizard build. All routes are already behind the one-time-token middleware.
"""

from __future__ import annotations

import secrets
from pathlib import Path


def register(app) -> None:
    from fastapi import Request

    @app.post("/api/session")
    async def new_session():
        sid = secrets.token_hex(8)
        ws = Path(app.state.workspace) / sid
        ws.mkdir(parents=True, exist_ok=True)
        app.state.sessions[sid] = {"dir": str(ws), "files": [], "clinical": None}
        return {"session": sid}
