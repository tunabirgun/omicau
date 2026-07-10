"""Local FastAPI server for the opt-in omicau web UI.

Single-user, local-first: binds 127.0.0.1 on an auto-selected free port, gates
every ``/api`` route behind a one-time token minted at launch, and keeps a
per-session workspace under the system temp dir. No data leaves the machine.
FastAPI / uvicorn are imported lazily so the core package never depends on them.
"""

from __future__ import annotations

import secrets
import socket
import tempfile
import threading
import webbrowser
from pathlib import Path

from omicau import __version__

STATIC = Path(__file__).parent / "static"


def _require_ui():
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        return True
    except ImportError as exc:  # pragma: no cover - optional dep
        from omicau._hints import extra_hint
        raise ImportError(
            "The local UI needs the optional 'ui' extra (FastAPI + uvicorn). "
            f"Add it with:  {extra_hint('ui')}\n"
            "omicau itself runs fully as a CLI without it."
        ) from exc


def _free_port(preferred: int | None = None) -> int:
    if preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", preferred)) != 0:
                return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _app_css() -> str:
    """Serve fonts + the shared dashboard design system + UI-specific styles."""
    from omicau.reporting._assets import FONT_FACES, DASHBOARD_CSS
    ui_css = (STATIC / "ui.css").read_text(encoding="utf-8")
    return f"{FONT_FACES}\n{DASHBOARD_CSS}\n{ui_css}"


def create_app(token: str, workspace: Path):
    """Build the FastAPI app (token-gated ``/api`` routes)."""
    _require_ui()
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    app = FastAPI(title="omicau", version=__version__, docs_url=None, redoc_url=None)
    app.state.token = token
    app.state.workspace = workspace
    app.state.sessions = {}  # session_id -> state (populated in later phases)

    @app.middleware("http")
    async def _gate(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            supplied = (request.headers.get("x-omicau-token")
                        or request.query_params.get("token")
                        or request.cookies.get("omicau_token"))
            if supplied != app.state.token:
                return JSONResponse({"error": "invalid or missing token"}, status_code=403)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def index(token: str | None = None):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        resp = HTMLResponse(html)
        if token:  # persist the launch token so the SPA can call /api
            resp.set_cookie("omicau_token", token, httponly=False, samesite="strict")
        return resp

    @app.get("/assets/app.css")
    async def app_css():
        return PlainTextResponse(_app_css(), media_type="text/css")

    @app.get("/assets/app.js")
    async def app_js():
        return PlainTextResponse((STATIC / "app.js").read_text(encoding="utf-8"),
                                 media_type="text/javascript")

    @app.get("/api/health")
    async def health():
        return {"ok": True, "version": __version__, "workspace": str(workspace)}

    # Data-facing routes (upload, inspect, validate, run, results) are registered
    # by the wizard module so this skeleton stays importable on its own.
    try:
        from omicau.ui import routes
        routes.register(app)
    except Exception:  # noqa: BLE001 - routes are added incrementally
        pass

    return app


def launch(host: str = "127.0.0.1", port: int | None = None, open_browser: bool = True,
           token: str | None = None, workspace: str | Path | None = None,
           echo=print) -> None:
    """Start the local UI server (blocking) and open the browser."""
    _require_ui()          # clean 'pip install omicau[ui]' message before touching uvicorn
    import uvicorn

    token = token or secrets.token_urlsafe(16)
    port = _free_port(port)
    ws = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="omicau_ui_"))
    ws.mkdir(parents=True, exist_ok=True)
    app = create_app(token, ws)
    url = f"http://{host}:{port}/?token={token}"

    echo(f"omicau UI running at {url}")
    echo("  (local only — data stays on this machine; press Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
