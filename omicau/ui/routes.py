"""Token-gated API routes for the local UI.

Registered onto the FastAPI app by ``server.create_app`` (already behind the
one-time-token middleware). Every route is a thin wrapper over the stack-portable
core in :mod:`omicau.ui.inspect` plus the same ``run_audit`` library entry point
the CLI uses. Files are written only under the per-session workspace; nothing
leaves the machine. Long runs execute in a background thread and stream named
stages via polling.
"""

import secrets
import threading
import traceback
from pathlib import Path


def register(app) -> None:  # noqa: C901 - a flat set of small handlers
    from fastapi import HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse

    from omicau.ui import inspect as I

    def _sess(sid: str) -> dict:
        s = app.state.sessions.get(sid)
        if s is None:
            raise HTTPException(404, f"unknown session '{sid}'")
        return s

    def _parse(fn, *a, **k):
        """Turn a data-parsing failure into a clean 400, not a 500 traceback."""
        try:
            return fn(*a, **k)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - user-supplied files
            raise HTTPException(400, f"Could not read the data: {exc}")

    @app.post("/api/session")
    async def new_session():
        sid = secrets.token_hex(8)
        ws = Path(app.state.workspace) / sid
        ws.mkdir(parents=True, exist_ok=True)
        app.state.sessions[sid] = {"dir": str(ws), "files": [], "clinical_map": {}, "run": None}
        return {"session": sid}

    @app.post("/api/session/{sid}/upload")
    async def upload(sid: str, files: list[UploadFile]):
        s = _sess(sid)
        out = []
        for f in files:
            safe = Path(f.filename or "file.csv").name
            dest = Path(s["dir"]) / safe
            dest.write_bytes(await f.read())
            try:
                info = I.inspect_matrix(dest, safe)
            except Exception as exc:  # noqa: BLE001 - user-supplied files
                raise HTTPException(400, f"Could not read '{safe}': {exc}")
            rec = {"filename": safe, "path": str(dest), "role": info["role"],
                   "orientation": "samples_as_rows"}
            s["files"] = [x for x in s["files"] if x["filename"] != safe] + [rec]
            out.append(info)
        return {"files": out}

    @app.post("/api/session/{sid}/roles")
    async def set_roles(sid: str, payload: dict):
        s = _sess(sid)
        roles = payload.get("roles", {})            # {filename: role}
        for rec in s["files"]:
            if rec["filename"] in roles:
                rec["role"] = roles[rec["filename"]]
        clinical = [r for r in s["files"] if r["role"] == "clinical"]
        omics = [r for r in s["files"] if r["role"] != "clinical"]
        omic_roles = [r["role"] for r in omics]
        errors = []
        if len(clinical) != 1:
            errors.append("Assign exactly one file as the clinical table.")
        if len(omic_roles) != len(set(omic_roles)):
            errors.append("Two files share an omic role — give each layer a distinct role.")
        if not omics:
            errors.append("Add at least one omic-data file.")
        return {"ok": not errors, "errors": errors,
                "omics": [r["role"] for r in omics], "clinical": bool(clinical)}

    @app.post("/api/session/{sid}/orient")
    async def set_orient(sid: str, payload: dict):
        s = _sess(sid)
        for rec in s["files"]:
            if rec["filename"] in payload.get("orientation", {}):
                rec["orientation"] = payload["orientation"][rec["filename"]]
        return {"ok": True}

    def _clinical_path(s: dict) -> str:
        c = [r for r in s["files"] if r["role"] == "clinical"]
        if not c:
            raise HTTPException(400, "no clinical table assigned")
        return c[0]["path"]

    @app.get("/api/session/{sid}/clinical")
    async def clinical_columns(sid: str):
        return _parse(I.inspect_clinical, _clinical_path(_sess(sid)))

    @app.get("/api/session/{sid}/consequence")
    async def consequence(sid: str, column: str, kind: str, task: str = "auto",
                          target: str | None = None):
        path = _clinical_path(_sess(sid))
        if kind == "target":
            return _parse(I.target_consequence, path, column, task)
        if kind == "group":
            return _parse(I.group_consequence, path, column)
        if kind == "batch":
            return _parse(I.batch_consequence, path, column, target)
        raise HTTPException(400, f"unknown consequence kind '{kind}'")

    @app.post("/api/session/{sid}/clinical-map")
    async def clinical_map(sid: str, payload: dict):
        _sess(sid)["clinical_map"] = {
            k: payload.get(k) for k in ("target", "sample_id", "group", "batch", "task")
        }
        return {"ok": True}

    @app.post("/api/session/{sid}/options")
    async def options(sid: str, payload: dict):
        s = _sess(sid)
        for k in ("n_splits", "neural", "run_name"):
            if k in payload:
                s[k] = payload[k]
        return {"ok": True}

    def _modalities(s: dict) -> list[dict]:
        return [{"name": r["role"], "path": r["path"], "orientation": r["orientation"]}
                for r in s["files"] if r["role"] != "clinical"]

    @app.post("/api/session/{sid}/align")
    async def align(sid: str):
        s = _sess(sid)
        cm = s.get("clinical_map", {})
        return _parse(I.alignment_preview, _modalities(s), _clinical_path(s), cm.get("sample_id"))

    def _config_dict(s: dict) -> dict:
        cm = s.get("clinical_map", {})
        return I.build_config_dict({
            "run_name": s.get("run_name") or "omicau_ui_run",
            "output_dir": str(Path(s["dir"]) / "run"),
            "modalities": _modalities(s),
            "clinical": {"path": _clinical_path(s), **cm},
            "n_splits": s.get("n_splits", 5),
            "neural": s.get("neural", True),
        })

    @app.post("/api/session/{sid}/preflight")
    async def preflight(sid: str):
        from omicau.config import OmicauConfig
        from omicau.data.alignment import load_and_align
        from omicau.models.classical import resolve_cores
        from omicau.cli import estimate_runtime
        s = _sess(sid)
        cfg = OmicauConfig.from_dict(_config_dict(s))
        try:
            aligned = load_and_align(cfg)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Could not align the data: {exc}")
        cores = resolve_cores(cfg)
        cost = estimate_runtime(aligned, cfg, "cpu", cores)
        return {"ok": True, "n_samples": aligned.n_samples, "task": aligned.task,
                "feature_counts": aligned.feature_counts(),
                "provenance_hash": aligned.provenance_hash,
                "cost": {"human_readable": cost["human_readable"],
                         "total_seconds": cost["total_seconds"]}}

    @app.post("/api/session/{sid}/run")
    async def run(sid: str):
        s = _sess(sid)
        if s.get("run") and s["run"]["status"] == "running":
            return {"ok": True, "already": True}
        s["run"] = {"status": "running", "stages": [], "error": None, "report": None,
                    "provenance": None}
        cfg_dict = _config_dict(s)

        def _worker():
            try:
                from omicau.config import OmicauConfig
                from omicau.cli import run_audit
                cfg = OmicauConfig.from_dict(cfg_dict)
                audit = run_audit(cfg, cores=None, device="auto", llm=False,
                                  echo=lambda m: s["run"]["stages"].append(str(m)))
                # write payload fields BEFORE flipping status, so a reader that sees
                # status=="done" always sees a consistent report/provenance.
                s["run"]["provenance"] = audit["meta"]["provenance_hash"]
                s["run"]["report"] = audit.get("_assets", {}).get("html")
                s["run"]["status"] = "done"          # published last
            except Exception as exc:  # noqa: BLE001
                s["run"]["error"] = f"{exc}"
                s["run"]["trace"] = traceback.format_exc()
                s["run"]["status"] = "error"         # published last

        threading.Thread(target=_worker, daemon=True).start()
        return {"ok": True, "started": True}

    @app.get("/api/session/{sid}/progress")
    async def progress(sid: str):
        r = _sess(sid).get("run") or {"status": "idle", "stages": []}
        status = r["status"]                          # read the published flag once
        return {"status": status, "stages": list(r.get("stages", [])),
                "error": r.get("error") if status == "error" else None,
                "report_ready": status == "done",     # derive from status, never from a mid-update field
                "provenance": r.get("provenance") if status == "done" else None}

    @app.get("/api/session/{sid}/report")
    async def report(sid: str):
        r = _sess(sid).get("run") or {}
        path = r.get("report")
        if not path or not Path(path).exists():
            raise HTTPException(404, "report not ready")
        return FileResponse(path, media_type="text/html")
