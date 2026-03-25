"""FastAPI application for Soup Web UI."""

import logging
import os
import secrets
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Max file read size to prevent memory exhaustion
_MAX_INSPECT_LIMIT = 500


class TrainRequest(PydanticBaseModel):
    """Request body for starting a training run."""
    config_yaml: str


class TrainStatus(PydanticBaseModel):
    """Current training process status."""
    running: bool
    pid: Optional[int] = None
    config_path: Optional[str] = None


class DataInspectRequest(PydanticBaseModel):
    """Request body for data inspection."""
    path: str
    limit: int = Field(default=50, ge=1, le=_MAX_INSPECT_LIMIT)


# Global state for training process
_train_process: Optional[subprocess.Popen] = None
_train_config_path: Optional[str] = None
_train_lock = threading.Lock()

# Auth token generated at startup — printed to console for the user
_auth_token: str = secrets.token_urlsafe(32)


def get_auth_token() -> str:
    """Return the current auth token (for printing at startup)."""
    return _auth_token


def create_app(host: str = "127.0.0.1", port: int = 7860):
    """Create the Soup Web UI FastAPI application."""
    from fastapi import Depends, FastAPI, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Soup Web UI", version="1.0.0")

    # Restrict CORS to the origin we actually serve
    allowed_origin = f"http://{host}:{port}"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[allowed_origin],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    def _verify_token(request: Request):
        """Verify Bearer token on mutating endpoints."""
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {_auth_token}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    # --- Static files ---

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_path = STATIC_DIR / "index.html"
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # --- Runs API ---

    @app.get("/api/runs")
    def list_runs(limit: int = Query(default=50, ge=1, le=500)):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            runs = tracker.list_runs(limit=limit)
            return {"runs": runs}
        finally:
            tracker.close()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            run = tracker.get_run(run_id)
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")
            return run
        finally:
            tracker.close()

    @app.get("/api/runs/{run_id}/metrics")
    def get_run_metrics(run_id: str):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            run = tracker.get_run(run_id)
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")
            metrics = tracker.get_metrics(run_id)
            return {"run_id": run_id, "metrics": metrics}
        finally:
            tracker.close()

    @app.delete("/api/runs/{run_id}", dependencies=[Depends(_verify_token)])
    def delete_run(run_id: str):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            deleted = tracker.delete_run(run_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Run not found")
            return {"deleted": True, "run_id": run_id}
        finally:
            tracker.close()

    @app.get("/api/runs/{run_id}/eval")
    def get_run_eval(run_id: str):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            results = tracker.get_eval_results(run_id=run_id)
            return {"run_id": run_id, "eval_results": results}
        finally:
            tracker.close()

    # --- GPU / System Info ---

    @app.get("/api/system")
    def system_info():
        from soup_cli import __version__
        from soup_cli.utils.gpu import detect_device, get_gpu_info

        device, device_name = detect_device()
        gpu_info = get_gpu_info()
        return {
            "version": __version__,
            "device": device,
            "device_name": device_name,
            "gpu_info": gpu_info,
            "python_version": sys.version.split()[0],
        }

    # --- Templates ---

    @app.get("/api/templates")
    def list_templates():
        from soup_cli.config.schema import TEMPLATES

        return {"templates": {name: yaml_str for name, yaml_str in TEMPLATES.items()}}

    # --- Config Validation ---

    @app.post("/api/config/validate", dependencies=[Depends(_verify_token)])
    def validate_config(body: dict):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = body.get("yaml", "")
        if not yaml_str:
            raise HTTPException(status_code=400, detail="Empty config")
        try:
            config = load_config_from_string(yaml_str)
            return {"valid": True, "config": config.model_dump()}
        except Exception as exc:
            return {"valid": False, "error": str(exc)}

    # --- Training ---

    @app.post("/api/train/start", dependencies=[Depends(_verify_token)])
    def start_training(req: TrainRequest):
        global _train_process, _train_config_path

        with _train_lock:
            if _train_process and _train_process.poll() is None:
                raise HTTPException(
                    status_code=409, detail="Training already in progress"
                )

            # Validate config before writing to disk
            from soup_cli.config.loader import load_config_from_string

            try:
                load_config_from_string(req.config_yaml)
            except Exception as exc:
                logger.warning("Invalid training config: %s", exc)
                raise HTTPException(
                    status_code=400, detail="Invalid training configuration"
                )

            # Write config to a fixed safe location (never user-controlled path)
            config_path = os.path.join(
                tempfile.gettempdir(), "soup_ui_config.yaml"
            )
            with open(config_path, "w", encoding="utf-8") as fh:
                fh.write(req.config_yaml)

            _train_config_path = config_path
            _train_process = subprocess.Popen(
                [sys.executable, "-m", "soup_cli", "train", "--config", config_path, "--yes"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return {"started": True, "pid": _train_process.pid}

    @app.get("/api/train/status")
    def train_status():
        global _train_process
        with _train_lock:
            if _train_process is None:
                return TrainStatus(running=False)
            poll = _train_process.poll()
            if poll is None:
                return TrainStatus(
                    running=True,
                    pid=_train_process.pid,
                    config_path=_train_config_path,
                )
            return TrainStatus(running=False, pid=_train_process.pid)

    @app.post("/api/train/stop", dependencies=[Depends(_verify_token)])
    def stop_training():
        global _train_process
        with _train_lock:
            if _train_process and _train_process.poll() is None:
                _train_process.terminate()
                return {"stopped": True}
            return {"stopped": False, "detail": "No training in progress"}

    # --- Data Inspection ---

    @app.post("/api/data/inspect", dependencies=[Depends(_verify_token)])
    def inspect_data(req: DataInspectRequest):
        from soup_cli.data.loader import load_raw_data

        # Path traversal protection: resolve and check against cwd
        allowed_root = Path.cwd().resolve()
        try:
            resolved = Path(req.path).resolve()
        except (ValueError, OSError):
            raise HTTPException(status_code=400, detail="Invalid path")

        if not str(resolved).startswith(str(allowed_root)):
            raise HTTPException(
                status_code=403, detail="Access denied: path outside working directory"
            )

        if not resolved.exists():
            raise HTTPException(status_code=404, detail="File not found")

        try:
            raw_data = load_raw_data(resolved)
        except Exception as exc:
            logger.warning("Data inspect error: %s", exc)
            raise HTTPException(status_code=400, detail="Failed to load data file")

        total = len(raw_data)
        sample = raw_data[: req.limit]

        # Detect format
        from soup_cli.data.formats import detect_format

        fmt = detect_format(raw_data[:5]) if raw_data else "unknown"

        # Basic stats
        keys = set()
        for entry in sample:
            keys.update(entry.keys())

        return {
            "path": str(resolved),
            "total": total,
            "format": fmt,
            "keys": sorted(keys),
            "sample": sample,
        }

    # --- Health ---

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    return app
