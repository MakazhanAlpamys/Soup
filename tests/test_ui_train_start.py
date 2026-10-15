"""Tests for Web UI /api/train/start temp config cleanup (Refs #1464).

Verifies that:
1. A finished training run deletes its temporary soup_ui_config_*.yaml.
2. Sequential finished runs leave no temp config files behind.
3. Stopping a run via /api/train/stop deletes the temp config.
4. Process spawn failures clean up the temp config and re-raise.
5. Unit test: _drain_stdout_worker deletes config when stdout is None.
6. Unit test: _drain_stdout_worker handles already-deleted config gracefully.
7. Control test: invalid YAML returns 400 and leaves no temp config.
8. Control test: child process can read the config while running.
9. Control test: /api/train/status reports running=True with config_path during a run.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import soup_cli.ui.app as ui_app_module
from soup_cli.ui.app import TrainLogBuffer, _drain_stdout_worker, create_app, get_auth_token

VALID_CONFIG = "base: test\ndata:\n  train: ./data.jsonl\n"


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_auth_token()}"}


def _config_files(temp_dir: pathlib.Path) -> list[pathlib.Path]:
    return list(temp_dir.glob("soup_ui_config_*.yaml"))


def _wait_for_run_completion(timeout: float = 10.0) -> None:
    proc = ui_app_module._train_process
    if proc is not None:
        proc.wait(timeout=timeout)
    drain_thread = ui_app_module._train_drain_thread
    if drain_thread is not None:
        drain_thread.join(timeout=timeout)


@pytest.fixture(autouse=True)
def _reset_train_state(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    ui_app_module._train_process = None
    ui_app_module._train_log_buffer = None
    ui_app_module._train_drain_thread = None
    ui_app_module._train_config_path = None
    yield
    proc = ui_app_module._train_process
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            pass
    drain_thread = ui_app_module._train_drain_thread
    if drain_thread is not None and drain_thread.is_alive():
        drain_thread.join(timeout=2.0)
    ui_app_module._train_process = None
    ui_app_module._train_log_buffer = None
    ui_app_module._train_drain_thread = None
    ui_app_module._train_config_path = None


def test_single_run_deletes_temp_config(tmp_path: pathlib.Path) -> None:
    """A single finished training run deletes its temporary config file."""
    client = TestClient(create_app())
    with patch(
        "soup_cli.ui.app._resolve_train_argv",
        return_value=[sys.executable, "-c", "pass"],
    ):
        response = client.post(
            "/api/train/start",
            json={"config_yaml": VALID_CONFIG},
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        assert response.json()["started"] is True
        _wait_for_run_completion(10.0)
        assert _config_files(tmp_path) == []


def test_three_sequential_runs_leave_no_temp_configs(tmp_path: pathlib.Path) -> None:
    """Multiple sequential finished runs leave 0 temporary config files."""
    client = TestClient(create_app())
    with patch(
        "soup_cli.ui.app._resolve_train_argv",
        return_value=[sys.executable, "-c", "pass"],
    ):
        for _ in range(3):
            response = client.post(
                "/api/train/start",
                json={"config_yaml": VALID_CONFIG},
                headers=_auth_headers(),
            )
            assert response.status_code == 200
            assert response.json()["started"] is True
            _wait_for_run_completion(10.0)
        assert _config_files(tmp_path) == []


def test_stop_training_deletes_temp_config(tmp_path: pathlib.Path) -> None:
    """Terminating training via /api/train/stop deletes the temporary config."""
    client = TestClient(create_app())
    with patch(
        "soup_cli.ui.app._resolve_train_argv",
        return_value=[sys.executable, "-c", "import time; time.sleep(30)"],
    ):
        response = client.post(
            "/api/train/start",
            json={"config_yaml": VALID_CONFIG},
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        stop_response = client.post("/api/train/stop", headers=_auth_headers())
        assert stop_response.status_code == 200
        assert stop_response.json()["stopped"] is True
        _wait_for_run_completion(10.0)
        assert _config_files(tmp_path) == []


def test_spawn_failure_deletes_temp_config(tmp_path: pathlib.Path) -> None:
    """If process spawn fails with OSError, the temp config is deleted and error raised."""
    client = TestClient(create_app(), raise_server_exceptions=False)
    with patch("subprocess.Popen", side_effect=OSError("Exec format error")):
        response = client.post(
            "/api/train/start",
            json={"config_yaml": VALID_CONFIG},
            headers=_auth_headers(),
        )
        assert response.status_code == 500
        assert _config_files(tmp_path) == []


def test_drain_worker_stdout_none_deletes_temp_config(tmp_path: pathlib.Path) -> None:
    """Unit test: _drain_stdout_worker removes config even if stdout is None."""
    temp_config = tmp_path / "soup_ui_config_dummy.yaml"
    temp_config.write_text("base: test\n", encoding="utf-8")
    assert temp_config.exists()

    proc = subprocess.Popen([sys.executable, "-c", "pass"], stdout=None)
    buffer = TrainLogBuffer(maxlen=100)
    _drain_stdout_worker(proc, buffer, config_path=str(temp_config))

    assert not temp_config.exists()
    assert buffer.is_done()


def test_drain_worker_already_deleted_config_does_not_raise(tmp_path: pathlib.Path) -> None:
    """Unit test: calling _drain_stdout_worker with a missing file does not raise."""
    deleted_path = str(tmp_path / "soup_ui_config_missing.yaml")
    assert not os.path.exists(deleted_path)

    proc = subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.PIPE)
    buffer = TrainLogBuffer(maxlen=100)
    _drain_stdout_worker(proc, buffer, config_path=deleted_path)
    assert buffer.is_done()


def test_invalid_yaml_returns_400_and_leaves_no_temp_config(tmp_path: pathlib.Path) -> None:
    """Control test: invalid YAML returns 400 before creating any file."""
    client = TestClient(create_app())
    response = client.post(
        "/api/train/start",
        json={"config_yaml": ": invalid: yaml: {"},
        headers=_auth_headers(),
    )
    assert response.status_code == 400
    assert _config_files(tmp_path) == []


def test_child_can_read_config_while_running(tmp_path: pathlib.Path) -> None:
    """Control test: the child process can read the temporary config while executing."""
    client = TestClient(create_app())
    script = (
        "import sys, time, pathlib\n"
        "target = pathlib.Path(sys.argv[1])\n"
        "assert target.exists(), 'Config does not exist'\n"
        "content = target.read_text(encoding='utf-8')\n"
        "assert 'base: test' in content\n"
        "time.sleep(1.0)\n"
    )
    with patch(
        "soup_cli.ui.app._resolve_train_argv",
        side_effect=lambda cfg_path: [sys.executable, "-c", script, cfg_path],
    ):
        response = client.post(
            "/api/train/start",
            json={"config_yaml": VALID_CONFIG},
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        proc = ui_app_module._train_process
        assert proc is not None

        active_configs = _config_files(tmp_path)
        assert len(active_configs) == 1
        assert active_configs[0].exists()

        _wait_for_run_completion(10.0)
        assert proc.poll() == 0


def test_train_status_reports_config_path_during_run(tmp_path: pathlib.Path) -> None:
    """Control test: /api/train/status returns running=True and config_path during a run."""
    client = TestClient(create_app())
    with patch(
        "soup_cli.ui.app._resolve_train_argv",
        return_value=[sys.executable, "-c", "import time; time.sleep(1.0)"],
    ):
        response = client.post(
            "/api/train/start",
            json={"config_yaml": VALID_CONFIG},
            headers=_auth_headers(),
        )
        assert response.status_code == 200

        status_response = client.get("/api/train/status", headers=_auth_headers())
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["running"] is True
        assert status_data["config_path"] is not None
        assert os.path.exists(status_data["config_path"])

        _wait_for_run_completion(10.0)
