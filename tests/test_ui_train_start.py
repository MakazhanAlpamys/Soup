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
10. Unit test: _drain_stdout_worker waits for child before deleting config (wait-order guard).
11. Popen failure does not close a reused file descriptor.
12. fdopen failure closes raw fd and cleans up temp config.
13. fdopen failure suppresses os.close error and preserves original exception.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Iterator
from unittest.mock import patch

import pytest

import soup_cli.ui.app as ui_app_module
from soup_cli.ui.app import TrainLogBuffer, _drain_stdout_worker, get_auth_token
from tests.ui_helpers import ui_client

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
    client = ui_client()
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
    client = ui_client()
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
    client = ui_client()
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
    client = ui_client(raise_server_exceptions=False)
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
    client = ui_client()
    response = client.post(
        "/api/train/start",
        json={"config_yaml": ": invalid: yaml: {"},
        headers=_auth_headers(),
    )
    assert response.status_code == 400
    assert _config_files(tmp_path) == []


def test_child_can_read_config_while_running(tmp_path: pathlib.Path) -> None:
    """Control test: the child process can read the temporary config while executing."""
    client = ui_client()
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
    client = ui_client()
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


class FakeStdout:
    """Fake stdout stream returning immediate EOF and tracking close calls."""

    def __init__(self) -> None:
        self.close_calls = 0

    def readline(self) -> bytes:
        return b""

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize(
    "has_stdout",
    [False, True],
    ids=["no_stdout", "stream"],
)
def test_drain_worker_wait_order_guard(tmp_path: pathlib.Path, has_stdout: bool) -> None:
    """Unit test: _drain_stdout_worker calls proc.wait() before deleting config, and
    marks done first, for both stdout=None and active stream cases.
    """
    temp_config = tmp_path / "soup_ui_config_wait_guard.yaml"
    temp_config.write_text("base: test\n", encoding="utf-8")
    assert temp_config.exists()

    buffer = TrainLogBuffer(maxlen=100)

    class FakeProc:
        def __init__(self, stdout: FakeStdout | None) -> None:
            self.stdout = stdout
            self.wait_calls = 0
            self.config_existed: bool | None = None
            self.done_at_wait: bool | None = None

        def wait(self) -> None:
            self.wait_calls += 1
            self.config_existed = temp_config.exists()
            self.done_at_wait = buffer.is_done()

    stdout_stream = FakeStdout() if has_stdout else None
    proc = FakeProc(stdout_stream)
    _drain_stdout_worker(proc, buffer, config_path=str(temp_config))

    assert proc.wait_calls == 1
    assert proc.config_existed is True
    assert proc.done_at_wait is True
    assert not temp_config.exists()
    if has_stdout:
        assert proc.stdout is not None
        assert proc.stdout.close_calls == 1


def test_popen_failure_does_not_close_reused_descriptor(tmp_path: pathlib.Path) -> None:
    """If Popen fails after fdopen, the outer except must NOT close someone else's descriptor."""
    created_fd: list[int] = []
    real_mkstemp = tempfile.mkstemp

    def mkstemp_spy(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created_fd.append(fd)
        return fd, path

    opened_handles: list[int] = []

    def popen_side_effect(*args, **kwargs):
        assert created_fd, "mkstemp must have run before Popen"
        target_fd = created_fd[0]
        found = False
        for attempt in range(64):
            dummy_file = tmp_path / f"spare_probe_{attempt}.tmp"
            dummy_file.touch()
            allocated_fd = os.open(str(dummy_file), os.O_RDONLY)
            opened_handles.append(allocated_fd)
            if allocated_fd == target_fd:
                found = True
                break
        if not found:
            pytest.fail(f"Could not reacquire fd {target_fd} within 64 attempts")
        raise OSError("Popen spawn failed")

    try:
        client = ui_client(raise_server_exceptions=False)
        with patch("tempfile.mkstemp", side_effect=mkstemp_spy), patch(
            "subprocess.Popen", side_effect=popen_side_effect
        ):
            response = client.post(
                "/api/train/start",
                json={"config_yaml": VALID_CONFIG},
                headers=_auth_headers(),
            )
            assert response.status_code == 500

            # created_fd[0] was reused by our probe. The outer except handler must NOT
            # have closed it. In the buggy code, os.close(fd) closed created_fd[0],
            # causing os.fstat to raise OSError: [Errno 9] Bad file descriptor.
            assert os.fstat(created_fd[0]) is not None
            assert _config_files(tmp_path) == []
    finally:
        for descriptor in opened_handles:
            with contextlib.suppress(OSError):
                os.close(descriptor)


def test_fdopen_failure_closes_fd_and_deletes_temp_config(tmp_path: pathlib.Path) -> None:
    """When os.fdopen fails, the raw fd is closed and temp config deleted."""
    created_fd: list[int] = []
    real_mkstemp = tempfile.mkstemp

    def mkstemp_spy(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created_fd.append(fd)
        return fd, path

    client = ui_client(raise_server_exceptions=True)
    with patch("tempfile.mkstemp", side_effect=mkstemp_spy), patch(
        "os.fdopen", side_effect=OSError("Disk full on fdopen")
    ):
        with pytest.raises(OSError, match="Disk full on fdopen"):
            client.post(
                "/api/train/start",
                json={"config_yaml": VALID_CONFIG},
                headers=_auth_headers(),
            )

    assert created_fd, "mkstemp must have been called"
    with pytest.raises(OSError):
        os.fstat(created_fd[0])
    assert _config_files(tmp_path) == []


def test_fdopen_failure_os_close_error_does_not_mask_original_exception(
    tmp_path: pathlib.Path,
) -> None:
    """When os.fdopen fails and os.close also raises, the fdopen error is not masked."""
    created_fd: list[int] = []
    real_mkstemp = tempfile.mkstemp
    real_close = os.close

    def mkstemp_spy(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created_fd.append(fd)
        return fd, path

    closed_fds: list[int] = []

    def close_wrapper(fd: int) -> None:
        if created_fd and fd == created_fd[0]:
            real_close(fd)
            closed_fds.append(fd)
            raise OSError("Close error on fd")
        real_close(fd)
        closed_fds.append(fd)

    client = ui_client(raise_server_exceptions=True)
    with patch("tempfile.mkstemp", side_effect=mkstemp_spy), patch(
        "os.fdopen", side_effect=ValueError("Invalid fd parameter")
    ), patch("os.close", side_effect=close_wrapper):
        with pytest.raises(ValueError, match="Invalid fd parameter"):
            client.post(
                "/api/train/start",
                json={"config_yaml": VALID_CONFIG},
                headers=_auth_headers(),
            )

    assert created_fd, "mkstemp must have been called"
    assert created_fd[0] in closed_fds
    assert _config_files(tmp_path) == []
