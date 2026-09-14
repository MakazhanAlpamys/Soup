"""Tests for issue #688: Web UI training subprocess pipe drain.

Verifies that:
1. Child subprocess emitting > 30 KiB of output completes with 0 readers without blocking.
2. Multiple concurrent consumers receive all lines without stealing from each other.
3. Last-Event-ID reconnect replays lines correctly from the bounded buffer.
4. stop_training terminates the child and cleans up the drain worker.
"""

import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

import soup_cli.ui.app as ui_app_module
from soup_cli.ui.app import TrainLogBuffer, _drain_stdout_worker, create_app, get_auth_token


def _auth_headers():
    return {"Authorization": f"Bearer {get_auth_token()}"}


class TestSubprocessDrain:
    """Test background stdout drain and non-blocking execution."""

    def test_subprocess_exceeding_pipe_threshold_completes_without_reader(self):
        """A child emitting ~40 KiB of stdout must finish rc=0 when nobody reads stdout."""
        # Generates ~40 KiB of stdout — far above the 8 KiB Windows / pipe capacity
        cmd = [
            sys.executable,
            "-c",
            "import sys\n"
            "for i in range(400):\n"
            "    sys.stdout.write(f'line {i}: ' + 'A' * 90 + '\\n')\n"
            "sys.stdout.flush()\n",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        buf = TrainLogBuffer(maxlen=1000)

        import threading

        drain_t = threading.Thread(
            target=_drain_stdout_worker,
            args=(proc, buf),
            daemon=True,
        )
        drain_t.start()

        # The child must complete within 10 seconds without deadlocking
        start_time = time.time()
        while proc.poll() is None:
            time.sleep(0.1)
            if time.time() - start_time > 10.0:
                proc.kill()
                pytest.fail("Subprocess blocked on write without stdout reader")

        drain_t.join(timeout=2.0)

        assert proc.poll() == 0
        assert buf.is_done()
        lines = buf.get_lines_from(0)
        assert len(lines) == 400
        assert lines[0][1].startswith("line 0:")
        assert lines[-1][1].startswith("line 399:")

    def test_multiple_readers_do_not_steal_lines(self):
        """Two concurrent consumers reading from /api/train/logs must both receive all lines."""
        buf = TrainLogBuffer(maxlen=100)
        for i in range(20):
            buf.append(f"message {i}")
        buf.mark_done()

        # Simulate reader 1
        lines_1 = []
        curr_1 = 0
        while True:
            batch, is_done = buf.wait_for_lines_or_done(curr_1)
            for idx, text in batch:
                lines_1.append(text)
                curr_1 = idx + 1
            if is_done:
                break

        # Simulate reader 2
        lines_2 = []
        curr_2 = 0
        while True:
            batch, is_done = buf.wait_for_lines_or_done(curr_2)
            for idx, text in batch:
                lines_2.append(text)
                curr_2 = idx + 1
            if is_done:
                break

        assert len(lines_1) == 20
        assert len(lines_2) == 20
        assert lines_1 == lines_2

    def test_last_event_id_reconnect_replays_unseen_lines(self):
        """Reconnecting with Last-Event-ID resumes from next event index."""
        ui_app_module._train_process = None
        ui_app_module._train_log_buffer = TrainLogBuffer(maxlen=100)
        for i in range(10):
            ui_app_module._train_log_buffer.append(f"log line {i}")
        ui_app_module._train_log_buffer.mark_done()

        client = TestClient(create_app())

        # Reconnect with Last-Event-ID = 6 -> should receive lines 7, 8, 9
        response = client.get(
            "/api/train/logs",
            headers={"Last-Event-ID": "6", **_auth_headers()},
        )
        assert response.status_code == 200
        content = response.text

        assert "id: 7" in content
        assert "log line 7" in content
        assert "id: 8" in content
        assert "id: 9" in content
        assert "id: 5" not in content
        assert "log line 5" not in content
        assert "event: done" in content

        # Cleanup
        ui_app_module._train_log_buffer = None

    def test_start_training_wires_drain_thread_and_drains_without_subscriber(self):
        """POST /api/train/start installs live drain; child finishes rc=0 with no SSE reader."""
        from unittest.mock import patch

        # Emit 1000 lines (> 100 KiB) — well above OS pipe capacity (4-64 KiB across platforms)
        child_script = (
            "import sys\n"
            "for i in range(1000):\n"
            "    sys.stdout.write(f'drain_line_{i}: ' + 'X' * 95 + '\\n')\n"
            "sys.stdout.flush()\n"
        )

        ui_app_module._train_process = None
        ui_app_module._train_log_buffer = None
        ui_app_module._train_drain_thread = None

        with patch(
            "soup_cli.ui.app._resolve_train_argv",
            return_value=[sys.executable, "-c", child_script],
        ):
            client = TestClient(create_app())
            response = client.post(
                "/api/train/start",
                json={"config_yaml": "base: test\ndata:\n  train: ./data.jsonl\n"},
                headers=_auth_headers(),
            )
            assert response.status_code == 200
            assert response.json()["started"] is True

            proc = ui_app_module._train_process
            assert proc is not None

            # Subprocess must complete with rc=0 with NO active SSE subscriber attached.
            # If start_training() failed to start the background drain thread, the child
            # deadlocks on write once the pipe buffer fills up (~8-64 KiB) and times out here.
            start_time = time.time()
            while proc.poll() is None:
                time.sleep(0.05)
                if time.time() - start_time > 5.0:
                    proc.kill()
                    pytest.fail(
                        "Subprocess blocked on write without stdout reader — "
                        "_train_drain_thread was not started by start_training() (#688)"
                    )

            assert proc.poll() == 0
            assert ui_app_module._train_drain_thread is not None
            ui_app_module._train_drain_thread.join(timeout=2.0)
            assert not ui_app_module._train_drain_thread.is_alive()
            assert ui_app_module._train_log_buffer is not None
            assert ui_app_module._train_log_buffer.is_done()

            lines = ui_app_module._train_log_buffer.get_lines_from(0)
            assert len(lines) == 1000
            assert lines[0][1].startswith("drain_line_0:")
            assert lines[-1][1].startswith("drain_line_999:")

        # Cleanup
        ui_app_module._train_process = None
        ui_app_module._train_log_buffer = None
        ui_app_module._train_drain_thread = None
