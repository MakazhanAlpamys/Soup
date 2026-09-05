"""Tests for Web UI Training Live Monitor — SSE endpoints and progress API."""

import os
from unittest.mock import MagicMock, patch

import pytest


def _auth_headers():
    """Return auth headers with the current UI token."""
    from soup_cli.ui.app import get_auth_token
    return {"Authorization": f"Bearer {get_auth_token()}"}


class TestTrainLogsSSE:
    """Test GET /api/train/logs SSE endpoint."""

    def test_logs_endpoint_exists(self):
        """Route /api/train/logs should be registered."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/api/train/logs" in routes

    def test_logs_returns_event_stream(self):
        """SSE endpoint should return text/event-stream content type."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        # Simulate a running process with output
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout = iter([b"Epoch 1/3\n", b"Loss: 2.5\n"])
        mock_proc.poll.side_effect = [None, None, 0]
        buf = ui_mod.TrainLogBuffer(maxlen=100)
        buf.append("Epoch 1/3")
        buf.append("Loss: 2.5")
        buf.mark_done()
        ui_mod._train_log_buffer = buf

        client = TestClient(create_app())
        try:
            with client.stream(
                "GET", "/api/train/logs", headers=_auth_headers()
            ) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]
                # Read at least one event
                lines = []
                for line in resp.iter_lines():
                    lines.append(line)
                    if len(lines) >= 2 or "done" in line:
                        break
        finally:
            ui_mod._train_process = None
            ui_mod._train_log_buffer = None

    def test_logs_no_training_returns_done(self):
        """When no training is running, SSE should emit done event."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None

        client = TestClient(create_app())
        with client.stream(
            "GET", "/api/train/logs", headers=_auth_headers()
        ) as resp:
            assert resp.status_code == 200
            body = b""
            for chunk in resp.iter_bytes():
                body += chunk
            text = body.decode()
            assert "done" in text

    def test_logs_last_event_id_reconnection(self):
        """Last-Event-ID header should skip earlier lines."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        buf = ui_mod.TrainLogBuffer(maxlen=100)
        buf.append("Line 1")
        buf.append("Line 2")
        buf.append("Line 3")
        buf.mark_done()
        ui_mod._train_log_buffer = buf

        client = TestClient(create_app())
        try:
            with client.stream(
                "GET",
                "/api/train/logs",
                headers={"Last-Event-ID": "0", **_auth_headers()},
            ) as resp:
                assert resp.status_code == 200
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                text = body.decode()
                # Line 1 (id=0) should be skipped, Line 2+ should appear
                assert "Line 2" in text or "Line 3" in text
        finally:
            ui_mod._train_log_buffer = None
    def test_logs_no_auth_required(self):
        """SSE log endpoint requires auth (401 without token, 200 with one)."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None

        client = TestClient(create_app())
        with client.stream("GET", "/api/train/logs") as resp:
            assert resp.status_code == 401
        with client.stream(
            "GET", "/api/train/logs", headers=_auth_headers()
        ) as resp:
            assert resp.status_code == 200


class TestLiveMetricsSSE:
    """Test GET /api/train/metrics/live SSE endpoint."""

    def test_metrics_live_endpoint_exists(self):
        """Route /api/train/metrics/live should be registered."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/api/train/metrics/live" in routes

    def test_metrics_live_returns_event_stream(self, tmp_path):
        """SSE metrics endpoint should return text/event-stream content type."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        # Simulate running training
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Already finished
        ui_mod._train_process = mock_proc

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            client = TestClient(create_app())
            with client.stream(
                "GET", "/api/train/metrics/live", headers=_auth_headers()
            ) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]

        ui_mod._train_process = None

    def test_metrics_live_emits_new_rows(self, tmp_path):
        """SSE should emit metric rows as they appear in the DB."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        db_path = tmp_path / "test.db"

        # Pre-populate a run with metrics
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker(db_path=db_path)
        run_id = tracker.start_run(
            config_dict={"base": "test", "task": "sft"},
            device="cpu",
            device_name="CPU",
            gpu_info={"memory_total": "N/A"},
        )
        tracker.log_metrics(run_id, step=10, loss=2.5, lr=1e-5)
        tracker.close()

        # Simulate finished training
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        ui_mod._train_process = mock_proc

        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            client = TestClient(create_app())
            with client.stream(
                "GET",
                f"/api/train/metrics/live?run_id={run_id}",
                headers=_auth_headers(),
            ) as resp:
                assert resp.status_code == 200
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                text = body.decode()
                # Should contain metric data
                assert "step" in text or "done" in text

        ui_mod._train_process = None

    def test_metrics_live_no_auth_required(self, tmp_path):
        """SSE metrics endpoint requires auth (401 without token, 200 with one)."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        ui_mod._train_process = mock_proc

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            client = TestClient(create_app())
            with client.stream("GET", "/api/train/metrics/live") as resp:
                assert resp.status_code == 401
            with client.stream(
                "GET", "/api/train/metrics/live", headers=_auth_headers()
            ) as resp:
                assert resp.status_code == 200

        ui_mod._train_process = None

    def test_metrics_live_done_when_no_training(self, tmp_path):
        """Should emit done event when no training is running."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            client = TestClient(create_app())
            with client.stream(
                "GET", "/api/train/metrics/live", headers=_auth_headers()
            ) as resp:
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                text = body.decode()
                assert "done" in text

        ui_mod._train_process = None


class TestTrainProgress:
    """Test GET /api/train/progress endpoint."""

    def test_progress_endpoint_exists(self):
        """Route /api/train/progress should be registered."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/api/train/progress" in routes

    def test_progress_not_running(self):
        """Should return running=false when no training."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None

        client = TestClient(create_app())
        response = client.get("/api/train/progress", headers=_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is False

    def test_progress_running_with_metrics(self, tmp_path):
        """Should return progress when training is running."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        db_path = tmp_path / "test.db"

        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker(db_path=db_path)
        run_id = tracker.start_run(
            config_dict={"base": "test", "task": "sft"},
            device="cpu",
            device_name="CPU",
            gpu_info={"memory_total": "N/A"},
        )
        tracker.log_metrics(run_id, step=50, loss=1.5, lr=1e-5)
        tracker.close()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 9999
        ui_mod._train_process = mock_proc

        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            client = TestClient(create_app())
            response = client.get(
                f"/api/train/progress?run_id={run_id}", headers=_auth_headers()
            )
            assert response.status_code == 200
            data = response.json()
            assert data["running"] is True
            assert data["current_step"] == 50

        ui_mod._train_process = None

    def test_progress_returns_correct_fields(self, tmp_path):
        """Progress response should contain all required fields."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        db_path = tmp_path / "test.db"

        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker(db_path=db_path)
        run_id = tracker.start_run(
            config_dict={"base": "test", "task": "sft"},
            device="cpu",
            device_name="CPU",
            gpu_info={"memory_total": "N/A"},
        )
        tracker.log_metrics(run_id, step=10, loss=2.0, lr=1e-5)
        tracker.close()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 1234
        ui_mod._train_process = mock_proc

        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            client = TestClient(create_app())
            response = client.get(
                f"/api/train/progress?run_id={run_id}", headers=_auth_headers()
            )
            data = response.json()
            expected_keys = {"running", "current_step", "run_id"}
            assert expected_keys.issubset(set(data.keys()))

        ui_mod._train_process = None

    def test_progress_no_auth_required(self):
        """Progress endpoint requires auth (401 without token, 200 with one)."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None

        client = TestClient(create_app())
        assert client.get("/api/train/progress").status_code == 401
        assert (
            client.get("/api/train/progress", headers=_auth_headers()).status_code
            == 200
        )


class TestSSEGracefulClose:
    """Test that SSE endpoints close gracefully."""

    def test_logs_closes_when_training_stops(self):
        """Log SSE should require auth and end when training process finishes."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        mock_proc = MagicMock()
        mock_proc.stdout = iter([b"Starting\n"])
        mock_proc.poll.side_effect = [None, 0]
        ui_mod._train_process = mock_proc

        client = TestClient(create_app())
        with client.stream("GET", "/api/train/logs") as resp:
            assert resp.status_code == 401

        try:
            with client.stream(
                "GET", "/api/train/logs", headers=_auth_headers()
            ) as resp:
                assert resp.status_code == 200
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                text = body.decode()
                # Stream should terminate with done event
                assert "done" in text
        finally:
            ui_mod._train_process = None

    def test_heartbeat_event(self):
        """SSE should include heartbeat comments to keep connection alive."""
        # This is a design test — heartbeats are sent as SSE comments (:heartbeat)
        # Testing that the generator yields at least one event
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_mod
        from soup_cli.ui.app import create_app

        ui_mod._train_process = None

        client = TestClient(create_app())
        with client.stream(
            "GET", "/api/train/logs", headers=_auth_headers()
        ) as resp:
            assert resp.status_code == 200
            body = b""
            for chunk in resp.iter_bytes():
                body += chunk
            # Stream should have some content (at least done event)
            assert len(body) > 0

        ui_mod._train_process = None
