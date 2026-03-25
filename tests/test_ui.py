"""Tests for soup ui — Web UI command and API endpoints."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


def _auth_headers():
    """Return auth headers with the current UI token."""
    from soup_cli.ui.app import get_auth_token
    return {"Authorization": f"Bearer {get_auth_token()}"}


class TestUICommand:
    """Test the soup ui CLI command."""

    def test_ui_command_registered(self):
        """soup ui should be a registered command."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["ui", "--help"])
        assert result.exit_code == 0
        assert "web ui" in result.output.lower() or "experiments" in result.output.lower()

    def test_ui_command_fastapi_import_error(self):
        """soup ui should fail gracefully if FastAPI not installed."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        with patch.dict("sys.modules", {"fastapi": None, "uvicorn": None}):
            result = runner.invoke(app, ["ui"])
            assert result.exit_code != 0

    def test_ui_command_options(self):
        """soup ui should accept --port, --host, --no-browser options."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["ui", "--help"])
        # Rich markup wraps dashes with ANSI codes, so check without codes
        import re
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--port" in clean
        assert "--host" in clean
        assert "--no-browser" in clean


class TestCreateApp:
    """Test the FastAPI app creation."""

    def test_create_app_returns_fastapi_instance(self):
        """create_app should return a FastAPI app."""
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_has_required_routes(self):
        """App should have all required API routes."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]

        assert "/" in routes
        assert "/api/health" in routes
        assert "/api/runs" in routes
        assert "/api/runs/{run_id}" in routes
        assert "/api/runs/{run_id}/metrics" in routes
        assert "/api/runs/{run_id}/eval" in routes
        assert "/api/system" in routes
        assert "/api/templates" in routes
        assert "/api/config/validate" in routes
        assert "/api/train/start" in routes
        assert "/api/train/status" in routes
        assert "/api/train/stop" in routes
        assert "/api/data/inspect" in routes


class TestHealthEndpoint:
    """Test the /api/health endpoint."""

    def test_health_returns_ok(self):
        """Health endpoint should return ok status."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestIndexEndpoint:
    """Test the index page serving."""

    def test_index_returns_html(self):
        """Root should serve HTML page."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Soup" in response.text

    def test_index_contains_pages(self):
        """Index HTML should contain all page sections."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.get("/")
        html = response.text
        assert "page-dashboard" in html
        assert "page-training" in html
        assert "page-data" in html
        assert "page-chat" in html


class TestSystemEndpoint:
    """Test the /api/system endpoint."""

    def test_system_info(self):
        """System endpoint should return version and device info."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.get("/api/system")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "device" in data
        assert "device_name" in data
        assert "gpu_info" in data
        assert "python_version" in data


class TestTemplatesEndpoint:
    """Test the /api/templates endpoint."""

    def test_list_templates(self):
        """Templates endpoint should return all built-in templates."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.get("/api/templates")
        assert response.status_code == 200
        data = response.json()
        templates = data["templates"]
        assert "chat" in templates
        assert "code" in templates
        assert "reasoning" in templates
        assert "vision" in templates
        assert "medical" in templates

    def test_templates_contain_yaml(self):
        """Each template should contain valid YAML content."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.get("/api/templates")
        templates = response.json()["templates"]
        for name, yaml_str in templates.items():
            assert "base:" in yaml_str, f"Template {name} missing 'base:'"
            assert "data:" in yaml_str, f"Template {name} missing 'data:'"


class TestConfigValidation:
    """Test the /api/config/validate endpoint."""

    def test_valid_config(self):
        """Should validate a correct config."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        yaml_str = """
base: meta-llama/Llama-3.1-8B
data:
  train: ./data/train.jsonl
  format: alpaca
"""
        response = client.post(
            "/api/config/validate",
            json={"yaml": yaml_str},
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "config" in data
        assert data["config"]["base"] == "meta-llama/Llama-3.1-8B"

    def test_invalid_config(self):
        """Should report errors for invalid config."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/config/validate",
            json={"yaml": "invalid: true"},
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "error" in data

    def test_empty_config(self):
        """Should reject empty config."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/config/validate",
            json={"yaml": ""},
            headers=_auth_headers(),
        )
        assert response.status_code == 400


class TestRunsEndpoint:
    """Test the /api/runs endpoints."""

    def test_list_runs_empty(self, tmp_path):
        """Should return empty list when no runs exist."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.ui.app import create_app

            client = TestClient(create_app())
            response = client.get("/api/runs")
            assert response.status_code == 200
            assert response.json()["runs"] == []

    def test_list_runs_with_data(self, tmp_path):
        """Should return runs from the database."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.experiment.tracker import ExperimentTracker
            from soup_cli.ui.app import create_app

            # Create a run
            tracker = ExperimentTracker(db_path=db_path)
            run_id = tracker.start_run(
                config_dict={"base": "test-model", "task": "sft"},
                device="cpu",
                device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.close()

            client = TestClient(create_app())
            response = client.get("/api/runs")
            assert response.status_code == 200
            runs = response.json()["runs"]
            assert len(runs) == 1
            assert runs[0]["run_id"] == run_id

    def test_get_run_detail(self, tmp_path):
        """Should return run details by ID."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.experiment.tracker import ExperimentTracker
            from soup_cli.ui.app import create_app

            tracker = ExperimentTracker(db_path=db_path)
            run_id = tracker.start_run(
                config_dict={"base": "llama", "task": "sft"},
                device="cpu",
                device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.close()

            client = TestClient(create_app())
            response = client.get(f"/api/runs/{run_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["run_id"] == run_id
            assert data["base_model"] == "llama"

    def test_get_run_not_found(self, tmp_path):
        """Should return 404 for nonexistent run."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.ui.app import create_app

            client = TestClient(create_app())
            response = client.get("/api/runs/nonexistent_run_id")
            assert response.status_code == 404

    def test_delete_run(self, tmp_path):
        """Should delete a run and return success."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.experiment.tracker import ExperimentTracker
            from soup_cli.ui.app import create_app

            tracker = ExperimentTracker(db_path=db_path)
            run_id = tracker.start_run(
                config_dict={"base": "test"},
                device="cpu",
                device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.close()

            client = TestClient(create_app())
            response = client.delete(
                f"/api/runs/{run_id}", headers=_auth_headers()
            )
            assert response.status_code == 200
            assert response.json()["deleted"] is True

            # Verify deleted
            response = client.get(f"/api/runs/{run_id}")
            assert response.status_code == 404

    def test_delete_run_not_found(self, tmp_path):
        """Should return 404 when deleting nonexistent run."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.ui.app import create_app

            client = TestClient(create_app())
            response = client.delete(
                "/api/runs/nonexistent", headers=_auth_headers()
            )
            assert response.status_code == 404


class TestRunMetrics:
    """Test the /api/runs/{id}/metrics endpoint."""

    def test_get_metrics(self, tmp_path):
        """Should return metrics for a run."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.experiment.tracker import ExperimentTracker
            from soup_cli.ui.app import create_app

            tracker = ExperimentTracker(db_path=db_path)
            run_id = tracker.start_run(
                config_dict={"base": "test"},
                device="cpu",
                device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.log_metrics(run_id, step=10, loss=2.5, lr=1e-5)
            tracker.log_metrics(run_id, step=20, loss=2.0, lr=9e-6)
            tracker.log_metrics(run_id, step=30, loss=1.5, lr=8e-6)
            tracker.close()

            client = TestClient(create_app())
            response = client.get(f"/api/runs/{run_id}/metrics")
            assert response.status_code == 200
            data = response.json()
            assert data["run_id"] == run_id
            assert len(data["metrics"]) == 3
            assert data["metrics"][0]["step"] == 10
            assert data["metrics"][0]["loss"] == 2.5
            assert data["metrics"][2]["step"] == 30

    def test_get_metrics_not_found(self, tmp_path):
        """Should return 404 for metrics of nonexistent run."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.ui.app import create_app

            client = TestClient(create_app())
            response = client.get("/api/runs/nonexistent/metrics")
            assert response.status_code == 404


class TestRunEval:
    """Test the /api/runs/{id}/eval endpoint."""

    def test_get_eval_results(self, tmp_path):
        """Should return eval results for a run."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.experiment.tracker import ExperimentTracker
            from soup_cli.ui.app import create_app

            tracker = ExperimentTracker(db_path=db_path)
            run_id = tracker.start_run(
                config_dict={"base": "test"},
                device="cpu",
                device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.save_eval_result(
                model_path="./output",
                benchmark="mmlu",
                score=0.75,
                details={"subjects": {"math": 0.8}},
                run_id=run_id,
            )
            tracker.close()

            client = TestClient(create_app())
            response = client.get(f"/api/runs/{run_id}/eval")
            assert response.status_code == 200
            data = response.json()
            assert len(data["eval_results"]) == 1
            assert data["eval_results"][0]["benchmark"] == "mmlu"
            assert data["eval_results"][0]["score"] == 0.75


class TestDataInspect:
    """Test the /api/data/inspect endpoint."""

    def test_inspect_jsonl(self, tmp_path):
        """Should inspect a JSONL file and return entries."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        data_file = tmp_path / "train.jsonl"
        entries = [
            {"instruction": "Say hi", "input": "", "output": "Hello!"},
            {"instruction": "Say bye", "input": "", "output": "Goodbye!"},
            {"instruction": "Count", "input": "to 3", "output": "1, 2, 3"},
        ]
        data_file.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

        client = TestClient(create_app())
        with patch("soup_cli.ui.app.Path.cwd", return_value=tmp_path):
            response = client.post(
                "/api/data/inspect",
                json={"path": str(data_file), "limit": 10},
                headers=_auth_headers(),
            )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["format"] == "alpaca"
        assert "instruction" in data["keys"]
        assert len(data["sample"]) == 3

    def test_inspect_file_not_found(self, tmp_path):
        """Should return 404 for nonexistent file."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        with patch("soup_cli.ui.app.Path.cwd", return_value=tmp_path):
            response = client.post(
                "/api/data/inspect",
                json={"path": str(tmp_path / "nonexistent.jsonl")},
                headers=_auth_headers(),
            )
        assert response.status_code == 404

    def test_inspect_with_limit(self, tmp_path):
        """Should respect the limit parameter."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        data_file = tmp_path / "data.jsonl"
        entries = [
            {"instruction": f"Task {i}", "input": "", "output": f"Result {i}"}
            for i in range(20)
        ]
        data_file.write_text(
            "\n".join(json.dumps(e) for e in entries), encoding="utf-8"
        )

        client = TestClient(create_app())
        with patch("soup_cli.ui.app.Path.cwd", return_value=tmp_path):
            response = client.post(
                "/api/data/inspect",
                json={"path": str(data_file), "limit": 5},
                headers=_auth_headers(),
            )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 20
        assert len(data["sample"]) == 5

    def test_inspect_json_file(self, tmp_path):
        """Should inspect a JSON array file."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        data_file = tmp_path / "data.json"
        entries = [
            {"instruction": "Hi", "input": "", "output": "Hello"},
        ]
        data_file.write_text(json.dumps(entries), encoding="utf-8")

        client = TestClient(create_app())
        with patch("soup_cli.ui.app.Path.cwd", return_value=tmp_path):
            response = client.post(
                "/api/data/inspect",
                json={"path": str(data_file)},
                headers=_auth_headers(),
            )
        assert response.status_code == 200
        assert response.json()["total"] == 1


class TestTrainEndpoints:
    """Test the /api/train/* endpoints."""

    def test_train_status_not_running(self):
        """Should report not running when no training in progress."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_app_module
        from soup_cli.ui.app import create_app

        # Reset global state
        ui_app_module._train_process = None

        client = TestClient(create_app())
        response = client.get("/api/train/status")
        assert response.status_code == 200
        assert response.json()["running"] is False

    def test_stop_training_not_running(self):
        """Should report no training to stop."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_app_module
        from soup_cli.ui.app import create_app

        ui_app_module._train_process = None

        client = TestClient(create_app())
        response = client.post("/api/train/stop", headers=_auth_headers())
        assert response.status_code == 200
        assert response.json()["stopped"] is False

    def test_start_training(self, tmp_path):
        """Should start training subprocess."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_app_module
        from soup_cli.ui.app import create_app

        ui_app_module._train_process = None

        mock_popen = MagicMock()
        mock_popen.poll.return_value = None
        mock_popen.pid = 12345

        with patch("soup_cli.ui.app.subprocess.Popen", return_value=mock_popen):
            client = TestClient(create_app())
            response = client.post(
                "/api/train/start",
                json={
                    "config_yaml": "base: test\ndata:\n  train: ./data.jsonl\n"
                },
                headers=_auth_headers(),
            )
            assert response.status_code == 200
            data = response.json()
            assert data["started"] is True
            assert data["pid"] == 12345

        # Cleanup
        ui_app_module._train_process = None

    def test_start_training_conflict(self, tmp_path):
        """Should reject second training when one is running."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        import soup_cli.ui.app as ui_app_module
        from soup_cli.ui.app import create_app

        # Simulate running process
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        ui_app_module._train_process = mock_proc

        client = TestClient(create_app())
        response = client.post(
            "/api/train/start",
            json={
                "config_yaml": "base: test\ndata:\n  train: ./data.jsonl\n"
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 409

        # Cleanup
        ui_app_module._train_process = None


class TestConfigLoader:
    """Test the load_config_from_string helper."""

    def test_load_valid_config(self):
        """Should parse valid YAML config."""
        from soup_cli.config.loader import load_config_from_string

        config = load_config_from_string("""
base: meta-llama/Llama-3.1-8B
data:
  train: ./data/train.jsonl
  format: alpaca
""")
        assert config.base == "meta-llama/Llama-3.1-8B"
        assert config.data.train == "./data/train.jsonl"

    def test_load_empty_config(self):
        """Should raise ValueError for empty config."""
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="empty"):
            load_config_from_string("")

    def test_load_invalid_config(self):
        """Should raise ValueError for invalid config."""
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError):
            load_config_from_string("invalid: true\nno_base: 1")

    def test_load_config_with_all_fields(self):
        """Should parse config with all fields."""
        from soup_cli.config.loader import load_config_from_string

        config = load_config_from_string("""
base: codellama/CodeLlama-7b-Instruct-hf
task: sft
data:
  train: ./data.jsonl
  format: sharegpt
  val_split: 0.15
  max_length: 4096
training:
  epochs: 5
  lr: 1e-5
  lora:
    r: 128
    alpha: 32
output: ./my_output
""")
        assert config.task == "sft"
        assert config.data.val_split == 0.15
        assert config.training.epochs == 5
        assert config.training.lora.r == 128
        assert config.output == "./my_output"


class TestStaticFiles:
    """Test static file serving."""

    def test_static_css(self):
        """Should serve CSS file."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.get("/static/style.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]

    def test_static_js(self):
        """Should serve JavaScript file."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.get("/static/app.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]

    def test_static_dir_exists(self):
        """Static directory should exist with required files."""
        from soup_cli.ui.app import STATIC_DIR

        assert STATIC_DIR.exists()
        assert (STATIC_DIR / "index.html").exists()
        assert (STATIC_DIR / "style.css").exists()
        assert (STATIC_DIR / "app.js").exists()


class TestRunsLimitParam:
    """Test the limit query parameter for runs."""

    def test_runs_limit_default(self, tmp_path):
        """Should use default limit of 50."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.ui.app import create_app

            client = TestClient(create_app())
            response = client.get("/api/runs")
            assert response.status_code == 200

    def test_runs_custom_limit(self, tmp_path):
        """Should accept custom limit parameter."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.ui.app import create_app

            client = TestClient(create_app())
            response = client.get("/api/runs?limit=10")
            assert response.status_code == 200
