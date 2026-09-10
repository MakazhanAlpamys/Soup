"""Tests for Web UI Enhanced Metrics & Eval Display."""

import os
from unittest.mock import patch

import pytest


def _auth_headers():
    """Return auth headers with the current UI token."""
    from soup_cli.ui.app import get_auth_token
    return {"Authorization": f"Bearer {get_auth_token()}"}


class TestMetricsFullFields:
    """Test that metrics response includes all fields."""

    def test_metrics_includes_grad_norm(self, tmp_path):
        """Metrics should include grad_norm field."""
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
                config_dict={"base": "test", "task": "sft"},
                device="cpu",
                device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.log_metrics(
                run_id, step=10, loss=2.5, lr=1e-5,
                grad_norm=1.23, speed=100.5, gpu_mem="4.2GB",
            )
            tracker.close()

            client = TestClient(create_app())
            response = client.get(
                f"/api/runs/{run_id}/metrics", headers=_auth_headers()
            )
            assert response.status_code == 200
            metrics = response.json()["metrics"]
            assert len(metrics) == 1
            m = metrics[0]
            assert m["grad_norm"] == 1.23
            assert m["speed"] == 100.5
            assert m["gpu_mem"] == "4.2GB"

    def test_metrics_includes_epoch(self, tmp_path):
        """Metrics should include epoch field."""
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
                config_dict={"base": "test", "task": "sft"},
                device="cpu",
                device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.log_metrics(run_id, step=10, epoch=1.5, loss=2.0, lr=1e-5)
            tracker.close()

            client = TestClient(create_app())
            response = client.get(
                f"/api/runs/{run_id}/metrics", headers=_auth_headers()
            )
            metrics = response.json()["metrics"]
            assert metrics[0]["epoch"] == 1.5


class TestRunsCompareEndpoint:
    """Test GET /api/runs/compare endpoint."""

    def test_compare_endpoint_exists(self):
        """Route /api/runs/compare should be registered."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/api/runs/compare" in routes

    def test_compare_returns_metrics(self, tmp_path):
        """Compare endpoint should return metrics for multiple runs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.experiment.tracker import ExperimentTracker
            from soup_cli.ui.app import create_app

            tracker = ExperimentTracker(db_path=db_path)
            run1 = tracker.start_run(
                config_dict={"base": "model-a", "task": "sft"},
                device="cpu", device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.log_metrics(run1, step=10, loss=2.5, lr=1e-5)
            run2 = tracker.start_run(
                config_dict={"base": "model-b", "task": "dpo"},
                device="cpu", device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.log_metrics(run2, step=10, loss=1.8, lr=5e-6)
            tracker.close()

            client = TestClient(create_app())
            response = client.get(
                f"/api/runs/compare?ids={run1},{run2}", headers=_auth_headers()
            )
            assert response.status_code == 200
            data = response.json()
            assert "runs" in data
            assert len(data["runs"]) == 2
            assert data["runs"][0]["run_id"] == run1
            assert data["runs"][1]["run_id"] == run2
            assert len(data["runs"][0]["metrics"]) == 1
            assert len(data["runs"][1]["metrics"]) == 1

    def test_compare_rejects_too_many_runs(self, tmp_path):
        """Compare endpoint should reject >5 runs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.ui.app import create_app

            client = TestClient(create_app())
            ids = ",".join([f"run_{i}" for i in range(6)])
            response = client.get(
                f"/api/runs/compare?ids={ids}", headers=_auth_headers()
            )
            assert response.status_code == 400

    def test_compare_validates_run_ids(self, tmp_path):
        """Compare endpoint returns entries for nonexistent runs (empty metrics)."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test_validate.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.ui.app import create_app

            client = TestClient(create_app())
            response = client.get(
                "/api/runs/compare?ids=nonexistent1,nonexistent2",
                headers=_auth_headers(),
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["runs"]) == 2

    def test_compare_empty_ids(self, tmp_path):
        """Compare endpoint should reject empty ids."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test_empty.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.ui.app import create_app

            client = TestClient(create_app())
            response = client.get(
                "/api/runs/compare?ids=", headers=_auth_headers()
            )
            assert response.status_code == 400

    def test_compare_no_auth_required(self, tmp_path):
        """Compare requires auth (401 without token, 200 with one)."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test_noauth.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.ui.app import create_app

            client = TestClient(create_app())
            assert client.get("/api/runs/compare?ids=run1,run2").status_code == 401
            assert (
                client.get(
                    "/api/runs/compare?ids=run1,run2", headers=_auth_headers()
                ).status_code
                == 200
            )


class TestEvalResultsEndpoint:
    """Test eval results display with parsed details."""

    def test_eval_results_include_details(self, tmp_path):
        """Eval results should include parsed details_json."""
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
                device="cpu", device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.save_eval_result(
                model_path="./output",
                benchmark="mmlu",
                score=0.75,
                details={"subjects": {"math": 0.8, "science": 0.7}},
                run_id=run_id,
            )
            tracker.close()

            client = TestClient(create_app())
            response = client.get(
                f"/api/runs/{run_id}/eval", headers=_auth_headers()
            )
            data = response.json()
            assert len(data["eval_results"]) == 1
            result = data["eval_results"][0]
            assert result["benchmark"] == "mmlu"
            assert result["score"] == 0.75
            # details_json should be present as a string (from SQLite)
            assert "details_json" in result

    def test_eval_results_empty(self, tmp_path):
        """Empty eval results should return empty array, not error."""
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
                device="cpu", device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.close()

            client = TestClient(create_app())
            response = client.get(
                f"/api/runs/{run_id}/eval", headers=_auth_headers()
            )
            assert response.status_code == 200
            data = response.json()
            assert data["eval_results"] == []

    def test_compare_max_five_runs(self, tmp_path):
        """Compare endpoint allows exactly 5 runs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            from soup_cli.experiment.tracker import ExperimentTracker
            from soup_cli.ui.app import create_app

            tracker = ExperimentTracker(db_path=db_path)
            run_ids = []
            for idx in range(5):
                rid = tracker.start_run(
                    config_dict={"base": f"model-{idx}"},
                    device="cpu", device_name="CPU",
                    gpu_info={"memory_total": "N/A"},
                )
                run_ids.append(rid)
            tracker.close()

            client = TestClient(create_app())
            ids_str = ",".join(run_ids)
            response = client.get(
                f"/api/runs/compare?ids={ids_str}", headers=_auth_headers()
            )
            assert response.status_code == 200
            assert len(response.json()["runs"]) == 5


class TestCompareMetricsContent:
    """Test compare endpoint returns full metric data."""

    def test_compare_includes_all_metric_fields(self, tmp_path):
        """Compare should return metrics with all fields."""
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
                config_dict={"base": "test", "task": "sft"},
                device="cpu", device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.log_metrics(
                run_id, step=10, epoch=1.0, loss=2.5, lr=1e-5,
                grad_norm=0.5, speed=200.0, gpu_mem="8GB",
            )
            tracker.close()

            client = TestClient(create_app())
            response = client.get(
                f"/api/runs/compare?ids={run_id}", headers=_auth_headers()
            )
            data = response.json()
            m = data["runs"][0]["metrics"][0]
            assert m["step"] == 10
            assert m["loss"] == 2.5
            assert m["grad_norm"] == 0.5
            assert m["speed"] == 200.0

    def test_compare_includes_config(self, tmp_path):
        """Compare should include run config for diff display."""
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
                config_dict={"base": "llama-8b", "task": "sft"},
                device="cpu", device_name="CPU",
                gpu_info={"memory_total": "N/A"},
            )
            tracker.close()

            client = TestClient(create_app())
            response = client.get(
                f"/api/runs/compare?ids={run_id}", headers=_auth_headers()
            )
            data = response.json()
            assert "config" in data["runs"][0]
