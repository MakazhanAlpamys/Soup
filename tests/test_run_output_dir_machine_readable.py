"""#2725 follow-up: machine-readable surfaces relativize or shorten output_dir.

The tracker stores output_dir as an absolute path so CLI readers find a run's
artifacts from any directory. Machine-readable surfaces (FastAPI UI endpoints
and MCP server tools) must not expose more local filesystem layout than before:
they show the path relative to the directory the server was started in when the
output directory is under it, and only the last path component otherwise.

CLI commands (`soup runs show`) keep printing the full stored path.
The database storage itself is unchanged.
"""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.mcp_server import registry as reg
from soup_cli.utils.paths import sanitize_run_output_dir, sanitize_run_row
from tests.conftest import strip_ansi
from tests.ui_helpers import ui_client

runner = CliRunner()


def _auth_headers():
    from soup_cli.ui.app import get_auth_token

    return {"Authorization": f"Bearer {get_auth_token()}"}


def _create_finished_run(db_path: Path, output_dir: str | None) -> str:
    tracker = ExperimentTracker(db_path=db_path)
    try:
        run_id = tracker.start_run(
            config_dict={"base": "test-model", "task": "sft"},
            device="cpu",
            device_name="CPU",
            gpu_info={"memory_total": "N/A"},
        )
        tracker.finish_run(
            run_id=run_id,
            initial_loss=1.0,
            final_loss=0.5,
            total_steps=10,
            duration_secs=5.0,
            output_dir=output_dir,
        )
        return run_id
    finally:
        tracker.close()


class TestSanitizeRunOutputDir:
    def test_output_dir_none_or_empty(self, tmp_path: Path):
        assert sanitize_run_output_dir(None, base=tmp_path) is None
        assert sanitize_run_output_dir("", base=tmp_path) == ""

    def test_output_dir_under_base(self, tmp_path: Path):
        output = tmp_path / "output"
        assert sanitize_run_output_dir(str(output), base=tmp_path) == "output"

    def test_output_dir_nested_under_base(self, tmp_path: Path):
        output = tmp_path / "runs" / "exp1" / "output"
        expected = os.path.join("runs", "exp1", "output")
        assert sanitize_run_output_dir(str(output), base=tmp_path) == expected

    def test_output_dir_same_as_base(self, tmp_path: Path):
        assert sanitize_run_output_dir(str(tmp_path), base=tmp_path) == "."

    def test_output_dir_outside_base(self, tmp_path: Path):
        outside = tmp_path / "other_project" / "custom_output"
        base = tmp_path / "my_project"
        assert sanitize_run_output_dir(str(outside), base=base) == "custom_output"

    def test_output_dir_outside_base_trailing_slash(self, tmp_path: Path):
        outside = str(tmp_path / "other" / "output") + os.sep
        assert sanitize_run_output_dir(outside, base=tmp_path / "my_project") == "output"

    def test_legacy_relative_output_dir(self, tmp_path: Path):
        assert sanitize_run_output_dir("output", base=tmp_path) == "output"

    def test_sanitize_run_row(self, tmp_path: Path):
        row = {"run_id": "r1", "status": "completed", "output_dir": str(tmp_path / "output")}
        sanitized = sanitize_run_row(row, base=tmp_path)
        assert sanitized["output_dir"] == "output"
        assert sanitized["run_id"] == "r1"
        assert sanitized["status"] == "completed"
        # Original row is not mutated
        assert row["output_dir"] == str(tmp_path / "output")

    def test_sanitize_run_row_none_or_missing(self, tmp_path: Path):
        assert sanitize_run_row(None, base=tmp_path) is None
        row = {"run_id": "r1"}
        assert sanitize_run_row(row, base=tmp_path) == {"run_id": "r1"}


class TestUiApiRunsOutputDir:
    @pytest.fixture(autouse=True)
    def setup_env(self, tmp_path: Path, monkeypatch):
        self.db_path = tmp_path / "experiments.db"
        self.project_dir = tmp_path / "project"
        self.project_dir.mkdir()
        self.outside_dir = tmp_path / "outside"
        self.outside_dir.mkdir()
        monkeypatch.setenv("SOUP_DB_PATH", str(self.db_path))
        monkeypatch.chdir(self.project_dir)

    def test_get_api_runs_under_and_outside_working_dir(self):
        from soup_cli.ui.app import create_app

        run_under = _create_finished_run(self.db_path, str(self.project_dir / "output"))
        run_outside = _create_finished_run(self.db_path, str(self.outside_dir / "models_out"))

        client = ui_client(create_app(server_dir=self.project_dir))
        response = client.get("/api/runs", headers=_auth_headers())
        assert response.status_code == 200

        runs = {r["run_id"]: r["output_dir"] for r in response.json()["runs"]}
        assert runs[run_under] == "output"
        assert runs[run_outside] == "models_out"

    def test_get_api_runs_by_id_under_working_dir(self):
        from soup_cli.ui.app import create_app

        run_under = _create_finished_run(self.db_path, str(self.project_dir / "output"))

        client = ui_client(create_app(server_dir=self.project_dir))
        response = client.get(f"/api/runs/{run_under}", headers=_auth_headers())
        assert response.status_code == 200
        assert response.json()["output_dir"] == "output"

    def test_get_api_runs_by_id_outside_working_dir(self):
        from soup_cli.ui.app import create_app

        run_outside = _create_finished_run(self.db_path, str(self.outside_dir / "models_out"))

        client = ui_client(create_app(server_dir=self.project_dir))
        response = client.get(f"/api/runs/{run_outside}", headers=_auth_headers())
        assert response.status_code == 200
        assert response.json()["output_dir"] == "models_out"


class TestMcpRunsOutputDir:
    @pytest.fixture(autouse=True)
    def setup_env(self, tmp_path: Path, monkeypatch):
        self.db_path = tmp_path / "experiments.db"
        self.project_dir = tmp_path / "project"
        self.project_dir.mkdir()
        self.outside_dir = tmp_path / "outside"
        self.outside_dir.mkdir()
        monkeypatch.setenv("SOUP_DB_PATH", str(self.db_path))
        monkeypatch.chdir(self.project_dir)

    def test_runs_list_under_and_outside_working_dir(self):
        run_under = _create_finished_run(self.db_path, str(self.project_dir / "output"))
        run_outside = _create_finished_run(self.db_path, str(self.outside_dir / "mcp_out"))

        out = reg.tool_runs_list({})
        assert out["count"] >= 2
        runs = {r["run_id"]: r["output_dir"] for r in out["runs"]}
        assert runs[run_under] == "output"
        assert runs[run_outside] == "mcp_out"

    def test_runs_show_under_working_dir(self):
        run_under = _create_finished_run(self.db_path, str(self.project_dir / "output"))

        out = reg.tool_runs_show({"run_id": run_under})
        assert out["output_dir"] == "output"

    def test_runs_show_outside_working_dir(self):
        run_outside = _create_finished_run(self.db_path, str(self.outside_dir / "mcp_out"))

        out = reg.tool_runs_show({"run_id": run_outside})
        assert out["output_dir"] == "mcp_out"


class TestCliRunsShowAndStorageUnchanged:
    @pytest.fixture(autouse=True)
    def setup_env(self, tmp_path: Path, monkeypatch):
        from rich.console import Console

        self.db_path = tmp_path / "experiments.db"
        self.project_dir = tmp_path / "project"
        self.project_dir.mkdir()
        self.outside_dir = tmp_path / "outside"
        self.outside_dir.mkdir()
        monkeypatch.setenv("SOUP_DB_PATH", str(self.db_path))
        monkeypatch.setenv("COLUMNS", "500")
        monkeypatch.setattr("soup_cli.commands.runs.console", Console(width=500))
        monkeypatch.chdir(self.project_dir)

    def test_cli_runs_show_prints_full_path_for_run_under_cwd(self):
        full_path = str(self.project_dir / "output")
        run_id = _create_finished_run(self.db_path, full_path)

        result = runner.invoke(app, ["runs", "show", run_id])
        assert result.exit_code == 0
        plain_output = " ".join(strip_ansi(result.output).split())
        assert f"Output: {full_path}" in plain_output

    def test_cli_runs_show_prints_full_path_for_run_outside_cwd(self):
        full_path = str(self.outside_dir / "external_model")
        run_id = _create_finished_run(self.db_path, full_path)

        result = runner.invoke(app, ["runs", "show", run_id])
        assert result.exit_code == 0
        plain_output = " ".join(strip_ansi(result.output).split())
        assert f"Output: {full_path}" in plain_output

    def test_database_stores_absolute_path_unchanged(self):
        full_path = str(self.outside_dir / "stored_dir")
        run_id = _create_finished_run(self.db_path, full_path)

        tracker = ExperimentTracker(db_path=self.db_path)
        try:
            stored_run = tracker.get_run(run_id)
            assert stored_run["output_dir"] == full_path
        finally:
            tracker.close()
