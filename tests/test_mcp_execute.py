"""Tests for MCP execution capability (Part E - Execution Slice)."""

import os
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

import soup_cli.mcp_server.registry as reg
from soup_cli.mcp_server.execution import ExecutionError, ExecutionManager

_MIN_CONFIG = "base: Qwen/Qwen2.5-0.5B\ntask: sft\ndata:\n  train: data.jsonl\n"


def _spec(name: str, *, allow_mutating: bool = False, allow_execute: bool = False, execution=None):
    specs = reg.build_registry(
        allow_mutating=allow_mutating, allow_execute=allow_execute, execution=execution
    )
    for s in specs:
        if s.name == name:
            return s
    raise ValueError(f"unknown spec {name}")


class TestMcpExecutionGates:
    def test_execution_disabled_by_default_refuses_execute_tools(self):
        with pytest.raises(reg.McpToolError) as exc:
            _spec("train_execute", allow_mutating=False, allow_execute=False).handler(
                {"confirmation_token": "token123"}
            )
        assert "allow-execute" in str(exc.value).lower()

        with pytest.raises(reg.McpToolError) as exc:
            _spec("export_execute", allow_mutating=False, allow_execute=False).handler(
                {"confirmation_token": "token123"}
            )
        assert "allow-execute" in str(exc.value).lower()

    def test_allow_mutating_alone_cannot_execute(self):
        with pytest.raises(reg.McpToolError) as exc:
            _spec("train_execute", allow_mutating=True, allow_execute=False).handler(
                {"confirmation_token": "token123"}
            )
        assert "allow-execute" in str(exc.value).lower()

        with pytest.raises(reg.McpToolError) as exc:
            _spec("export_execute", allow_mutating=True, allow_execute=False).handler(
                {"confirmation_token": "token123"}
            )
        assert "allow-execute" in str(exc.value).lower()

    def test_allow_execute_enables_execution(self):
        manager = ExecutionManager()
        manager.issue(kind="train", argv=["soup"], display_command="soup")
        spec = _spec("train_execute", allow_execute=True, execution=manager)
        assert spec.mutating is True
        assert spec.annotations == {"readOnlyHint": False, "destructiveHint": True}

    def test_tokens_issued_only_when_execution_enabled(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_MIN_CONFIG, encoding="utf-8")
        (tmp_path / "model.bin").write_text("weights", encoding="utf-8")

        # allow_mutating=True alone -> no token
        out1 = _spec("train_start", allow_mutating=True, allow_execute=False).handler(
            {"config": "soup.yaml"}
        )
        assert "confirmation_token" not in out1

        out_exp1 = _spec("export", allow_mutating=True, allow_execute=False).handler(
            {"model": "model.bin", "format": "gguf"}
        )
        assert "confirmation_token" not in out_exp1

        # allow_execute=True -> issues token
        manager = ExecutionManager()
        out2 = _spec("train_start", allow_execute=True, execution=manager).handler(
            {"config": "soup.yaml"}
        )
        assert "confirmation_token" in out2
        assert isinstance(out2["confirmation_token"], str)

        out_exp2 = _spec("export", allow_execute=True, execution=manager).handler(
            {"model": "model.bin", "format": "gguf"}
        )
        assert "confirmation_token" in out_exp2
        assert isinstance(out_exp2["confirmation_token"], str)


class TestTokenSecurity:
    def test_valid_token_executes_and_replayed_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        token = manager.issue(
            kind="train", argv=[sys.executable, "--version"], display_command="test"
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 1234
            mock_proc.wait.return_value = 0
            mock_popen.return_value = mock_proc

            res = manager.execute(token=token, kind="train")
            assert res["status"] == "running"
            assert res["pid"] == 1234

        # Replayed token rejected
        with pytest.raises(ExecutionError) as exc:
            manager.execute(token=token, kind="train")
        assert "consumed" in str(exc.value).lower()

    def test_invalid_token_rejected(self):
        manager = ExecutionManager()
        with pytest.raises(ExecutionError) as exc:
            manager.execute(token="invalid_token_xyz", kind="train")
        assert "unknown or expired" in str(exc.value).lower()

    def test_expired_token_rejected(self):
        manager = ExecutionManager(ttl_seconds=0)
        token = manager.issue(kind="train", argv=["echo"], display_command="test")
        time.sleep(0.01)
        with pytest.raises(ExecutionError) as exc:
            manager.execute(token=token, kind="train")
        assert "unknown or expired" in str(exc.value).lower()

    def test_wrong_tool_token_rejected(self):
        manager = ExecutionManager()
        token = manager.issue(kind="train", argv=["echo"], display_command="test")
        with pytest.raises(ExecutionError) as exc:
            manager.execute(token=token, kind="export")
        assert "not valid for this execution tool" in str(exc.value).lower()

    def test_identical_plans_receive_different_tokens(self):
        manager = ExecutionManager()
        t1 = manager.issue(kind="train", argv=["echo"], display_command="test")
        t2 = manager.issue(kind="train", argv=["echo"], display_command="test")
        assert t1 != t2

    def test_protected_input_mutation_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = tmp_path / "soup.yaml"
        cfg_path.write_text(_MIN_CONFIG, encoding="utf-8")

        manager = ExecutionManager()
        spec = _spec("train_start", allow_execute=True, execution=manager)
        out = spec.handler({"config": "soup.yaml"})
        token = out["confirmation_token"]

        # Mutate the file
        cfg_path.write_text(
            "base: altered-model\ntask: sft\ndata:\n  train: data.jsonl\n", encoding="utf-8"
        )

        with pytest.raises(ExecutionError) as exc:
            manager.execute(token=token, kind="train")
        assert "planned input changed" in str(exc.value).lower()


class TestSubprocessIsolationAndConcurrency:
    def test_subprocess_args_and_environment_isolation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        token = manager.issue(
            kind="train",
            argv=[sys.executable, "-m", "soup_cli.cli", "train"],
            display_command="soup train",
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 9999
            mock_proc.wait.return_value = 0
            mock_popen.return_value = mock_proc

            res = manager.execute(token=token, kind="train")
            assert res["pid"] == 9999

            assert mock_popen.called
            call_kwargs = mock_popen.call_args.kwargs
            call_args = mock_popen.call_args.args[0]

            assert call_args == [sys.executable, "-m", "soup_cli.cli", "train"]
            assert call_kwargs["shell"] is False
            assert call_kwargs["stdin"] == subprocess.DEVNULL
            assert call_kwargs["cwd"] == str(tmp_path)
            assert "SOUP_MCP_RUN_ID" in call_kwargs["env"]

    def test_export_execution_end_to_end_isolation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model_file = tmp_path / "model.bin"
        model_file.write_text("weights", encoding="utf-8")
        model_real = os.path.realpath(str(model_file))

        manager = ExecutionManager()
        export_spec = _spec("export", allow_execute=True, execution=manager)
        execute_spec = _spec("export_execute", allow_execute=True, execution=manager)

        plan_res = export_spec.handler({"model": "model.bin", "format": "gguf"})
        assert "confirmation_token" in plan_res
        token = plan_res["confirmation_token"]

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 8888
            mock_proc.wait.return_value = 0
            mock_popen.return_value = mock_proc

            exec_res = execute_spec.handler({"confirmation_token": token})
            assert exec_res["status"] == "running"
            assert exec_res["pid"] == 8888

            assert mock_popen.called
            call_args = mock_popen.call_args.args[0]
            call_kwargs = mock_popen.call_args.kwargs

            assert call_args == [
                sys.executable,
                "-m",
                "soup_cli.cli",
                "export",
                "--model",
                model_real,
                "--format",
                "gguf",
            ]
            assert call_kwargs["shell"] is False
            assert call_kwargs["stdin"] == subprocess.DEVNULL
            assert call_kwargs["cwd"] == str(tmp_path)
            assert "SOUP_MCP_RUN_ID" in call_kwargs["env"]

    def test_client_cannot_pass_extra_args_to_execute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        spec = _spec("train_execute", allow_execute=True, execution=manager)
        with pytest.raises(reg.McpToolError) as exc:
            spec.handler({"confirmation_token": "tok", "command": "rm -rf /"})
        assert "requires only 'confirmation_token'" in str(exc.value)

    def test_one_active_execution_per_server_process(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        t1 = manager.issue(kind="train", argv=["sleep", "10"], display_command="test1")
        t2 = manager.issue(kind="train", argv=["sleep", "10"], display_command="test2")

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 1001
            mock_proc.wait.side_effect = lambda: time.sleep(0.5) or 0
            mock_popen.return_value = mock_proc

            manager.execute(token=t1, kind="train")

            # Second execution refused
            with pytest.raises(ExecutionError) as exc:
                manager.execute(token=t2, kind="train")
            assert "already active" in str(exc.value).lower()

    def test_capacity_released_after_completion(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        t1 = manager.issue(kind="train", argv=["echo"], display_command="test1")
        t2 = manager.issue(kind="train", argv=["echo"], display_command="test2")

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 1001
            mock_proc.wait.return_value = 0
            mock_popen.return_value = mock_proc

            manager.execute(token=t1, kind="train")
            time.sleep(0.05)  # Let watcher thread complete

            # Second execution now succeeds
            res2 = manager.execute(token=t2, kind="train")
            assert res2["status"] == "running"

    def test_capacity_released_after_spawn_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        t1 = manager.issue(kind="train", argv=["nonexistent_executable"], display_command="test1")
        t2 = manager.issue(kind="train", argv=["echo"], display_command="test2")

        with patch("subprocess.Popen", side_effect=OSError("spawn failed")):
            with pytest.raises(ExecutionError) as exc:
                manager.execute(token=t1, kind="train")
            assert "could not spawn" in str(exc.value).lower()

        # Capacity was released on spawn failure
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 1002
            mock_proc.wait.return_value = 0
            mock_popen.return_value = mock_proc

            res2 = manager.execute(token=t2, kind="train")
            assert res2["status"] == "running"

    def test_capacity_released_after_tracker_launch_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        t1 = manager.issue(kind="train", argv=["echo"], display_command="test1")
        t2 = manager.issue(kind="train", argv=["echo"], display_command="test2")

        with patch(
            "soup_cli.experiment.tracker.ExperimentTracker.launch_run",
            side_effect=RuntimeError("db failed"),
        ):
            with pytest.raises(ExecutionError) as exc:
                manager.execute(token=t1, kind="train")
            assert "could not spawn" in str(exc.value).lower()

        # Capacity was released on tracker launch failure
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 1003
            mock_proc.wait.return_value = 0
            mock_popen.return_value = mock_proc

            res2 = manager.execute(token=t2, kind="train")
            assert res2["status"] == "running"


class TestRunTrackingIntegration:
    def test_run_tracking_updates_on_launch_running_finish(self, tmp_path, monkeypatch):
        from soup_cli.experiment.tracker import ExperimentTracker

        db_path = tmp_path / "exp.db"
        monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
        monkeypatch.chdir(tmp_path)

        manager = ExecutionManager()
        token = manager.issue(
            kind="train", argv=[sys.executable, "train.py"], display_command="test"
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 5555
            mock_proc.wait.return_value = 0
            mock_popen.return_value = mock_proc

            res = manager.execute(token=token, kind="train")
            run_id = res["run_id"]

            tracker = ExperimentTracker(db_path=db_path)
            run_data = tracker.get_run(run_id)
            assert run_data["run_id"] == run_id
            assert run_data["pid"] == 5555
            assert run_data["run_kind"] == "train"

            time.sleep(0.05)  # Let watcher thread complete
            updated_run = tracker.get_run(run_id)
            assert updated_run["status"] == "completed"
            assert updated_run["exit_code"] == 0
