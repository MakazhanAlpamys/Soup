"""Tests for MCP execution capability (Part E - Execution Slice)."""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import soup_cli.mcp_server.registry as reg
from soup_cli.mcp_server.execution import ExecutionError, ExecutionManager, digest_file

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


class TestDigestFileAndDirectoryTrees:
    def test_digest_file_computes_sha256_chunked(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        test_file = tmp_path / "test.bin"
        content = b"hello world 12345" * 10000
        test_file.write_bytes(content)

        protected = digest_file("test.bin", "field")
        expected_digest = hashlib.sha256(content).hexdigest()
        assert protected.digest == expected_digest
        assert protected.path == os.path.realpath(str(test_file))

    def test_digest_directory_tree_content_sha256(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "model_dir"
        test_dir.mkdir()
        (test_dir / "config.json").write_text('{"arch": "test"}', encoding="utf-8")
        (test_dir / "shard-00001.safetensors").write_bytes(b"tensors_data")

        protected = digest_file("model_dir", "field")
        assert len(protected.digest) == 64
        assert protected.path == os.path.realpath(str(test_dir))

    def test_directory_content_modification_changes_digest(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "ckpt"
        test_dir.mkdir()
        f1 = test_dir / "config.json"
        f2 = test_dir / "shard-00001.safetensors"
        f1.write_text('{"hidden": 768}', encoding="utf-8")
        f2.write_bytes(b"original_weights_data")

        d1 = digest_file("ckpt", "ckpt").digest

        # Modify shard in-place (same length, different bytes)
        f2.write_bytes(b"modified_weights_data")
        d2 = digest_file("ckpt", "ckpt").digest

        assert d1 != d2

    def test_directory_add_file_changes_digest(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "ckpt"
        test_dir.mkdir()
        (test_dir / "config.json").write_text("{}", encoding="utf-8")

        d1 = digest_file("ckpt", "ckpt").digest

        (test_dir / "added.bin").write_bytes(b"new_data")
        d2 = digest_file("ckpt", "ckpt").digest

        assert d1 != d2

    def test_directory_remove_file_changes_digest(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "ckpt"
        test_dir.mkdir()
        (test_dir / "f1.txt").write_text("1", encoding="utf-8")
        (test_dir / "f2.txt").write_text("2", encoding="utf-8")

        d1 = digest_file("ckpt", "ckpt").digest

        (test_dir / "f2.txt").unlink()
        d2 = digest_file("ckpt", "ckpt").digest

        assert d1 != d2

    def test_directory_rename_file_changes_digest(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "ckpt"
        test_dir.mkdir()
        (test_dir / "old_name.txt").write_text("content", encoding="utf-8")

        d1 = digest_file("ckpt", "ckpt").digest

        (test_dir / "old_name.txt").rename(test_dir / "new_name.txt")
        d2 = digest_file("ckpt", "ckpt").digest

        assert d1 != d2

    def test_identical_directory_trees_produce_identical_digests(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dir1 = tmp_path / "tree1"
        dir2 = tmp_path / "tree2"
        for d in (dir1, dir2):
            d.mkdir()
            sub = d / "sub"
            sub.mkdir()
            (sub / "file.txt").write_text("hello", encoding="utf-8")
            (d / "root.bin").write_bytes(b"\x00\x01\x02")

        d1 = digest_file("tree1", "t1").digest
        d2 = digest_file("tree2", "t2").digest
        assert d1 == d2

    def test_directory_file_count_limit_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "many_files"
        test_dir.mkdir()
        for i in range(5):
            (test_dir / f"f{i}.txt").write_text(str(i), encoding="utf-8")

        # Exceeds max_files=3 limit
        with pytest.raises(ExecutionError) as exc:
            digest_file("many_files", "dir_field", max_files=3)
        assert "exceeds maximum file count limit" in str(exc.value)

    def test_directory_byte_limit_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "big_dir"
        test_dir.mkdir()
        (test_dir / "f1.bin").write_bytes(b"a" * 100)
        (test_dir / "f2.bin").write_bytes(b"b" * 100)

        # Exceeds max_bytes=150 limit
        with pytest.raises(ExecutionError) as exc:
            digest_file("big_dir", "dir_field", max_bytes=150)
        assert "exceeds maximum byte limit" in str(exc.value)

    def test_revalidate_rejects_inplace_modification_of_directory_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model_dir = tmp_path / "model_weights"
        model_dir.mkdir()
        cfg_file = model_dir / "config.json"
        weight_file = model_dir / "shard-00001.safetensors"
        cfg_file.write_text('{"hidden": 256}', encoding="utf-8")
        weight_file.write_bytes(b"weights_v1")

        manager = ExecutionManager()
        export_spec = _spec("export", allow_execute=True, execution=manager)
        plan_res = export_spec.handler({"model": "model_weights", "format": "gguf"})
        token = plan_res["confirmation_token"]

        # In-place modification of weights file inside model directory
        weight_file.write_bytes(b"weights_v2")

        with pytest.raises(ExecutionError) as exc:
            manager.execute(token=token, kind="export")
        assert "planned input changed" in str(exc.value).lower()


class TestConfigSnapshottingAndPlanning:
    def test_plan_creates_expected_snapshot(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = tmp_path / "soup.yaml"
        cfg_path.write_text(_MIN_CONFIG, encoding="utf-8")

        manager = ExecutionManager()
        spec = _spec("train_start", allow_execute=True, execution=manager)
        out = spec.handler({"config": "soup.yaml"})

        token = out["confirmation_token"]
        plan = manager._plans[token]

        snapshot_file = Path(tmp_path) / ".soup" / "mcp-runs" / plan.run_id / "config.yaml"
        assert snapshot_file.exists()
        assert snapshot_file.read_text(encoding="utf-8") == _MIN_CONFIG

    def test_snapshot_contains_exact_validated_content(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        custom_yaml = (
            "base: Qwen/Qwen2.5-0.5B\ntask: sft\ndata:\n  train: d.jsonl\n  val_split: 0.2\n"
        )
        (tmp_path / "soup.yaml").write_text(custom_yaml, encoding="utf-8")

        manager = ExecutionManager()
        spec = _spec("train_start", allow_execute=True, execution=manager)
        out = spec.handler({"config": "soup.yaml"})

        token = out["confirmation_token"]
        plan = manager._plans[token]
        snapshot_file = Path(tmp_path) / ".soup" / "mcp-runs" / plan.run_id / "config.yaml"
        assert snapshot_file.read_text(encoding="utf-8") == custom_yaml

    def test_original_config_modification_after_plan_does_not_affect_execution(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        cfg_path = tmp_path / "soup.yaml"
        cfg_path.write_text(_MIN_CONFIG, encoding="utf-8")

        manager = ExecutionManager()
        spec = _spec("train_start", allow_execute=True, execution=manager)
        out = spec.handler({"config": "soup.yaml"})
        token = out["confirmation_token"]

        # Mutate the original soup.yaml completely
        cfg_path.write_text(
            "base: totally-different-model\ntask: sft\ndata:\n  train: other.jsonl\n",
            encoding="utf-8",
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 4321
            mock_proc.wait.return_value = 0
            mock_popen.return_value = mock_proc

            res = manager.execute(token=token, kind="train")
            assert res["status"] == "running"
            assert res["pid"] == 4321

            # Execution argv used the snapshot, not the modified soup.yaml
            call_args = mock_popen.call_args.args[0]
            config_idx = call_args.index("--config")
            used_config = call_args[config_idx + 1]
            assert ".soup" in used_config
            assert "mcp-runs" in used_config
            assert "config.yaml" in used_config
            assert Path(used_config).read_text(encoding="utf-8") == _MIN_CONFIG

    def test_execution_argv_references_snapshot(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = tmp_path / "soup.yaml"
        cfg_path.write_text(_MIN_CONFIG, encoding="utf-8")

        manager = ExecutionManager()
        spec = _spec("train_start", allow_execute=True, execution=manager)
        out = spec.handler({"config": "soup.yaml"})
        token = out["confirmation_token"]

        plan = manager._plans[token]
        expected_snapshot = str(
            Path(tmp_path) / ".soup" / "mcp-runs" / plan.run_id / "config.yaml"
        )
        assert "--config" in plan.argv
        cfg_arg_idx = plan.argv.index("--config") + 1
        assert os.path.realpath(plan.argv[cfg_arg_idx]) == os.path.realpath(expected_snapshot)

    def test_snapshot_run_id_matches_tracker_and_env(self, tmp_path, monkeypatch):
        from soup_cli.experiment.tracker import ExperimentTracker

        db_path = tmp_path / "exp.db"
        monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
        monkeypatch.chdir(tmp_path)

        (tmp_path / "soup.yaml").write_text(_MIN_CONFIG, encoding="utf-8")

        manager = ExecutionManager()
        spec = _spec("train_start", allow_execute=True, execution=manager)
        out = spec.handler({"config": "soup.yaml"})
        token = out["confirmation_token"]
        planned_run_id = manager._plans[token].run_id

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 7777
            mock_proc.wait.return_value = 0
            mock_popen.return_value = mock_proc

            exec_res = manager.execute(token=token, kind="train")
            assert exec_res["run_id"] == planned_run_id

            call_env = mock_popen.call_args.kwargs["env"]
            assert call_env["SOUP_MCP_RUN_ID"] == planned_run_id

            tracker = ExperimentTracker(db_path=db_path)
            run_data = tracker.get_run(planned_run_id)
            assert run_data is not None
            assert run_data["run_id"] == planned_run_id
            assert run_data["pid"] == 7777

    def test_external_protected_inputs_revalidated_before_execution(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"instruction": "hi", "output": "hello"}\n', encoding="utf-8")
        cfg_path = tmp_path / "soup.yaml"
        cfg_path.write_text(_MIN_CONFIG, encoding="utf-8")

        manager = ExecutionManager()
        spec = _spec("train_start", allow_execute=True, execution=manager)
        out = spec.handler({"config": "soup.yaml"})
        token = out["confirmation_token"]

        # Mutate external data input
        data_file.write_text('{"instruction": "tampered", "output": "pwned"}\n', encoding="utf-8")

        with pytest.raises(ExecutionError) as exc:
            manager.execute(token=token, kind="train")
        assert "planned input changed" in str(exc.value).lower()

    def test_invalid_config_planning_fails_without_plan(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bad_cfg = tmp_path / "bad.yaml"
        bad_cfg.write_text("task: invalid_task_xyz\n", encoding="utf-8")

        manager = ExecutionManager()
        spec = _spec("train_start", allow_execute=True, execution=manager)
        with pytest.raises(reg.McpToolError):
            spec.handler({"config": "bad.yaml"})

        assert len(manager._plans) == 0


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
            deadline = time.monotonic() + 3.0
            while manager._active_run_id is not None and time.monotonic() < deadline:
                time.sleep(0.005)
            assert manager._active_run_id is None

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

            deadline = time.monotonic() + 3.0
            updated_run = None
            while time.monotonic() < deadline:
                updated_run = tracker.get_run(run_id)
                if updated_run and updated_run["status"] in ("completed", "failed"):
                    break
                time.sleep(0.005)

            assert updated_run is not None
            assert updated_run["status"] == "completed"
            assert updated_run["exit_code"] == 0
