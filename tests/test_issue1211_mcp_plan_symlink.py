"""#1211 — MCP plan time: snapshot_config writes through a symlinked or junctioned .soup directory.

Follow-up to #1158 / #1171:
PR #1171 hardened the MCP run log at execution time against a symlinked or
junctioned directory above it. This suite covers the plan-time half:
when snapshot_config copies the validated config into .soup/mcp-runs/<id>/config.yaml,
any directory symlink or junction at .soup or .soup/mcp-runs must be refused
before directory creation and write.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from soup_cli.mcp_server.execution import ExecutionError, ExecutionManager
from soup_cli.mcp_server.registry import McpToolError, build_registry


def _get_train_start_spec(manager: ExecutionManager):
    specs = build_registry(allow_mutating=True, allow_execute=True, execution=manager)
    for tool in specs:
        if tool.name == "train_start":
            return tool
    raise KeyError("train_start")


def _setup_valid_config(tmp_path: Path) -> Path:
    data_file = tmp_path / "data.jsonl"
    data_file.write_text('{"text": "example"}\n', encoding="utf-8")
    cfg_file = tmp_path / "soup.yaml"
    cfg_file.write_text(
        "base: Qwen/Qwen2.5-0.5B\ntask: sft\ndata:\n  train: data.jsonl\n",
        encoding="utf-8",
    )
    return cfg_file


class TestPlanTimeDirectorySymlinkAndJunction:
    @pytest.mark.requires_symlink
    def test_plan_refuses_a_directory_symlink_one_level_up(self, tmp_path, monkeypatch):
        """A directory symlink at .soup/mcp-runs -> evildir is refused at plan time."""
        monkeypatch.chdir(tmp_path)
        _setup_valid_config(tmp_path)
        manager = ExecutionManager()

        evildir = tmp_path / "evildir"
        evildir.mkdir()
        soup_dir = tmp_path / ".soup"
        soup_dir.mkdir()
        os.symlink(str(evildir), str(soup_dir / "mcp-runs"), target_is_directory=True)

        # 1. Direct snapshot_config refusal
        with pytest.raises(ExecutionError, match="symbolic link or junction"):
            manager.snapshot_config("planrun1", "dummy: content")
        assert not any(evildir.iterdir())

        # 2. End-to-end plan tool refusal
        spec = _get_train_start_spec(manager)
        with pytest.raises(McpToolError, match="symbolic link or junction"):
            spec.handler({"config": "soup.yaml"})
        assert not any(evildir.iterdir())

    @pytest.mark.requires_symlink
    def test_plan_refuses_symlink_at_soup_directory_itself(self, tmp_path, monkeypatch):
        """A symlink at .soup itself is refused without creating mcp-runs in target."""
        monkeypatch.chdir(tmp_path)
        _setup_valid_config(tmp_path)
        manager = ExecutionManager()

        evildir = tmp_path / "evildir"
        evildir.mkdir()
        os.symlink(str(evildir), str(tmp_path / ".soup"), target_is_directory=True)

        # 1. Direct snapshot_config refusal
        with pytest.raises(ExecutionError, match="symbolic link or junction"):
            manager.snapshot_config("planrun2", "dummy: content")
        assert not (evildir / "mcp-runs").exists()
        assert not any(evildir.iterdir())

        # 2. End-to-end plan tool refusal
        spec = _get_train_start_spec(manager)
        with pytest.raises(McpToolError, match="symbolic link or junction"):
            spec.handler({"config": "soup.yaml"})
        assert not (evildir / "mcp-runs").exists()
        assert not any(evildir.iterdir())

    @pytest.mark.skipif(os.name != "nt", reason="Windows junctions only exist on Windows")
    def test_plan_refuses_a_directory_junction_one_level_up(self, tmp_path, monkeypatch):
        """A directory junction at .soup/mcp-runs -> evildir is refused at plan time."""
        import _winapi

        monkeypatch.chdir(tmp_path)
        _setup_valid_config(tmp_path)
        manager = ExecutionManager()

        evildir = tmp_path / "evildir"
        evildir.mkdir()
        soup_dir = tmp_path / ".soup"
        soup_dir.mkdir()
        _winapi.CreateJunction(str(evildir), str(soup_dir / "mcp-runs"))

        with pytest.raises(ExecutionError, match="symbolic link or junction"):
            manager.snapshot_config("junctionrun1", "dummy: content")
        assert not any(evildir.iterdir())

        spec = _get_train_start_spec(manager)
        with pytest.raises(McpToolError, match="symbolic link or junction"):
            spec.handler({"config": "soup.yaml"})
        assert not any(evildir.iterdir())

    @pytest.mark.skipif(os.name != "nt", reason="Windows junctions only exist on Windows")
    def test_plan_refuses_junction_at_soup_directory_itself(self, tmp_path, monkeypatch):
        """A junction at .soup itself is refused without creating mcp-runs in target."""
        import _winapi

        monkeypatch.chdir(tmp_path)
        _setup_valid_config(tmp_path)
        manager = ExecutionManager()

        evildir = tmp_path / "evildir"
        evildir.mkdir()
        _winapi.CreateJunction(str(evildir), str(tmp_path / ".soup"))

        with pytest.raises(ExecutionError, match="symbolic link or junction"):
            manager.snapshot_config("junctionrun2", "dummy: content")
        assert not (evildir / "mcp-runs").exists()
        assert not any(evildir.iterdir())

        spec = _get_train_start_spec(manager)
        with pytest.raises(McpToolError, match="symbolic link or junction"):
            spec.handler({"config": "soup.yaml"})
        assert not (evildir / "mcp-runs").exists()
        assert not any(evildir.iterdir())

    def test_plan_ordinary_path_creates_regular_snapshot(self, tmp_path, monkeypatch):
        """Ordinary path creates a regular directory and file under working directory."""
        monkeypatch.chdir(tmp_path)
        _setup_valid_config(tmp_path)
        manager = ExecutionManager()

        spec = _get_train_start_spec(manager)
        result = spec.handler({"config": "soup.yaml"})

        assert "confirmation_token" in result
        token = result["confirmation_token"]
        assert isinstance(token, str)

        snapshot_dir = tmp_path / ".soup" / "mcp-runs"
        assert snapshot_dir.exists()
        assert not os.path.islink(snapshot_dir)

        run_dirs = list(snapshot_dir.iterdir())
        assert len(run_dirs) == 1
        cfg_snapshot = run_dirs[0] / "config.yaml"
        assert cfg_snapshot.exists()
        assert not os.path.islink(cfg_snapshot)
        assert "base: Qwen/Qwen2.5-0.5B" in cfg_snapshot.read_text(encoding="utf-8")

    @pytest.mark.requires_symlink
    def test_repaired_layout_allows_subsequent_plan(self, tmp_path, monkeypatch):
        """After removing the directory link, a subsequent plan succeeds cleanly."""
        monkeypatch.chdir(tmp_path)
        _setup_valid_config(tmp_path)
        manager = ExecutionManager()

        evildir = tmp_path / "evildir"
        evildir.mkdir()
        link = tmp_path / ".soup"
        os.symlink(str(evildir), str(link), target_is_directory=True)

        with pytest.raises(ExecutionError, match="symbolic link or junction"):
            manager.snapshot_config("blocked", "content")

        (os.rmdir if os.name == "nt" else os.unlink)(link)

        # Fresh plan succeeds
        spec = _get_train_start_spec(manager)
        res = spec.handler({"config": "soup.yaml"})
        assert "confirmation_token" in res
