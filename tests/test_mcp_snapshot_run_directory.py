"""MCP plan time: the config snapshot's run directory is checked and created fresh.

``ExecutionManager.snapshot_config`` writes ``.soup/mcp-runs/<run_id>/config.yaml``.
#1492 refuses a symbolic link or junction at ``.soup`` and ``.soup/mcp-runs``;
these tests cover the run directory itself, a directory that already exists
there, a link that replaces a parent directory while the snapshot is being
created, and the snapshot file, which is created exclusively without following
a link. Ordinary planning writes the same bytes as before.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

import soup_cli.mcp_server.execution as execution_module
from soup_cli.mcp_server.execution import ExecutionError, ExecutionManager
from soup_cli.mcp_server.registry import McpToolError, build_registry
from soup_cli.utils.paths import atomic_write_text

RUN_ID = "run_20260101_000000_0123abcd"
CONTENT = "base: Qwen/Qwen2.5-0.5B\ntask: sft\ndata:\n  train: data.jsonl\n"
LINK_REFUSAL = "symbolic link or junction"
EXISTS_REFUSAL = "already exists"


def _manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExecutionManager:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.jsonl").write_text('{"text": "example"}\n', encoding="utf-8")
    (tmp_path / "soup.yaml").write_text(CONTENT, encoding="utf-8")
    return ExecutionManager()


def _train_start(manager: ExecutionManager, monkeypatch: pytest.MonkeyPatch):
    """The real ``train_start`` handler, planning under the fixed ``RUN_ID``."""
    monkeypatch.setattr(manager, "allocate_run_id", lambda: RUN_ID)
    for tool in build_registry(allow_mutating=True, allow_execute=True, execution=manager):
        if tool.name == "train_start":
            return tool.handler
    raise KeyError("train_start")


def _runs(tmp_path: Path) -> Path:
    return tmp_path / ".soup" / "mcp-runs"


def _files_under(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()]


class TestALinkAtTheRunDirectory:
    @pytest.mark.requires_symlink
    def test_directory_symlink_at_the_run_directory_is_refused(self, tmp_path, monkeypatch):
        manager = _manager(tmp_path, monkeypatch)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        _runs(tmp_path).mkdir(parents=True)
        os.symlink(str(elsewhere), str(_runs(tmp_path) / RUN_ID), target_is_directory=True)

        with pytest.raises(ExecutionError, match=LINK_REFUSAL) as excinfo:
            manager.snapshot_config(RUN_ID, CONTENT)
        assert not any(elsewhere.iterdir())
        assert str(elsewhere) not in str(excinfo.value)
        assert "elsewhere" not in str(excinfo.value)

        handler = _train_start(manager, monkeypatch)
        with pytest.raises(McpToolError, match=LINK_REFUSAL):
            handler({"config": "soup.yaml"})
        assert not any(elsewhere.iterdir())

    @pytest.mark.requires_symlink
    def test_dangling_symlink_at_the_run_directory_is_refused(self, tmp_path, monkeypatch):
        manager = _manager(tmp_path, monkeypatch)
        missing = tmp_path / "not-created"
        _runs(tmp_path).mkdir(parents=True)
        os.symlink(str(missing), str(_runs(tmp_path) / RUN_ID), target_is_directory=True)

        with pytest.raises(ExecutionError, match=LINK_REFUSAL):
            manager.snapshot_config(RUN_ID, CONTENT)
        assert not os.path.lexists(missing)

    @pytest.mark.skipif(os.name != "nt", reason="Windows junctions only exist on Windows")
    def test_junction_at_the_run_directory_is_refused(self, tmp_path, monkeypatch):
        import _winapi

        manager = _manager(tmp_path, monkeypatch)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        _runs(tmp_path).mkdir(parents=True)
        _winapi.CreateJunction(str(elsewhere), str(_runs(tmp_path) / RUN_ID))

        with pytest.raises(ExecutionError, match=LINK_REFUSAL) as excinfo:
            manager.snapshot_config(RUN_ID, CONTENT)
        assert not any(elsewhere.iterdir())
        assert str(elsewhere) not in str(excinfo.value)

        handler = _train_start(manager, monkeypatch)
        with pytest.raises(McpToolError, match=LINK_REFUSAL):
            handler({"config": "soup.yaml"})
        assert not any(elsewhere.iterdir())


class TestSomethingAlreadyAtTheRunDirectory:
    def test_existing_directory_is_refused_and_left_alone(self, tmp_path, monkeypatch):
        manager = _manager(tmp_path, monkeypatch)
        run_dir = _runs(tmp_path) / RUN_ID
        run_dir.mkdir(parents=True)
        (run_dir / "notes.txt").write_text("keep\n", encoding="utf-8")

        with pytest.raises(ExecutionError, match=EXISTS_REFUSAL) as excinfo:
            manager.snapshot_config(RUN_ID, CONTENT)
        assert not (run_dir / "config.yaml").exists()
        assert sorted(p.name for p in run_dir.iterdir()) == ["notes.txt"]
        assert str(tmp_path) not in str(excinfo.value)

        handler = _train_start(manager, monkeypatch)
        with pytest.raises(McpToolError, match=EXISTS_REFUSAL):
            handler({"config": "soup.yaml"})
        assert not (run_dir / "config.yaml").exists()

    def test_existing_file_at_the_run_directory_is_refused(self, tmp_path, monkeypatch):
        manager = _manager(tmp_path, monkeypatch)
        _runs(tmp_path).mkdir(parents=True)
        (_runs(tmp_path) / RUN_ID).write_text("keep\n", encoding="utf-8")

        with pytest.raises(ExecutionError, match=EXISTS_REFUSAL):
            manager.snapshot_config(RUN_ID, CONTENT)
        assert (_runs(tmp_path) / RUN_ID).read_text(encoding="utf-8") == "keep\n"


class TestALinkThatAppearsWhileTheSnapshotIsCreated:
    @pytest.mark.requires_symlink
    def test_parent_replaced_after_the_first_check(self, tmp_path, monkeypatch):
        """A link swapped in after the run path was checked creates nothing in its target."""
        manager = _manager(tmp_path, monkeypatch)
        runs = _runs(tmp_path)
        runs.mkdir(parents=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        real_refuse = execution_module.refuse_linked_dirs
        swapped = []

        def refuse_then_swap(path, *, stop_at=None):
            real_refuse(path, stop_at=stop_at)
            if not swapped:
                swapped.append(path)
                runs.rmdir()
                os.symlink(str(elsewhere), str(runs), target_is_directory=True)

        monkeypatch.setattr(execution_module, "refuse_linked_dirs", refuse_then_swap)
        with pytest.raises(ExecutionError, match=LINK_REFUSAL):
            manager.snapshot_config(RUN_ID, CONTENT)
        assert swapped
        assert not any(elsewhere.iterdir())

    @pytest.mark.requires_symlink
    def test_parent_replaced_just_before_the_snapshot_is_opened(self, tmp_path, monkeypatch):
        """The open re-checks the parent directories: nothing is written through the link."""
        manager = _manager(tmp_path, monkeypatch)
        runs = _runs(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        (elsewhere / RUN_ID).mkdir(parents=True)
        real_open = execution_module.open_no_follow
        swapped = []

        def swap_then_open(path, *args, **kwargs):
            if not swapped:
                swapped.append(path)
                os.rename(str(runs), str(tmp_path / "moved-runs"))
                os.symlink(str(elsewhere), str(runs), target_is_directory=True)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(execution_module, "open_no_follow", swap_then_open)
        with pytest.raises(ExecutionError, match=LINK_REFUSAL):
            manager.snapshot_config(RUN_ID, CONTENT)
        assert swapped
        assert _files_under(elsewhere) == []

    def test_snapshot_file_created_by_someone_else_is_not_overwritten(
        self, tmp_path, monkeypatch
    ):
        manager = _manager(tmp_path, monkeypatch)
        real_open = execution_module.open_no_follow
        other = "written: elsewhere\n"
        created = []

        def create_then_open(path, *args, **kwargs):
            if not created:
                created.append(path)
                Path(path).write_text(other, encoding="utf-8")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(execution_module, "open_no_follow", create_then_open)
        with pytest.raises(ExecutionError, match=EXISTS_REFUSAL):
            manager.snapshot_config(RUN_ID, CONTENT)
        assert created
        snapshot = _runs(tmp_path) / RUN_ID / "config.yaml"
        assert snapshot.read_text(encoding="utf-8") == other


class TestAnyOtherWriteFailure:
    def test_is_reported_by_type_without_a_path(self, tmp_path, monkeypatch):
        manager = _manager(tmp_path, monkeypatch)
        full_disk = str(tmp_path / ".soup" / "mcp-runs" / RUN_ID / "config.yaml")

        def no_space(*args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device", full_disk)

        monkeypatch.setattr(execution_module, "open_no_follow", no_space)
        expected = "cannot plan: could not write the config snapshot (OSError)"

        with pytest.raises(ExecutionError) as excinfo:
            manager.snapshot_config(RUN_ID, CONTENT)
        assert str(excinfo.value) == expected
        assert str(tmp_path) not in str(excinfo.value)

        monkeypatch.setattr(manager, "allocate_run_id", lambda: RUN_ID + "b")
        handler = None
        for tool in build_registry(allow_mutating=True, allow_execute=True, execution=manager):
            if tool.name == "train_start":
                handler = tool.handler
        with pytest.raises(McpToolError) as tool_error:
            handler({"config": "soup.yaml"})
        assert str(tool_error.value) == expected


class TestTheRunIdIsOnePathComponent:
    """The snapshot path is built from the run id, so it must not walk anywhere."""

    @pytest.mark.parametrize("run_id", ["", ".", "..", "nested/run", "../run", "run\x00x"])
    def test_run_id_that_is_not_one_path_component_is_refused(
        self, tmp_path, monkeypatch, run_id
    ):
        manager = _manager(tmp_path, monkeypatch)

        with pytest.raises(ExecutionError, match="invalid run id"):
            manager.snapshot_config(run_id, CONTENT)
        assert not (tmp_path / ".soup").exists()

    def test_run_id_cannot_place_the_snapshot_outside_the_working_directory(
        self, tmp_path, monkeypatch
    ):
        manager = _manager(tmp_path, monkeypatch)
        outside = tmp_path.parent / f"{tmp_path.name}-outside"

        with pytest.raises(ExecutionError, match="invalid run id"):
            manager.snapshot_config(f"../../../{outside.name}", CONTENT)
        assert not outside.exists()


class TestOrdinaryPlanning:
    def test_snapshot_is_written_as_before(self, tmp_path, monkeypatch):
        manager = _manager(tmp_path, monkeypatch)

        path = manager.snapshot_config(RUN_ID, CONTENT)

        expected = _runs(tmp_path) / RUN_ID / "config.yaml"
        assert path == os.path.realpath(expected)
        assert not os.path.islink(expected)
        # The bytes the previous writer (atomic_write_text) produced.
        control = atomic_write_text(CONTENT, str(tmp_path / "control.yaml"))
        assert Path(path).read_bytes() == Path(control).read_bytes()
        assert stat.S_IMODE(os.stat(path).st_mode) == stat.S_IMODE(os.stat(control).st_mode)

    def test_each_plan_gets_its_own_run_directory(self, tmp_path, monkeypatch):
        manager = _manager(tmp_path, monkeypatch)
        handler = None
        for tool in build_registry(allow_mutating=True, allow_execute=True, execution=manager):
            if tool.name == "train_start":
                handler = tool.handler
        assert handler is not None

        first = handler({"config": "soup.yaml"})
        second = handler({"config": "soup.yaml"})

        assert first["confirmation_token"] != second["confirmation_token"]
        run_dirs = sorted(_runs(tmp_path).iterdir())
        assert len(run_dirs) == 2
        for run_dir in run_dirs:
            snapshot = run_dir / "config.yaml"
            assert snapshot.read_text(encoding="utf-8") == CONTENT
