"""Tests for Issue #1762: copy_bundle_to explicit containment root.

Ensures copy_bundle_to(name, output_path, *, root=None) allows callers
with their own workspace (such as Soup Zero) to copy demo bundles safely
under their workspace root without needing cwd chdir or reading the private
_bundle_source_path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from soup_cli.utils.demo_bundles import copy_bundle_to


class TestCopyBundleToContainmentRoot:
    def test_default_root_stays_under_cwd(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        # Inside cwd succeeds
        out = copy_bundle_to("alpaca_demo", "./alpaca.jsonl")
        assert Path(out).is_file()
        assert Path(out).parent == workspace

        # Outside cwd fails
        outside = str(tmp_path / "outside.jsonl")
        with pytest.raises(ValueError, match="output_path must stay under cwd"):
            copy_bundle_to("alpaca_demo", outside)

    def test_explicit_root_str_outside_cwd_succeeds(self, tmp_path, monkeypatch):
        # cwd is one directory
        cwd_dir = tmp_path / "cwd_dir"
        cwd_dir.mkdir()
        monkeypatch.chdir(cwd_dir)

        # workspace is a completely separate directory outside cwd
        workspace = tmp_path / "server_workspace"
        workspace.mkdir()

        target = str(workspace / "alpaca.jsonl")
        written = copy_bundle_to("alpaca_demo", target, root=str(workspace))

        assert os.path.isabs(written)
        assert Path(written).is_file()
        assert Path(written).parent == workspace
        # Validate JSON content
        with open(written, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) > 0

    def test_explicit_root_path_object_succeeds(self, tmp_path, monkeypatch):
        cwd_dir = tmp_path / "cwd_dir"
        cwd_dir.mkdir()
        monkeypatch.chdir(cwd_dir)

        workspace = tmp_path / "workspace_path_obj"
        workspace.mkdir()

        target = str(workspace / "dpo.jsonl")
        written = copy_bundle_to("dpo_demo", target, root=workspace)

        assert Path(written).is_file()
        assert Path(written).parent == workspace

    def test_explicit_root_rejects_paths_outside_root(self, tmp_path, monkeypatch):
        cwd_dir = tmp_path / "cwd_dir"
        cwd_dir.mkdir()
        monkeypatch.chdir(cwd_dir)

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Path pointing into cwd_dir or another folder, but root is set to workspace
        foreign_target = str(cwd_dir / "escaped.jsonl")
        with pytest.raises(ValueError, match="output_path must stay under root"):
            copy_bundle_to("alpaca_demo", foreign_target, root=str(workspace))

    def test_nested_subdirectory_under_root_creates_parents(self, tmp_path, monkeypatch):
        cwd_dir = tmp_path / "cwd_dir"
        cwd_dir.mkdir()
        monkeypatch.chdir(cwd_dir)

        workspace = tmp_path / "nested_workspace"
        workspace.mkdir()

        target = str(workspace / "sub1" / "sub2" / "bundle.jsonl")
        written = copy_bundle_to("sharegpt_demo", target, root=workspace)

        assert Path(written).is_file()
        assert Path(written).parent == workspace / "sub1" / "sub2"

    def test_root_validation_empty_string(self, tmp_path):
        target = str(tmp_path / "out.jsonl")
        with pytest.raises(ValueError, match="root must be a non-empty string or Path"):
            copy_bundle_to("alpaca_demo", target, root="")

    def test_root_validation_null_byte(self, tmp_path):
        target = str(tmp_path / "out.jsonl")
        with pytest.raises(ValueError, match="root must not contain null bytes"):
            copy_bundle_to("alpaca_demo", target, root="some\x00path")

    def test_root_validation_invalid_type(self, tmp_path):
        target = str(tmp_path / "out.jsonl")
        with pytest.raises(ValueError, match="root must be a non-empty string or Path"):
            copy_bundle_to("alpaca_demo", target, root=123)  # type: ignore[arg-type]

    def test_explicit_root_refuses_overwrite(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(tmp_path)

        target = workspace / "existing.jsonl"
        target.write_text("old content\n", encoding="utf-8")

        with pytest.raises(FileExistsError):
            copy_bundle_to("alpaca_demo", str(target), root=workspace)

    @pytest.mark.requires_symlink
    def test_explicit_root_staging_symlink_defense(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(tmp_path)

        target = workspace / "target.jsonl"
        outside_sentinel = tmp_path / "sentinel.txt"
        outside_sentinel.write_text("sentinel", encoding="utf-8")

        # Symlink at target.tmp
        os.symlink(str(outside_sentinel), str(target) + ".tmp")

        with pytest.raises(ValueError, match="symlink"):
            copy_bundle_to("alpaca_demo", str(target), root=workspace)

    def test_explicit_root_rejects_parent_traversal(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        target = str(root / ".." / "outside" / "x.jsonl")
        with pytest.raises(ValueError, match="output_path must stay under root"):
            copy_bundle_to("alpaca_demo", target, root=root)

    def test_explicit_root_rejects_sibling_prefix_directory(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        sibling = tmp_path / "root2"
        sibling.mkdir()
        target = str(sibling / "x.jsonl")
        with pytest.raises(ValueError, match="output_path must stay under root"):
            copy_bundle_to("alpaca_demo", target, root=root)

    @pytest.mark.requires_symlink
    def test_explicit_root_rejects_directory_symlink_pointing_outside(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside_dir"
        outside.mkdir()

        link_inside = root / "symlink_dir"
        os.symlink(str(outside), str(link_inside))

        target = str(link_inside / "escaped.jsonl")
        with pytest.raises(ValueError, match="output_path must stay under root"):
            copy_bundle_to("alpaca_demo", target, root=root)

        assert not (outside / "escaped.jsonl").exists()

    def test_explicit_root_empty_path_object_resolves_to_cwd(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        out = copy_bundle_to("alpaca_demo", "alpaca.jsonl", root=Path(""))
        assert Path(out).is_file()
        assert Path(out).parent == workspace

    def test_explicit_root_nonexistent_root_is_created(self, tmp_path):
        nonexistent_root = tmp_path / "new_root" / "sub"
        target = str(nonexistent_root / "bundle.jsonl")
        written = copy_bundle_to("alpaca_demo", target, root=nonexistent_root)
        assert Path(written).is_file()
        assert Path(written).parent == nonexistent_root
