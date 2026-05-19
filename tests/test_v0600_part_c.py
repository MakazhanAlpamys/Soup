"""Tests for v0.60.0 Part C — ``--strict-safetensors`` mode.

Coverage:
- ``StrictSafetensorsReport`` frozen dataclass + verdicts
- ``find_unsafe_weight_files`` walker
- ``check_strict_safetensors`` strict + lenient
- CLI smoke (``soup adapters check-safetensors``)
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app


def _make_safetensors_only(tmp_path: Path) -> Path:
    target = tmp_path / "safe_adapter"
    target.mkdir()
    (target / "adapter_model.safetensors").write_bytes(b"weights")
    (target / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    return target


def _make_with_pickle(tmp_path: Path) -> Path:
    target = tmp_path / "pickle_adapter"
    target.mkdir()
    (target / "adapter_model.bin").write_bytes(b"PK\x05\x06pickled-bytes")
    (target / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    return target


class TestStrictSafetensors:
    def test_imports(self):
        from soup_cli.utils.strict_safetensors import (
            UNSAFE_EXTENSIONS,
            StrictSafetensorsReport,
            check_strict_safetensors,
            find_unsafe_weight_files,
        )
        assert callable(check_strict_safetensors)
        assert callable(find_unsafe_weight_files)
        assert isinstance(UNSAFE_EXTENSIONS, frozenset)
        assert dataclasses.is_dataclass(StrictSafetensorsReport)

    def test_unsafe_extensions_includes_bin_pt(self):
        from soup_cli.utils.strict_safetensors import UNSAFE_EXTENSIONS

        assert ".bin" in UNSAFE_EXTENSIONS
        assert ".pt" in UNSAFE_EXTENSIONS
        assert ".pth" in UNSAFE_EXTENSIONS
        assert ".ckpt" in UNSAFE_EXTENSIONS

    def test_find_unsafe_clean(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_safetensors_only(tmp_path)
        from soup_cli.utils.strict_safetensors import find_unsafe_weight_files

        found = find_unsafe_weight_files(str(adapter))
        assert found == ()

    def test_find_unsafe_flags_bin(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_with_pickle(tmp_path)
        from soup_cli.utils.strict_safetensors import find_unsafe_weight_files

        found = find_unsafe_weight_files(str(adapter))
        assert len(found) == 1
        assert found[0].endswith(".bin")

    def test_check_strict_clean_passes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_safetensors_only(tmp_path)
        from soup_cli.utils.strict_safetensors import check_strict_safetensors

        report = check_strict_safetensors(str(adapter), strict=True)
        assert report.ok is True
        assert report.unsafe_files == ()

    def test_check_strict_pickle_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_with_pickle(tmp_path)
        from soup_cli.utils.strict_safetensors import check_strict_safetensors

        with pytest.raises(ValueError, match="(?i)pickle|.bin|unsafe"):
            check_strict_safetensors(str(adapter), strict=True)

    def test_check_lenient_pickle_returns_report(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_with_pickle(tmp_path)
        from soup_cli.utils.strict_safetensors import check_strict_safetensors

        report = check_strict_safetensors(str(adapter), strict=False)
        assert report.ok is False
        assert len(report.unsafe_files) == 1

    def test_check_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.strict_safetensors import check_strict_safetensors

        with pytest.raises(ValueError):
            check_strict_safetensors(str(tmp_path.parent / "outside"))

    def test_check_missing_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.strict_safetensors import check_strict_safetensors

        with pytest.raises(FileNotFoundError):
            check_strict_safetensors(str(tmp_path / "nope"))

    def test_report_frozen(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_safetensors_only(tmp_path)
        from soup_cli.utils.strict_safetensors import check_strict_safetensors

        report = check_strict_safetensors(str(adapter))
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.ok = False  # type: ignore[misc]

    def test_strict_not_bool_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_safetensors_only(tmp_path)
        from soup_cli.utils.strict_safetensors import check_strict_safetensors

        with pytest.raises(TypeError):
            check_strict_safetensors(str(adapter), strict="yes")  # type: ignore[arg-type]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
    def test_symlinked_unsafe_file_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "symlink_adapter"
        adapter.mkdir()
        external = tmp_path / "external.bin"
        external.write_bytes(b"x")
        os.symlink(str(external), str(adapter / "weights.bin"))
        (adapter / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
        from soup_cli.utils.strict_safetensors import find_unsafe_weight_files

        # Even a symlinked weights.bin is reported as unsafe (we read by name).
        found = find_unsafe_weight_files(str(adapter))
        assert any(p.endswith("weights.bin") for p in found)


class TestStrictSafetensorsCli:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["adapters", "check-safetensors", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_clean_passes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_safetensors_only(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app, ["adapters", "check-safetensors", str(adapter.relative_to(tmp_path))]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_pickle_strict_exits_3(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_with_pickle(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app, [
                "adapters", "check-safetensors",
                str(adapter.relative_to(tmp_path)),
                "--strict",
            ]
        )
        # Exit code 3 is the distinct strict-fail code (planned).
        assert result.exit_code == 3, (result.output, repr(result.exception))

    def test_pickle_lenient_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_with_pickle(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app, [
                "adapters", "check-safetensors",
                str(adapter.relative_to(tmp_path)),
            ]
        )
        # Lenient: exit 1, not 0 (still flags the issue).
        assert result.exit_code == 1


class TestSourceWiring:
    def test_module_imports(self):
        from soup_cli.utils import strict_safetensors as m

        assert hasattr(m, "check_strict_safetensors")
        assert hasattr(m, "find_unsafe_weight_files")
        assert hasattr(m, "UNSAFE_EXTENSIONS")
