"""Keep Typer invocation and Click's filesystem lifecycle in the shared runner."""

from __future__ import annotations

import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

import pytest
from typer.testing import CliRunner as TyperCliRunner

from soup_cli import __version__
from soup_cli.cli import app
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi


@pytest.mark.parametrize("force_color", [False, True])
def test_invoke_is_the_installed_typer_implementation(force_color: bool) -> None:
    assert CliRunner.invoke is TyperCliRunner.invoke
    result = CliRunner().invoke(
        app, ["version"], color=force_color, env={"FORCE_COLOR": "1" if force_color else "0"},
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert __version__ in strip_ansi(result.output)


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_filesystem_restores_cwd_and_respects_directory_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preserve: bool, fail: bool,
) -> None:
    _assert_filesystem_lifecycle(tmp_path, monkeypatch, preserve, fail)


@pytest.mark.requires_symlink
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_filesystem_with_symlinked_temporary_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preserve: bool, fail: bool,
) -> None:
    target = tmp_path / "temporary-root"
    target.mkdir()
    alias = tmp_path / "temporary-alias"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(tempfile, "tempdir", str(alias))
    _assert_filesystem_lifecycle(tmp_path, monkeypatch, preserve, fail)


def _assert_filesystem_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preserve: bool, fail: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    expectation = pytest.raises(RuntimeError, match="inside isolation") if fail else nullcontext()
    directory: Path | None = None

    with expectation:
        with CliRunner().isolated_filesystem(temp_dir=tmp_path if preserve else None) as fs:
            assert isinstance(fs, str)
            directory = Path(fs)
            assert os.path.realpath(Path.cwd()) == os.path.realpath(directory)
            if preserve:
                assert os.path.realpath(directory.parent) == os.path.realpath(tmp_path)
            Path("artifact.txt").write_text("kept artifact", encoding="utf-8")
            if fail:
                raise RuntimeError("inside isolation")

    assert os.path.realpath(Path.cwd()) == os.path.realpath(tmp_path)
    assert directory is not None
    assert directory.exists() is preserve
    if preserve:
        assert (directory / "artifact.txt").read_text(encoding="utf-8") == "kept artifact"
