"""Keep Typer invocation and Click's filesystem lifecycle in the shared runner."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest
from typer.testing import CliRunner as TyperCliRunner

from soup_cli import __version__
from soup_cli.cli import app
from tests.cli_runner import CliRunner


def test_invoke_is_the_installed_typer_implementation() -> None:
    assert CliRunner.invoke is TyperCliRunner.invoke
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert __version__ in result.output


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_filesystem_restores_cwd_and_respects_directory_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preserve: bool, fail: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    expectation = pytest.raises(RuntimeError, match="inside isolation") if fail else nullcontext()
    directory: Path | None = None

    with expectation:
        with CliRunner().isolated_filesystem(temp_dir=tmp_path if preserve else None) as fs:
            assert isinstance(fs, str)
            directory = Path(fs)
            assert Path.cwd() == directory
            if preserve:
                assert directory.parent == tmp_path
            Path("artifact.txt").write_text("kept artifact", encoding="utf-8")
            if fail:
                raise RuntimeError("inside isolation")

    assert Path.cwd() == tmp_path
    assert directory is not None
    assert directory.exists() is preserve
    if preserve:
        assert (directory / "artifact.txt").read_text(encoding="utf-8") == "kept artifact"
