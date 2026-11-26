"""soup merge-sharded-fsdp-weights shows typed paths, shard errors and its result as typed."""

import os
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import merge_sharded_fsdp_weights as merge_cmd
from soup_cli.utils.fsdp_consolidate import ConsolidationResult
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# A backslash separates directories on Windows, so a name that holds one cannot be a file there.
_NO_BACKSLASH_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="a backslash separates directories on Windows"
)

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close, an emoji code, and a tag with control characters, so the
# strip and the escape are pinned separately. A backslash before an unmatched
# bracket and a trailing pair of backslashes are what markup rewrites.
_TEXTS = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    pytest.param("x\\[n", id="backslash-bracket"),
    pytest.param("n\\\\", id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]
# A file name cannot hold a slash, so no closing tag with nothing to close.
_NAMES = [
    "x[red]n",
    pytest.param("n:warning:", marks=skip_colon_on_windows, id="emoji-code"),
    pytest.param("x\\[n", marks=_NO_BACKSLASH_ON_WINDOWS, id="backslash-bracket"),
    pytest.param("n\\\\", marks=_NO_BACKSLASH_ON_WINDOWS, id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr(merge_cmd, "console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


@pytest.fixture
def torch_lib():
    """The live consolidation needs torch and safetensors."""
    pytest.importorskip("safetensors")
    return pytest.importorskip("torch")


def _coloured(monkeypatch):
    monkeypatch.setattr(
        merge_cmd,
        "console",
        Console(
            width=500, height=100, force_terminal=True, no_color=False, color_system="standard"
        ),
    )


def _shards(name: str = "shards") -> str:
    """A shard directory whose one shard is a placeholder: planning only reads the names."""
    Path(name).mkdir()
    Path(name, "pytorch_model_fsdp_0.bin").write_bytes(b"")
    return name


def _merge(shard_dir: str, *extra: str, output: str = "merged.safetensors"):
    return runner.invoke(
        app, ["merge-sharded-fsdp-weights", shard_dir, "--output", output, *extra]
    )


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _lines(result) -> list:
    """Every output line, its whitespace collapsed and the panel's side borders cut off."""
    lines = strip_ansi(result.output).splitlines()
    return [" ".join(line.strip("\u2502").split()) for line in lines]


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _raising(error):
    def fail(*args, **kwargs):
        raise error

    return fail


def _consolidating(monkeypatch, *, result=None, error=None):
    """Stand in for the torch side, so the display is tested without a checkpoint."""

    def consolidate(plan):
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(merge_cmd, "consolidate_shards", consolidate)


def _result(output_path: str) -> ConsolidationResult:
    return ConsolidationResult(
        output_path=output_path, num_tensors=3, num_shards=1, total_bytes=2_500_000
    )


# --- refusals while planning -------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_shard_directory_is_shown_as_typed(name):
    result = _merge(f"gone-{name}")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"shard_dir not found: gone-{_shown(name)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_shard_directory_outside_the_working_directory_is_shown_as_typed(tmp_path, name):
    result = _merge(str(tmp_path / "elsewhere" / name))

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"shard_dir is outside cwd: {_shown(name)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_an_output_outside_the_working_directory_is_shown_as_typed(tmp_path, name):
    result = _merge(_shards(), output=str(tmp_path / "elsewhere" / f"{name}.safetensors"))

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"output_path is outside cwd: {_shown(name)}.safetensors" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, FileNotFoundError, RuntimeError])
@pytest.mark.parametrize("value", _TEXTS)
def test_a_refused_plan_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(merge_cmd, "plan_consolidation", _raising(error(f"plan said {value}")))

    result = _merge("shards")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"plan said {_shown(value)}" in text
    assert _not_shown(text)


# --- the plan ----------------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_the_plan_shows_the_directory_and_the_output_as_typed(name):
    result = _merge(_shards(name), "--plan-only", output=f"{name}.safetensors")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    lines = _lines(result)
    assert "Shards found: 1" in lines
    assert f"Source dir: {_shown(os.path.realpath(name))}" in lines
    assert f"Output (target): {_shown(os.path.realpath(f'{name}.safetensors'))}" in lines
    assert _not_shown(_plain(result))
    assert not Path(f"{name}.safetensors").exists()


def test_a_plain_plan_keeps_its_lines():
    result = _merge(_shards(), "--plan-only")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    raw = strip_ansi(result.output)
    assert "FSDP Consolidation Plan" in raw
    assert re.search(r"^\u2502 Shards found:    1 +\u2502$", raw, re.M)
    assert f"Source dir:      {os.path.realpath('shards')} " in raw
    assert f"Output (target): {os.path.realpath('merged.safetensors')} " in raw
    lines = _lines(result)
    target = lines.index(f"Output (target): {os.path.realpath('merged.safetensors')}")
    assert lines[target + 1] == ""
    assert lines[target + 2] == (
        "Plan-only run \u2014 pass without --plan-only to write the file."
    )


# --- the consolidation -------------------------------------------------------


@pytest.mark.parametrize("error", [TypeError, ValueError])
@pytest.mark.parametrize("value", _TEXTS)
def test_a_failed_consolidation_is_shown_as_typed(monkeypatch, error, value):
    _consolidating(monkeypatch, error=error(f"consolidation said {value}"))

    result = _merge(_shards())

    assert result.exit_code == 2, (result.output, repr(result.exception))
    lines = _lines(result)
    assert lines == ["Consolidating 1 shard(s)...", f"consolidation said {_shown(value)}"]
    assert _not_shown(_plain(result))


def test_a_missing_torch_keeps_its_install_hint(monkeypatch):
    _consolidating(monkeypatch, error=ImportError("no torch"))

    result = _merge(_shards())

    assert result.exit_code == 1, (result.output, repr(result.exception))
    hint = (
        "torch + safetensors are required for consolidation. "
        'Install with: pip install "soup-cli[train]"'
    )
    assert _lines(result)[1:] == [hint]


@pytest.mark.parametrize("value", _TEXTS)
def test_the_result_shows_the_output_path_as_typed(monkeypatch, value):
    _consolidating(monkeypatch, result=_result(value))

    result = _merge(_shards())

    assert result.exit_code == 0, (result.output, repr(result.exception))
    lines = _lines(result)
    assert "Tensors written: 3" in lines
    assert "Shards merged: 1" in lines
    assert "Size: 2.50 MB" in lines
    assert f"Output: {_shown(value)}" in lines
    assert _not_shown(_plain(result))


@pytest.mark.usefixtures("torch_lib")
@pytest.mark.parametrize("name", _NAMES)
def test_a_live_consolidation_shows_its_output_path_as_typed(name):
    import torch

    shards = Path(name)
    shards.mkdir()
    torch.save({"a.weight": torch.ones(2, 2)}, shards / "pytorch_model_fsdp_0.bin")
    torch.save({"b.weight": torch.zeros(2, 2)}, shards / "pytorch_model_fsdp_1.bin")

    result = _merge(name, output=f"{name}.safetensors")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    lines = _lines(result)
    assert "Consolidating 2 shard(s)..." in lines
    assert "Tensors written: 2" in lines
    assert "Shards merged: 2" in lines
    assert f"Output: {_shown(os.path.realpath(f'{name}.safetensors'))}" in lines
    assert _not_shown(_plain(result))
    assert Path(f"{name}.safetensors").is_file()


@pytest.mark.usefixtures("torch_lib")
@pytest.mark.parametrize("key", _TEXTS)
def test_a_tensor_name_read_from_a_shard_is_shown_as_stored(key):
    import torch

    Path("shards").mkdir()
    torch.save({key: 3}, Path("shards") / "pytorch_model_fsdp_0.bin")

    result = _merge("shards")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    # The message quotes the name with repr(), which already escapes controls.
    assert f"shard pytorch_model_fsdp_0.bin key {key!r} is not a tensor (got int)" in text
    assert _not_shown(text)


# --- styles ------------------------------------------------------------------


def test_the_refusals_keep_their_red(monkeypatch):
    _coloured(monkeypatch)
    _consolidating(monkeypatch, error=ValueError("a shard said no"))

    planned = _merge("gone")
    failed = _merge(_shards())

    assert planned.output == "\x1b[31mshard_dir not found: gone\x1b[0m\n"
    assert failed.output.endswith("\x1b[31ma shard said no\x1b[0m\n")


def test_the_plan_panel_keeps_its_cyan_frame_and_an_unstyled_body(monkeypatch):
    _coloured(monkeypatch)

    result = _merge(_shards(), "--plan-only")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[36m FSDP Consolidation Plan \x1b[0m" in result.output
    assert "\x1b[36m\u2502\x1b[0m Shards found:    1 " in result.output
    assert f"\x1b[36m\u2502\x1b[0m Source dir:      {os.path.realpath('shards')} " in result.output


def test_the_result_panel_keeps_its_green_frame_and_an_unstyled_body(monkeypatch):
    _coloured(monkeypatch)
    _consolidating(monkeypatch, result=_result("merged.safetensors"))

    result = _merge(_shards())

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[1;32mFSDP Consolidation Done\x1b[0m" in result.output
    assert "\x1b[32m\u2502\x1b[0m Tensors written: 3 " in result.output
    assert "\x1b[32m\u2502\x1b[0m Shards merged:   1 " in result.output
    assert "\x1b[32m\u2502\x1b[0m Size:            2.50 MB " in result.output
    assert "\x1b[32m\u2502\x1b[0m Output:          merged.safetensors " in result.output
