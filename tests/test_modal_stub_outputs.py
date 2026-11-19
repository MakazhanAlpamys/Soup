"""`soup train --cloud modal` keeps run outputs on a Modal volume and downloads them.

The rendered stub is executed against a fake ``modal`` module (no SDK import, no
network). The fake mirrors the modal 1.5.5 API the stub relies on:
``Volume.from_name(name, create_if_missing=...)``, ``Volume.commit()``,
``Volume.listdir(path, recursive=...) -> list[FileEntry]`` (``.path`` / ``.type``),
``Volume.read_file(path)`` yielding bytes, and ``modal.volume.FileEntryType``.
"""

from __future__ import annotations

import ast
import enum
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.conftest import strip_ansi

_SOUP_YAML = (
    "base: hf-internal-testing/tiny-random-gpt2\n"
    "task: sft\n"
    "data:\n  train: data.jsonl\n  format: chatml\n"
    "output: ./output\n"
)
_RUN = "soup-test01"


class _FileEntryType(enum.IntEnum):
    UNSPECIFIED = 0
    FILE = 1
    DIRECTORY = 2


@dataclass(frozen=True)
class _Entry:
    path: str
    type: _FileEntryType


class _FakeVolume:
    def __init__(
        self, root: Path, *, leading_slash: bool, extra_entries=(), listdir_error=None
    ):
        self.root = root
        self.leading_slash = leading_slash
        self.extra_entries = list(extra_entries)
        self.listdir_error = listdir_error
        self.commits = 0
        self.from_name_kwargs: dict = {}

    def commit(self) -> None:
        self.commits += 1

    def listdir(self, path: str, *, recursive: bool = False):
        assert recursive is True
        if self.listdir_error is not None:
            raise self.listdir_error
        base = self.root / path.strip("/")
        entries = []
        for item in sorted(base.rglob("*")):
            rel = item.relative_to(self.root).as_posix()
            if self.leading_slash:
                rel = "/" + rel
            kind = _FileEntryType.FILE if item.is_file() else _FileEntryType.DIRECTORY
            entries.append(_Entry(rel, kind))
        return entries + self.extra_entries

    def read_file(self, path: str):
        data = (self.root / path.strip("/")).read_bytes()
        half = len(data) // 2
        yield data[:half]
        yield data[half:]


class _FakeFunction:
    def __init__(self, fn, kwargs):
        self.fn = fn
        self.kwargs = kwargs

    def remote(self):
        return self.fn()


class _FakeApp:
    def __init__(self, name: str):
        self.name = name

    def function(self, **kwargs):
        return lambda fn: _FakeFunction(fn, kwargs)

    def local_entrypoint(self):
        return lambda fn: fn


class _FakeImage:
    @classmethod
    def debian_slim(cls):
        return cls()

    def pip_install(self, *specs):
        return self


def _install_fake_modal(monkeypatch, volume: _FakeVolume) -> None:
    fake = types.ModuleType("modal")
    volume_mod = types.ModuleType("modal.volume")
    volume_mod.FileEntryType = _FileEntryType

    def from_name(name, create_if_missing=False):
        volume.from_name_kwargs = {"name": name, "create_if_missing": create_if_missing}
        return volume

    fake.App = _FakeApp
    fake.Image = _FakeImage
    fake.Volume = types.SimpleNamespace(from_name=from_name)
    fake.volume = volume_mod
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setitem(sys.modules, "modal.volume", volume_mod)


def _render(
    tmp_path: Path,
    vol_root: Path,
    output_dir: str | None = None,
    windows_separators: bool = False,
) -> str:
    from soup_cli.cloud.modal import render_modal_stub

    stub = render_modal_stub(
        _SOUP_YAML,
        gpu="a100",
        output_dir=str(tmp_path / "local-out") if output_dir is None else output_dir,
        soup_version="0.75.0",
        run_name=_RUN,
        windows_separators=windows_separators,
    )
    # Test-only relocation of the container paths onto the temp volume root.
    for old, new in (
        ('_REMOTE_ROOT = "/outputs"', f"_REMOTE_ROOT = {str(vol_root)!r}"),
        (
            '_CONFIG_PATH = "/root/soup.yaml"',
            f"_CONFIG_PATH = {str(tmp_path / 'container-soup.yaml')!r}",
        ),
    ):
        assert stub.count(old) == 1, old
        stub = stub.replace(old, new)
    return stub


def _exec_stub(stub: str) -> dict:
    namespace: dict = {"__name__": "soup_modal_app"}
    exec(compile(stub, "soup_modal_app.py", "exec"), namespace)
    return namespace


def test_stub_is_valid_python_and_uses_a_volume():
    from soup_cli.cloud.modal import MODAL_OUTPUT_VOLUME, render_modal_stub

    stub = render_modal_stub(
        _SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.75.0"
    )
    ast.parse(stub)
    assert MODAL_OUTPUT_VOLUME == "soup-outputs"
    assert "modal.Volume.from_name" in stub
    assert "create_if_missing=True" in stub
    assert "volumes=" in stub
    assert ".commit()" in stub
    assert "download checkpoints to" not in stub


def test_default_run_name_is_generated_and_unique():
    from soup_cli.cloud.modal import render_modal_stub

    stubs = {
        render_modal_stub(_SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.75.0")
        for _ in range(2)
    }
    assert len(stubs) == 2
    for stub in stubs:
        tree = ast.parse(stub)
        names = [
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_RUN_NAME"
        ]
        assert len(names) == 1
        assert names[0].startswith("soup-")


@pytest.mark.parametrize(
    "bad",
    ["Bad Name", "x'); import os; ('", "", "-leading", "a" * 64, "run/../x", 7],
)
def test_run_name_validated(bad):
    from soup_cli.cloud.modal import render_modal_stub

    with pytest.raises(ValueError, match="run_name"):
        render_modal_stub(
            _SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.75.0", run_name=bad
        )


@pytest.mark.parametrize("leading_slash", [False, True])
def test_generated_app_downloads_outputs(tmp_path, monkeypatch, capsys, leading_slash):
    vol_root = tmp_path / "volume"
    vol_root.mkdir()
    volume = _FakeVolume(vol_root, leading_slash=leading_slash)
    _install_fake_modal(monkeypatch, volume)
    calls = []

    def fake_run(argv, check, cwd):
        calls.append((argv, cwd))
        assert Path(cwd) == vol_root / _RUN
        out = Path(cwd) / "output"
        (out / "checkpoint-5").mkdir(parents=True)
        (out / "adapter_model.safetensors").write_bytes(b"adapter-bytes")
        (out / "checkpoint-5" / "optimizer.pt").write_bytes(b"optimizer-bytes")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    namespace = _exec_stub(_render(tmp_path, vol_root))
    assert volume.from_name_kwargs == {"name": "soup-outputs", "create_if_missing": True}
    assert namespace["train"].kwargs["volumes"] == {str(vol_root): volume}

    namespace["main"]()

    local = tmp_path / "local-out"
    assert (local / "output" / "adapter_model.safetensors").read_bytes() == b"adapter-bytes"
    assert (local / "output" / "checkpoint-5" / "optimizer.pt").read_bytes() == (
        b"optimizer-bytes"
    )
    assert len(calls) == 1
    assert volume.commits == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "Downloaded 2 file(s)" in out  # ansi-ok: mock modal subprocess stdout is uncoloured
    assert f"modal volume get soup-outputs /{_RUN}" in out


def test_download_runs_even_when_training_fails(tmp_path, monkeypatch):
    vol_root = tmp_path / "volume"
    vol_root.mkdir()
    volume = _FakeVolume(vol_root, leading_slash=False)
    _install_fake_modal(monkeypatch, volume)

    def failing_run(argv, check, cwd):
        out = Path(cwd) / "output" / "checkpoint-1"
        out.mkdir(parents=True)
        (out / "adapter_model.safetensors").write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", failing_run)
    namespace = _exec_stub(_render(tmp_path, vol_root))

    with pytest.raises(subprocess.CalledProcessError):
        namespace["main"]()

    saved = tmp_path / "local-out" / "output" / "checkpoint-1" / "adapter_model.safetensors"
    assert saved.read_bytes() == b"partial"
    assert volume.commits == 1


def test_failed_download_does_not_hide_the_training_error(tmp_path, monkeypatch, capsys):
    vol_root = tmp_path / "volume"
    vol_root.mkdir()
    volume = _FakeVolume(
        vol_root,
        leading_slash=False,
        listdir_error=FileNotFoundError("run directory does not exist"),
    )
    _install_fake_modal(monkeypatch, volume)

    def failing_run(argv, check, cwd):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", failing_run)
    namespace = _exec_stub(_render(tmp_path, vol_root))

    with pytest.raises(subprocess.CalledProcessError):
        namespace["main"]()

    out = " ".join(capsys.readouterr().out.split())
    assert "Could not download outputs after the failed run" in out
    assert "run directory does not exist" in out


def test_download_error_propagates_after_successful_training(tmp_path, monkeypatch):
    vol_root = tmp_path / "volume"
    vol_root.mkdir()
    volume = _FakeVolume(
        vol_root,
        leading_slash=False,
        listdir_error=FileNotFoundError("run directory does not exist"),
    )
    _install_fake_modal(monkeypatch, volume)
    monkeypatch.setattr(
        subprocess, "run", lambda argv, check, cwd: subprocess.CompletedProcess(argv, 0)
    )
    namespace = _exec_stub(_render(tmp_path, vol_root))

    with pytest.raises(FileNotFoundError, match="run directory does not exist"):
        namespace["main"]()


def test_download_refuses_escaping_entry(tmp_path, monkeypatch):
    vol_root = tmp_path / "volume"
    vol_root.mkdir()
    volume = _FakeVolume(
        vol_root,
        leading_slash=False,
        extra_entries=[_Entry(f"/{_RUN}/../../evil.txt", _FileEntryType.FILE)],
    )
    _install_fake_modal(monkeypatch, volume)
    monkeypatch.setattr(
        subprocess, "run", lambda argv, check, cwd: subprocess.CompletedProcess(argv, 0)
    )
    namespace = _exec_stub(_render(tmp_path, vol_root))

    with pytest.raises(RuntimeError, match="refusing to write outside"):
        namespace["main"]()

    assert not (tmp_path / "evil.txt").exists()
    assert not list(tmp_path.rglob("evil.txt"))


# --- the retry hint is a command a person pastes: the output name is repeated as a name


_PLACEHOLDER = "<local output directory>"


def _run_stub_and_capture(
    tmp_path, monkeypatch, capsys, output_dir: str, windows_separators: bool = False
) -> str:
    """Run the rendered app against the fake volume (no files) and return its stdout."""
    vol_root = tmp_path / "volume"
    vol_root.mkdir(exist_ok=True)
    volume = _FakeVolume(vol_root, leading_slash=False)
    _install_fake_modal(monkeypatch, volume)
    monkeypatch.setattr(
        subprocess, "run", lambda argv, check, cwd: subprocess.CompletedProcess(argv, 0)
    )
    monkeypatch.chdir(tmp_path)
    namespace = _exec_stub(_render(tmp_path, vol_root, output_dir, windows_separators))
    namespace["main"]()
    return capsys.readouterr().out


def _retry_command_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if "modal volume get" in line]


@pytest.mark.parametrize(
    ("output_dir", "pasted"),
    [
        ("./out", "./out"),
        ("results/run 1", '"results/run 1"'),
        ("out (v2)", '"out (v2)"'),
        ("out;cmd", '"out;cmd"'),
        ("out&calc", '"out&calc"'),
        ("out|calc", '"out|calc"'),
        ("out>X", '"out>X"'),
        ("~/out", '"~/out"'),
        ("-rf", "./-rf"),
        ("--force", "./--force"),
        ("результаты", "результаты"),
        ("out$(cmd)", _PLACEHOLDER),
        ("out$HOME", _PLACEHOLDER),
        ("out`cmd`", _PLACEHOLDER),
        ('out"x', _PLACEHOLDER),
        ("out'; cmd; echo '", '"out\'; cmd; echo \'"'),
        ('out"; cmd; echo "', _PLACEHOLDER),
        ("out%COMSPEC%", _PLACEHOLDER),
        ("out!x", _PLACEHOLDER),
        ("out\\dir", _PLACEHOLDER),
        ("out\x1b[2Jx", _PLACEHOLDER),
        ("out\u201cx", _PLACEHOLDER),
        ("out\u2013force", _PLACEHOLDER),
    ],
)
def test_retry_command_repeats_the_output_name_as_one_argument(
    tmp_path, monkeypatch, capsys, output_dir, pasted
):
    out = _run_stub_and_capture(tmp_path, monkeypatch, capsys, output_dir)

    lines = _retry_command_lines(out)
    assert len(lines) == 1, out
    assert lines[0].strip() == f"modal volume get soup-outputs /{_RUN} {pasted} --force", lines
    # The command is on its own line: no prose in front of it and nothing after it, so a
    # copied line pastes as exactly the command.
    assert lines[0].startswith("  modal volume get ")
    assert lines[0].endswith(" --force")
    # No raw escape character reaches the terminal, whatever the name holds.
    assert "\x1b" not in out


@pytest.mark.parametrize("output_dir", ["out$(cmd)", "out`cmd`", "out%COMSPEC%"])
def test_a_name_left_out_of_the_command_is_shown_only_as_quoted_text(
    tmp_path, monkeypatch, capsys, output_dir
):
    out = _run_stub_and_capture(tmp_path, monkeypatch, capsys, output_dir)

    shown = [line for line in out.splitlines() if output_dir in line]
    assert shown, out
    for line in shown:
        assert "modal volume get" not in line, line
        # The shown path is wrapped in single quotes (utils.paths.quote_path).
        assert f"'{output_dir}" in line, line


def test_placeholder_explains_why_the_path_is_not_repeated(tmp_path, monkeypatch, capsys):
    out = " ".join(
        _run_stub_and_capture(tmp_path, monkeypatch, capsys, "out$(cmd)").split()
    )

    assert "is not repeated in the command" in out
    assert _PLACEHOLDER in out


def test_plain_name_prints_no_placeholder_note(tmp_path, monkeypatch, capsys):
    out = _run_stub_and_capture(tmp_path, monkeypatch, capsys, "./out")

    assert "is not repeated in the command" not in out
    assert _PLACEHOLDER not in out


def test_download_still_writes_into_an_output_name_with_a_space(
    tmp_path, monkeypatch, capsys
):
    vol_root = tmp_path / "volume"
    vol_root.mkdir()
    volume = _FakeVolume(vol_root, leading_slash=False)
    _install_fake_modal(monkeypatch, volume)

    def fake_run(argv, check, cwd):
        out_dir = Path(cwd) / "output"
        out_dir.mkdir(parents=True)
        (out_dir / "adapter_model.safetensors").write_bytes(b"adapter-bytes")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    local = tmp_path / "my outputs"
    namespace = _exec_stub(_render(tmp_path, vol_root, str(local)))

    namespace["main"]()

    assert (local / "output" / "adapter_model.safetensors").read_bytes() == b"adapter-bytes"
    assert "Downloaded 1 file(s)" in strip_ansi(capsys.readouterr().out)


@pytest.mark.parametrize(
    "output_dir",
    [
        "x'); import os; os.system('cmd'); ('",
        'x"); import os; os.system("cmd"); ("',
        "x' + __import__('os').getcwd() + '",
        "x\\' ; y = 1 ; '",
    ],
)
def test_the_output_name_stays_data_in_the_rendered_source(output_dir):
    from soup_cli.cloud.modal import render_modal_stub

    stub = render_modal_stub(
        _SOUP_YAML, gpu="a100", output_dir=output_dir, soup_version="0.75.0"
    )
    tree = ast.parse(stub)

    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "system" not in calls
    assert "getcwd" not in calls
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "y" not in names


@pytest.mark.parametrize(
    ("output_dir", "windows_separators", "pasted"),
    [
        ("runs\\x", True, "runs/x"),
        ("runs\\my x", True, '"runs/my x"'),
        ("runs\\x", False, _PLACEHOLDER),
        ("runs\\$x", True, _PLACEHOLDER),
    ],
)
def test_a_backslash_path_is_repeated_with_forward_slashes_only_on_a_windows_host(
    tmp_path, monkeypatch, capsys, output_dir, windows_separators, pasted
):
    out = _run_stub_and_capture(tmp_path, monkeypatch, capsys, output_dir, windows_separators)

    (line,) = _retry_command_lines(out)
    assert line.strip() == f"modal volume get soup-outputs /{_RUN} {pasted} --force"


def test_the_shown_path_has_no_invisible_or_reordering_characters(
    tmp_path, monkeypatch, capsys
):
    out = _run_stub_and_capture(tmp_path, monkeypatch, capsys, "out\u202ex\u200by")

    assert "\u202e" not in out
    assert "\u200b" not in out
    assert "'outxy'" in out
