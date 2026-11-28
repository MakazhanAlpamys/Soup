"""Check the optional expert reader sources, wheel contents and import contract."""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

import soup_cli

EXT_DIR = Path(soup_cli.__file__).resolve().parent / "utils" / "native_expert_reader"
SOURCE = EXT_DIR / "native_stream_async.cpp"
EXTENSION_NAME = "_native_expert_stream_async"
BUILD_COMMAND = (
    "python -m soup_cli.utils.native_expert_reader.build_ext "
    "build_ext --build-lib /tmp/soup-native --build-temp /tmp/soup-native/obj"
)
MISSING_EXTENSION = f"{EXTENSION_NAME} is not built. Run {BUILD_COMMAND}."


def extension_or_skip() -> ModuleType:
    """Load torch's shared libraries before importing a discoverable extension."""
    if importlib.util.find_spec(EXTENSION_NAME) is None:
        pytest.skip(MISSING_EXTENSION)
    importlib.import_module("torch")
    return importlib.import_module(EXTENSION_NAME)


def test_the_vendored_source_uses_the_macos_file_time_field() -> None:
    """macOS ``struct stat`` has ``st_mtimespec``; ``st_mtim`` is the Linux spelling."""
    text = SOURCE.read_text(encoding="utf-8")
    block = re.search(r"#if defined\(__APPLE__\)(.*?)#else(.*?)#endif", text, re.S)
    assert block is not None, "the file time fields need an __APPLE__ branch"
    apple, other = block.group(1), block.group(2)
    assert "st_mtimespec.tv_sec" in apple and "st_mtimespec.tv_nsec" in apple
    assert "info.st_mtim." not in apple
    assert "info.st_mtim.tv_sec" in other and "info.st_mtim.tv_nsec" in other


def test_the_build_script_and_cuda_decode_source_are_vendored() -> None:
    assert (EXT_DIR / "README.md").is_file()
    assert (EXT_DIR / "build_ext.py").is_file()
    assert (EXT_DIR / "nf4_decode.cu").is_file()


def test_importing_soup_cli_does_not_load_the_native_extension() -> None:
    """A clean checkout has no compiled module; the package must import without one."""
    source_root = Path(soup_cli.__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    code = (
        "import sys, soup_cli\n"
        "assert soup_cli.__version__\n"
        "assert '_native_expert_stream_async' not in sys.modules\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'torch.utils.cpp_extension' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-c", code], cwd=str(source_root), env=env, check=True,
        capture_output=True, text=True, timeout=30,
    )


def test_the_compiled_extension_exports_the_reader() -> None:
    assert callable(extension_or_skip().PackedStreamAsync)


def test_an_absent_extension_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, EXTENSION_NAME, None)
    with pytest.raises(pytest.skip.Exception) as caught:
        extension_or_skip()
    assert str(caught.value) == MISSING_EXTENSION


def test_a_discoverable_broken_extension_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / f"{EXTENSION_NAME}.py").write_text(
        'raise ImportError("native expert reader ABI mismatch")\n', encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, EXTENSION_NAME, raising=False)
    # The synthetic extension needs no torch libraries.
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))

    with pytest.raises(ImportError, match="^native expert reader ABI mismatch$"):
        extension_or_skip()


@pytest.mark.skipif(importlib.util.find_spec("hatchling") is None, reason="needs hatchling")
def test_wheel_contains_native_sources_and_build_script(tmp_path: Path) -> None:
    root = Path(__file__).absolute().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(tmp_path)],
        cwd=root, capture_output=True, text=True, check=False, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    (wheel,) = tmp_path.glob("soup_cli-*.whl")
    prefix = "soup_cli/utils/native_expert_reader/"
    expected = {"README.md", "build_ext.py", "native_stream_async.cpp", "nf4_decode.cu"}
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in expected:
            assert names.count(prefix + name) == 1, name
        assert not any(Path(name).name.startswith(EXTENSION_NAME) for name in names)
