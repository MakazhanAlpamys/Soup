"""The CUDA 13 / glibc 2.43 ``rsqrt`` conflict is diagnosed before ``nvcc`` runs.

CUDA 13.0's ``crt/math_functions.h`` declares ``rsqrt(double)`` and ``rsqrtf(float)``
without the ``noexcept (true)`` that glibc 2.43 gives the same declarations. ``nvcc`` stops
inside the header before it reaches ``nf4_decode.cu``, and the message names the toolkit's
header rather than anything the operator set. The build script must explain the conflict and
the documented two-line patch, and it must NOT edit the user's environment: only read the
header and refuse.

These checks feed the pure diagnosis real headers from ``tmp_path`` so they run on every
machine, patched or not, with or without a toolkit.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Tuple

import pytest

BUILD_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "soup_cli"
    / "utils"
    / "native_expert_reader"
    / "build_ext.py"
)

DECLARATION_LINES = {
    629: "extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double  rsqrt(double x)",
    653: "extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float   rsqrtf(float x)",
}


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("soup_native_build_ext", BUILD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _header(tmp_path: Path, *, patched: bool) -> Path:
    """A CUDA-shaped header whose only relevant lines are 629 and 653."""
    lines = [f"/* filler line {number} */" for number in range(1, 700)]
    for number, declaration in DECLARATION_LINES.items():
        suffix = " noexcept (true);" if patched else ";"
        lines[number - 1] = declaration + suffix
    path = tmp_path / "math_functions.h"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_an_unpatched_header_names_both_lines_the_backup_and_the_rollback(
    tmp_path: Path,
) -> None:
    message = _module().diagnose_rsqrt_conflict(
        _header(tmp_path, patched=False), glibc=(2, 43), cuda=(13, 0)
    )
    assert message is not None
    for expected in (
        "629",
        "653",
        "rsqrt(double x)",
        "rsqrtf(float x)",
        "noexcept (true)",
        "math_functions.h.soup-backup",
        "nvidia-cuda-crt",
        "--force-reinstall",
        "glibc 2.43",
        "CUDA 13.0",
    ):
        assert expected in message, expected
    assert "This script did not modify anything: the check only read the header." in message


def test_a_patched_header_needs_no_diagnosis(tmp_path: Path) -> None:
    module = _module()
    assert module.diagnose_rsqrt_conflict(
        _header(tmp_path, patched=True), glibc=(2, 43), cuda=(13, 0)
    ) is None


@pytest.mark.parametrize(
    ("glibc", "cuda"),
    [((2, 42), (13, 0)), ((2, 43), (12, 8)), ((2, 43), None), (None, (13, 0))],
)
def test_only_the_glibc_2_43_and_cuda_13_pairing_is_diagnosed(
    tmp_path: Path, glibc: Tuple[int, int] | None, cuda: Tuple[int, int] | None
) -> None:
    assert _module().diagnose_rsqrt_conflict(
        _header(tmp_path, patched=False), glibc=glibc, cuda=cuda
    ) is None


def test_glibc_version_parsing_reads_the_confstr_spelling() -> None:
    module = _module()
    assert module.parse_glibc_version("glibc 2.43") == (2, 43)
    assert module.parse_glibc_version("glibc 2.39-0ubuntu8.5") == (2, 39)
    assert module.parse_glibc_version("musl libc (x86_64)") is None
    assert module.parse_glibc_version("") is None


def test_the_toolkit_version_comes_from_version_json(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "version.json").write_text(
        '{"cuda": {"name": "CUDA SDK", "version": "13.0.88"}}', encoding="utf-8"
    )
    assert module._cuda_version(tmp_path) == (13, 0)
    (tmp_path / "version.json").write_text(
        '{"cuda": {"version": "12.8.61"}}', encoding="utf-8"
    )
    assert module._cuda_version(tmp_path) == (12, 8)


def test_the_check_refuses_without_touching_the_header(tmp_path: Path) -> None:
    module = _module()
    header = _header(tmp_path, patched=False)
    before = header.read_bytes()
    before_stat = header.stat().st_mtime_ns
    with pytest.raises(module.ToolchainConflictError) as caught:
        module.check_toolchain(header=header, glibc=(2, 43), cuda=(13, 0))
    assert str(caught.value).startswith("CUDA 13.0 and glibc 2.43 disagree about rsqrt")
    assert header.read_bytes() == before
    assert header.stat().st_mtime_ns == before_stat


def test_a_missing_header_is_not_a_conflict(tmp_path: Path) -> None:
    module = _module()
    assert module.diagnose_rsqrt_conflict(
        tmp_path / "absent.h", glibc=(2, 43), cuda=(13, 0)
    ) is None


def test_main_runs_the_check_before_any_extension_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """``nvcc`` must never start: the check raises before ``CUDAExtension`` exists."""
    module = _module()
    calls: List[str] = []

    def refuse() -> None:
        calls.append("check")
        raise module.ToolchainConflictError("CUDA 13.0 and glibc 2.43 disagree about rsqrt")

    monkeypatch.setattr(module, "check_toolchain", refuse)
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")
    monkeypatch.setattr(
        cpp_extension, "CUDAExtension", lambda *args, **kwargs: calls.append("cuda-extension")
    )
    monkeypatch.setattr(
        cpp_extension, "CppExtension", lambda *args, **kwargs: calls.append("cpp-extension")
    )
    monkeypatch.setattr(module, "setup", lambda *args, **kwargs: calls.append("setup"))
    monkeypatch.setattr(module.sys, "argv", ["build_ext.py", "build_ext", "--with-cuda-nf4"])
    with pytest.raises(
        module.ToolchainConflictError, match=r"CUDA 13\.0 and glibc 2\.43 disagree about rsqrt"
    ):
        module.main()
    assert calls == ["check"]


def test_a_cpu_only_build_does_not_probe_the_cuda_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: List[str] = []

    def unexpected() -> None:
        calls.append("check")

    monkeypatch.setattr(module, "check_toolchain", unexpected)
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")
    monkeypatch.setattr(
        cpp_extension, "CUDAExtension", lambda *args, **kwargs: calls.append("cuda-extension")
    )
    monkeypatch.setattr(
        cpp_extension, "CppExtension", lambda *args, **kwargs: calls.append("cpp-extension")
    )
    monkeypatch.setattr(module, "setup", lambda *args, **kwargs: calls.append("setup"))
    monkeypatch.setattr(
        sys, "argv", ["build_ext.py", "build_ext", "--build-lib", str(tmp_path)]
    )
    module.main()
    assert "check" not in calls
    assert "cpp-extension" in calls and "setup" in calls


def test_cuda_build_keeps_subnormals_and_compiles_both_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")
    module = _module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["build_ext.py", "build_ext", "--build-lib", "lib", "--with-cuda-nf4"],
    )
    events = []

    def cuda_extension(**kwargs):
        events.append("cuda-extension")
        assert {Path(source).name for source in kwargs["sources"]} == {
            "native_stream_async.cpp", "nf4_decode.cu"
        }
        assert kwargs["define_macros"] == [("SOUP_WITH_CUDA_NF4", "1")]
        assert "--ftz=false" in kwargs["extra_compile_args"]["nvcc"]
        return object()

    def setup(**kwargs):
        events.append("setup")
        assert sys.argv[1:] == ["build_ext", "--build-lib", str((tmp_path / "lib").resolve())]
        assert len(kwargs["ext_modules"]) == 1

    monkeypatch.setattr(module, "check_toolchain", lambda: events.append("check"))
    monkeypatch.setattr(cpp_extension, "CUDAExtension", cuda_extension)
    monkeypatch.setattr(module, "setup", setup)
    module.main()
    assert events == ["check", "cuda-extension", "setup"]
    assert Path.cwd() == tmp_path.resolve()
