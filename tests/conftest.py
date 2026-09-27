"""Shared pytest fixtures and helpers."""

import functools
import json
import os
import re
import tempfile
from pathlib import Path

import pytest

#: The skip message for a ``gpu``-marked test on a machine without CUDA (#833).
GPU_SKIP_REASON = "needs a CUDA device (run the GPU subset with: pytest -m gpu --no-cov)"


def _probe_initial_device_count() -> int:
    """Probe the visible CUDA device count before any test initialises CUDA (#1128).

    Under CUDA_VISIBLE_DEVICES="", PyTorch may report device_count() == 0 pre-init
    but report 1 if a CUDA tensor is allocated post-init. Pinning the pre-init
    count at session start keeps the gpu marker deterministic across the suite.
    """
    try:
        import torch
    except Exception:
        return 0
    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001 — broken CUDA install or driver
        return 0


_INITIAL_CUDA_DEVICE_COUNT: int = _probe_initial_device_count()


def _cuda_device_available() -> bool:
    """Check if a usable CUDA device was visible at session start (#1128)."""
    return _INITIAL_CUDA_DEVICE_COUNT > 0


@functools.lru_cache(maxsize=None)
def cuda_available() -> bool:
    """The build probe for test bodies; the gpu marker uses _cuda_device_available() (#833).

    Fifteen modules had grown a private copy of this, and nine of them turned it
    into an identical ``requires_cuda`` skipif, so ``pytest -m gpu`` had nothing
    to select. Gate a test that needs CUDA with ``@pytest.mark.gpu`` (optionally
    ``@pytest.mark.gpu(reason="...")``) rather than calling this in a skip;
    ``tests/test_issue833_gpu_marker.py`` holds that line.
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — a broken CUDA install counts as no CUDA
        return False


#: Rich/Pygments emit SGR escapes *between* the tokens of one logical line, so a
#: multi-token substring like "modality: text" is absent from raw output and
#: yaml.safe_load rejects \x1b outright (#633). 38 test files had grown their
#: own copy of this regex; new code should import this one.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: "str | None") -> str:
    """Return ``text`` with SGR escape sequences removed."""
    return _ANSI_ESCAPE.sub("", text or "")


#: Why a ``requires_symlink`` test was skipped. It names the account's missing capability, not
#: the platform: Windows creates symlinks fine with Developer Mode on or when elevated, which is
#: how the Windows CI cells run them (#832).
SYMLINK_SKIP_REASON = "this account cannot create symlinks (enable Developer Mode or run elevated)"


@functools.lru_cache(maxsize=None)
def can_symlink() -> bool:
    """Whether this process can create a symlink. Probed once per session."""
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "target")
        with open(target, "w", encoding="utf-8"):
            pass
        try:
            os.symlink(target, os.path.join(tmp, "link"))
        except (OSError, NotImplementedError, AttributeError):
            return False
    return True


#: Why a ``requires_hardlink`` test was skipped (#1158).
HARDLINK_SKIP_REASON = "this filesystem does not support hard links"


@functools.lru_cache(maxsize=None)
def can_hardlink() -> bool:
    """Whether this filesystem/account supports creating hard links (#1158).

    Probed once per session.
    """
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "target")
        with open(target, "w", encoding="utf-8"):
            pass
        try:
            os.link(target, os.path.join(tmp, "link"))
        except (OSError, NotImplementedError, AttributeError):
            return False
    return True


def pytest_runtest_setup(item: pytest.Item) -> None:
    """The one collection hook. All capability markers go through here (#832, #833, #1158).

    pytest calls a plugin hook once per definition, and a module can only hold one
    ``pytest_runtest_setup`` -- a second ``def`` silently replaces the first, taking its
    marker with it. Keeping the checks in one function is what makes that impossible
    rather than merely unlikely, so add the next capability marker here too.
    """
    gpu = item.get_closest_marker("gpu")
    if gpu is not None and not _cuda_device_available():
        why = gpu.kwargs.get("reason")
        pytest.skip(f"{GPU_SKIP_REASON}: {why}" if why else GPU_SKIP_REASON)

    if item.get_closest_marker("requires_symlink") is not None and not can_symlink():
        pytest.skip(SYMLINK_SKIP_REASON)

    if item.get_closest_marker("requires_hardlink") is not None and not can_hardlink():
        pytest.skip(HARDLINK_SKIP_REASON)


@pytest.fixture(autouse=True)
def _isolate_experiments_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the experiments DB at a per-test temp file.

    The MCP capacity gate now reads persisted runs from the tracker (issue
    #402), so a test must not see 'running' rows left in the real
    ``~/.soup/experiments.db`` by earlier tests or by the developer's own runs.
    A test that needs a specific DB overrides ``SOUP_DB_PATH`` itself; this only
    provides a clean, isolated default.
    """
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))


@pytest.fixture
def aten_half_matmuls(monkeypatch: pytest.MonkeyPatch):
    """Keep a CPU bf16/fp16 step off oneDNN and off MKL's half-precision GEMMs.

    For a test that runs a real step under CPU autocast: its half-precision matmuls
    run on ATen's precompiled kernels instead, and attention runs in fp32. Two routes
    in such a step pick their instruction set at run time from the library's own CPU
    probe:

    * a bf16 matmul whose ``m*n*k`` exceeds 16**3 goes to oneDNN whenever oneDNN
      reports bf16 for the CPU (an AVX-512 or AVX2-VNNI-2 host), and to ATen's
      precompiled ``gemm`` otherwise (``mkldnn_gemm``,
      ``aten/src/ATen/native/mkldnn/Matmul.cpp``); fp16 goes the same way;
    * the fused CPU SDPA kernel multiplies its bf16/fp16 blocks with MKL's
      ``cblas_gemm_bf16bf16f32`` / ``cblas_gemm_f16f16f32`` (``MKL_VERBOSE=1`` logs
      128 ``GEMM_BF16BF16F32`` calls in one step of #1235's test).

    On part of GitHub's ``windows-latest`` fleet the first such matmul, a oneDNN one,
    kills the interpreter with ``0xc000001d`` (main CI run 36251313755), while the
    same hosts run fp32 training (MKL SGEMM) and bf16 element-wise ops (ATen) without
    a fault. ``torch.backends.mkldnn.enabled = False`` sends the first route to ATen
    (it also gates the SDPA brgemm), and SDPA's math backend upcasts q/k/v to fp32,
    which puts attention on the SGEMM an fp32 run already uses.

    Unlike ``skip_on_windows_ci`` the test keeps running on every cell under the same
    autocast. A box without AVX-512 never reached oneDNN here anyway
    (``ONEDNN_VERBOSE=1`` logs no primitive for these steps).
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    monkeypatch.setattr(torch.backends.mkldnn, "enabled", False)
    with sdpa_kernel([SDPBackend.MATH]):
        yield


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temp directory with sample training data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_alpaca_data(tmp_data_dir: Path) -> Path:
    """Create a sample alpaca-format JSONL file."""
    path = tmp_data_dir / "train.jsonl"
    samples = [
        {
            "instruction": "What is Python?",
            "input": "",
            "output": "Python is a programming language.",
        },
        {
            "instruction": "Explain gravity",
            "input": "",
            "output": "Gravity is a fundamental force.",
        },
        {
            "instruction": "Translate hello to Spanish",
            "input": "hello",
            "output": "hola",
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return path


@pytest.fixture
def sample_config(tmp_path: Path, sample_alpaca_data: Path) -> Path:
    """Create a sample soup.yaml config."""
    config_path = tmp_path / "soup.yaml"
    config_path.write_text(
        f"""base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: {sample_alpaca_data}
  format: alpaca
  val_split: 0.1
training:
  epochs: 1
  lr: 2e-5
  batch_size: 1
  lora:
    r: 8
    alpha: 16
  quantization: 4bit
output: {tmp_path / 'output'}
""",
        encoding="utf-8",
    )
    return config_path
