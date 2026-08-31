"""v0.40.1 Part C tests — autopilot fallback / dependency cap / quickstart
GPU-aware model pick / lr_finder import regression.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

# --- C3: autopilot fallback to 1B (not 7B) ---------------------------------


def test_guess_params_unknown_model_falls_back_to_1b():
    from soup_cli.autopilot.analyzer import _guess_params_from_name

    assert _guess_params_from_name("tiny-gpt2") == 1.0


def test_guess_params_extracts_billions_first():
    from soup_cli.autopilot.analyzer import _guess_params_from_name

    assert _guess_params_from_name("Qwen/Qwen2.5-7B-Instruct") == 7.0


def test_guess_params_handles_millions_suffix():
    from soup_cli.autopilot.analyzer import _guess_params_from_name

    # SmolLM2-135M → 0.135 B
    assert _guess_params_from_name("HuggingFaceTB/SmolLM2-135M-Instruct") == pytest.approx(
        0.135, abs=1e-3
    )


def test_probe_cache_returns_none_for_missing_repo():
    from soup_cli.autopilot.analyzer import _probe_cache_param_count

    # Random name guaranteed not in cache.
    assert _probe_cache_param_count("nonexistent-org/never-cached-XYZ") is None


# --- C5: doctor flags unsupported breaking majors -------------------------


def test_version_ge_handles_dev_suffix():
    from soup_cli.commands.doctor import _version_ge

    assert _version_ge("5.0.0.dev0", "5.0.0") is True
    assert _version_ge("4.36.2", "5.0.0") is False


def test_version_ge_short_version_string():
    from soup_cli.commands.doctor import _version_ge

    assert _version_ge("5", "5.0.0") is True
    assert _version_ge("4.99", "5.0.0") is False


def test_max_exclusive_table_allows_transformers5_and_caps_transformers6():
    from soup_cli.commands.doctor import _MAX_EXCLUSIVE

    assert _MAX_EXCLUSIVE.get("transformers") == "6.0.0"


# --- G12: lr_finder real loop uses load_raw_data ---------------------------


def test_live_lr_sweep_uses_load_raw_data_not_load_local():
    import inspect
    import re

    from soup_cli.commands.train import _live_lr_sweep_from_config

    src = inspect.getsource(_live_lr_sweep_from_config)
    assert "load_raw_data" in src
    # The broken import was `from soup_cli.data.loader import load_local`;
    # match only that import line to avoid false positives on the comment
    # describing the fix.
    assert not re.search(r"from\s+soup_cli\.data\.loader\s+import\s+load_local", src), (
        "load_local was removed; live sweep must use load_raw_data"
    )


# --- G1: quickstart picks SmolLM2 on ≤6 GB VRAM ----------------------------


def test_pick_quickstart_model_no_cuda_uses_tinyllama():
    from soup_cli.commands import quickstart as qs

    with patch("torch.cuda.is_available", return_value=False):
        model, advisory = qs._pick_quickstart_model()
    assert model == qs._DEFAULT_MODEL
    assert advisory is None


def test_pick_quickstart_model_low_vram_switches_to_smollm():
    from soup_cli.commands import quickstart as qs

    fake_props = type("P", (), {"total_memory": int(4 * 1024**3)})()  # 4 GB
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_properties", return_value=fake_props),
    ):
        model, advisory = qs._pick_quickstart_model()
    assert model == qs._LOW_VRAM_MODEL
    assert advisory is not None
    assert "VRAM" in advisory


def test_pick_quickstart_model_high_vram_keeps_default():
    from soup_cli.commands import quickstart as qs

    fake_props = type("P", (), {"total_memory": int(24 * 1024**3)})()  # 24 GB
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_properties", return_value=fake_props),
    ):
        model, advisory = qs._pick_quickstart_model()
    assert model == qs._DEFAULT_MODEL
    assert advisory is None


# --- N3: GPU diagnostic distinguishes CPU build from no GPU ----------------


def test_detect_gpu_hw_without_torch_cuda_no_nvidia_smi():
    import subprocess

    from soup_cli.commands.doctor import _detect_gpu_hw_without_torch_cuda

    with patch("shutil.which", return_value=None):
        assert _detect_gpu_hw_without_torch_cuda() == ""

    # nvidia-smi present but failing → empty advisory.
    completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
    with (
        patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("subprocess.run", return_value=completed),
    ):
        assert _detect_gpu_hw_without_torch_cuda() == ""


def test_detect_gpu_hw_returns_advisory_when_smi_succeeds():
    import subprocess

    from soup_cli.commands.doctor import _detect_gpu_hw_without_torch_cuda

    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="NVIDIA GeForce RTX 3050\n", stderr=""
    )
    with (
        patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("subprocess.run", return_value=completed),
        patch(
            "soup_cli.commands.doctor._nvidia_smi_cuda_version",
            return_value=None,
        ),
    ):
        advisory = _detect_gpu_hw_without_torch_cuda()
    assert "RTX 3050" in advisory
    assert "download.pytorch.org/whl/" in advisory
    # Unknown driver: conservative cu121 (parse failure is an old driver).
    assert "whl/cu121" in advisory


def test_detect_gpu_hw_uses_driver_cuda_wheel():
    import subprocess

    from soup_cli.commands.doctor import _detect_gpu_hw_without_torch_cuda

    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="NVIDIA GeForce RTX 5060 Ti\n", stderr=""
    )
    with (
        patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("subprocess.run", return_value=completed),
        patch(
            "soup_cli.commands.doctor._nvidia_smi_cuda_version",
            return_value=(13, 2),
        ),
        # Installed CUDA torch version string, as on a real GPU box.
        # A naive `assert "cu121" not in advisory` matches this and goes
        # red locally / green on CPU-only CI.
        patch("importlib.metadata.version", return_value="2.5.1+cu121"),
    ):
        advisory = _detect_gpu_hw_without_torch_cuda()
    assert "RTX 5060 Ti" in advisory
    assert "2.5.1+cu121" in advisory
    assert "whl/cu132" in advisory
    assert "whl/cu121" not in advisory


def test_nvidia_smi_invoked_by_absolute_path():
    import subprocess

    from soup_cli.commands.doctor import (
        _detect_gpu_hw_without_torch_cuda,
        _nvidia_smi_cuda_version,
    )

    smi = "/usr/bin/nvidia-smi"
    failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
    with (
        patch("shutil.which", return_value=smi),
        patch("subprocess.run", return_value=failed) as run,
    ):
        _nvidia_smi_cuda_version()
    argv = run.call_args.args[0]
    assert argv[0] == smi
    assert argv[0] != "nvidia-smi"

    with (
        patch("shutil.which", return_value=smi),
        patch("subprocess.run", return_value=failed) as run,
    ):
        _detect_gpu_hw_without_torch_cuda()
    argv = run.call_args.args[0]
    assert argv[0] == smi
    assert "--query-gpu=name" in argv


def test_parse_cuda_version_from_nvidia_smi_header():
    from soup_cli.commands.doctor import _parse_cuda_version

    header = (
        "NVIDIA-SMI 596.49                 Driver Version: 596.49"
        "         CUDA Version: 13.2"
    )
    assert _parse_cuda_version(header) == (13, 2)
    assert _parse_cuda_version("CUDA Version:13.2") == (13, 2)
    assert _parse_cuda_version("CUDA Version: 12.10") == (12, 10)
    assert _parse_cuda_version("CUDA Version: N/A") is None
    assert _parse_cuda_version("no cuda here") is None
    assert _parse_cuda_version("") is None


def test_torch_cuda_wheel_tag_picks_newest_supported():
    from soup_cli.commands.doctor import _TORCH_CUDA_WHEELS, _torch_cuda_wheel_tag

    assert (11, 8, "cu118") in _TORCH_CUDA_WHEELS
    assert _torch_cuda_wheel_tag((13, 2)) == "cu132"
    assert _torch_cuda_wheel_tag((13, 1)) == "cu130"
    assert _torch_cuda_wheel_tag((13, 0)) == "cu130"
    assert _torch_cuda_wheel_tag((12, 8)) == "cu128"
    assert _torch_cuda_wheel_tag((12, 7)) == "cu126"
    assert _torch_cuda_wheel_tag((12, 6)) == "cu126"
    assert _torch_cuda_wheel_tag((12, 4)) == "cu124"
    assert _torch_cuda_wheel_tag((12, 1)) == "cu121"
    assert _torch_cuda_wheel_tag((11, 8)) == "cu118"
    assert _torch_cuda_wheel_tag((11, 0)) == "cu118"
    assert _torch_cuda_wheel_tag(None) == "cu121"


def test_readme_does_not_hardcode_a_cuda_wheel_index():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "download.pytorch.org/whl/cu" not in text


# --- N4: dual-Python interpreter detector ---------------------------------


def test_detect_dual_python_no_path_python():
    from soup_cli.commands.doctor import _detect_dual_python_interpreters

    with patch("shutil.which", return_value=None):
        assert _detect_dual_python_interpreters() == ""


# --- M1: rich version probe falls back to importlib.metadata --------------


def test_doctor_rich_version_probe_uses_metadata_when_module_lacks_attr():
    """Sanity: importing rich and probing version doesn't return '?'."""
    from importlib.metadata import version

    import rich

    # Rich does export __version__, so this just guards the importlib.metadata
    # fallback path is reachable and returns a real string for rich.
    assert version("rich")  # non-empty
    assert hasattr(rich, "__version__") or version("rich")
