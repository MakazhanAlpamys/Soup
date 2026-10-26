"""Distinguish a deliberately hidden GPU from a broken CUDA installation."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from soup_cli.commands import doctor


@pytest.fixture
def cuda_hardware(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Present an NVIDIA card and CUDA torch without requiring either locally."""
    monkeypatch.setattr(doctor, "_nvidia_smi_executable", lambda: "/fake/nvidia-smi")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="NVIDIA Test GPU\n", stderr=""
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda="13.0"))
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2.14.0+cu130")
    driver_probe = Mock(return_value=(13, 0))
    monkeypatch.setattr(doctor, "_nvidia_smi_cuda_version", driver_probe)
    return driver_probe


@pytest.mark.parametrize("driver_version", [(13, 0), None], ids=["known-driver", "unknown-driver"])
def test_hidden_cuda_gpu_reports_visibility_without_reinstall(
    monkeypatch: pytest.MonkeyPatch, cuda_hardware: Mock, driver_version: tuple[int, int] | None
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    cuda_hardware.return_value = driver_version

    advisory = doctor._detect_gpu_hw_without_torch_cuda()

    assert "NVIDIA Test GPU" in advisory
    assert "CUDA_VISIBLE_DEVICES=-1" in advisory
    assert "hides" in advisory
    assert "Unset CUDA_VISIBLE_DEVICES to use the GPU" in advisory
    assert "--force-reinstall" not in advisory
    assert "pip install" not in advisory
    assert "could not initialise" not in advisory
    cuda_hardware.assert_not_called()


@pytest.mark.parametrize(
    "visibility", [None, "0", "-1,0"], ids=["unset", "device-zero", "other-value"]
)
def test_other_visibility_preserves_cuda_reinstall_advice(
    monkeypatch: pytest.MonkeyPatch, cuda_hardware: Mock, visibility: str | None
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if visibility is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)

    advisory = doctor._detect_gpu_hw_without_torch_cuda()

    assert "CUDA build (13.0) could not initialise" in advisory
    assert '--force-reinstall "torch>=2.6.0"' in advisory
    assert "https://download.pytorch.org/whl/cu130" in advisory
    cuda_hardware.assert_called_once_with()


@pytest.mark.parametrize("hardware_state", ["missing", "failed"])
def test_hidden_variable_without_detected_hardware_has_no_advisory(
    monkeypatch: pytest.MonkeyPatch, cuda_hardware: Mock, hardware_state: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    if hardware_state == "missing":
        monkeypatch.setattr(doctor, "_nvidia_smi_executable", lambda: None)
    else:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="driver unavailable"
            ),
        )

    assert doctor._detect_gpu_hw_without_torch_cuda() == ""
    cuda_hardware.assert_not_called()
