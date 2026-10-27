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
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda="13.0")))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2.14.0+cu130")
    driver_probe = Mock(return_value=(13, 0))
    monkeypatch.setattr(doctor, "_nvidia_smi_cuda_version", driver_probe)
    return driver_probe


@pytest.mark.parametrize("driver_version", [(13, 0), None], ids=["known-driver", "unknown-driver"])
@pytest.mark.parametrize(
    "visibility",
    [
        "-1",
        " -1",
        "-1 ",
        "-1,0",
        "-2",
        "99",
        "abc",
        "NoDevFiles",
        "",
        "gpu-unknown",
        "mig-unknown",
        # Windows CUDA can normalize fullwidth zero to ASCII; this keeps the non-ASCII case.
        pytest.param("٠", id="arabic-indic-zero"),
    ],
)
def test_hidden_cuda_gpu_reports_visibility_without_reinstall(
    monkeypatch: pytest.MonkeyPatch,
    cuda_hardware: Mock,
    driver_version: tuple[int, int] | None,
    visibility: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    cuda_hardware.return_value = driver_version

    advisory = doctor._detect_gpu_hw_without_torch_cuda()

    assert "NVIDIA Test GPU" in advisory
    assert f"CUDA_VISIBLE_DEVICES={visibility}" in advisory
    assert "hides" in advisory
    assert "Unset CUDA_VISIBLE_DEVICES to use the GPU" in advisory
    assert "--force-reinstall" not in advisory
    assert "pip install" not in advisory
    assert "could not initialise" not in advisory
    cuda_hardware.assert_not_called()


@pytest.mark.parametrize(
    "visibility",
    [None, "0", "0,-1", " 0 ", "GPU-unknown", "MIG-GPU-unknown/1/2"],
    ids=[
        "unset",
        "device-zero",
        "invalid-after-visible",
        "padded-visible",
        "unknown-uuid",
        "unknown-mig",
    ],
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


@pytest.mark.parametrize(
    ("visibility", "gpu_count", "hidden"), [("1", 1, True), ("1", 2, False), ("2", 2, True)]
)
def test_device_index_uses_all_detected_gpu_rows(
    monkeypatch: pytest.MonkeyPatch,
    cuda_hardware: Mock,
    visibility: str,
    gpu_count: int,
    hidden: bool,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="NVIDIA Test GPU\n" * gpu_count, stderr=""
        ),
    )

    advisory = doctor._detect_gpu_hw_without_torch_cuda()

    if hidden:
        assert f"CUDA_VISIBLE_DEVICES={visibility}" in advisory
        assert "Unset CUDA_VISIBLE_DEVICES" in advisory
        assert "pip install" not in advisory
        cuda_hardware.assert_not_called()
    else:
        assert "CUDA build (13.0) could not initialise" in advisory
        assert "--force-reinstall" in advisory
        cuda_hardware.assert_called_once_with()


def test_hidden_visibility_value_is_safe_inside_rich_markup(
    monkeypatch: pytest.MonkeyPatch, cuda_hardware: Mock
) -> None:
    from rich.text import Text

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "[red]x\x1b[31m")

    advisory = doctor._detect_gpu_hw_without_torch_cuda()

    assert r"CUDA_VISIBLE_DEVICES=\[red]x[31m" in advisory
    assert "\x1b" not in advisory
    rendered = Text.from_markup(f"[dim]{advisory}[/]")
    assert "CUDA_VISIBLE_DEVICES=[red]x[31m" in rendered.plain
    assert all(str(span.style) == "dim" for span in rendered.spans)
    assert "pip install" not in advisory
    cuda_hardware.assert_not_called()


@pytest.mark.parametrize("visibility", [None, "-1"], ids=["existing-reinstall", "hidden"])
def test_gpu_name_is_escaped_in_both_advisories(
    monkeypatch: pytest.MonkeyPatch, cuda_hardware: Mock, visibility: str | None
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if visibility is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="[red]Test\n", stderr=""
        ),
    )

    advisory = doctor._detect_gpu_hw_without_torch_cuda()

    assert r"\[red]Test" in advisory


def test_oversized_invalid_index_does_not_crash_doctor(
    monkeypatch: pytest.MonkeyPatch, cuda_hardware: Mock
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9" * 5000)

    advisory = doctor._detect_gpu_hw_without_torch_cuda()

    assert "hides" in advisory
    assert "Unset CUDA_VISIBLE_DEVICES" in advisory
    assert "pip install" not in advisory
    cuda_hardware.assert_not_called()
