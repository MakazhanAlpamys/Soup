"""Tests for soup doctor command."""

import re
import sys
from unittest.mock import patch

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.doctor import _version_ok

runner = CliRunner()

# Rich/Typer emits per-character ANSI escapes when colour is forced
# (FORCE_COLOR=1), which would break substring assertions on commands.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# --- _version_ok tests ---


def test_version_ok_exact():
    assert _version_ok("2.0.0", "2.0.0") is True


def test_version_ok_higher():
    assert _version_ok("2.1.0", "2.0.0") is True


def test_version_ok_lower():
    assert _version_ok("1.9.0", "2.0.0") is False


def test_version_ok_patch():
    assert _version_ok("2.0.1", "2.0.0") is True


def test_version_ok_major_higher():
    assert _version_ok("3.0.0", "2.0.0") is True


def test_version_ok_two_part():
    assert _version_ok("6.0", "6.0") is True


def test_version_ok_unparseable():
    """Unparseable versions should return True (assume OK)."""
    assert _version_ok("unknown", "2.0.0") is True


def test_version_ok_dev_suffix():
    """Version with dev suffix (can't fully parse)."""
    assert _version_ok("2.1.0.dev0", "2.0.0") is True


# --- doctor CLI tests ---


def test_doctor_runs():
    """soup doctor runs without crashing."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Soup Doctor" in result.output


def test_doctor_shows_system_info():
    """soup doctor shows system info panel."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python" in result.output
    assert "Platform" in result.output


def test_doctor_shows_dependencies():
    """soup doctor shows dependency table."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Dependencies" in result.output
    assert "Package" in result.output


def test_gpu_arch_mismatch_advisory_uses_driver_cuda_wheel(monkeypatch):
    from soup_cli.commands.doctor import _detect_gpu_arch_mismatch_advisory

    monkeypatch.setattr(
        "soup_cli.commands.doctor._nvidia_smi_cuda_version",
        lambda: (13, 2),
    )

    advisory = _detect_gpu_arch_mismatch_advisory()

    assert "download.pytorch.org/whl/cu132" in advisory
    assert "pip install torch" in advisory


def test_format_gpu_capability_handles_missing_torch_cuda_attribute():
    from types import SimpleNamespace

    from soup_cli.commands.doctor import _format_gpu_capability

    fake_torch = SimpleNamespace(cuda=SimpleNamespace())

    assert _format_gpu_capability(fake_torch, 0) == ("unknown", False)


def test_gpu_architecture_name_maps_legacy_nvidia_architectures():
    from soup_cli.commands.doctor import _gpu_architecture_name

    assert _gpu_architecture_name(6, 0) == "Pascal"
    assert _gpu_architecture_name(6, 1) == "Pascal"
    assert _gpu_architecture_name(7, 0) == "Volta"
    assert _gpu_architecture_name(7, 2) == "Volta"
    assert _gpu_architecture_name(7, 5) == "Turing"


def test_torch_gpu_arch_supported_accepts_suffixed_sm_target():
    from types import SimpleNamespace

    from soup_cli.commands.doctor import _torch_gpu_arch_supported

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_arch_list=lambda: ["sm_90a", "sm_100a"],
        )
    )

    assert _torch_gpu_arch_supported(fake_torch, 9, 0) is True
    assert _torch_gpu_arch_supported(fake_torch, 10, 0) is True


def test_torch_gpu_arch_supported_accepts_exact_sm_target():
    from types import SimpleNamespace

    from soup_cli.commands.doctor import _torch_gpu_arch_supported

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_arch_list=lambda: ["sm_90", "sm_120"],
        )
    )

    assert _torch_gpu_arch_supported(fake_torch, 12, 0) is True


def test_torch_gpu_arch_supported_accepts_lower_ptx_target():
    from types import SimpleNamespace

    from soup_cli.commands.doctor import _torch_gpu_arch_supported

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_arch_list=lambda: ["sm_90", "compute_90"],
        )
    )

    assert _torch_gpu_arch_supported(fake_torch, 12, 0) is True


def test_torch_gpu_arch_supported_rejects_higher_only_ptx_target():
    from types import SimpleNamespace

    from soup_cli.commands.doctor import _torch_gpu_arch_supported

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_arch_list=lambda: ["sm_80", "compute_120"],
        )
    )

    assert _torch_gpu_arch_supported(fake_torch, 9, 0) is False


def test_doctor_gpu_panel_shows_compute_capability_and_precision(monkeypatch):
    """GPU panel reports compute capability and precision feature gates."""
    import importlib.machinery
    import types

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda idx: "NVIDIA GeForce RTX 5070 Laptop GPU",
        get_device_properties=lambda idx: types.SimpleNamespace(
            total_memory=8 * 1024**3
        ),
        get_device_capability=lambda idx: (12, 0),
        get_arch_list=lambda: ["sm_75", "sm_80", "sm_90", "sm_120"],
        is_bf16_supported=lambda: True,
    )

    fake_torch = types.SimpleNamespace(
        __spec__=importlib.machinery.ModuleSpec("torch", None),
        cuda=fake_cuda,
        version=types.SimpleNamespace(cuda="13.0"),
    )

    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(
        "soup_cli.commands.doctor._installed_version_str",
        lambda import_name, pkg_name: None,
    )
    monkeypatch.setattr(
        "soup_cli.utils.fp8.is_fp8_gpu_supported",
        lambda: True,
    )
    monkeypatch.setattr(
        "soup_cli.utils.fp8.is_fp8_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "soup_cli.utils.advanced_precision.is_blackwell_gpu",
        lambda: True,
    )
    monkeypatch.setattr(
        "soup_cli.utils.advanced_precision._torchao_available",
        lambda: True,
    )

    fake_torchao = types.ModuleType("torchao")
    fake_quantization = types.ModuleType("torchao.quantization")
    fake_quantization.NVFP4Config = object
    fake_quantization.quantize_ = lambda *args, **kwargs: None
    fake_torchao.quantization = fake_quantization

    monkeypatch.setitem(__import__("sys").modules, "torchao", fake_torchao)
    monkeypatch.setitem(
        __import__("sys").modules,
        "torchao.quantization",
        fake_quantization,
    )

    result = runner.invoke(app, ["doctor"])
    out = result.output

    assert result.exit_code == 0
    assert "sm_120 (Blackwell)" in out
    assert "Precision features" in out
    assert "BF16: hardware=yes" in out
    assert "FP8: hardware=yes, software=yes" in out
    assert "NVFP4: hardware=yes, software=yes" in out


def test_get_precision_capabilities_reports_negative_bf16(monkeypatch):
    """BF16 hardware support must reflect the real Torch probe."""
    import types

    fake_cuda = types.SimpleNamespace(
        is_bf16_supported=lambda: False,
    )
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)

    monkeypatch.setattr(
        "soup_cli.utils.fp8.is_fp8_gpu_supported",
        lambda: False,
    )
    monkeypatch.setattr(
        "soup_cli.utils.advanced_precision.is_blackwell_gpu",
        lambda: False,
    )
    monkeypatch.setattr(
        "soup_cli.utils.advanced_precision._torchao_available",
        lambda: False,
    )

    from soup_cli.commands.doctor import _get_precision_capabilities

    result = _get_precision_capabilities(fake_torch)

    assert result["BF16"] == (False, None)


def test_doctor_gpu_panel_warns_when_torch_lacks_gpu_arch(monkeypatch):
    """GPU panel warns when Torch does not include the detected GPU architecture."""
    import importlib.machinery
    import types

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda idx: "NVIDIA GeForce RTX 5070 Laptop GPU",
        get_device_properties=lambda idx: types.SimpleNamespace(
            total_memory=8 * 1024**3
        ),
        get_device_capability=lambda idx: (12, 0),
        get_arch_list=lambda: ["sm_75", "sm_80", "sm_90"],
        is_bf16_supported=lambda: True,
    )

    fake_torch = types.SimpleNamespace(
        __spec__=importlib.machinery.ModuleSpec("torch", None),
        cuda=fake_cuda,
        version=types.SimpleNamespace(cuda="13.0"),
    )

    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(
        "soup_cli.commands.doctor._installed_version_str",
        lambda import_name, pkg_name: None,
    )
    monkeypatch.setattr(
        "soup_cli.utils.fp8.is_fp8_gpu_supported",
        lambda: True,
    )
    monkeypatch.setattr(
        "soup_cli.utils.fp8.is_fp8_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "soup_cli.utils.advanced_precision.is_blackwell_gpu",
        lambda: True,
    )
    monkeypatch.setattr(
        "soup_cli.utils.advanced_precision._torchao_available",
        lambda: True,
    )

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "sm_120 (Blackwell)" in result.output

    from tests.conftest import strip_ansi

    normalized = " ".join(strip_ansi(result.output).split())
    assert "Torch build does not include this GPU architecture" in normalized


def test_doctor_shows_gpu_section():
    """soup doctor shows GPU section."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "GPU" in result.output


def test_doctor_shows_system_resources():
    """soup doctor shows System Resources section."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "System Resources" in result.output
    assert "RAM" in result.output
    assert "Disk" in result.output


def test_doctor_checks_torch():
    """soup doctor checks for torch."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "torch" in result.output


def test_doctor_checks_pydantic():
    """soup doctor checks for pydantic."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "pydantic" in result.output


def test_doctor_checks_optional_deps():
    """soup doctor shows optional deps."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "optional" in result.output


def test_doctor_requires_torchao_070_for_optional_feature_support(monkeypatch):
    """Doctor must report the production torchao floor when it is outdated."""
    import importlib.machinery
    import types

    fake_torchao = types.SimpleNamespace(
        __version__="0.6.0",
        __spec__=importlib.machinery.ModuleSpec("torchao", None),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "torchao",
        fake_torchao,
    )

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "outdated" in result.output
    assert ">=0.7.0" in result.output


def test_doctor_missing_dep():
    """soup doctor reports a missing required dep and exits non-zero (#828)."""
    with patch(
        "soup_cli.commands.doctor.DEPS",
        [
            ("nonexistent_fake_pkg_xyz", "nonexistent-pkg", "1.0.0", True),
        ],
    ):
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1
        assert "MISSING" in result.output


def test_doctor_outdated_dep():
    """soup doctor reports outdated dep."""
    with patch(
        "soup_cli.commands.doctor.DEPS",
        [
            ("sys", "sys", "999.0.0", True),  # sys has no __version__ but import won't fail
        ],
    ):
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        # Either outdated or OK (depends on version attr presence)


# --- NCCL Check tests ---


def test_doctor_missing_train_extra_suggests_extra(monkeypatch):
    """A core-only install suggests the [train] extra, not bare floors (#828)."""
    for name in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "bitsandbytes",
        "accelerate",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    out = _strip_ansi(result.output)
    assert 'pip install "soup-cli[train]"' in out
    assert "torch>=2.6.0" not in out
    assert "transformers>=5.16.1" not in out
    assert "peft>=0.20.0" not in out
    assert "trl>=0.29.0" not in out
    assert "datasets>=2.14.0" not in out
    assert "bitsandbytes>=0.41.0" not in out
    assert "accelerate>=0.27.0" not in out


def test_doctor_partial_train_extra_names_missing_members(monkeypatch):
    """A partial [train] install names the absent members, not the whole stack (#875)."""
    installed = {
        "torch": "2.6.0",
        "transformers": "5.16.1",
        "peft": "0.20.0",
        "trl": "0.29.0",
        "datasets": "2.14.0",
        "accelerate": "0.27.0",
    }

    def _fake_version(import_name, pkg_name):
        return installed.get(pkg_name)

    monkeypatch.setattr("soup_cli.commands.doctor._installed_version_str", _fake_version)
    monkeypatch.setattr("soup_cli.commands.doctor._nvidia_smi_cuda_version", lambda: None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    out = _strip_ansi(result.output)
    assert "Training stack not installed" not in out
    assert "Training stack incomplete, missing: bitsandbytes" in out
    assert 'pip install "soup-cli[train]"' in out
    assert "All checks passed!" not in out


def test_doctor_missing_train_extra_message_unchanged_when_none_installed(monkeypatch):
    """With no [train] members, the existing not-installed message is kept (#875)."""
    monkeypatch.setattr(
        "soup_cli.commands.doctor._installed_version_str", lambda import_name, pkg_name: None
    )
    monkeypatch.setattr("soup_cli.commands.doctor._nvidia_smi_cuda_version", lambda: None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    out = _strip_ansi(result.output)
    assert 'Training stack not installed: pip install "soup-cli[train]"' in out
    assert "Training stack incomplete" not in out
    assert "All checks passed!" not in out


def test_doctor_missing_train_extra_with_unsupported_driver_falls_back_to_extra(monkeypatch):
    """Do not construct a whl/None URL when the driver has no supported wheel."""
    for name in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "bitsandbytes",
        "accelerate",
    ):
        monkeypatch.setitem(sys.modules, name, None)

    monkeypatch.setattr(
        "soup_cli.commands.doctor._nvidia_smi_cuda_version",
        lambda: (11, 7),
    )
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    out = _strip_ansi(result.output)
    assert 'pip install "soup-cli[train]"' in out
    assert "download.pytorch.org/whl/" not in out
    assert "whl/None" not in out


def test_doctor_suggestion_is_colour_safe(monkeypatch):
    """The [train] suggestion survives Rich highlighting (#828 review)."""
    from rich.console import Console

    monkeypatch.setattr("soup_cli.commands.doctor.console", Console(force_terminal=True))
    for name in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "bitsandbytes",
        "accelerate",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert 'pip install "soup-cli[train]"' in _strip_ansi(result.output)


def test_doctor_fix_all_line_keeps_extra_marker(monkeypatch):
    """The Fix all line must not drop [train] to Rich markup (#828 review)."""
    for name in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "bitsandbytes",
        "accelerate",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    out = _strip_ansi(result.output)
    assert "Fix all:" in out
    after = out.split("Fix all:", 1)[1]
    assert 'pip install "soup-cli[train]"' in after


def test_doctor_table_keeps_extra_marker(monkeypatch):
    """The Required column must render [train] literally, not blank (#828 review)."""
    for name in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "bitsandbytes",
        "accelerate",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    result = runner.invoke(app, ["doctor"])
    out = _strip_ansi(result.output)
    torch_rows = [line for line in out.splitlines() if "torch" in line]
    assert torch_rows, "expected a table row for torch"
    assert any("[train]" in line for line in torch_rows)


def test_doctor_incompatible_train_member_is_reported(monkeypatch):
    """A [train] member past its breaking-major ceiling must add an issue (#828 review)."""
    monkeypatch.setattr(
        "soup_cli.commands.doctor.EXTRA_GROUPS",
        [("train", [("transformers", "transformers", "5.16.1")])],
    )
    monkeypatch.setattr(
        "soup_cli.commands.doctor._installed_version_str", lambda import_name, pkg_name: "6.1.0"
    )
    result = runner.invoke(app, ["doctor"])
    out = _strip_ansi(result.output)
    assert "INCOMPATIBLE" in out
    assert 'Downgrade transformers: pip install "transformers>=5.16.1,<6.0.0"' in out
    assert "All checks passed!" not in out


def test_doctor_out_of_range_train_member_is_reported(monkeypatch):
    """An installed-but-out-of-range [train] member must add an issue (#828 review)."""
    monkeypatch.setattr(
        "soup_cli.commands.doctor.EXTRA_GROUPS",
        [("train", [("pydantic", "pydantic", "999.0.0")])],
    )
    result = runner.invoke(app, ["doctor"])
    out = _strip_ansi(result.output)
    assert "outdated" in out
    assert "All checks passed!" not in out


def test_doctor_incompatible_train_member_exits_nonzero(monkeypatch):
    """A [train] member past its ceiling makes `soup doctor` exit non-zero (#874)."""
    monkeypatch.setattr(
        "soup_cli.commands.doctor.EXTRA_GROUPS",
        [("train", [("transformers", "transformers", "5.16.1")])],
    )
    monkeypatch.setattr(
        "soup_cli.commands.doctor._installed_version_str", lambda import_name, pkg_name: "6.1.0"
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "INCOMPATIBLE" in _strip_ansi(result.output)


def test_doctor_incompatible_core_dependency_exits_nonzero(monkeypatch):
    """A core dependency past its ceiling makes `soup doctor` exit non-zero (#874)."""
    monkeypatch.setattr("soup_cli.commands.doctor.DEPS", [("typer", "typer", "0.1.0", True)])
    monkeypatch.setattr("soup_cli.commands.doctor._MAX_EXCLUSIVE", {"typer": "0.2.0"})
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "INCOMPATIBLE" in _strip_ansi(result.output)


def test_doctor_outdated_train_member_exits_zero(monkeypatch):
    """An outdated [train] member stays advisory: only beyond-ceiling blocks (#874)."""
    monkeypatch.setattr(
        "soup_cli.commands.doctor.EXTRA_GROUPS",
        [("train", [("transformers", "transformers", "5.16.1")])],
    )
    monkeypatch.setattr(
        "soup_cli.commands.doctor._installed_version_str", lambda import_name, pkg_name: "4.0.0"
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "outdated" in _strip_ansi(result.output)


def test_doctor_partial_train_group_in_range_exits_zero(monkeypatch):
    """A partially installed [train] group with in-range members stays exit 0 (#874)."""
    monkeypatch.setattr(
        "soup_cli.commands.doctor.EXTRA_GROUPS",
        [
            (
                "train",
                [
                    ("transformers", "transformers", "5.16.1"),
                    ("bitsandbytes", "bitsandbytes", "0.41.0"),
                ],
            )
        ],
    )
    monkeypatch.setattr(
        "soup_cli.commands.doctor._installed_version_str",
        lambda import_name, pkg_name: "5.16.1" if pkg_name == "transformers" else None,
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    out = _strip_ansi(result.output)
    assert "OK" in out
    assert "not installed" in out


def test_doctor_nvidia_train_suggestion_is_two_step(monkeypatch):
    """On an NVIDIA box the [train] suggestion installs torch from its own index (#828 review)."""
    monkeypatch.setattr(
        "soup_cli.commands.doctor._nvidia_smi_cuda_version", lambda: (13, 0)
    )
    for name in ("torch", "transformers", "peft", "trl", "datasets", "bitsandbytes", "accelerate"):
        monkeypatch.setitem(sys.modules, name, None)
    result = runner.invoke(app, ["doctor"])
    out = _strip_ansi(result.output)
    assert "pip install torch --index-url https://download.pytorch.org/whl/" in out
    assert 'pip install "soup-cli[train]"' in out
    # The broken single-step form must be gone.
    assert '["soup-cli[train]" --index-url' not in out and '[train]" --index-url' not in out


def test_doctor_nvidia_partial_stack_with_torch_missing(monkeypatch):
    """NVIDIA + partial [train] + torch missing keeps the two-step index URL (#884)."""
    installed = {
        "transformers": "5.16.1",
        "peft": "0.20.0",
        "trl": "0.29.0",
        "datasets": "2.14.0",
        "bitsandbytes": "0.41.0",
        "accelerate": "0.27.0",
    }

    def _fake_version(import_name, pkg_name):
        return installed.get(pkg_name)

    monkeypatch.setattr("soup_cli.commands.doctor._installed_version_str", _fake_version)
    monkeypatch.setattr(
        "soup_cli.commands.doctor._nvidia_smi_cuda_version", lambda: (13, 0)
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    out = _strip_ansi(result.output)
    assert "Training stack incomplete, missing: torch" in out
    assert "pip install torch --index-url https://download.pytorch.org/whl/" in out
    assert 'pip install "soup-cli[train]"' in out
    assert "All checks passed!" not in out
    assert "Training stack not installed" not in out


def test_doctor_nvidia_partial_stack_with_torch_present(monkeypatch):
    """NVIDIA + partial [train] + torch installed drops the index URL (#884)."""
    installed = {
        "torch": "2.6.0",
        "transformers": "5.16.1",
        "peft": "0.20.0",
        "trl": "0.29.0",
        "datasets": "2.14.0",
        "accelerate": "0.27.0",
    }  # bitsandbytes missing, torch present
    monkeypatch.setattr(
        "soup_cli.commands.doctor._installed_version_str",
        lambda import_name, pkg_name: installed.get(pkg_name),
    )
    monkeypatch.setattr(
        "soup_cli.commands.doctor._nvidia_smi_cuda_version", lambda: (13, 0)
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    out = _strip_ansi(result.output)
    assert "Training stack incomplete, missing: bitsandbytes" in out
    assert "index-url" not in out


def test_doctor_missing_core_dependency_exits_nonzero(monkeypatch):
    """A missing core dependency makes `soup doctor` exit non-zero (#828)."""
    monkeypatch.setitem(sys.modules, "plotext", None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code != 0
    assert "MISSING" in result.output


def test_doctor_full_install_exits_zero():
    """Runs against the full dev install: missing [train] stays advisory (#828)."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0


def test_installed_extras_lists_train_and_mcp(monkeypatch):
    """_installed_extras() derives train and mcp from dist metadata (#828)."""
    import importlib.metadata

    class _FakeMeta:
        def get_all(self, key):
            if key == "Provides-Extra":
                return ["train", "mcp", "serve", "data"]
            return None

    requires = [
        "torch>=2.6.0; extra == 'train'",
        "transformers>=5.16.1; extra == 'train'",
        "mcp>=1.10.0; extra == 'mcp'",
        "fastapi>=0.104.0; extra == 'serve'",
        "scikit-learn>=1.3.0; extra == 'data'",
    ]
    present = {"torch", "transformers", "mcp", "fastapi", "scikit-learn"}

    def _distribution(name):
        if name not in present:
            raise importlib.metadata.PackageNotFoundError(name)
        return object()

    monkeypatch.setattr(importlib.metadata, "metadata", lambda name: _FakeMeta())
    monkeypatch.setattr(importlib.metadata, "requires", lambda name: list(requires))
    monkeypatch.setattr(importlib.metadata, "distribution", _distribution)

    from soup_cli.cli import _installed_extras

    extras = _installed_extras()
    assert "train" in extras
    assert "mcp" in extras
    assert "data" in extras


def test_doctor_nccl_no_gpu():
    """--nccl with <2 GPUs prints a skip message."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.distributed.is_available", return_value=True),
        patch(
            "soup_cli.utils.topology.detect_topology",
            return_value={"gpu_count": 1, "nvlink_pairs": 0, "interconnect": "single"},
        ),
    ):
        result = runner.invoke(app, ["doctor", "--nccl"])
        assert result.exit_code == 0
        # _strip_ansi, like the newer tests in this file: under colour Rich
        # splits the message with SGR codes and the substring match fails
        # for a reason unrelated to NCCL (#886). These two simply predate
        # the helper defined at the top of this module.
        assert "NCCL bandwidth requires >=2 GPUs" in _strip_ansi(result.output)


def test_doctor_nccl_mocked_success():
    """--nccl with 2 GPUs runs the check and displays result."""

    # We mock mp.spawn to just set a value in the return_dict instead of actually running processes.
    def mock_spawn(func, args, nprocs, join):
        return_dict = args[0]
        return_dict["gb_per_sec"] = 350.0  # mock value

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.distributed.is_available", return_value=True),
        patch(
            "soup_cli.utils.topology.detect_topology",
            return_value={"gpu_count": 2, "nvlink_pairs": 1, "interconnect": "nvlink"},
        ),
        patch("torch.cuda.get_device_name", return_value="NVIDIA H100 80GB HBM3"),
        patch("torch.multiprocessing.spawn", side_effect=mock_spawn),
    ):
        result = runner.invoke(app, ["doctor", "--nccl"])
        assert result.exit_code == 0
        assert "Measuring NCCL bandwidth" in _strip_ansi(result.output)
        assert "Result (H100 over NVLINK)" in _strip_ansi(result.output)
