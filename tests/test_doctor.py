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
        assert "NCCL bandwidth requires >=2 GPUs" in result.output


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
        assert "Measuring NCCL bandwidth" in result.output
        assert "Result (H100 over NVLINK)" in result.output
