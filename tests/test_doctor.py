"""Tests for soup doctor command."""

import sys
from unittest.mock import patch

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.doctor import _version_ok

runner = CliRunner()


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
        assert result.exit_code != 0
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
    assert 'pip install "soup-cli[train]"' in result.output
    assert "torch>=2.6.0" not in result.output
    assert "transformers>=5.16.1" not in result.output
    assert "peft>=0.20.0" not in result.output
    assert "trl>=0.29.0" not in result.output
    assert "datasets>=2.14.0" not in result.output
    assert "bitsandbytes>=0.41.0" not in result.output
    assert "accelerate>=0.27.0" not in result.output


def test_doctor_missing_core_dependency_exits_nonzero(monkeypatch):
    """A missing core dependency makes `soup doctor` exit non-zero (#828)."""
    monkeypatch.setitem(sys.modules, "plotext", None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code != 0
    assert "MISSING" in result.output


def test_doctor_healthy_core_exit_zero():
    """A healthy core-only install still exits 0 despite missing [train] (#828)."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0


def test_installed_extras_lists_train_and_mcp(monkeypatch):
    """_installed_extras() derives train and mcp from dist metadata (#828)."""
    import importlib.metadata
    import importlib.util

    class _FakeMeta:
        def get_all(self, key):
            if key == "Provides-Extra":
                return ["train", "mcp", "serve"]
            return None

    requires = [
        "torch>=2.6.0; extra == 'train'",
        "transformers>=5.16.1; extra == 'train'",
        "mcp>=1.10.0; extra == 'mcp'",
        "fastapi>=0.104.0; extra == 'serve'",
    ]
    present = {"torch", "transformers", "mcp", "fastapi"}

    monkeypatch.setattr(importlib.metadata, "metadata", lambda name: _FakeMeta())
    monkeypatch.setattr(importlib.metadata, "requires", lambda name: list(requires))
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name in present else None
    )

    from soup_cli.cli import _installed_extras

    extras = _installed_extras()
    assert "train" in extras
    assert "mcp" in extras


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
