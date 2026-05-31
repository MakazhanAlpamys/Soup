"""v0.64.0 review fix follow-ups.

Covers wave-1 review-fix gaps:
- HIGH H2/H3/H4 — symlink rejection on the new READ paths
  (plan/apply YAML loader, tunability load_report, env_lock read_lock,
  terraform_plan read_state).
- HIGH H5 — compute_dataset_sha cwd-containment + symlink rejection.
- HIGH H6 — symlink rejection on dataset_path through `soup plan`.
- MEDIUM M1 — compute_config_sha rejects non-JSON-serialisable values.
- MEDIUM M3 — read_state rejects non-bool `applied`.
- MEDIUM M4 — _activation_bytes overflow-clamps at sanity cap.
- MEDIUM M6 — flag_downstream_risk tight Llama-family allowlist.
- MEDIUM M7 — Windows CUDA path parse (vN.M pattern).
- MEDIUM M8 — load_report containment-before-existence ordering.
- LOW  L3  — source-grep regression: atomic_write_text usage in all 3 writers.
- LOW  L4  — license-advisor MAU upper cap.
- LOW  L5  — `Sequence` imported from `collections.abc` not `typing`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# HIGH H2 — plan `_load_yaml_config` rejects symlink on YAML path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_plan_yaml_symlink_rejected(tmp_path, monkeypatch):
    from soup_cli.commands.plan import _load_yaml_config

    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.yaml"
    real.write_text("base: x\n")
    link = tmp_path / "link.yaml"
    os.symlink(real, link)
    with pytest.raises(ValueError, match="symlink"):
        _load_yaml_config(str(link))


def test_plan_yaml_outside_cwd_rejected(tmp_path, monkeypatch):
    from soup_cli.commands.plan import _load_yaml_config

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "evil.yaml"
    outside.write_text("base: x\n")
    with pytest.raises(ValueError, match="cwd"):
        _load_yaml_config(str(outside))


def test_plan_yaml_null_byte_rejected(tmp_path, monkeypatch):
    from soup_cli.commands.plan import _load_yaml_config

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="null"):
        _load_yaml_config("config\x00.yaml")


# ---------------------------------------------------------------------------
# HIGH H3 — tunability load_report rejects symlink + null-byte
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_tunability_load_report_symlink_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.tunability import load_report

    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    os.symlink(real, link)
    with pytest.raises(ValueError, match="symlink"):
        load_report(str(link))


def test_tunability_load_report_null_byte_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.tunability import load_report

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="null"):
        load_report("report\x00.json")


def test_tunability_load_report_containment_before_existence(tmp_path, monkeypatch):
    """Outside-cwd path must raise ValueError(cwd), NOT FileNotFoundError.

    Otherwise an attacker can distinguish "file exists out of cwd" from
    "file missing" via the exception type — MEDIUM M8 review fix.
    """
    from soup_cli.utils.tunability import load_report

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "definitely-missing.json"
    with pytest.raises(ValueError, match="cwd"):
        load_report(str(outside))


# ---------------------------------------------------------------------------
# HIGH H4 — env read_lock rejects symlink + null-byte + ordering
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_env_read_lock_symlink_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.env_lock import read_lock

    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.lock"
    real.write_text("{}")
    link = tmp_path / "link.lock"
    os.symlink(real, link)
    with pytest.raises(ValueError, match="symlink"):
        read_lock(str(link))


def test_env_read_lock_null_byte_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.env_lock import read_lock

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="null"):
        read_lock("lock\x00.lock")


def test_env_read_lock_outside_cwd_before_existence(tmp_path, monkeypatch):
    from soup_cli.utils.env_lock import read_lock

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "missing.lock"
    with pytest.raises(ValueError, match="cwd"):
        read_lock(str(outside))


# ---------------------------------------------------------------------------
# HIGH H4 (terraform read_state) — symlink rejection
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_terraform_read_state_symlink_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import read_state

    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.tfstate"
    real.write_text("{}")
    link = tmp_path / "link.tfstate"
    os.symlink(real, link)
    with pytest.raises(ValueError, match="symlink"):
        read_state(str(link))


def test_terraform_read_state_null_byte_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import read_state

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="null"):
        read_state("state\x00.tfstate")


def test_terraform_read_state_outside_cwd_before_existence(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import read_state

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "missing.tfstate"
    with pytest.raises(ValueError, match="cwd"):
        read_state(str(outside))


# ---------------------------------------------------------------------------
# HIGH H5 — compute_dataset_sha cwd containment + symlink rejection
# ---------------------------------------------------------------------------


def test_compute_dataset_sha_empty_returns_zero():
    from soup_cli.utils.terraform_plan import compute_dataset_sha

    assert compute_dataset_sha("") == "0" * 64


def test_compute_dataset_sha_outside_cwd_returns_zero(tmp_path, monkeypatch):
    """Out-of-cwd path silently returns zero-hash (no file read).

    Defends against `data.train: /etc/shadow` smuggling file contents
    into the SHA — H5 review fix.
    """
    from soup_cli.utils.terraform_plan import compute_dataset_sha

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "evil.jsonl"
    if outside.exists():
        # Don't depend on the parent dir being writable.
        return
    sha = compute_dataset_sha(str(outside))
    assert sha == "0" * 64


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_compute_dataset_sha_symlink_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import compute_dataset_sha

    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.jsonl"
    real.write_text("{}\n")
    link = tmp_path / "link.jsonl"
    os.symlink(real, link)
    with pytest.raises(ValueError, match="symlink"):
        compute_dataset_sha(str(link))


def test_compute_dataset_sha_null_byte_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import compute_dataset_sha

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="null"):
        compute_dataset_sha("data\x00.jsonl")


def test_compute_dataset_sha_missing_returns_zero(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import compute_dataset_sha

    monkeypatch.chdir(tmp_path)
    sha = compute_dataset_sha("missing.jsonl")
    assert sha == "0" * 64


# ---------------------------------------------------------------------------
# MEDIUM M1 — compute_config_sha rejects non-JSON-serialisable values
# ---------------------------------------------------------------------------


def test_compute_config_sha_strict_no_default_str():
    """A non-serialisable value must raise, not silently `str()` it."""
    from soup_cli.utils.terraform_plan import compute_config_sha

    # `set` is not JSON-serialisable
    with pytest.raises(TypeError):
        compute_config_sha({"a": {1, 2, 3}})


# ---------------------------------------------------------------------------
# MEDIUM M3 — read_state rejects non-bool `applied`
# ---------------------------------------------------------------------------


def test_read_state_rejects_non_bool_applied(tmp_path, monkeypatch):
    import json

    from soup_cli.utils.terraform_plan import read_state

    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "bad.tfstate"
    payload = {
        "plan": {
            "base": "m",
            "task": "sft",
            "config_sha": "a" * 64,
            "dataset_sha": "b" * 64,
            "estimated_cost_usd": 0.5,
            "estimated_minutes": 10.0,
            "peak_vram_gb": 8.0,
            "spot_price_usd_per_hour": 0.30,
        },
        "applied": "yes",  # str, not bool — must raise
        "applied_at": None,
        "run_id": None,
    }
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="applied"):
        read_state(str(bad))


# ---------------------------------------------------------------------------
# MEDIUM M4 — _activation_bytes overflow clamp
# ---------------------------------------------------------------------------


def test_activation_bytes_clamped_at_extreme():
    """Even at schema max seq_len * batch_size, result stays finite."""
    import math

    from soup_cli.utils.hardware_fit import HardwareFitInput, estimate_peak_vram_gb

    inp = HardwareFitInput(
        params_b=1000.0,
        seq_len=1_048_576,
        batch_size=1024,
        optimizer="adamw_torch",
        quant="none",
        peft="full",
        gradient_checkpointing=False,
    )
    bd = estimate_peak_vram_gb(inp)
    assert math.isfinite(bd.total_gb)
    assert math.isfinite(bd.activations_gb)


# ---------------------------------------------------------------------------
# MEDIUM M6 — Llama community license allowlist (no .startswith match)
# ---------------------------------------------------------------------------


def test_flag_downstream_risk_unknown_llama_variant_does_not_trip_mau_gate():
    """A hypothetical `llama-permissive-2030` should NOT trigger MAU block.

    Tight allowlist defends against future Meta licence id surprises.
    """
    from soup_cli.utils.license_advisor import flag_downstream_risk

    # `llama-permissive-2030` is not in `_LLAMA_COMMUNITY_LICENSES` AND
    # is not in the known matrix, so it should fall through to the
    # "unknown license -> warn" branch, not "block by MAU".
    r = flag_downstream_risk(
        license_id="llama-permissive-2030",
        target="b2c",
        monthly_active_users=800_000_000,
    )
    # severity might be warn (unknown) but NOT block-by-MAU.
    assert "monthly active users" not in r.reason.lower() or r.severity != "block"


def test_flag_downstream_risk_llama_3_under_cap_is_warn_not_block():
    from soup_cli.utils.license_advisor import flag_downstream_risk

    r = flag_downstream_risk(
        license_id="llama-3",
        target="b2c",
        monthly_active_users=100,
    )
    # Restricted-use on B2C with low MAU = warn, never block.
    assert r.severity != "block"


# ---------------------------------------------------------------------------
# MEDIUM M7 — Windows CUDA path parse (vN.M)
# ---------------------------------------------------------------------------


def test_detect_cuda_version_windows_path(monkeypatch):
    """`v12.1` token at end of a Windows-style path is extracted."""
    from soup_cli.utils import env_lock

    monkeypatch.setenv("CUDA_HOME", r"C:\Program Files\NVIDIA\CUDA\v12.1")
    # Ensure no other env var pre-empts
    monkeypatch.delenv("CUDA_VERSION", raising=False)
    # Sidestep torch — drop it from sys.modules
    monkeypatch.setattr(env_lock.sys, "modules", {**env_lock.sys.modules, "torch": None})
    version = env_lock._detect_cuda_version()
    assert version is not None
    assert "12.1" in version


def test_detect_cuda_version_posix_path(monkeypatch):
    from soup_cli.utils import env_lock

    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-12.1")
    monkeypatch.delenv("CUDA_VERSION", raising=False)
    monkeypatch.setattr(env_lock.sys, "modules", {**env_lock.sys.modules, "torch": None})
    version = env_lock._detect_cuda_version()
    assert version is not None
    assert "12.1" in version


def test_detect_cuda_version_none_when_no_env(monkeypatch):
    from soup_cli.utils import env_lock

    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_VERSION", raising=False)
    monkeypatch.setattr(env_lock.sys, "modules", {**env_lock.sys.modules, "torch": None})
    # Should silently return None — no env, no torch.
    assert env_lock._detect_cuda_version() is None


# ---------------------------------------------------------------------------
# LOW L3 — source-grep regression: atomic_write_text in all 3 writers
# ---------------------------------------------------------------------------


def test_tunability_uses_atomic_write_text():
    src = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "utils" / "tunability.py"
    text = src.read_text(encoding="utf-8")
    assert "atomic_write_text" in text, "tunability.write_report must use shared helper"


def test_terraform_plan_uses_atomic_write_text():
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "soup_cli" / "utils" / "terraform_plan.py"
    )
    text = src.read_text(encoding="utf-8")
    assert "atomic_write_text" in text, "terraform_plan.write_state must use shared helper"


def test_env_lock_uses_atomic_write_text():
    src = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "utils" / "env_lock.py"
    text = src.read_text(encoding="utf-8")
    assert "atomic_write_text" in text, "env_lock.write_lock must use shared helper"


# ---------------------------------------------------------------------------
# LOW L4 — MAU upper cap
# ---------------------------------------------------------------------------


def test_mau_oversize_rejected():
    from soup_cli.utils.license_advisor import flag_downstream_risk

    with pytest.raises(ValueError, match="monthly_active_users"):
        flag_downstream_risk(
            license_id="apache-2.0",
            target="b2c",
            monthly_active_users=10**18,
        )


# ---------------------------------------------------------------------------
# LOW L5 — Sequence imported from collections.abc, not typing
# ---------------------------------------------------------------------------


def test_tunability_sequence_imported_from_collections_abc():
    """`typing.Sequence` doesn't work with isinstance on every Py 3.9 build.

    Use `collections.abc.Sequence` for runtime isinstance checks.
    """
    src = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "utils" / "tunability.py"
    text = src.read_text(encoding="utf-8")
    # Either explicit `from collections.abc import Sequence`, or NO
    # mention of `typing.Sequence` in an isinstance context.
    assert "from collections.abc import Sequence" in text, (
        "tunability.py must import Sequence from collections.abc"
    )


# ---------------------------------------------------------------------------
# Drift-on-modify regression — `soup plan` -> mutate -> `soup apply`
# ---------------------------------------------------------------------------


def test_apply_refuses_drift_end_to_end(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "soup.yaml"
    cfg.write_text(
        "base: meta-llama/Llama-3.2-1B\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data.jsonl\n"
        "training:\n"
        "  epochs: 1\n"
        "  lr: 0.00005\n"
        "  batch_size: 4\n"
    )
    (tmp_path / "data.jsonl").write_text("{}\n")
    r = runner.invoke(app, ["plan", "--config", str(cfg)])
    assert r.exit_code == 0, r.output
    cfg.write_text(cfg.read_text().replace("epochs: 1", "epochs: 5"))
    r2 = runner.invoke(app, ["apply", "--config", str(cfg), "--dry-run"])
    assert r2.exit_code == 3, r2.output  # exit 3 = drift refused
    assert "drift" in r2.output.lower()
