"""v0.64.0 Part F — License advisor tests."""

from __future__ import annotations

import dataclasses

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def test_module_imports():
    from soup_cli.utils import license_advisor

    assert hasattr(license_advisor, "DEPLOY_TARGETS")
    assert hasattr(license_advisor, "validate_deploy_target")
    assert hasattr(license_advisor, "LicenseRecommendation")
    assert hasattr(license_advisor, "advise_license_for_target")
    assert hasattr(license_advisor, "flag_downstream_risk")
    assert hasattr(license_advisor, "DownstreamRisk")


# ---------------------------------------------------------------------------
# DEPLOY_TARGETS
# ---------------------------------------------------------------------------


def test_deploy_targets_exact():
    from soup_cli.utils.license_advisor import DEPLOY_TARGETS

    assert "b2c" in DEPLOY_TARGETS
    assert "defense" in DEPLOY_TARGETS
    assert "embedded" in DEPLOY_TARGETS


def test_deploy_targets_is_frozenset():
    from soup_cli.utils.license_advisor import DEPLOY_TARGETS

    assert isinstance(DEPLOY_TARGETS, frozenset)


# ---------------------------------------------------------------------------
# validate_deploy_target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", ["b2c", "defense", "embedded"])
def test_validate_deploy_target_happy(v):
    from soup_cli.utils.license_advisor import validate_deploy_target

    assert validate_deploy_target(v) == v


def test_validate_deploy_target_case_insensitive():
    from soup_cli.utils.license_advisor import validate_deploy_target

    assert validate_deploy_target("B2C") == "b2c"


@pytest.mark.parametrize(
    "bad",
    [True, False, None, "", "consumer", "x" * 33],
)
def test_validate_deploy_target_rejects(bad):
    from soup_cli.utils.license_advisor import validate_deploy_target

    with pytest.raises((TypeError, ValueError)):
        validate_deploy_target(bad)


def test_validate_deploy_target_null_byte():
    from soup_cli.utils.license_advisor import validate_deploy_target

    with pytest.raises(ValueError, match="null"):
        validate_deploy_target("b\x002c")


# ---------------------------------------------------------------------------
# LicenseRecommendation
# ---------------------------------------------------------------------------


def test_license_recommendation_frozen():
    from soup_cli.utils.license_advisor import LicenseRecommendation

    r = LicenseRecommendation(
        target="b2c",
        recommended_licenses=("apache-2.0", "mit"),
        forbidden_licenses=("cc-by-nc-4.0",),
        reason="permissive only",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.target = "other"  # type: ignore[misc]


def test_license_recommendation_tuples():
    from soup_cli.utils.license_advisor import LicenseRecommendation

    # Lists rejected (frozen=True doesn't make lists immutable)
    with pytest.raises(TypeError, match="tuple"):
        LicenseRecommendation(
            target="b2c",
            recommended_licenses=["apache-2.0"],  # type: ignore[arg-type]
            forbidden_licenses=(),
            reason="x",
        )


# ---------------------------------------------------------------------------
# advise_license_for_target
# ---------------------------------------------------------------------------


def test_advise_license_b2c():
    from soup_cli.utils.license_advisor import advise_license_for_target

    rec = advise_license_for_target("b2c")
    # B2C must include broadly-permissive options
    assert "apache-2.0" in rec.recommended_licenses
    assert "mit" in rec.recommended_licenses
    # Non-commercial licenses must be forbidden for B2C
    assert "cc-by-nc-4.0" in rec.forbidden_licenses


def test_advise_license_defense():
    from soup_cli.utils.license_advisor import advise_license_for_target

    rec = advise_license_for_target("defense")
    # Defense forbids restricted-use community licenses
    assert "llama-3.1" in rec.forbidden_licenses or "llama-2" in rec.forbidden_licenses


def test_advise_license_embedded():
    from soup_cli.utils.license_advisor import advise_license_for_target

    rec = advise_license_for_target("embedded")
    # Embedded systems should avoid strong copyleft
    assert "gpl-3.0" in rec.forbidden_licenses


def test_advise_license_rejects_unknown_target():
    from soup_cli.utils.license_advisor import advise_license_for_target

    with pytest.raises(ValueError):
        advise_license_for_target("bogus")


def test_advise_license_rejects_bool():
    from soup_cli.utils.license_advisor import advise_license_for_target

    with pytest.raises(TypeError):
        advise_license_for_target(True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DownstreamRisk
# ---------------------------------------------------------------------------


def test_downstream_risk_frozen():
    from soup_cli.utils.license_advisor import DownstreamRisk

    r = DownstreamRisk(ok=True, severity="ok", reason="permissive license")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.ok = False  # type: ignore[misc]


def test_downstream_risk_severity_allowlist():
    from soup_cli.utils.license_advisor import DownstreamRisk

    with pytest.raises(ValueError, match="severity"):
        DownstreamRisk(ok=False, severity="weird", reason="x")


# ---------------------------------------------------------------------------
# flag_downstream_risk
# ---------------------------------------------------------------------------


def test_flag_downstream_risk_apache_b2c_ok():
    from soup_cli.utils.license_advisor import flag_downstream_risk

    r = flag_downstream_risk(license_id="apache-2.0", target="b2c", monthly_active_users=10_000)
    assert r.ok is True
    assert r.severity == "ok"


def test_flag_downstream_risk_llama_community_high_mau():
    """Llama community license + commercial use > 700M MAU is the canonical risk."""
    from soup_cli.utils.license_advisor import flag_downstream_risk

    r = flag_downstream_risk(
        license_id="llama-3", target="b2c", monthly_active_users=800_000_000
    )
    assert r.ok is False
    assert r.severity in ("warn", "block")


def test_flag_downstream_risk_llama_community_low_mau_warns():
    """Llama community on small B2C: warning, not block."""
    from soup_cli.utils.license_advisor import flag_downstream_risk

    r = flag_downstream_risk(
        license_id="llama-3", target="b2c", monthly_active_users=10_000
    )
    # Either OK with note or a warn — not a hard block
    assert r.severity in ("ok", "warn")


def test_flag_downstream_risk_nc_b2c_blocked():
    from soup_cli.utils.license_advisor import flag_downstream_risk

    r = flag_downstream_risk(
        license_id="cc-by-nc-4.0", target="b2c", monthly_active_users=100
    )
    assert r.ok is False


def test_flag_downstream_risk_rejects_negative_mau():
    from soup_cli.utils.license_advisor import flag_downstream_risk

    with pytest.raises(ValueError, match="monthly_active_users"):
        flag_downstream_risk(
            license_id="apache-2.0", target="b2c", monthly_active_users=-1
        )


def test_flag_downstream_risk_rejects_bool_mau():
    from soup_cli.utils.license_advisor import flag_downstream_risk

    with pytest.raises(TypeError, match="bool"):
        flag_downstream_risk(
            license_id="apache-2.0", target="b2c",
            monthly_active_users=True,  # type: ignore[arg-type]
        )


def test_flag_downstream_risk_unknown_license_warns():
    from soup_cli.utils.license_advisor import flag_downstream_risk

    r = flag_downstream_risk(
        license_id="unknown-soup-test-license", target="b2c", monthly_active_users=1
    )
    # Unknown license should not block silently — surface as a warn
    assert r.severity in ("warn", "block")


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_license_advisor_help():
    from soup_cli.cli import app

    result = runner.invoke(app, ["license-advisor", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_cli_license_advisor_b2c():
    from soup_cli.cli import app

    result = runner.invoke(app, ["license-advisor", "--target", "b2c"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_cli_license_advisor_unknown_target():
    from soup_cli.cli import app

    result = runner.invoke(app, ["license-advisor", "--target", "bogus"])
    assert result.exit_code != 0


def test_cli_license_advisor_check_license():
    """--license is the per-license-id risk-check mode."""
    from soup_cli.cli import app

    result = runner.invoke(
        app,
        [
            "license-advisor",
            "--target",
            "b2c",
            "--license",
            "apache-2.0",
            "--monthly-active-users",
            "10000",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_cli_license_advisor_check_llama_block():
    from soup_cli.cli import app

    result = runner.invoke(
        app,
        [
            "license-advisor",
            "--target",
            "b2c",
            "--license",
            "llama-3",
            "--monthly-active-users",
            "800000000",
        ],
    )
    # Either exits non-zero on block, OR prints a block message
    assert result.exit_code != 0 or "block" in result.output.lower()


# ---------------------------------------------------------------------------
# Source-wiring regression
# ---------------------------------------------------------------------------


def test_no_heavy_top_level_imports():
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "soup_cli" / "utils" / "license_advisor.py"
    )
    text = src.read_text(encoding="utf-8")
    import re
    for bad in ["^import torch", "^from torch", "^import transformers", "^from transformers"]:
        assert not re.search(bad, text, re.MULTILINE)
