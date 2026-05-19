"""Tests for v0.60.0 Part E — License-conflict matrix at adapter merge.

Coverage:
- ``LicenseCompatibilityMatrix`` closed allowlist + immutability
- ``check_license_compat`` happy + conflict matrix
- ``LicenseConflictReport`` frozen + reason explainability
- ``--license-override <reason>`` opt-in trail
- Integration: ``adapter_merge`` refuses on conflict (helper used; CLI test
  smokes the flag wiring)
"""

from __future__ import annotations

import dataclasses
import types

import pytest


class TestLicenseMatrix:
    def test_imports(self):
        from soup_cli.utils.license_matrix import (
            KNOWN_LICENSES,
            LICENSE_MATRIX,
            LicenseConflictReport,
            check_license_compat,
            normalise_license_id,
        )
        assert callable(check_license_compat)
        assert callable(normalise_license_id)
        assert isinstance(KNOWN_LICENSES, frozenset)
        assert isinstance(LICENSE_MATRIX, types.MappingProxyType)
        assert dataclasses.is_dataclass(LicenseConflictReport)

    def test_known_licenses_includes_common(self):
        from soup_cli.utils.license_matrix import KNOWN_LICENSES

        for lic in (
            "apache-2.0", "mit", "bsd-3-clause",
            "llama-3", "llama-3.1", "llama-community",
            "gemma", "qwen-research",
            "gpl-3.0", "agpl-3.0",
            "cc-by-4.0", "cc-by-nc-4.0",
            "openrail", "creativeml-openrail-m",
            "openai-tos",
        ):
            assert lic in KNOWN_LICENSES, f"missing license: {lic}"

    def test_matrix_immutable(self):
        from soup_cli.utils.license_matrix import LICENSE_MATRIX

        with pytest.raises(TypeError):
            LICENSE_MATRIX["evil"] = ("anything",)  # type: ignore[index]

    def test_normalise_license_id(self):
        from soup_cli.utils.license_matrix import normalise_license_id

        assert normalise_license_id("Apache-2.0") == "apache-2.0"
        assert normalise_license_id("MIT") == "mit"
        assert normalise_license_id("  llama-3  ") == "llama-3"

    def test_normalise_rejects_null_byte(self):
        from soup_cli.utils.license_matrix import normalise_license_id

        with pytest.raises(ValueError):
            normalise_license_id("apache\x00")

    def test_normalise_rejects_bool(self):
        from soup_cli.utils.license_matrix import normalise_license_id

        with pytest.raises(TypeError):
            normalise_license_id(True)


class TestCheckCompat:
    def test_clean_apache_only_passes(self):
        from soup_cli.utils.license_matrix import check_license_compat

        report = check_license_compat(["apache-2.0", "apache-2.0"])
        assert report.ok is True
        assert report.conflict_pair is None

    def test_apache_with_mit_passes(self):
        from soup_cli.utils.license_matrix import check_license_compat

        report = check_license_compat(["apache-2.0", "mit"])
        assert report.ok is True

    def test_apache_with_gpl_conflict(self):
        from soup_cli.utils.license_matrix import check_license_compat

        # Locked verdict: GPL-3.0 (strong copyleft) alongside Apache-2.0
        # (permissive) is a refused conflict in the v0.60.0 matrix. A
        # future relax requires both a matrix change and this test
        # change in the same PR — drift guard.
        report = check_license_compat(["apache-2.0", "gpl-3.0"])
        assert report.ok is False
        assert report.conflict_pair is not None
        assert "copyleft" in report.reason.lower()

    def test_nc_license_with_apache_conflict(self):
        from soup_cli.utils.license_matrix import check_license_compat

        # NC (non-commercial) cannot be combined with any permissive license
        # that doesn't carry the same restriction.
        report = check_license_compat(["apache-2.0", "cc-by-nc-4.0"])
        assert report.ok is False
        assert "nc" in report.reason.lower() or "non-commercial" in report.reason.lower()

    def test_unknown_license_treated_as_conflict_or_warn(self):
        from soup_cli.utils.license_matrix import check_license_compat

        report = check_license_compat(["apache-2.0", "weird-unknown-license-xyz"])
        assert report.ok is False
        assert "unknown" in report.reason.lower()

    def test_empty_list_rejected(self):
        from soup_cli.utils.license_matrix import check_license_compat

        with pytest.raises(ValueError):
            check_license_compat([])

    def test_single_license_passes(self):
        from soup_cli.utils.license_matrix import check_license_compat

        report = check_license_compat(["apache-2.0"])
        assert report.ok is True

    def test_rejects_non_list(self):
        from soup_cli.utils.license_matrix import check_license_compat

        with pytest.raises(TypeError):
            check_license_compat("apache-2.0")  # type: ignore[arg-type]

    def test_llama_with_apache_conflict(self):
        from soup_cli.utils.license_matrix import check_license_compat

        # Locked verdict: Llama community license adds acceptable-use
        # restrictions that Apache does not satisfy. v0.60.0 design is
        # strict — refuse and force operator override.
        report = check_license_compat(["llama-3", "apache-2.0"])
        assert report.ok is False
        assert report.conflict_pair is not None
        assert "restricted" in report.reason.lower() or "acceptable-use" in report.reason.lower()

    def test_case_insensitive(self):
        from soup_cli.utils.license_matrix import check_license_compat

        report = check_license_compat(["Apache-2.0", "MIT"])
        assert report.ok is True

    def test_report_frozen(self):
        from soup_cli.utils.license_matrix import check_license_compat

        report = check_license_compat(["apache-2.0"])
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.ok = False  # type: ignore[misc]


class TestOverride:
    def test_override_reason_required(self):
        from soup_cli.utils.license_matrix import (
            validate_license_override_reason,
        )
        # Empty reason rejected.
        with pytest.raises(ValueError):
            validate_license_override_reason("")
        with pytest.raises(TypeError):
            validate_license_override_reason(None)  # type: ignore[arg-type]

    def test_override_reason_min_length(self):
        from soup_cli.utils.license_matrix import (
            validate_license_override_reason,
        )
        # A reason of 1-2 chars is too short — defends against `--license-override y`.
        with pytest.raises(ValueError):
            validate_license_override_reason("ok")

    def test_override_reason_oversize_rejected(self):
        from soup_cli.utils.license_matrix import (
            validate_license_override_reason,
        )
        # 4kb cap to prevent log bloat.
        with pytest.raises(ValueError):
            validate_license_override_reason("x" * 5000)

    def test_override_reason_exact_boundary(self):
        """Exact-boundary test at 4096 / 4097 chars."""
        from soup_cli.utils.license_matrix import (
            validate_license_override_reason,
        )
        # Accepted: exactly 4096 chars.
        result = validate_license_override_reason("x" * 4096)
        assert len(result) == 4096
        # Rejected: 4097.
        with pytest.raises(ValueError):
            validate_license_override_reason("x" * 4097)

    def test_override_reason_min_boundary(self):
        """Exact-boundary test at 7 / 8 chars (min cap)."""
        from soup_cli.utils.license_matrix import (
            validate_license_override_reason,
        )
        # Rejected: 7 chars.
        with pytest.raises(ValueError):
            validate_license_override_reason("x" * 7)
        # Accepted: 8 chars.
        result = validate_license_override_reason("x" * 8)
        assert result == "x" * 8

    def test_override_reason_null_byte(self):
        from soup_cli.utils.license_matrix import (
            validate_license_override_reason,
        )
        with pytest.raises(ValueError):
            validate_license_override_reason("legal cleared\x00")

    def test_override_reason_happy(self):
        from soup_cli.utils.license_matrix import (
            validate_license_override_reason,
        )
        result = validate_license_override_reason(
            "legal-cleared by alice@example.com on 2026-05-19"
        )
        assert result.startswith("legal-cleared")


class TestMergeIntegration:
    """Regression: license gate is wired into `soup adapters merge`."""

    def test_merge_help_lists_license_flags(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["adapters", "merge", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--license" in result.output
        assert "--license-override" in result.output

    def test_merge_refuses_license_conflict_without_override(self, tmp_path, monkeypatch):
        """Two adapters with incompatible licenses + no override → exit 3."""
        from typer.testing import CliRunner

        from soup_cli.cli import app
        monkeypatch.chdir(tmp_path)
        # Create two minimal adapter dirs — the license gate fires before
        # the actual merge math, so we don't need real safetensors.
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            (d / "adapter_config.json").write_text("{}", encoding="utf-8")
            (d / "adapter_model.safetensors").write_bytes(b"x")
        runner = CliRunner()
        result = runner.invoke(
            app, [
                "adapters", "merge", "a", "b",
                "-o", "out",
                "--license", "apache-2.0",
                "--license", "cc-by-nc-4.0",
            ]
        )
        assert result.exit_code == 3, (result.output, repr(result.exception))
        assert "license conflict" in result.output.lower()

    def test_merge_override_accepted_with_reason(self, tmp_path, monkeypatch):
        """Conflict + valid --license-override reason → merge proceeds.

        The merge math will fail downstream because the fixture adapters
        are not real safetensors; we assert the license gate let us
        through (exit != 3 — gate-pass, even if merge math fails later).
        """
        from typer.testing import CliRunner

        from soup_cli.cli import app
        monkeypatch.chdir(tmp_path)
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            (d / "adapter_config.json").write_text("{}", encoding="utf-8")
            (d / "adapter_model.safetensors").write_bytes(b"x")
        runner = CliRunner()
        result = runner.invoke(
            app, [
                "adapters", "merge", "a", "b",
                "-o", "out",
                "--license", "apache-2.0",
                "--license", "cc-by-nc-4.0",
                "--license-override", "legal-cleared 2026-05-19 by alice",
            ]
        )
        # Gate passed (no exit 3); downstream merge math likely fails on
        # the synthetic safetensors fixture.
        assert result.exit_code != 3, (result.output, repr(result.exception))
        # Override message surfaced (either the explicit "overridden"
        # banner, or absence of the unmitigated "License conflict refused"
        # banner shows the gate did pass).
        out_lower = result.output.lower()
        assert ("overridden" in out_lower) or ("license conflict refused" not in out_lower)

    def test_merge_override_reason_too_short_rejected(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        monkeypatch.chdir(tmp_path)
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            (d / "adapter_config.json").write_text("{}", encoding="utf-8")
            (d / "adapter_model.safetensors").write_bytes(b"x")
        runner = CliRunner()
        result = runner.invoke(
            app, [
                "adapters", "merge", "a", "b",
                "-o", "out",
                "--license", "apache-2.0",
                "--license", "cc-by-nc-4.0",
                "--license-override", "ok",  # < 8 chars
            ]
        )
        assert result.exit_code == 2
        assert "short" in result.output.lower() or "8" in result.output

    def test_merge_license_count_mismatch_rejected(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        monkeypatch.chdir(tmp_path)
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            (d / "adapter_config.json").write_text("{}", encoding="utf-8")
            (d / "adapter_model.safetensors").write_bytes(b"x")
        runner = CliRunner()
        # Only one license for two adapters — mismatch.
        result = runner.invoke(
            app, [
                "adapters", "merge", "a", "b",
                "-o", "out",
                "--license", "apache-2.0",
            ]
        )
        assert result.exit_code == 2
        assert "must match" in result.output.lower() or "count" in result.output.lower()


class TestSourceWiring:
    def test_module_imports(self):
        from soup_cli.utils import license_matrix as m

        assert hasattr(m, "check_license_compat")
        assert hasattr(m, "KNOWN_LICENSES")
        assert hasattr(m, "LICENSE_MATRIX")

    def test_matrix_has_at_least_20_entries(self):
        from soup_cli.utils.license_matrix import KNOWN_LICENSES

        # Plan says top-30 licenses — at least 20 must be cataloged.
        assert len(KNOWN_LICENSES) >= 20
