"""Tests for v0.59.0 — Governance & Provenance.

Coverage:
- Part A: BOM emit (CycloneDX 1.6 + ML-BOM + SPDX 2.3 + AI profile)
- Part B: Attestation (in-toto Statement + SLSA-3 provenance shape)
- Part C: Annex XI/XII auto-doc rendering + FLOPs/CO2 capture
- Part D: Audit log JSONL append + redaction + rotation
- Part E: Reproducibility receipt (seeds + kernel versions + GPU + OS)
- Part F: CO2 + energy capture stubs + electricityMap SSRF policy
- CLI: bom emit / attest / audit-log / train --annex-xi / train --repro-receipt
- Source-grep wiring guards
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli
from soup_cli.cli import app

# ---------- Part A: BOM ----------


class TestBomSpec:
    def test_imports(self):
        from soup_cli.utils.bom import (
            BomEntry,
            build_cyclonedx_bom,
            build_spdx_bom,
            render_bom,
            write_bom,
        )
        assert callable(build_cyclonedx_bom)
        assert callable(build_spdx_bom)
        assert callable(render_bom)
        assert callable(write_bom)
        assert dataclasses.is_dataclass(BomEntry)

    def test_bom_entry_frozen(self):
        from soup_cli.utils.bom import BomEntry

        entry = BomEntry(
            name="adapter-v1",
            version="0.1.0",
            base_model="meta-llama/Llama-3.1-8B",
            base_sha="0" * 64,
            config_sha="1" * 64,
            data_sha="2" * 64,
            task="sft",
            license="apache-2.0",
            parents=(),
            artifacts=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        assert entry.name == "adapter-v1"
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.name = "evil"  # type: ignore[misc]

    def test_bom_entry_validation_rejects_null_byte(self):
        from soup_cli.utils.bom import BomEntry

        with pytest.raises(ValueError):
            BomEntry(
                name="adapter\x00",
                version="0.1.0",
                base_model="m",
                base_sha="0" * 64,
                config_sha="1" * 64,
                data_sha=None,
                task="sft",
                license=None,
                parents=(),
                artifacts=(),
                created_at="2026-05-18T12:00:00+00:00",
            )

    def test_bom_entry_validation_rejects_bad_sha(self):
        from soup_cli.utils.bom import BomEntry

        # too short
        with pytest.raises(ValueError):
            BomEntry(
                name="a",
                version="0.1.0",
                base_model="m",
                base_sha="00",
                config_sha="1" * 64,
                data_sha=None,
                task="sft",
                license=None,
                parents=(),
                artifacts=(),
                created_at="2026-05-18T12:00:00+00:00",
            )

    def test_cyclonedx_shape(self):
        from soup_cli.utils.bom import BomEntry, build_cyclonedx_bom

        entry = BomEntry(
            name="adapter-v1",
            version="0.1.0",
            base_model="meta-llama/Llama-3.1-8B",
            base_sha="a" * 64,
            config_sha="b" * 64,
            data_sha="c" * 64,
            task="sft",
            license="apache-2.0",
            parents=(),
            artifacts=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        doc = build_cyclonedx_bom(entry)
        assert doc["bomFormat"] == "CycloneDX"
        assert doc["specVersion"] == "1.6"
        assert doc["serialNumber"].startswith("urn:uuid:")
        assert doc["metadata"]["component"]["name"] == "adapter-v1"
        assert doc["metadata"]["component"]["version"] == "0.1.0"
        # ML-BOM: component should have type machine-learning-model
        assert doc["metadata"]["component"]["type"] == "machine-learning-model"
        # Should include base model component
        comps = doc.get("components", [])
        assert any(c["name"] == "meta-llama/Llama-3.1-8B" for c in comps)

    def test_cyclonedx_license_chain(self):
        from soup_cli.utils.bom import BomEntry, build_cyclonedx_bom

        entry = BomEntry(
            name="adapter-v1",
            version="0.1.0",
            base_model="meta-llama/Llama-3.1-8B",
            base_sha="a" * 64,
            config_sha="b" * 64,
            data_sha="c" * 64,
            task="sft",
            license="apache-2.0",
            parents=(),
            artifacts=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        doc = build_cyclonedx_bom(entry)
        license_field = doc["metadata"]["component"].get("licenses", [])
        assert len(license_field) >= 1
        assert license_field[0]["license"]["id"].lower() == "apache-2.0"

    def test_spdx_shape(self):
        from soup_cli.utils.bom import BomEntry, build_spdx_bom

        entry = BomEntry(
            name="adapter-v1",
            version="0.1.0",
            base_model="meta-llama/Llama-3.1-8B",
            base_sha="a" * 64,
            config_sha="b" * 64,
            data_sha="c" * 64,
            task="sft",
            license="apache-2.0",
            parents=(),
            artifacts=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        doc = build_spdx_bom(entry)
        assert doc["spdxVersion"] == "SPDX-2.3"
        assert doc["dataLicense"] == "CC0-1.0"
        assert doc["name"] == "adapter-v1"
        assert doc["SPDXID"] == "SPDXRef-DOCUMENT"
        # AI profile annotation
        pkgs = doc.get("packages", [])
        assert any(p.get("primaryPackagePurpose") == "AI-MODEL" for p in pkgs)

    def test_render_bom_format_dispatch(self):
        from soup_cli.utils.bom import BomEntry, render_bom

        entry = BomEntry(
            name="adapter",
            version="0.1.0",
            base_model="m",
            base_sha="a" * 64,
            config_sha="b" * 64,
            data_sha=None,
            task="sft",
            license=None,
            parents=(),
            artifacts=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        cy = render_bom(entry, "cyclonedx")
        assert "CycloneDX" in cy
        sp = render_bom(entry, "spdx")
        assert "SPDX-2.3" in sp
        with pytest.raises(ValueError):
            render_bom(entry, "xml")

    def test_write_bom_atomic_and_containment(self, tmp_path, monkeypatch):
        from soup_cli.utils.bom import BomEntry, write_bom

        monkeypatch.chdir(tmp_path)
        entry = BomEntry(
            name="adapter",
            version="0.1.0",
            base_model="m",
            base_sha="a" * 64,
            config_sha="b" * 64,
            data_sha=None,
            task="sft",
            license=None,
            parents=(),
            artifacts=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        out = tmp_path / "bom.json"
        write_bom(entry, "cyclonedx", str(out))
        assert out.is_file()
        data = json.loads(out.read_text())
        assert data["bomFormat"] == "CycloneDX"

    def test_write_bom_rejects_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.utils.bom import BomEntry, write_bom

        monkeypatch.chdir(tmp_path)
        entry = BomEntry(
            name="adapter",
            version="0.1.0",
            base_model="m",
            base_sha="a" * 64,
            config_sha="b" * 64,
            data_sha=None,
            task="sft",
            license=None,
            parents=(),
            artifacts=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        with pytest.raises(ValueError):
            write_bom(entry, "cyclonedx", "/tmp/evil/bom.json")

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink rejection")
    def test_write_bom_rejects_symlink_target(self, tmp_path, monkeypatch):
        from soup_cli.utils.bom import BomEntry, write_bom

        monkeypatch.chdir(tmp_path)
        entry = BomEntry(
            name="adapter",
            version="0.1.0",
            base_model="m",
            base_sha="a" * 64,
            config_sha="b" * 64,
            data_sha=None,
            task="sft",
            license=None,
            parents=(),
            artifacts=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        target = tmp_path / "bom.json"
        os.symlink("/etc/passwd", str(target))
        with pytest.raises(ValueError):
            write_bom(entry, "cyclonedx", str(target))

    def test_bom_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["bom", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "emit" in result.output.lower() or "bom" in result.output.lower()

    def test_bom_emit_cli_smoke(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "bom.json"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "bom", "emit",
                "--name", "adapter-v1",
                "--version", "0.1.0",
                "--base-model", "meta-llama/Llama-3.1-8B",
                "--base-sha", "a" * 64,
                "--config-sha", "b" * 64,
                "--task", "sft",
                "--license", "apache-2.0",
                "--format", "cyclonedx",
                "--output", str(out),
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert out.is_file()
        data = json.loads(out.read_text())
        assert data["bomFormat"] == "CycloneDX"


# ---------- Part B: Attestation ----------


class TestAttestation:
    def test_imports(self):
        from soup_cli.utils.attest import (
            AttestationStatement,
            build_in_toto_statement,
            build_slsa_provenance,
            write_attestation,
        )
        assert callable(build_in_toto_statement)
        assert callable(build_slsa_provenance)
        assert callable(write_attestation)
        assert dataclasses.is_dataclass(AttestationStatement)

    def test_attestation_statement_frozen(self):
        from soup_cli.utils.attest import AttestationStatement

        s = AttestationStatement(
            stage="train",
            subject_name="adapter-v1",
            subject_sha256="a" * 64,
            builder_id="soup-cli@0.59.0",
            invocation={"command": "soup train"},
            materials=({"uri": "hf://meta-llama/Llama-3.1-8B", "digest": "b" * 64},),
            created_at="2026-05-18T12:00:00+00:00",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.stage = "evil"  # type: ignore[misc]

    def test_attestation_stage_allowlist(self):
        from soup_cli.utils.attest import AttestationStatement

        with pytest.raises(ValueError):
            AttestationStatement(
                stage="evil",
                subject_name="a",
                subject_sha256="0" * 64,
                builder_id="b",
                invocation={},
                materials=(),
                created_at="2026-05-18T12:00:00+00:00",
            )

    def test_in_toto_shape(self):
        from soup_cli.utils.attest import (
            AttestationStatement,
            build_in_toto_statement,
        )

        s = AttestationStatement(
            stage="train",
            subject_name="adapter-v1",
            subject_sha256="a" * 64,
            builder_id="soup-cli@0.59.0",
            invocation={"command": "soup train"},
            materials=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        doc = build_in_toto_statement(s)
        assert doc["_type"] == "https://in-toto.io/Statement/v1"
        assert doc["predicateType"].startswith("https://slsa.dev/provenance/v1")
        subjects = doc["subject"]
        assert subjects[0]["name"] == "adapter-v1"
        assert subjects[0]["digest"]["sha256"] == "a" * 64

    def test_slsa_provenance_shape(self):
        from soup_cli.utils.attest import (
            AttestationStatement,
            build_slsa_provenance,
        )

        s = AttestationStatement(
            stage="train",
            subject_name="adapter-v1",
            subject_sha256="a" * 64,
            builder_id="soup-cli@0.59.0",
            invocation={"command": "soup train"},
            materials=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        prov = build_slsa_provenance(s)
        assert prov["buildDefinition"]["buildType"].endswith("/build/v1")
        assert prov["runDetails"]["builder"]["id"] == "soup-cli@0.59.0"

    def test_subject_sha_must_be_64_hex(self):
        from soup_cli.utils.attest import AttestationStatement

        with pytest.raises(ValueError):
            AttestationStatement(
                stage="train",
                subject_name="a",
                subject_sha256="not-hex",
                builder_id="b",
                invocation={},
                materials=(),
                created_at="2026-05-18T12:00:00+00:00",
            )

    def test_attestation_write_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.attest import (
            AttestationStatement,
            write_attestation,
        )

        monkeypatch.chdir(tmp_path)
        s = AttestationStatement(
            stage="train",
            subject_name="adapter-v1",
            subject_sha256="a" * 64,
            builder_id="soup-cli@0.59.0",
            invocation={},
            materials=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        with pytest.raises(ValueError):
            write_attestation(s, "/tmp/x.json")

    def test_attestation_write_atomic(self, tmp_path, monkeypatch):
        from soup_cli.utils.attest import (
            AttestationStatement,
            write_attestation,
        )

        monkeypatch.chdir(tmp_path)
        s = AttestationStatement(
            stage="train",
            subject_name="adapter-v1",
            subject_sha256="a" * 64,
            builder_id="soup-cli@0.59.0",
            invocation={},
            materials=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        out = tmp_path / "att.json"
        write_attestation(s, str(out))
        assert out.is_file()
        data = json.loads(out.read_text())
        assert data["_type"] == "https://in-toto.io/Statement/v1"

    def test_sign_attestation_stub_signature(self):
        """Sigstore signing deferred to v0.59.1 — ed25519 fallback returns marker."""
        from soup_cli.utils.attest import SignatureBackend, sign_attestation

        out = sign_attestation(b"payload", backend=SignatureBackend.UNSIGNED)
        assert out["signature"] == ""
        assert out["backend"] == "unsigned"

    def test_sign_attestation_rejects_unknown_backend(self):
        from soup_cli.utils.attest import sign_attestation

        with pytest.raises(ValueError):
            sign_attestation(b"payload", backend="weird")  # type: ignore[arg-type]

    def test_attest_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["attest", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))


# ---------- Part C: Annex XI/XII ----------


class TestAnnexXI:
    def test_imports(self):
        from soup_cli.utils.annex_xi import (
            AnnexXIData,
            render_annex_xi_markdown,
            render_annex_xii_markdown,
            write_annex_doc,
        )
        assert dataclasses.is_dataclass(AnnexXIData)
        assert callable(render_annex_xi_markdown)
        assert callable(render_annex_xii_markdown)
        assert callable(write_annex_doc)

    def test_annex_xi_data_frozen(self):
        from soup_cli.utils.annex_xi import AnnexXIData

        d = AnnexXIData(
            model_name="adapter-v1",
            base_model="meta-llama/Llama-3.1-8B",
            task="sft",
            dataset_summary="Anthropic HH-RLHF (filtered)",
            modalities=("text",),
            train_compute_flops=1.0e18,
            train_energy_kwh=12.5,
            train_co2_kg=4.0,
            top_domains=(),
            soup_version="0.59.0",
            run_id="run-abc",
            created_at="2026-05-18T12:00:00+00:00",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.model_name = "evil"  # type: ignore[misc]

    def test_annex_xi_markdown_contains_sections(self):
        from soup_cli.utils.annex_xi import (
            AnnexXIData,
            render_annex_xi_markdown,
        )

        d = AnnexXIData(
            model_name="adapter-v1",
            base_model="meta-llama/Llama-3.1-8B",
            task="sft",
            dataset_summary="HH-RLHF",
            modalities=("text",),
            train_compute_flops=1.0e18,
            train_energy_kwh=12.5,
            train_co2_kg=4.0,
            top_domains=(("huggingface.co", 0.6), ("github.com", 0.4)),
            soup_version="0.59.0",
            run_id="run-abc",
            created_at="2026-05-18T12:00:00+00:00",
        )
        md = render_annex_xi_markdown(d)
        # Annex XI Section 1 & 2 headers must appear
        assert "Annex XI" in md
        assert "Section 1" in md
        assert "Section 2" in md
        assert "adapter-v1" in md
        # FLOPs / energy / CO2 must be cited
        assert "kWh" in md
        assert "CO" in md
        # Top-10 domains
        assert "huggingface.co" in md

    def test_annex_xii_summary_markdown(self):
        from soup_cli.utils.annex_xi import (
            AnnexXIData,
            render_annex_xii_markdown,
        )

        d = AnnexXIData(
            model_name="adapter-v1",
            base_model="meta-llama/Llama-3.1-8B",
            task="sft",
            dataset_summary="HH-RLHF",
            modalities=("text",),
            train_compute_flops=1.0e18,
            train_energy_kwh=12.5,
            train_co2_kg=4.0,
            top_domains=(),
            soup_version="0.59.0",
            run_id="run-abc",
            created_at="2026-05-18T12:00:00+00:00",
        )
        md = render_annex_xii_markdown(d)
        assert "Annex XII" in md
        assert "Article 53" in md

    def test_top_domains_capped_at_10(self):
        from soup_cli.utils.annex_xi import AnnexXIData

        many = tuple((f"d{i}.com", 1.0) for i in range(20))
        # Should be accepted but capped on render (kept on data for round-trip)
        d = AnnexXIData(
            model_name="a",
            base_model="b",
            task="sft",
            dataset_summary="",
            modalities=("text",),
            train_compute_flops=0.0,
            train_energy_kwh=0.0,
            train_co2_kg=0.0,
            top_domains=many,
            soup_version="0.59.0",
            run_id="r",
            created_at="2026-05-18T12:00:00+00:00",
        )
        from soup_cli.utils.annex_xi import render_annex_xi_markdown
        md = render_annex_xi_markdown(d)
        # We slice to top-10
        for i in range(10):
            assert f"d{i}.com" in md

    def test_negative_flops_rejected(self):
        from soup_cli.utils.annex_xi import AnnexXIData

        with pytest.raises(ValueError):
            AnnexXIData(
                model_name="a", base_model="b", task="sft",
                dataset_summary="", modalities=("text",),
                train_compute_flops=-1.0, train_energy_kwh=0.0,
                train_co2_kg=0.0, top_domains=(), soup_version="0.59.0",
                run_id="r", created_at="2026-05-18T12:00:00+00:00",
            )

    def test_null_byte_model_name_rejected(self):
        from soup_cli.utils.annex_xi import AnnexXIData

        with pytest.raises(ValueError):
            AnnexXIData(
                model_name="bad\x00", base_model="b", task="sft",
                dataset_summary="", modalities=("text",),
                train_compute_flops=0.0, train_energy_kwh=0.0,
                train_co2_kg=0.0, top_domains=(), soup_version="0.59.0",
                run_id="r", created_at="2026-05-18T12:00:00+00:00",
            )

    def test_write_annex_doc_atomic(self, tmp_path, monkeypatch):
        from soup_cli.utils.annex_xi import AnnexXIData, write_annex_doc

        monkeypatch.chdir(tmp_path)
        d = AnnexXIData(
            model_name="a", base_model="b", task="sft",
            dataset_summary="", modalities=("text",),
            train_compute_flops=0.0, train_energy_kwh=0.0,
            train_co2_kg=0.0, top_domains=(), soup_version="0.59.0",
            run_id="r", created_at="2026-05-18T12:00:00+00:00",
        )
        out = tmp_path / "annex.md"
        write_annex_doc(d, "xi", str(out))
        assert out.is_file()
        assert "Annex XI" in out.read_text()

    def test_write_annex_unknown_section_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.annex_xi import AnnexXIData, write_annex_doc

        monkeypatch.chdir(tmp_path)
        d = AnnexXIData(
            model_name="a", base_model="b", task="sft",
            dataset_summary="", modalities=("text",),
            train_compute_flops=0.0, train_energy_kwh=0.0,
            train_co2_kg=0.0, top_domains=(), soup_version="0.59.0",
            run_id="r", created_at="2026-05-18T12:00:00+00:00",
        )
        with pytest.raises(ValueError):
            write_annex_doc(d, "xx", str(tmp_path / "x.md"))


# ---------- Part D: Audit log ----------


class TestAuditLog:
    def test_imports(self):
        from soup_cli.utils.audit_log import (
            AuditEvent,
            append_audit_event,
            redact_event,
            rotate_if_needed,
        )
        assert dataclasses.is_dataclass(AuditEvent)
        assert callable(append_audit_event)
        assert callable(redact_event)
        assert callable(rotate_if_needed)

    def test_audit_event_frozen_and_validated(self):
        from soup_cli.utils.audit_log import AuditEvent

        ev = AuditEvent(
            timestamp="2026-05-18T12:00:00+00:00",
            command="train",
            args=("--config", "soup.yaml"),
            exit_code=0,
            host_id="laptop-01",
            operator_id="alpamys",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.command = "x"  # type: ignore[misc]
        assert ev.exit_code == 0

    def test_audit_event_null_byte_rejected(self):
        from soup_cli.utils.audit_log import AuditEvent

        with pytest.raises(ValueError):
            AuditEvent(
                timestamp="t",
                command="train\x00",
                args=(),
                exit_code=0,
                host_id="h",
                operator_id="o",
            )

    def test_redact_event_strips_secrets(self):
        from soup_cli.utils.audit_log import AuditEvent, redact_event

        ev = AuditEvent(
            timestamp="t",
            command="push",
            args=("--token", "hf_aaaaaaaaaaaaaaaa", "--api-key", "sk-abcdef1234567890"),
            exit_code=0,
            host_id="h",
            operator_id="o",
        )
        red = redact_event(ev)
        assert "hf_aaaaaaaaaaaaaaaa" not in " ".join(red.args)
        assert "sk-abcdef1234567890" not in " ".join(red.args)
        assert "<redacted>" in " ".join(red.args)

    def test_append_audit_event_writes_line(self, tmp_path):
        from soup_cli.utils.audit_log import AuditEvent, append_audit_event

        log_path = tmp_path / "audit.jsonl"
        ev = AuditEvent(
            timestamp="2026-05-18T12:00:00+00:00",
            command="train",
            args=("--config", "soup.yaml"),
            exit_code=0,
            host_id="h",
            operator_id="o",
        )
        append_audit_event(ev, str(log_path))
        line = log_path.read_text().strip()
        rec = json.loads(line)
        assert rec["command"] == "train"
        assert rec["exit_code"] == 0

    def test_append_audit_event_appends_not_overwrites(self, tmp_path):
        from soup_cli.utils.audit_log import AuditEvent, append_audit_event

        log_path = tmp_path / "audit.jsonl"
        for cmd in ("train", "eval", "push"):
            append_audit_event(
                AuditEvent(
                    timestamp="t",
                    command=cmd,
                    args=(),
                    exit_code=0,
                    host_id="h",
                    operator_id="o",
                ),
                str(log_path),
            )
        lines = [
            line for line in log_path.read_text().splitlines() if line.strip()
        ]
        assert len(lines) == 3

    def test_rotate_if_needed_renames_at_cap(self, tmp_path):
        from soup_cli.utils.audit_log import rotate_if_needed

        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("x" * 200)
        # cap at 100 bytes -> rotate
        rotated = rotate_if_needed(str(log_path), cap_bytes=100)
        assert rotated is True
        assert (tmp_path / "audit.jsonl.1").is_file()

    def test_rotate_does_nothing_under_cap(self, tmp_path):
        from soup_cli.utils.audit_log import rotate_if_needed

        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("small")
        rotated = rotate_if_needed(str(log_path), cap_bytes=100)
        assert rotated is False

    def test_rotate_rejects_invalid_cap(self, tmp_path):
        from soup_cli.utils.audit_log import rotate_if_needed

        with pytest.raises(ValueError):
            rotate_if_needed(str(tmp_path / "x"), cap_bytes=0)
        with pytest.raises(ValueError):
            rotate_if_needed(str(tmp_path / "x"), cap_bytes=True)  # type: ignore[arg-type]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX 0o600 perms")
    def test_audit_log_perms_0o600(self, tmp_path):
        from soup_cli.utils.audit_log import AuditEvent, append_audit_event

        log_path = tmp_path / "audit.jsonl"
        ev = AuditEvent(
            timestamp="t", command="train", args=(), exit_code=0,
            host_id="h", operator_id="o",
        )
        append_audit_event(ev, str(log_path))
        mode = stat.S_IMODE(os.stat(str(log_path)).st_mode)
        assert mode == 0o600


# ---------- Part E: Reproducibility receipt ----------


class TestReproReceipt:
    def test_imports(self):
        from soup_cli.utils.repro_receipt import (
            ReproReceipt,
            build_repro_receipt,
            write_repro_receipt,
        )
        assert dataclasses.is_dataclass(ReproReceipt)
        assert callable(build_repro_receipt)
        assert callable(write_repro_receipt)

    def test_receipt_captures_basic_env(self):
        from soup_cli.utils.repro_receipt import build_repro_receipt

        r = build_repro_receipt(
            seeds={"torch": 42, "numpy": 42, "python": 0},
            run_id="abc",
        )
        d = dataclasses.asdict(r)
        assert d["seeds"]["torch"] == 42
        # OS, python_version captured
        assert d.get("python_version")
        assert d.get("os")

    def test_receipt_frozen(self):
        from soup_cli.utils.repro_receipt import build_repro_receipt

        r = build_repro_receipt(seeds={"torch": 0}, run_id="abc")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.run_id = "evil"  # type: ignore[misc]

    def test_receipt_seeds_validation(self):
        from soup_cli.utils.repro_receipt import build_repro_receipt

        # non-int seed
        with pytest.raises(ValueError):
            build_repro_receipt(seeds={"torch": "abc"}, run_id="x")  # type: ignore[dict-item]

    def test_receipt_run_id_validation(self):
        from soup_cli.utils.repro_receipt import build_repro_receipt

        with pytest.raises(ValueError):
            build_repro_receipt(seeds={}, run_id="bad\x00")

    def test_write_repro_receipt_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.utils.repro_receipt import build_repro_receipt, write_repro_receipt

        monkeypatch.chdir(tmp_path)
        r = build_repro_receipt(seeds={}, run_id="abc")
        with pytest.raises(ValueError):
            write_repro_receipt(r, "/tmp/x.json")

    def test_write_repro_receipt_atomic(self, tmp_path, monkeypatch):
        from soup_cli.utils.repro_receipt import build_repro_receipt, write_repro_receipt

        monkeypatch.chdir(tmp_path)
        r = build_repro_receipt(seeds={"torch": 1}, run_id="abc")
        out = tmp_path / "repro.json"
        write_repro_receipt(r, str(out))
        assert out.is_file()
        data = json.loads(out.read_text())
        assert data["run_id"] == "abc"
        assert data["seeds"]["torch"] == 1


# ---------- Part F: CO2 + energy ----------


class TestEnergy:
    def test_imports(self):
        from soup_cli.utils.energy import (
            EnergyMeasurement,
            adjust_for_pue,
            measure_run_energy,
            validate_electricity_map_endpoint,
        )
        assert dataclasses.is_dataclass(EnergyMeasurement)
        assert callable(measure_run_energy)
        assert callable(validate_electricity_map_endpoint)
        assert callable(adjust_for_pue)

    def test_measurement_frozen(self):
        from soup_cli.utils.energy import EnergyMeasurement

        m = EnergyMeasurement(
            energy_kwh=1.0, co2_kg=0.4, pue=1.2,
            grid_intensity_g_per_kwh=400.0,
            source="codecarbon",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.energy_kwh = 0.0  # type: ignore[misc]

    def test_measure_returns_zero_when_codecarbon_missing(self):
        from soup_cli.utils.energy import measure_run_energy

        # No live codecarbon — returns None or zero, never crashes
        out = measure_run_energy(duration_seconds=0.0)
        # Returns either None or a zero-energy measurement
        assert out is None or out.energy_kwh == 0.0

    def test_validate_electricity_map_endpoint_loopback_ok(self):
        from soup_cli.utils.energy import validate_electricity_map_endpoint

        assert validate_electricity_map_endpoint(
            "http://localhost:8080/co2"
        ) == "http://localhost:8080/co2"

    def test_validate_electricity_map_endpoint_https_ok(self):
        from soup_cli.utils.energy import validate_electricity_map_endpoint

        v = validate_electricity_map_endpoint("https://api.electricitymap.org/v3")
        assert v.startswith("https://")

    def test_validate_electricity_map_rejects_lan_http(self):
        from soup_cli.utils.energy import validate_electricity_map_endpoint

        with pytest.raises(ValueError):
            validate_electricity_map_endpoint("http://10.0.0.1/co2")

    def test_validate_electricity_map_rejects_null_byte(self):
        from soup_cli.utils.energy import validate_electricity_map_endpoint

        with pytest.raises(ValueError):
            validate_electricity_map_endpoint("http://localhost\x00/x")

    def test_validate_electricity_map_rejects_bad_scheme(self):
        from soup_cli.utils.energy import validate_electricity_map_endpoint

        with pytest.raises(ValueError):
            validate_electricity_map_endpoint("file:///etc/passwd")

    def test_adjust_for_pue(self):
        from soup_cli.utils.energy import adjust_for_pue

        assert adjust_for_pue(1.0, 1.5) == pytest.approx(1.5)
        assert adjust_for_pue(0.0, 1.5) == 0.0

    def test_adjust_for_pue_rejects_bad_inputs(self):
        from soup_cli.utils.energy import adjust_for_pue

        with pytest.raises(ValueError):
            adjust_for_pue(-1.0, 1.5)
        with pytest.raises(ValueError):
            adjust_for_pue(1.0, 0.0)
        with pytest.raises(ValueError):
            adjust_for_pue(1.0, float("nan"))
        with pytest.raises(ValueError):
            adjust_for_pue(True, 1.5)  # type: ignore[arg-type]


# ---------- BOM + Energy integration ----------


class TestBomEnergyAttach:
    def test_bom_with_energy_metadata(self):
        from soup_cli.utils.bom import BomEntry, attach_energy, build_cyclonedx_bom
        from soup_cli.utils.energy import EnergyMeasurement

        entry = BomEntry(
            name="adapter",
            version="0.1.0",
            base_model="m",
            base_sha="a" * 64,
            config_sha="b" * 64,
            data_sha=None,
            task="sft",
            license="apache-2.0",
            parents=(),
            artifacts=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        m = EnergyMeasurement(
            energy_kwh=12.5, co2_kg=4.0, pue=1.2,
            grid_intensity_g_per_kwh=400.0,
            source="codecarbon",
        )
        entry2 = attach_energy(entry, m)
        doc = build_cyclonedx_bom(entry2)
        props = doc["metadata"].get("properties", [])
        names = {p["name"] for p in props}
        assert "soup:energy_kwh" in names
        assert "soup:co2_kg" in names


# ---------- Source-grep wiring ----------


class TestSourceWiring:
    def test_cli_registers_bom(self):
        cli_path = Path(soup_cli.__file__).parent / "cli.py"
        text = cli_path.read_text()
        # bom and attest must be wired
        assert "bom" in text.lower()
        assert "attest" in text.lower()

    def test_version_is_0_59(self):
        assert soup_cli.__version__ == "0.59.0"

    def test_no_top_level_heavy_imports(self):
        """v0.59 modules must not import torch/transformers at module top."""
        utils = Path(soup_cli.__file__).parent / "utils"
        for name in ("bom.py", "attest.py", "annex_xi.py", "audit_log.py",
                     "repro_receipt.py", "energy.py"):
            text = (utils / name).read_text()
            # Allow lazy imports inside functions; reject top-level only
            top = text.split("def ")[0]
            assert "import torch" not in top, name
            assert "import transformers" not in top, name


# ---------- Annex XI: integration via train flag ----------


class TestTrainAnnexXIFlag:
    def test_train_annex_xi_flag_present_in_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--annex-xi" in result.output

    def test_train_repro_receipt_flag_present_in_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--repro-receipt" in result.output


# ---------- Audit CLI ----------


class TestAuditCli:
    def test_audit_log_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["audit-log", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_audit_log_list_empty(self, tmp_path, monkeypatch):
        # Ensure a fresh empty location and don't crash
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
        runner = CliRunner()
        result = runner.invoke(app, ["audit-log", "tail", "--limit", "10"])
        # Should not crash even with empty/missing log
        assert result.exit_code == 0, (result.output, repr(result.exception))


# ---------- Review-wave follow-ups (v0.59.0 review fixes) ----------


class TestReviewFollowups:
    """Tests for the v0.59.0 review wave fixes (security/code/python/TDD)."""

    # --- Security HIGH H3 / Code review #9: redaction extended to all fields ---
    def test_redact_event_redacts_host_and_operator_fields(self):
        from soup_cli.utils.audit_log import AuditEvent, redact_event

        ev = AuditEvent(
            timestamp="2026-05-18T12:00:00+00:00",
            command="train",
            args=(),
            exit_code=0,
            host_id="host-Bearer abcdefgh12345",
            operator_id="op-hf_xxxxxxxxxxxxxxxx",
        )
        red = redact_event(ev)
        assert "<redacted>" in red.host_id
        assert "<redacted>" in red.operator_id

    def test_redact_event_handles_bearer_in_args(self):
        from soup_cli.utils.audit_log import AuditEvent, redact_event

        ev = AuditEvent(
            timestamp="t", command="serve",
            args=("--token", "Bearer abcdefgh12345"),
            exit_code=0, host_id="h", operator_id="o",
        )
        red = redact_event(ev)
        joined = " ".join(red.args)
        assert "Bearer abcdefgh12345" not in joined
        assert "<redacted>" in joined
        # length preserved
        assert len(red.args) == len(ev.args)

    # --- Security HIGH H1: rotation symlink rejection at backup path ---
    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink rejection")
    def test_rotate_refuses_symlink_backup(self, tmp_path):
        from soup_cli.utils.audit_log import rotate_if_needed

        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("x" * 200)
        # Pre-plant a symlink at the backup target.
        backup = tmp_path / "audit.jsonl.1"
        os.symlink("/etc/passwd", str(backup))
        # Rotation must refuse rather than overwrite the symlink.
        rotated = rotate_if_needed(str(log_path), cap_bytes=100)
        assert rotated is False
        # Original file still intact (NOT moved).
        assert log_path.is_file()

    # --- Security M2: env override path containment ---
    def test_default_log_path_rejects_unsafe_env_override(self, monkeypatch, tmp_path):
        from soup_cli.utils.audit_log import default_log_path

        # /etc/cron.d is outside $HOME / $CWD / $TMPDIR — fall back to default.
        monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", "/etc/cron.d/x")
        monkeypatch.chdir(tmp_path)
        resolved = default_log_path()
        # The default is ~/.soup/audit.jsonl — confirm we didn't honour the override.
        assert "/etc/cron.d/x" not in resolved

    def test_default_log_path_honours_in_bounds_env(self, monkeypatch, tmp_path):
        from soup_cli.utils.audit_log import default_log_path

        target = tmp_path / "audit.jsonl"
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", str(target))
        resolved = default_log_path()
        assert resolved == str(target)

    def test_default_log_path_rejects_null_byte_env(self, monkeypatch, tmp_path):
        from soup_cli.utils.audit_log import default_log_path

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", "/tmp/\x00/audit.jsonl")
        resolved = default_log_path()
        assert "\x00" not in resolved

    # --- Code review #2 / Security L1: artifact size_bytes validation ---
    def test_bom_artifact_size_bytes_non_int_rejected(self):
        from soup_cli.utils.bom import BomEntry, build_cyclonedx_bom

        entry = BomEntry(
            name="a", version="0.1", base_model="m",
            base_sha="a" * 64, config_sha="b" * 64, data_sha=None,
            task="sft", license=None,
            parents=(),
            artifacts=({"kind": "adapter", "sha256": "c" * 64, "size_bytes": "not-int"},),
            created_at="2026-05-18T12:00:00+00:00",
        )
        with pytest.raises(ValueError):
            build_cyclonedx_bom(entry)

    def test_bom_artifact_size_bytes_bool_rejected(self):
        from soup_cli.utils.bom import BomEntry, build_cyclonedx_bom

        entry = BomEntry(
            name="a", version="0.1", base_model="m",
            base_sha="a" * 64, config_sha="b" * 64, data_sha=None,
            task="sft", license=None,
            parents=(),
            artifacts=({"kind": "adapter", "sha256": "c" * 64, "size_bytes": True},),
            created_at="2026-05-18T12:00:00+00:00",
        )
        with pytest.raises(ValueError):
            build_cyclonedx_bom(entry)

    # --- Security M1: Annex markdown escape ---
    def test_annex_xi_escapes_markdown_active_chars(self):
        from soup_cli.utils.annex_xi import AnnexXIData, render_annex_xi_markdown

        d = AnnexXIData(
            model_name="adapter|injected]more",
            base_model="meta-llama/Llama",
            task="sft",
            dataset_summary="summary [click](javascript:evil)",
            modalities=("text",),
            train_compute_flops=0.0,
            train_energy_kwh=0.0,
            train_co2_kg=0.0,
            top_domains=(),
            soup_version="0.59.0",
            run_id="r",
            created_at="2026-05-18T12:00:00+00:00",
        )
        md = render_annex_xi_markdown(d)
        # Pipe + brackets neutralised
        assert "\\|" in md
        assert "\\[" in md
        assert "\\(" in md
        # The literal `[click]` markdown link shape must not survive
        assert "[click]" not in md

    def test_annex_xi_neutralises_newline_in_model_name(self):
        from soup_cli.utils.annex_xi import AnnexXIData, render_annex_xi_markdown

        # A newline injected into model_name would otherwise forge a heading.
        d = AnnexXIData(
            model_name="evil\n## Forged Heading",
            base_model="b", task="sft",
            dataset_summary="", modalities=("text",),
            train_compute_flops=0.0, train_energy_kwh=0.0,
            train_co2_kg=0.0, top_domains=(),
            soup_version="0.59.0", run_id="r",
            created_at="2026-05-18T12:00:00+00:00",
        )
        md = render_annex_xi_markdown(d)
        # The forged heading line should not appear (newline replaced with space).
        assert "\n## Forged Heading" not in md

    # --- Python review #2: public default_log_path ---
    def test_default_log_path_public_symbol(self):
        from soup_cli.utils import audit_log

        assert hasattr(audit_log, "default_log_path")
        assert callable(audit_log.default_log_path)

    # --- TDD #1 / TOCTOU: symlink rejection on each write helper ---
    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink rejection")
    def test_write_attestation_rejects_symlink_target(self, tmp_path, monkeypatch):
        from soup_cli.utils.attest import AttestationStatement, write_attestation

        monkeypatch.chdir(tmp_path)
        s = AttestationStatement(
            stage="train", subject_name="a", subject_sha256="a" * 64,
            builder_id="b", invocation={}, materials=(),
            created_at="2026-05-18T12:00:00+00:00",
        )
        target = tmp_path / "att.json"
        os.symlink("/etc/passwd", str(target))
        with pytest.raises(ValueError):
            write_attestation(s, str(target))

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink rejection")
    def test_write_annex_doc_rejects_symlink_target(self, tmp_path, monkeypatch):
        from soup_cli.utils.annex_xi import AnnexXIData, write_annex_doc

        monkeypatch.chdir(tmp_path)
        d = AnnexXIData(
            model_name="a", base_model="b", task="sft",
            dataset_summary="", modalities=("text",),
            train_compute_flops=0.0, train_energy_kwh=0.0,
            train_co2_kg=0.0, top_domains=(),
            soup_version="0.59.0", run_id="r",
            created_at="2026-05-18T12:00:00+00:00",
        )
        target = tmp_path / "annex.md"
        os.symlink("/etc/passwd", str(target))
        with pytest.raises(ValueError):
            write_annex_doc(d, "xi", str(target))

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink rejection")
    def test_write_repro_receipt_rejects_symlink_target(self, tmp_path, monkeypatch):
        from soup_cli.utils.repro_receipt import build_repro_receipt, write_repro_receipt

        monkeypatch.chdir(tmp_path)
        r = build_repro_receipt(seeds={}, run_id="abc")
        target = tmp_path / "repro.json"
        os.symlink("/etc/passwd", str(target))
        with pytest.raises(ValueError):
            write_repro_receipt(r, str(target))

    # --- TDD #2: append_audit_event rejects outside-cwd null-byte path ---
    def test_append_audit_event_null_byte_path_rejected(self):
        from soup_cli.utils.audit_log import AuditEvent, append_audit_event

        ev = AuditEvent(
            timestamp="t", command="train", args=(),
            exit_code=0, host_id="h", operator_id="o",
        )
        with pytest.raises(ValueError):
            append_audit_event(ev, "/tmp/\x00/x")

    def test_append_audit_event_empty_path_rejected(self):
        from soup_cli.utils.audit_log import AuditEvent, append_audit_event

        ev = AuditEvent(
            timestamp="t", command="train", args=(),
            exit_code=0, host_id="h", operator_id="o",
        )
        with pytest.raises(ValueError):
            append_audit_event(ev, "")

    # --- TDD #3: rotate boundary at exactly cap_bytes (returns False) ---
    def test_rotate_at_exact_cap_does_not_rotate(self, tmp_path):
        from soup_cli.utils.audit_log import rotate_if_needed

        log = tmp_path / "audit.jsonl"
        log.write_text("x" * 100)
        rotated = rotate_if_needed(str(log), cap_bytes=100)
        assert rotated is False

    def test_rotate_just_over_cap_rotates(self, tmp_path):
        from soup_cli.utils.audit_log import rotate_if_needed

        log = tmp_path / "audit.jsonl"
        log.write_text("x" * 101)
        rotated = rotate_if_needed(str(log), cap_bytes=100)
        assert rotated is True
        assert (tmp_path / "audit.jsonl.1").is_file()

    # --- TDD #4: read_audit_tail validation ---
    def test_read_audit_tail_bool_limit_rejected(self):
        from soup_cli.utils.audit_log import read_audit_tail

        with pytest.raises(ValueError):
            read_audit_tail(limit=True)  # type: ignore[arg-type]

    def test_read_audit_tail_zero_limit_rejected(self):
        from soup_cli.utils.audit_log import read_audit_tail

        with pytest.raises(ValueError):
            read_audit_tail(limit=0)

    def test_read_audit_tail_missing_file_returns_empty(self, tmp_path):
        from soup_cli.utils.audit_log import read_audit_tail

        out = read_audit_tail(str(tmp_path / "missing.jsonl"), limit=10)
        assert out == []

    def test_read_audit_tail_skips_malformed_lines(self, tmp_path):
        from soup_cli.utils.audit_log import read_audit_tail

        log = tmp_path / "audit.jsonl"
        log.write_text(
            '{"command": "train", "exit_code": 0}\n'
            "this-is-not-json\n"
            '{"command": "eval", "exit_code": 0}\n',
            encoding="utf-8",
        )
        out = read_audit_tail(str(log), limit=10)
        # 2 valid lines, 1 malformed skipped.
        assert len(out) == 2

    # --- TDD #6: bool-as-int rejection on AnnexXIData numeric fields ---
    def test_annex_xi_flops_bool_rejected(self):
        from soup_cli.utils.annex_xi import AnnexXIData

        with pytest.raises(ValueError):
            AnnexXIData(
                model_name="a", base_model="b", task="sft",
                dataset_summary="", modalities=("text",),
                train_compute_flops=True,  # type: ignore[arg-type]
                train_energy_kwh=0.0,
                train_co2_kg=0.0, top_domains=(),
                soup_version="0.59.0", run_id="r",
                created_at="2026-05-18T12:00:00+00:00",
            )

    def test_annex_xi_kwh_bool_rejected(self):
        from soup_cli.utils.annex_xi import AnnexXIData

        with pytest.raises(ValueError):
            AnnexXIData(
                model_name="a", base_model="b", task="sft",
                dataset_summary="", modalities=("text",),
                train_compute_flops=0.0,
                train_energy_kwh=True,  # type: ignore[arg-type]
                train_co2_kg=0.0, top_domains=(),
                soup_version="0.59.0", run_id="r",
                created_at="2026-05-18T12:00:00+00:00",
            )

    def test_audit_event_exit_code_bool_rejected(self):
        from soup_cli.utils.audit_log import AuditEvent

        with pytest.raises(ValueError):
            AuditEvent(
                timestamp="t", command="train", args=(),
                exit_code=True,  # type: ignore[arg-type]
                host_id="h", operator_id="o",
            )

    def test_repro_seeds_bool_value_rejected(self):
        from soup_cli.utils.repro_receipt import build_repro_receipt

        with pytest.raises(ValueError):
            build_repro_receipt(
                seeds={"torch": True},  # type: ignore[dict-item]
                run_id="r",
            )

    # --- TDD #8: tighter CLI help assertion ---
    def test_bom_cli_help_explicitly_lists_emit_subcommand(self):
        runner = CliRunner()
        result = runner.invoke(app, ["bom", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        # Specifically check for the 'emit' command name in the help output.
        assert "emit" in result.output

    # --- TDD #12: top_domains cap absent ---
    def test_top_domains_only_first_10_rendered(self):
        from soup_cli.utils.annex_xi import (
            AnnexXIData,
            render_annex_xi_markdown,
        )

        many = tuple((f"d{i}.com", 0.05) for i in range(15))
        d = AnnexXIData(
            model_name="a", base_model="b", task="sft",
            dataset_summary="", modalities=("text",),
            train_compute_flops=0.0, train_energy_kwh=0.0,
            train_co2_kg=0.0, top_domains=many,
            soup_version="0.59.0", run_id="r",
            created_at="2026-05-18T12:00:00+00:00",
        )
        md = render_annex_xi_markdown(d)
        # d0..d9 present, d10..d14 absent.
        for i in range(10):
            assert f"d{i}.com" in md
        for i in range(10, 15):
            assert f"d{i}.com" not in md

    # --- Code review #6: atomic_write_text central helper ---
    def test_atomic_write_text_helper_importable(self):
        from soup_cli.utils.paths import atomic_write_text

        assert callable(atomic_write_text)

    def test_atomic_write_text_rejects_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.utils.paths import atomic_write_text

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            atomic_write_text("hi", "/tmp/x.txt")

    def test_atomic_write_text_happy(self, tmp_path, monkeypatch):
        from soup_cli.utils.paths import atomic_write_text

        monkeypatch.chdir(tmp_path)
        out = tmp_path / "out.txt"
        atomic_write_text("hello", str(out))
        assert out.read_text() == "hello"
