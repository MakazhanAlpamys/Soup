"""Tests for v0.71.3 — Compliance / annex / audit / energy.

Closes:
- #180 Live CodeCarbon hook (EnergyTracker context manager) + `soup train --track-energy`
- #181 PDF rendering for Annex XI/XII (`write_annex_doc(..., fmt="pdf")`)
- #182 Soup Can manifest v3 `attestations` field + embed in-toto Statement
- #183 Audit-log auto-instrumentation via the top-level CLI entry point
- #184 Auto-populate AnnexXIData.top_domains from the data.train manifest
- #188 Auto-attach the repro-receipt into `soup airgap-bundle`

Pure-Python / config / sqlite — no GPU required for these tests. The live
codecarbon energy reading + reportlab PDF body are exercised in the Step-6
local smoke (codecarbon + reportlab installed); here we test the graceful /
schema / wiring surface.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# #180 — Live CodeCarbon hook: EnergyTracker context manager
# ---------------------------------------------------------------------------
class TestEnergyTracker:
    def test_imports(self):
        from soup_cli.utils.energy import EnergyMeasurement, EnergyTracker  # noqa: F401

    def test_tracker_graceful_when_codecarbon_missing(self, monkeypatch):
        """When codecarbon is absent the tracker yields, never crashes, and
        leaves measurement=None."""
        import builtins

        real_import = builtins.__import__

        def _no_codecarbon(name, *args, **kwargs):
            if name == "codecarbon" or name.startswith("codecarbon."):
                raise ImportError("no codecarbon")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_codecarbon)
        from soup_cli.utils.energy import EnergyTracker

        with EnergyTracker(pue=1.1) as tracker:
            pass  # no work
        assert tracker.measurement is None

    def test_tracker_exit_swallows_exceptions(self, monkeypatch):
        """A failure inside the codecarbon stop() must not crash __exit__ and
        must NOT mask an exception raised in the body."""
        import builtins

        real_import = builtins.__import__

        def _no_codecarbon(name, *args, **kwargs):
            if name == "codecarbon" or name.startswith("codecarbon."):
                raise ImportError("no codecarbon")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_codecarbon)
        from soup_cli.utils.energy import EnergyTracker

        with pytest.raises(RuntimeError, match="boom"):
            with EnergyTracker() as tracker:
                raise RuntimeError("boom")
        assert tracker.measurement is None

    def test_tracker_rejects_bad_pue(self):
        from soup_cli.utils.energy import EnergyTracker

        with pytest.raises(ValueError):
            EnergyTracker(pue=0.5)

    def test_tracker_rejects_bad_country(self):
        from soup_cli.utils.energy import EnergyTracker

        with pytest.raises(ValueError):
            EnergyTracker(country_iso_code="USAA")  # not 3-letter
        with pytest.raises(ValueError):
            EnergyTracker(country_iso_code="12")

    def test_tracker_builds_measurement_from_fake_codecarbon(self, monkeypatch):
        """Inject a fake codecarbon module so we exercise the real start/stop
        lifecycle math without installing codecarbon."""
        import types

        fake = types.ModuleType("codecarbon")

        class _Data:
            energy_consumed = 0.5  # kWh
            emissions = 0.2  # kg

        class _FakeTracker:
            def __init__(self, **kwargs):
                self.final_emissions_data = _Data()
                self._started = False

            def start(self):
                self._started = True

            def stop(self):
                return 0.2  # kg CO2

        fake.OfflineEmissionsTracker = _FakeTracker
        monkeypatch.setitem(sys.modules, "codecarbon", fake)
        from soup_cli.utils.energy import EnergyTracker

        with EnergyTracker(pue=1.2, country_iso_code="USA") as tracker:
            pass
        m = tracker.measurement
        assert m is not None
        # energy is PUE-scaled: 0.5 * 1.2
        assert abs(m.energy_kwh - 0.6) < 1e-9
        # co2 PUE-scaled: 0.2 * 1.2
        assert abs(m.co2_kg - 0.24) < 1e-9
        # grid intensity computed from raw: 0.2 / 0.5 * 1000 = 400 g/kWh
        assert abs(m.grid_intensity_g_per_kwh - 400.0) < 1e-6
        assert m.pue == 1.2
        assert m.source == "codecarbon-offline"

    def test_tracker_zero_energy_uses_default_grid(self, monkeypatch):
        import types

        fake = types.ModuleType("codecarbon")

        class _Data:
            energy_consumed = 0.0
            emissions = 0.0

        class _FakeTracker:
            def __init__(self, **kwargs):
                self.final_emissions_data = _Data()

            def start(self):
                pass

            def stop(self):
                return 0.0

        fake.OfflineEmissionsTracker = _FakeTracker
        monkeypatch.setitem(sys.modules, "codecarbon", fake)
        from soup_cli.utils.energy import EnergyTracker

        with EnergyTracker(grid_intensity_g_per_kwh=350.0) as tracker:
            pass
        m = tracker.measurement
        assert m is not None
        assert m.energy_kwh == 0.0
        assert m.grid_intensity_g_per_kwh == 350.0

    def test_train_track_energy_flag_in_help(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        out = "".join(ch for ch in result.output if ord(ch) < 128)
        assert "track-energy" in out.replace("─", "")


# ---------------------------------------------------------------------------
# #184 — Auto-populate AnnexXIData.top_domains from the training corpus
# ---------------------------------------------------------------------------
class TestTopDomains:
    def test_imports(self):
        from soup_cli.utils.annex_xi import (  # noqa: F401
            extract_top_domains,
            load_top_domains_from_jsonl,
        )

    def test_extract_top_domains_counts_and_shares(self):
        from soup_cli.utils.annex_xi import extract_top_domains

        rows = [
            {"text": "see https://example.com/a and https://example.com/b"},
            {"text": "also http://other.org/x"},
            {"content": "visit https://example.com/c"},
        ]
        out = extract_top_domains(rows)
        # example.com appears 3x, other.org 1x → 4 urls total
        assert out[0] == ("example.com", 0.75)
        assert out[1] == ("other.org", 0.25)

    def test_extract_top_domains_strips_port_lowercases(self):
        from soup_cli.utils.annex_xi import extract_top_domains

        rows = [{"text": "https://Example.COM:8443/x https://example.com/y"}]
        out = extract_top_domains(rows)
        assert out == (("example.com", 1.0),)

    def test_extract_top_domains_messages_field(self):
        from soup_cli.utils.annex_xi import extract_top_domains

        rows = [
            {"messages": [{"role": "user", "content": "https://hf.co/datasets/x"}]},
        ]
        out = extract_top_domains(rows)
        assert out == (("hf.co", 1.0),)

    def test_extract_top_domains_empty_returns_empty(self):
        from soup_cli.utils.annex_xi import extract_top_domains

        assert extract_top_domains([]) == ()
        assert extract_top_domains([{"text": "no urls here"}]) == ()

    def test_extract_top_domains_caps_at_top_n(self):
        from soup_cli.utils.annex_xi import extract_top_domains

        rows = [{"text": " ".join(f"https://d{i}.com/x" for i in range(20))}]
        out = extract_top_domains(rows, top_n=5)
        assert len(out) == 5

    def test_extract_top_domains_non_iterable_rejected(self):
        from soup_cli.utils.annex_xi import extract_top_domains

        with pytest.raises(TypeError):
            extract_top_domains(42)

    def test_extract_top_domains_bad_top_n(self):
        from soup_cli.utils.annex_xi import extract_top_domains

        with pytest.raises(ValueError):
            extract_top_domains([], top_n=0)
        with pytest.raises(ValueError):
            extract_top_domains([], top_n=True)

    def test_load_from_jsonl(self, tmp_path, monkeypatch):
        from soup_cli.utils.annex_xi import load_top_domains_from_jsonl

        monkeypatch.chdir(tmp_path)
        p = tmp_path / "train.jsonl"
        p.write_text(
            '{"text": "https://example.com/a"}\n'
            '{"text": "https://example.com/b https://other.org/c"}\n',
            encoding="utf-8",
        )
        out = load_top_domains_from_jsonl("train.jsonl")
        assert out[0][0] == "example.com"

    def test_load_from_jsonl_missing_returns_empty(self, tmp_path, monkeypatch):
        from soup_cli.utils.annex_xi import load_top_domains_from_jsonl

        monkeypatch.chdir(tmp_path)
        assert load_top_domains_from_jsonl("nope.jsonl") == ()

    def test_load_from_jsonl_outside_cwd_returns_empty(self, tmp_path, monkeypatch):
        from soup_cli.utils.annex_xi import load_top_domains_from_jsonl

        monkeypatch.chdir(tmp_path)
        # an absolute path outside cwd — best-effort returns ()
        outside = tmp_path.parent / "x.jsonl"
        outside.write_text('{"text": "https://example.com/a"}\n', encoding="utf-8")
        assert load_top_domains_from_jsonl(str(outside)) == ()

    def test_load_from_jsonl_non_string_returns_empty(self):
        from soup_cli.utils.annex_xi import load_top_domains_from_jsonl

        assert load_top_domains_from_jsonl(None) == ()
        assert load_top_domains_from_jsonl("") == ()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink semantics")
    def test_load_from_jsonl_symlink_returns_empty(self, tmp_path, monkeypatch):
        from soup_cli.utils.annex_xi import load_top_domains_from_jsonl

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real.jsonl"
        real.write_text('{"text": "https://example.com/a"}\n', encoding="utf-8")
        link = tmp_path / "link.jsonl"
        os.symlink(real, link)
        assert load_top_domains_from_jsonl("link.jsonl") == ()


# ---------------------------------------------------------------------------
# #181 — PDF rendering for Annex XI/XII
# ---------------------------------------------------------------------------
def _sample_annex_data():
    from soup_cli.utils.annex_xi import AnnexXIData

    return AnnexXIData(
        model_name="my-model",
        base_model="meta-llama/Llama-3.2-1B",
        task="sft",
        dataset_summary="data/train.jsonl",
        modalities=("text",),
        train_compute_flops=1.5e18,
        train_energy_kwh=0.5,
        train_co2_kg=0.2,
        top_domains=(("example.com", 0.6), ("other.org", 0.4)),
        soup_version="0.71.3",
        run_id="run-1",
        created_at="2026-06-01T00:00:00+00:00",
    )


class TestAnnexPdf:
    def test_atomic_write_bytes_imports(self):
        from soup_cli.utils.paths import atomic_write_bytes  # noqa: F401

    def test_atomic_write_bytes_roundtrip(self, tmp_path, monkeypatch):
        from soup_cli.utils.paths import atomic_write_bytes

        monkeypatch.chdir(tmp_path)
        out = atomic_write_bytes(b"%PDF-1.4\nhello", "x.pdf")
        assert Path(out).read_bytes() == b"%PDF-1.4\nhello"

    def test_atomic_write_bytes_rejects_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.utils.paths import atomic_write_bytes

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            atomic_write_bytes(b"x", str(tmp_path.parent / "y.pdf"))

    def test_render_annex_pdf_starts_with_pdf_magic(self):
        from soup_cli.utils.annex_xi import render_annex_pdf

        data = _sample_annex_data()
        pdf = render_annex_pdf(data, "xi")
        assert isinstance(pdf, bytes)
        assert pdf.startswith(b"%PDF")

    def test_render_annex_pdf_xii(self):
        from soup_cli.utils.annex_xi import render_annex_pdf

        pdf = render_annex_pdf(_sample_annex_data(), "xii")
        assert pdf.startswith(b"%PDF")

    def test_render_annex_pdf_rejects_non_data(self):
        from soup_cli.utils.annex_xi import render_annex_pdf

        with pytest.raises(TypeError):
            render_annex_pdf({"not": "data"}, "xi")

    def test_render_annex_pdf_rejects_bad_section(self):
        from soup_cli.utils.annex_xi import render_annex_pdf

        with pytest.raises(ValueError):
            render_annex_pdf(_sample_annex_data(), "zz")

    def test_write_annex_doc_markdown_default(self, tmp_path, monkeypatch):
        from soup_cli.utils.annex_xi import write_annex_doc

        monkeypatch.chdir(tmp_path)
        out = write_annex_doc(_sample_annex_data(), "xi", "annex.md")
        body = Path(out).read_text(encoding="utf-8")
        assert "Annex XI" in body

    def test_write_annex_doc_pdf_format(self, tmp_path, monkeypatch):
        from soup_cli.utils.annex_xi import write_annex_doc

        monkeypatch.chdir(tmp_path)
        out = write_annex_doc(_sample_annex_data(), "xi", "annex.pdf", fmt="pdf")
        assert Path(out).read_bytes().startswith(b"%PDF")

    def test_write_annex_doc_rejects_bad_fmt(self, tmp_path, monkeypatch):
        from soup_cli.utils.annex_xi import write_annex_doc

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            write_annex_doc(_sample_annex_data(), "xi", "annex.x", fmt="docx")


# ---------------------------------------------------------------------------
# #182 — Soup Can manifest v3 `attestations` field
# ---------------------------------------------------------------------------
def _sample_statement():
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": "model", "digest": {"sha256": "a" * 64}}],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {"buildDefinition": {}, "runDetails": {}},
    }


class TestCanManifestV3:
    def test_version_bumped_to_3(self):
        from soup_cli.cans.schema import (
            CAN_FORMAT_VERSION,
            SUPPORTED_CAN_FORMAT_VERSIONS,
        )

        assert CAN_FORMAT_VERSION == 3
        assert SUPPORTED_CAN_FORMAT_VERSIONS == (1, 2, 3)

    def test_v1_and_v2_still_load(self):
        from soup_cli.cans.schema import Manifest

        for v in (1, 2, 3):
            m = Manifest(
                can_format_version=v, name="x", author="a",
                created_at="2026-01-01", base_hash="h",
            )
            assert m.can_format_version == v

    def test_attestations_default_empty(self):
        from soup_cli.cans.schema import Manifest

        m = Manifest(
            can_format_version=3, name="x", author="a",
            created_at="2026-01-01", base_hash="h",
        )
        assert m.attestations == []

    def test_attestations_accepts_valid_statement(self):
        from soup_cli.cans.schema import Manifest

        m = Manifest(
            can_format_version=3, name="x", author="a",
            created_at="2026-01-01", base_hash="h",
            attestations=[_sample_statement()],
        )
        assert len(m.attestations) == 1
        assert m.attestations[0]["_type"].endswith("Statement/v1")

    def test_attestations_none_coerced_to_empty(self):
        from soup_cli.cans.schema import Manifest

        m = Manifest(
            can_format_version=3, name="x", author="a",
            created_at="2026-01-01", base_hash="h",
            attestations=None,
        )
        assert m.attestations == []

    def test_attestations_rejects_non_dict_entry(self):
        from pydantic import ValidationError

        from soup_cli.cans.schema import Manifest

        with pytest.raises(ValidationError):
            Manifest(
                can_format_version=3, name="x", author="a",
                created_at="2026-01-01", base_hash="h",
                attestations=["not-a-dict"],
            )

    def test_attestations_rejects_missing_type(self):
        from pydantic import ValidationError

        from soup_cli.cans.schema import Manifest

        with pytest.raises(ValidationError):
            Manifest(
                can_format_version=3, name="x", author="a",
                created_at="2026-01-01", base_hash="h",
                attestations=[{"predicateType": "x"}],
            )

    def test_attestations_too_many_rejected(self):
        from pydantic import ValidationError

        from soup_cli.cans.schema import Manifest

        with pytest.raises(ValidationError):
            Manifest(
                can_format_version=3, name="x", author="a",
                created_at="2026-01-01", base_hash="h",
                attestations=[_sample_statement() for _ in range(200)],
            )

    def test_validate_attestation_statement_helper(self):
        from soup_cli.cans.schema import validate_attestation_statement

        out = validate_attestation_statement(_sample_statement())
        assert out["_type"]
        with pytest.raises(ValueError):
            validate_attestation_statement({"_type": "x"})  # missing predicateType
        with pytest.raises(ValueError):
            validate_attestation_statement("nope")

    def test_pack_entry_embeds_attestations(self, tmp_path, monkeypatch):
        """pack_entry threads attestations into the written manifest.yaml."""
        import tarfile

        import yaml

        from soup_cli.cans import pack as pack_mod

        monkeypatch.chdir(tmp_path)

        class _FakeStore:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def resolve(self, eid):
                return "eid"

            def get(self, eid):
                return {
                    "name": "my-recipe", "base_model": "b", "task": "sft",
                    "entry_hash": "h", "config_json": "{}", "notes": "n",
                    "tags": [],
                }

        monkeypatch.setattr(pack_mod, "RegistryStore", lambda: _FakeStore())
        out = pack_mod.pack_entry(
            entry_id="eid", out_path="x.can", author="a",
            attestations=[_sample_statement()],
        )
        with tarfile.open(out, "r:gz") as tar:
            manifest = yaml.safe_load(
                tar.extractfile("manifest.yaml").read().decode("utf-8")
            )
        assert manifest["can_format_version"] == 3
        assert len(manifest["attestations"]) == 1

    def test_can_pack_attest_flag_in_help(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["can", "pack", "--help"])
        assert result.exit_code == 0
        out = "".join(ch for ch in result.output if ord(ch) < 128)
        assert "attest" in out


# ---------------------------------------------------------------------------
# #183 — Audit-log auto-instrumentation via the top-level CLI entry point
# ---------------------------------------------------------------------------
class TestAuditInstrumentation:
    def test_imports(self):
        from soup_cli.cli import _emit_audit_event, _split_command_args  # noqa: F401

    def test_split_command_args_basic(self):
        from soup_cli.cli import _split_command_args

        cmd, args = _split_command_args(["soup", "train", "--config", "x"])
        assert cmd == "train"
        assert args == ["--config", "x"]

    def test_split_command_args_skips_log_level_value(self):
        from soup_cli.cli import _split_command_args

        cmd, args = _split_command_args(
            ["soup", "--log-level", "debug", "train", "--yes"]
        )
        assert cmd == "train"
        assert args == ["--yes"]

    def test_split_command_args_root_only(self):
        from soup_cli.cli import _split_command_args

        cmd, args = _split_command_args(["soup"])
        assert cmd == "(root)"
        assert args == []

    def test_split_command_args_help_flag(self):
        from soup_cli.cli import _split_command_args

        cmd, args = _split_command_args(["soup", "--help"])
        assert cmd == "(root)"
        assert args == ["--help"]

    def test_emit_audit_event_writes_line(self, tmp_path, monkeypatch):
        import soup_cli.cli as cli_mod

        log = tmp_path / "audit.jsonl"
        monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", str(log))
        monkeypatch.delenv("SOUP_NO_AUDIT_LOG", raising=False)
        monkeypatch.setattr(cli_mod, "_audit_disabled", False)
        cli_mod._emit_audit_event(["soup", "version"], 0)
        rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        assert len(rows) == 1
        assert rows[0]["command"] == "version"
        assert rows[0]["exit_code"] == 0

    def test_emit_audit_event_env_opt_out(self, tmp_path, monkeypatch):
        import soup_cli.cli as cli_mod

        log = tmp_path / "audit.jsonl"
        monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", str(log))
        monkeypatch.setenv("SOUP_NO_AUDIT_LOG", "1")
        monkeypatch.setattr(cli_mod, "_audit_disabled", False)
        cli_mod._emit_audit_event(["soup", "version"], 0)
        assert not log.exists()

    def test_emit_audit_event_flag_opt_out(self, tmp_path, monkeypatch):
        import soup_cli.cli as cli_mod

        log = tmp_path / "audit.jsonl"
        monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", str(log))
        monkeypatch.delenv("SOUP_NO_AUDIT_LOG", raising=False)
        monkeypatch.setattr(cli_mod, "_audit_disabled", True)
        cli_mod._emit_audit_event(["soup", "version"], 0)
        assert not log.exists()

    def test_emit_audit_event_never_raises(self, monkeypatch):
        import soup_cli.cli as cli_mod

        # Point at an impossible path via a broken default — must swallow.
        monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", "\x00bad")
        monkeypatch.delenv("SOUP_NO_AUDIT_LOG", raising=False)
        monkeypatch.setattr(cli_mod, "_audit_disabled", False)
        # Should not raise even with a garbage path.
        cli_mod._emit_audit_event(["soup", "x"], 1)

    def test_no_audit_log_flag_in_help(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        out = "".join(ch for ch in result.output if ord(ch) < 128)
        assert "no-audit-log" in out.replace("─", "")

    def test_subprocess_emits_audit_line(self, tmp_path):
        """End-to-end: a real `soup` invocation via run() writes an audit line."""
        log = tmp_path / "audit.jsonl"
        env = dict(os.environ)
        env["SOUP_AUDIT_LOG_PATH"] = str(log)
        env.pop("SOUP_NO_AUDIT_LOG", None)
        env["PYTHONUTF8"] = "1"
        r = subprocess.run(
            [sys.executable, "-m", "soup_cli", "version"],
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, (r.stdout, r.stderr)
        assert log.exists(), (r.stdout, r.stderr)
        rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        assert any(row["command"] == "version" for row in rows)

    def test_subprocess_no_audit_log_env_suppresses(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        env = dict(os.environ)
        env["SOUP_AUDIT_LOG_PATH"] = str(log)
        env["SOUP_NO_AUDIT_LOG"] = "1"
        env["PYTHONUTF8"] = "1"
        r = subprocess.run(
            [sys.executable, "-m", "soup_cli", "version"],
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, (r.stdout, r.stderr)
        assert not log.exists()


# ---------------------------------------------------------------------------
# #188 — Auto-attach the repro-receipt into `soup airgap-bundle`
# ---------------------------------------------------------------------------
def _make_model_dir(tmp_path, *, with_receipt=False):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"x": 1}', encoding="utf-8")
    if with_receipt:
        (model / "repro-receipt.json").write_text(
            json.dumps({"run_id": "r1", "seeds": {"torch": 42}}),
            encoding="utf-8",
        )
    return model


class TestAirgapReproReceipt:
    def test_plan_accepts_repro_receipt_field(self, tmp_path, monkeypatch):
        from soup_cli.utils.airgap_bundle import AirgapBundlePlan

        monkeypatch.chdir(tmp_path)
        plan = AirgapBundlePlan(
            output="b.tar", model_dir="model", dataset_dirs=(),
            wheel_dirs=(), kernel_dirs=(),
            bundle_size_cap_bytes=1024 * 1024,
            repro_receipt={"run_id": "r1"},
        )
        assert plan.repro_receipt == {"run_id": "r1"}

    def test_plan_repro_receipt_defaults_none(self, tmp_path, monkeypatch):
        from soup_cli.utils.airgap_bundle import AirgapBundlePlan

        monkeypatch.chdir(tmp_path)
        plan = AirgapBundlePlan(
            output="b.tar", model_dir="model", dataset_dirs=(),
            wheel_dirs=(), kernel_dirs=(),
            bundle_size_cap_bytes=1024 * 1024,
        )
        assert plan.repro_receipt is None

    def test_plan_repro_receipt_must_be_mapping(self, tmp_path, monkeypatch):
        from soup_cli.utils.airgap_bundle import AirgapBundlePlan

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            AirgapBundlePlan(
                output="b.tar", model_dir="model", dataset_dirs=(),
                wheel_dirs=(), kernel_dirs=(),
                bundle_size_cap_bytes=1024 * 1024,
                repro_receipt="not-a-dict",
            )

    def test_build_embeds_receipt_member(self, tmp_path, monkeypatch):
        import tarfile

        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
            inspect_airgap_bundle,
        )

        monkeypatch.chdir(tmp_path)
        _make_model_dir(tmp_path)
        plan = AirgapBundlePlan(
            output="b.tar", model_dir="model", dataset_dirs=(),
            wheel_dirs=(), kernel_dirs=(),
            bundle_size_cap_bytes=10 * 1024 * 1024,
            repro_receipt={"run_id": "r1", "seeds": {"torch": 42}},
        )
        manifest = build_airgap_bundle(plan)
        # The receipt is a manifest member.
        assert any(f.name == "repro-receipt.json" for f in manifest.files)
        # Tar actually contains the file.
        with tarfile.open("b.tar", "r") as tar:
            names = tar.getnames()
        assert "repro-receipt.json" in names
        # inspect round-trips the receipt back.
        read = inspect_airgap_bundle("b.tar")
        assert read.repro_receipt == {"run_id": "r1", "seeds": {"torch": 42}}

    def test_build_without_receipt_has_none(self, tmp_path, monkeypatch):
        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
            inspect_airgap_bundle,
        )

        monkeypatch.chdir(tmp_path)
        _make_model_dir(tmp_path)
        plan = AirgapBundlePlan(
            output="b.tar", model_dir="model", dataset_dirs=(),
            wheel_dirs=(), kernel_dirs=(),
            bundle_size_cap_bytes=10 * 1024 * 1024,
        )
        build_airgap_bundle(plan)
        read = inspect_airgap_bundle("b.tar")
        assert read.repro_receipt is None
        assert not any(f.name == "repro-receipt.json" for f in read.files)

    def test_cli_repro_receipt_flag_in_help(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["airgap-bundle", "--help"])
        assert result.exit_code == 0
        out = "".join(ch for ch in result.output if ord(ch) < 128)
        assert "repro-receipt" in out.replace("─", "")

    def test_cli_explicit_repro_receipt(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils.airgap_bundle import inspect_airgap_bundle

        monkeypatch.chdir(tmp_path)
        _make_model_dir(tmp_path)
        receipt = tmp_path / "receipt.json"
        receipt.write_text(json.dumps({"run_id": "explicit"}), encoding="utf-8")
        result = runner.invoke(
            app,
            ["airgap-bundle", "-o", "b.tar", "--model", "model",
             "--repro-receipt", "receipt.json"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        read = inspect_airgap_bundle("b.tar")
        assert read.repro_receipt == {"run_id": "explicit"}

    def test_cli_auto_detects_receipt_in_model_dir(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils.airgap_bundle import inspect_airgap_bundle

        monkeypatch.chdir(tmp_path)
        _make_model_dir(tmp_path, with_receipt=True)
        result = runner.invoke(
            app, ["airgap-bundle", "-o", "b.tar", "--model", "model"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        read = inspect_airgap_bundle("b.tar")
        assert read.repro_receipt == {"run_id": "r1", "seeds": {"torch": 42}}


# ---------------------------------------------------------------------------
# Review follow-ups (security M1 + python H1/H2 + tdd missing tests)
# ---------------------------------------------------------------------------
class TestReviewFollowups:
    # --- #180 EnergyTracker boundary (tdd HIGH 1 / LOW 16) ---
    @pytest.mark.parametrize("bad", [-5.0, True, float("nan"), float("inf")])
    def test_energy_tracker_rejects_bad_grid(self, bad):
        from soup_cli.utils.energy import EnergyTracker

        with pytest.raises(ValueError):
            EnergyTracker(grid_intensity_g_per_kwh=bad)

    def test_energy_tracker_normalises_country_to_upper(self, monkeypatch):
        import types

        captured = {}
        fake = types.ModuleType("codecarbon")

        class _Data:
            energy_consumed = 0.1
            emissions = 0.05

        class _FakeTracker:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.final_emissions_data = _Data()

            def start(self):
                pass

            def stop(self):
                return 0.05

        fake.OfflineEmissionsTracker = _FakeTracker
        monkeypatch.setitem(sys.modules, "codecarbon", fake)
        from soup_cli.utils.energy import EnergyTracker

        with EnergyTracker(country_iso_code="usa"):
            pass
        assert captured.get("country_iso_code") == "USA"

    # --- #184 extract_top_domains (tdd HIGH 2/3, LOW 18) ---
    def test_extract_top_domains_skips_non_mapping_rows(self):
        from soup_cli.utils.annex_xi import extract_top_domains

        out = extract_top_domains([42, None, "https://x.com/a", {"text": "no"}])
        assert out == (("x.com", 1.0),)

    def test_extract_top_domains_tie_break_domain_ascending(self):
        from soup_cli.utils.annex_xi import extract_top_domains

        out = extract_top_domains([{"text": "https://z.com/1 https://a.com/2"}])
        # equal counts → ascending domain
        assert out == (("a.com", 0.5), ("z.com", 0.5))

    def test_extract_top_domains_urls_per_row_cap(self, monkeypatch):
        import soup_cli.utils.annex_xi as ax

        monkeypatch.setattr(ax, "_MAX_URLS_PER_ROW", 2)
        text = " ".join(f"https://d{i}.com/x" for i in range(10))
        out = ax.extract_top_domains([{"text": text}])
        # Only the first 2 URLs counted.
        assert sum(round(s * 2) for _, s in out) == 2

    # --- #181 atomic_write_bytes (tdd HIGH 4/5) ---
    def test_atomic_write_bytes_rejects_non_bytes(self, tmp_path, monkeypatch):
        from soup_cli.utils.paths import atomic_write_bytes

        monkeypatch.chdir(tmp_path)
        with pytest.raises(TypeError):
            atomic_write_bytes("not-bytes", "x.pdf")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink semantics")
    def test_atomic_write_bytes_rejects_symlink_target(self, tmp_path, monkeypatch):
        from soup_cli.utils.paths import atomic_write_bytes

        monkeypatch.chdir(tmp_path)
        (tmp_path / "real").write_bytes(b"x")
        os.symlink(tmp_path / "real", tmp_path / "link.pdf")
        with pytest.raises(ValueError):
            atomic_write_bytes(b"y", "link.pdf")

    def test_render_annex_pdf_escapes_operator_markup(self):
        from soup_cli.utils.annex_xi import AnnexXIData, render_annex_pdf

        data = AnnexXIData(
            model_name="<b>evil</b> & co",
            base_model="b", task="sft", dataset_summary="d",
            modalities=("text",), train_compute_flops=0.0,
            train_energy_kwh=0.0, train_co2_kg=0.0, top_domains=(),
            soup_version="0.71.3", run_id="r",
            created_at="2026-06-01T00:00:00+00:00",
        )
        # Must not raise — reportlab paragraph parser would choke on raw <b>.
        assert render_annex_pdf(data, "xi").startswith(b"%PDF")

    # --- #182 attestation caps (security M1 / tdd HIGH 6/7) ---
    def test_validate_attestation_oversize_rejected(self):
        from soup_cli.cans.schema import validate_attestation_statement

        big = {**_sample_statement(), "pad": "a" * (1024 * 1024 + 10)}
        with pytest.raises(ValueError, match="too large"):
            validate_attestation_statement(big)

    def test_validate_attestation_non_serialisable_rejected(self):
        from soup_cli.cans.schema import validate_attestation_statement

        bad = {**_sample_statement(), "bad": {object()}}
        with pytest.raises(ValueError):
            validate_attestation_statement(bad)

    def test_manifest_rejects_oversize_attestation(self):
        from pydantic import ValidationError

        from soup_cli.cans.schema import Manifest

        big = {**_sample_statement(), "pad": "a" * (1024 * 1024 + 10)}
        with pytest.raises(ValidationError):
            Manifest(
                can_format_version=3, name="x", author="a",
                created_at="2026-01-01", base_hash="h",
                attestations=[big],
            )

    def test_load_attestation_file_oversize_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.can import _load_attestation_files

        monkeypatch.chdir(tmp_path)
        big = tmp_path / "big.json"
        big.write_text(
            json.dumps({**_sample_statement(), "pad": "a" * (1024 * 1024 + 50)}),
            encoding="utf-8",
        )
        with pytest.raises(typer.Exit):
            _load_attestation_files(["big.json"])

    # --- #183 audit splitter + exit codes (tdd MED 8/9/10/11) ---
    def test_split_command_args_value_opt_followed_by_flag(self):
        from soup_cli.cli import _split_command_args

        cmd, args = _split_command_args(
            ["soup", "--log-level", "--yes", "train", "--config", "x"]
        )
        # --yes is consumed as the --log-level value; train is the command.
        assert cmd == "train"
        assert args == ["--config", "x"]

    def test_split_command_args_trailing_value_opt(self):
        from soup_cli.cli import _split_command_args

        # Trailing value-opt: i += 2 overshoots, loop exits cleanly (no
        # IndexError) and the leftover token is returned as args.
        cmd, args = _split_command_args(["soup", "--log-level"])
        assert cmd == "(root)"
        assert args == ["--log-level"]

    def test_emit_audit_event_nonzero_exit_code(self, tmp_path, monkeypatch):
        import soup_cli.cli as cli_mod

        log = tmp_path / "audit.jsonl"
        monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", str(log))
        monkeypatch.delenv("SOUP_NO_AUDIT_LOG", raising=False)
        monkeypatch.setattr(cli_mod, "_audit_disabled", False)
        cli_mod._emit_audit_event(["soup", "train", "--config", "x"], 3)
        rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        assert rows[0]["command"] == "train"
        assert rows[0]["exit_code"] == 3
        assert "--config" in rows[0]["args"]

    def test_subprocess_no_audit_log_flag_suppresses(self, tmp_path):
        """The --no-audit-log FLAG (not env) flows through the real run()."""
        log = tmp_path / "audit.jsonl"
        env = dict(os.environ)
        env["SOUP_AUDIT_LOG_PATH"] = str(log)
        env.pop("SOUP_NO_AUDIT_LOG", None)
        env["PYTHONUTF8"] = "1"
        r = subprocess.run(
            [sys.executable, "-m", "soup_cli", "--no-audit-log", "version"],
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, (r.stdout, r.stderr)
        assert not log.exists()

    # --- #188 airgap repro-receipt edges (tdd MED 12/13/14/15, LOW 17) ---
    def test_build_receipt_over_size_cap_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
        )

        monkeypatch.chdir(tmp_path)
        _make_model_dir(tmp_path)
        plan = AirgapBundlePlan(
            output="b.tar", model_dir="model", dataset_dirs=(),
            wheel_dirs=(), kernel_dirs=(),
            bundle_size_cap_bytes=10,
            repro_receipt={"x": "a" * 1000},
        )
        with pytest.raises(ValueError, match="exceeds cap"):
            build_airgap_bundle(plan)

    def test_plan_accepts_mappingproxy_receipt(self, tmp_path, monkeypatch):
        from types import MappingProxyType

        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
            inspect_airgap_bundle,
        )

        monkeypatch.chdir(tmp_path)
        _make_model_dir(tmp_path)
        plan = AirgapBundlePlan(
            output="b.tar", model_dir="model", dataset_dirs=(),
            wheel_dirs=(), kernel_dirs=(),
            bundle_size_cap_bytes=10 * 1024 * 1024,
            repro_receipt=MappingProxyType({"run_id": "proxy"}),
        )
        build_airgap_bundle(plan)
        assert inspect_airgap_bundle("b.tar").repro_receipt == {"run_id": "proxy"}

    def test_manifest_from_payload_coerces_bad_receipt_to_none(self):
        from soup_cli.utils.airgap_bundle import _manifest_from_payload

        m = _manifest_from_payload({
            "soup_version": "0.71.3", "created_at": "x", "model_dir": "model",
            "datasets": [], "wheels": [], "kernels": [], "files": [],
            "total_bytes": 0, "repro_receipt": ["not", "a", "dict"],
        })
        assert m.repro_receipt is None

    def test_cli_auto_detect_malformed_receipt_is_none(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils.airgap_bundle import inspect_airgap_bundle

        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        (model / "repro-receipt.json").write_text("not json{", encoding="utf-8")
        result = runner.invoke(
            app, ["airgap-bundle", "-o", "b.tar", "--model", "model"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert inspect_airgap_bundle("b.tar").repro_receipt is None

    def test_cli_explicit_missing_receipt_exits_2(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        _make_model_dir(tmp_path)
        result = runner.invoke(
            app,
            ["airgap-bundle", "-o", "b.tar", "--model", "model",
             "--repro-receipt", "nope.json"],
        )
        assert result.exit_code == 2, (result.output, repr(result.exception))
