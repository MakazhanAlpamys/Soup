"""#1446: the provenance and compliance artifacts state what was measured, not a default.

Five documents written for audits and supply-chain tooling stated something false or
dropped what the user supplied: the SLSA statement lost ``--invocation``; Annex XI/XII
printed ``FLOPs: 0`` and ``0.000 kWh`` for quantities nobody measured; the CycloneDX 1.6
``serialNumber`` was not a UUID URN and ``--license`` went verbatim into ``license.id``;
the SPDX data relationship was reversed; the SR 11-7 receipt never filled
``driver_version``. Each artifact is pinned here against its specification.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.annex_xi import AnnexXIData, render_annex_xi_markdown, render_annex_xii_markdown
from soup_cli.utils.attest import AttestationStatement, build_slsa_provenance
from soup_cli.utils.bom import BomEntry, build_cyclonedx_bom, build_spdx_bom, render_bom
from soup_cli.utils.repro_receipt import build_repro_receipt, receipt_to_dict
from soup_cli.utils.spdx_license_ids import SPDX_LICENSE_IDS, canonical_spdx_id

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cyclonedx"
_SHA = "a" * 64
_UUID_URN = re.compile(
    r"^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _entry(license: str | None, data_sha: str | None = "c" * 64) -> BomEntry:
    return BomEntry(
        name="m", version="1", base_model="b", base_sha=_SHA, config_sha="b" * 64,
        data_sha=data_sha, task="sft", license=license, parents=(), artifacts=(),
        created_at="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# 1. soup attest emit --invocation
# ---------------------------------------------------------------------------

_INVOCATION = "soup train --config soup.yaml"


def _external_parameters(path: Path) -> dict:
    statement = json.loads(path.read_text(encoding="utf-8"))
    return statement["predicate"]["buildDefinition"]["externalParameters"]


class TestAttestationInvocation:
    def test_build_slsa_provenance_puts_the_command_in_external_parameters(self) -> None:
        s = AttestationStatement(
            stage="train", subject_name="m", subject_sha256=_SHA, builder_id="soup-cli",
            invocation={"command": _INVOCATION}, materials=(), created_at="2026-01-01T00:00:00Z",
        )
        params = build_slsa_provenance(s)["buildDefinition"]["externalParameters"]
        assert params["invocation"] == _INVOCATION
        assert params["stage"] == "train"

    def test_control_no_invocation_means_no_key(self) -> None:
        s = AttestationStatement(
            stage="train", subject_name="m", subject_sha256=_SHA, builder_id="soup-cli",
            invocation={"command": ""}, materials=(), created_at="2026-01-01T00:00:00Z",
        )
        assert "invocation" not in build_slsa_provenance(s)["buildDefinition"]["externalParameters"]

    def test_cli_unsigned_statement_carries_the_invocation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, [
            "attest", "emit", "--stage", "train", "--subject", "m", "--sha", _SHA,
            "--invocation", _INVOCATION, "-o", "att.json",
        ])
        assert result.exit_code == 0, result.output
        assert _external_parameters(tmp_path / "att.json")["invocation"] == _INVOCATION

    def test_cli_ed25519_signed_statement_carries_the_invocation_and_verifies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cryptography = pytest.importorskip("cryptography")
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519

        del cryptography
        monkeypatch.chdir(tmp_path)
        key = ed25519.Ed25519PrivateKey.generate()
        (tmp_path / "key.pem").write_bytes(key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
        result = CliRunner().invoke(app, [
            "attest", "emit", "--stage", "train", "--subject", "m", "--sha", _SHA,
            "--invocation", _INVOCATION, "--sign", "ed25519", "--key", "key.pem", "-o", "s.json",
        ])
        assert result.exit_code == 0, result.output
        assert _external_parameters(tmp_path / "s.json")["invocation"] == _INVOCATION
        verify = CliRunner().invoke(
            app, ["attest", "verify", "s.json", "--signature", "s.json.sig"]
        )
        assert verify.exit_code == 0, verify.output


# ---------------------------------------------------------------------------
# 2. Annex XI / XII: unmeasured quantities say so
# ---------------------------------------------------------------------------

def _annex(**overrides) -> AnnexXIData:
    fields = dict(
        model_name="adapter-v1", base_model="b", task="sft", dataset_summary="d",
        modalities=("text",), train_compute_flops=None, train_energy_kwh=None, train_co2_kg=None,
        top_domains=(), soup_version="0.75.1", run_id="run-1", created_at="2026-01-01T00:00:00Z",
    )
    fields.update(overrides)
    return AnnexXIData(**fields)


_LABELS = ("Training compute", "Energy consumed", "CO\u2082 emissions")


def _compute_lines(md: str) -> list[str]:
    return [line.strip() for line in md.splitlines() if any(label in line for label in _LABELS)]


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", text))


class TestAttestationInvocationLimits:
    def test_the_statement_printed_without_output_is_verbatim_json(self, monkeypatch) -> None:
        """Rich parsed `[...]` as markup and folded long lines at 80 columns (CI has
        no terminal), so the JSON on stdout no longer parsed (review of #1569)."""
        from rich.console import Console

        monkeypatch.setattr("soup_cli.commands.attest.console", Console(width=80))
        invocation = "soup train --set lora.target_modules=[q_proj,v_proj] --tag [/x] " + "x" * 80
        result = CliRunner().invoke(app, ["attest", "emit", "--stage", "train", "--subject", "m",
                                          "--sha", _SHA, "--invocation", invocation])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        text = result.stdout
        statement = json.loads(text[: text.rindex("}") + 1])
        params = statement["predicate"]["buildDefinition"]["externalParameters"]
        assert params["invocation"] == invocation

    def test_an_invocation_over_the_cap_is_refused_not_cut(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        base = ["attest", "emit", "--stage", "train", "--subject", "m", "--sha", _SHA]
        ok = CliRunner().invoke(app, [*base, "--invocation", "x" * 4096, "-o", "ok.json"])
        assert ok.exit_code == 0, ok.output
        assert _external_parameters(tmp_path / "ok.json")["invocation"] == "x" * 4096
        too_long = CliRunner().invoke(app, [*base, "--invocation", "x" * 4097, "-o", "no.json"])
        assert too_long.exit_code == 2, too_long.output
        assert "4096" in _plain(too_long.output)
        assert not (tmp_path / "no.json").exists()


class TestAnnexNotMeasured:
    @pytest.mark.parametrize("render", [render_annex_xi_markdown, render_annex_xii_markdown])
    def test_unmeasured_values_render_as_not_measured_never_zero(self, render) -> None:
        lines = _compute_lines(render(_annex()))
        assert len(lines) == 3, lines
        assert all("not measured" in line for line in lines), lines
        assert not any(re.search(r"\b0(\.000)?\b", line) for line in lines), lines

    @pytest.mark.parametrize("render", [render_annex_xi_markdown, render_annex_xii_markdown])
    def test_measured_values_are_still_written(self, render) -> None:
        md = render(_annex(train_compute_flops=1.0e18, train_energy_kwh=1.234, train_co2_kg=0.456))
        lines = _compute_lines(md)
        assert any("1.00e18" in line for line in lines), lines
        assert any("1.234 kWh" in line for line in lines), lines
        assert any("0.456 kg" in line for line in lines), lines
        assert not any("not measured" in line for line in lines)

    def test_a_negative_measurement_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="train_energy_kwh"):
            _annex(train_energy_kwh=-1.0)

    def test_the_real_writer_without_energy_says_not_measured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.commands import train as train_cmd
        from soup_cli.config.loader import load_config_from_string

        monkeypatch.chdir(tmp_path)
        (tmp_path / "train.jsonl").write_text(
            '{"instruction": "hi", "output": "hello"}\n', encoding="utf-8"
        )
        cfg = load_config_from_string(
            "base: HuggingFaceTB/SmolLM2-135M\ntask: sft\n"
            "data:\n  train: ./train.jsonl\n  format: alpaca\noutput: ./out\n"
        )
        train_cmd._write_annex_xi("annex.md", "run-1", cfg, energy=None)
        lines = _compute_lines((tmp_path / "annex.md").read_text(encoding="utf-8"))
        assert len(lines) == 3 and all("not measured" in line for line in lines), lines

    def test_the_real_writer_with_energy_writes_the_measurement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.commands import train as train_cmd
        from soup_cli.config.loader import load_config_from_string

        monkeypatch.chdir(tmp_path)
        (tmp_path / "train.jsonl").write_text(
            '{"instruction": "hi", "output": "hello"}\n', encoding="utf-8"
        )
        cfg = load_config_from_string(
            "base: HuggingFaceTB/SmolLM2-135M\ntask: sft\n"
            "data:\n  train: ./train.jsonl\n  format: alpaca\noutput: ./out\n"
        )
        energy = SimpleNamespace(energy_kwh=1.234, co2_kg=0.456)
        train_cmd._write_annex_xi("annex.md", "run-1", cfg, energy=energy)
        lines = _compute_lines((tmp_path / "annex.md").read_text(encoding="utf-8"))
        assert any("1.234 kWh" in line for line in lines), lines
        assert any("0.456 kg" in line for line in lines), lines
        assert any("FLOPs" in line and "not measured" in line for line in lines), lines


# ---------------------------------------------------------------------------
# 3. CycloneDX 1.6: schema-valid serialNumber and licenses
# ---------------------------------------------------------------------------

def _cyclonedx_validator():
    jsonschema = pytest.importorskip("jsonschema", minversion="4.18")
    referencing = pytest.importorskip("referencing")

    schemas = {name: json.loads((_FIXTURES / name).read_text(encoding="utf-8"))
               for name in ("bom-1.6.schema.json", "spdx.schema.json", "jsf-0.82.schema.json")}
    registry = referencing.Registry().with_resources(
        (f"http://cyclonedx.org/schema/{name}", referencing.Resource.from_contents(schema))
        for name, schema in schemas.items()
    )
    return jsonschema.Draft7Validator(schemas["bom-1.6.schema.json"], registry=registry)


class TestCycloneDx:
    def test_the_runtime_spdx_id_list_matches_the_vendored_schema(self) -> None:
        enum = json.loads((_FIXTURES / "spdx.schema.json").read_text(encoding="utf-8"))["enum"]
        assert set(enum) == set(SPDX_LICENSE_IDS)
        assert canonical_spdx_id("apache-2.0") == "Apache-2.0"
        assert canonical_spdx_id("Llama 3.1 Community") is None

    @pytest.mark.parametrize("license", ["apache-2.0", "Apache-2.0", "Llama 3.1 Community",
                                         "Apache-2.0 OR MIT", None])
    def test_the_document_validates_against_the_1_6_schema(self, license) -> None:
        doc = build_cyclonedx_bom(_entry(license))
        errors = sorted(_cyclonedx_validator().iter_errors(doc), key=lambda e: list(e.path))
        assert not errors, "\n".join(f"{list(e.path)}: {e.message[:160]}" for e in errors)

    def test_serial_number_is_an_rfc_4122_urn_every_time(self) -> None:
        serials = {build_cyclonedx_bom(_entry(None))["serialNumber"] for _ in range(50)}
        assert len(serials) == 50
        assert all(_UUID_URN.match(s) for s in serials), sorted(serials)[:3]

    @pytest.mark.parametrize("given,expected", [
        ("apache-2.0", [{"license": {"id": "Apache-2.0"}}]),
        ("Apache-2.0", [{"license": {"id": "Apache-2.0"}}]),
        ("mit", [{"license": {"id": "MIT"}}]),
        ("GPL-2.0+", [{"license": {"id": "GPL-2.0+"}}]),  # listed itself, so an id
        ("Llama 3.1 Community", [{"license": {"name": "Llama 3.1 Community"}}]),
        ("Apache-2.0 OR MIT", [{"expression": "Apache-2.0 OR MIT"}]),
    ])
    def test_license_lands_in_id_name_or_expression(self, given, expected) -> None:
        assert build_cyclonedx_bom(_entry(given))["metadata"]["component"]["licenses"] == expected

    def test_render_bom_writes_the_same_values(self) -> None:
        doc = json.loads(render_bom(_entry("apache-2.0"), "cyclonedx"))
        assert _UUID_URN.match(doc["serialNumber"])
        assert doc["metadata"]["component"]["licenses"] == [{"license": {"id": "Apache-2.0"}}]


# ---------------------------------------------------------------------------
# 4. SPDX 2.3: the data is a build dependency OF the model
# ---------------------------------------------------------------------------

class TestSpdxLicenseFields:
    @pytest.mark.parametrize("given,expected", [
        ("apache-2.0", "Apache-2.0"),
        ("MIT", "MIT"),
        ("GPL-2.0+", "GPL-2.0+"),  # listed as an id itself, so not an expression
        ("Apache-2.0 OR MIT", "Apache-2.0 OR MIT"),
        (None, "NOASSERTION"),
    ])
    def test_concluded_and_declared_carry_a_valid_expression(self, given, expected) -> None:
        pkg = build_spdx_bom(_entry(given))["packages"][0]
        assert (pkg["licenseConcluded"], pkg["licenseDeclared"]) == (expected, expected)
        assert "hasExtractedLicensingInfos" not in build_spdx_bom(_entry(given))

    def test_a_non_spdx_name_becomes_a_license_ref_with_its_extracted_info(self) -> None:
        doc = build_spdx_bom(_entry("Llama 3.1 Community"))
        pkg = doc["packages"][0]
        assert pkg["licenseConcluded"] == pkg["licenseDeclared"] == "LicenseRef-Llama-3.1-Community"
        assert doc["hasExtractedLicensingInfos"] == [{
            "licenseId": "LicenseRef-Llama-3.1-Community",
            "name": "Llama 3.1 Community",
            "extractedText": "Llama 3.1 Community",
        }]


    def test_a_name_with_no_idstring_safe_characters_gets_a_content_hash_ref(self) -> None:
        doc = build_spdx_bom(_entry("???"))
        ref = doc["packages"][0]["licenseConcluded"]
        assert re.fullmatch(r"LicenseRef-[0-9a-f]{12}", ref), ref
        assert doc["hasExtractedLicensingInfos"][0]["licenseId"] == ref
        assert build_spdx_bom(_entry("???"))["packages"][0]["licenseConcluded"] == ref  # stable


class TestLicenseNamesThatOnlyLookLikeExpressions:
    @pytest.mark.parametrize("given", [
        "Llama 3.2 Community License and Acceptable Use Policy",
        "Gemma Terms of Use and Prohibited Use Policy",
        "Apache 2.0 with Commons Clause",
        "CC-BY-NC-4.0 or commercial license",
        "apache-2.0 or mit",
        "Apache-2.0 OR Llama-Community",
        "MIT OR",
        "OR MIT",
        "(MIT OR Apache-2.0",
        "MIT) OR (Apache-2.0",  # a close before any open: the depth guard
        "MIT OR Apache-2.0 and BSD-3-Clause",  # one lower-case operator among upper-case ones
    ])
    def test_a_name_that_only_looks_like_an_expression_is_a_name(self, given) -> None:
        licenses = build_cyclonedx_bom(_entry(given))["metadata"]["component"]["licenses"]
        assert licenses == [{"license": {"name": given}}]
        concluded = build_spdx_bom(_entry(given))["packages"][0]["licenseConcluded"]
        assert re.fullmatch(r"LicenseRef-[A-Za-z0-9.-]+", concluded), concluded

    @pytest.mark.parametrize("given,expected", [
        ("Apache-2.0 OR MIT", "Apache-2.0 OR MIT"),
        ("apache-2.0 OR mit", "Apache-2.0 OR MIT"),
        ("(MIT OR Apache-2.0) AND BSD-3-Clause", "(MIT OR Apache-2.0) AND BSD-3-Clause"),
        ("GPL-2.0-only WITH Classpath-exception-2.0", "GPL-2.0-only WITH Classpath-exception-2.0"),
        ("Apache-2.0 OR LicenseRef-Llama-Community", "Apache-2.0 OR LicenseRef-Llama-Community"),
        ("apache-2.0+ OR mit", "Apache-2.0+ OR MIT"),  # the `+` suffix survives canonicalisation
        ("apache-2.0+", "Apache-2.0+"),  # a lone `id+` is a simple expression
        ("LicenseRef-Llama-Community", "LicenseRef-Llama-Community"),  # so is a lone ref
    ])
    def test_a_real_expression_is_kept_with_canonical_ids(self, given, expected) -> None:
        licenses = build_cyclonedx_bom(_entry(given))["metadata"]["component"]["licenses"]
        assert licenses == [{"expression": expected}]
        assert build_spdx_bom(_entry(given))["packages"][0]["licenseConcluded"] == expected

    def test_a_license_ref_operand_is_defined_in_the_document(self) -> None:
        """SPDX requires every ``LicenseRef-`` a field uses to have an
        ``hasExtractedLicensingInfos`` entry, operands inside an expression included."""
        # two distinct refs, one of them repeated: one entry per distinct ref, in order
        given = "LicenseRef-Llama OR MIT OR LicenseRef-Llama OR LicenseRef-Gemma"
        doc = build_spdx_bom(_entry(given))
        assert doc["packages"][0]["licenseConcluded"] == given
        assert doc["hasExtractedLicensingInfos"] == [
            {"licenseId": ref, "name": ref, "extractedText": ref}
            for ref in ("LicenseRef-Llama", "LicenseRef-Gemma")
        ]

    def test_a_document_ref_qualified_operand_makes_the_value_a_name(self) -> None:
        """The document emits no ``externalDocumentRefs``, so a ref into another document
        cannot be honoured; the value is kept verbatim as a name instead."""
        given = "MIT OR DocumentRef-other:LicenseRef-Custom"
        licenses = build_cyclonedx_bom(_entry(given))["metadata"]["component"]["licenses"]
        assert licenses == [{"license": {"name": given}}]
        doc = build_spdx_bom(_entry(given))
        concluded = doc["packages"][0]["licenseConcluded"]
        assert re.fullmatch(r"LicenseRef-[A-Za-z0-9.-]+", concluded), concluded
        assert doc["hasExtractedLicensingInfos"][0]["extractedText"] == given


class TestSpdxRelationship:
    def test_data_is_the_build_dependency_of_the_model(self) -> None:
        rels = {(r["spdxElementId"], r["relationshipType"], r["relatedSpdxElement"])
                for r in build_spdx_bom(_entry(None))["relationships"]}
        assert ("SPDXRef-Data", "BUILD_DEPENDENCY_OF", "SPDXRef-Model") in rels
        assert ("SPDXRef-Model", "BUILD_DEPENDENCY_OF", "SPDXRef-Data") not in rels
        assert ("SPDXRef-Model", "DERIVED_FROM", "SPDXRef-Base") in rels  # unchanged

    def test_control_no_data_sha_no_data_relationship(self) -> None:
        rels = build_spdx_bom(_entry(None, data_sha=None))["relationships"]
        assert [r["relationshipType"] for r in rels] == ["DERIVED_FROM"]


# ---------------------------------------------------------------------------
# 5. SR 11-7 receipt: driver_version on a CUDA host
# ---------------------------------------------------------------------------

class TestReceiptDriverVersion:
    def test_cuda_host_records_the_driver_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        from soup_cli.bench import train_run

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "NVIDIA test GPU")
        # one level down, at the bench query the issue points to, so the wrapper's own
        # body runs (review of #1569: stubbing the wrapper let `return None` through)
        monkeypatch.setattr(train_run, "_driver_version", lambda: "550.90.07")
        receipt = receipt_to_dict(build_repro_receipt({"torch": 0}, "r1"))
        assert receipt["accelerator_backend"] == "cuda"
        assert receipt["gpu_models"] == ["NVIDIA test GPU"]
        assert receipt["driver_version"] == "550.90.07"

    def test_control_without_cuda_the_driver_query_is_not_made(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        from soup_cli.utils import repro_receipt

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        # recorded, not raised: the probe swallows exceptions, so an AssertionError
        # inside it could never fail this control (review of #1569)
        calls: list[int] = []
        monkeypatch.setattr(
            repro_receipt, "_driver_version", lambda: calls.append(1) or "550.90.07"
        )
        receipt = receipt_to_dict(build_repro_receipt({"torch": 0}, "r1"))
        assert calls == []
        assert receipt["driver_version"] is None
