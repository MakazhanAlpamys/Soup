"""``soup export --format gguf-ud`` treats a flavour the same however it is spelled.

``--gguf-flavour`` is validated case-insensitively, so ``iq2_m`` is accepted as
``IQ2_M``. The pipeline after the check must then run as it does for ``IQ2_M``:
the importance matrix built from ``--calibration-data``, the refusal without it,
and the ``UD-`` prefix stripped before llama-quantize (#3102). Only the three
llama.cpp tools are faked here, at ``subprocess.run``; every Soup step runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import gguf_quant
from soup_cli.utils.gguf_quant import export_advanced_gguf
from tests.conftest import strip_ansi

CALIB_TEXT = "one sample\nanother\n"


@pytest.fixture
def toolchain(tmp_path, monkeypatch):
    """A model dir, a llama.cpp dir with its three tools, and the calls they get."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    llama = tmp_path / "llama"
    (llama / "build" / "bin").mkdir(parents=True)
    (llama / "convert_hf_to_gguf.py").write_text("# stub", encoding="utf-8")
    for tool in ("llama-quantize", "llama-imatrix"):
        (llama / "build" / "bin" / tool).write_text("# stub", encoding="utf-8")
    calib = tmp_path / "calib.jsonl"
    calib.write_text('{"text": "one sample"}\n{"text": "another"}\n', encoding="utf-8")

    calls: list[tuple[str, list[str]]] = []
    calib_seen: list[str] = []

    def run(argv, **kwargs):
        argv = [str(a) for a in argv]
        name = Path(argv[0]).name
        if name.startswith("llama-imatrix"):
            tool, out = "imatrix", argv[argv.index("-o") + 1]
            calib_seen.append(Path(argv[argv.index("-f") + 1]).read_text(encoding="utf-8"))
        elif name.startswith("llama-quantize"):
            tool, out = "quantize", argv[-2]
        else:  # python convert_hf_to_gguf.py ...
            tool, out = "convert", argv[argv.index("--outfile") + 1]
        calls.append((tool, argv))
        Path(out).write_bytes(b"GGUF")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(gguf_quant.subprocess, "run", run)
    return {
        "model": model, "llama": llama, "calib": calib, "calls": calls,
        "calib_seen": calib_seen, "dir": tmp_path,
    }


def _refusal(canonical: str) -> str:
    return (
        f"flavour {canonical!r} requires --calibration-data <jsonl>. "
        "UD ladder + low-bit IQ flavours need an importance matrix."
    )


def _spellings(name: str) -> list[str]:
    """``name`` as written, in lower case, and in two mixed cases.

    Every letter of the 36 names is upper case, so ``name.swapcase()`` is
    ``name.lower()``. ``capitalize()`` and its ``swapcase()`` give the mixed
    spellings (``Ud-iq2_m``, ``uD-IQ2_M``); a name with one letter has none.
    """
    mixed = name.capitalize()
    out: list[str] = []
    for spelling in (name, name.lower(), name.swapcase(), mixed, mixed.swapcase()):
        if spelling not in out:
            out.append(spelling)
    return out


_SPELLINGS = [
    (spelling, name)
    for name in sorted(gguf_quant.ALL_ADVANCED_GGUF_FORMATS)
    for spelling in _spellings(name)
]
# The Kelvin sign (U+212A) lower-cases to "k", so this is accepted as Q4_K_M.
_SPELLINGS.append(("Q4_\u212a_M", "Q4_K_M"))


@pytest.mark.parametrize("calibration", [True, False], ids=["calib", "no-calib"])
@pytest.mark.parametrize(("spelling", "canonical"), _SPELLINGS, ids=[s for s, _ in _SPELLINGS])
def test_every_spelling_runs_as_its_canonical_flavour(toolchain, spelling, canonical, calibration):
    needs_imatrix = canonical in gguf_quant._REQUIRES_IMATRIX
    out = toolchain["dir"] / "out.gguf"

    def export():
        export_advanced_gguf(
            model_dir=str(toolchain["model"]),
            output_path=str(out),
            flavour=spelling,
            calibration_data=str(toolchain["calib"]) if calibration else None,
            llama_cpp_dir=str(toolchain["llama"]),
        )

    if needs_imatrix and not calibration:
        with pytest.raises(ValueError) as exc:
            export()
        assert str(exc.value) == _refusal(canonical)
        assert toolchain["calls"] == []
        return

    export()
    assert out.is_file()
    calls = dict(toolchain["calls"])
    quantize = calls["quantize"]
    assert ("imatrix" in calls) is needs_imatrix
    assert ("--imatrix" in quantize) is needs_imatrix
    if needs_imatrix:
        imatrix = calls["imatrix"]
        assert quantize[quantize.index("--imatrix") + 1] == imatrix[imatrix.index("-o") + 1]
        assert toolchain["calib_seen"] == [CALIB_TEXT]
    assert quantize[-1] == canonical.removeprefix("UD-")


# --- through `soup export --format gguf-ud` ---------------------------------

runner = CliRunner()


def _soup_export(*args: str):
    result = runner.invoke(
        app, ["export", "--model", "model", "--format", "gguf-ud", "--llama-cpp", "llama", *args],
    )
    return result, " ".join(strip_ansi(result.output).split())


def test_cli_refuses_lower_case_imatrix_flavour_without_calibration_data(toolchain):
    result, out = _soup_export("--gguf-flavour", "iq3_xxs", "-o", "out.gguf")

    assert result.exit_code == 1, (out, repr(result.exception))
    assert out.endswith(f"Advanced GGUF export failed: {_refusal('IQ3_XXS')}"), out
    assert toolchain["calls"] == []
    assert not (toolchain["dir"] / "out.gguf").exists()


def test_cli_uses_calibration_data_for_lower_case_flavour(toolchain):
    result, out = _soup_export(
        "--gguf-flavour", "iq3_xs", "--calibration-data", "calib.jsonl", "-o", "out.gguf",
    )

    assert result.exit_code == 0, (out, repr(result.exception))
    assert [tool for tool, _ in toolchain["calls"]] == ["convert", "imatrix", "quantize"]
    calls = dict(toolchain["calls"])
    quantize = calls["quantize"]
    assert quantize[quantize.index("--imatrix") + 1] == calls["imatrix"][
        calls["imatrix"].index("-o") + 1
    ]
    assert quantize[-2:] == ["out.gguf", "IQ3_XS"]
    assert toolchain["calib_seen"] == [CALIB_TEXT]
    assert "Calib: calib.jsonl" in out


def test_cli_panel_and_default_output_name_keep_the_typed_spelling(toolchain):
    result, out = _soup_export("--gguf-flavour", "Ud-Iq2_M", "--calibration-data", "calib.jsonl")

    assert result.exit_code == 0, (out, repr(result.exception))
    assert (toolchain["dir"] / "model.Ud-Iq2_M.gguf").is_file()
    quantize = dict(toolchain["calls"])["quantize"]
    assert quantize[-2:] == ["model.Ud-Iq2_M.gguf", "IQ2_M"]
    assert out.count("Flavour: Ud-Iq2_M") == 2, out  # the start and the completion panel
    assert "Output: model.Ud-Iq2_M.gguf" in out
