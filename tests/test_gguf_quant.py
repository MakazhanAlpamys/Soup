"""``export_advanced_gguf`` treats a flavour the same however it is spelled.

``--gguf-flavour`` is validated case-insensitively, so ``iq2_m`` is accepted as
``IQ2_M``. The pipeline after the check must then run as it does for ``IQ2_M``:
the importance matrix built from ``--calibration-data``, the refusal without it,
and the ``UD-`` prefix stripped before llama-quantize. Only the external tools are
faked here; every Soup function in the pipeline runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from soup_cli.utils import gguf_quant
from soup_cli.utils.gguf_quant import export_advanced_gguf


@pytest.fixture
def toolchain(tmp_path, monkeypatch):
    """A model dir, a llama.cpp dir with its three tools, and the calls they get."""
    monkeypatch.chdir(tmp_path)
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

    def run(argv, **kwargs):
        argv = [str(a) for a in argv]
        name = Path(argv[0]).name
        if name.startswith("llama-imatrix"):
            tool, out = "imatrix", argv[argv.index("-o") + 1]
        elif name.startswith("llama-quantize"):
            tool, out = "quantize", argv[-2]
        else:  # python convert_hf_to_gguf.py ...
            tool, out = "convert", argv[argv.index("--outfile") + 1]
        calls.append((tool, argv))
        Path(out).write_bytes(b"GGUF")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(gguf_quant.subprocess, "run", run)
    return {"model": model, "llama": llama, "calib": calib, "calls": calls, "dir": tmp_path}


def _export(tc, flavour, *, calibration=True):
    out = tc["dir"] / "out.gguf"
    export_advanced_gguf(
        model_dir=str(tc["model"]),
        output_path=str(out),
        flavour=flavour,
        calibration_data=str(tc["calib"]) if calibration else None,
        llama_cpp_dir=str(tc["llama"]),
    )
    assert out.is_file()
    return dict(tc["calls"])


@pytest.mark.parametrize(
    ("spelling", "quant_type"),
    [
        ("IQ2_M", "IQ2_M"),
        ("iq2_m", "IQ2_M"),
        ("Iq3_xxs", "IQ3_XXS"),
        ("iq3_xs", "IQ3_XS"),
        ("ud-iq2_m", "IQ2_M"),
        ("Ud-Iq1_M", "IQ1_M"),
    ],
)
def test_imatrix_flavour_builds_and_uses_the_importance_matrix(toolchain, spelling, quant_type):
    calls = _export(toolchain, spelling)

    assert "imatrix" in calls, f"{spelling!r} skipped llama-imatrix"
    quantize = calls["quantize"]
    assert "--imatrix" in quantize
    assert quantize[quantize.index("--imatrix") + 1] == calls["imatrix"][
        calls["imatrix"].index("-o") + 1
    ]
    assert quantize[-1] == quant_type


@pytest.mark.parametrize("spelling", ["iq2_m", "iq3_xxs", "iq1_s", "ud-q4_k_xl", "ud-iq3_m"])
def test_imatrix_flavour_without_calibration_data_is_refused(toolchain, spelling):
    with pytest.raises(ValueError, match="requires --calibration-data"):
        _export(toolchain, spelling, calibration=False)
    assert toolchain["calls"] == []


@pytest.mark.parametrize(("spelling", "quant_type"), [("Q5_K_M", "Q5_K_M"), ("q5_k_m", "Q5_K_M")])
def test_flavour_without_imatrix_still_quantizes_directly(toolchain, spelling, quant_type):
    calls = _export(toolchain, spelling, calibration=False)

    assert "imatrix" not in calls
    assert "--imatrix" not in calls["quantize"]
    assert calls["quantize"][-1] == quant_type
