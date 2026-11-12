"""The resident parity gate (#2104): Soup's native resident K2 load vs K2's own code.

Acceptance gate named in the issue: the resident route (``load_resident_model``) must
reproduce, on the tiny FP8 K2 checkpoint, the final-position logits K2's own published
remote code computes on the same weights. The fixtures
(``tests/fixtures/kimi_k2/resident_parity_fixtures.npz``) were recorded by
``benchmarks/harness/kimi_k2_resident_parity_record.py`` on a pinned old transformers in an
isolated venv, running ``modeling_deepseek.py`` at moonshotai/Kimi-K2-Base revision
``ce72df012259dcc55d945e890f815fe7ef69159c`` (sha256
``ce753fc9076ce13ce9557e680e2f5bd68ae8929a8225772968a9a22a1d17b7d4``); the manifest pins
every hash.

The npz holds two references, both from K2's own code on the same seeded inputs. The
bf16-source reference (``logits_i``) is the pristine comparison, and the bf16 cell below
loads the unquantised checkpoint through the resident route against it: transformers' own
native class on the current stack reproduces it to ~1e-7 (reported in
``benchmarks/gate-kimi-k2-resident.md``). The FP8 cell, though, loads the quantised
checkpoint: the e4m3 quantise→dequant round trip moves the final logits by ~1e-2 on this
tiny model — three orders of magnitude over this gate's tolerance — so it references the
second recording (``logits_fp8_i``), K2's own code on the dequantised FP8 round trip, the
exact weights the FP8 cell materialises. The round trip itself (per-block dequant
exactness) is gated bit-exactly by ``tests/test_fp8_block_streaming.py``, and the identity
of Soup's FP8 load with an independent dequantisation is pinned there; this gate therefore
measures the resident route against K2's own computation, not the quantisation error.

Mutation discipline (each of these production changes makes this file's parity test fail):
- bypassing or breaking the per-tensor FP8 dequant in ``load_resident_model``
  (``soup_cli/utils/kimi_k2.py``) — the resident load then carries quantisation noise the
  fixtures do not have;
- dropping ``routed_scaling_factor`` (or any other K2 field) from the native config
  mapping — the router's expert combine lands on a different scale;
- loading the rope with the wrong theta (the published 50000.0 vs the class default) or
  the wrong yarn parameters — the controls below prove the logits move on exactly these.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests.kimi_k2_tiny import build_tiny_k2_checkpoints
from tests.test_kimi_k2_streaming import _refuse_remote_code

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "kimi_k2"
FIXTURES_NPZ = FIXTURES_DIR / "resident_parity_fixtures.npz"
MANIFEST = FIXTURES_DIR / "manifest.json"

#: The admission gate's pinned sha256 for modeling_deepseek.py; the manifest must record it.
MODELING_SHA256 = "ce753fc9076ce13ce9557e680e2f5bd68ae8929a8225772968a9a22a1d17b7d4"
#: Comparisons run at fp32 on CPU. Bit-equality is not demanded across two implementations
#: of the same architecture; the tolerance below is pinned and a failing check reports its
#: maximum absolute difference.
RTOL = 1e-5
ATOL = 1e-6


def _fixtures():
    if not FIXTURES_NPZ.exists() or not MANIFEST.exists():
        pytest.skip(
            "resident parity fixtures missing; record them with: python "
            "benchmarks/harness/kimi_k2_resident_parity_record.py"
        )
    with np.load(FIXTURES_NPZ) as fixtures:
        payload = {name: fixtures[name] for name in fixtures.files}
    return payload, json.loads(MANIFEST.read_text(encoding="utf-8"))


def _importorskip_stack():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("numpy")
    return torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_resident(fp8_dir: str, torch):
    from soup_cli.utils.kimi_k2 import load_resident_model

    model = load_resident_model(str(fp8_dir), model_kwargs={"dtype": torch.float32})
    return model.eval()


def _ids_tensor(torch, ids):
    # The npz stores each input as a flat 1-D id sequence; forward needs batch 1.
    return torch.as_tensor(np.asarray(ids), dtype=torch.int64).reshape(1, -1)


def _final_logits(model, torch, ids):
    with torch.no_grad():
        output = model(input_ids=_ids_tensor(torch, ids))
    return output.logits[0, -1].to(torch.float32)


def test_the_fixtures_match_their_manifest():
    torch = _importorskip_stack()
    fixtures, _manifest = _fixtures()  # skips when absent
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["modeling_deepseek.py"]["sha256"] == MODELING_SHA256
    assert manifest["fixtures"]["sha256"] == _sha256(FIXTURES_NPZ)
    # The manifest mirrors the exact ids; the npz must agree, and both reference sets
    # (pristine bf16 source; the FP8 round trip the gate's FP8 cell references) must be
    # present and must DIFFER — identical arrays would mean the round trip was never run.
    with np.load(FIXTURES_NPZ) as npz:
        for index, ids in enumerate(manifest["inputs"]["ids"]):
            assert npz[f"ids_{index}"].tolist() == ids
            assert npz[f"logits_{index}"].shape == npz[f"logits_fp8_{index}"].shape
        assert sorted(npz.files) == sorted(manifest["fixtures"]["entries"])
    for index, ids in enumerate(manifest["inputs"]["ids"]):
        pristine = torch.as_tensor(fixtures[f"logits_{index}"]).to(torch.float32)
        round_tripped = torch.as_tensor(fixtures[f"logits_fp8_{index}"]).to(torch.float32)
        diff = float((pristine - round_tripped).abs().max())
        assert not torch.allclose(pristine, round_tripped, rtol=RTOL, atol=ATOL), (
            f"input {index}: the bf16-source and FP8 references agree to the gate "
            f"tolerance (max abs diff {diff}); the FP8 reference carries no round trip"
        )


def test_the_resident_fp8_load_matches_k2s_own_remote_code(tmp_path, monkeypatch):
    """The gate cell: load_resident_model on the FP8 tiny checkpoint vs the fixtures."""
    torch = _importorskip_stack()
    fixtures, manifest = _fixtures()
    _refuse_remote_code(monkeypatch)

    built = build_tiny_k2_checkpoints(tmp_path / "checkpoints")
    model = _load_resident(built["fp8"][0], torch)
    assert manifest["transformers_pin"], "manifest must record the validated transformers pin"
    for index, ids in enumerate(manifest["inputs"]["ids"]):
        got = _final_logits(model, torch, ids)
        want = torch.as_tensor(fixtures[f"logits_fp8_{index}"]).to(torch.float32)
        assert got.shape == want.shape
        assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
            f"input {index}: max abs diff {float((got - want).abs().max())} "
            f"exceeds rtol={RTOL} atol={ATOL}"
        )
        want_loss = float(fixtures[f"loss_fp8_{index}"])
        got_loss = _loss(model, torch, ids)
        assert abs(got_loss - want_loss) <= ATOL + RTOL * abs(want_loss), (
            f"input {index}: loss {got_loss} vs fixture {want_loss}"
        )


def test_the_resident_bf16_load_matches_k2s_own_remote_code(tmp_path, monkeypatch):
    """The pristine chain: load_resident_model on the unquantised bf16 tiny checkpoint
    (stale ``inv_freq`` keys and all) against K2's own code on the same checkpoint
    (``logits_i``/``loss_i``), at the same pinned tolerance. Without this cell the
    pristine fixtures would be checked for integrity and difference only — deleting them
    would break nothing that loads them."""
    torch = _importorskip_stack()
    fixtures, manifest = _fixtures()
    _refuse_remote_code(monkeypatch)

    built = build_tiny_k2_checkpoints(tmp_path / "checkpoints")
    model = _load_resident(built["bf16"][0], torch)
    for index, ids in enumerate(manifest["inputs"]["ids"]):
        got = _final_logits(model, torch, ids)
        want = torch.as_tensor(fixtures[f"logits_{index}"]).to(torch.float32)
        assert got.shape == want.shape
        assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
            f"input {index}: max abs diff {float((got - want).abs().max())} "
            f"exceeds rtol={RTOL} atol={ATOL}"
        )
        want_loss = float(fixtures[f"loss_{index}"])
        got_loss = _loss(model, torch, ids)
        assert abs(got_loss - want_loss) <= ATOL + RTOL * abs(want_loss), (
            f"input {index}: loss {got_loss} vs fixture {want_loss}"
        )


def _loss(model, torch, ids):
    tensor = _ids_tensor(torch, ids)
    with torch.no_grad():
        return float(model(input_ids=tensor, labels=tensor).loss)


@pytest.mark.parametrize("control", ["default-rope", "perturbed-router-bias"])
def test_the_gate_comparison_is_live(tmp_path, monkeypatch, control):
    """Controls: each perturbation must move the logits well outside the gate tolerance,
    or the parity cell above could be passing on an insensitive comparison."""
    torch = _importorskip_stack()
    fixtures, manifest = _fixtures()
    _refuse_remote_code(monkeypatch)

    built = build_tiny_k2_checkpoints(tmp_path / "checkpoints")
    if control == "default-rope":
        folder = tmp_path / "default-rope"
        shutil.copytree(built["fp8"][0], folder)
        config_path = folder / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["rope_scaling"] = None  # admitted by the guard; computes the default rope
        config_path.write_text(json.dumps(config), encoding="utf-8")
        model = _load_resident(str(folder), torch)
    else:
        model = _load_resident(built["fp8"][0], torch)
        # Zeroing the correction bias changes every router score; a small constant shift
        # would not do (sigmoid saturation leaves the top-k and weights unchanged, which
        # this control must not rely on).
        with torch.no_grad():
            for layer in model.model.layers[1:]:
                layer.mlp.gate.e_score_correction_bias.zero_()
    for index, ids in enumerate(manifest["inputs"]["ids"]):
        got = _final_logits(model, torch, ids)
        want = torch.as_tensor(fixtures[f"logits_fp8_{index}"]).to(torch.float32)
        diff = float((got - want).abs().max())
        assert not torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
            f"control {control}: input {index} still within tolerance (max abs diff {diff})"
        )
