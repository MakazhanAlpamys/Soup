"""CPU checks for Part B's rule in benchmarks/harness/lora_hook_real_model.py.

The adapter-effect error is compared with three base gaps in effect units,
plus 0.02 slack, in bf16 or f32. These tests plant engine outputs with a
known deviation and a known effect and check the verdicts.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness" / "lora_hook_real_model.py"


@pytest.fixture(scope="module")
def harness():
    original_path = sys.path[:]
    names = ("_lora_hook_real_model", "lora_hook_parity")
    original_modules = {name: sys.modules.get(name) for name in names}
    spec = importlib.util.spec_from_file_location(names[0], HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        for name, previous in original_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _case(noise_scale, effect_scale=0.1, engine_effect=1.0, seed=0):
    """The engine's deviation from the reference is deterministic: a fixed base gap, plus a
    further deviation of the same size once the adapter is applied. A dropped adapter computes
    the base again, so its logits equal the engine's base logits exactly."""
    rng = np.random.default_rng(seed)
    ref0 = rng.normal(size=(16, 64))
    effect = effect_scale * rng.normal(size=ref0.shape)
    eng0 = ref0 + noise_scale * rng.normal(size=ref0.shape)
    if engine_effect == 0.0:
        return ref0, ref0 + effect, eng0, eng0.copy()
    eng1 = eng0 + engine_effect * effect + noise_scale * rng.normal(size=ref0.shape)
    return ref0, ref0 + effect, eng0, eng1


def _verdict(harness, ref0, ref1, eng0, eng1):
    metrics = harness.parity.effect_metrics(ref0, ref1, eng0, eng1)
    return harness.verdict_b(
        base_ok=True,
        s_ref=harness.parity.effect_size(ref0, ref1),
        convert_ok=True,
        load_ok=True,
        r=metrics["r"],
        rho=metrics["rho"],
        floor=harness.noise_floor(eng0, ref0, ref1),
    )


def test_noise_floor_is_base_gap_over_effect(harness):
    ref0 = np.zeros((2, 3))
    ref1 = ref0 + 2.0
    eng0 = ref0 + 0.5
    assert harness.noise_floor(eng0, ref0, ref1) == pytest.approx(0.25)


def test_effect_reproduced_under_base_level_noise_is_applied(harness):
    assert _verdict(harness, *_case(noise_scale=0.01)) == "APPLIED"


def test_dropped_adapter_is_dropped(harness):
    assert _verdict(harness, *_case(noise_scale=0.01, engine_effect=0.0)) == "DROPPED"


def test_half_applied_adapter_is_wrong(harness):
    assert _verdict(harness, *_case(noise_scale=0.001, engine_effect=0.5)) == "WRONG"


def test_base_gap_too_large_for_the_effect_gives_no_verdict(harness):
    """Otherwise the APPLIED band would admit even a dropped adapter (r = 1)."""
    assert _verdict(harness, *_case(noise_scale=0.05, engine_effect=0.0)) == "TOO NOISY"


def test_base_disagreement_voids_before_anything_else(harness):
    verdict = harness.verdict_b(base_ok=False, s_ref=0.5, convert_ok=False, load_ok=None)
    assert verdict == "VOID"


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_disabled_is_base", False),
        ("touches_routed_experts", True),
        ("reference_deterministic", False),
        ("reference_deterministic", None),
    ],
)
def test_invalid_reference_stops_before_export(harness, monkeypatch, tmp_path, field, value):
    row = {
        "adapter_dir": tmp_path / "adapter",
        "z_ref1": np.ones((2, 3)),
        "reference_disabled_is_base": True,
        "touches_routed_experts": False,
        "reference_deterministic": True,
        "s_ref": 0.5,
    }
    row[field] = value

    def refuse_export(*args):
        pytest.fail("invalid reference must not reach the external converter")

    monkeypatch.setattr(harness.parity, "convert_adapter", refuse_export)
    result = harness.engine_variant(
        None, tmp_path, tmp_path / "base.gguf", "control", row, {}, np.zeros((2, 3)), True
    )
    assert result["verdict"] == "VOID"


@pytest.mark.parametrize(
    "engine_fault,effect,expected",
    [
        (None, 0.1, "APPLIED"),
        ("token-count", 0.1, "VOID"),
        ("token-ids", 0.1, "VOID"),
        ("repeat-logits", 0.1, "VOID"),
        ("repeat-failed", 0.1, "VOID"),
        (None, 0.0, "TOO WEAK"),
        ("repeat-failed", 0.0, "VOID"),
    ],
)
def test_variant_controls_precede_effect_metrics(
    harness, monkeypatch, tmp_path, engine_fault, effect, expected,
):
    parity = harness.parity
    ref0 = np.ones((2, 3), dtype=np.float32)
    ref1 = ref0 + effect
    tokens = np.array([0, 1])
    engine_tokens, eng1 = tokens.copy(), ref1.copy()
    if engine_fault == "token-count":
        engine_tokens = np.array([0, 1, 2])
        eng1 = np.vstack((ref1, ref1[:1]))
    elif engine_fault == "token-ids":
        engine_tokens = np.array([1, 0])
    repeat = eng1 + 0.1 if engine_fault == "repeat-logits" else eng1.copy()
    ok = parity.Step(True, 0, "")
    runs = iter([
        (ok, engine_tokens, eng1),
        (parity.Step(False, 1, "repeat failed"), None, None)
        if engine_fault == "repeat-failed" else (ok, engine_tokens.copy(), repeat),
    ])
    monkeypatch.setattr(parity, "convert_adapter", lambda *args: ok)
    monkeypatch.setattr(parity, "engine_logits", lambda *args: next(runs))
    ctx = parity.Context(tmp_path / "engine", tmp_path, sys.executable, tmp_path, "", 1)
    row = {
        "adapter_dir": tmp_path / "adapter",
        "z_ref1": ref1,
        "reference_disabled_is_base": True,
        "touches_routed_experts": False,
        "reference_deterministic": True,
        "s_ref": parity.effect_size(ref0, ref1),
        "digests": {},
    }

    result = harness.engine_variant(
        ctx, tmp_path / "model", tmp_path / "base.gguf", "q_proj", row,
        {"tokens": tokens, "z_eng0": ref0.copy()}, ref0, True,
    )

    assert result["verdict"] == expected
    if expected == "APPLIED":
        assert result["r"] == pytest.approx(0.0)
        assert result["rho"] == pytest.approx(1.0)
        assert result["floor"] == pytest.approx(0.0)
    else:
        assert not {"r", "rho", "floor", "t"}.intersection(result)
    if engine_fault == "token-count":
        assert result["tokens_match"] is False
    if engine_fault == "repeat-failed":
        assert result["engine_deterministic"] is False
        assert result["engine_repeat"]["returncode"] == 1
