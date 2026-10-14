"""CPU checks for the decision rule of benchmarks/harness/lora_hook_parity.py.

The harness turns logits into verdicts (APPLIED, DROPPED, WRONG, ...). These
tests plant engine outputs with a known relation to the reference and check
that the rule in probe-lora-hook-engines.md §2 names them correctly.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness" / "lora_hook_parity.py"


@pytest.fixture(scope="module")
def harness():
    name = "_lora_hook_parity"
    previous = sys.modules.get(name)
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


@pytest.fixture()
def logits():
    """Reference logits without and with an adapter that moves them by ~10%."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(12, 50)).astype(np.float32)
    effect = 0.1 * rng.normal(size=base.shape).astype(np.float32)
    return base, base + effect


def _verdict_for(harness, ref0, ref1, eng0, eng1):
    metrics = harness.effect_metrics(ref0, ref1, eng0, eng1)
    return harness.verdict(
        base_ok=harness.base_error(eng0, ref0) <= harness.E_BASE_MAX,
        s_ref=harness.effect_size(ref0, ref1),
        convert_ok=True,
        load_ok=True,
        r=metrics["r"],
        rho=metrics["rho"],
    )


def test_engine_reproducing_the_effect_is_applied(harness, logits):
    ref0, ref1 = logits
    assert _verdict_for(harness, ref0, ref1, ref0.copy(), ref1.copy()) == "APPLIED"


def test_engine_ignoring_the_adapter_is_dropped(harness, logits):
    ref0, ref1 = logits
    assert _verdict_for(harness, ref0, ref1, ref0.copy(), ref0.copy()) == "DROPPED"


def test_engine_scaling_the_adapter_wrongly_is_wrong(harness, logits):
    ref0, ref1 = logits
    eng1 = ref0 + 1.1 * (ref1 - ref0)
    metrics = harness.effect_metrics(ref0, ref1, ref0, eng1)
    assert metrics["r"] == pytest.approx(0.1, rel=1e-4)
    assert _verdict_for(harness, ref0, ref1, ref0, eng1) == "WRONG"


def test_partially_applied_adapter_is_wrong_not_dropped(harness, logits):
    ref0, ref1 = logits
    eng1 = ref0 + 0.5 * (ref1 - ref0)
    assert _verdict_for(harness, ref0, ref1, ref0, eng1) == "WRONG"


def test_effect_is_measured_against_the_engines_own_base(harness, logits):
    """A tiny base offset shared by both engine runs must not count against the adapter."""
    ref0, ref1 = logits
    offset = np.float32(1e-5)
    assert _verdict_for(harness, ref0, ref1, ref0 + offset, ref1 + offset) == "APPLIED"


def test_base_mismatch_voids_even_a_perfect_adapter(harness, logits):
    ref0, ref1 = logits
    shift = 2e-3 * np.abs(ref0).max()
    assert _verdict_for(harness, ref0, ref1, ref0 + shift, ref1 + shift) == "VOID"


def test_rule_rows_apply_in_order(harness):
    # A weak adapter is reported as such before any engine outcome.
    assert harness.verdict(base_ok=True, s_ref=1e-3, convert_ok=False, load_ok=None) == "TOO WEAK"
    assert harness.verdict(base_ok=True, s_ref=0.5, convert_ok=False, load_ok=None) == (
        "CONVERT-FAILED"
    )
    assert harness.verdict(base_ok=True, s_ref=0.5, convert_ok=True, load_ok=False) == (
        "LOAD-FAILED"
    )
    # The APPLIED band includes its boundary; DROPPED is checked only when not APPLIED.
    boundary = harness.R_APPLIED_MAX
    assert harness.verdict(
        base_ok=True, s_ref=0.5, convert_ok=True, load_ok=True, r=boundary, rho=1.0
    ) == "APPLIED"
    assert harness.verdict(
        base_ok=True, s_ref=0.5, convert_ok=True, load_ok=True, r=1.0, rho=0.0
    ) == "DROPPED"


def _models(qwen_verdicts, dsv3_verdicts):
    def wrap(verdicts):
        return {"variants": {name: {
            "verdict": value,
            "reference_disabled_is_base": True,
            "touches_routed_experts": False,
            "reference_deterministic": True,
            "engine_deterministic": True,
            "tokens_match": True,
        } for name, value in verdicts.items()}}

    return {"qwen35moe-tiny": wrap(qwen_verdicts), "dsv3-tiny": wrap(dsv3_verdicts)}


DSV3_SINGLES = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "shared")
CONTROL_OK = {"q_proj": "APPLIED", "v_proj": "APPLIED", "shared": "APPLIED"}


def test_failed_positive_control_gives_no_verdict(harness):
    qwen = dict(CONTROL_OK, v_proj="DROPPED")
    dsv3 = {name: "APPLIED" for name in DSV3_SINGLES}
    assert harness.probe_verdict(_models(qwen, dsv3))["verdict"] == "NO VERDICT"


def test_hook_lists_exactly_the_modules_not_applied(harness):
    dsv3 = {name: "APPLIED" for name in DSV3_SINGLES}
    dsv3.update(q_a_proj="DROPPED", kv_b_proj="CONVERT-FAILED")
    outcome = harness.probe_verdict(_models(CONTROL_OK, dsv3))
    assert outcome["verdict"] == "HOOK NEEDED"
    assert outcome["hook_modules"] == {"q_a_proj": "DROPPED", "kv_b_proj": "CONVERT-FAILED"}


def test_all_modules_applied_needs_no_hook(harness):
    dsv3 = {name: "APPLIED" for name in DSV3_SINGLES}
    assert harness.probe_verdict(_models(CONTROL_OK, dsv3))["verdict"] == "NO HOOK NEEDED"


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_disabled_is_base", False),
        ("reference_disabled_is_base", None),
        ("touches_routed_experts", True),
        ("touches_routed_experts", None),
        ("reference_deterministic", False),
        ("reference_deterministic", None),
        ("engine_deterministic", False),
        ("engine_deterministic", None),
        ("tokens_match", False),
        ("verdict", "VOID"),
    ],
)
def test_failed_adapter_control_invalidates_probe(harness, field, value):
    models = _models(CONTROL_OK, {name: "APPLIED" for name in DSV3_SINGLES})
    models["dsv3-tiny"]["variants"]["q_a_proj"][field] = value
    assert harness.probe_verdict(models)["verdict"] == "NO VERDICT"


def test_bitwise_determinism_rejects_signed_zero_change(harness):
    left = np.array([[0.0, 1.0]], dtype=np.float32)
    right = np.array([[-0.0, 1.0]], dtype=np.float32)
    assert harness.bitwise_equal(left, left.copy())
    assert not harness.bitwise_equal(left, right)
    assert not harness.bitwise_equal(left, left.reshape(-1))
    assert not harness.bitwise_equal(left, left.astype(np.float64))


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
    ok = harness.Step(True, 0, "")
    runs = iter([
        (ok, engine_tokens, eng1),
        (harness.Step(False, 1, "repeat failed"), None, None)
        if engine_fault == "repeat-failed" else (ok, engine_tokens.copy(), repeat),
    ])
    monkeypatch.setattr(
        harness, "make_adapter", lambda *args: ["model.layers.0.self_attn.q_proj"],
    )
    monkeypatch.setattr(harness, "reference_logits", lambda *args: (ref1.copy(), ref0.copy()))
    monkeypatch.setattr(harness, "convert_adapter", lambda *args: ok)
    monkeypatch.setattr(harness, "engine_logits", lambda *args: next(runs))
    ctx = harness.Context(tmp_path / "engine", tmp_path, sys.executable, tmp_path, "", 1)
    base = harness.Base(tmp_path / "model", tmp_path / "base.gguf", tokens, ref0, ref0.copy())

    result = harness.run_variant(ctx, base, "q_proj", ["q_proj"], 17)

    assert result["verdict"] == expected
    if expected == "APPLIED":
        assert result["r"] == pytest.approx(0.0)
        assert result["rho"] == pytest.approx(1.0)
    else:
        assert "r" not in result
        assert "rho" not in result
    if effect == 0.0:
        assert result["s_ref"] == 0.0
    if engine_fault == "token-count":
        assert result["tokens_match"] is False
    if engine_fault == "repeat-failed":
        assert result["engine_deterministic"] is False
        assert result["engine_repeat"]["returncode"] == 1
