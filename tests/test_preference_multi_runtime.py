"""v0.40.1 Part B — Multi-objective preference live runtime tests.

Closes the v0.40.0 Part D NotImplementedError stub. The wrapper now
combines 2-5 preference losses on the same forward pass via
:mod:`soup_cli.utils.preference_combine`.
"""

from __future__ import annotations

import math

import pytest

from soup_cli.utils import preference_combine as pc

# ---------------------------------------------------------------------------
# Pure-function math tests (no torch surrogate; we use real torch tensors)
# ---------------------------------------------------------------------------


@pytest.fixture
def torch_module():
    torch = pytest.importorskip("torch")
    return torch


def test_combine_losses_single_loss_passthrough(torch_module):
    torch = torch_module
    losses = {"dpo": torch.tensor(0.42)}
    out = pc.combine_losses(losses, {"dpo": 1.0})
    assert math.isclose(out.item(), 0.42, abs_tol=1e-6)


def test_combine_losses_two_losses_weighted_average(torch_module):
    torch = torch_module
    losses = {"dpo": torch.tensor(1.0), "simpo": torch.tensor(2.0)}
    out = pc.combine_losses(losses, {"dpo": 0.25, "simpo": 0.75})
    # 0.25*1 + 0.75*2 = 1.75
    assert math.isclose(out.item(), 1.75, abs_tol=1e-6)


def test_combine_losses_rejects_key_mismatch(torch_module):
    torch = torch_module
    losses = {"dpo": torch.tensor(1.0)}
    with pytest.raises(ValueError, match="loss keys"):
        pc.combine_losses(losses, {"simpo": 1.0})


def test_combine_losses_rejects_extra_loss_key(torch_module):
    """Inverse of key_mismatch — losses key absent from weights."""
    torch = torch_module
    losses = {"dpo": torch.tensor(1.0), "extra": torch.tensor(0.5)}
    with pytest.raises(ValueError, match="loss keys"):
        pc.combine_losses(losses, {"dpo": 1.0})


def test_combine_losses_rejects_empty_weights(torch_module):
    """Empty weights mapping must raise (defence-in-depth — schema enforces 2-5)."""
    torch_module  # noqa
    with pytest.raises(ValueError, match="empty"):
        pc.combine_losses({}, {})


def test_combine_losses_propagates_nan_loudly(torch_module):
    """A NaN tensor input should produce a NaN combined loss (caller handles
    the recovery via loss_watchdog) — but never silently zero out."""
    torch = torch_module
    losses = {
        "dpo": torch.tensor(float("nan")),
        "simpo": torch.tensor(1.0),
    }
    out = pc.combine_losses(losses, {"dpo": 0.5, "simpo": 0.5})
    # NaN must propagate — not get mistakenly zeroed by `0.0 * NaN`-style arithmetic.
    assert torch.isnan(out).item(), "combine_losses silently swallowed NaN"


def test_combine_losses_rejects_bool_weight(torch_module):
    """Defence-in-depth: bool is a subclass of int but the project policy
    rejects bool weights (matches v0.30.0 Candidate / v0.34.0 cost policy)."""
    torch = torch_module
    losses = {"dpo": torch.tensor(1.0), "simpo": torch.tensor(2.0)}
    with pytest.raises(TypeError, match="bool"):
        pc.combine_losses(losses, {"dpo": True, "simpo": False})


def test_combine_losses_rejects_unnormalised_weights(torch_module):
    torch = torch_module
    losses = {"dpo": torch.tensor(1.0), "simpo": torch.tensor(2.0)}
    with pytest.raises(ValueError, match="sum to 1"):
        pc.combine_losses(losses, {"dpo": 0.5, "simpo": 0.4})


def test_combine_losses_propagates_gradients(torch_module):
    torch = torch_module
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    losses = {"dpo": a * 1.0, "ipo": b * 1.0}
    out = pc.combine_losses(losses, {"dpo": 0.5, "ipo": 0.5})
    out.backward()
    # d/da (0.5 a) = 0.5; d/db (0.5 b) = 0.5
    assert math.isclose(a.grad.item(), 0.5, abs_tol=1e-6)
    assert math.isclose(b.grad.item(), 0.5, abs_tol=1e-6)


def test_dpo_term_matches_known_formula(torch_module):
    torch = torch_module
    pol_chosen = torch.tensor([1.0])
    pol_rejected = torch.tensor([0.0])
    ref_chosen = torch.tensor([0.5])
    ref_rejected = torch.tensor([0.0])
    beta = 0.1
    out = pc.compute_dpo_term(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta)
    # logits = 0.1 * ((1-0) - (0.5-0)) = 0.1 * 0.5 = 0.05
    # loss = -logσ(0.05) ≈ 0.6682
    expected = -math.log(1 / (1 + math.exp(-0.05)))
    assert math.isclose(out.item(), expected, abs_tol=1e-4)


def test_dpo_term_requires_reference_logps(torch_module):
    torch = torch_module
    pol_chosen = torch.tensor([1.0])
    pol_rejected = torch.tensor([0.0])
    with pytest.raises(ValueError, match="reference"):
        pc.compute_dpo_term(pol_chosen, pol_rejected, None, None, 0.1)


def test_ipo_term_rejects_non_positive_beta(torch_module):
    torch = torch_module
    z = torch.tensor([0.0])
    with pytest.raises(ValueError, match="beta"):
        pc.compute_ipo_term(z, z, z, z, 0.0)


def test_simpo_term_length_normalised(torch_module):
    torch = torch_module
    pol_chosen = torch.tensor([2.0])
    pol_rejected = torch.tensor([2.0])
    chosen_lens = torch.tensor([1])
    rejected_lens = torch.tensor([2])
    out = pc.compute_simpo_term(
        pol_chosen, pol_rejected, beta=1.0, gamma=0.0,
        chosen_lens=chosen_lens, rejected_lens=rejected_lens,
    )
    # Normalised: chosen=2/1=2, rejected=2/2=1; logits=1*(2-1)-0=1
    expected = -math.log(1 / (1 + math.exp(-1.0)))
    assert math.isclose(out.item(), expected, abs_tol=1e-4)


def test_describe_blend_format():
    out = pc.describe_blend({"dpo": 0.6, "simpo": 0.4})
    # Sorted alphabetically.
    assert out == "0.60·dpo + 0.40·simpo"


def test_describe_blend_empty():
    assert pc.describe_blend(None) == "(none)"
    assert pc.describe_blend({}) == "(none)"


# ---------------------------------------------------------------------------
# Compatibility validation
# ---------------------------------------------------------------------------


def test_validate_weight_compat_paired_only_ok():
    pc.validate_weight_compat({"dpo": 0.5, "simpo": 0.5})  # no raise


def test_validate_weight_compat_bco_alone_ok():
    pc.validate_weight_compat({"bco": 1.0})  # no raise — schema already filters


def test_validate_weight_compat_bco_mixed_with_paired_rejected():
    with pytest.raises(ValueError, match="bco"):
        pc.validate_weight_compat({"bco": 0.5, "dpo": 0.5})


def test_needs_reference_model_dpo():
    assert pc.needs_reference_model({"dpo": 1.0}) is True


def test_needs_reference_model_simpo_orpo_no_ref():
    assert pc.needs_reference_model({"simpo": 0.5, "orpo": 0.5}) is False


def test_needs_reference_model_mixed():
    assert pc.needs_reference_model({"dpo": 0.5, "simpo": 0.5}) is True


# ---------------------------------------------------------------------------
# PreferenceTrainerWrapper integration — runtime path no longer raises
# ---------------------------------------------------------------------------


def _make_multi_objective_cfg(weights):
    from soup_cli.config.schema import (
        DataConfig,
        SoupConfig,
        TrainingConfig,
    )

    return SoupConfig(
        base="HuggingFaceTB/SmolLM2-135M-Instruct",
        task="preference",
        data=DataConfig(train="data.jsonl", format="dpo"),
        training=TrainingConfig(preference_loss_weights=weights),
        output="./out",
    )


def test_wrapper_setup_no_longer_raises_for_paired_blend(tmp_path, monkeypatch):
    """The v0.40.0 NotImplementedError stub is gone for paired blends."""
    from soup_cli.trainer.preference import PreferenceTrainerWrapper

    cfg = _make_multi_objective_cfg({"dpo": 0.6, "simpo": 0.4})
    wrapper = PreferenceTrainerWrapper(cfg, device="cpu")

    # We bypass real model loading by intercepting _build_multi_objective.
    # The point is that setup() no longer aborts at the stub-then-live gate.
    called = {"build": False}

    def _stub_build():
        called["build"] = True
        return None  # No real inner trainer — just exercise the gate.

    monkeypatch.setattr(wrapper, "_build_multi_objective", _stub_build)
    wrapper.setup({"train": []})
    assert called["build"], "wrapper did not enter multi-objective path"


def test_wrapper_setup_rejects_bco_mixed_with_paired(monkeypatch):
    from soup_cli.trainer.preference import PreferenceTrainerWrapper

    cfg = _make_multi_objective_cfg({"bco": 0.5, "dpo": 0.5})
    wrapper = PreferenceTrainerWrapper(cfg, device="cpu")
    with pytest.raises(ValueError, match="bco"):
        wrapper.setup({"train": []})


def test_wrapper_describes_active_blend():
    """The advisory message is owned by ``_build_multi_objective``; verify
    the helper's source contains the user-facing blend description."""
    import inspect

    from soup_cli.trainer.preference import PreferenceTrainerWrapper

    src = inspect.getsource(PreferenceTrainerWrapper._build_multi_objective)
    assert "describe_blend" in src
    assert "primary" in src.lower()
