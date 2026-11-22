"""``soup_cli.kernels.packed_lora.policy``: the per-run policy, rounding and int8 switch (CPU)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from soup_cli.kernels.packed_lora import policy
from soup_cli.kernels.packed_lora.config import PackedLoraConfig
from soup_cli.kernels.packed_lora.policy import Int8Switch, Policy, Rounding


def _config(**block):
    return PackedLoraConfig(enabled=True, **block)


class TestPolicyFromTheBlock:
    def test_defaults(self):
        pol = Policy.from_block(_config())
        assert pol == Policy(gpu_budget_bytes=None, activations="auto", int8_backward="auto",
                             int8_rounding="grad", on_unsupported="error")

    def test_every_field_is_carried(self):
        pol = Policy.from_block(_config(gpu_budget_mib=7000, activations="offload",
                                         int8_backward="stash", int8_rounding="stochastic",
                                         on_unsupported="stock"))
        assert pol.gpu_budget_bytes == 7000 * (1 << 20)
        assert (pol.activations, pol.int8_backward) == ("offload", "stash")
        assert (pol.int8_rounding, pol.on_unsupported) == ("stochastic", "stock")

    def test_the_config_defaults_and_the_policy_defaults_agree(self):
        block = PackedLoraConfig()
        assert Policy.from_block(block) == Policy()

    @pytest.mark.parametrize("kwargs,word", [
        (dict(activations="gpu"), "activations"),
        (dict(int8_backward="maybe"), "int8_backward"),
        (dict(int8_rounding="sloppy"), "int8_rounding"),
        (dict(on_unsupported="ignore"), "on_unsupported"),
        (dict(gpu_budget_bytes=0), "gpu_budget_bytes"),
        (dict(gpu_budget_bytes=True), "gpu_budget_bytes"),
        (dict(gpu_budget_bytes=1.5), "gpu_budget_bytes"),
    ])
    def test_a_policy_built_by_hand_is_validated(self, kwargs, word):
        with pytest.raises(ValueError, match=word):
            Policy(**kwargs)

    def test_the_policy_is_frozen(self):
        with pytest.raises(Exception):
            Policy().activations = "offload"  # type: ignore[misc]


class TestRounding:
    @pytest.mark.parametrize("mode,grad,weights", [
        ("nearest", False, False), ("grad", True, False), ("stochastic", True, True)])
    def test_modes(self, mode, grad, weights):
        rounding = Rounding(mode)
        assert (rounding.grad, rounding.weights) == (grad, weights)

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="int8 rounding must be one of"):
            Rounding("sloppy")

    def test_seeds_are_a_hashed_counter_below_two_to_the_31(self):
        rounding = Rounding("grad")
        seeds = [rounding.next_seed() for _ in range(1000)]
        assert seeds[0] == (1 * 2654435761 + 12345) % 2147483647
        assert len(set(seeds)) == 1000 and max(seeds) < 2**31

    def test_two_runs_draw_the_same_seeds(self):
        first, second = Rounding("grad"), Rounding("grad")
        assert [first.next_seed() for _ in range(5)] == [second.next_seed() for _ in range(5)]

    def test_a_seed_is_drawn_only_where_the_rounding_is_stochastic(self):
        nearest = Rounding("nearest")
        assert nearest.grad_seed() == 0 and nearest.weight_seed() == 0
        grad = Rounding("grad")
        assert grad.weight_seed() == 0 and grad.grad_seed() != 0
        both = Rounding("stochastic")
        assert both.grad_seed() != 0 and both.weight_seed() != 0

    def test_the_check_bar_follows_the_rounding(self):
        assert Rounding("nearest").cosine_min == policy.COSINE_MIN == 0.999
        assert Rounding("grad").cosine_min == policy.COSINE_MIN_SR == 0.998
        assert Rounding("stochastic").cosine_min == 0.998

    def test_the_policy_builds_a_fresh_one_per_run(self):
        pol = Policy()
        assert pol.rounding() is not pol.rounding()
        assert pol.rounding().mode == "grad"


class TestInt8Switch:
    def test_auto_is_on_until_the_check_says_otherwise(self):
        switch = Int8Switch("auto")
        assert switch.needs_check and switch.enabled() and switch.gemm_enabled()
        assert switch.decide(False, 0.5, 0.9) is False
        assert not switch.needs_check and not switch.enabled() and not switch.gemm_enabled()
        assert switch.check == {"cosine": 0.5, "rel_err": 0.9}

    def test_auto_stays_on_when_the_check_passes(self):
        switch = Int8Switch("auto")
        switch.decide(True, 0.9991, 0.043)
        assert switch.enabled() and switch.gemm_enabled() and not switch.needs_check

    @pytest.mark.parametrize("mode,enabled,gemm", [
        ("on", True, True), ("stash", True, False), ("off", False, False)])
    def test_fixed_modes_never_need_the_check(self, mode, enabled, gemm):
        switch = Int8Switch(mode)
        assert not switch.needs_check
        assert (switch.enabled(), switch.gemm_enabled()) == (enabled, gemm)

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="int8_backward must be one of"):
            Int8Switch("maybe")

    def test_two_switches_do_not_share_a_verdict(self):
        first, second = Policy().int8_switch(), Policy().int8_switch()
        first.decide(False)
        assert second.enabled() and second.needs_check


def test_the_module_keeps_no_state_and_reads_no_environment(monkeypatch):
    monkeypatch.setenv("KZ_INT8_BWD", "0")
    monkeypatch.setenv("KZ_GPU_BUDGET_MIB", "123")
    assert Policy() == Policy.from_block(PackedLoraConfig())
    assert Policy().int8_switch().enabled()


def test_importing_the_pure_half_does_not_import_torch():
    code = (
        "import sys\n"
        "from soup_cli.kernels.packed_lora import policy, budget, support, layout, config, loader\n"
        "from soup_cli.config import schema\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "assert 'triton' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr
