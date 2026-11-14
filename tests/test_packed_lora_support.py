"""``soup_cli.kernels.packed_lora.support``: what the kernels support, as reasons (CPU).

Each function returns a list of sentences a user would read in the preflight; the tests pin
the sentence and the one condition behind it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.kernels.packed_lora import support

GOOD_ENV = dict(system="Linux", has_triton=True, compute_capability=(8, 9), bf16_supported=True,
                world_size=1)


def _env(**over):
    return support.environment_problems(**{**GOOD_ENV, **over})


class TestEnvironment:
    def test_a_good_machine_has_no_problem(self):
        assert _env() == []

    @pytest.mark.parametrize("cap", [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)])
    def test_every_ampere_or_newer_card_is_fine(self, cap):
        assert _env(compute_capability=cap) == []

    @pytest.mark.parametrize("system", ["Windows", "Darwin", ""])
    def test_linux_only(self, system):
        (problem,) = _env(system=system)
        assert problem.startswith("runs on Linux (and WSL2) only; this is ")

    def test_triton_is_needed(self):
        (problem,) = _env(has_triton=False)
        assert problem.startswith("needs the triton package")

    def test_a_cuda_device_is_needed(self):
        (problem,) = _env(compute_capability=None)
        assert problem == "needs a CUDA GPU; none is available."

    def test_a_t4_is_named_as_not_validated(self):
        (problem,) = _env(compute_capability=(7, 5))
        assert "compute capability 8.0 or newer" in problem and "this one is 7.5" in problem
        assert "fp16 on a T4" in problem

    def test_bf16_is_needed(self):
        (problem,) = _env(bf16_supported=False)
        assert problem == "needs bf16 support on the GPU; torch reports none."

    def test_bf16_is_not_asked_of_a_card_already_refused(self):
        assert len(_env(compute_capability=(7, 5), bf16_supported=False)) == 1

    def test_one_gpu_only(self):
        (problem,) = _env(world_size=2)
        assert "supports one GPU only; WORLD_SIZE is 2" in problem

    def test_all_problems_are_listed(self):
        problems = _env(system="Windows", has_triton=False, compute_capability=None,
                        world_size=4)
        assert len(problems) == 4


def _config(**over):
    base = dict(model_type="qwen2", hidden_act="silu", hidden_size=1536, num_attention_heads=12,
                num_key_value_heads=2, intermediate_size=8960, use_sliding_window=False)
    return SimpleNamespace(**{**base, **over})


class TestModelConfig:
    def test_qwen2_5_1_5b_is_supported(self):
        assert support.model_config_problems(_config()) == []

    def test_qwen2_5_7b_and_72b_shapes_are_supported(self):
        seven = _config(hidden_size=3584, num_attention_heads=28, num_key_value_heads=4,
                        intermediate_size=18944)
        seventy_two = _config(hidden_size=8192, num_attention_heads=64, num_key_value_heads=8,
                              intermediate_size=29568)
        assert support.model_config_problems(seven) == []
        assert support.model_config_problems(seventy_two) == []

    def test_another_architecture(self):
        (problem,) = support.model_config_problems(_config(model_type="llama"))
        assert problem == "supports Qwen2ForCausalLM only; this model has model_type='llama'."

    def test_qwen3_is_ported_but_closed(self):
        (problem,) = support.model_config_problems(_config(model_type="qwen3"))
        assert problem.startswith("Qwen3 is ported but not enabled")
        assert "no training run on a real Qwen3 has matched" in problem

    def test_qwen3_opens_for_the_layer_checks(self):
        ported = tuple(support.PORTED_ARCHITECTURES.values())
        assert support.model_config_problems(_config(model_type="qwen3"), ported) == []

    def test_sliding_window(self):
        (problem,) = support.model_config_problems(_config(use_sliding_window=True))
        assert "sliding-window attention" in problem

    def test_sliding_window_layer_types(self):
        config = _config(layer_types=["full_attention", "sliding_attention"])
        (problem,) = support.model_config_problems(config)
        assert "sliding-window attention" in problem

    def test_activation(self):
        (problem,) = support.model_config_problems(_config(hidden_act="gelu"))
        assert "uses the 'gelu' activation" in problem

    @pytest.mark.parametrize("heads,kv", [(12, 5), (12, 3), (0, 0), (12, 0)])
    def test_kv_heads_must_divide_the_heads_and_be_a_power_of_two(self, heads, kv):
        problems = support.model_config_problems(
            _config(num_attention_heads=heads, num_key_value_heads=kv))
        assert any("kv heads" in problem for problem in problems)

    def test_an_odd_head_size(self):
        problems = support.model_config_problems(
            _config(hidden_size=63, num_attention_heads=1, num_key_value_heads=1))
        assert any("head size 63" in problem for problem in problems)

    def test_an_explicit_head_dim_is_used(self):
        config = _config(head_dim=64, hidden_size=1000, num_attention_heads=8,
                         num_key_value_heads=2)
        assert support.model_config_problems(config) == []

    def test_intermediate_size_must_pack(self):
        problems = support.model_config_problems(_config(intermediate_size=8961))
        assert any("intermediate size 8961" in problem for problem in problems)


class TestAttentionSignature:
    def test_the_installed_torch_has_the_signatures_the_kernels_call(self):
        pytest.importorskip("torch")
        assert support.attention_signature_problems() == []

    def _fake(self, forward_names=None, backward_names=None, kwargs=("scale",), outputs=6):
        def argument(name, kwarg=False):
            return SimpleNamespace(name=name, kwarg_only=kwarg)

        fwd = SimpleNamespace(
            arguments=[argument(n) for n in (forward_names or support.ATTENTION_FORWARD_ARGS)]
            + [argument(k, True) for k in kwargs],
            returns=[object()] * outputs)
        bwd = SimpleNamespace(
            arguments=[argument(n) for n in (backward_names or support.ATTENTION_BACKWARD_ARGS)]
            + [argument(k, True) for k in ("scale", "num_splits_key")],
            returns=[object()] * 4)
        return fwd, bwd

    def test_a_renamed_argument_is_reported(self):
        fwd, _ = self._fake(forward_names=("q",) + support.ATTENTION_FORWARD_ARGS[1:])
        problem = support._signature_problem(
            "_efficient_attention_forward", fwd, support.ATTENTION_FORWARD_ARGS,
            support.ATTENTION_FORWARD_KWARGS, support.ATTENTION_FORWARD_OUTPUTS)
        assert problem.startswith("torch's private aten._efficient_attention_forward takes (q,")

    def test_a_missing_keyword_is_reported(self):
        fwd, _ = self._fake(kwargs=())
        problem = support._signature_problem(
            "_efficient_attention_forward", fwd, support.ATTENTION_FORWARD_ARGS,
            support.ATTENTION_FORWARD_KWARGS, support.ATTENTION_FORWARD_OUTPUTS)
        assert problem == ("torch's private aten._efficient_attention_forward has no keyword "
                           "argument scale.")

    def test_a_different_number_of_outputs_is_reported(self):
        fwd, _ = self._fake(outputs=5)
        problem = support._signature_problem(
            "_efficient_attention_forward", fwd, support.ATTENTION_FORWARD_ARGS,
            support.ATTENTION_FORWARD_KWARGS, support.ATTENTION_FORWARD_OUTPUTS)
        assert problem == ("torch's private aten._efficient_attention_forward returns 5 values; "
                           "the kernels read 6.")

    def test_a_matching_signature_has_no_problem(self):
        fwd, bwd = self._fake()
        assert support._signature_problem(
            "f", fwd, support.ATTENTION_FORWARD_ARGS, support.ATTENTION_FORWARD_KWARGS, 6) is None
        assert support._signature_problem(
            "b", bwd, support.ATTENTION_BACKWARD_ARGS, support.ATTENTION_BACKWARD_KWARGS, 4
        ) is None

    def test_a_torch_without_the_ops_says_so(self, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch, "ops", SimpleNamespace(aten=SimpleNamespace()))
        problems = support.attention_signature_problems()
        assert problems and "memory-efficient attention ops are not available" in problems[0]


class TestMemory:
    def test_host_memory_that_fits(self):
        assert support.host_memory_problems(8 << 30, 12 << 30) == []
        assert support.host_memory_problems(8 << 30, 8 << 30) == []

    def test_host_memory_not_measurable_says_nothing(self):
        assert support.host_memory_problems(80 << 30, None) == []

    def test_host_memory_short(self):
        (problem,) = support.host_memory_problems(9 << 30, 6 << 30)
        assert "needs about 9.0 GiB of pinned host memory" in problem
        assert "6.0 GiB of RAM is free" in problem and "swaps" in problem

    def test_gpu_memory(self):
        assert support.gpu_memory_problems(5000 << 20, 6000 << 20) == []
        assert support.gpu_memory_problems(5000 << 20, None) == []
        (problem,) = support.gpu_memory_problems(5000 << 20, 4000 << 20)
        assert "5000 MiB of free GPU memory" in problem and "4000 MiB is free" in problem


def test_an_architecture_left_out_of_the_accepted_list_is_named():
    (problem,) = support.model_config_problems(_config(), architectures=())
    assert problem == "Qwen2ForCausalLM is not enabled for this run."
