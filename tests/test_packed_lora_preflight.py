"""``soup train``'s check of a ``training.packed_lora`` run before any model is loaded (CPU).

The machine is a value (``Machine``), so every refusal is reached with one offending fact and
asserted on the words a user would grep for. The model is a ``transformers`` config with the
shape of Qwen2.5-1.5B or 7B; the plan is the planner's, checked against the planner itself.
"""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from soup_cli.config.schema import SoupConfig
from soup_cli.kernels.packed_lora import budget, preflight, support
from soup_cli.kernels.packed_lora.preflight import Machine
from soup_cli.kernels.packed_lora.preflight import preflight as run_preflight

GIB = 1 << 30
MIB = 1 << 20

QWEN_7B = dict(
    vocab_size=152064, hidden_size=3584, intermediate_size=18944, num_hidden_layers=28,
    num_attention_heads=28, num_key_value_heads=4, tie_word_embeddings=False)
QWEN_1_5B = dict(
    vocab_size=151936, hidden_size=1536, intermediate_size=8960, num_hidden_layers=28,
    num_attention_heads=12, num_key_value_heads=2, tie_word_embeddings=True)


def model_config(**shape):
    from transformers import Qwen2Config

    return Qwen2Config(**(shape or QWEN_7B))


def soup(packed=None, **training):
    packed = {"enabled": True} if packed is None else packed
    return SoupConfig(
        base="Qwen/Qwen2.5-7B",
        data={"train": "tests/fixtures/sample_train.jsonl", "max_length": 512},
        training={"quantization": "4bit", "batch_size": 8,
                  "lora": {"r": 16, "alpha": 32, "dropout": 0.0}, "packed_lora": packed,
                  **training},
    )


def machine(**changes):
    base = Machine(
        system="Linux", has_triton=True, compute_capability=(12, 0), bf16_supported=True,
        world_size=1, free_ram_bytes=64 * GIB, free_vram_bytes=23000 * MIB,
        total_vram_bytes=24000 * MIB)
    return dataclasses.replace(base, **changes)


def laptop(**changes):
    """The 8 GB card of the dev box, with 8000 MiB of it free."""
    return machine(free_vram_bytes=8000 * MIB, total_vram_bytes=8151 * MIB, **changes)


@pytest.fixture(autouse=True)
def attention_signature_fine(monkeypatch):
    """The installed torch's private attention ops are checked in their own tests."""
    monkeypatch.setattr(support, "attention_signature_problems", lambda: [])


def only_problem(result):
    assert len(result.problems) == 1, result.problems
    return result.problems[0]


def only_fit_problem(result):
    """The reason the run does not fit right now (the kernels would take it)."""
    assert result.problems == [] and len(result.fit_problems) == 1, (
        result.problems, result.fit_problems)
    return result.fit_problems[0]


# --- a run the kernels can take ------------------------------------------------------------


class TestAnAcceptedRun:
    def test_a_7b_run_at_a_budget_that_fits_the_laptop_card(self):
        result = run_preflight(soup({"enabled": True, "gpu_budget_mib": 7000}), model_config(),
                               laptop())
        assert result.ok and result.problems == []
        assert result.tokens == 8 * 512
        # the planner's own plan for the shapes of the config, not a second opinion
        shape = budget.Shape.of_config(model_config(), rank=16)
        want = budget.Plan(shape, 7000 * MIB, 4096, True)
        assert result.plan.k == want.k and result.plan.est_peak == want.est_peak
        assert result.estimate_bytes == int(want.est_peak) <= 7000 * MIB
        assert result.host_bytes == want.host_bytes() > 4 * GIB

    def test_the_lora_rank_sizes_the_plan(self):
        low = run_preflight(soup({"enabled": True, "gpu_budget_mib": 7000}, lora={
            "r": 8, "alpha": 16, "dropout": 0.0}), model_config(), machine())
        high = run_preflight(soup({"enabled": True, "gpu_budget_mib": 7000}, lora={
            "r": 64, "alpha": 128, "dropout": 0.0}), model_config(), machine())
        assert low.plan.shape.lora_params < high.plan.shape.lora_params

    def test_without_a_budget_there_is_no_plan_and_the_floor_is_the_estimate(self):
        result = run_preflight(soup(), model_config(**QWEN_1_5B), machine())
        assert result.ok and result.plan is None
        shape = budget.Shape.of_config(model_config(**QWEN_1_5B), rank=16)
        assert result.estimate_bytes == int(shape.peak(True, True, 0, 4096, True))
        # all layers' pinned buffers are made up front unless activations stay resident
        assert result.host_bytes == 28 * shape.layer(4096, True)

    def test_resident_activations_need_no_pinned_memory(self):
        result = run_preflight(soup({"enabled": True, "activations": "resident"}),
                               model_config(**QWEN_1_5B), machine())
        assert result.ok and result.host_bytes == 0

    def test_the_int8_off_stash_is_larger(self):
        on = run_preflight(soup(), model_config(**QWEN_1_5B), machine())
        off = run_preflight(soup({"enabled": True, "int8_backward": "off"}),
                            model_config(**QWEN_1_5B), machine())
        assert off.host_bytes > on.host_bytes

    def test_a_config_that_did_not_switch_it_on_has_nothing_to_check(self):
        with pytest.raises(ValueError, match="enabled is not true"):
            run_preflight(soup({"enabled": False}), model_config(), machine())


# --- the machine ---------------------------------------------------------------------------


class TestTheMachine:
    def test_not_linux(self):
        result = run_preflight(soup(), model_config(), machine(system="Windows"))
        assert "Linux" in only_problem(result) and "Windows" in only_problem(result)

    def test_no_triton(self):
        result = run_preflight(soup(), model_config(), machine(has_triton=False))
        assert "triton" in only_problem(result)

    def test_no_cuda_device(self):
        result = run_preflight(soup(), model_config(), machine(
            compute_capability=None, bf16_supported=False, free_vram_bytes=None,
            total_vram_bytes=None))
        assert "needs a CUDA GPU" in only_problem(result)

    def test_a_pre_ampere_card(self):
        result = run_preflight(soup(), model_config(), machine(compute_capability=(7, 5)))
        assert "compute capability 8.0" in only_problem(result)
        assert "7.5" in only_problem(result)

    def test_no_bf16(self):
        result = run_preflight(soup(), model_config(), machine(bf16_supported=False))
        assert "bf16" in only_problem(result)

    def test_several_processes(self):
        result = run_preflight(soup(), model_config(), machine(world_size=2))
        assert "one GPU only" in only_problem(result) and "WORLD_SIZE is 2" in only_problem(
            result)

    def test_a_sharded_single_process_run(self):
        result = run_preflight(soup(), model_config(), machine(), sharding="--fsdp full_shard")
        assert "one GPU only; --fsdp full_shard is set" in only_problem(result)

    def test_the_private_attention_ops_are_checked_when_the_machine_is_fine(self, monkeypatch):
        monkeypatch.setattr(support, "attention_signature_problems",
                            lambda: ["torch's private aten.x changed."])
        result = run_preflight(soup(), model_config(), machine())
        assert only_problem(result) == "training.packed_lora torch's private aten.x changed."

    def test_and_not_when_it_is_not(self, monkeypatch):
        def explode():
            raise AssertionError("asked about the attention ops on a machine that cannot run")

        monkeypatch.setattr(support, "attention_signature_problems", explode)
        result = run_preflight(soup(), model_config(), machine(system="Darwin"))
        assert result.problems


# --- the model -----------------------------------------------------------------------------


class TestTheModel:
    def test_another_architecture(self):
        from transformers import LlamaConfig

        result = run_preflight(soup(), LlamaConfig(), machine())
        assert "supports Qwen2ForCausalLM only" in only_problem(result)
        assert result.plan is None

    def test_qwen3_is_ported_but_not_enabled(self):
        from transformers import Qwen3Config

        result = run_preflight(soup(), Qwen3Config(), machine())
        assert "Qwen3 is ported but not enabled" in only_problem(result)

    def test_a_config_that_could_not_be_read_waits_for_the_load(self):
        result = run_preflight(soup(), None, machine())
        assert result.ok and "checked at the load" in result.notes[0]

    def test_a_machine_that_cannot_run_is_reported_even_without_the_model_config(self):
        result = run_preflight(soup(), None, machine(system="Windows"))
        assert "Linux" in only_problem(result)

    def test_an_auto_batch_size_cannot_be_planned(self):
        config = soup()
        config.training.batch_size = "auto"
        result = run_preflight(config, model_config(), machine())
        assert "explicit training.batch_size" in only_problem(result)
        assert result.plan is None


# --- the plan against the card -------------------------------------------------------------


class TestThePlanAgainstTheCard:
    def test_a_budget_below_the_floor_is_refused_with_the_minimum(self):
        result = run_preflight(soup({"enabled": True, "gpu_budget_mib": 3000}), model_config(),
                               laptop())
        message = only_problem(result)
        assert "gpu_budget_mib=3000 is below the minimum of" in message
        minimum = int(message.split("minimum of ")[1].split(" MiB")[0])
        assert 5000 < minimum < 7000
        # the minimum it names is one the planner accepts
        assert run_preflight(soup({"enabled": True, "gpu_budget_mib": minimum}),
                             model_config(), laptop()).ok

    def test_resident_activations_that_do_not_fit_the_budget(self):
        config = soup({"enabled": True, "gpu_budget_mib": 7000, "activations": "resident"})
        result = run_preflight(config, model_config(), machine())
        assert "activations: resident" in only_problem(result)

    def test_a_budget_larger_than_the_card(self):
        result = run_preflight(soup({"enabled": True, "gpu_budget_mib": 12000}), model_config(),
                               laptop())
        assert "gpu_budget_mib=12000 is more than the GPU has (8151 MiB)" in only_problem(result)

    def test_a_card_that_is_busy(self):
        result = run_preflight(soup({"enabled": True, "gpu_budget_mib": 7000}), model_config(),
                               machine(free_vram_bytes=5000 * MIB))
        message = only_fit_problem(result)
        assert "only 5000 MiB of GPU memory is free right now" in message
        assert "gpu_budget_mib=7000" in message

    def test_without_a_budget_the_floor_has_to_fit_the_free_memory(self):
        result = run_preflight(soup(), model_config(), machine(free_vram_bytes=3000 * MIB))
        assert "free GPU memory for its floor" in only_fit_problem(result)
        assert "3000 MiB is free" in only_fit_problem(result)

    def test_not_enough_pinned_host_memory(self):
        config = soup({"enabled": True, "gpu_budget_mib": 7000})
        result = run_preflight(config, model_config(), machine(free_ram_bytes=3 * GIB))
        message = only_fit_problem(result)
        assert "pinned host memory" in message and "3.0 GiB of RAM is free" in message

    def test_a_fit_problem_is_not_an_unsupported_run_and_the_other_way_round(self):
        busy = run_preflight(soup(), model_config(), machine(free_vram_bytes=2000 * MIB))
        assert busy.problems == [] and len(busy.fit_problems) == 1 and not busy.ok
        windows = run_preflight(soup(), model_config(), machine(system="Windows"))
        assert windows.fit_problems == [] and len(windows.problems) == 1 and not windows.ok

    def test_an_unreadable_ram_figure_is_not_checked(self):
        config = soup({"enabled": True, "gpu_budget_mib": 7000})
        assert run_preflight(config, model_config(), machine(free_ram_bytes=None)).ok

    def test_an_unreadable_gpu_figure_is_not_checked(self):
        config = soup({"enabled": True, "gpu_budget_mib": 7000})
        assert run_preflight(config, model_config(), machine(
            free_vram_bytes=None, total_vram_bytes=None)).ok

    def test_every_reason_is_reported_not_only_the_first(self):
        result = run_preflight(soup(), model_config(), machine(
            system="Windows", has_triton=False, world_size=2))
        assert len(result.problems) == 3 and result.fit_problems == []


def test_tokens_per_step():
    assert preflight.tokens_per_step(soup()) == 4096
    config = soup()
    config.training.batch_size = "auto"
    assert preflight.tokens_per_step(config) is None


# --- reading the machine -------------------------------------------------------------------


class TestReadMachine:
    def _cuda(self, monkeypatch, available=True):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (12, 0))
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index=0: (7 * GIB, 8 * GIB))

    def test_the_figures_of_a_cuda_machine(self, monkeypatch):
        self._cuda(monkeypatch)
        monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
        monkeypatch.setattr(preflight.loader, "triton_available", lambda: True)
        monkeypatch.setenv("WORLD_SIZE", "1")
        fake = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(
            available=12 * GIB))
        monkeypatch.setitem(sys.modules, "psutil", fake)
        read = preflight.read_machine()
        assert read == Machine(
            system="Linux", has_triton=True, compute_capability=(12, 0), bf16_supported=True,
            world_size=1, free_ram_bytes=12 * GIB, free_vram_bytes=7 * GIB,
            total_vram_bytes=8 * GIB)

    def test_no_cuda_device(self, monkeypatch):
        self._cuda(monkeypatch, available=False)
        read = preflight.read_machine()
        assert read.compute_capability is None and read.bf16_supported is False
        assert read.free_vram_bytes is None and read.total_vram_bytes is None

    @pytest.mark.parametrize("value,want", [("4", 4), ("", 1), ("many", 1)])
    def test_world_size(self, monkeypatch, value, want):
        self._cuda(monkeypatch, available=False)
        monkeypatch.setenv("WORLD_SIZE", value)
        assert preflight.read_machine().world_size == want

    def test_a_missing_psutil_is_an_unknown_figure(self, monkeypatch):
        self._cuda(monkeypatch, available=False)
        monkeypatch.setitem(sys.modules, "psutil", None)
        assert preflight.read_machine().free_ram_bytes is None


class TestEveryReasonNamesItsSetting:
    @pytest.mark.parametrize("run,model,box", [
        (soup(), model_config(), machine(system="Windows")),
        (soup(), model_config(), machine(has_triton=False, bf16_supported=False)),
        (soup(), model_config(), machine(world_size=3)),
        (soup({"enabled": True, "gpu_budget_mib": 3000}), model_config(), laptop()),
        (soup({"enabled": True, "gpu_budget_mib": 12000}), model_config(), laptop()),
        (soup({"enabled": True, "gpu_budget_mib": 7000}), model_config(),
         machine(free_ram_bytes=1 * GIB)),
        (soup(), model_config(), machine(free_vram_bytes=2000 * MIB)),
    ])
    def test_the_sentence_starts_with_the_setting(self, run, model, box):
        result = run_preflight(run, model, box)
        reasons = result.problems + result.fit_problems
        assert reasons and not result.ok
        assert all(p.startswith("training.packed_lora") for p in reasons), reasons

    def test_a_reason_that_already_names_it_is_not_named_twice(self):
        result = run_preflight(soup({"enabled": True, "gpu_budget_mib": 3000}), model_config(),
                               laptop())
        assert not only_problem(result).startswith("training.packed_lora training.packed_lora")

    def test_model_reasons_too(self):
        from transformers import LlamaConfig

        result = run_preflight(soup(), LlamaConfig(), machine())
        assert only_problem(result).startswith("training.packed_lora supports Qwen2")
