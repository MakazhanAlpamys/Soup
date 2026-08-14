"""#336 — DeepSpeed fails with LoRA on every stage.

Two independent defects, measured on 4xH100 and recorded in
``benchmarks/gate-h100-validation.md`` STEP 20.

**Defect 1 — the empty no-decay parameter group.** HF's ``create_optimizer``
always builds *two* groups (decay / no-decay). With LoRA every trainable
tensor is a 2-D ``lora_A``/``lora_B`` weight, so the no-decay group comes out
**empty** — measured group sizes ``[192, 0]`` on one GPU. DeepSpeed then drops
the empty group when it builds its own partitioning, leaving **one** group
against a scheduler whose ``base_lrs`` still has length two, and torch 2.13's
``zip(..., strict=True)`` in ``torch/optim/lr_scheduler.py:296`` raises
``ValueError: zip() argument 2 is longer than argument 1`` at the first
``lr_scheduler.step()``. The control that isolates the trigger is full
fine-tuning: it puts biases and layernorms in the second group, both groups
are non-empty, and the same config trains to completion.

**Defect 2 — ZeRO++ quantisation dtype.** The ``zero++`` preset declares
``bf16.enabled: true`` *and* ``zero_quantized_weights/gradients: true`` in the
same file. DeepSpeed's quantiser is the fp16 CUDA kernel, so the dequantised
all-gather comes back ``c10::Half`` and meets a ``c10::BFloat16`` activation:
``RuntimeError: expected mat1 and mat2 to have the same dtype``. Fixing the
parameter groups does **not** make ZeRO++ run. The same preset also hardcoded
``zero_hpz_partition_size: 8`` regardless of how many GPUs the run has.

Every test here has a control, because both fixes are shaped like
special-casing and a special case that quietly breaks the ordinary path would
be a worse bug than the one being fixed.
"""

import copy
import json

import pytest

from soup_cli.utils.deepspeed import (
    CONFIGS,
    get_deepspeed_config,
    prune_empty_param_groups,
    resolve_deepspeed_config,
    resolve_hpz_partition_size,
    resolve_world_size,
    write_deepspeed_config,
)

# --------------------------------------------------------------------------
# Defect 1 — the pure half: pruning empty parameter groups
# --------------------------------------------------------------------------


class _FakeOptimizer:
    """The only surface ``prune_empty_param_groups`` touches.

    A real ``torch.optim.Optimizer`` cannot be built without torch, and the
    function under test reads and rewrites exactly one attribute, so the fake
    is the whole contract rather than a convenience.
    """

    def __init__(self, param_groups):
        self.param_groups = param_groups


class TestPruneEmptyParamGroups:
    def test_the_empty_group_is_dropped(self):
        opt = _FakeOptimizer([
            {"params": ["a", "b"], "weight_decay": 0.01},
            {"params": [], "weight_decay": 0.0},
        ])
        dropped = prune_empty_param_groups(opt)
        assert dropped == 1
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["params"] == ["a", "b"]

    def test_control_two_populated_groups_are_left_alone(self):
        """The full-fine-tuning shape. A fix that also rewrote this would
        change the optimiser of every non-LoRA DeepSpeed run."""
        groups = [
            {"params": ["w"], "weight_decay": 0.01},
            {"params": ["bias"], "weight_decay": 0.0},
        ]
        opt = _FakeOptimizer(copy.deepcopy(groups))
        dropped = prune_empty_param_groups(opt)
        assert dropped == 0
        assert len(opt.param_groups) == 2
        assert [g["weight_decay"] for g in opt.param_groups] == [0.01, 0.0]

    def test_the_surviving_group_keeps_its_hyperparameters(self):
        """Dropping a group must not silently reset weight decay or LR on the
        one that survives — that would train a different model."""
        opt = _FakeOptimizer([
            {"params": ["a"], "weight_decay": 0.123, "lr": 5e-5, "betas": (0.9, 0.95)},
            {"params": [], "weight_decay": 0.0, "lr": 5e-5},
        ])
        prune_empty_param_groups(opt)
        kept = opt.param_groups[0]
        assert kept["weight_decay"] == 0.123
        assert kept["lr"] == 5e-5
        assert kept["betas"] == (0.9, 0.95)

    def test_every_group_empty_is_left_untouched(self):
        """Nothing is trainable. Pruning to zero groups would replace a clear
        'no trainable parameters' failure with an obscure one inside
        DeepSpeed, so the honest move is to leave it and let the real error
        surface."""
        opt = _FakeOptimizer([{"params": []}, {"params": []}])
        dropped = prune_empty_param_groups(opt)
        assert dropped == 0
        assert len(opt.param_groups) == 2

    def test_no_param_groups_at_all_is_a_no_op(self):
        opt = _FakeOptimizer([])
        assert prune_empty_param_groups(opt) == 0
        assert opt.param_groups == []

    def test_a_group_with_no_params_key_counts_as_empty(self):
        opt = _FakeOptimizer([{"params": ["a"]}, {}])
        assert prune_empty_param_groups(opt) == 1
        assert len(opt.param_groups) == 1

    def test_it_mutates_in_place_rather_than_rebinding(self):
        """``torch.optim.Optimizer.param_groups`` is read by the scheduler
        that was built over it, so the list object has to stay the same one."""
        groups = [{"params": ["a"]}, {"params": []}]
        opt = _FakeOptimizer(groups)
        prune_empty_param_groups(opt)
        assert opt.param_groups is groups
        assert len(groups) == 1


# --------------------------------------------------------------------------
# Defect 2 — the ZeRO++ preset
# --------------------------------------------------------------------------


class TestZeroPlusPlusQuantisationDtype:
    def test_bf16_preset_no_longer_ships_fp16_quantisation(self):
        """The shipped preset declared bf16 and fp16 quantisation in the same
        file. DeepSpeed's quantiser is the fp16 kernel, so the dequantised
        all-gather met a bf16 activation and the forward pass died."""
        cfg, notes = resolve_deepspeed_config(get_deepspeed_config("zero++"), gpu_count=4)
        zero = cfg["zero_optimization"]
        assert zero["zero_quantized_weights"] is False
        assert zero["zero_quantized_gradients"] is False
        joined = " ".join(notes).lower()
        assert "bf16" in joined
        assert "quantiz" in joined or "quantis" in joined

    def test_control_an_fp16_run_keeps_quantisation(self):
        """The fix must be a dtype decision, not 'quantisation is always off'
        — otherwise ZeRO++ loses its communication saving for everyone."""
        raw = get_deepspeed_config("zero++")
        raw["bf16"] = {"enabled": False}
        raw["fp16"] = {"enabled": True}
        cfg, _notes = resolve_deepspeed_config(raw, gpu_count=4)
        zero = cfg["zero_optimization"]
        assert zero["zero_quantized_weights"] is True
        assert zero["zero_quantized_gradients"] is True

    def test_hierarchical_partitioning_survives_the_dtype_fix(self):
        """Disabling quantisation must not turn ZeRO++ into plain ZeRO-3 —
        hierarchical partitioning is the other half of the feature."""
        cfg, _ = resolve_deepspeed_config(get_deepspeed_config("zero++"), gpu_count=4)
        assert cfg["zero_optimization"]["zero_hpz_partition_size"] == 4


class TestHpzPartitionSizeFollowsGpuCount:
    @pytest.mark.parametrize("gpus,expected", [(1, 1), (2, 2), (4, 4), (8, 8)])
    def test_it_equals_the_gpu_count(self, gpus, expected):
        cfg, _ = resolve_deepspeed_config(get_deepspeed_config("zero++"), gpu_count=gpus)
        assert cfg["zero_optimization"]["zero_hpz_partition_size"] == expected

    def test_the_shipped_preset_was_wrong_for_anything_but_eight(self):
        """Characterisation: the raw preset hardcodes 8. DeepSpeed requires
        world_size %% hpz == 0, so on the measured 4-GPU box it was invalid."""
        assert CONFIGS["zero++"]["zero_optimization"]["zero_hpz_partition_size"] == 8

    def test_it_never_exceeds_the_world_size(self):
        cfg, _ = resolve_deepspeed_config(get_deepspeed_config("zero++"), gpu_count=3)
        hpz = cfg["zero_optimization"]["zero_hpz_partition_size"]
        assert hpz <= 3
        assert 3 % hpz == 0

    def test_ranks_per_node_is_preferred_when_the_launcher_reports_it(self, monkeypatch):
        """Hierarchical partitioning exists to keep the secondary shard inside
        one node's interconnect, so on 2 nodes x 8 GPUs the partition is 8,
        not 16."""
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        assert resolve_hpz_partition_size(16) == 8

    def test_control_single_node_gives_the_same_answer_either_way(self, monkeypatch):
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        assert resolve_hpz_partition_size(4) == 4

    def test_a_local_size_that_does_not_divide_the_world_is_ignored(self, monkeypatch):
        """A non-divisor would fail DeepSpeed's own check, so the world size —
        which always divides — wins over a launcher value that cannot work."""
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "3")
        assert resolve_hpz_partition_size(8) == 8

    def test_junk_local_world_size_is_ignored(self, monkeypatch):
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "banana")
        assert resolve_hpz_partition_size(4) == 4

    def test_an_unknown_gpu_count_falls_back_to_one(self):
        """Zero GPUs means the probe failed; hpz=1 disables hierarchical
        partitioning rather than asserting a partition size that cannot
        divide the real world size."""
        cfg, _ = resolve_deepspeed_config(get_deepspeed_config("zero++"), gpu_count=0)
        assert cfg["zero_optimization"]["zero_hpz_partition_size"] == 1


class TestTheResolverLeavesOtherPresetsAlone:
    @pytest.mark.parametrize("stage", ["zero2", "zero3", "zero2_offload", "zero3_offload"])
    def test_non_zero_pp_presets_are_byte_identical_after_resolving(self, stage):
        """The control for defect 2: only ZeRO++ carries the quantisation and
        hpz keys, so nothing else may move."""
        before = get_deepspeed_config(stage)
        after, notes = resolve_deepspeed_config(before, gpu_count=4)
        assert after == get_deepspeed_config(stage)
        assert notes == []

    def test_the_resolver_does_not_mutate_its_input(self):
        original = get_deepspeed_config("zero++")
        snapshot = copy.deepcopy(original)
        resolve_deepspeed_config(original, gpu_count=2)
        assert original == snapshot


class TestConfigPlumbing:
    def test_get_deepspeed_config_still_returns_a_copy(self):
        first = get_deepspeed_config("zero2")
        first["zero_optimization"]["stage"] = 99
        assert get_deepspeed_config("zero2")["zero_optimization"]["stage"] == 2

    def test_unknown_stage_still_raises_and_names_the_options(self):
        with pytest.raises(ValueError) as exc:
            get_deepspeed_config("zero9000")
        assert "zero2" in str(exc.value)

    def test_written_config_is_the_resolved_one(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "2")
        path = write_deepspeed_config("zero++")
        written = json.loads(open(path, encoding="utf-8").read())
        zero = written["zero_optimization"]
        assert zero["zero_hpz_partition_size"] == 2
        assert zero["zero_quantized_weights"] is False

    def test_zero_pp_alias_resolves_identically(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "2")
        a, _ = resolve_deepspeed_config(get_deepspeed_config("zero++"), gpu_count=2)
        b, _ = resolve_deepspeed_config(get_deepspeed_config("zero_pp"), gpu_count=2)
        assert a == b


class TestResolveWorldSize:
    def test_world_size_env_wins(self, monkeypatch):
        """Inside an ``accelerate launch`` rank, WORLD_SIZE is the authority —
        ``--gpus 2`` on an 8-GPU box must not produce hpz=8."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        assert resolve_world_size() == 2

    def test_a_junk_world_size_is_ignored(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "not-a-number")
        assert resolve_world_size() >= 0

    def test_a_nonpositive_world_size_is_ignored(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "0")
        assert resolve_world_size() >= 0

    def test_an_explicit_count_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "8")
        cfg, _ = resolve_deepspeed_config(get_deepspeed_config("zero++"), gpu_count=2)
        assert cfg["zero_optimization"]["zero_hpz_partition_size"] == 2


# --------------------------------------------------------------------------
# Defect 1 — the live half: a real HF optimizer over a real LoRA model
# --------------------------------------------------------------------------


def _tiny_causal_lm():
    """A 2-layer Llama built from config — small enough for CPU CI, real
    enough that HF's decay/no-decay split behaves exactly as it does at 8B."""
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    return transformers.LlamaForCausalLM(cfg)


def _lora_wrap(model):
    peft = pytest.importorskip("peft")
    return peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )


def _hf_trainer(model, tmp_path):
    transformers = pytest.importorskip("transformers")
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path), report_to=[], weight_decay=0.01, learning_rate=2e-4,
    )
    return transformers.Trainer(model=model, args=args)


class TestTheEmptyGroupIsRealAndLoraOnly:
    def test_lora_produces_an_empty_no_decay_group(self, tmp_path):
        """The precondition the whole issue rests on, asserted rather than
        assumed. Measured on the box as ``[192, 0]``."""
        pytest.importorskip("torch")
        trainer = _hf_trainer(_lora_wrap(_tiny_causal_lm()), tmp_path)
        opt = trainer.create_optimizer()
        sizes = [len(g["params"]) for g in opt.param_groups]
        assert len(sizes) == 2
        assert sizes[0] > 0
        assert sizes[1] == 0, f"expected an empty no-decay group, got {sizes}"

    def test_control_full_finetune_populates_both_groups(self, tmp_path):
        """The control from the issue: replace LoRA with full FT and the
        no-decay group picks up the norms, so nothing needs pruning."""
        pytest.importorskip("torch")
        trainer = _hf_trainer(_tiny_causal_lm(), tmp_path)
        opt = trainer.create_optimizer()
        sizes = [len(g["params"]) for g in opt.param_groups]
        assert len(sizes) == 2
        assert all(n > 0 for n in sizes), f"expected both groups populated, got {sizes}"


def _scheduler_zip_is_strict() -> bool:
    """Does this torch's LRScheduler refuse a base_lrs / param_groups mismatch?

    Probed BEHAVIOURALLY on a throwaway optimizer, not by reading
    `inspect.getsource(LRScheduler.step)` for "strict=True" — the first version of
    this helper did exactly that, and it was wrong on CI's torch, where the
    strictness lives somewhere the source scan did not look. The test then stepped
    an unguarded scheduler expecting no exception and got one.

    The same defect is a hard ValueError on torch 2.13 and a silent wrong-LR on
    2.5.1, so this has to be detected rather than assumed either way.
    """
    import torch

    first = torch.nn.Parameter(torch.zeros(1))
    second = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD(
        [{"params": [first]}, {"params": [second]}], lr=0.1
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    optimizer.param_groups.pop()  # 1 group against 2 base_lrs
    try:
        scheduler.step()
    except ValueError:
        return True
    except Exception:  # noqa: BLE001 - any other refusal still counts as strict
        return True
    return False


class TestTheGuardOnARealTrainer:
    def test_the_guard_leaves_one_group_for_lora(self, tmp_path):
        from soup_cli.utils.deepspeed import attach_empty_param_group_guard

        pytest.importorskip("torch")
        trainer = _hf_trainer(_lora_wrap(_tiny_causal_lm()), tmp_path)
        assert attach_empty_param_group_guard(trainer) is True
        opt = trainer.create_optimizer()
        assert len(opt.param_groups) == 1
        assert len(opt.param_groups[0]["params"]) > 0

    def test_control_the_guard_leaves_full_finetune_at_two_groups(self, tmp_path):
        """A fix that special-cases LoRA must not disturb the path that
        already worked — full FT trained to completion under ZeRO-2."""
        from soup_cli.utils.deepspeed import attach_empty_param_group_guard

        pytest.importorskip("torch")
        trainer = _hf_trainer(_tiny_causal_lm(), tmp_path)
        attach_empty_param_group_guard(trainer)
        opt = trainer.create_optimizer()
        assert len(opt.param_groups) == 2

    def test_scheduler_base_lrs_match_the_param_groups(self, tmp_path):
        """THE regression test. torch 2.13 zips ``optimizer.param_groups``
        against the scheduler's per-group values with ``strict=True``, so the
        two lengths agreeing *is* the bug's absence."""
        from soup_cli.utils.deepspeed import attach_empty_param_group_guard

        pytest.importorskip("torch")
        trainer = _hf_trainer(_lora_wrap(_tiny_causal_lm()), tmp_path)
        attach_empty_param_group_guard(trainer)
        opt = trainer.create_optimizer()
        sched = trainer.create_scheduler(num_training_steps=10, optimizer=opt)
        assert len(sched.base_lrs) == len(opt.param_groups)
        sched.step()  # would raise ValueError under torch>=2.13 if mismatched

    def test_control_without_the_guard_the_lengths_disagree_after_a_drop(self, tmp_path):
        """Proves the test above has teeth: unguarded, the scheduler carries two
        base_lrs, and once DeepSpeed drops the empty group there is a 1-vs-2
        mismatch. Simulated here by dropping it after the fact, exactly as
        DeepSpeed does.

        The MISMATCH is the defect and is asserted unconditionally. Whether it
        RAISES is torch's business and is not stable: `lr_scheduler` gained
        `zip(..., strict=True)` in a later torch, so this crashes on 2.13 (where
        #336 was reported) and silently drops the extra base_lr on 2.5.1 (the dev
        box). Pinning the exception alone would make this control red on one
        supported stack and green on the other while the real defect - the
        lengths disagreeing at all - went unasserted on both.
        """
        pytest.importorskip("torch")
        trainer = _hf_trainer(_lora_wrap(_tiny_causal_lm()), tmp_path)
        opt = trainer.create_optimizer()
        sched = trainer.create_scheduler(num_training_steps=10, optimizer=opt)
        assert len(sched.base_lrs) == 2
        opt.param_groups[:] = [g for g in opt.param_groups if g["params"]]
        assert len(opt.param_groups) == 1
        assert len(sched.base_lrs) != len(opt.param_groups), (
            "the drop did not create the mismatch this control exists to produce"
        )
        if _scheduler_zip_is_strict():
            with pytest.raises(ValueError):
                sched.step()
        else:
            # Quieter on this torch, not harmless: the surplus base_lr is
            # silently ignored, so a group's learning rate stops being applied.
            sched.step()

    def test_the_guard_is_idempotent(self, tmp_path):
        from soup_cli.utils.deepspeed import attach_empty_param_group_guard

        pytest.importorskip("torch")
        trainer = _hf_trainer(_lora_wrap(_tiny_causal_lm()), tmp_path)
        assert attach_empty_param_group_guard(trainer) is True
        assert attach_empty_param_group_guard(trainer) is False
        opt = trainer.create_optimizer()
        assert len(opt.param_groups) == 1

    def test_the_guard_returns_the_trainers_own_optimizer(self, tmp_path):
        """HF reads ``trainer.optimizer`` after calling ``create_optimizer``;
        a wrapper that returned a different object would leave the trainer
        holding the unpruned one."""
        from soup_cli.utils.deepspeed import attach_empty_param_group_guard

        pytest.importorskip("torch")
        trainer = _hf_trainer(_lora_wrap(_tiny_causal_lm()), tmp_path)
        attach_empty_param_group_guard(trainer)
        opt = trainer.create_optimizer()
        assert opt is trainer.optimizer


class TestTheSftWrapperWiresTheGuard:
    """The end-to-end proof is the four real multi-GPU runs recorded in
    ``benchmarks/gate-h100-validation.md``; constructing a DeepSpeed-enabled
    ``TrainingArguments`` in CI would require the ``deepspeed`` package and a
    distributed context. What is checkable here is that the call site exists
    and is conditional, which is what a future refactor would silently drop.
    """

    def _sft_source(self):
        from pathlib import Path

        import soup_cli.trainer.sft as sft_mod

        return Path(sft_mod.__file__).read_text(encoding="utf-8")

    def test_sft_calls_the_guard(self):
        assert "attach_empty_param_group_guard" in self._sft_source()

    def test_the_call_is_conditional_on_deepspeed(self):
        src = self._sft_source()
        idx = src.index("attach_empty_param_group_guard(")
        window = src[max(0, idx - 600):idx]
        assert "self.deepspeed_config" in window, (
            "the guard must only fire under DeepSpeed — attaching it "
            "unconditionally would change every LoRA run's optimizer"
        )
