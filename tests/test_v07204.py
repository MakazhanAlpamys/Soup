"""v0.72.4 — preference losses (DPO / ORPO / SimPO / KTO) over layer streaming.

The slot's whole risk is ONE property (plan.md §7.1): DPO needs a reference
model, and implemented naively as a second model instance it doubles memory and
defeats the feature entirely. It must be the SAME streamed base with adapters
disabled. **A passing loss curve does not detect a second instance** — only a
memory assertion does, which is why the assertions here are about bytes and
about object identity, not about the loss going down.

Gate numbers + method: `.claude/v0724-gate-results.md`.

A correction to the brief established while gating: ORPO and SimPO are genuinely
reference-free, but **KTO is not** — `KTOTrainer.__init__` has byte-for-byte the
same three-branch reference selection as DPO, so it inherits the same trap and
is asserted the same way here.
"""

import json
import os

import pytest


# ==========================================================================
# fixtures (mirroring tests/test_v07200.py so the two cannot drift)
# ==========================================================================
def _cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover - torch always present in CI
        return False


def _torch_version():
    """Reported in the KTO xfail message: the failure it tolerates is a torch
    version property, so the version is the one datum that makes the record
    useful when the issue is picked up."""
    try:
        import torch

        return torch.__version__
    except Exception:  # pragma: no cover - torch always present in CI
        return "unknown"


def _mps_is_the_accelerator():
    try:
        import torch

        return (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and not torch.cuda.is_available()
        )
    except Exception:  # pragma: no cover
        return False


def _tiny_llama_dir(tmp_path, n_layers=2, tie=True, vocab=64, hidden=64):
    """A real (tiny) Llama checkpoint on disk: config.json + model.safetensors.

    ``hidden`` is 64, not 32: bitsandbytes' CPU 4-bit repack computes
    ``blocks_per_row = 16 // (hidden // 2)`` and silently yields 0 at 32 (#323).
    """
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    if tie:
        state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights), model, config


def _write_tiny_tokenizer(directory):
    """A real, offline PreTrainedTokenizerFast so `setup()` can tokenize."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in (
        "hello",
        "world",
        "hi",
        "yo",
        "the",
        "cat",
        "sat",
        "on",
        "mat",
        "good",
        "bad",
        "answer",
        "question",
        "soup",
    ):
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(os.path.join(directory, "tokenizer.json"))
    with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "model_max_length": 128,
                "clean_up_tokenization_spaces": False,
            },
            fh,
        )


def _randomise_lora_b(model, seed=11):
    """PEFT initialises ``lora_B = 0``, so until B is load-bearing the adapter
    contributes NOTHING and every "did the adapter path run?" assertion — very
    much including "is the reference different from the policy?" — passes
    vacuously."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(
                    torch.randn(param.shape, generator=gen).to(param.device, param.dtype) * 0.05
                )


def _sync_adapters(dst, src):
    """Copy LoRA weights src -> dst across the ``.inner.`` wrapper difference.

    Returns the number copied; 0 means the comparison would be vacuous.
    """
    import torch

    def norm(key):
        return key.replace(".inner.", "")

    source = {norm(k): v.detach().clone() for k, v in src.state_dict().items() if "lora_" in k}
    copied = 0
    with torch.no_grad():
        for key, tensor in dst.state_dict().items():
            if "lora_" not in key:
                continue
            match = source.get(norm(key))
            if match is not None:
                tensor.copy_(match.to(tensor.device, tensor.dtype))
                copied += 1
    return copied


def _batch_on(model, batch):
    """Move a TRL batch onto the model's device.

    `TrainingArguments` picks CUDA whenever it is available, so a test that
    deliberately pins the MODEL to CPU (for exact float32 arithmetic) still gets
    CUDA batch tensors from the dataloader.
    """
    device = next(model.parameters()).device
    return {
        key: (value.to(device) if hasattr(value, "to") else value) for key, value in batch.items()
    }


def _loss_of(trainer, model, batch):
    """Call the trainer's loss for `model`, across TRL's signature differences.

    `KTOTrainer.get_batch_loss_metrics` takes (model, batch); DPO / ORPO / CPO
    take (model, batch, train_eval).
    """
    import inspect

    fn = trainer.get_batch_loss_metrics
    if "train_eval" in inspect.signature(fn).parameters:
        loss, _ = fn(model, batch, "train")
    else:
        loss, _ = fn(model, batch)
    return loss


def _match_streamed_dtype(resident, streamed):
    """Put the resident reference on the streamed model's device AND dtype.

    Streaming picks bf16 on CUDA and float32 on CPU. Comparing a float32
    resident model against a bf16 streamed one measures the dtype gap, not the
    streaming path — that mistake produced a 9.96e-04 "failure" that was
    entirely the test's own.
    """
    param = next(streamed.parameters())
    return resident.to(device=param.device, dtype=param.dtype)


def _pref_rows(n=4):
    return [{"prompt": "hi", "chosen": " good answer", "rejected": " bad"} for _ in range(n)]


def _kto_rows(n=4):
    return [{"prompt": "hi", "completion": " good answer", "label": i % 2 == 0} for i in range(n)]


def _sft_rows(n=4):
    return [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello world"},
            ]
        }
        for _ in range(n)
    ]


_TASK_ROWS = {
    "sft": _sft_rows,
    "dpo": _pref_rows,
    "orpo": _pref_rows,
    "simpo": _pref_rows,
    "kto": _kto_rows,
}

#: KTO refuses a per-device batch size of 1 outright ("the KL term will be
#: equivalent to the implied reward"), so it is only streamable at all because
#: v0.72.3 lifted layer streaming's batch-1 restriction.
_MIN_BATCH = {"kto": 2}


def _stream_cfg(weights, out_dir, task="dpo", **training):
    import yaml

    from soup_cli.config.loader import load_config_from_string

    tcfg = {
        # KTO is refused below batch 2 (its KL term is degenerate at 1), so the
        # per-task floor is the default here; an explicit kwarg still wins,
        # which is what TestKtoBatchIsRefusedEarly relies on.
        "batch_size": _MIN_BATCH.get(task, 1),
        "gradient_accumulation_steps": 1,
        "quantization": "none",
        "stream_layers": True,
        "epochs": 1,
        "logging_steps": 1,
        "save_steps": 1000,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    tcfg.update(training)
    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": weights,
                "task": task,
                "backend": "transformers",
                "modality": "text",
                "data": {
                    "train": "train.jsonl",
                    "max_length": 64,
                    # v0.36.0 removed the silent f-string fallback; SFT needs one.
                    "chat_template": "chatml",
                },
                "training": tcfg,
                "output": str(out_dir),
            }
        )
    )


def _wrapper_for(task):
    from soup_cli.trainer.dpo import DPOTrainerWrapper
    from soup_cli.trainer.kto import KTOTrainerWrapper
    from soup_cli.trainer.orpo import ORPOTrainerWrapper
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.trainer.simpo import SimPOTrainerWrapper

    return {
        "sft": SFTTrainerWrapper,
        "dpo": DPOTrainerWrapper,
        "orpo": ORPOTrainerWrapper,
        "simpo": SimPOTrainerWrapper,
        "kto": KTOTrainerWrapper,
    }[task]


def _build_streamed_wrapper(
    tmp_path,
    monkeypatch,
    task="dpo",
    n_layers=2,
    device=None,
    hidden=64,
    vocab=64,
    **training,
):
    """Build a task wrapper through the REAL `setup()` path, streaming.

    ``device`` defaults to the real accelerator, because `TrainingArguments`
    picks CUDA when it is available and forcing CPU there would only produce a
    device mismatch no user would ever hit. Numerical-equality tests pass
    ``device='cpu'`` deliberately: the streaming path uses float32 on CPU and
    bf16 on CUDA, and "bit-exact" is only a meaningful assertion in the former
    (a bf16 logp of -12.75 cannot represent a change smaller than ~0.05).
    """
    weights, resident, _ = _tiny_llama_dir(tmp_path, n_layers=n_layers, hidden=hidden, vocab=vocab)
    _write_tiny_tokenizer(weights)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    training.setdefault("batch_size", _MIN_BATCH.get(task, 1))
    cfg = _stream_cfg(weights, tmp_path / "out", task=task, **training)
    if device is None:
        device = "cuda" if _cuda_available() else "cpu"
    wrapper = _wrapper_for(task)(cfg, device=device)
    wrapper.setup({"train": _TASK_ROWS[task](8)})
    return wrapper, resident, weights


# ==========================================================================
# item 1 -- schema: which tasks may stream
# ==========================================================================
_REFERENCE_USING = ("dpo", "kto")
_ALL_PREFERENCE = ("dpo", "orpo", "simpo", "kto")


class TestStreamingTaskGate:
    """v0.72.0-.3 hard-coded ``task == 'sft'``. v0.72.4 opens exactly four more
    and keeps refusing the rest — GRPO/PPO *permanently*, because rollouts
    re-read the model per generated token and destroy the amortisation that
    makes streaming viable at all (plan §3.2)."""

    def _cfg(self, tmp_path, task):
        return _stream_cfg(str(tmp_path / "m"), tmp_path / "out", task=task)

    @pytest.mark.parametrize("task", ("sft",) + _ALL_PREFERENCE)
    def test_supported_tasks_are_accepted(self, tmp_path, task):
        cfg = self._cfg(tmp_path, task)
        assert cfg.task == task
        assert cfg.training.stream_layers is True

    @pytest.mark.parametrize("task", ("grpo", "ppo"))
    def test_rollout_tasks_are_refused_for_a_reason_that_does_not_expire(self, tmp_path, task):
        """Not "not yet" — never. The refusal must say why, or a future
        maintainer will read it as an unfinished slot and wire it up."""
        with pytest.raises(ValueError) as excinfo:
            self._cfg(tmp_path, task)
        message = str(excinfo.value)
        assert "stream_layers" in message
        assert "generation" in message or "rollout" in message
        assert "v0.72" not in message, (
            "a permanent refusal must not name a release that would lift it"
        )

    @pytest.mark.parametrize("task", ("reward_model", "pretrain", "embedding"))
    def test_other_tasks_still_refused(self, tmp_path, task):
        # `distill` is deliberately not in this list: it is rejected earlier by
        # its own `teacher_model` gate, so it would assert nothing about
        # streaming.
        with pytest.raises(ValueError, match="stream_layers"):
            self._cfg(tmp_path, task)


# ==========================================================================
# item 1 -- THE trap: one model, one stream
# ==========================================================================
class TestNoSecondModelInstance:
    """The brief's whole slot. Asserted on bytes and object identity, never on
    the loss."""

    @pytest.mark.parametrize("task", _REFERENCE_USING)
    def test_the_reference_is_the_same_model_with_adapters_disabled(
        self, tmp_path, monkeypatch, task
    ):
        wrapper, _, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task=task)
        trainer = wrapper.trainer
        assert hasattr(trainer, "ref_model"), (
            "TRL renamed `ref_model`; the no-second-instance property must be "
            "re-verified against the new API before this test is adjusted"
        )
        assert trainer.ref_model is None, (
            f"{task} built a SECOND model instance for the reference — that "
            f"doubles memory and defeats layer streaming entirely"
        )
        assert getattr(trainer, "is_peft_model", False) is True

    @pytest.mark.parametrize("task", _ALL_PREFERENCE)
    def test_exactly_one_weight_store_is_constructed(self, tmp_path, monkeypatch, task):
        """Counts RamSource constructions across the whole `setup()`. Two would
        mean two copies of the base in host RAM."""
        from soup_cli.utils import layer_stream_runtime as lsr

        calls = {"n": 0}
        real = lsr.RamSource

        class Counting(real):
            def __init__(self, *a, **k):
                calls["n"] += 1
                super().__init__(*a, **k)

        monkeypatch.setattr(lsr, "RamSource", Counting)
        _build_streamed_wrapper(tmp_path, monkeypatch, task=task)
        assert calls["n"] == 1, f"{task} constructed {calls['n']} weight stores"

    def test_reference_logps_actually_differ_from_the_policy(self, tmp_path, monkeypatch):
        """THE silent-failure check.

        The streamed layer substitutes base weights through `functional_call`
        rather than the module's own forward, so `disable_adapter()` being a
        no-op through that path is entirely plausible. If it were, the reference
        would BE the policy, every log-ratio would be 0, and the DPO loss would
        sit at -logsigmoid(0) = 0.6931 forever — which reads as "training
        slowly", not as a bug. Measured 8.2e-01 in the gate.
        """
        import torch

        wrapper, _, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task="dpo", device="cpu")
        _randomise_lora_b(wrapper.model)
        trainer = wrapper.trainer
        batch = _batch_on(wrapper.model, next(iter(trainer.get_train_dataloader())))
        wrapper.model.eval()
        with torch.no_grad():
            policy = trainer.concatenated_forward(wrapper.model, batch)
            ref_chosen, ref_rejected = trainer.compute_ref_log_probs(batch)
        assert (policy["chosen_logps"] - ref_chosen).abs().max().item() > 1e-4
        assert (policy["rejected_logps"] - ref_rejected).abs().max().item() > 1e-4

    def test_the_difference_really_comes_from_the_adapter(self, tmp_path, monkeypatch):
        """CONTROL for the test above: with ``lora_B = 0`` the adapter
        contributes nothing, so disabling it must change NOTHING and the two
        must be exactly equal. Without this, the previous test passes for any
        model whose two forwards merely differ for some other reason."""
        import torch

        wrapper, _, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task="dpo", device="cpu")
        with torch.no_grad():
            for name, param in wrapper.model.named_parameters():
                if "lora_B" in name:
                    param.zero_()
        trainer = wrapper.trainer
        batch = _batch_on(wrapper.model, next(iter(trainer.get_train_dataloader())))
        wrapper.model.eval()
        with torch.no_grad():
            policy = trainer.concatenated_forward(wrapper.model, batch)
            ref_chosen, _ = trainer.compute_ref_log_probs(batch)
        diff = (policy["chosen_logps"] - ref_chosen).abs().max().item()
        assert diff == 0.0, diff


@pytest.mark.skipif(not _cuda_available(), reason="peak VRAM needs CUDA")
class TestPeakVramIsNotDoubled:
    """The brief's literal assertion, on the real device."""

    def _peak(self, tmp_path, monkeypatch, task, second_reference=False):
        import gc

        import torch

        # The buffer pool is held by reference CYCLES in the module tree
        # (measured: close() alone retains it, close() + gc.collect() retains
        # 0.00 MB). Without collecting here the next arm's baseline carries the
        # previous pool and every peak is inflated by one pool's worth — which
        # is exactly how an early version of the gate produced a bogus 2x.
        gc.collect()
        torch.cuda.empty_cache()
        wrapper, _, weights = _build_streamed_wrapper(tmp_path, monkeypatch, task=task, n_layers=4)
        stats = wrapper._stream_runtime.stats()
        forced = None
        if second_reference:
            from transformers import AutoModelForCausalLM

            forced = AutoModelForCausalLM.from_pretrained(
                weights, torch_dtype=next(wrapper.model.parameters()).dtype
            ).to("cuda")
            wrapper.trainer.ref_model = forced
        # Reset AFTER setup: the pre-flight's own GEMM ceiling probe allocates
        # three 4096^3 matrices, which would otherwise dominate the "peak".
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        wrapper.trainer.args.max_steps = 1
        wrapper.trainer.train()
        peak = torch.cuda.max_memory_allocated()
        wrapper._close_stream_runtime()
        del wrapper, forced
        gc.collect()
        torch.cuda.empty_cache()
        return peak, stats

    @pytest.mark.parametrize("task", _REFERENCE_USING)
    def test_weight_bearing_terms_are_identical_to_sft(self, tmp_path, monkeypatch, task):
        """One store, one pool. The gate measured 729.91 MB / 60.83 MB for both
        arms on a 730 MB model; here the sizes are tiny but the EQUALITY is the
        size-independent form of the same claim."""
        sft_peak, sft_stats = self._peak(tmp_path / "a", monkeypatch, "sft")
        pref_peak, pref_stats = self._peak(tmp_path / "b", monkeypatch, task)
        assert pref_stats["store_bytes"] == sft_stats["store_bytes"]
        assert pref_stats["buffer_bytes"] == sft_stats["buffer_bytes"]
        # ...and the brief's literal metric, on the numbers actually measured.
        # `stats` describe the ONE streaming runtime's own bookkeeping and would
        # read identically even if a second RESIDENT reference model were built
        # outside it — which is the bug this slot exists to prevent.
        assert pref_peak < sft_peak * 1.5, (task, sft_peak, pref_peak)

    @pytest.mark.parametrize("task", _REFERENCE_USING)
    def test_the_peak_assertion_would_notice_a_second_instance(self, tmp_path, monkeypatch, task):
        """CONTROL for the test above. "DPO is not 2x SFT" means nothing unless
        the same harness can show what a second instance costs: the gate
        measured +730.44 MB against 730.44 MB of weights, i.e. exactly one
        copy."""
        implicit, stats = self._peak(tmp_path / "c", monkeypatch, task)
        forced, _ = self._peak(tmp_path / "d", monkeypatch, task, second_reference=True)
        assert forced > implicit, (implicit, forced)


class TestBitExactVsResident:
    """The rule every slot in the series inherits: a streamed run is bit-exact
    against the RESIDENT run of the same numerics. What changes per slot is the
    reference, not the standard — here it is a resident run of the same loss."""

    @pytest.mark.skipif(
        _mps_is_the_accelerator(),
        reason="MPS is untested for layer streaming (CUDA + CPU only)",
    )
    @pytest.mark.parametrize("task", _ALL_PREFERENCE)
    def test_loss_matches_a_resident_run_of_the_same_loss(self, tmp_path, monkeypatch, task):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model

        wrapper, resident, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task=task)
        _randomise_lora_b(wrapper.model)

        resident_peft = get_peft_model(
            resident,
            LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        copied = _sync_adapters(resident_peft, wrapper.model)
        assert copied > 0, "vacuous: no adapter tensors copied"

        _match_streamed_dtype(resident_peft, wrapper.model).eval()
        wrapper.model.eval()
        batch = _batch_on(wrapper.model, next(iter(wrapper.trainer.get_train_dataloader())))

        streamed_trainer = wrapper.trainer
        # Same trainer object, different model: the ONLY difference under test
        # is where the base weights come from.
        with torch.no_grad():
            streamed_loss = _loss_of(streamed_trainer, wrapper.model, batch)
            resident_loss = _loss_of(streamed_trainer, resident_peft, batch)
        diff = (streamed_loss - resident_loss).abs().max().item()
        assert diff == 0.0, f"{task}: streamed vs resident loss differs by {diff}"

    @pytest.mark.skipif(
        _mps_is_the_accelerator(),
        reason="MPS is untested for layer streaming (CUDA + CPU only)",
    )
    @pytest.mark.parametrize("task", _ALL_PREFERENCE)
    def test_layer_zero_adapter_receives_gradient(self, tmp_path, monkeypatch, task):
        """plan P2: a `detach()`/`no_grad()` anywhere in the base forward severs
        the graph. The lower adapters then never train while the loss still
        falls, because the upper ones still learn."""
        wrapper, _, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task=task)
        _randomise_lora_b(wrapper.model)
        wrapper.model.train()
        wrapper.model.zero_grad(set_to_none=True)
        batch = _batch_on(wrapper.model, next(iter(wrapper.trainer.get_train_dataloader())))
        loss = _loss_of(wrapper.trainer, wrapper.model, batch)
        loss.mean().backward()
        grads = [
            param.grad.abs().max().item()
            for name, param in wrapper.model.named_parameters()
            if "layers.0." in name and "lora_" in name and param.grad is not None
        ]
        assert grads and max(grads) > 0.0, (
            f"{task}: layer-0 adapter gradient is zero — the graph is severed"
        )


# ==========================================================================
# item 1 -- the pre-flight must not under-predict for a concatenating loss
# ==========================================================================
class TestVramPreflightAccountsForPairedRows:
    """DPO / ORPO / SimPO build their forward through `concatenated_inputs` +
    `torch.cat`, so **2 x batch_size** rows reach the model in ONE tensor.
    v0.72.3's estimator was validated on the property that it NEVER
    under-predicts; reusing it at 1x rows for these three would break exactly
    that, and on Windows the consequence is not an exception but a silent WDDM
    spill that makes the run an order of magnitude slower.

    KTO instead runs its KL batch as a SEPARATE forward, so its row count is 1x.
    """

    @pytest.mark.parametrize(
        "task,expected", [("sft", 1), ("dpo", 2), ("orpo", 2), ("simpo", 2), ("kto", 1)]
    )
    def test_rows_per_example_is_declared_per_task(self, task, expected):
        from soup_cli.trainer.dpo import DPOTrainerWrapper
        from soup_cli.trainer.kto import KTOTrainerWrapper
        from soup_cli.trainer.orpo import ORPOTrainerWrapper
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from soup_cli.trainer.simpo import SimPOTrainerWrapper

        cls = {
            "sft": SFTTrainerWrapper,
            "dpo": DPOTrainerWrapper,
            "orpo": ORPOTrainerWrapper,
            "simpo": SimPOTrainerWrapper,
            "kto": KTOTrainerWrapper,
        }[task]
        assert cls._STREAM_ROWS_PER_EXAMPLE == expected

    @pytest.mark.parametrize("task", ("dpo", "orpo", "simpo"))
    def test_the_budget_uses_the_multiplier(self, tmp_path, monkeypatch, task):
        """Behavioural, per wrapper: asserting the class attribute alone would
        miss a wrapper that reads the wrong one. The predicted peak for a paired
        loss at batch 1 must be the prediction for 2 rows, because that IS the
        tensor TRL builds."""
        from soup_cli.utils.layer_stream import estimate_stream_peak_vram

        captured = []
        import soup_cli.trainer.stream_setup as setup_mod

        real = (
            setup_mod.estimate_stream_peak_vram
            if hasattr(setup_mod, "estimate_stream_peak_vram")
            else estimate_stream_peak_vram
        )

        def spy(**kwargs):
            captured.append(kwargs["batch_size"])
            return real(**kwargs)

        monkeypatch.setattr("soup_cli.utils.layer_stream.estimate_stream_peak_vram", spy)
        _build_streamed_wrapper(tmp_path, monkeypatch, task=task)
        assert captured, "the VRAM pre-flight never ran"
        assert captured[0] == 2, (
            f"{task} at batch_size=1 budgeted for {captured[0]} rows; the "
            f"concatenated chosen+rejected tensor is 2"
        )


# ==========================================================================
# item 1 -- resource release
# ==========================================================================
class TestStreamRuntimeIsReleased:
    """v0.72.3's code review established this for SFT: `close()` after the
    training call is SKIPPED when training raises, and an OOM is realistic on
    exactly the cards this feature targets. On the disk tier that leaks one open
    shard handle per decoder layer."""

    @pytest.mark.parametrize("task", _ALL_PREFERENCE)
    def test_close_runs_even_when_training_raises(self, tmp_path, monkeypatch, task):
        wrapper, _, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task=task)
        closed = {"n": 0}
        real_close = wrapper._stream_runtime.close

        def counting_close():
            closed["n"] += 1
            return real_close()

        wrapper._stream_runtime.close = counting_close

        def boom(*_a, **_k):
            raise RuntimeError("CUDA out of memory")

        wrapper.trainer.train = boom
        with pytest.raises(RuntimeError, match="out of memory"):
            wrapper.train()
        assert closed["n"] >= 1, f"{task} leaked the streaming weight source"

    @pytest.mark.parametrize("task", _ALL_PREFERENCE)
    def test_close_is_a_noop_without_streaming(self, tmp_path, monkeypatch, task):
        """A non-streaming run must not acquire a `_stream_runtime` attribute
        and must not fail when the release hook runs anyway."""
        cls = _wrapper_for(task)
        import yaml

        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            yaml.safe_dump(
                {
                    "base": "sshleifer/tiny-gpt2",
                    "task": task,
                    "data": {"train": "train.jsonl"},
                    "training": {"quantization": "none"},
                    "output": str(tmp_path / "o"),
                }
            )
        )
        wrapper = cls(cfg, device="cpu")
        wrapper._close_stream_runtime()  # must not raise
        assert getattr(wrapper, "_stream_runtime", None) is None


# ==========================================================================
# the shared setup must be shared, not copied
# ==========================================================================
class TestStreamingSetupIsSharedNotCopied:
    """Four wrappers x ~270 lines of streaming setup is a drift machine: the
    NF4 pre-flight, the tier fallback and the fit refusal would each have five
    places to be fixed."""

    def test_every_streaming_wrapper_uses_the_one_mixin(self):
        from soup_cli.trainer.dpo import DPOTrainerWrapper
        from soup_cli.trainer.kto import KTOTrainerWrapper
        from soup_cli.trainer.orpo import ORPOTrainerWrapper
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from soup_cli.trainer.simpo import SimPOTrainerWrapper
        from soup_cli.trainer.stream_setup import StreamingSetupMixin

        for cls in (
            SFTTrainerWrapper,
            DPOTrainerWrapper,
            ORPOTrainerWrapper,
            SimPOTrainerWrapper,
            KTOTrainerWrapper,
        ):
            assert issubclass(cls, StreamingSetupMixin)
            # and does not shadow the shared implementation
            assert "_setup_streaming_transformers" not in vars(cls)

    def test_the_mixin_is_import_light(self):
        """`stream_setup` is imported by five trainer modules; a top-level torch
        there would be a wide blast radius. (The authority on the CLI startup
        property is tests/test_cli_startup_is_light.py — this is the cheap
        syntactic first line, per CLAUDE.md's evidence rule.)"""
        import ast
        import pathlib

        import soup_cli.trainer.stream_setup as mod

        tree = ast.parse(pathlib.Path(mod.__file__).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in {"torch", "peft", "trl"}
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in {"torch", "peft", "trl"}


# ==========================================================================
# findings the gate produced that would otherwise not be pinned
# ==========================================================================
class TestRowMultiplierIsLoadBearing:
    """The gate's own first attempt at this was NOT evidence: at vocab 64 the
    logits term is ~0, so budgeting 1x and 2x rows predict almost the same
    number and the check passed for either answer. Re-measured at vocab 32000
    the two are 71.3% apart."""

    def test_the_two_answers_are_far_apart_at_a_realistic_vocab(self):
        from soup_cli.utils.layer_stream import estimate_stream_peak_vram

        def predict(rows):
            return estimate_stream_peak_vram(
                layer_bytes=4_000_000,
                buffers=2,
                extras_bytes=32_000_000,
                adapter_params=300_000,
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=1024,
                n_layers=8,
                seq_len=128,
                batch_size=rows,
            )

        one, two = predict(1), predict(2)
        assert (two - one) / one > 0.15, (
            f"1x={one} 2x={two} are only {(two - one) / one:.1%} apart, so any "
            f"test of the multiplier at this shape would be vacuous"
        )

    def test_a_paired_loss_budgets_strictly_more_than_an_unpaired_one(self):
        """Behavioural consequence: the same batch_size costs more under a
        concatenating loss, because it IS a bigger tensor."""
        from soup_cli.trainer.dpo import DPOTrainerWrapper
        from soup_cli.trainer.kto import KTOTrainerWrapper
        from soup_cli.trainer.sft import SFTTrainerWrapper

        assert (
            DPOTrainerWrapper._STREAM_ROWS_PER_EXAMPLE > SFTTrainerWrapper._STREAM_ROWS_PER_EXAMPLE
        )
        # KTO does NOT concatenate — it runs the KL batch as a separate forward
        # — so it must NOT be lumped in with the paired losses.
        assert (
            KTOTrainerWrapper._STREAM_ROWS_PER_EXAMPLE == SFTTrainerWrapper._STREAM_ROWS_PER_EXAMPLE
        )

    def test_kto_budgets_its_real_batch(self, tmp_path, monkeypatch):
        """KTO's multiplier is 1, but it requires batch_size >= 2, so the budget
        must still see 2 rows. Asserting the multiplier alone would miss that."""
        captured = []
        from soup_cli.utils import layer_stream as ls

        real = ls.estimate_stream_peak_vram

        def spy(**kwargs):
            captured.append(kwargs["batch_size"])
            return real(**kwargs)

        monkeypatch.setattr(ls, "estimate_stream_peak_vram", spy)
        _build_streamed_wrapper(tmp_path, monkeypatch, task="kto")
        assert captured and captured[0] == 2, captured


class TestKtoNeedsMoreThanOneRow:
    """TRL refuses `per_device_train_batch_size == 1` for KTO outright ("the KL
    term will be equivalent to the implied reward"), so KTO is streamable ONLY
    because v0.72.3 lifted layer streaming's own batch-1 restriction. If a
    future change re-freezes batch to 1, KTO silently stops being usable."""

    def test_batch_two_is_accepted_by_the_streaming_schema(self, tmp_path):
        cfg = _stream_cfg(str(tmp_path / "m"), tmp_path / "o", task="kto", batch_size=2)
        assert cfg.training.batch_size == 2

    @pytest.mark.skipif(
        _mps_is_the_accelerator(),
        reason="MPS is untested for layer streaming (CUDA + CPU only)",
    )
    def test_kto_streams_at_batch_two(self, tmp_path, monkeypatch):
        """Runs EVERYWHERE, and tolerates exactly one known failure signature.

        v0.72.4 first gated this on `skipif(not cuda)` with the rationale that "a
        streamed model on CPU is a test convenience rather than a configuration".
        That was a generalisation applied to an unexplained failure, and it was
        wrong three ways. CI has no GPU runners (ubuntu/windows/macos), so the
        gate made this test dead in CI and alive only on the dev box, under the
        one torch where it happens to pass. The same CI run had
        `test_v07200.py::test_one_training_step_actually_runs` — the identical
        streamed `train()` for SFT — pass on that CPU runner. And executing this
        body on the dev box with CUDA masked passes too (torch 2.5.1). The
        variable is the torch version, not the device.

        The real defect is a streaming property: `Tensor on device cpu is not on
        the expected device meta!` comes from `check_same_device`, i.e. an op
        received a `meta` placeholder next to a real tensor. A meta placeholder
        is reachable from KTO's second (KL) forward, and newer torch decomposes
        more ops, which is why only the newer stack surfaces it. Tracked as #328
        rather than absorbed into a device rule.

        So: tolerate that one signature on CPU and nothing else. Any other
        exception, and the same signature on CUDA, is a hard failure — and when
        the leak is fixed this XPASSes instead of quietly staying skipped.
        """
        import math

        wrapper, _, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task="kto", batch_size=2)
        assert wrapper._stream_runtime.stats()["n_layers"] == 2
        wrapper.trainer.args.max_steps = 1
        try:
            wrapper.trainer.train()
        except RuntimeError as exc:
            if not _cuda_available() and "expected device meta" in str(exc):
                pytest.xfail(
                    "#328: known meta leak in KTO's KL forward on CPU under "
                    f"newer torch (this run: {_torch_version()})"
                )
            raise
        losses = [entry["loss"] for entry in wrapper.trainer.state.log_history if "loss" in entry]
        assert losses and math.isfinite(losses[0]), losses
        assert wrapper._stream_runtime.pool.loads > 0, "no layer was streamed"


class TestTheReferenceForwardActuallyHappens:
    """Memory-wise the reference is free; time-wise it is not. DPO traverses the
    stack three times per step (policy forward + reference forward + checkpoint
    recompute) against SFT's two — measured 1.52x layer reads on a 24-layer
    model. Pinned so an "optimisation" that silently drops or caches the
    reference forward cannot pass unnoticed."""

    @pytest.mark.skipif(
        _mps_is_the_accelerator(),
        reason="MPS is untested for layer streaming (CUDA + CPU only)",
    )
    @pytest.mark.parametrize("task", _REFERENCE_USING)
    def test_a_reference_using_loss_reads_more_layers_than_sft(self, tmp_path, monkeypatch, task):
        def reads_for(task, root):
            # The real device, deliberately: KTO's `get_batch_loss_metrics`
            # moves the batch to `self.accelerator.device` itself
            # (kto_trainer.py:1349), so pinning the model to CPU here would only
            # manufacture a device mismatch. This test counts reads; it needs no
            # exact arithmetic.
            wrapper, _, _ = _build_streamed_wrapper(root, monkeypatch, task=task, n_layers=4)
            _randomise_lora_b(wrapper.model)
            pool = wrapper._stream_runtime.pool
            counted = {"n": 0}
            original = pool.load_async

            def counting(idx, source, stream=None):
                counted["n"] += 1
                return original(idx, source, stream)

            pool.load_async = counting
            wrapper.model.train()
            wrapper.model.zero_grad(set_to_none=True)
            batch = _batch_on(wrapper.model, next(iter(wrapper.trainer.get_train_dataloader())))
            if task == "sft":
                # TRL's SFTTrainer has no `get_batch_loss_metrics`; its step is
                # a plain causal-LM forward + backward, which is the point of
                # the comparison anyway.
                ids = batch["input_ids"]
                out = wrapper.model(
                    input_ids=ids, attention_mask=batch.get("attention_mask"), labels=ids
                )
                out.loss.backward()
            else:
                _loss_of(wrapper.trainer, wrapper.model, batch).mean().backward()
            pool.load_async = original
            wrapper._close_stream_runtime()
            return counted["n"]

        sft_reads = reads_for("sft", tmp_path / "s")
        pref_reads = reads_for(task, tmp_path / "d")
        assert pref_reads > sft_reads, (
            f"{task} did {pref_reads} layer reads vs SFT's {sft_reads}: the "
            f"reference forward appears not to be running at all"
        )


class TestKtoBatchIsRefusedEarly:
    """TRL raises "Actual (not effective) batch size must be > 1" from
    `KTOTrainer.__init__`, which under streaming runs only AFTER the RAM
    pre-flight, the checkpoint sharding and (at `quantization: 4bit`) the NF4
    quantisation — minutes of disk I/O on a real base, to fail on a config that
    was invalid before any of it started."""

    def test_batch_one_is_refused_at_parse_time(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            _stream_cfg(str(tmp_path / "m"), tmp_path / "o", task="kto", batch_size=1)
        message = str(excinfo.value)
        assert "kto" in message.lower()
        assert "batch_size" in message

    def test_batch_two_still_parses(self, tmp_path):
        cfg = _stream_cfg(str(tmp_path / "m"), tmp_path / "o", task="kto", batch_size=2)
        assert cfg.training.batch_size == 2

    def test_the_other_streaming_tasks_are_unaffected(self, tmp_path):
        """Control: batch 1 is perfectly valid for the other four, so a gate
        that refused it everywhere would be a regression, not a fix."""
        for task in ("sft", "dpo", "orpo", "simpo"):
            cfg = _stream_cfg(str(tmp_path / "m"), tmp_path / "o", task=task, batch_size=1)
            assert cfg.training.batch_size == 1

    def test_non_streaming_kto_is_left_alone(self, tmp_path):
        """Scoped to streaming deliberately: resident KTO at batch 1 fails the
        same way, but that is pre-existing behaviour outside this slot, and
        widening the gate here could reject configs that parse today."""
        import yaml

        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            yaml.safe_dump(
                {
                    "base": "sshleifer/tiny-gpt2",
                    "task": "kto",
                    "data": {"train": "t.jsonl"},
                    "training": {"batch_size": 1, "quantization": "none"},
                    "output": str(tmp_path / "o"),
                }
            )
        )
        assert cfg.training.batch_size == 1

    def test_trl_itself_still_refuses_batch_one(self, tmp_path):
        """Pins the UPSTREAM behaviour our schema gate mirrors.

        The gate duplicates a TRL threshold. This box runs trl 0.19.1 while CI
        runs 1.9.2 (#323), so if a TRL version drops or changes the requirement
        our gate would silently refuse configs that upstream now accepts. Assert
        against the INSTALLED trl rather than against a version number, so
        whichever one CI has is the one that gets checked.
        """
        import torch
        from trl import KTOConfig, KTOTrainer

        weights, _, _ = _tiny_llama_dir(tmp_path)
        _write_tiny_tokenizer(weights)
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(weights, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(weights)
        dataset = Dataset.from_list(_kto_rows(4))
        from soup_cli.trainer._trl_compat import prompt_length_kwargs

        args = KTOConfig(
            output_dir=str(tmp_path / "o"),
            per_device_train_batch_size=1,
            report_to=[],
            max_length=32,
            # #326 — incidental setup, not what this test asserts. trl removed
            # `max_prompt_length` from KTOConfig at 0.27.0, so hard-coding it
            # made this test raise TypeError there and stop reaching the
            # batch-size check it exists to pin.
            **prompt_length_kwargs(KTOConfig, 16),
            # Newer TRL configs enable bf16 by default, which a CPU-only runner
            # rejects with "Your setup doesn't support bf16/gpu" BEFORE reaching
            # the batch-size check this test exists to pin.
            bf16=False,
        )
        with pytest.raises(ValueError, match="batch size"):
            KTOTrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                processing_class=tokenizer,
            )


class TestKtoReferenceIsAlsoTheDisabledAdapter:
    """CRITICAL gap found in review: KTO's reference forward is
    `self.forward(self.model, batch)` (`kto_trainer.py:1400`) — the trainer's
    OWN bound model, ignoring whatever `model` argument reaches
    `get_batch_loss_metrics`. So a streamed-vs-resident loss comparison computes
    the SAME reference term in both arms and exercises only the policy forward.
    KTO's reference path therefore needs its own direct assertion, the analogue
    of the DPO one."""

    def _forwards(self, wrapper):
        import torch

        trainer = wrapper.trainer
        batch = _batch_on(wrapper.model, next(iter(trainer.get_train_dataloader())))
        wrapper.model.eval()
        with torch.no_grad():
            policy = trainer.forward(wrapper.model, batch)[0]
            with trainer.null_ref_context():
                reference = trainer.forward(wrapper.model, batch)[0]
        return policy, reference

    def test_reference_differs_from_policy(self, tmp_path, monkeypatch):
        wrapper, _, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task="kto", device="cpu")
        _randomise_lora_b(wrapper.model)
        policy, reference = self._forwards(wrapper)
        assert (policy - reference).abs().max().item() > 1e-4, (
            "KTO's null_ref_context() is a no-op through the streamed layer's "
            "functional_call, so the reference IS the policy"
        )

    def test_the_difference_really_comes_from_the_adapter(self, tmp_path, monkeypatch):
        """CONTROL: with `lora_B = 0` the adapter contributes nothing, so
        disabling it must change nothing at all."""
        import torch

        wrapper, _, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task="kto", device="cpu")
        with torch.no_grad():
            for name, param in wrapper.model.named_parameters():
                if "lora_B" in name:
                    param.zero_()
        policy, reference = self._forwards(wrapper)
        assert (policy - reference).abs().max().item() == 0.0


class TestNf4CombinesWithEveryPreferenceLoss:
    """The schema allows `quantization: 4bit` with all four, and the NF4
    `total_params` override was hand-copied into each wrapper (PEFT sizes a
    `meta` Params4bit placeholder as `numel * 2 * itemsize`, over-reporting a
    streamed NF4 model ~6.5x — measured 878,154,048 vs a true 134,515,008 on
    SmolLM2-135M). A name drift in any one copy is otherwise invisible."""

    @pytest.mark.skipif(
        _mps_is_the_accelerator(),
        reason="bitsandbytes has no 4-bit MPS kernels",
    )
    @pytest.mark.parametrize("task", _ALL_PREFERENCE)
    def test_nf4_streaming_sets_up_and_reports_honest_parameters(self, tmp_path, monkeypatch, task):
        pytest.importorskip("bitsandbytes")
        wrapper, _, _ = _build_streamed_wrapper(
            tmp_path, monkeypatch, task=task, quantization="4bit"
        )
        assert wrapper.trainer is not None
        honest = wrapper._stream_runtime.total_params
        assert honest > 0
        peft_says = wrapper.model.get_nb_trainable_parameters()[1]
        assert honest != peft_says, (
            "PEFT's total now agrees with the sharder's, so the override this "
            "test guards is either unnecessary or no longer wired"
        )
        real = sum(p.numel() for p in wrapper.model.parameters() if not p.is_meta)
        assert honest > real, (honest, real)
        wrapper._close_stream_runtime()

    def test_auto_batch_still_reports_its_own_reason_for_kto(self, tmp_path):
        """The `batch_size='auto'` refusal and the KTO `>= 2` refusal are two
        different guards over the same field. `auto` is not an int, so the KTO
        branch must not swallow it and report the wrong reason."""
        with pytest.raises(ValueError) as excinfo:
            _stream_cfg(str(tmp_path / "m"), tmp_path / "o", task="kto", batch_size="auto")
        message = str(excinfo.value)
        assert "auto" in message
        assert "RESIDENT" in message or "probe" in message
