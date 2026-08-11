"""v0.73.0 (#331) — the repair: keep streamed NF4 weights out of ``MatMul4Bit``.

WHY THIS EXISTS

``bitsandbytes.autograd._functions.MatMul4Bit.forward`` stashes the packed weight
and the ``quant_state`` on ``ctx`` as PLAIN ATTRIBUTES::

    ctx.state = quant_state
    ctx.tensors = (None, B)

They never go through ``save_for_backward``, so ``torch.utils.checkpoint`` cannot
discard and recompute them. The reference is taken in the forward, it ALIASES the
streaming buffer pool, and it is read in the backward after that slot has already
been refilled with a different layer.

Symptom, measured on 8xH100 against a resident NF4 reference (see
``benchmarks/gate-h100-validation.md``): the forward stays bit-exact, the loss curve
looks healthy, and the gradients are wrong on every layer except the last
``stream_buffers``. It bites NF4 above roughly 165 MiB per layer, so 32B and 72B, and
never bf16 — which goes through ``MmBackward0`` with a normal ``save_for_backward``.

Both de-aliasing repairs were measured and rejected: because bnb holds the reference
across the whole forward-to-backward span, ANY de-aliasing keeps one copy of every
layer alive for that span and costs O(model), not O(window). On real 32B that was
peak VRAM 4 220 -> 19 720 MiB, which deletes the feature's premise.

THE REPAIR: do not send a streamed NF4 weight through ``MatMul4Bit`` at all.
Dequantise inside the checkpointed region and use a native matmul. ``F.linear``
saves the dequantised weight properly, so checkpointing DOES discard and recompute
it, and the transient lives only inside the recomputed block — O(window) by
construction.

WHY THIS IS NOT A NUMERICS CHANGE AT TRAINING SHAPES (STEP 13 of the record)

``bitsandbytes::gemm_4bit`` dispatches on M (tokens)::

    _gemm_4bit_custom_max_m = 1536      # CUDA
    if M > _gemm_4bit_custom_max_m: -> _dequant_linear_fallback

and on real projection shapes it takes that fallback at every M measured from 8 to
2048. So at 8B/32B shapes bitsandbytes is ALREADY doing what this repair does; the
repair makes it explicit and moves it inside the checkpoint. Measured over 423 rows,
the gradient is bit-exact in every one of them, worst ``max_abs`` exactly 0.0 — by
construction, since bnb's own backward is already dequantise-then-matmul.

The forward differs only where the fused kernel genuinely runs (small M), and then by
one bf16 ulp: worst 3.95e-3 relative to scale against 2^-8 = 3.9e-3.
"""

import os
import sys
import types

import pytest

# The NF4 streamed/resident pair builders live in test_v07202 and are deliberately
# NOT duplicated here: two copies of a fixture drift, and this file's whole subject
# is a numerical comparison between the two models they build.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from test_v07202 import _nf4_stream, _resident_nf4  # noqa: E402


def _count_matmul_4bit(monkeypatch):
    """Count ``bnb.matmul_4bit`` calls.

    ``bitsandbytes/nn/modules.py`` does ``import bitsandbytes as bnb`` and then calls
    ``bnb.matmul_4bit(...)``, i.e. the attribute is looked up on the module object at
    CALL time, so patching the module attribute intercepts it.
    """
    import bitsandbytes as bnb

    calls = {"n": 0}
    real = bnb.matmul_4bit

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(bnb, "matmul_4bit", counting)
    return calls


class TestStreamedNF4AvoidsMatMul4Bit:
    """The repair, and the control that makes it mean something.

    ``0 calls`` on its own is equally consistent with "the counter never
    intercepted anything" — which is exactly how an earlier path control in this
    investigation was fooled. The resident control must COUNT, in the same test
    session, or the streamed assertion proves nothing.
    """

    def test_resident_nf4_does_reach_matmul_4bit(self, tmp_path, monkeypatch):
        """CONTROL. Without this, the assertion below is unfalsifiable."""
        _, _, weights, _, _ = _nf4_stream(tmp_path)
        resident = _resident_nf4(weights)
        calls = _count_matmul_4bit(monkeypatch)
        with torch.no_grad():
            resident(input_ids=torch.randint(0, 64, (1, 8)))
        assert calls["n"] > 0, (
            "the counter did not intercept bnb.matmul_4bit at all, so it cannot "
            "detect the streamed path avoiding it either"
        )

    def test_streamed_nf4_forward_does_not_reach_matmul_4bit(self, tmp_path, monkeypatch):
        model, _, _, _, _ = _nf4_stream(tmp_path)
        calls = _count_matmul_4bit(monkeypatch)
        with torch.no_grad():
            model(input_ids=torch.randint(0, 64, (1, 8)))
        assert calls["n"] == 0, (
            f"streamed NF4 still routed {calls['n']} call(s) through MatMul4Bit, which "
            "captures the packed weight outside save_for_backward and therefore aliases "
            "the buffer pool across the checkpoint boundary (#331)"
        )


# ==========================================================================
# #328 — HF gradient checkpointing must stay OFF on a streamed preference run
# ==========================================================================
PREFERENCE_TASKS = ("dpo", "orpo", "simpo", "kto")


class TestStreamedPreferenceLossesDisableHfGradientCheckpointing:
    """``StreamedDecoderLayer`` already wraps every layer in
    ``checkpoint(use_reentrant=False)``. ``should_enable_hf_gradient_checkpointing``
    exists so the HF Trainer does not ALSO checkpoint a streamed model — but only
    ``sft.py`` ever consulted it. The four preference wrappers never did, and TRL's
    ``DPOConfig`` / ``ORPOConfig`` / ``CPOConfig`` / ``KTOConfig`` all default
    ``gradient_checkpointing=True``, so it arrived switched on through the default
    rather than through anything the user wrote.

    On torch 2.5.1 that is the documented ~1.5x silent double-recompute. On torch
    2.13 + CUDA it is fatal: HF's checkpoint wraps the INNER ``LlamaDecoderLayer``,
    whose recompute runs long after ``functional_call``'s reparametrisation context
    has exited and restored the ``meta`` placeholders, so the recompute multiplies a
    ``meta`` ``input_layernorm.weight`` by a real ``cuda`` activation::

        RuntimeError: Tensor on device cuda:0 is not on the expected device meta!

    Measured on 8xH100: all four preference losses fail, SFT passes, and forcing the
    flag back off makes the DPO failure disappear with the wrapper's own checkpoint
    still active — which is what makes HF's checkpointing the cause rather than a
    correlate.
    """

    @pytest.mark.parametrize("task", PREFERENCE_TASKS)
    def test_streaming_turns_hf_gradient_checkpointing_off(self, task, tmp_path, monkeypatch):
        from test_v07204 import _build_streamed_wrapper

        wrapper, _, _ = _build_streamed_wrapper(
            tmp_path, monkeypatch, task=task, device="cpu"
        )
        assert wrapper.trainer.args.gradient_checkpointing is False, (
            f"{task}: HF gradient checkpointing reached the Trainer switched ON for a "
            "streamed run, so HF checkpoints the inner decoder layer as well as the "
            "streaming wrapper. On torch 2.13 CUDA that recompute sees the restored "
            "meta placeholders and dies with 'expected device meta' (#328)"
        )

    @pytest.mark.parametrize("asked", [True, False])
    def test_a_non_streaming_run_gets_what_the_config_asked_for(
        self, asked, tmp_path, monkeypatch
    ):
        """CONTROL, and the second half of the same defect.

        Two things have to be true at once, and only asserting one of them would
        hide the other:

        * the repair must switch HF checkpointing off ONLY where streaming already
          checkpoints — pinning the flag to a constant ``False`` would satisfy the
          streamed test above while silently removing checkpointing from every
          ordinary DPO run;
        * the wrapper must not inherit TRL's default either way.

        Stating it as "the config value survives to the Trainer" is deliberately
        version-independent. TRL's own default for these configs is NOT stable —
        measured ``False`` on trl 0.19.1 and ``True`` on trl 0.26.2 — so asserting
        the default directly would be red on one supported stack and green on the
        other while the actual bug (the wrapper never setting it) went unnoticed on
        both. On the older stack a user's explicit ``gradient_checkpointing: true``
        was being silently dropped; on the newer one TRL's ``True`` arrived
        uninvited. One cause, two opposite symptoms.
        """
        from test_v07204 import _build_streamed_wrapper

        wrapper, _, _ = _build_streamed_wrapper(
            tmp_path,
            monkeypatch,
            task="dpo",
            device="cpu",
            stream_layers=False,
            gradient_checkpointing=asked,
        )
        assert wrapper.trainer.args.gradient_checkpointing is asked


# ==========================================================================
# the CI fixture must exercise the kernel path production actually takes
# ==========================================================================
def _cuda_available():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="the fused-kernel window is a CUDA dispatch")
class TestFixtureIsOutsideTheFusedKernelWindow:
    """The streamed-vs-resident gate was comparing a code path no real model uses.

    ``bitsandbytes::gemm_4bit`` dispatches on M. Measured on this fixture (hidden 64,
    so every projection is [64x64] or [32x64]), on an H100:

    ==========  ==========================  ================
    tokens      resident calls via fallback  streamed-vs-resident
    ==========  ==========================  ================
    8, 16, 32   0 of 14 — all FUSED          3.906250e-03 (= 2^-8, one bf16 ulp)
    64 and up   14 of 14 — all fallback      0.0
    ==========  ==========================  ================

    Real 8B / 32B projections take the fallback at every M from 8 to 2048, so the
    fused branch is one the shipped models never reach — and the gate was sitting in
    it, at 16 tokens. The window boundary and the bit-exactness boundary were
    measured to coincide EXACTLY at 64, which is what identifies the dispatch as the
    cause of the ulp rather than something else that merely correlates with size.

    This test pins the fixture outside that window, so shrinking the sequence back
    for speed fails here with the reason attached instead of silently returning the
    gate to the wrong kernel.
    """

    def test_every_resident_4bit_call_takes_the_dequant_fallback(self, tmp_path):
        import bitsandbytes as bnb

        bnb_ops = pytest.importorskip(
            "bitsandbytes.backends.cuda.ops",
            reason="the M-dispatch lives in the CUDA backend module",
        )
        if not hasattr(bnb_ops, "_dequant_linear_fallback"):
            pytest.skip(
                "this bitsandbytes has no _dequant_linear_fallback (added around "
                "0.50); the branch it names cannot be counted here. The property "
                "still holds — TestNF4ParityOnCuda asserts the observable "
                "consequence, an exactly-zero streamed-vs-resident difference."
            )

        model, _, weights, _, _ = _nf4_stream(tmp_path, device="cuda", dtype="bfloat16")
        resident = _resident_nf4(weights, dtype="bfloat16", device=0)
        del model

        counts = {"fallback": 0, "total": 0}
        real_fb = bnb_ops._dequant_linear_fallback
        real_matmul = bnb.matmul_4bit

        def counting_fb(*args, **kwargs):
            counts["fallback"] += 1
            return real_fb(*args, **kwargs)

        def counting_matmul(*args, **kwargs):
            counts["total"] += 1
            return real_matmul(*args, **kwargs)

        ids = torch.randint(
            0, 64, (1, 128), generator=torch.Generator().manual_seed(11)
        ).cuda()
        bnb_ops._dequant_linear_fallback = counting_fb
        bnb.matmul_4bit = counting_matmul
        try:
            with torch.no_grad():
                resident(input_ids=ids, attention_mask=torch.ones_like(ids))
            torch.cuda.synchronize()
        finally:
            bnb_ops._dequant_linear_fallback = real_fb
            bnb.matmul_4bit = real_matmul

        assert counts["total"] > 0, (
            "the counter never intercepted bnb.matmul_4bit, so it cannot tell which "
            "branch was taken either"
        )
        assert counts["fallback"] == counts["total"], (
            f"{counts['total'] - counts['fallback']} of {counts['total']} resident "
            "4-bit calls took the FUSED kernel at the token count the CUDA gates use. "
            "Real models never take that branch, so the gate would be pinning a path "
            "no user runs — and the streamed arm, which always dequantises, differs "
            "from it by one bf16 ulp. Raise the fixture's sequence length."
        )


# ==========================================================================
# #78 — two defects the FlashAttention / Liger benchmark surfaced
# ==========================================================================
def _sft_wrapper(tmp_path, monkeypatch, **training):
    """A NON-streaming SFT wrapper built through the real ``setup()`` path.

    Deliberately not reusing ``_build_streamed_wrapper``: both defects below are
    about what reaches TRL's ``SFTConfig`` on an ORDINARY run, which is what every
    user gets, and routing through the streaming helper would test a narrower path
    than the one that is broken.
    """
    import yaml
    from test_v07202 import _tiny_llama_dir, _write_tiny_tokenizer

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper

    weights, _, _ = _tiny_llama_dir(tmp_path)
    _write_tiny_tokenizer(weights)
    monkeypatch.chdir(tmp_path)
    max_length = training.pop("max_length", 64)
    tcfg = {
        "batch_size": 1,
        "quantization": "none",
        "epochs": 1,
        "logging_steps": 1,
        "save_steps": 1000,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    tcfg.update(training)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": weights,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": {
                    "train": "train.jsonl",
                    "max_length": max_length,
                    "chat_template": "chatml",
                },
                "training": tcfg,
                "output": str(tmp_path / "out"),
            }
        )
    )
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": [{"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]}] * 4})
    return wrapper


class TestDataMaxLengthReachesTrl:
    """``data.max_length`` above 1024 was silently ignored on every SFT run.

    Soup builds a ``transformers.TrainingArguments``, not a ``trl.SFTConfig``.
    ``SFTTrainer.__init__`` converts one to the other with
    ``SFTConfig(**args.to_dict())``, and ``max_length`` is an SFT-specific field
    that ``TrainingArguments`` does not carry — so it took ``SFTConfig``'s own
    default of 1024 no matter what the config said.

    Measured on 8xH100 before the fix: ``data.max_length=512`` gave 512 tokens per
    sample, ``data.max_length=4096`` gave **1024**. Every long-context SFT run was
    quietly truncated, and no warning was emitted. It also caps how much
    FlashAttention can ever help, since its advantage grows with sequence length.
    """

    def test_max_length_above_the_trl_default_survives(self, tmp_path, monkeypatch):
        wrapper = _sft_wrapper(tmp_path, monkeypatch, max_length=2048)
        assert wrapper.trainer.args.max_length == 2048, (
            f"TRL received max_length={wrapper.trainer.args.max_length} where the "
            "config asked for 2048 — sequences are being truncated silently"
        )

    def test_max_length_below_the_trl_default_survives_too(self, tmp_path, monkeypatch):
        """CONTROL. A fix that pinned max_length to some large constant would pass
        the test above while breaking every short-sequence run."""
        wrapper = _sft_wrapper(tmp_path, monkeypatch, max_length=128)
        assert wrapper.trainer.args.max_length == 128


class TestLigerTellsTrlItIsOn:
    """``training.use_liger: true`` crashed at step 0 — the feature never ran.

    Soup patches the model with ``apply_liger_kernel()`` but never set
    ``TrainingArguments.use_liger_kernel``, which is the flag TRL and HF read to
    know the fused path will return ``logits=None``. TRL guards its entropy metric
    with ``if not self.args.use_liger_kernel:``; with the flag left False the guard
    cannot fire and ``entropy_from_logits(None)`` raises
    ``AttributeError: 'NoneType' object has no attribute 'shape'``.

    Reproduced on trl 0.26.2 AND trl 0.19.1 — i.e. across the whole supported pin,
    so this was never version-specific bad luck.
    """

    def test_flag_is_set_when_liger_is_requested(self, tmp_path, monkeypatch):
        pytest.importorskip("liger_kernel", reason="use_liger needs the liger extra")
        wrapper = _sft_wrapper(tmp_path, monkeypatch, use_liger=True)
        assert wrapper.trainer.args.use_liger_kernel is True

    def test_flag_stays_off_when_liger_is_not_requested(self, tmp_path, monkeypatch):
        """CONTROL. Setting it unconditionally would make TRL expect None logits
        from an unpatched model, which is the same crash with the arms swapped."""
        wrapper = _sft_wrapper(tmp_path, monkeypatch)
        assert wrapper.trainer.args.use_liger_kernel is False


# ==========================================================================
# #76 — SGLang generation was broken for every request
# ==========================================================================
class TestSglangDecodesItsRuntimeResponse:
    """`--backend sglang` loaded and then returned 500 on every generation.

    ``sglang.Runtime.generate`` ends with ``return json.dumps(response.json())`` —
    a **string**. ``utils/sglang.py`` did ``response["text"]`` on it, so every
    request raised ``TypeError: string indices must be integers``. Deterministic,
    not a race, and the backend had genuinely never been run: verified on
    sglang 0.5.16 on an H100.

    Older sglang returned a dict, so the decode has to accept BOTH shapes rather
    than swapping one hard assumption for another.
    """

    def test_a_json_string_is_decoded(self):
        import json

        from soup_cli.utils.sglang import decode_sglang_response

        payload = {"text": " Paris.", "meta_info": {"prompt_tokens": 17}}
        out = decode_sglang_response(json.dumps(payload))
        assert out["text"] == " Paris."
        assert out["meta_info"]["prompt_tokens"] == 17

    def test_a_dict_still_works(self):
        """CONTROL. The older sglang shape must keep working — a fix that only
        handled strings would break every previously-working version."""
        from soup_cli.utils.sglang import decode_sglang_response

        payload = {"text": " Paris.", "meta_info": {"prompt_tokens": 17}}
        assert decode_sglang_response(payload) is payload

    def test_an_undecodable_response_raises_something_actionable(self):
        from soup_cli.utils.sglang import decode_sglang_response

        with pytest.raises(ValueError, match="SGLang"):
            decode_sglang_response("not json at all")


# ==========================================================================
# #77 — `soup train --gpus N` never launched at all
# ==========================================================================
class TestMultiGpuLaunchArgvIsRunnable:
    """The whole multi-GPU entry point was dead.

    ``accelerate launch`` takes a **script path** (or ``--module NAME``), and Soup
    handed it ``sys.executable``. accelerate then opened the Python ELF binary and
    parsed it as source, so every rank died before the trainer existed::

        File "/root/venv/bin/python", line 1
          ELF
        SyntaxError: source code cannot contain null bytes

    Measured on 4xH100: every arm of the #77 matrix had to be launched by hand.
    The failure is total and immediate, which is why it is worth a test that reads
    the argv rather than one that mocks the launch.
    """

    def test_the_python_dash_m_form_becomes_accelerates_module_flag(self):
        import sys as _sys

        from soup_cli.utils.launcher import build_accelerate_argv

        argv = build_accelerate_argv(
            num_processes=4,
            script_args=[_sys.executable, "-m", "soup_cli.cli", "train", "-c", "s.yaml"],
        )
        assert _sys.executable not in argv, (
            "the Python interpreter is still being passed where accelerate expects "
            "a training script; accelerate will parse the ELF binary as source"
        )
        assert "--module" in argv
        assert argv[argv.index("--module") + 1] == "soup_cli.cli"
        assert argv[-3:] == ["train", "-c", "s.yaml"]
        assert argv[:2] == ["accelerate", "launch"]

    def test_a_real_script_path_is_left_alone(self):
        """CONTROL. Translating unconditionally would break the documented
        ``accelerate launch train.py`` form, which is a script path and must stay
        positional."""
        from soup_cli.utils.launcher import build_accelerate_argv

        argv = build_accelerate_argv(
            num_processes=2, script_args=["train.py", "--config", "s.yaml"]
        )
        assert "--module" not in argv
        assert argv[-3:] == ["train.py", "--config", "s.yaml"]

    def test_single_process_is_still_directly_executable(self):
        """CONTROL. At one process there is no accelerate wrapper and the argv is
        exec'd as-is, so the interpreter form must survive untouched — turning it
        into ``--module`` there would produce a command with no program to run."""
        import sys as _sys

        from soup_cli.utils.launcher import build_accelerate_argv

        argv = build_accelerate_argv(
            num_processes=1, script_args=[_sys.executable, "-m", "soup_cli.cli", "train"]
        )
        assert argv == [_sys.executable, "-m", "soup_cli.cli", "train"]


# ==========================================================================
# #335 — use_fsdp2_compile wrote an adapter that reloads as all zeros
# ==========================================================================
def _write_adapter(directory, prefix=""):
    """Write a two-tensor LoRA adapter, optionally through torch.compile's prefix.

    Module-level so #351's checkpoint tests below build the file the same way
    #335's do. Both are assertions about one exact key spelling, and two copies
    of the fixture that produces it would drift.
    """
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        f"{prefix}base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
            torch.ones(4, 8),
        f"{prefix}base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
            torch.full((8, 4), 0.007),
    }
    path = directory / "adapter_model.safetensors"
    save_file(tensors, str(path))
    return path


class TestCompiledAdapterKeysAreCanonical:
    """A run that completes, exits 0, and writes a file that silently does nothing.

    Under ``training.use_fsdp2_compile`` every saved key carried torch.compile's
    wrapper prefix::

        _orig_mod.base_model.model...     # written with compile
        base_model.model...               # written without it

    The tensors ARE trained (max|lora_B| 7.0e-3 measured), but
    ``PeftModel.from_pretrained`` matches none of them, emits only
    ``UserWarning: Found missing adapter keys``, and leaves ``lora_B`` at its zero
    init — **0 of 96 non-zero**, reproduced 3/3 on 4xH100, against 96/96 for the
    paired non-compile run.

    This is the same failure shape v0.72.1 fixed for the streaming wrapper's
    ``.inner.`` segment, and it is the worst one this project has: nothing raises.
    """

    def test_the_compile_prefix_is_stripped(self, tmp_path):
        from safetensors.torch import load_file

        from soup_cli.utils.peft_wiring import strip_compile_prefix

        path = _write_adapter(tmp_path / "a", prefix="_orig_mod.")
        changed = strip_compile_prefix(str(tmp_path / "a"))
        assert changed == 2, f"expected both keys rewritten, got {changed}"
        keys = set(load_file(str(path)))
        assert not any(k.startswith("_orig_mod.") for k in keys), keys
        assert all(k.startswith("base_model.model.") for k in keys), keys

    def test_the_values_survive_the_rewrite(self, tmp_path):
        """Renaming must not quietly re-initialise anything — the whole defect is
        that trained numbers stop being found, so a fix that loses them is the
        same bug with a different mechanism."""
        from safetensors.torch import load_file

        from soup_cli.utils.peft_wiring import strip_compile_prefix

        path = _write_adapter(tmp_path / "a", prefix="_orig_mod.")
        strip_compile_prefix(str(tmp_path / "a"))
        loaded = load_file(str(path))
        b = loaded["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
        assert float(b.abs().max()) == pytest.approx(0.007), "trained values lost"

    def test_a_normal_adapter_is_untouched(self, tmp_path):
        """CONTROL. Rewriting unconditionally would churn every ordinary run's
        adapter file for nothing, and any bug in the rewrite would then reach
        users who never enabled compile."""
        from safetensors.torch import load_file

        from soup_cli.utils.peft_wiring import strip_compile_prefix

        path = _write_adapter(tmp_path / "a")
        before = set(load_file(str(path)))
        assert strip_compile_prefix(str(tmp_path / "a")) == 0
        assert set(load_file(str(path))) == before

    def test_a_missing_adapter_file_is_not_an_error(self, tmp_path):
        """Full fine-tuning writes no adapter; the normaliser must not turn that
        into a crash at the end of a completed run."""
        from soup_cli.utils.peft_wiring import strip_compile_prefix

        (tmp_path / "empty").mkdir()
        assert strip_compile_prefix(str(tmp_path / "empty")) == 0


# ==========================================================================
# #351: the repair above ran on the final save only
# ==========================================================================
def _fire_on_save(callback, output_dir, step, *, should_save=True, world_zero=None):
    """Drive one ``on_save`` the way HF's CallbackHandler does.

    ``args`` / ``state`` are the two attributes the callback reads, so a
    SimpleNamespace is the whole Trainer this needs, the same shape
    ``tests/test_hf_integration.py`` uses to drive ``HFPushCallback.on_save``.

    ``world_zero`` defaults to following ``should_save``, which is what every
    run Soup can produce today does. Pass it explicitly to drive them apart, the
    way ``save_on_each_node`` does.
    """
    args = types.SimpleNamespace(output_dir=str(output_dir), should_save=should_save)
    state = types.SimpleNamespace(
        global_step=step,
        is_world_process_zero=should_save if world_zero is None else world_zero,
    )
    control = types.SimpleNamespace()
    callback.on_save(args, state, control)


class TestCheckpointAdaptersAreCanonicalToo:
    """#335's repair fires once, after the final ``save_model``. The
    ``checkpoint-*`` directories written on the way there keep the prefix.

    Both are written by the SAME ``Trainer.save_model`` (``_save_checkpoint``
    calls it with ``output_dir=<run>/checkpoint-N``), so they come out identical
    and only the last one was ever normalised. Measured at 70B on 8xH100
    (``benchmarks/gate-h100-validation.md``, STEP 28): 320 canonical keys in the
    output root, 320 prefixed ones in ``checkpoint-100``.

    Resuming is the case that decides how bad this is. ``from_pretrained`` at
    least warns. ``Trainer._load_from_checkpoint`` calls
    ``model.load_adapter(...)`` and drops the return value, and ``load_adapter``
    deliberately does not warn: it hands the missing keys back in the load
    result instead, which nothing reads. The unexpected ``_orig_mod.`` keys go
    to ``load_state_dict(strict=False)``, which discards them without a word. So
    a resumed run continues from a re-zeroed ``lora_B`` in total silence: the
    #335 failure shape with the last warning removed.
    """

    def test_a_checkpoint_written_under_compile_is_normalised(self, tmp_path):
        from safetensors.torch import load_file

        from soup_cli.utils.peft_wiring import build_compile_prefix_callback

        path = _write_adapter(tmp_path / "checkpoint-100", prefix="_orig_mod.")
        _fire_on_save(build_compile_prefix_callback(), tmp_path, 100)

        keys = set(load_file(str(path)))
        assert not any(k.startswith("_orig_mod.") for k in keys), keys
        assert all(k.startswith("base_model.model.") for k in keys), keys

    def test_the_trained_values_survive_in_a_checkpoint(self, tmp_path):
        """The standard #335's repair was held to. A rename that loses the
        numbers is the same defect wearing a different mechanism."""
        from safetensors.torch import load_file

        from soup_cli.utils.peft_wiring import build_compile_prefix_callback

        path = _write_adapter(tmp_path / "checkpoint-100", prefix="_orig_mod.")
        _fire_on_save(build_compile_prefix_callback(), tmp_path, 100)

        loaded = load_file(str(path))
        b = loaded["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
        assert float(b.abs().max()) == pytest.approx(0.007), "trained values lost"

    def test_a_repaired_checkpoint_is_byte_identical_to_an_uncompiled_one(
        self, tmp_path
    ):
        """What ``resume_from_checkpoint`` actually needs.

        A real resume needs FSDP2 and several GPUs, but the thing being resumed
        is a file, and peft matches it by name. So the assertion that carries
        the same weight off-GPU is that the repaired checkpoint is
        indistinguishable from what the paired non-compile run writes: same
        keys, same values. Anything peft finds in one it finds in the other.
        """
        from safetensors.torch import load_file

        from soup_cli.utils.peft_wiring import build_compile_prefix_callback

        repaired = _write_adapter(tmp_path / "checkpoint-100", prefix="_orig_mod.")
        _fire_on_save(build_compile_prefix_callback(), tmp_path, 100)
        reference = _write_adapter(tmp_path / "no-compile")

        got, want = load_file(str(repaired)), load_file(str(reference))
        assert set(got) == set(want)
        for key, tensor in want.items():
            assert torch.equal(got[key], tensor), key
        assert all(float(v.abs().max()) > 0 for k, v in got.items() if "lora_B" in k)

    def test_only_the_rank_that_wrote_the_file_rewrites(self, tmp_path):
        """``save_model`` writes the adapter only where ``args.should_save`` is
        true, but ``on_save`` is dispatched on EVERY rank. Unguarded, the 8 ranks
        of the run this was measured on would all rewrite one file at once."""
        from safetensors.torch import load_file

        from soup_cli.utils.peft_wiring import build_compile_prefix_callback

        path = _write_adapter(tmp_path / "checkpoint-100", prefix="_orig_mod.")
        before = set(load_file(str(path)))
        _fire_on_save(build_compile_prefix_callback(), tmp_path, 100, should_save=False)
        assert set(load_file(str(path))) == before

    def test_a_writing_rank_that_is_not_world_zero_still_repairs(self, tmp_path):
        """The reason the guard reads ``args.should_save`` and not
        ``state.is_world_process_zero``.

        ``TrainingArguments.should_save`` is ``local_process_index == 0`` when
        ``save_on_each_node`` is set and ``process_index == 0`` otherwise, and
        ``save_model`` gates the write on it. So under ``save_on_each_node`` the
        local rank 0 of every node after the first writes a checkpoint while
        reporting ``is_world_process_zero=False``, and a guard on world-zero
        would leave exactly those nodes holding an unrepaired adapter. Soup sets
        no such flag today, which is what makes this the kind of thing that goes
        unnoticed until someone does.
        """
        from safetensors.torch import load_file

        from soup_cli.utils.peft_wiring import build_compile_prefix_callback

        path = _write_adapter(tmp_path / "checkpoint-100", prefix="_orig_mod.")
        _fire_on_save(
            build_compile_prefix_callback(),
            tmp_path,
            100,
            should_save=True,
            world_zero=False,
        )

        keys = set(load_file(str(path)))
        assert not any(k.startswith("_orig_mod.") for k in keys), keys

    def test_a_checkpoint_without_an_adapter_is_not_an_error(self, tmp_path):
        """Full fine-tuning checkpoints carry no adapter file. Mid-run is a worse
        place to raise than the end of a run, so this must stay a no-op."""
        from soup_cli.utils.peft_wiring import build_compile_prefix_callback

        (tmp_path / "checkpoint-100").mkdir()
        _fire_on_save(build_compile_prefix_callback(), tmp_path, 100)

    def test_a_rewrite_that_fails_warns_and_keeps_training(
        self, tmp_path, monkeypatch, caplog
    ):
        """A broken rewrite must not kill a multi-hour run, but it must not pass
        in silence either, because a checkpoint left with the prefix is a dead
        adapter and that is the whole bug."""
        from soup_cli.utils import peft_wiring

        _write_adapter(tmp_path / "checkpoint-100", prefix="_orig_mod.")

        def boom(_output_dir):
            raise OSError("disk full")

        monkeypatch.setattr(peft_wiring, "strip_compile_prefix", boom)
        with caplog.at_level("WARNING"):
            _fire_on_save(peft_wiring.build_compile_prefix_callback(), tmp_path, 100)

        assert "checkpoint-100" in caplog.text, caplog.text

    def test_the_callback_answers_every_trainer_event(self):
        """#308. HF's ``CallbackHandler.call_event`` dispatches through
        ``getattr(cb, event)`` with no ``hasattr`` guard, so a duck-typed
        callback dies on the first ``on_epoch_begin`` rather than at wiring
        time. Inheriting ``TrainerCallback`` is what supplies the no-op stubs."""
        from transformers import TrainerCallback

        from soup_cli.utils.peft_wiring import build_compile_prefix_callback

        callback = build_compile_prefix_callback()
        assert isinstance(callback, TrainerCallback)
        callback.on_epoch_begin(
            types.SimpleNamespace(), types.SimpleNamespace(), types.SimpleNamespace()
        )

    def test_it_is_wired_only_when_compile_is_on(self):
        """CONTROL. Every ordinary run would otherwise carry a callback that
        reopens each checkpoint for nothing."""
        from soup_cli.utils.peft_wiring import attach_compile_prefix_callback

        class FakeTrainer:
            def __init__(self):
                self.callbacks = []

            def add_callback(self, callback):
                self.callbacks.append(callback)

        off = FakeTrainer()
        assert attach_compile_prefix_callback(
            off, types.SimpleNamespace(use_fsdp2_compile=False), "out"
        ) is False
        assert off.callbacks == []

        on = FakeTrainer()
        assert attach_compile_prefix_callback(
            on, types.SimpleNamespace(use_fsdp2_compile=True), "out"
        ) is True
        assert len(on.callbacks) == 1


# ==========================================================================
# #77 (low) — the --no-reexec hint dropped the user's own flags
# ==========================================================================
class TestNoReexecHintKeepsTheUsersFlags:
    """`--gpus 4 --fsdp full_shard --no-reexec` printed a command WITHOUT
    `--fsdp`, so following the hint literally trained without FSDP.

    The hint was hand-built as ``["soup", "train", "-c", config]`` a few lines
    above the auto-reexec path, which assembles the real flag list. Two copies of
    "what the user typed", and only one of them was maintained. The hint is now
    derived from the same ``script_args``, so they cannot drift.
    """

    def _advice(self, tmp_path, monkeypatch, extra_args):
        import re

        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.utils import topology as topo_mod

        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(
            "base: test/model\n"
            "task: sft\n"
            "data: {train: data.jsonl, format: alpaca}\n"
            "training: {epochs: 1, lr: 1e-4, batch_size: 1}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            topo_mod, "detect_topology",
            lambda: {"gpu_count": 4, "interconnect": "PCIe"},
        )
        monkeypatch.setattr(topo_mod, "resolve_num_gpus", lambda spec: 4)
        result = CliRunner().invoke(
            app,
            ["train", "--config", "soup.yaml", "--gpus", "4", "--no-reexec", "--yes",
             *extra_args],
        )
        # Rich wraps and may inject ANSI; strip both before matching, the way
        # every other CLI-output assertion in this repo has had to since v0.71.26.
        return re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("\n", " ")

    def test_the_hint_repeats_the_flags_the_user_passed(self, tmp_path, monkeypatch):
        out = self._advice(
            tmp_path, monkeypatch, ["--fsdp", "full_shard", "--trust-remote-code"]
        )
        if "Multi-GPU launch required" not in out:
            pytest.skip("did not reach the advisory (config/data validation ran first)")
        assert "--fsdp" in out, out
        assert "full_shard" in out, out
        assert "--trust-remote-code" in out, out

    def test_the_hint_does_not_tell_you_to_pass_no_reexec(self, tmp_path, monkeypatch):
        """Under `accelerate launch` the run is already distributed and never
        re-execs, so repeating the flag would be noise the user has to reason
        about. CONTROL for the derivation: it proves the hint is filtered rather
        than a raw echo of the reexec argv."""
        out = self._advice(tmp_path, monkeypatch, ["--fsdp", "full_shard"])
        if "Multi-GPU launch required" not in out:
            pytest.skip("did not reach the advisory (config/data validation ran first)")
        assert "--no-reexec" not in out, out


# ==========================================================================
# Liger was silently inert for every local model path
# ==========================================================================
class TestLigerDetectsArchitectureFromTheConfig:
    """`apply_liger_kernel` matched on a SUBSTRING OF THE MODEL PATH.

    It first tried `AutoLigerKernelForCausalLM._apply_liger_kernel(model_name)` —
    which **does not exist** in liger-kernel 0.8.1 (`AttributeError: type object
    ... has no attribute '_apply_liger_kernel'`, verified on the box) — and then
    fell back to `_apply_liger_manual(model_name.lower())`, which looks for
    "llama" / "mistral" / "qwen2" and friends **in the name**.

    So for any model loaded from a local directory the name carries no
    architecture, nothing matched, and the run printed "no matching architecture
    found" and trained without Liger. Silently, on a flag the user explicitly set.

    The architecture is in the config, where it cannot be renamed away: liger 0.8.1
    exposes a module-level `_apply_liger_kernel(model_type: str)`, and
    `AutoConfig.model_type` is exactly that string.
    """

    def _fake_liger(self, monkeypatch, calls):
        """Inject a stand-in liger so this runs where liger is not installed —
        which is the machine the maintainer develops on, and the reason the defect
        survived."""
        import sys
        import types

        mod = types.ModuleType("liger_kernel.transformers")

        def _apply(model_type, **kwargs):
            calls.append(model_type)

        mod._apply_liger_kernel = _apply
        mod._apply_liger_kernel_to_instance = lambda model, **kw: None
        parent = types.ModuleType("liger_kernel")
        parent.transformers = mod
        monkeypatch.setitem(sys.modules, "liger_kernel", parent)
        monkeypatch.setitem(sys.modules, "liger_kernel.transformers", mod)
        return calls

    def test_a_local_path_with_no_architecture_in_its_name_is_still_detected(
        self, tmp_path, monkeypatch
    ):
        from test_v07202 import _tiny_llama_dir

        from soup_cli.utils import liger as liger_mod

        weights, _, _ = _tiny_llama_dir(tmp_path)
        assert "llama" not in weights.lower(), (
            "the fixture path must NOT contain the architecture name, or this test "
            "passes for the substring matcher too and proves nothing"
        )
        calls: list = []
        self._fake_liger(monkeypatch, calls)
        monkeypatch.setattr(liger_mod, "check_liger_available", lambda: True)

        assert liger_mod.apply_liger_kernel(weights) is True
        assert calls == ["llama"], calls

    def test_an_unknown_architecture_still_reports_false(self, tmp_path, monkeypatch):
        """CONTROL. Detection by config must not turn into "always True" — an
        architecture liger does not support has to keep reporting False so the
        caller keeps printing its warning and does not set the TRL flag."""
        import json

        from soup_cli.utils import liger as liger_mod

        directory = tmp_path / "weird"
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps({"model_type": "not_a_real_arch"}), encoding="utf-8"
        )
        calls: list = []
        self._fake_liger(monkeypatch, calls)

        def _raise(model_type, **kwargs):
            raise NotImplementedError(model_type)

        import sys

        sys.modules["liger_kernel.transformers"]._apply_liger_kernel = _raise
        monkeypatch.setattr(liger_mod, "check_liger_available", lambda: True)

        assert liger_mod.apply_liger_kernel(str(directory)) is False


# ==========================================================================
# #351 review: the normalisation has to reach the file before anything
# that PUBLISHES it does
# ==========================================================================
class TestTheNormalisationRunsBeforeAnythingThatPublishes:
    """``HFPushCallback.on_save`` uploads ``checkpoint-{step}`` on the same event
    this normalisation runs on, so the order the two are dispatched in decides
    what ``--push-as`` puts on the Hub.

    ``CallbackHandler.add_callback`` appends and ``call_event`` iterates in
    insertion order, and ``--push-as`` adds its callback to the trainer after
    ``setup()`` has returned (``commands/train.py``). Attaching the
    normalisation in ``setup()`` is therefore what keeps it in front: attached in
    ``train()`` instead, it landed second, the Hub received the prefixed adapter
    and only local disk got the repaired one. That is the same dead-adapter
    defect on the path where it is least recoverable, since what is published is
    the dead file.

    Asserted through the real ``callback_handler`` rather than by comparing list
    indices, because the property that matters is what a later callback SEES.
    """

    def test_a_later_added_on_save_sees_the_repaired_keys(self, tmp_path, monkeypatch):
        from pathlib import Path

        from safetensors.torch import load_file
        from transformers import TrainerCallback

        wrapper = _sft_wrapper(tmp_path, monkeypatch, use_fsdp2_compile=True)
        trainer = wrapper.trainer
        seen = {}

        class Spy(TrainerCallback):
            """Stands in for ``HFPushCallback``: reads the checkpoint it is told
            was just written, which is what an upload does."""

            def on_save(self, args, state, control, **kwargs):
                adapter = (
                    Path(args.output_dir)
                    / f"checkpoint-{state.global_step}"
                    / "adapter_model.safetensors"
                )
                seen["keys"] = set(load_file(str(adapter)))

        # Exactly what `--push-as` does, and it can only ever happen after
        # `setup()` has run.
        trainer.add_callback(Spy())

        _write_adapter(
            Path(trainer.args.output_dir) / "checkpoint-100", prefix="_orig_mod."
        )
        trainer.state.global_step = 100
        trainer.control = trainer.callback_handler.on_save(
            trainer.args, trainer.state, trainer.control
        )

        assert seen, "the spy's on_save never fired; the wiring changed shape"
        assert not any(k.startswith("_orig_mod.") for k in seen["keys"]), (
            f"a callback added after setup() saw the PREFIXED adapter, so "
            f"--push-as would publish a dead one: {sorted(seen['keys'])}"
        )

    def test_the_callback_is_attached_by_setup_not_by_train(self, tmp_path, monkeypatch):
        """The position itself, so a move back into ``train()`` fails here rather
        than only in the subtler assertion above."""
        wrapper = _sft_wrapper(tmp_path, monkeypatch, use_fsdp2_compile=True)
        names = [
            type(cb).__name__ for cb in wrapper.trainer.callback_handler.callbacks
        ]
        assert "CompilePrefixCallback" in names, names
