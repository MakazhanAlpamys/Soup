"""Reference and checkpoint checks for streamed IPO (#2766)."""

from pathlib import Path

import pytest

from tests.test_v07204 import (
    _pref_rows,
    _randomise_lora_b,
    _stream_cfg,
    _sync_adapters,
    _tiny_llama_dir,
    _write_tiny_tokenizer,
)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    weights, _, _ = _tiny_llama_dir(tmp_path, tie=False)
    _write_tiny_tokenizer(weights)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    return Path(weights)


def _build(checkpoint, name, *, device="cpu", quant="none", dispatched=False, stream=True):
    from soup_cli.trainer.ipo import IPOTrainerWrapper
    from soup_cli.trainer.preference import PreferenceTrainerWrapper

    cfg = _stream_cfg(
        str(checkpoint), checkpoint.parent / name,
        task="preference" if dispatched else "ipo",
        **({"preference_loss": "ipo"} if dispatched else {}),
        stream_layers=stream, **({"stream_source": "ram"} if stream else {}),
        quantization=quant, batch_size=1,
        seed=7, warmup_ratio=0.0, save_steps=1,
        lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    )
    wrapper = (PreferenceTrainerWrapper if dispatched else IPOTrainerWrapper)(
        cfg, device=device, fsdp_config={"use_cpu": True} if device == "cpu" else None,
    )
    rows = _pref_rows(4)
    rows[1] = {"prompt": "hello", "chosen": " world", "rejected": " bad answer"}
    try:
        wrapper.setup({"train": rows})
    except BaseException:
        inner = wrapper._inner if dispatched else wrapper
        if inner is not None:
            inner._close_stream_runtime()
        raise
    inner = wrapper._inner if dispatched else wrapper
    assert type(inner) is IPOTrainerWrapper
    assert inner.config.task == "ipo" and inner.config.training.stream_layers is stream
    inner.trainer.args.logging_nan_inf_filter = False
    return wrapper, inner


def _trained_adapters(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()
            if ".default." in name and "lora_" in name}


def _assert_equal(left, right):
    import torch

    assert left and left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0, msg=key)


def test_streamed_budget_counts_the_loss_batch_shape(checkpoint):
    _, inner = _build(checkpoint, "budget")
    try:
        rows, length, vocab = inner._stream_step_shape(
            inner.config, inner.config.training, inner.model.config,
        )
        assert (rows, length, vocab) == (2, 64, 64)
        assert inner._stream_large_budget_bytes(128, 2) == 256
    finally:
        inner._close_stream_runtime()


def test_wrapper_closes_stream_runtime_when_training_raises(checkpoint, monkeypatch):
    wrapper, inner = _build(checkpoint, "failure")
    closed = []
    close = inner._stream_runtime.close

    def record_close():
        close()
        closed.append(True)

    def fail_training(**kwargs):
        raise RuntimeError("training failure")

    monkeypatch.setattr(inner._stream_runtime, "close", record_close)
    monkeypatch.setattr(inner.trainer, "train", fail_training)
    with pytest.raises(RuntimeError, match="training failure"):
        wrapper.train()
    assert closed == [True]


def test_untied_streamed_loss_and_gradients_match_resident(checkpoint, monkeypatch):
    from soup_cli.utils import layer_stream_runtime

    sources = []

    class CountedSource(layer_stream_runtime.RamSource):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sources.append(self)

    monkeypatch.setattr(layer_stream_runtime, "RamSource", CountedSource)
    _, streamed = _build(checkpoint, "stream")
    try:
        _, resident = _build(checkpoint, "resident", stream=False)
        assert len(sources) == 1 and streamed.trainer.ref_model is None
        refs = [value for key, value in streamed.model.named_parameters() if ".ref." in key]
        assert refs and all(not value.is_meta and not value.requires_grad for value in refs)
        _randomise_lora_b(streamed.model)
        assert _sync_adapters(resident.model, streamed.model) > 0
        batch = next(iter(streamed.trainer.get_train_dataloader()))
        assert next(iter(resident.trainer.get_train_dataloader())).keys() == batch.keys()

        def run(wrapper):
            model = wrapper.model
            model.train()
            value = wrapper.trainer.compute_loss(model, batch)
            value.backward()
            grads = {name.replace(".inner.", "."): param.grad for name, param in
                     model.named_parameters() if param.grad is not None and "lora_" in name}
            return value.detach(), grads

        actual, actual_grads = run(streamed)
        expected, expected_grads = run(resident)
        _assert_equal({"loss": actual}, {"loss": expected})
        _assert_equal(actual_grads, expected_grads)
        assert any(grad.count_nonzero().item() for grad in actual_grads.values())
    finally:
        streamed._close_stream_runtime()


def test_streamed_resume_restores_adapter_and_reference_state(checkpoint):
    from safetensors.torch import load_file
    from transformers import TrainerCallback

    full, original = _build(checkpoint, "full")
    original.trainer.args.max_steps = 2
    full.train()
    saved = checkpoint.parent / "full" / "checkpoint-1"
    default = load_file(saved / "adapter_model.safetensors")
    ref = load_file(saved / "ref" / "adapter_model.safetensors")
    resumed, inner = _build(checkpoint, "resumed", dispatched=True)
    inner.trainer.args.max_steps = 2
    seen = []

    class ObserveResume(TrainerCallback):
        def on_train_begin(self, args, state, control, model, **kwargs):
            from peft import get_peft_model_state_dict

            assert state.global_step == 1
            _assert_equal(get_peft_model_state_dict(model), default)
            _assert_equal(get_peft_model_state_dict(model, adapter_name="ref"), ref)
            seen.append(True)

    inner.trainer.add_callback(ObserveResume())
    result = resumed.train(resume_from_checkpoint=str(saved))
    assert seen and result["total_steps"] == 2
    _assert_equal(_trained_adapters(inner.model), _trained_adapters(original.model))


@pytest.mark.parametrize("where", ["checkpoint", "final"])
def test_wrapper_rejects_empty_streamed_adapter(checkpoint, monkeypatch, where):
    from safetensors.torch import save_file

    wrapper, inner = _build(checkpoint, "empty")
    inner.trainer.args.max_steps = 1
    save = inner.trainer.save_model

    def empty_on_save(output_dir=None, *args, **kwargs):
        save(output_dir, *args, **kwargs)
        output = Path(output_dir or inner.trainer.args.output_dir)
        if output.name.startswith("checkpoint-") == (where == "checkpoint"):
            save_file({}, output / "adapter_model.safetensors")

    monkeypatch.setattr(inner.trainer, "save_model", empty_on_save)
    with pytest.raises(RuntimeError, match="holds 0 LoRA tensors"):
        wrapper.train()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dispatched,quant,initial_scale",
    [
        pytest.param(False, "none", 1024, id="ipo-none-1024"),
        pytest.param(False, "4bit", 1024, id="ipo-nf4-1024"),
        pytest.param(True, "none", 1024, id="preference-none-1024"),
        pytest.param(True, "4bit", 1024, id="preference-nf4-1024"),
        pytest.param(False, "none", None, id="ipo-none-default"),
    ],
)
@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "resident"])
def test_cuda_first_applied_update(checkpoint, dispatched, quant, initial_scale, stream):
    import json
    import math
    import os
    from hashlib import sha256

    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, TrainerCallback

    torch.cuda.reset_peak_memory_stats()
    wrapper, inner = _build(
        checkpoint, "cuda", device="cuda", quant=quant, dispatched=dispatched, stream=stream,
    )
    trainer = inner.trainer
    try:
        assert trainer.args.fp16 and not trainer.args.bf16
        assert trainer.ref_model is None
        assert trainer.beta == inner.config.training.ipo_tau == 0.1
        scaler = trainer.accelerator.scaler
        default_scale = scaler.get_scale()
        if initial_scale is not None:
            # Qualification harness only. The default control never changes the scaler.
            state = scaler.state_dict()
            state["scale"] = float(initial_scale)
            scaler.load_state_dict(state)
        effective_scale = scaler.get_scale()
        assert effective_scale == (initial_scale or default_scale)
        cap = 1 if initial_scale is not None else 12
        trainer.args.max_steps = cap
        before = _trained_adapters(inner.model)
        attempts = []

        class ObserveUpdates(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                self.scale_before = scaler.get_scale()

            def on_step_end(self, args, state, control, **kwargs):
                after = _trained_adapters(inner.model)
                attempts.append({
                    "step": state.global_step,
                    "scale_before": self.scale_before,
                    "scale_after": scaler.get_scale(),
                    "optimizer_skipped": trainer.accelerator.optimizer_step_was_skipped,
                    "adapter_max_delta": max(
                        (after[key] - before[key]).abs().max().item() for key in before
                    ),
                })
                if not attempts[-1]["optimizer_skipped"]:
                    control.should_training_stop = True

        trainer.add_callback(ObserveUpdates())
        result = wrapper.train()
        logs = {item["step"]: item for item in trainer.state.log_history if "loss" in item}
        for attempt in attempts:
            for key in ("loss", "grad_norm"):
                value = logs[attempt["step"]][key]
                attempt[key] = value if math.isfinite(value) else str(value)
        applied = next((attempt for attempt in attempts if not attempt["optimizer_skipped"]), None)
        task = "preference" if dispatched else "ipo"
        mode = "streamed" if stream else "resident"
        values = {
            "task": task, "mode": mode, "quantization": quant,
            "instrument": "initial scale 1024, set in the harness only"
            if initial_scale is not None else "default initial scale",
            "default_initial_scale": default_scale, "effective_initial_scale": effective_scale,
            "attempt_cap": cap, "attempts": attempts,
            "skipped_steps": sum(attempt["optimizer_skipped"] for attempt in attempts),
            "first_applied_step": applied["step"] if applied else None,
            "fixture_sha256": sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest(),
            "layer_reads": inner._stream_runtime.pool.loads if stream else None,
        }
        evidence = os.environ.get("SOUP_TEST_2766_EVIDENCE_DIR")
        row_path = (
            Path(evidence) / f"ipo-{mode}-{task}-{quant}-{initial_scale or 'default'}.json"
            if evidence else None
        )
        if row_path:
            row_path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")

        assert len(attempts) == result["total_steps"] <= cap
        assert applied is not None, f"No applied update inside {cap} attempts: {attempts}"
        assert attempts[-1] == applied
        assert all(step["adapter_max_delta"] == 0 for step in attempts[:-1])
        assert applied["adapter_max_delta"] > 0
        assert math.isfinite(applied["loss"]) and math.isfinite(applied["grad_norm"])
        assert not stream or values["layer_reads"] > 0

        output = Path(result["output_dir"])
        saved = output / f"checkpoint-{applied['step']}"
        base = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
        restored = PeftModel.from_pretrained(base, output, local_files_only=True)
        restored.load_adapter(output / "ref", adapter_name="ref", local_files_only=True)
        for directory, adapter in ((Path("."), "default"), (Path("ref"), "ref")):
            final = load_file(output / directory / "adapter_model.safetensors")
            assert len(final) == 8
            _assert_equal(load_file(saved / directory / "adapter_model.safetensors"), final)
            reloaded = get_peft_model_state_dict(restored, adapter_name=adapter)
            # PEFT may promote saved half-precision adapters to FP32 on load.
            # Promote both sides so a lossy reload still fails exact comparison.
            _assert_equal(
                {key: value.float() for key, value in reloaded.items()},
                {key: value.float() for key, value in final.items()},
            )
            values[f"{adapter}_adapter_keys"] = sorted(final)
        values["adapter_reload"] = True
        values["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        if row_path:
            row_path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    finally:
        inner._close_stream_runtime()
