"""Reference, checkpoint and CUDA smoke checks for streamed BCO (#2766)."""

import json
import math
import os
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
    from soup_cli.trainer.bco import BCOTrainerWrapper
    from soup_cli.trainer.preference import PreferenceTrainerWrapper

    cfg = _stream_cfg(
        str(checkpoint), checkpoint.parent / name,
        task="preference" if dispatched else "bco",
        **({"preference_loss": "bco"} if dispatched else {}),
        stream_layers=stream, **({"stream_source": "ram"} if stream else {}),
        quantization=quant, batch_size=2,
        seed=7, warmup_ratio=0.0, save_steps=1,
        lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    )
    wrapper = (PreferenceTrainerWrapper if dispatched else BCOTrainerWrapper)(
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
    assert type(inner) is BCOTrainerWrapper
    assert inner.config.task == "bco" and inner.config.training.stream_layers is stream
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
        assert set(streamed.model.peft_config) == {"default"}
        _randomise_lora_b(streamed.model)
        assert _sync_adapters(resident.model, streamed.model) > 0
        batch = next(iter(streamed.trainer.get_train_dataloader()))
        assert next(iter(resident.trainer.get_train_dataloader())).keys() == batch.keys()
        assert set(batch["label"]) == {True, False}
        seen = []
        lora = next(module for module in streamed.model.modules() if hasattr(module, "lora_A"))
        hook = streamed.model.register_forward_pre_hook(
            lambda *_: seen.append(lora.disable_adapters)
        )

        def run(wrapper):
            model = wrapper.model
            model.train()
            value = wrapper.trainer.compute_loss(model, batch)
            value.backward()
            grads = {name.replace(".inner.", "."): param.grad for name, param in
                     model.named_parameters() if param.grad is not None and "lora_" in name}
            return value.detach(), grads

        try:
            actual, actual_grads = run(streamed)
        finally:
            hook.remove()
        expected, expected_grads = run(resident)
        _assert_equal({"loss": actual}, {"loss": expected})
        _assert_equal(actual_grads, expected_grads)
        assert any(grad.count_nonzero().item() for grad in actual_grads.values())
        assert seen == [False, True]
    finally:
        streamed._close_stream_runtime()


def test_streamed_resume_restores_adapter_and_running_state(checkpoint):
    from safetensors.torch import load_file
    from transformers import TrainerCallback

    full, original = _build(checkpoint, "full")
    _randomise_lora_b(original.model)
    original.trainer.args.max_steps = 2
    full.train()
    saved = checkpoint.parent / "full" / "checkpoint-1"
    default = load_file(saved / "adapter_model.safetensors")
    running = json.loads((saved / "running.json").read_text())
    resumed, inner = _build(checkpoint, "resumed", dispatched=True)
    inner.trainer.args.max_steps = 2
    seen = []

    class ObserveResume(TrainerCallback):
        def on_train_begin(self, args, state, control, model, **kwargs):
            from peft import get_peft_model_state_dict

            assert state.global_step == 1
            _assert_equal(get_peft_model_state_dict(model), default)
            assert running["count"] > 0
            assert all(running[key] == getattr(inner.trainer.running, key) for key in running)
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
@pytest.mark.parametrize("dispatched", [False, True], ids=["legacy", "preference"])
@pytest.mark.parametrize("quant", ["none", "4bit"])
def test_one_cuda_update_and_adapter_round_trip(checkpoint, dispatched, quant):
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    torch.cuda.reset_peak_memory_stats()
    wrapper, inner = _build(
        checkpoint, "cuda", device="cuda", quant=quant, dispatched=dispatched,
    )
    try:
        assert inner.trainer.ref_model is None
        batch = next(iter(inner.trainer.get_train_dataloader()))
        assert set(batch["label"]) == {True, False}
        before = _trained_adapters(inner.model)
        inner.trainer.args.max_steps = 1
        result = wrapper.train()
        row = next(item for item in inner.trainer.state.log_history if "loss" in item)
        assert result["total_steps"] == 1
        assert math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"])
        assert not inner.trainer.accelerator.optimizer_step_was_skipped
        assert inner._stream_runtime.pool.loads > 0
        after = _trained_adapters(inner.model)
        adapter_delta = max((after[key] - before[key]).abs().max().item() for key in before)
        assert adapter_delta > 0
        peak = torch.cuda.max_memory_allocated()
        output = Path(result["output_dir"])
        checkpoint_tensors = load_file(output / "checkpoint-1" / "adapter_model.safetensors")
        final = load_file(output / "adapter_model.safetensors")
        _assert_equal(checkpoint_tensors, final)
        base = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
        restored = PeftModel.from_pretrained(base, output, local_files_only=True)
        _assert_equal(get_peft_model_state_dict(restored), final)
        evidence = os.environ.get("SOUP_TEST_2766_EVIDENCE_DIR")
        if evidence:
            from hashlib import sha256

            fixture_hash = sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
            values = {
                "loss": "bco", "task": "preference" if dispatched else "bco",
                "quantization": quant, "step": result["total_steps"],
                "raw_loss": row["loss"], "grad_norm": row["grad_norm"],
                "adapter_max_delta": adapter_delta, "peak_allocated_bytes": peak,
                "layer_reads": inner._stream_runtime.pool.loads,
                "checkpoint_keys": sorted(checkpoint_tensors), "final_keys": sorted(final),
                "checkpoint_sha256": fixture_hash,
                "adapter_reload": True,
            }
            name = f"bco-{values['task']}-{quant}.json"
            (Path(evidence) / name).write_text(json.dumps(values, indent=2) + "\n")
    finally:
        inner._close_stream_runtime()
