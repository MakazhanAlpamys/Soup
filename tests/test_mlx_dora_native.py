"""Native adapter lifecycles (DoRA and SFT watchdog); no model downloads.

Kept together because mlx-smoke already selects this file on Apple Silicon.
"""

import json
import math

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.llama import Model, ModelArgs  # noqa: E402
from mlx_lm.tuner.dora import DoRALinear  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from soup_cli.config.schema import SoupConfig  # noqa: E402
from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper  # noqa: E402


def _checkpoint(path, quantized=False, *, layers=1):
    path.mkdir()
    mx.random.seed(7)
    config = dict(
        model_type="llama", hidden_size=64, num_hidden_layers=layers, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=4, rms_norm_eps=1e-5,
        vocab_size=32, tie_word_embeddings=False,
    )
    model = Model(ModelArgs(**config))
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
        config["quantization"] = {"group_size": 32, "bits": 4}
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config))
    vocab = {word: i for i, word in enumerate(
        ["<unk>", "<eos>", "<pad>", "user", "assistant", "alpha", "beta", "gamma"]
    )}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "<eos> {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    tokenizer.save_pretrained(path)
    return path


def _config(base, output, *, dora=True, quantized=False, targets="auto"):
    return SoupConfig(
        base=str(base), backend="mlx", output=str(output),
        data={"train": "unused.jsonl", "train_on_responses_only": False, "max_length": 64},
        training={
            "quantization": "4bit" if quantized else "none", "epochs": 1,
            "batch_size": 1, "gradient_accumulation_steps": 1, "lr": 0.005,
            "logging_steps": 1, "save_steps": 2, "scheduler": "constant",
            "warmup_ratio": 0, "weight_decay": 0,
            "lora": {"r": 4, "alpha": 8, "dropout": 0, "use_dora": dora,
                     "target_modules": targets},
        },
    )


def _rows():
    return {"train": [{"text": "alpha beta gamma alpha beta gamma"}] * 4}


@pytest.mark.parametrize("quantized, checkpoint, accumulation", [
    (False, False, 1), (True, False, 1), (False, True, 2), (True, True, 2),
])
def test_native_training_reload_warm_start_and_fusion(
    tmp_path, quantized, checkpoint, accumulation, monkeypatch
):
    base = _checkpoint(tmp_path / "base", quantized)
    cfg = _config(base, tmp_path / "adapters", quantized=quantized)
    cfg.training.gradient_checkpointing = checkpoint
    cfg.training.gradient_accumulation_steps = accumulation
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.setup(_rows())
    original = dict(tree_flatten(wrapper.model.parameters()))
    initial = {}
    apply = wrapper._apply_lora

    def capture_initial(model):
        apply(model)
        initial.update(tree_flatten(model.trainable_parameters()))
        mx.eval(initial)

    monkeypatch.setattr(wrapper, "_apply_lora", capture_initial)
    result = wrapper.train()
    assert result["total_steps"] == 4
    assert math.isfinite(result["final_loss"])
    assert isinstance(wrapper.model.layers[0].self_attn.q_proj, DoRALinear)
    trainable = dict(tree_flatten(wrapper.model.trainable_parameters()))
    assert len(trainable) == 6
    assert {name.rsplit(".", 1)[-1] for name in trainable} == {"lora_a", "lora_b", "m"}
    assert all(not mx.array_equal(value, initial[name]).item()
               for name, value in trainable.items())
    from mlx_lm.tuner.trainer import default_loss

    (_, _), gradients = nn.value_and_grad(wrapper.model, default_loss)(
        wrapper.model, mx.array([[5, 6, 7, 5]]), mx.array([[0, 4]])
    )
    assert all(mx.isfinite(value).all().item() for _, value in tree_flatten(gradients))
    # Every frozen base tensor, including quantization scales/biases, is unchanged.
    trained = dict(tree_flatten(wrapper.model.parameters()))
    for name, value in original.items():
        parent, leaf = name.rsplit(".", 1)
        adapted_name = f"{parent}.linear.{leaf}"
        actual = trained[adapted_name] if adapted_name in trained else trained[name]
        assert mx.array_equal(actual, value).item(), name
    for projection in ("q_proj", "v_proj"):
        layer = getattr(wrapper.model.layers[0].self_attn, projection)
        assert not mx.array_equal(layer.lora_b, mx.zeros_like(layer.lora_b)).item()
        assert not mx.array_equal(
            layer.m, mx.linalg.norm(layer._dequantized_weight().astype(mx.float32), axis=1)
        ).item()

    tokens = mx.array([[5, 6, 7, 5]])
    wrapper.model.eval()
    expected = wrapper.model(tokens)
    metadata = json.loads((tmp_path / "adapters" / "adapter_config.json").read_text())
    assert metadata["fine_tune_type"] == "dora"
    assert metadata["lora_parameters"]["scale"] == 2
    restored, _ = load(str(base), adapter_path=str(tmp_path / "adapters"))
    restored.eval()
    assert isinstance(restored.layers[0].self_attn.q_proj, DoRALinear)
    assert mx.allclose(restored(tokens), expected, atol=1e-5, rtol=1e-5).item()

    resumed = MLXSFTTrainerWrapper(cfg)
    resumed.setup(_rows())
    resumed._apply_lora(resumed.model)
    resumed._load_checkpoint_weights(str(tmp_path / "adapters" / "adapters.safetensors"))
    resumed.model.eval()
    assert mx.allclose(resumed.model(tokens), expected, atol=1e-5, rtol=1e-5).item()

    # This is the operation used by mlx_lm.fuse --dequantize. Avoid requantization
    # error when asserting that the adapter's mathematical effect is preserved.
    restored.update_modules(tree_unflatten([
        (name, module.fuse(dequantize=True))
        for name, module in restored.named_modules() if isinstance(module, DoRALinear)
    ]))
    assert mx.allclose(restored(tokens), expected, atol=2e-5, rtol=2e-5).item()


@pytest.mark.parametrize("quantized", [False, True])
def test_public_cli_trains_a_reloadable_dora_adapter(tmp_path, quantized, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    base = _checkpoint(tmp_path / "base", quantized)
    cfg = _config(base, tmp_path / "out", quantized=quantized)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"messages": [
        {"role": "user", "content": "alpha beta"},
        {"role": "assistant", "content": "gamma alpha beta"},
    ]}) for _ in range(4)) + "\n")
    cfg.data.train = str(data)
    cfg.data.format = "chatml"
    cfg.data.val_split = 0
    config = tmp_path / "soup.json"
    # The loader uses YAML, whose JSON subset keeps the serialized schema exact.
    config.write_text(cfg.model_dump_json())
    outcome = CliRunner().invoke(app, ["train", "--config", str(config), "--yes"])
    assert outcome.exit_code == 0, (outcome.output, repr(outcome.exception))
    restored, _ = load(str(base), adapter_path=str(tmp_path / "out"))
    assert isinstance(restored.layers[0].self_attn.q_proj, DoRALinear)
    assert mx.isfinite(restored(mx.array([[5, 6, 7]]))).all().item()
    resumed = CliRunner().invoke(
        app, ["train", "--config", str(config), "--yes", "--resume", "auto"]
    )
    assert resumed.exit_code == 0, (resumed.output, repr(resumed.exception))
    assert "loading checkpoint weights" in resumed.output
    reloaded, _ = load(str(base), adapter_path=str(tmp_path / "out"))
    assert isinstance(reloaded.layers[0].self_attn.q_proj, DoRALinear)
    assert mx.isfinite(reloaded(mx.array([[5, 6, 7]]))).all().item()


@pytest.mark.parametrize("change", ["lora", "rank", "targets", "missing", "extra"])
def test_dora_refuses_incompatible_warm_start_before_loading(tmp_path, change):
    base = _checkpoint(tmp_path / "base")
    cfg = _config(base, tmp_path / "out")
    target = MLXSFTTrainerWrapper(cfg)
    target.setup(_rows())
    target._apply_lora(target.model)
    before = dict(tree_flatten(target.model.trainable_parameters()))
    source_cfg = _config(base, tmp_path / "other", dora=change != "lora")
    if change == "rank":
        source_cfg.training.lora.r = 2
    if change == "targets":
        source_cfg.training.lora.target_modules = ["self_attn.k_proj"]
    source = MLXSFTTrainerWrapper(source_cfg)
    source.setup(_rows())
    source._apply_lora(source.model)
    weights = dict(tree_flatten(source.model.trainable_parameters()))
    if change == "missing":
        weights.pop(next(iter(weights)))
    if change == "extra":
        weights["unexpected.m"] = mx.ones((1,))
    checkpoint = tmp_path / "incompatible.safetensors"
    mx.save_safetensors(str(checkpoint), weights)
    with pytest.raises(ValueError, match="[Dd][Oo][Rr][Aa].*incompatible"):
        target._load_checkpoint_weights(str(checkpoint))
    after = dict(tree_flatten(target.model.trainable_parameters()))
    assert all(mx.array_equal(value, after[name]).item() for name, value in before.items())


@pytest.mark.parametrize("invalid", ["missing", "switch", "embedding", "8bit"])
def test_unsupported_targets_fail_before_any_conversion(tmp_path, invalid):
    from mlx_lm.models.switch_layers import SwitchLinear

    base = _checkpoint(tmp_path / "base")
    targets = ["self_attn.q_proj", "self_attn.v_proj"]
    if invalid == "missing":
        targets.append("self_attn.typo")
    wrapper = MLXSFTTrainerWrapper(_config(base, tmp_path / "out", targets=targets))
    wrapper.setup(_rows())
    attention = wrapper.model.layers[0].self_attn
    if invalid == "switch":
        attention.v_proj = SwitchLinear(64, 64, 2)
    elif invalid == "embedding":
        attention.v_proj = nn.Embedding(32, 64)
    elif invalid == "8bit":
        attention.v_proj = nn.QuantizedLinear.from_linear(attention.v_proj, group_size=32, bits=8)
    before = dict(tree_flatten(wrapper.model.trainable_parameters()))
    with pytest.raises(ValueError, match="MLX DoRA requires dense linear targets"):
        wrapper._apply_lora(wrapper.model)
    assert isinstance(attention.q_proj, nn.Linear)
    assert set(dict(tree_flatten(wrapper.model.trainable_parameters()))) == set(before)


def test_plain_lora_still_constructs_lora_layers(tmp_path):
    from mlx_lm.tuner.lora import LoRALinear

    base = _checkpoint(tmp_path / "base")
    wrapper = MLXSFTTrainerWrapper(_config(base, tmp_path / "out", dora=False))
    wrapper.setup(_rows())
    wrapper._apply_lora(wrapper.model)
    assert isinstance(wrapper.model.layers[0].self_attn.q_proj, LoRALinear)
    assert not any(name.endswith(".m")
                   for name, _ in tree_flatten(wrapper.model.trainable_parameters()))


def test_native_generation_preserves_tokenizer_and_model_eos(tmp_path):
    """The dependency floor otherwise drops tokenizer EOS when model EOS differs."""
    from soup_cli.utils.mlx_generation import generate_native, load_native_model

    base = _checkpoint(tmp_path / "base")
    config = json.loads((base / "config.json").read_text())
    config["eos_token_id"] = [2, 3]
    (base / "config.json").write_text(json.dumps(config))
    tokenizer_config = json.loads((base / "tokenizer_config.json").read_text())
    tokenizer_config["eos_token"] = "<unk>"
    (base / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    weights = mx.load(str(base / "model.safetensors"))
    # Equal logits deterministically sample token zero, the tokenizer's declared EOS.
    weights["lm_head.weight"] = mx.zeros_like(weights["lm_head.weight"])
    mx.save_safetensors(str(base / "model.safetensors"), weights)

    model, tokenizer = load_native_model(str(base))
    assert tokenizer.eos_token_id == 0
    assert tokenizer.eos_token_ids == {0, 2, 3}
    assert generate_native(
        model, tokenizer, [{"role": "user", "content": "alpha"}],
        max_tokens=4, temperature=0,
    ) == ("", 1)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("targets", [["lm_head"], "auto"], ids=["head", "auto-two-layers"])
def test_native_generation_fresh_process_consumers_and_strict_weights(
    tmp_path, quantized, targets, monkeypatch,
):
    """A real trained LoRA survives inference in a fresh process."""
    import os
    import subprocess
    import sys

    monkeypatch.chdir(tmp_path)
    base = _checkpoint(tmp_path / "base", quantized, layers=2 if targets == "auto" else 1)
    adapter = tmp_path / "adapter"
    cfg = _config(base, adapter, dora=False, quantized=quantized, targets=targets)
    cfg.training.lr = 0.01
    cfg.training.epochs = 4
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.setup(_rows())
    assert isinstance(wrapper.model.lm_head, nn.QuantizedLinear) is quantized
    if quantized:
        assert wrapper.model.lm_head.bits == 4
        assert wrapper.model.lm_head.weight.dtype == mx.uint32
    frozen = dict(tree_flatten(wrapper.model.parameters()))
    mx.eval(frozen)
    outcome = wrapper.train()
    assert math.isfinite(outcome["final_loss"])
    trained = dict(tree_flatten(wrapper.model.parameters()))
    for name, value in frozen.items():
        parent, leaf = name.rsplit(".", 1)
        actual = trained.get(f"{parent}.linear.{leaf}", trained.get(name))
        assert mx.array_equal(value, actual).item(), name
    saved = mx.load(str(adapter / "adapters.safetensors"))
    if targets == "auto":
        assert set(saved) == {
            f"model.layers.{layer}.self_attn.{projection}.lora_{matrix}"
            for layer in (0, 1) for projection in ("q_proj", "v_proj") for matrix in ("a", "b")
        }
    assert any(not mx.array_equal(value, mx.zeros_like(value)).item()
               for name, value in saved.items() if name.endswith("lora_b"))
    wrapper.model.eval()
    expected_logits = wrapper.model(mx.array([[5, 6, 7, 5]]))
    mx.save_safetensors(str(tmp_path / "trained-logits.safetensors"), {"logits": expected_logits})
    metadata = json.loads((adapter / "adapter_config.json").read_text())
    assert metadata["model"] == str(base)
    assert metadata["lora_parameters"]["scale"] == 2.0
    if targets == "auto":
        assert metadata["num_layers"] == 2
        assert set(metadata["lora_parameters"]["keys"]) == {"self_attn.q_proj", "self_attn.v_proj"}
    script = r'''
import json, sys
from pathlib import Path
import mlx.core as mx
from typer.testing import CliRunner
from soup_cli.cli import app
from soup_cli.utils.mlx_generation import load_native_model, generate_native
model, tokenizer = load_native_model("adapter")
base, _ = load_native_model("base")
tokens = mx.array([[5, 6, 7, 5]])
assert mx.allclose(model(tokens), mx.load("trained-logits.safetensors")["logits"],
                   atol=1e-5, rtol=1e-5).item()
assert not mx.allclose(model(tokens), base(tokens), atol=1e-5, rtol=1e-5).item()
messages = [{"role": "user", "content": "alpha"}]
expected, count = generate_native(model, tokenizer, messages, max_tokens=4, temperature=0)
control, _ = generate_native(base, tokenizer, messages, max_tokens=4, temperature=0)
if sys.argv[1] == "head":
    assert expected != control, (expected, control)
Path("prompts.jsonl").write_text('{"prompt":"alpha"}\n{"prompt":"beta"}\n')
runner = CliRunner()
result = runner.invoke(app, ["infer", "--backend", "mlx", "--model", "adapter",
                            "--input", "prompts.jsonl", "--output", "results.jsonl",
                            "--temperature", "0", "--max-tokens", "4"])
assert result.exit_code == 0, (result.output, repr(result.exception))
rows = [json.loads(line) for line in Path("results.jsonl").read_text().splitlines()]
assert [row["prompt"] for row in rows] == ["alpha", "beta"]
assert rows[0]["response"] == expected
assert rows[0]["tokens_generated"] == count
Path("consumer-evidence.json").write_text(json.dumps({"response": expected, "base": control,
                                                    "tokens_generated": count}))
'''
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    env["PYTHONPATH"] = str(__import__("pathlib").Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script, "auto" if targets == "auto" else "head"],
        cwd=tmp_path, env=env,
        text=True, capture_output=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = json.loads((tmp_path / "consumer-evidence.json").read_text())
    if targets != "auto":
        assert evidence["response"] != evidence["base"]

    from soup_cli.utils.mlx_generation import load_native_model

    for mutation in ("missing", "extra", "shape", "nonfinite", "integer", "truncated"):
        weights = dict(saved)
        name = next(iter(weights))
        if mutation == "missing":
            del weights[name]
        elif mutation == "extra":
            weights["not.an.adapter"] = mx.zeros((1,))
        elif mutation == "shape":
            weights[name] = mx.zeros((1, 1))
        elif mutation == "nonfinite":
            weights[name] = mx.full(weights[name].shape, float("nan"))
        else:
            weights[name] = mx.zeros(weights[name].shape, dtype=mx.int32)
        if mutation == "truncated":
            (adapter / "adapters.safetensors").write_bytes(b"truncated")
        else:
            mx.save_safetensors(str(adapter / "adapters.safetensors"), weights)
        diagnostic = (
            "finite floating" if mutation in ("nonfinite", "integer") else "tensor keys/shapes"
        )
        if mutation == "truncated":
            diagnostic = "Cannot read native MLX adapter weights"
        with pytest.raises(ValueError, match=diagnostic):
            load_native_model(str(adapter))
        from typer.testing import CliRunner

        from soup_cli.cli import app

        args = ["infer", "--backend", "mlx", "--model", str(adapter),
                "--input", "prompts.jsonl", "--output", "corrupt-results.jsonl"]
        failed = CliRunner().invoke(app, args)
        assert failed.exit_code == 1, failed.output
        assert "Model loaded" not in failed.output
        assert not (tmp_path / "corrupt-results.jsonl").exists()
    mx.save_safetensors(str(adapter / "adapters.safetensors"), saved)


def _watchdog_wrapper(base, output, *, accum=1, enabled=True, patience=2, epochs=3, train_count=4):
    mx.random.seed(17)
    cfg = _config(base, output, dora=False)
    cfg.training.epochs = epochs
    cfg.training.gradient_accumulation_steps = accum
    cfg.training.loss_watchdog = enabled
    cfg.training.loss_watchdog_threshold = 0.000001
    cfg.training.loss_watchdog_patience = patience
    cfg.training.logging_steps = 1
    cfg.training.rewind_log = False
    # Validate through the same serialized configuration loader as the CLI.
    from soup_cli.config.loader import load_config_from_string

    wrapper = MLXSFTTrainerWrapper(load_config_from_string(cfg.model_dump_json()))
    wrapper.setup({"train": _rows()["train"][:1] * train_count})
    return wrapper


@pytest.mark.parametrize("accum, logging, patience", [(1, 1, 2), (2, 1, 2), (3, 1, 2), (2, 2, 1)])
def test_native_watchdog_stops_at_exact_update_and_reloads(
    tmp_path, monkeypatch, accum, logging, patience
):
    import soup_cli.trainer.mlx_optim as optim

    base = _checkpoint(tmp_path / "base")
    built = []
    build = optim.build_optimizer

    def record(plan):
        optimizer = build(plan)
        built.append(optimizer)
        return optimizer

    monkeypatch.setattr(optim, "build_optimizer", record)
    wrapper = _watchdog_wrapper(base, tmp_path / "stopped", accum=accum, patience=patience)
    wrapper.config.training.logging_steps = logging
    result = wrapper.train()
    assert result["stop_reason"] == "loss_watchdog"
    assert result["training_outcome"] == "stopped"
    assert result["completed_updates"] == 2
    assert result["total_steps"] == 2 * accum
    assert built[0].step.item() == 2
    metadata = json.loads((tmp_path / "stopped" / "adapter_config.json").read_text())
    assert metadata["completed_updates"] == 2
    assert metadata["completed_iterations"] == 2 * accum
    assert metadata["training_outcome"] == "stopped"
    assert metadata["stop_reason"] == "loss_watchdog"
    assert metadata["watchdog_report_unit"] == "optimizer updates"
    restored, _ = load(str(base), adapter_path=str(tmp_path / "stopped"))
    wrapper.model.eval()
    tokens = mx.array([[5, 6, 7]])
    assert mx.allclose(restored(tokens), wrapper.model(tokens), atol=1e-5, rtol=1e-5).item()

    # Identical initial model/data/constant LR: the stopped parameters must equal
    # exactly two updates, and differ from a disabled run's complete budget.
    control = _watchdog_wrapper(base, tmp_path / "control", accum=accum, enabled=False,
                                epochs=1, train_count=2 * accum)
    control.train()
    stopped_weights = dict(tree_flatten(wrapper.model.trainable_parameters()))
    control_weights = dict(tree_flatten(control.model.trainable_parameters()))
    assert built[1].step.item() == 2
    assert all(mx.allclose(value, control_weights[name], atol=1e-6, rtol=1e-6).item()
               for name, value in stopped_weights.items())
    full = _watchdog_wrapper(base, tmp_path / "full", accum=accum, enabled=False)
    full_result = full.train()
    full_weights = dict(tree_flatten(full.model.trainable_parameters()))
    assert full_result["total_steps"] > result["total_steps"]
    assert built[2].step.item() > 2
    assert any(not mx.array_equal(value, full_weights[name]).item()
               for name, value in stopped_weights.items())


def test_public_cli_watchdog_reports_early_stop_and_saves(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    base = _checkpoint(tmp_path / "base")
    wrapper = _watchdog_wrapper(base, tmp_path / "out", accum=2)
    wrapper.config.training.rewind_log = True
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in _rows()["train"]) + "\n")
    wrapper.config.data.train = str(data)
    wrapper.config.data.format = "plaintext"
    wrapper.config.data.val_split = 0
    config = tmp_path / "soup.json"
    config.write_text(wrapper.config.model_dump_json())
    outcome = CliRunner().invoke(app, ["train", "--config", str(config), "--yes"])
    assert outcome.exit_code == 0, (outcome.output, repr(outcome.exception))
    assert "Loss watchdog" in outcome.output
    assert "Training Stopped" in outcome.output
    assert "Training Complete!" not in outcome.output
    metadata = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert metadata["completed_updates"] == 2
    assert metadata["stop_reason"] == "loss_watchdog"
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    run = tracker.list_runs()[0]
    assert run["status"] == "completed"  # successful finalization, documented separately
    assert run["total_steps"] == 4
    assert [row["step"] for row in tracker.get_metrics(run["run_id"])
            if row["loss"] is not None] == [2, 4]
    records = [json.loads(line) for line in (tmp_path / "out" / "rewind.jsonl")
               .read_text().splitlines()]
    assert [(row["step"], row["micro"]) for row in records if row["kind"] == "batch"] == [
        (1, 0), (1, 1), (2, 0), (2, 1),
    ]
    restored, _ = load(str(base), adapter_path=str(tmp_path / "out"))
    assert mx.isfinite(restored(mx.array([[5, 6, 7]]))).all().item()


def test_native_watchdog_runs_final_eval_at_stopped_update(tmp_path):
    base = _checkpoint(tmp_path / "base")
    wrapper = _watchdog_wrapper(base, tmp_path / "out", accum=2)
    wrapper.config.training.eval_steps = 3
    wrapper._dataset["val"] = _rows()["train"][:1]
    result = wrapper.train()
    assert result["completed_updates"] == 2
    assert result["eval_updates"] == [2]
    assert wrapper._eval_log[0]["optimizer_step"] == 2
    assert math.isfinite(result["val_loss"])
