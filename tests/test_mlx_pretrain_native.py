"""Real offline native objectives, updates, artifacts and warm starts for #1983."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from soup_cli.config.schema import SoupConfig  # noqa: E402
from soup_cli.data.loader import load_dataset  # noqa: E402
from soup_cli.data.mlx_pretrain import TokenWindow  # noqa: E402
from soup_cli.trainer.dispatch import build_trainer  # noqa: E402
from soup_cli.trainer.mlx_pretrain import (  # noqa: E402
    causal_loss_sum,
    evaluate_windows,
    token_weighted_update,
)


def save_base(path, *, quantized=False, bits=4):
    """A genuine reloadable local Llama and tokenizer; no patched loader or downloads."""
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.utils import quantize_model, save_config, save_model
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    mx.random.seed(13)
    config = dict(model_type="llama", hidden_size=32, num_hidden_layers=1,
                  intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                  rms_norm_eps=1e-5, vocab_size=32, tie_word_embeddings=True)
    model = Model(ModelArgs(**config))
    if quantized:
        model, config = quantize_model(model, config, 32, bits)
    path.mkdir(parents=True, exist_ok=True)
    save_model(path, model)
    save_config(config, path / "config.json")
    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2,
             **{f"t{i}": i for i in range(3, 32)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", 1)],
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       bos_token="[BOS]", eos_token="[EOS]", pad_token="[EOS]")
    tokenizer.save_pretrained(path)
    return path


def loaded(path, *, quantized=False, base=None, train=None, val=None, **settings):
    path.mkdir(parents=True, exist_ok=True)
    if base is None:
        base = save_base(path / "base", quantized=quantized)
    train_path, val_path = path / "train.jsonl", path / "val.jsonl"
    rows = train if train is not None else ["t3 t4 t5 t6 " * 4, "t3 t4 t5 t6", "t3"]
    heldout = val if val is not None else ["t3 t4 t5 t6 " * 3, "t3"]
    for file, texts in ((train_path, rows), (val_path, heldout)):
        file.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    training = dict(quantization="4bit" if quantized else "none", batch_size=2,
                    gradient_accumulation_steps=2, epochs=2, lr=.03, seed=7,
                    data_seed=11, scheduler="constant", warmup_ratio=0., weight_decay=0.,
                    lora=dict(r=4, alpha=8, dropout=0))
    training.update(settings)
    cfg = SoupConfig(
        base=str(base), backend="mlx", task="pretrain", output=str(path / "output"),
        data=dict(train=str(train_path), validation=str(val_path), format="plaintext",
                  val_split=0, max_length=64),
        training=training,
    )
    wrapper = build_trainer(cfg)
    wrapper.setup(load_dataset(cfg.data, native_pretrain=True))
    return wrapper


def flat(tree):
    return {k: np.array(v) for k, v in tree_flatten(tree)}


def test_causal_loss_and_gradients_match_independent_float64_oracle():
    logits = np.array([[[.1, -.5, 1.2], [.2, .7, -.3], [.9, -.2, .4]],
                       [[.6, -.8, .3], [.2, .1, -.4], [-1., .5, .8]]], np.float32)
    targets = np.array([[2, 1, 2], [0, 2, 2]], np.int32)
    mask = np.array([[1, 1, 1], [1, 0, 0]], np.float32)
    z = logits.astype(np.float64)
    probability = np.exp(z - z.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    expected = -(np.log(np.take_along_axis(probability, targets[..., None], -1)[..., 0])
                 * mask).sum()
    expected_grad = probability.copy()
    for row in range(2):
        for col in range(3):
            expected_grad[row, col, targets[row, col]] -= 1
    expected_grad *= mask[..., None]

    def objective(value):
        return causal_loss_sum(value, mx.array(targets), mx.array(mask))

    loss, grad = mx.value_and_grad(objective)(mx.array(logits))
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    np.testing.assert_allclose(np.array(grad), expected_grad, atol=1e-7, rtol=1e-6)
    changed = logits.copy()
    changed[1, 2, 2] += 30  # PAD=EOS: a padded EOS has no effect.
    other_loss, other_grad = mx.value_and_grad(objective)(mx.array(changed))
    assert other_loss.item() == loss.item()
    np.testing.assert_array_equal(np.array(other_grad), np.array(grad))
    changed = logits.copy()
    changed[0, 2, 2] += 3  # A real final EOS must affect both value and derivative.
    other_loss, other_grad = mx.value_and_grad(objective)(mx.array(changed))
    assert abs(other_loss.item() - loss.item()) > .1
    assert np.max(np.abs(np.array(other_grad) - np.array(grad))) > .1
    with pytest.raises(ValueError, match="no supervised targets"):
        causal_loss_sum(mx.array(logits), mx.array(targets), mx.zeros_like(mx.array(mask)))
    with pytest.raises(ValueError, match="shapes must agree"):
        causal_loss_sum(mx.array(logits[:, :-1]), mx.array(targets), mx.array(mask))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = mx.array(np.arange(64, dtype=np.float32).reshape(8, 8) / 41.)

    def __call__(self, x):
        return self.table[x]


@pytest.mark.parametrize("max_norm", [.01, 100.])
def test_unequal_microbatches_match_token_weighted_reference_then_clip(max_norm):
    from soup_cli.trainer.mlx_sft import _clipping_optimizer

    model, reference = TinyLM(), TinyLM()
    original = np.array(model.table)
    batches = [[TokenWindow((1, 3, 4, 5), 0)], [TokenWindow((1, 6), 1)]]
    # Independent complete batch, shifted once; EOS-like padding is position masked.
    x, y = mx.array([[1, 3, 4], [1, 2, 2]]), mx.array([[3, 4, 5], [6, 2, 2]])
    mask = mx.array([[1., 1., 1.], [1., 0., 0.]])

    def loss(m):
        z = m(x)
        lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
        return -(mx.take_along_axis(lp, y[..., None], axis=-1)[..., 0] * mask).sum() / 4

    expected, grads = nn.value_and_grad(reference, loss)(reference)
    # Clip after normalizing, using a threshold that actually binds.
    norm = float(mx.sqrt((grads["table"] ** 2).sum()).item())
    assert norm > .01
    manual = original - .2 * np.array(grads["table"]) * min(1., max_norm / norm)
    optimizer = _clipping_optimizer(optim.SGD(learning_rate=.2), max_norm)
    measured, count = token_weighted_update(model, optimizer, batches, pad_id=2)
    assert count == 4 and measured == pytest.approx(expected.item(), abs=1e-6)
    assert int(optimizer.step.item()) == 1
    np.testing.assert_allclose(np.array(model.table), manual, atol=1e-7, rtol=1e-6)
    assert np.max(np.abs(np.array(model.table) - original)) > 1e-4


def test_evaluation_weights_short_tail_and_does_not_update_or_consume_rng():
    model = TinyLM()
    model.train()
    before = flat(model.parameters())
    rng_before = np.array(mx.random.state[0])
    windows = [TokenWindow((1, 3, 4), 0), TokenWindow((1, 6), 1), TokenWindow((1, 2), 2)]
    z = before["table"].astype(np.float64)
    def ce(i, j):
        return np.log(np.exp(z[i]).sum()) - z[i, j]
    expected = (ce(1, 3) + ce(3, 4) + ce(1, 6) + ce(1, 2)) / 4
    assert evaluate_windows(model, windows, batch_size=2, pad_id=2) == pytest.approx(
        expected, abs=1e-6,
    )
    assert model.training
    np.testing.assert_array_equal(np.array(mx.random.state[0]), rng_before)
    for name, value in flat(model.parameters()).items():
        np.testing.assert_array_equal(value, before[name])
    with pytest.raises(ValueError, match="no supervised targets"):
        evaluate_windows(model, [], batch_size=2, pad_id=2)


@pytest.mark.parametrize("quantized", [False, True], ids=["dense", "4bit"])
def test_real_wrapper_learning_frozen_base_and_fresh_process_reload(tmp_path, quantized):
    wrapper = loaded(tmp_path, quantized=quantized)
    original = flat(wrapper.model.parameters())
    if quantized:
        assert wrapper._base_quantization["bits"] == 4
        assert isinstance(wrapper.model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    result = wrapper.train()
    assert result["total_steps"] == 2
    assert result["consumed_targets"] == wrapper.train_stats["supervised_tokens"] * 2
    assert result["val_loss"] < result["initial_val_loss"] - .05
    after = flat(wrapper.model.parameters())
    for name, value in original.items():
        parent, leaf = name.rsplit(".", 1)
        mapped = name if name in after else f"{parent}.linear.{leaf}"
        np.testing.assert_array_equal(after[mapped], value, err_msg=name)
    adapters = flat(wrapper.model.trainable_parameters())
    assert {k.rsplit(".", 1)[-1] for k in adapters} == {"lora_a", "lora_b"}
    assert all(np.abs(v).max() > 1e-4 for k, v in adapters.items() if k.endswith("lora_b"))
    tokens = [[1, 3, 4, 5, 6]]
    live = np.array(wrapper.model(mx.array(tokens)))
    script = """
import json, sys
import mlx.core as mx
import numpy as np
from soup_cli.utils.mlx_generation import load_native_model
model, _ = load_native_model(sys.argv[1])
print(json.dumps(np.array(model(mx.array([[1,3,4,5,6]]))).tolist()))
"""
    def run(path):
        completed = subprocess.run([sys.executable, "-c", script, str(path)],
                                   capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return np.array(json.loads(completed.stdout.strip().splitlines()[-1]))
    reloaded = run(wrapper.config.output)
    np.testing.assert_allclose(reloaded, live, atol=1e-5, rtol=1e-5)
    assert np.abs(run(wrapper.config.base) - live).max() > .05
    meta = json.loads((Path(wrapper.config.output) / "adapter_config.json").read_text())
    assert meta["task"] == "pretrain" and meta["contract"]["task"] == "pretrain"
    assert meta["lora_parameters"]["keys"] == ["self_attn.q_proj", "self_attn.v_proj"]
    assert meta["completed_updates"] == 2 and meta["supervised_tokens_per_second"] > 0


def test_checkpoints_explicit_and_auto_warm_start_and_mismatches(tmp_path):
    from soup_cli.commands.train import _resolve_mlx_checkpoint
    from soup_cli.utils.mlx_pretrain_checkpoint import load_snapshot, read_snapshot

    wrapper = loaded(tmp_path, save_steps=1, eval_steps=1)
    wrapper.train()
    output = Path(wrapper.config.output)
    first, second = (output / f"{i:07d}_adapters.safetensors" for i in (1, 2))
    assert read_snapshot(first)["completed_updates"] == 1
    assert read_snapshot(second)["completed_updates"] == 2
    assert _resolve_mlx_checkpoint("auto", str(output)) == str(second)
    assert _resolve_mlx_checkpoint(str(first), str(output)) == str(first)
    warm = loaded(tmp_path / "warm", base=wrapper.config.base)
    # Seeded init plus a strict restore before any new optimizer update.
    mx.random.seed(7)
    warm._apply_lora(warm.model)
    contract = read_snapshot(second)["contract"]
    load_snapshot(warm.model, second, contract)
    for name, value in flat(wrapper.model.trainable_parameters()).items():
        np.testing.assert_array_equal(flat(warm.model.trainable_parameters())[name], value)
    continued = loaded(tmp_path / "continued", base=wrapper.config.base)
    result = continued.train(resume_from_checkpoint=str(first))
    assert result["total_steps"] == 2 and int(continued.optimizer.step.item()) == 2
    for field in ("task", "model", "tokenizer_sha256", "lora_parameters", "window_policy",
                  "base_quantization"):
        with pytest.raises(ValueError, match="incompatible warm-start contract"):
            load_snapshot(warm.model, second, {**contract, field: "wrong"})
    saved = second.read_bytes()
    second.write_bytes(saved[:17])
    with pytest.raises(ValueError, match="incomplete or corrupt"):
        load_snapshot(warm.model, second, contract)
    second.write_bytes(saved)
    second.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="cannot warm-start"):
        load_snapshot(warm.model, second, contract)


def test_invalid_targets_and_quantization_cannot_succeed(tmp_path):
    wrapper = loaded(tmp_path)
    wrapper.config.training.lora.target_modules = ["does.not.exist"]
    with pytest.raises(ValueError, match="resolved linear LoRA target keys"):
        wrapper.train()
    wrapper = loaded(tmp_path / "other")
    wrapper.config.training.quantization = "4bit"
    with pytest.raises(ValueError, match="does not match actual base tensors"):
        wrapper.train()


def test_failed_snapshot_is_not_a_valid_artifact(tmp_path, monkeypatch):
    from soup_cli.utils.mlx_pretrain_checkpoint import read_snapshot, save_snapshot

    wrapper = loaded(tmp_path)
    wrapper._apply_lora(wrapper.model)
    path = tmp_path / "adapters.safetensors"
    def fail(name, weights):
        Path(name).write_bytes(b"partial")
        raise OSError("simulated interrupted save")
    monkeypatch.setattr(mx, "save_safetensors", fail)
    with pytest.raises(OSError, match="interrupted save"):
        save_snapshot(wrapper.model, path, {"contract": {"task": "pretrain"}})
    assert not path.exists() and not path.with_suffix(".json").exists()
    assert not list(tmp_path.glob("*.partial*"))
    with pytest.raises(ValueError, match="cannot warm-start"):
        read_snapshot(path)


@pytest.mark.parametrize("count,batch,accum,epochs,steps", [(1, 4, 4, 1, 1),
    (5, 2, 2, 3, 6), (4, 2, 2, 2, 2)])
def test_real_partial_groups_schedules_evaluation_and_seed(tmp_path, count, batch, accum,
                                                          epochs, steps):
    wrapper = loaded(tmp_path, train=["t3 t4 t5"] * count, val=["t3"],
                     batch_size=batch, gradient_accumulation_steps=accum, epochs=epochs,
                     eval_steps=4, save_steps=2, scheduler="linear", warmup_ratio=.5)
    result = wrapper.train()
    assert result["total_steps"] == steps == int(wrapper.optimizer.step.item())
    assert result["consumed_targets"] == count * 4 * epochs
    assert result["eval_updates"] == ([4, 6] if steps == 6 else [steps])
    meta = json.loads((Path(wrapper.config.output) / "adapter_config.json").read_text())
    assert meta["warmup_updates"] == int(.5 * steps)
    assert meta["total_updates"] == steps
    repeat = loaded(tmp_path / "repeat", base=wrapper.config.base,
                    train=["t3 t4 t5"] * count, val=["t3"], batch_size=batch,
                    gradient_accumulation_steps=accum, epochs=epochs,
                    eval_steps=4, save_steps=2, scheduler="linear", warmup_ratio=.5)
    repeated = repeat.train()
    assert repeated["losses"] == result["losses"]
    for name, value in flat(wrapper.model.trainable_parameters()).items():
        np.testing.assert_array_equal(value, flat(repeat.model.trainable_parameters())[name])


@pytest.mark.parametrize("damage", ["keys", "shape", "nan"])
def test_even_a_digest_matching_snapshot_cannot_drop_or_corrupt_tensors(tmp_path, damage):
    from soup_cli.data.mlx_pretrain import file_sha256
    from soup_cli.utils.mlx_pretrain_checkpoint import load_snapshot, read_snapshot

    wrapper = loaded(tmp_path)
    wrapper.train()
    path = Path(wrapper.config.output) / "adapters.safetensors"
    meta = read_snapshot(path)
    weights = mx.load(str(path))
    key = next(iter(weights))
    if damage == "keys":
        weights["unresolved.auto"] = weights.pop(key)
    elif damage == "shape":
        weights[key] = weights[key][:1]
    else:
        weights[key] = mx.full(weights[key].shape, float("nan"))
    mx.save_safetensors(str(path), weights)
    meta["weights_sha256"] = file_sha256(path)
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="finite floating" if damage == "nan" else "keys/shapes"):
        load_snapshot(wrapper.model, path, meta["contract"])


def test_cli_dense_and_resume_auto_in_a_clean_process(tmp_path):
    wrapper = loaded(tmp_path, epochs=1, save_steps=1)
    config = tmp_path / "soup.yaml"
    import yaml

    config.write_text(yaml.safe_dump(wrapper.config.model_dump(mode="json", exclude_unset=True)))
    script = """
import json, sys
from typer.testing import CliRunner
from soup_cli.cli import app
argv = ['train', '--config', sys.argv[1], '--yes', '--no-reexec']
if len(sys.argv) > 2:
    argv += ['--resume', 'auto']
r = CliRunner().invoke(app, argv)
print(r.output)
if r.exception:
    raise r.exception
assert r.exit_code == 0
assert not any(name in sys.modules for name in ('torch', 'trl', 'peft'))
"""
    for extra in ([], ["resume"]):
        completed = subprocess.run([sys.executable, "-c", script, str(config), *extra],
                                   cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        if extra:
            assert "weights-only warm start" in completed.stdout
    meta = json.loads((Path(wrapper.config.output) / "adapter_config.json").read_text())
    assert meta["warm_start"] is True and meta["completed_updates"] == 1


def test_configured_initialization_seed_actually_changes_the_adapter(tmp_path):
    first = loaded(tmp_path / "first", seed=7, epochs=1)
    second = loaded(tmp_path / "second", seed=8, epochs=1, base=first.config.base)
    first.train()
    second.train()
    a, b = flat(first.model.trainable_parameters()), flat(second.model.trainable_parameters())
    assert any(np.abs(a[k] - b[k]).max() > 1e-3 for k in a)
