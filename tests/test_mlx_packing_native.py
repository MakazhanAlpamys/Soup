"""Native isolated packing contracts on real local Llama models."""

import json

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from typer.testing import CliRunner  # noqa: E402

from soup_cli.cli import app  # noqa: E402
from tests.test_mlx_dora_native import _config, _selective_fixture  # noqa: E402


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("bf16", [False, True])
def test_public_packing_concatenates_real_masked_examples(
    tmp_path, monkeypatch, quantized, accumulation, bf16,
):
    from mlx_lm.tuner import trainer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    base = tmp_path / "base"
    messages, ids = _selective_fixture(base, quantized)
    if bf16:
        _cast_checkpoint_bf16(base)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"messages": messages}) for _ in range(4)) + "\n")
    cfg = _config(base, tmp_path / "out", dora=False, quantized=quantized)
    cfg.data.train = str(data)
    cfg.data.format = "chatml"
    cfg.data.val_split = 0.0
    cfg.data.train_on_messages_with_train_field = True
    cfg.training.packing = True
    cfg.training.gradient_accumulation_steps = accumulation
    config = tmp_path / "config.json"
    config.write_text(cfg.model_dump_json())
    observed = []
    real_train = trainer.train

    def observe_train(**kwargs):
        iterator = kwargs.get("iterate_batches", trainer.iterate_batches)

        def observe_batches(*args, **options):
            for batch in iterator(*args, **options):
                observed.append(batch[0].tolist())
                yield batch

        return real_train(**{**kwargs, "iterate_batches": observe_batches})

    monkeypatch.setattr(trainer, "train", observe_train)
    result = CliRunner().invoke(app, ["train", "--config", str(config), "--yes"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert observed
    assert observed[0][0][:2 * len(ids)] == ids * 2
    assert len(observed) == 2  # Four records fit into two context-bounded packs.
    import os
    import subprocess
    import sys

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load

    original, _ = load(str(base))
    trained, _ = load(str(base), adapter_path=str(tmp_path / "out"))
    frozen = dict(tree_flatten(original.parameters()))
    reloaded = dict(tree_flatten(trained.parameters()))
    for name, value in frozen.items():
        parent, leaf = name.rsplit(".", 1)
        assert mx.array_equal(reloaded.get(f"{parent}.linear.{leaf}", reloaded.get(name)),
                              value).item(), name
    weights = mx.load(str(tmp_path / "out" / "adapters.safetensors"))
    assert any(mx.abs(value).sum().item() > 0 for name, value in weights.items()
               if name.endswith("lora_b"))
    probe = mx.array([[2, 3, 4, 1]])
    trained.eval()
    original.eval()
    expected = trained(probe)
    assert mx.abs(expected - original(probe)).max().item() > 1e-5
    script = """
import sys
import mlx.core as mx
from mlx_lm import load
model, _ = load(sys.argv[1], adapter_path=sys.argv[2])
model.eval()
mx.save_safetensors(sys.argv[3], {'logits': model(mx.array([[2,3,4,1]]))})
"""
    checkpoint = tmp_path / "fresh-logits.safetensors"
    child = subprocess.run(
        [sys.executable, "-c", script, str(base), str(tmp_path / "out"), str(checkpoint)],
        capture_output=True, text=True, timeout=90,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert mx.allclose(mx.load(str(checkpoint))["logits"], expected, atol=1e-6).item()
    metadata = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert metadata["packing"]["supervised_tokens"] > 0
    assert metadata["packing"]["packs"] == 2
    assert not (tmp_path / "out" / "rewind.jsonl").exists()
    assert "Rewind log off" in result.output


def _cast_checkpoint_bf16(path):
    import mlx.core as mx

    file = path / "model.safetensors"
    weights = mx.load(str(file))
    converted = {
        name: value.astype(mx.bfloat16) if value.dtype == mx.float32 else value
        for name, value in weights.items()
    }
    mx.eval(*converted.values())
    assert any(value.dtype == mx.bfloat16 and mx.abs(value).sum().item() > 0
               for value in converted.values())
    # Older MLX can retain lazy file-backed inputs: never truncate their source
    # while saving the converted checkpoint.
    temporary = path / "bf16.safetensors"
    mx.save_safetensors(str(temporary), converted)
    temporary.replace(file)


def _adapted_model(path, quantized, grouped=False, bf16=False):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
    from tests.test_mlx_dora_native import _checkpoint

    _checkpoint(path, quantized)
    if grouped:
        from mlx_lm.models.llama import Model, ModelArgs

        config = json.loads((path / "config.json").read_text())
        config["num_key_value_heads"] = 2
        config["rope_traditional"] = True
        model = Model(ModelArgs.from_dict(config))
        if quantized:
            import mlx.nn as nn

            nn.quantize(model, group_size=32, bits=4)
        mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
        (path / "config.json").write_text(json.dumps(config))
    if bf16:
        _cast_checkpoint_bf16(path)
    wrapper = MLXSFTTrainerWrapper(_config(path, path.parent / "adapter",
                                          dora=False, quantized=quantized))
    wrapper.setup({"train": [{"text": "alpha beta"}]})
    wrapper._apply_lora(wrapper.model)
    for name, value in tree_flatten(wrapper.model.trainable_parameters()):
        if name.endswith("lora_b"):
            wrapper.model.load_weights([(name, mx.ones_like(value) * 0.02)], strict=False)
    wrapper.model.eval()
    return wrapper.model


def _reference_sum(model, rows):
    import mlx.core as mx

    total = mx.array(0.0)
    count = 0
    for tokens, labels in rows:
        logits = model(mx.array([tokens[:-1]])).astype(mx.float32)
        targets = mx.array([labels[1:]])
        ce = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
            logits, mx.maximum(targets, 0)[..., None], axis=-1,
        ).squeeze(-1)
        total = total + (ce * (targets != -100)).sum()
        count += sum(label != -100 for label in labels[1:])
    return total / count


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_packed_logits_loss_gradients_and_earlier_segment_isolation(
    tmp_path, quantized, grouped, bf16,
):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from soup_cli.trainer.mlx_packing import (
        PackedDataset,
        packed_iterate_batches,
        packed_logits,
        packed_loss,
        validate_packing_model,
    )

    model = _adapted_model(tmp_path / "base", quantized, grouped, bf16)
    # BF16 projections round to a 7-bit mantissa; compare at that precision.
    tolerance = 0.008 if bf16 else 2e-5
    validate_packing_model(model)
    rows = [([2, 3, 4, 1], [-100, 3, 4, 1]),
            ([2, 5, 6, 7, 1], [-100, -100, 6, 7, 1])]
    dataset = PackedDataset(rows, kind="labels", max_length=16, vocab_size=32)
    batch, payload = next(packed_iterate_batches(dataset, 1, 16))
    labels, segments, positions, _ = payload
    logits = packed_logits(model, batch, segments, positions)
    for start, (tokens, _) in zip([0, 4], rows):
        expected = model(mx.array([tokens]))
        assert mx.allclose(logits[:, start:start + len(tokens)], expected,
                           atol=tolerance, rtol=tolerance).item()
    actual, gradient = nn.value_and_grad(model, packed_loss)(model, batch, payload)
    expected, reference = nn.value_and_grad(model, _reference_sum)(model, rows)
    assert actual[1].item() == 6
    assert actual[0].item() == pytest.approx(expected.item(), abs=tolerance)
    reference = dict(tree_flatten(reference))
    for name, value in tree_flatten(gradient):
        assert mx.allclose(value, reference[name], atol=tolerance, rtol=tolerance).item(), name

    perturbed = batch.at[0, 1].add(8)  # Earlier segment; causal attention alone cannot isolate it.
    changed = packed_logits(model, perturbed, segments, positions)
    assert mx.allclose(changed[:, 4:], logits[:, 4:], atol=1e-6).item()
    assert mx.abs(model(perturbed)[:, 4:] - model(batch)[:, 4:]).max().item() > 1e-4

    def later_loss(inner, tokens):
        values = packed_logits(inner, tokens, segments, positions)[:, 4:-1]
        return nn.losses.cross_entropy(values, batch[:, 5:]).mean()

    _, original_grad = nn.value_and_grad(model, later_loss)(model, batch)
    _, perturbed_grad = nn.value_and_grad(model, later_loss)(model, perturbed)
    changed_grad = dict(tree_flatten(perturbed_grad))
    for name, value in tree_flatten(original_grad):
        assert mx.allclose(value, changed_grad[name], atol=1e-6).item(), name


def test_unequal_pack_accumulation_uses_the_group_target_denominator(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from soup_cli.trainer.mlx_packing import PackedDataset, packed_iterate_batches, packed_loss

    model = _adapted_model(tmp_path / "base", False)
    rows = [([2, 3, 4, 5, 6, 1], [-100, 3, 4, 5, 6, 1]),
            ([2, 7, 8, 9, 1], [-100, -100, -100, 9, 1])]
    dataset = PackedDataset(rows, kind="labels", max_length=8, vocab_size=32)
    iterator = packed_iterate_batches(dataset, 1, 8, loop=True, seed=7, accumulation=2)
    gradients = []
    means = []
    for _ in range(2):
        batch, payload = next(iterator)
        actual, gradient = nn.value_and_grad(model, packed_loss)(model, batch, payload)
        assert actual[1].item() in (2, 5)
        gradients.append(dict(tree_flatten(gradient)))
        means.append(actual[0].item())
    expected, reference = nn.value_and_grad(model, _reference_sum)(model, rows)
    assert abs(means[0] - means[1]) > 1e-4  # Unequal counts and unequal losses are meaningful.
    for name, value in tree_flatten(reference):
        assert mx.allclose((gradients[0][name] + gradients[1][name]) / 2, value,
                           atol=2e-5, rtol=2e-5).item(), name
    assert mx.isfinite(expected).item()


@pytest.mark.parametrize("mode", ["responses_only", "mask_history", "cache"])
def test_public_target_modes_and_cached_validation_keep_their_exact_targets(
    tmp_path, monkeypatch, mode,
):
    import mlx.core as mx
    from datasets import Dataset, DatasetDict
    from mlx_lm.tuner import trainer

    from soup_cli.trainer.mlx_masking import MaskedChatDataset
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    monkeypatch.chdir(tmp_path)
    base = tmp_path / "base"
    messages, ids = _selective_fixture(base, False)
    rows = [{"messages": messages}] * 4
    source = tmp_path / "data.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    cfg = _config(base, tmp_path / "out", dora=False)
    cfg.data.train = str(source)
    cfg.data.format = "chatml"
    cfg.data.val_split = 0.0
    cfg.data.train_on_responses_only = True
    cfg.data.mask_history = mode == "mask_history"
    cfg.training.packing = True
    observed = {}
    setup = MLXSFTTrainerWrapper.setup

    def observe_setup(self, dataset):
        setup(self, dataset)
        observed["wrapper"] = self
        raw = MaskedChatDataset(rows, self.tokenizer, mask_history=mode == "mask_history")
        tokens, mask = raw.process(rows[0])
        observed["expected"] = [token if keep else -100 for token, keep in zip(tokens, mask)]

    if mode == "cache":
        labels = [-100] * len(ids)
        labels[-3:] = [3, 4, 1]  # Explicit targets differ from the stored input IDs.
        cache = tmp_path / "cache"
        records = Dataset.from_dict({"input_ids": [ids]*4, "labels": [labels]*4})
        DatasetDict(train=records, validation=records.select([0, 1])).save_to_disk(str(cache))
        # The public CLI still validates the configured source's declared format.
        source.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        cfg.data.format = "pre_tokenized"
        cfg.data.tokenized_path = str(cache)
        cfg.training.eval_steps = 1
        observed["cached_labels"] = labels
    monkeypatch.setattr(MLXSFTTrainerWrapper, "setup", observe_setup)
    real_train = trainer.train

    def observe_train(**kwargs):
        iterator = kwargs["iterate_batches"]

        def observe_batches(*args, **options):
            for batch in iterator(*args, **options):
                actual = batch[1][0][0, :2 * len(ids)].tolist()
                expected = observed.get("cached_labels", observed["expected"])
                expected = [-100, *expected[1:]] * 2
                assert actual == expected
                yield batch

        return real_train(**{**kwargs, "iterate_batches": observe_batches})

    monkeypatch.setattr(trainer, "train", observe_train)
    path = tmp_path / "config.json"
    path.write_text(cfg.model_dump_json())
    result = CliRunner().invoke(app, ["train", "--config", str(path), "--yes"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    if mode == "cache":
        wrapper = observed["wrapper"]
        wrapper.model.eval()
        rows = [(ids, observed["cached_labels"])] * 2
        expected = _reference_sum(wrapper.model, rows).item()
        assert wrapper._eval_log[-1]["val_loss"] == pytest.approx(expected, abs=2e-5)
        assert mx.isfinite(mx.array(expected)).item()


def test_an_incomplete_accumulation_group_is_refused_instead_of_dropping_a_pack(tmp_path):
    from soup_cli.trainer.mlx_packing import PackedDataset, packed_iterate_batches

    row = ([2, 3, 1], [-100, 3, 1])
    odd = PackedDataset([row]*3, kind="labels", max_length=3, vocab_size=8)
    with pytest.raises(ValueError, match="pack count divisible"):
        next(packed_iterate_batches(odd, 1, 3, loop=True, accumulation=2))
    even = PackedDataset([row]*2, kind="labels", max_length=3, vocab_size=8)
    assert next(packed_iterate_batches(even, 1, 3, loop=True, accumulation=2))[0].shape == (1, 3)


@pytest.mark.parametrize("variant", ["rope_scaling", "sliding_window"])
def test_public_setup_refuses_unqualified_llama_layout_and_keeps_packing_off_valid(
    tmp_path, variant,
):
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
    from tests.test_mlx_dora_native import _checkpoint

    base = _checkpoint(tmp_path / "base")
    config = json.loads((base / "config.json").read_text())
    config[variant] = {"type": "linear", "factor": 2.0} if variant == "rope_scaling" else 16
    (base / "config.json").write_text(json.dumps(config))
    cfg = _config(base, tmp_path / "out", dora=False)
    cfg.training.packing = False
    MLXSFTTrainerWrapper(cfg).setup({"train": [{"text": "alpha beta"}]})
    cfg.training.packing = True
    with pytest.raises(ValueError, match="full-attention Llama.*unscaled RoPE"):
        MLXSFTTrainerWrapper(cfg).setup({"train": [{"text": "alpha beta"}]})


@pytest.mark.parametrize("package,version,diagnostic", [
    ("core", "0.31.1", "requires mlx >="), ("core", "0.33.0", "requires mlx >="),
    ("lm", "0.31.2", "requires mlx-lm >="), ("lm", "0.33.0", "requires mlx-lm >="),
])
def test_packing_version_guard_has_a_real_supported_control(tmp_path, monkeypatch, package, version,
                                                          diagnostic):
    import mlx.core as mx
    import mlx_lm

    from soup_cli.trainer.mlx_packing import validate_packing_model

    model = _adapted_model(tmp_path / "base", False)
    validate_packing_model(model)
    monkeypatch.setattr(mx if package == "core" else mlx_lm, "__version__", version)
    with pytest.raises(ValueError, match=diagnostic):
        validate_packing_model(model)


def test_packed_distributed_refusal_reaches_the_worker_count_guard():
    from types import SimpleNamespace

    from soup_cli.trainer.mlx_packing import PackedDataset, packed_iterate_batches

    data = PackedDataset([([2, 3, 1], [-100, 3, 1])], kind="labels", max_length=8, vocab_size=8)
    assert next(packed_iterate_batches(data, 1, 8, comm_group=SimpleNamespace(size=lambda: 1)))
    with pytest.raises(ValueError, match="single-process"):
        next(packed_iterate_batches(data, 1, 8, comm_group=SimpleNamespace(size=lambda: 2)))


@pytest.mark.parametrize("quantized", [False, True])
def test_public_plain_text_packing_preserves_span_targets(tmp_path, monkeypatch, quantized):
    from mlx_lm.tuner import trainer

    from tests.test_mlx_dora_native import _checkpoint

    monkeypatch.chdir(tmp_path)
    base = _checkpoint(tmp_path / "base", quantized)
    source = tmp_path / "data.jsonl"
    source.write_text("\n".join(json.dumps({"text": "alpha beta gamma"}) for _ in range(4)))
    cfg = _config(base, tmp_path / "out", dora=False, quantized=quantized)
    cfg.data.train = str(source)
    cfg.data.format = "plaintext"
    cfg.data.val_split = 0
    cfg.training.packing = True
    cfg.training.save_steps = 0
    real_train = trainer.train
    observed = []

    def train(**kwargs):
        observed.append(kwargs["train_dataset"][0])
        return real_train(**kwargs)

    monkeypatch.setattr(trainer, "train", train)
    path = tmp_path / "config.json"
    path.write_text(cfg.model_dump_json())
    result = CliRunner().invoke(app, ["train", "--config", str(path), "--yes"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    tokens, labels, segments, positions = observed[0]
    # The native completion dataset appends one EOS per record.
    assert tokens == [5, 6, 7, 1] * 4
    assert labels == [-100, 6, 7, 1] * 4
    assert segments == [0]*4 + [1]*4 + [2]*4 + [3]*4
    assert positions == [0, 1, 2, 3] * 4


def test_actual_rope_receives_reset_positions_for_each_token(tmp_path, monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn

    from soup_cli.trainer.mlx_packing import PackedDataset, packed_iterate_batches, packed_logits

    model = _adapted_model(tmp_path / "base", False)
    data = PackedDataset([([2, 3, 1], [-100, 3, 1])]*2,
                         kind="labels", max_length=8, vocab_size=32)
    batch, (_, segments, positions, _) = next(packed_iterate_batches(data, 1, 8))
    actual_call = nn.RoPE.__call__
    seen = []

    def observe(self, values, offset=0):
        seen.append(offset.tolist())
        return actual_call(self, values, offset=offset)

    monkeypatch.setattr(nn.RoPE, "__call__", observe)
    logits = packed_logits(model, batch, segments, positions)
    mx.eval(logits)
    assert seen == [[0, 1, 2, 0, 1, 2]] * (2 * len(model.layers))


def test_public_training_refuses_an_incomplete_group_before_any_optimizer_step(
    tmp_path, monkeypatch,
):
    from mlx_lm.tuner import trainer

    monkeypatch.chdir(tmp_path)
    base = tmp_path / "base"
    messages, _ = _selective_fixture(base, False)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"messages": messages}) for _ in range(5)))
    cfg = _config(base, tmp_path / "out", dora=False)
    cfg.data.train = str(data)
    cfg.data.format = "chatml"
    cfg.data.val_split = 0
    cfg.data.train_on_messages_with_train_field = True
    cfg.training.packing = True
    cfg.training.gradient_accumulation_steps = 2
    path = tmp_path / "config.json"
    path.write_text(cfg.model_dump_json())

    def must_not_train(**kwargs):
        raise AssertionError("incomplete packs must fail before training")

    monkeypatch.setattr(trainer, "train", must_not_train)
    result = CliRunner().invoke(app, ["train", "--config", str(path), "--yes"])
    assert isinstance(result.exception, ValueError)
    assert "pack count divisible" in str(result.exception)
    assert not list((tmp_path / "out").glob("*.safetensors"))


def test_packing_refuses_embedding_adapters_with_an_ordinary_linear_control(tmp_path):
    from mlx_lm.tuner.lora import LoRALinear

    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
    from tests.test_mlx_dora_native import _checkpoint

    base = _checkpoint(tmp_path / "base")
    cfg = _config(base, tmp_path / "out", dora=False, targets=["self_attn.q_proj"])
    cfg.training.packing = True
    control = MLXSFTTrainerWrapper(cfg)
    control.setup({"train": [{"text": "alpha beta"}]})
    control._apply_lora(control.model)
    assert isinstance(control.model.layers[0].self_attn.q_proj, LoRALinear)
    cfg.training.lora.target_modules = ["model.embed_tokens"]
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.setup({"train": [{"text": "alpha beta"}]})
    with pytest.raises(ValueError, match="packing requires dense linear targets"):
        wrapper._apply_lora(wrapper.model)


def test_packing_refuses_actual_eight_bit_base_weights(tmp_path):
    import mlx.nn as nn

    from soup_cli.trainer.mlx_packing import validate_packing_model

    model = _adapted_model(tmp_path / "base", False)
    validate_packing_model(model)
    # Convert a previously unadapted projection, so this reaches the base guard.
    model.layers[0].self_attn.k_proj = nn.QuantizedLinear.from_linear(
        model.layers[0].self_attn.k_proj, group_size=32, bits=8,
    )
    assert model.layers[0].self_attn.k_proj.bits == 8
    with pytest.raises(ValueError, match="dense or 4-bit base weights"):
        validate_packing_model(model)
