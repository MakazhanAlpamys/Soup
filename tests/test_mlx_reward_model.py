"""Native MLX reward training, safe bundles, and portable input/config contracts."""

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.trainer.mlx_reward_model import MLXRewardModelTrainerWrapper
from soup_cli.utils.mlx_reward import (
    NativeRewardScorer,
    load_reward_scorer,
    refuse_reward_artifact,
    score_completions,
)
from soup_cli.utils.mlx_reward_bundle import base_identity, publish_bundle, read_bundle
from soup_cli.utils.mlx_reward_data import (
    RewardPair,
    encode_completion,
    padded_arrays,
    plan_windows,
    prepare_pairs,
)
from soup_cli.utils.mlx_reward_native import (
    accumulated_gradients,
    evaluate_pairs,
    make_reward_model,
    pair_loss,
)


def tiny_model(seed=42):
    mx.random.seed(1)
    args = ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=32,
    )
    model = Model(args)
    model.freeze()
    linear_to_lora_layers(
        model,
        2,
        {"rank": 2, "scale": 2.0, "dropout": 0.0, "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
    )
    return make_reward_model(model, seed)


def pairs():
    return [RewardPair(i, [1, 4 + i, 12, 2], [1, 4 + i, 13, 14, 2]) for i in range(5)]


def flat(value):
    return dict(tree_flatten(value))


@pytest.mark.usefixtures("native_stack")
def test_scalar_gradient_oracle():
    chosen = mx.array([1.0, -0.3, 0.8])
    rejected = mx.array([-0.2, 0.9, 0.4])

    class Fixed:
        def __init__(self, x):
            self.x = x

        def __call__(self, *_):
            return mx.concatenate([self.x, rejected])

    gradient = mx.grad(lambda x: pair_loss(Fixed(x), None, None))(chosen)
    expected = -mx.sigmoid(-(chosen - rejected)) / 3
    assert mx.allclose(gradient, expected, atol=1e-7).item()
    assert pair_loss(Fixed(chosen), None, None).item() > 0


@pytest.mark.usefixtures("native_stack")
def test_padding_eos_and_norm_pooling():
    model = tiny_model()
    model.eval()
    right = mx.array([[1, 4, 12, 2, 2], [1, 5, 12, 16, 2]])
    mask = mx.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=mx.bool_)
    left = mx.array([[2, 1, 4, 12, 2], [1, 5, 12, 16, 2]])
    leftmask = mx.array([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=mx.bool_)
    actual = model(right, mask)
    expected = model.backbone.model(right)[mx.arange(2), mx.array([3, 4])].astype(mx.float32)
    expected = (expected @ model.score["weight"].T).squeeze(-1)
    assert mx.allclose(actual, expected, atol=1e-6).item()
    assert mx.allclose(actual, model(left, leftmask), atol=1e-6).item()
    for index, ids in enumerate([[1, 4, 12, 2], [1, 5, 12, 16, 2]]):
        assert abs(model(*padded_arrays([ids], 2)).item() - actual[index].item()) < 1e-6


@pytest.mark.usefixtures("native_stack")
def test_pair_weighted_gradients_and_adamw_moments():
    whole = tiny_model()
    split = tiny_model()
    examples = pairs()
    loss1, g1 = accumulated_gradients(whole, [examples], 2)
    loss2, g2 = accumulated_gradients(split, [examples[:2], examples[2:4], examples[4:]], 2)
    assert abs(loss1 - loss2) < 1e-6
    for key, value in flat(g1).items():
        assert mx.allclose(value, flat(g2)[key], atol=1e-6).item(), key
    a = optim.AdamW(learning_rate=1e-3)
    b = optim.AdamW(learning_rate=1e-3)
    a.update(whole, g1)
    b.update(split, g2)
    mx.eval(a.state, b.state)
    for key, value in flat(a.state).items():
        assert mx.allclose(value, flat(b.state)[key], atol=1e-7).item(), key


@pytest.mark.usefixtures("native_stack")
def test_frozen_base_trainable_head_lora_and_eval_state():
    model = tiny_model()
    before = {k: mx.array(v) for k, v in flat(model.parameters()).items()}
    trainables = set(flat(model.trainable_parameters()))
    assert "score.weight" in trainables and len(trainables) == 9
    optimizer = optim.AdamW(learning_rate=0.01)
    for _ in range(4):
        _, grad = accumulated_gradients(model, [pairs()], 2)
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state)
    after = flat(model.parameters())
    for key in before:
        if key in trainables:
            assert not mx.array_equal(before[key], after[key]).item(), key
        else:
            assert mx.array_equal(before[key], after[key]).item(), key
    state = [mx.array(v) for v in mx.random.state]
    model.train()
    one = evaluate_pairs(model, pairs(), 2, 5)
    tail = evaluate_pairs(model, pairs(), 2, 2)
    for key in one:
        assert abs(one[key] - tail[key]) < 1e-6
    assert model.training
    assert all(mx.array_equal(a, b).item() for a, b in zip(state, mx.random.state))


@pytest.fixture
def trained(tmp_path, native_stack):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    base = tmp_path / "base"
    base.mkdir()
    vocab = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "Q": 3,
        "good": 4,
        "bad": 5,
        "long": 6,
        "answer": 7,
    }
    vocab.update({f"t{i}": i for i in range(8, 32)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<eos>",
    )
    tokenizer.save_pretrained(base)
    args = ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=32,
        max_position_embeddings=256,
    )
    mx.random.seed(21)
    backbone = Model(args)
    mx.save_safetensors(str(base / "model.safetensors"), flat(backbone.parameters()))
    (base / "config.json").write_text(json.dumps(asdict(args)))
    lora = SimpleNamespace(
        r=2,
        alpha=4,
        dropout=0.0,
        use_dora=False,
        use_rslora=False,
        target_modules=["self_attn.q_proj", "self_attn.v_proj"],
    )
    training = SimpleNamespace(
        lora=lora,
        quantization="none",
        gradient_checkpointing=False,
        seed=42,
        data_seed=43,
        batch_size=2,
        gradient_accumulation_steps=2,
        epochs=2,
        lr=0.01,
        optimizer="adamw_torch",
        scheduler="constant",
        warmup_ratio=0.0,
        weight_decay=0.01,
        max_grad_norm=1.0,
        save_steps=2,
        logging_steps=10,
        rewind_log=False,
    )
    cfg = SimpleNamespace(
        base=str(base),
        output=str(tmp_path / "out"),
        data=SimpleNamespace(max_length=64),
        training=training,
    )
    from soup_cli.config.schema import SoupConfig

    cfg = SoupConfig(
        base=cfg.base,
        output=cfg.output,
        backend="mlx",
        task="reward_model",
        data={"train": "unused.jsonl", "max_length": 64},
        training={**vars(training), "lora": vars(lora)},
    )
    trainer = MLXRewardModelTrainerWrapper(cfg)
    rows = [
        {"prompt": "Q", "chosen": "good " + f"t{i}", "rejected": "bad long " + f"t{i}"}
        for i in range(8, 13)
    ]
    validation = [
        {"prompt": "Q", "chosen": f"good t{i}", "rejected": f"bad long t{i}"}
        for i in range(13, 16)
    ]
    trainer.setup({"train": rows, "val": validation})
    result = trainer.train()
    return trainer, result, Path(cfg.output), base


@pytest.mark.usefixtures("native_stack")
def test_loop_tail_save_and_fresh_reload(trained):
    trainer, result, root, base = trained
    assert result["total_steps"] == 4
    assert len(result["generations"]) == 3
    assert result["train_pairs"] == 5
    scorer = load_reward_scorer(root)
    items = [("Q", "good t8"), ("Q", "bad long t9")]
    expected = score_completions(scorer, items, batch_size=2)
    assert len(expected) == 2 and expected[0] != expected[1]
    trainer.model.eval()
    from soup_cli.utils.mlx_reward import NativeRewardScorer

    assert expected == score_completions(
        NativeRewardScorer(trainer.model, trainer.tokenizer, 64), items, batch_size=2
    )
    script = (
        "from soup_cli.utils.mlx_reward import *; import json; "
        "print(json.dumps(score_completions(load_reward_scorer("
        + repr(str(root))
        + "),"
        + repr(items)
        + ",batch_size=2)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(proc.stdout) == expected
    assert len(list((root / "bundles").iterdir())) == 3


@pytest.mark.usefixtures("native_stack")
@pytest.mark.parametrize("name", ["reward_model.json", "reward.safetensors"])
@pytest.mark.requires_symlink
def test_broken_recognition(tmp_path, name):
    (tmp_path / name).symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="scoring-only"):
        refuse_reward_artifact(tmp_path)


@pytest.mark.usefixtures("native_stack")
def test_failed_publication_preserves_previous(trained, monkeypatch):
    trainer, result, root, base = trained
    old = (root / "reward_model.json").read_bytes()
    _, manifest = read_bundle(root)
    import soup_cli.utils.mlx_reward_bundle as bundles

    monkeypatch.setattr(bundles.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("failure")))
    contract = {
        k: v for k, v in manifest.items() if k not in ("files", "format_version", "artifact_type")
    }
    with pytest.raises(OSError):
        publish_bundle(root, trainer.model, trainer.tokenizer, contract)
    assert (root / "reward_model.json").read_bytes() == old
    assert len(list((root / "bundles").iterdir())) == 3
    load_reward_scorer(root)


@pytest.mark.usefixtures("native_stack")
@pytest.mark.parametrize(
    "bundle",
    [
        "../escape",
        "/absolute",
        "bundles/../escape",
        "bundles\\abc",
        "bundles/" + "A" * 32,
        "bundles/" + "a" * 31,
    ],
)
def test_pointer_path_refusal(trained, bundle):
    _, _, root, _ = trained
    pointer = json.loads((root / "reward_model.json").read_text())
    pointer["bundle"] = bundle
    (root / "reward_model.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="bundle must"):
        load_reward_scorer(root)


@pytest.mark.usefixtures("native_stack")
def test_tokenizer_inventory_and_base_change(trained):
    _, _, root, base = trained
    payload, _ = read_bundle(root)
    (payload / "tokenizer" / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        load_reward_scorer(root)
    (payload / "tokenizer" / "extra.json").unlink()
    config = json.loads((base / "config.json").read_text())
    config["rms_norm_eps"] = 0.001
    (base / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="expected SHA-256"):
        load_reward_scorer(root)


def repoint(root, mutate):
    import hashlib

    payload, manifest = read_bundle(root)
    mutate(payload, manifest)
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    (payload / "manifest.json").write_bytes(raw)
    pointer = json.loads((root / "reward_model.json").read_text())
    pointer["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    (root / "reward_model.json").write_text(json.dumps(pointer))


@pytest.mark.usefixtures("native_stack")
@pytest.mark.parametrize(
    "fault", ["missing_head", "lost_lora", "unexpected", "shape", "dtype", "nan"]
)
def test_exact_tensor_contract(trained, fault):
    import hashlib

    _, _, root, _ = trained

    def change(payload, manifest):
        path = payload / "reward.safetensors"
        weights = mx.load(str(path))
        if fault == "missing_head":
            del weights["score.weight"]
        elif fault == "lost_lora":
            del weights[next(k for k in weights if k != "score.weight")]
        elif fault == "unexpected":
            weights["extra.weight"] = mx.zeros((1,))
        elif fault == "shape":
            weights["score.weight"] = mx.zeros((1, 33))
        elif fault == "dtype":
            weights["score.weight"] = weights["score.weight"].astype(mx.float16)
        elif fault == "nan":
            weights["score.weight"] = mx.full((1, 32), float("nan"))
        mx.save_safetensors(str(path), weights)
        raw = path.read_bytes()
        manifest["files"]["reward.safetensors"] = {
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    repoint(root, change)
    with pytest.raises(ValueError, match="MLX reward: tensor"):
        load_reward_scorer(root)


@pytest.mark.usefixtures("native_stack")
def test_relocated_base_without_tokenizer(trained, tmp_path):
    import shutil

    _, _, root, base = trained
    expected = score_completions(load_reward_scorer(root), [("Q", "good")])
    new = tmp_path / "relocated"
    new.mkdir()
    for name in ("config.json", "model.safetensors"):
        shutil.copy2(base / name, new / name)
    assert score_completions(load_reward_scorer(root, base=new), [("Q", "good")]) == expected


@pytest.mark.usefixtures("native_stack")
@pytest.mark.requires_symlink
@pytest.mark.parametrize("member", ["file", "directory"])
def test_bundle_symlinks_are_refused(trained, tmp_path, member):
    _, _, root, _ = trained
    assert load_reward_scorer(root)
    payload, _ = read_bundle(root)
    path = payload / ("reward.safetensors" if member == "file" else "tokenizer")
    moved = tmp_path / "moved-member"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=member == "directory")
    with pytest.raises((ValueError, OSError)):
        load_reward_scorer(root)


@pytest.mark.usefixtures("native_stack")
@pytest.mark.requires_hardlink
def test_bundle_hardlinks_are_refused(trained, tmp_path):
    _, _, root, _ = trained
    assert load_reward_scorer(root)
    payload, _ = read_bundle(root)
    path = payload / "reward.safetensors"
    moved = tmp_path / "moved-weight"
    path.rename(moved)
    os.link(moved, path)
    with pytest.raises((ValueError, OSError)):
        load_reward_scorer(root)



@pytest.mark.usefixtures("native_stack")
def test_base_extra_shard_and_duplicate_keys(trained):
    _, _, _, base = trained
    mx.save_safetensors(
        str(base / "model-extra.safetensors"), {"model.embed_tokens.weight": mx.zeros((32, 32))}
    )
    with pytest.raises(ValueError, match="duplicate"):
        base_identity(base)


@pytest.mark.usefixtures("native_stack")
def test_checkpoint_gradients_and_no_global_class_change():
    plain = tiny_model()
    original = type(plain.backbone.layers[0]).__call__
    checkpointed = make_reward_model(tiny_model().backbone, 42, gradient_checkpointing=True)
    loss1, g1 = accumulated_gradients(plain, [pairs()], 2)
    loss2, g2 = accumulated_gradients(checkpointed, [pairs()], 2)
    assert abs(loss1 - loss2) < 1e-6
    for key, value in flat(g1).items():
        assert mx.allclose(value, flat(g2)[key], atol=1e-6).item(), key
    assert type(plain.backbone.layers[0]).__call__ is original


@pytest.mark.usefixtures("native_stack")
def test_save_preserves_rng_even_when_validating_previous_bundle(trained):
    trainer, _, root, _ = trained
    _, manifest = read_bundle(root)
    contract = {
        k: v for k, v in manifest.items() if k not in ("files", "format_version", "artifact_type")
    }
    mx.random.seed(999)
    before = [mx.array(v) for v in mx.random.state]
    publish_bundle(root, trainer.model, trainer.tokenizer, contract)
    assert all(mx.array_equal(a, b).item() for a, b in zip(before, mx.random.state))


@pytest.mark.usefixtures("native_stack")
def test_missing_base_member_names_remedy(trained):
    _, _, root, base = trained
    (base / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="base member 'model.safetensors' is unavailable"):
        load_reward_scorer(root)


@pytest.mark.usefixtures("native_stack")
def test_diagnostic_labels_escape_markup_controls_and_bound():
    from rich.text import Text

    from soup_cli.utils.mlx_reward_bundle import label

    value = label("[bold]\x1b" + "x" * 500)
    shown = Text.from_markup(value)
    assert "[bold]" in shown.plain and not shown.spans
    assert "\x1b" not in value and len(value) <= 260


@pytest.fixture
def native_stack():
    """No model downloads; native tests alone skip on non-MLX hosts."""
    import importlib

    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    llama = importlib.import_module("mlx_lm.models.llama")
    globals().update(
        mx=mx,
        nn=importlib.import_module("mlx.nn"),
        optim=importlib.import_module("mlx.optimizers"),
        tree_flatten=importlib.import_module("mlx.utils").tree_flatten,
        Model=llama.Model,
        ModelArgs=llama.ModelArgs,
        linear_to_lora_layers=importlib.import_module("mlx_lm.tuner.utils").linear_to_lora_layers,
    )


mx = nn = optim = tree_flatten = Model = ModelArgs = linear_to_lora_layers = None


class Tokenizer:
    eos_token_id = pad_token_id = 2
    is_fast = True

    def __call__(self, text, **kwargs):
        import re

        words = list(re.finditer(r"\S+", text))
        return {
            "input_ids": [1] + [3 + len(m.group()) for m in words],
            "offset_mapping": [(0, 0)] + [(m.start(), m.end()) for m in words],
        }


def test_validation_before_forward():
    scorer = NativeRewardScorer(None, Tokenizer(), 4)
    with pytest.raises(ValueError, match="batch_size"):
        score_completions(scorer, [], batch_size=True)
    assert score_completions(scorer, []) == []
    with pytest.raises(ValueError, match="item 1.*exceed"):
        score_completions(scorer, [("Q", "a"), ("Q", "one two three")])


def test_offsets_bos_eos_and_truncation():
    tokens, lost, count = encode_completion(Tokenizer(), "Q:", " x ", 4)
    assert tokens == [1, 5, 4, 2] and not lost and count == 4
    assert encode_completion(Tokenizer(), "Q:", " x ", 3)[1]


def test_malformed_rejected_cannot_hide_behind_lost_chosen():
    with pytest.raises(ValueError, match="row 0 rejected: blank"):
        prepare_pairs(
            [{"prompt": "a long prompt", "chosen": "answer", "rejected": " "}],
            Tokenizer(),
            3,
            "train",
        )


def test_pair_filter_retains_original_index():
    rows = [
        {"prompt": "", "chosen": "same", "rejected": "word"},
        {"prompt": "", "chosen": "a", "rejected": "longer"},
    ]
    pairs, stats = prepare_pairs(rows, Tokenizer(), 4, "train")
    assert [p.index for p in pairs] == [1]
    assert stats["skipped_collapsed"] == 1
    assert "row 0" in stats["first_skip"]


def test_planner_tail_and_private_rng():
    import random

    before = random.getstate()
    epochs = plan_windows(5, 2, 2, 2, 42)
    assert len(epochs) == 2
    for windows in epochs:
        assert [len(w) for w in windows] == [2, 1]
        assert sorted(i for w in windows for batch in w for i in batch) == list(range(5))
    assert random.getstate() == before


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2", None])
def test_invalid_batch_sizes_even_empty(value):
    with pytest.raises(ValueError, match="batch_size"):
        score_completions(NativeRewardScorer(None, Tokenizer(), 64), [], batch_size=value)


@pytest.mark.parametrize(
    "items,pattern",
    [
        ("text", "items must"),
        ([("x",)], "item 0: expected"),
        ([(1, "answer")], "item 0 prompt: expected"),
        ([("Q", None)], "item 0 completion: expected"),
        ([("Q", " \t ")], "item 0 completion: blank"),
    ],
)
def test_invalid_scoring_items(items, pattern):
    with pytest.raises(ValueError, match=pattern):
        score_completions(NativeRewardScorer(None, Tokenizer(), 64), items)


@pytest.mark.parametrize(
    "row,pattern",
    [
        ([], "expected a prompt/chosen/rejected"),
        ({"chosen": "a"}, "rejected: required"),
        ({"chosen": "a", "rejected": 1}, "rejected: expected"),
        ({"prompt": None, "chosen": "a", "rejected": "b"}, "prompt: expected"),
    ],
)
def test_malformed_rows_stop_with_index(row, pattern):
    with pytest.raises(ValueError, match=pattern):
        prepare_pairs([row], Tokenizer(), 64, "train")


def test_separator_overlap_and_zero_offset_bos():
    class OffsetTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            assert text == "Q: x"
            return {"input_ids": [1, 5, 8, 2], "offset_mapping": [(0, 0), (0, 2), (2, 4), (0, 0)]}

    assert encode_completion(OffsetTokenizer(), "Q:", "x", 4) == ([1, 5, 8, 2], False, 4)


def test_eos_only_rejected_not_hidden_by_lost_chosen():
    class SpecialTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            if text.endswith("<eos>"):
                return {"input_ids": [1, 2], "offset_mapping": [(0, 0), (0, len(text))]}
            return super().__call__(text, **kwargs)

    with pytest.raises(ValueError, match="rejected: no non-EOS content"):
        prepare_pairs(
            [{"prompt": "long long long", "chosen": "answer", "rejected": "<eos>"}],
            SpecialTokenizer(),
            3,
            "train",
        )


def test_filtered_validation_is_not_reported_as_zero_loss():
    with pytest.raises(ValueError, match="reward_model val: no usable"):
        prepare_pairs([{"chosen": "same", "rejected": "word"}], Tokenizer(), 64, "val")


def reward_config(**changes):
    from soup_cli.config.schema import SoupConfig

    values = {
        "base": "local/base",
        "backend": "mlx",
        "task": "reward_model",
        "data": {"train": "pairs.jsonl"},
        "training": {"quantization": "none", "rewind_log": False},
    }
    values.update(changes)
    if "training" in changes:
        values["training"] = {"quantization": "none", "rewind_log": False, **changes["training"]}
    return SoupConfig(**values)


def test_documented_reward_configuration_loads_and_dispatches():
    import yaml

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.mlx_routing import resolve_trainer

    page = (Path(__file__).parents[1] / "docs/mlx-reward-model.md").read_text()
    blocks = re.findall(r"```yaml\n(.*?)\n```", page, re.DOTALL)
    assert blocks
    config = SoupConfig(**yaml.safe_load(blocks[0]))
    assert (config.backend, config.task) == ("mlx", "reward_model")
    assert resolve_trainer(config)[0] is MLXRewardModelTrainerWrapper


def test_omitted_reward_rewind_is_disabled_and_config_remains_reloadable():
    from soup_cli.config.schema import SoupConfig, TrainingConfig

    config = SoupConfig(
        base="local", backend="mlx", task="reward_model", data={"train": "pairs.jsonl"},
        training={"quantization": "none"},
    )
    assert config.training.rewind_log is False
    assert SoupConfig(**config.model_dump()).model_dump() == config.model_dump()
    assert TrainingConfig().rewind_log is True
    assert reward_config(training={"rewind_log": False}).training.rewind_log is False
    with pytest.raises(ValueError, match="training.rewind_log"):
        reward_config(training={"rewind_log": True})


@pytest.mark.parametrize("value", ["4bit", "8bit"])
def test_quantization_diagnostic_names_the_actual_field_and_value(value):
    with pytest.raises(ValueError) as error:
        reward_config(training={"quantization": value})
    assert error.value.errors()[0]["msg"] == (
        f"Value error, MLX reward_model does not support training.quantization='{value}'. "
        "Set training.quantization to none for a dense base."
    )


@pytest.mark.usefixtures("native_stack")
def test_publish_does_not_hash_or_reload_the_base(trained, monkeypatch):
    import mlx_lm.utils

    import soup_cli.utils.mlx_reward_bundle as bundles

    trainer, _, root, _ = trained
    _, manifest = read_bundle(root)
    contract = {key: value for key, value in manifest.items()
                if key not in ("files", "format_version", "artifact_type")}

    def refuse(*args, **kwargs):
        raise AssertionError("saving rehashed or reloaded the base")

    monkeypatch.setattr(bundles, "verify_base", refuse)
    monkeypatch.setattr(bundles, "base_identity", refuse)
    monkeypatch.setattr(mlx_lm.utils, "load_model", refuse)
    before = (root / "reward_model.json").read_bytes()
    publish_bundle(root, trainer.model, trainer.tokenizer, contract)
    assert (root / "reward_model.json").read_bytes() != before
    payload, _ = read_bundle(root)
    saved = (root / "reward_model.json").read_bytes()
    (payload / "reward.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size or SHA-256 mismatch"):
        publish_bundle(root, trainer.model, trainer.tokenizer, contract)
    assert (root / "reward_model.json").read_bytes() == saved


@pytest.mark.usefixtures("native_stack")
def test_hub_download_only_selects_native_base_and_tokenizer_assets(trained, monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import filter_repo_objects

    from soup_cli.config.schema import SoupConfig

    trainer, _, root, base = trained
    calls = []

    def download(model, **kwargs):
        calls.append((model, kwargs))
        return str(base)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    values = trainer.config.model_dump()
    values.update(base="example/native-reward-base", output=str(root) + "-hub")
    wrapper = MLXRewardModelTrainerWrapper(SoupConfig(**values))
    wrapper.setup({"train": [{"prompt": "Q", "chosen": "good", "rejected": "bad"}]})
    assert len(calls) == 1 and calls[0][0] == values["base"]
    options = calls[0][1]
    allowed = {
        "config.json", "model.safetensors", "model-00001-of-00002.safetensors",
        "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "tokenizer.model", "tiktoken.model",
        "vocab.txt", "vocab.json", "merges.txt", "chat_template.jinja",
        "chat_templates/default.jinja", "tokenizer.tiktoken", "hf_quant_config.json",
    }
    refused = {"pytorch_model.bin", "model.bin", "model.npz", "model.py", "data.json",
               "data.jsonl", "other/model.safetensors", "model-dir/model.safetensors", "README.md"}
    selected = set(filter_repo_objects(
        sorted(allowed | refused), allow_patterns=options.get("allow_patterns"),
        ignore_patterns=options.get("ignore_patterns"),
    ))
    assert selected == allowed


@pytest.mark.usefixtures("native_stack")
@pytest.mark.parametrize("member", ["pytorch_model.bin", "model.npz", "hf_quant_config.json"])
def test_unsupported_base_member_is_named_and_not_deleted(trained, member):
    _, _, _, base = trained
    assert base_identity(base)["files"]
    path = base / member
    path.write_bytes(b"unused cached member")
    with pytest.raises(ValueError, match=re.escape(member)):
        base_identity(base)
    assert path.read_bytes() == b"unused cached member"


@pytest.mark.usefixtures("native_stack")
def test_unseen_pairs_improve_over_seeded_initial_head_and_zero_lora(trained):
    from soup_cli.config.schema import SoupConfig

    trainer, _, root, _ = trained
    train_tokens = {tuple(pair.chosen) for pair in trainer._pairs}
    train_tokens |= {tuple(pair.rejected) for pair in trainer._pairs}
    held_tokens = {tuple(pair.chosen) for pair in trainer._val_pairs}
    held_tokens |= {tuple(pair.rejected) for pair in trainer._val_pairs}
    assert len(trainer._val_pairs) == 3 and train_tokens.isdisjoint(held_tokens)
    held = [{"prompt": "Q", "chosen": f"good t{i}", "rejected": f"bad long t{i}"}
            for i in range(13, 16)]
    for index in range(13, 16):
        assert trainer.tokenizer.encode(f"t{index}", add_special_tokens=False) == [index]
    values = trainer.config.model_dump()
    values["output"] += "-initial"
    initial = MLXRewardModelTrainerWrapper(SoupConfig(**values))
    initial.setup({"train": held})
    lora_b = [value for key, value in flat(initial.model.trainable_parameters()).items()
              if key.endswith("lora_b")]
    assert lora_b and all(bool(mx.all(value == 0).item()) for value in lora_b)
    items = [(row["prompt"], row[side]) for row in held for side in ("chosen", "rejected")]
    before = score_completions(NativeRewardScorer(initial.model, initial.tokenizer, 64), items)
    after = score_completions(load_reward_scorer(root), items)
    initial_margins = [before[i] - before[i + 1] for i in range(0, len(items), 2)]
    trained_margins = [after[i] - after[i + 1] for i in range(0, len(items), 2)]
    assert all(margin > 0 for margin in trained_margins)
    # Both supported stacks improved by >5.8; retain a wide deterministic margin.
    assert sum(trained_margins) / 3 - sum(initial_margins) / 3 > 1.0


def test_public_registry_and_neighbor_routes():
    from soup_cli.config.schema import MLX_TASKS
    from soup_cli.trainer.mlx_routing import MLX_TRAINER_REGISTRY, resolve_trainer

    config = reward_config()
    assert "reward_model" in MLX_TASKS
    assert "reward_model" in MLX_TRAINER_REGISTRY
    assert "reward_model" in MLX_TRAINER_REGISTRY.keys()
    assert resolve_trainer(config)[0] is MLXRewardModelTrainerWrapper
    assert resolve_trainer(reward_config(backend="transformers"))[0] is None
    assert resolve_trainer(reward_config(task="sft"))[0].__name__ == "MLXSFTTrainerWrapper"


@pytest.mark.parametrize(
    "training,field",
    [
        ({"quantization": "4bit"}, "training.quantization"),
        ({"quantization": "8bit"}, "training.quantization"),
        ({"quantize_reward_model": True}, "quantize_reward_model"),
        ({"use_galore": True}, "use_galore"),
        ({"use_ring_attention": True}, "use_ring_attention"),
        ({"rewind_log": True}, "rewind_log"),
        ({"gradient_checkpointing": "unsloth"}, "gradient_checkpointing"),
        ({"packing": True}, "packing"),
        ({"lora": {"r": 0}}, "lora.r"),
        ({"lora": {"r": 4097}}, "lora.r"),
        ({"lora": {"target_modules": ["lm_head"]}}, "target_modules"),
        ({"lora": {"use_dora": True}}, "use_dora"),
        ({"lora": {"init_strategy": "pissa"}}, "pissa"),
        ({"eval_steps": 2}, "eval_steps"),
        ({"lora": {"rank_pattern": {"q_proj": 2}}}, "rank_pattern"),
        ({"loraplus_lr_ratio": 2}, "loraplus_lr_ratio"),
    ],
)
def test_unapplied_or_out_of_scope_training_refused(training, field):
    with pytest.raises(ValueError, match=field):
        reward_config(training=training)


@pytest.mark.parametrize(
    "data,field",
    [
        ({"tokenized_path": "cache"}, "tokenized_path"),
        ({"format": "pre_tokenized", "tokenized_path": "cache"}, "tokenized_path"),
        ({"format": "alpaca"}, "data.format"),
        ({"streaming": True}, "streaming"),
        ({"chat_template": "chatml"}, "chat_template"),
        ({"add_new_tokens": ["<new>"]}, "add_new_tokens"),
        ({"mask_history": True}, "mask_history"),
        ({"eval_on_each_dataset": True}, "eval_on_each_dataset"),
        ({"train_on_responses_only": False}, "train_on_responses_only"),
        ({"skip_prepare_dataset": True}, "skip_prepare_dataset"),
    ],
)
def test_reward_raw_data_scope(data, field):
    with pytest.raises(ValueError, match=field):
        reward_config(data={"train": "pairs.jsonl", **data})


def test_eval_policy_and_rslora_dropout_decision():
    with pytest.raises(ValueError, match="training.eval_gate.enabled"):
        reward_config(training={"eval_gate": {"enabled": True, "suite": "suite.yaml"}})
    with pytest.raises(ValueError, match="eval.auto_eval"):
        reward_config(eval={"auto_eval": True})
    config = reward_config(training={"lora": {"use_rslora": True, "dropout": 0.25}})
    assert config.training.lora.use_rslora and config.training.lora.dropout == 0.25


@pytest.mark.usefixtures("native_stack")
def test_dropout_and_rslora_are_applied_like_sft(trained):
    trainer, _, _, _ = trained
    values = trainer.config.model_dump()
    values["training"]["lora"].update(use_rslora=True, dropout=0.25)
    values["output"] += "-rslora"
    from soup_cli.config.schema import SoupConfig

    wrapper = MLXRewardModelTrainerWrapper(SoupConfig(**values))
    wrapper.setup({"train": [{"prompt": "Q", "chosen": "good", "rejected": "bad"}]})
    layer = wrapper.model.backbone.layers[0].self_attn.q_proj
    assert abs(layer.scale - 4 / (2**0.5)) < 1e-6
    layer.lora_b = mx.ones_like(layer.lora_b)
    features = mx.ones((16, 32))
    layer.train()
    first = layer(features)
    second = layer(features)
    mx.eval(first, second)
    assert not mx.array_equal(first, second).item()
    layer.eval()
    assert mx.array_equal(layer(features), layer(features)).item()
    wrapper.train()
    wrapper.model.eval()
    restored = load_reward_scorer(values["output"])
    assert restored.model.backbone.layers[0].self_attn.q_proj.scale == layer.scale
    assert score_completions(restored, [("Q", "good")]) == score_completions(
        NativeRewardScorer(wrapper.model, wrapper.tokenizer, 64), [("Q", "good")]
    )


def public_cli(config_path, *args):
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    environment.setdefault("COLUMNS", "240")
    return subprocess.run(
        [sys.executable, "-m", "soup_cli", "train", "--config", str(config_path), "--yes", *args],
        cwd=config_path.parent,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--push-as", "org/model"], "MLX reward_model does not support --push-as."),
        (["--gate", "missing.yaml"], "MLX reward_model does not support --gate."),
        (["--diagnose-gate", "missing.yaml"],
         "MLX reward_model does not support --diagnose-gate."),
        (["--capture-activations", "activations"],
         "MLX reward_model does not support --capture-activations."),
        (["--resume", "missing", "--push-as", "org/model"],
         "--resume is not supported for task 'reward_model' on backend 'mlx'"),
        (["--hf-resume", "--push-as", "org/model"],
         "--hf-resume is not supported for task 'reward_model' on backend 'mlx'"),
    ],
)
def test_public_cli_early_refusals(tmp_path, args, expected):
    import yaml

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(reward_config().model_dump(mode="json")))
    result = public_cli(config, *args)
    assert result.returncode == 1, result.stdout + result.stderr
    from tests.conftest import strip_ansi

    assert expected in " ".join(strip_ansi(result.stdout).split()), result.stdout + result.stderr
    assert "Setting up model" not in result.stdout
    assert not (tmp_path / ".soup").exists()


@pytest.mark.usefixtures("native_stack")
def test_public_cli_and_real_sweep_dispatch(trained, tmp_path, monkeypatch):
    import yaml

    from soup_cli.commands.sweep import _run_single
    from soup_cli.config.schema import SoupConfig

    trainer, _, _, _ = trained
    rows = [{"prompt": "Q", "chosen": f"good t{i}", "rejected": f"bad t{i}"}
            for i in range(8, 13)]
    train = tmp_path / "pairs.jsonl"
    train.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    values = trainer.config.model_dump(mode="json")
    values["data"].update(train=str(train), val_split=0.4)
    values["output"] = str(tmp_path / "cli-output")
    values["training"]["epochs"] = 1
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(values))
    result = public_cli(config)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "load_reward_scorer" in result.stdout
    assert "soup runs show" in result.stdout
    for forbidden in ("soup chat --model", "soup push --model", "soup merge --adapter",
                      "soup export --model"):
        assert forbidden not in result.stdout
    root = Path(values["output"])
    _, manifest = read_bundle(root)
    assert manifest["training"]["completed_updates"] == 1
    assert manifest["training"]["train_counts"]["retained"] == 3
    assert manifest["training"]["val_counts"]["retained"] == 2
    scored = score_completions(load_reward_scorer(root), [("Q", "good t8")])
    assert len(scored) == 1
    values["output"] = str(tmp_path / "sweep-output")
    monkeypatch.chdir(tmp_path)
    # Real sweep execution: config validation, loader, dispatch, setup, update and save.
    swept = _run_single(SoupConfig(**values), {"lr": 0.005}, "reward-test", config)
    assert swept["total_steps"] == 1
    assert swept["train_pairs"] == 3
    assert len(swept["evaluations"]) == 1
    assert len(score_completions(load_reward_scorer(values["output"]), [("Q", "good t8")])) == 1


@pytest.mark.usefixtures("native_stack")
def test_setup_seed_reproducibility_and_independent_data_seed(trained):
    from soup_cli.config.schema import SoupConfig

    trainer, _, _, _ = trained
    snapshots = []
    for index, (seed, data_seed) in enumerate(((None, 43), (42, 43), (7, 43), (42, 99))):
        values = trainer.config.model_dump()
        values["training"].update(seed=seed, data_seed=data_seed)
        values["output"] += f"-seed-{index}"
        mx.random.seed(1000 + index)  # Different ambient state must not leak into setup.
        wrapper = MLXRewardModelTrainerWrapper(SoupConfig(**values))
        wrapper.setup({"train": [{"prompt": "Q", "chosen": "good", "rejected": "bad"}]})
        weights = {key: mx.array(value) for key, value in flat(
            wrapper.model.trainable_parameters()
        ).items()}
        mx.eval(weights)
        snapshots.append(weights)
    for key, value in snapshots[0].items():
        assert mx.array_equal(value, snapshots[1][key]).item(), key
        assert mx.array_equal(value, snapshots[3][key]).item(), key
    for suffix in ("lora_a", "score.weight"):
        assert any(not mx.array_equal(value, snapshots[2][key]).item()
                   for key, value in snapshots[0].items() if key.endswith(suffix))
    expected = mx.random.normal((1, 32), dtype=mx.float32, key=mx.random.key(42)) * 0.02
    assert mx.array_equal(snapshots[0]["score.weight"], expected).item()


@pytest.mark.usefixtures("native_stack")
@pytest.mark.parametrize(
    "fault", ["duplicate_json", "nonfinite_json", "oversize_header", "custom_code"],
)
def test_bundle_corruption_reaches_the_specific_check(trained, monkeypatch, fault):
    import hashlib

    import mlx_lm.utils

    import soup_cli.utils.mlx_reward_bundle as bundles
    _, _, root, base = trained
    # A valid control is proved before planting exactly one fault.
    assert score_completions(load_reward_scorer(root), [("Q", "good t13")])
    payload, manifest = read_bundle(root)
    pointer_path = root / 'reward_model.json'
    pointer = json.loads(pointer_path.read_text())
    expected = 'invalid bounded JSON'
    if fault == 'duplicate_json':
        raw = json.dumps(pointer)
        assert raw.count('"format_version": 1') == 1
        pointer_path.write_text(raw.replace('"format_version": 1',
                                           '"format_version": 1, "format_version": 1'))
    else:
        if fault == 'nonfinite_json':
            manifest['training']['plan_probe'] = 1.0
        elif fault == 'oversize_header':
            path = payload / 'reward.safetensors'
            data = path.read_bytes()
            old_length = int.from_bytes(data[:8], 'little')
            new_length = bundles.MAX_JSON + 8
            path.write_bytes(new_length.to_bytes(8, 'little') +
                             data[8:8+old_length].ljust(new_length, b' ') +
                             data[8+old_length:])
            raw = path.read_bytes()
            manifest['files']['reward.safetensors'] = {
                'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            expected = 'invalid safetensors header length'
        else:
            path = base / 'config.json'
            config = json.loads(path.read_text())
            config['model_file'] = 'unused.py'
            path.write_text(json.dumps(config))
            raw = path.read_bytes()
            manifest['base']['files']['config.json'] = {
                'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            def refuse(*args, **kwargs):
                raise AssertionError('the custom-model route reached the model loader')
            monkeypatch.setattr(mlx_lm.utils, 'load_model', refuse)
            expected = 'dense Llama backbone required'
        raw = json.dumps(manifest, sort_keys=True).encode()
        if fault == 'nonfinite_json':
            assert raw.count(b'"plan_probe": 1.0') == 1
            raw = raw.replace(b'"plan_probe": 1.0', b'"plan_probe": 1e999')
        (payload / 'manifest.json').write_bytes(raw)
        pointer['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
        pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match=expected):
        load_reward_scorer(root)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}',
                                 b'{"x":-Infinity}', b'{"x":[1e999]}', b'[]'])
def test_json_parser_checks_complete_objects(raw):
    from soup_cli.utils.mlx_reward_bundle import _json

    assert _json(b'{"x":[1,2.5,{"y":true}]}') == {"x": [1, 2.5, {"y": True}]}
    with pytest.raises(ValueError, match="duplicate|nonfinite|expected JSON object"):
        _json(raw)


def test_json_byte_bound_is_checked_before_parsing(tmp_path):
    from soup_cli.utils.mlx_reward_bundle import MAX_JSON, read_json

    path = tmp_path / "bounded.json"
    path.write_bytes(b'{"x":1}'.ljust(MAX_JSON, b" "))
    assert read_json(path) == {"x": 1}
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bounded JSON"):
        read_json(path)


@pytest.mark.parametrize("version", [True, False, 1.0, "1", 2, None])
def test_version_requires_the_exact_integer_type(version):
    from soup_cli.utils.mlx_reward_bundle import ARTIFACT, _version

    _version({"format_version": 1, "artifact_type": ARTIFACT}, "pointer")
    with pytest.raises(ValueError, match="format_version: expected 1"):
        _version({"format_version": version, "artifact_type": ARTIFACT}, "pointer")


def _tensor_file(path, header, body=b"\0\0\0\0"):
    raw = json.dumps(header).encode()
    raw = raw.ljust((len(raw) + 7) // 8 * 8, b" ")
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + body)


@pytest.mark.parametrize("fault", ["dtype", "shape_bool", "shape_negative", "offset_bool",
                                   "offset_negative", "bytes", "overlap", "gap", "size"])
def test_tensor_header_layout_checks_have_valid_controls(tmp_path, fault):
    from soup_cli.utils.mlx_reward_bundle import tensor_header

    path = tmp_path / "weights.safetensors"
    meta = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
    _tensor_file(path, {"weight": meta})
    with path.open("rb") as stream:
        assert tensor_header(stream, path.name)["weight"] == meta
    body = b"\0" * 4
    header = {"weight": meta}
    if fault == "dtype":
        meta["dtype"] = "UNKNOWN"
    elif fault == "shape_bool":
        meta["shape"] = [True]
    elif fault == "shape_negative":
        meta["shape"] = [-1]
    elif fault == "offset_bool":
        meta["data_offsets"] = [False, 4]
    elif fault == "offset_negative":
        meta["data_offsets"] = [-1, 3]
    elif fault == "bytes":
        meta["data_offsets"] = [0, 8]
        body = b"\0" * 8
    elif fault == "overlap":
        header["other"] = dict(meta)
    elif fault == "gap":
        meta["data_offsets"] = [4, 8]
        body = b"\0" * 8
    else:
        body = b""
    _tensor_file(path, header, body)
    expected = "invalid tensor shape, dtype or byte range"
    if fault in ("overlap", "gap"):
        expected = "overlapping or non-contiguous tensor ranges"
    elif fault == "size":
        expected = "tensor byte ranges differ from file size"
    with path.open("rb") as stream, pytest.raises(ValueError, match=expected):
        tensor_header(stream, path.name)


@pytest.mark.requires_symlink
def test_json_read_refuses_linked_parent_and_contained_member(tmp_path):
    from soup_cli.utils.mlx_reward_bundle import contained, read_json

    target = tmp_path / "target"
    target.mkdir()
    (target / "data.json").write_text('{"x":1}')
    assert read_json(target / "data.json") == {"x": 1}
    assert contained(tmp_path, "target/data.json") == target / "data.json"
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="unreadable|linked"):
        read_json(linked / "data.json")
    with pytest.raises(ValueError, match="linked parent"):
        contained(tmp_path, "linked/data.json")


@pytest.mark.requires_hardlink
def test_json_read_refuses_hardlinks(tmp_path):
    from soup_cli.utils.mlx_reward_bundle import read_json

    path = tmp_path / "data.json"
    path.write_text('{"x":1}')
    assert read_json(path) == {"x": 1}
    os.link(path, tmp_path / "other.json")
    with pytest.raises(ValueError, match="unreadable|linked"):
        read_json(path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_checked_file_refuses_nonregular_files_without_reading(tmp_path):
    from soup_cli.utils.mlx_reward_bundle import checked_file

    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular non-linked file"), checked_file(path):
        pytest.fail("nonregular descriptor was accepted")


@pytest.mark.requires_symlink
@pytest.mark.parametrize("parent", [False, True])
def test_output_rejects_linked_root_or_parent_before_writing(tmp_path, parent):
    from soup_cli.utils.mlx_reward_bundle import validate_output

    target = tmp_path / "target"
    target.mkdir()
    validate_output(target)
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError, match="symbolic link|reparse"):
        validate_output(linked / "new" if parent else linked)
    assert list(target.iterdir()) == []


@pytest.mark.usefixtures("native_stack")
def test_weights_use_the_verified_descriptor_after_path_replacement(trained, monkeypatch):
    import soup_cli.utils.mlx_reward_bundle as bundles

    trainer, _, root, _ = trained
    payload, manifest = read_bundle(root)
    path = payload / "reward.safetensors"
    expected = flat(trainer.model.trainable_parameters())
    replacement = dict(expected)
    replacement["score.weight"] = expected["score.weight"] + 1
    changed = payload / "replacement.safetensors"
    mx.save_safetensors(str(changed), replacement)
    original = bundles.tensor_header

    def swap(stream, member):
        header = original(stream, member)
        os.replace(changed, path)
        return header

    monkeypatch.setattr(bundles, "tensor_header", swap)
    loaded = bundles.read_weights(path, expected, manifest["files"]["reward.safetensors"])
    assert not changed.exists()
    for key, value in expected.items():
        assert mx.array_equal(loaded[key], value).item(), key


@pytest.mark.usefixtures("native_stack")
@pytest.mark.parametrize("member", ["config.json", "tokenizer_config.json"])
@pytest.mark.parametrize("key", ["model_file", "auto_map"])
def test_saved_tokenizer_metadata_refuses_custom_routes_before_loading(
    trained, monkeypatch, member, key,
):
    from transformers import AutoTokenizer

    _, _, root, _ = trained
    assert load_reward_scorer(root)

    def change(payload, manifest):
        import hashlib

        path = payload / "tokenizer" / member
        config = json.loads(path.read_text()) if path.exists() else {"model_type": "llama"}
        config[key] = "unsupported"
        raw = json.dumps(config).encode()
        path.write_bytes(raw)
        manifest["files"][f"tokenizer/{member}"] = {
            "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        }

    repoint(root, change)

    def refuse(*args, **kwargs):
        raise AssertionError("custom metadata reached AutoTokenizer")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", refuse)
    with pytest.raises(ValueError, match="custom code is not supported"):
        load_reward_scorer(root)


@pytest.mark.parametrize('field,value,pattern', [
    ('max_length', True, 'max_length'), ('max_length', 1, 'max_length'),
    ('training', None, 'training metadata'), ('head.seed', True, 'head initialization'),
    ('lora.rank', True, 'LoRA metadata'), ('lora.rank', 4097, 'LoRA metadata'),
    ('lora.layers', 65537, 'LoRA metadata'), ('lora.keys', ['x' * 257], 'LoRA metadata'),
    ('lora.keys', [f'p{i}' for i in range(1025)], 'LoRA metadata'),
    ('lora.scale', False, 'LoRA scale'),
    pytest.param('lora.scale', 10 ** 4000, 'LoRA scale', id='overflowing-scale'),
    ('lora.dropout', 1.0, 'LoRA scale or dropout'),
    ('tokenizer.eos_id', True, 'tokenizer identity'),
    ('tokenizer.vocab_size', 2, 'tokenizer identity'),
])
def test_manifest_contract_bounds_are_independent(field, value, pattern):
    from soup_cli.utils.mlx_reward_bundle import POLICY, validate_contract

    manifest = dict(base={'path':'local', 'files':{
        'config.json':{'size':1, 'sha256':'a' * 64}}}, training={}, max_length=64,
        policy=dict(POLICY), head=dict(seed=42, std=0.02, dtype='float32', bias=False),
        lora=dict(rank=2, scale=2., dropout=0., keys=['self_attn.q_proj'], layers=2),
        tokenizer=dict(eos_id=2, vocab_size=32))
    validate_contract(manifest)
    if '.' in field:
        namespace, name = field.split('.')
        manifest[namespace][name] = value
    else:
        manifest[field] = value
    with pytest.raises(ValueError, match=pattern):
        validate_contract(manifest)


def test_public_cli_preserves_requested_terminal_width(tmp_path, monkeypatch):
    observed = {}
    monkeypatch.setenv('COLUMNS', '80')
    monkeypatch.setenv('FORCE_COLOR', '1')

    def run(*args, **kwargs):
        observed.update(kwargs['env'])

    monkeypatch.setattr(subprocess, 'run', run)
    public_cli(tmp_path / 'config.yaml', '--gate', 'missing.yaml')
    assert observed['COLUMNS'] == '80' and observed['FORCE_COLOR'] == '1'


@pytest.mark.usefixtures("native_stack")
@pytest.mark.parametrize("member", ["config.json", "tokenizer_config.json"])
@pytest.mark.parametrize("key", ["model_file", "auto_map"])
def test_base_metadata_checks_precede_identity_or_model_loading(trained, monkeypatch,
                                                              member, key):
    import hashlib

    import mlx_lm.utils

    _, _, root, base = trained
    assert load_reward_scorer(root)

    def change(payload, manifest):
        path = base / member
        config = json.loads(path.read_text())
        config[key] = "unsupported"
        raw = json.dumps(config).encode()
        path.write_bytes(raw)
        if member in manifest["base"]["files"]:
            manifest["base"]["files"][member] = {
                "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            }

    repoint(root, change)

    def refuse(*args, **kwargs):
        raise AssertionError("unsupported metadata reached the model loader")

    monkeypatch.setattr(mlx_lm.utils, "load_model", refuse)
    expected = "dense Llama backbone required" if member == "config.json" else "custom code"
    with pytest.raises(ValueError, match=expected):
        load_reward_scorer(root)


@pytest.mark.usefixtures("native_stack")
def test_failed_payload_rename_cannot_publish_a_pointer(trained, monkeypatch):
    trainer, _, root, _ = trained
    _, manifest = read_bundle(root)
    contract = {key: value for key, value in manifest.items()
                if key not in ("files", "format_version", "artifact_type")}
    before = (root / "reward_model.json").read_bytes()
    generations = set((root / "bundles").iterdir())
    original = Path.rename

    def fail_stage(path, target):
        if path.name.startswith('.staging-'):
            raise OSError('injected payload rename failure')
        return original(path, target)

    monkeypatch.setattr(Path, 'rename', fail_stage)
    with pytest.raises(OSError, match='injected payload rename'):
        publish_bundle(root, trainer.model, trainer.tokenizer, contract)
    assert (root / 'reward_model.json').read_bytes() == before
    assert set((root / 'bundles').iterdir()) == generations
    assert score_completions(load_reward_scorer(root), [('Q', 'good')])


@pytest.mark.requires_symlink
@pytest.mark.parametrize("entry", ["output", "contained", "json"])
def test_directory_checks_refuse_a_link_before_a_parent_component(tmp_path, entry):
    from soup_cli.utils.mlx_reward_bundle import contained, read_json, validate_output

    work, outside = tmp_path / "work", tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    (outside / "child").mkdir()
    (outside / "bundle").mkdir()
    (outside / "bundle" / "data.json").write_text('{"outside":true}')
    (work / "child").mkdir()
    (work / "bundle").mkdir()
    (work / "bundle" / "data.json").write_text('{"outside":false}')
    ordinary = work / "child" / ".." / "bundle"
    validate_output(ordinary)
    assert read_json(contained(ordinary, "data.json")) == {"outside": False}
    (work / "linked").symlink_to(outside / "child", target_is_directory=True)
    linked = work / "linked" / ".." / "bundle"
    operations = {
        "output": lambda: validate_output(linked),
        "contained": lambda: contained(linked, "data.json"),
        "json": lambda: read_json(linked / "data.json"),
    }
    with pytest.raises((OSError, ValueError), match="link|unreadable"):
        operations[entry]()
    assert list((outside / "bundle").iterdir()) == [outside / "bundle" / "data.json"]


@pytest.mark.usefixtures("native_stack")
@pytest.mark.requires_symlink
def test_publication_refuses_a_link_before_parent_without_side_effects(trained, tmp_path):
    trainer, _, root, _ = trained
    _, manifest = read_bundle(root)
    contract = {key: value for key, value in manifest.items()
                if key not in ("files", "format_version", "artifact_type")}
    work, outside = tmp_path / "work", tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    (outside / "child").mkdir()
    (work / "linked").symlink_to(outside / "child", target_is_directory=True)
    output = work / "linked" / ".." / "output"
    with pytest.raises((OSError, ValueError), match="link|unreadable"):
        publish_bundle(output, trainer.model, trainer.tokenizer, contract)
    assert not (outside / "output").exists()


@pytest.mark.usefixtures("native_stack")
@pytest.mark.parametrize("member", ["config.json", "model.safetensors"])
def test_base_loader_uses_checked_contents_when_original_paths_change(trained, monkeypatch, member):
    import hashlib

    import mlx_lm.utils

    _, _, root, base = trained
    control = score_completions(load_reward_scorer(root), [("Q", "good")])
    _, manifest = read_bundle(root)
    original = mlx_lm.utils.load_model
    observed = []

    def replace_then_load(path, *args, **kwargs):
        source = base / member
        if member == "config.json":
            config = json.loads(source.read_text())
            config["model_file"] = "not-a-real-module.py"
            source.write_text(json.dumps(config))
        else:
            source.write_bytes(b"not-the-verified-weights")
        # No custom code executes: inspect the loader's actual input first.
        actual = (Path(path) / member).read_bytes()
        assert hashlib.sha256(actual).hexdigest() == manifest["base"]["files"][member]["sha256"]
        observed.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(mlx_lm.utils, "load_model", replace_then_load)
    scorer = load_reward_scorer(root)
    assert observed
    assert score_completions(scorer, [("Q", "good")]) == control
    assert all(not path.exists() for path in observed)


@pytest.mark.usefixtures("native_stack")
def test_tokenizer_loader_uses_checked_contents_when_original_path_changes(trained, monkeypatch):
    import hashlib

    from transformers import AutoTokenizer

    _, _, root, _ = trained
    control = load_reward_scorer(root)
    expected = control.tokenizer.encode("good", add_special_tokens=False)
    payload, manifest = read_bundle(root)
    original = AutoTokenizer.from_pretrained
    observed = []

    def replace_then_load(path, *args, **kwargs):
        source = payload / "tokenizer" / "tokenizer.json"
        content = json.loads(source.read_text())
        vocab = content["model"]["vocab"]
        vocab["good"], vocab["bad"] = vocab["bad"], vocab["good"]
        source.write_text(json.dumps(content))
        actual = (Path(path) / "tokenizer.json").read_bytes()
        assert hashlib.sha256(actual).hexdigest() == manifest["files"]["tokenizer/tokenizer.json"][
            "sha256"
        ]
        observed.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", replace_then_load)
    scorer = load_reward_scorer(root)
    assert observed
    assert scorer.tokenizer.encode("good", add_special_tokens=False) == expected
    assert all(not path.exists() for path in observed)


@pytest.mark.usefixtures("native_stack")
@pytest.mark.parametrize("field", ["tokenizer_file", "vocab_file", "merges_file",
                                    "fast_tokenizer_files"])
def test_saved_tokenizer_refuses_external_file_arguments_before_loader(trained, monkeypatch, field):
    from transformers import AutoTokenizer

    _, _, root, _ = trained
    assert load_reward_scorer(root)

    def change(payload, manifest):
        import hashlib

        path = payload / "tokenizer" / "tokenizer_config.json"
        config = json.loads(path.read_text())
        value = "../../outside/tokenizer.0.0.0.json"
        config[field] = [value] if field == "fast_tokenizer_files" else value
        raw = json.dumps(config).encode()
        path.write_bytes(raw)
        manifest["files"]["tokenizer/tokenizer_config.json"] = {
            "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        }

    repoint(root, change)

    def refuse(*args, **kwargs):
        raise AssertionError("external tokenizer file argument reached the loader")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", refuse)
    with pytest.raises(ValueError, match="tokenizer|relative member"):
        load_reward_scorer(root)


@pytest.mark.usefixtures("native_stack")
@pytest.mark.parametrize("external", [False, True])
def test_versioned_tokenizer_files_stay_inside_the_checked_inventory(trained, external):
    import hashlib

    _, _, root, _ = trained
    expected = load_reward_scorer(root).tokenizer.encode("good", add_special_tokens=False)

    def change(payload, manifest):
        config_path = payload / "tokenizer" / "tokenizer_config.json"
        config = json.loads(config_path.read_text())
        version_name = "tokenizer.0.0.0.json"
        if external:
            version = payload.parent / "outside" / version_name
            config["fast_tokenizer_files"] = ["../../outside/" + version_name]
        else:
            version = payload / "tokenizer" / version_name
            config["fast_tokenizer_files"] = [version_name]
        version.parent.mkdir(exist_ok=True)
        raw = (payload / "tokenizer" / "tokenizer.json").read_bytes()
        version.write_bytes(raw)
        if not external:
            manifest["files"]["tokenizer/" + version_name] = {
                "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            }
        raw = json.dumps(config).encode()
        config_path.write_bytes(raw)
        manifest["files"]["tokenizer/tokenizer_config.json"] = {
            "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        }

    repoint(root, change)
    if external:
        with pytest.raises(ValueError, match="tokenizer|relative member"):
            load_reward_scorer(root)
    else:
        actual = load_reward_scorer(root).tokenizer.encode("good", add_special_tokens=False)
        assert actual == expected


@pytest.mark.usefixtures("native_stack")
@pytest.mark.requires_symlink
def test_private_base_loading_accepts_hub_cache_links(trained, tmp_path):
    _, _, root, base = trained
    expected = score_completions(load_reward_scorer(root), [("Q", "good")])
    cache = tmp_path / "cache"
    cache.mkdir()
    for member in ("config.json", "model.safetensors"):
        (cache / member).symlink_to(base / member)
    assert score_completions(load_reward_scorer(root, base=cache), [("Q", "good")]) == expected


@pytest.mark.usefixtures("native_stack")
@pytest.mark.parametrize("loader", ["model", "tokenizer"])
def test_private_loader_inputs_are_cleaned_on_error_without_changing_sources(
    trained, monkeypatch, loader,
):
    import mlx_lm.utils
    from transformers import AutoTokenizer

    _, _, root, base = trained
    before = (root / "reward_model.json").read_bytes(), (base / "config.json").read_bytes()
    inputs = []

    def refuse(path, *args, **kwargs):
        inputs.append(Path(path))
        assert inputs[-1].is_dir()
        raise ValueError("injected loader failure")

    if loader == "model":
        monkeypatch.setattr(mlx_lm.utils, "load_model", refuse)
    else:
        monkeypatch.setattr(AutoTokenizer, "from_pretrained", refuse)
    with pytest.raises(ValueError, match="injected loader failure"):
        load_reward_scorer(root)
    assert inputs and all(not path.exists() for path in inputs)
    after = (root / "reward_model.json").read_bytes(), (base / "config.json").read_bytes()
    assert after == before


@pytest.mark.usefixtures("native_stack")
def test_tokenizer_copy_refuses_bytes_changed_after_bundle_validation(trained, monkeypatch):
    import soup_cli.utils.mlx_reward_bundle as bundles

    _, _, root, _ = trained
    payload, _ = read_bundle(root)
    original = bundles.load_tokenizer

    def change_then_load(path, **kwargs):
        source = payload / "tokenizer" / "tokenizer.json"
        config = json.loads(source.read_text())
        vocab = config["model"]["vocab"]
        vocab["good"], vocab["bad"] = vocab["bad"], vocab["good"]
        source.write_text(json.dumps(config))
        return original(path, **kwargs)

    monkeypatch.setattr(bundles, "load_tokenizer", change_then_load)
    with pytest.raises(ValueError, match="size or SHA-256 mismatch"):
        load_reward_scorer(root)


@pytest.mark.usefixtures("native_stack")
def test_private_base_loading_preserves_a_sharded_index(trained):
    _, _, root, base = trained
    expected = score_completions(load_reward_scorer(root), [("Q", "good")])

    def change(payload, manifest):
        weights = list(mx.load(str(base / "model.safetensors")).items())
        (base / "model.safetensors").unlink()
        mapping = {}
        for index, shard in enumerate((weights[::2], weights[1::2]), 1):
            name = f"model-{index:05d}-of-00002.safetensors"
            mx.save_safetensors(str(base / name), dict(shard))
            mapping.update({key: name for key, _ in shard})
        (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        manifest["base"] = base_identity(base)

    repoint(root, change)
    assert score_completions(load_reward_scorer(root), [("Q", "good")]) == expected


def test_private_member_copy_checks_the_bound_and_exact_identity(tmp_path):
    import hashlib

    from soup_cli.utils.mlx_reward_bundle import _copy_member

    source = tmp_path / "source"
    source.write_bytes(b"abc")
    expected = {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    destination = tmp_path / "valid"
    _copy_member(source, destination, expected=expected, limit=3)
    assert destination.read_bytes() == b"abc"
    with pytest.raises(ValueError, match="size or SHA-256 mismatch"):
        _copy_member(source, tmp_path / "bad-hash", expected={**expected, "sha256": "0" * 64})
    destination = tmp_path / "too-large"
    with pytest.raises(ValueError, match="byte bound"):
        _copy_member(source, destination, limit=2)
    assert not destination.exists()
    assert source.read_bytes() == b"abc"


def test_private_member_copy_checks_growth_after_open(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import soup_cli.utils.mlx_reward_bundle as bundles

    source = tmp_path / "source"
    source.write_bytes(b"abc")
    original = bundles.checked_file

    @contextmanager
    def grow_after_stat(path):
        with original(path) as stream:
            class Growing:
                grown = False

                def fileno(self):
                    return stream.fileno()

                def read(self, size):
                    if not self.grown:
                        self.grown = True
                        with source.open("ab") as writer:
                            writer.write(b"def")
                    return stream.read(size)

            yield Growing()

    monkeypatch.setattr(bundles, "checked_file", grow_after_stat)
    destination = tmp_path / "bounded"
    with pytest.raises(ValueError, match="byte bound"):
        bundles._copy_member(source, destination, limit=3)
    assert destination.stat().st_size == 0


@pytest.mark.requires_symlink
def test_file_open_rechecks_parent_after_the_initial_directory_check(tmp_path, monkeypatch):
    import soup_cli.utils.mlx_reward_bundle as bundles

    parent, outside = tmp_path / "parent", tmp_path / "outside"
    parent.mkdir()
    outside.mkdir()
    (parent / "data.json").write_text('{"outside":false}')
    (outside / "data.json").write_text('{"outside":true}')
    assert bundles.read_json(parent / "data.json") == {"outside": False}
    original = bundles._checked_dirs

    def check_then_replace(path):
        original(path)
        if path == parent:
            parent.rename(tmp_path / "previous-parent")
            parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(bundles, "_checked_dirs", check_then_replace)
    with pytest.raises(ValueError, match="unreadable|linked"):
        bundles.read_json(parent / "data.json")


def test_tokenizer_version_metadata_requires_an_available_local_member(tmp_path):
    from soup_cli.utils.mlx_reward_bundle import _tokenizer_metadata

    config = tmp_path / "tokenizer_config.json"
    (tmp_path / "tokenizer.json").write_text("{}")
    config.write_text(json.dumps({"fast_tokenizer_files": ["tokenizer.json"]}))
    _tokenizer_metadata(tmp_path)
    config.write_text(json.dumps({"fast_tokenizer_files": ["missing-tokenizer.json"]}))
    with pytest.raises(ValueError, match="unavailable member"):
        _tokenizer_metadata(tmp_path)


@pytest.mark.requires_symlink
@pytest.mark.parametrize("entry", ["output", "contained", "json"])
def test_directory_checks_do_not_stop_at_a_cwd_alias(tmp_path, monkeypatch, entry):
    from soup_cli.utils.mlx_reward_bundle import contained, read_json, validate_output

    target = tmp_path / "target"
    target.mkdir()
    cwd = target / "child"
    cwd.mkdir()
    (cwd / "data.json").write_text('{"control":true}')
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.chdir(cwd)
    validate_output(cwd)
    assert read_json(contained(cwd, "data.json")) == {"control": True}
    linked = alias / "child"
    operations = {
        "output": lambda: validate_output(linked),
        "contained": lambda: contained(linked, "data.json"),
        "json": lambda: read_json(linked / "data.json"),
    }
    with pytest.raises((OSError, ValueError), match="link|unreadable"):
        operations[entry]()


@pytest.mark.usefixtures("native_stack")
@pytest.mark.requires_symlink
def test_publication_refuses_a_cwd_alias_before_writing(trained, tmp_path, monkeypatch):
    trainer, _, root, _ = trained
    _, manifest = read_bundle(root)
    contract = {key: value for key, value in manifest.items()
                if key not in ("files", "format_version", "artifact_type")}
    target = tmp_path / "target"
    target.mkdir()
    cwd = target / "child"
    cwd.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.chdir(cwd)
    with pytest.raises((OSError, ValueError), match="link|unreadable"):
        publish_bundle(alias / "child", trainer.model, trainer.tokenizer, contract)
    assert list(cwd.iterdir()) == []
