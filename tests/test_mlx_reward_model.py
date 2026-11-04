"""Native MLX reward training, safe bundles, and portable input/config contracts."""

import json
import os
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
    trainer.setup({"train": rows, "val": rows[:3]})
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
@pytest.mark.parametrize(
    "fault",
    [
        "file_symlink",
        "file_hardlink",
        "directory_symlink",
        "duplicate_json",
        "nonfinite_json",
        "oversize_header",
        "custom_code",
    ],
)
@pytest.mark.requires_symlink
def test_bundle_read_safety(trained, tmp_path, fault):
    _, _, root, base = trained
    payload, _ = read_bundle(root)
    if fault in ("duplicate_json", "nonfinite_json"):
        (root / "reward_model.json").write_text(
            '{"a":1,"a":2}' if fault == "duplicate_json" else '{"a":1e999}'
        )
    elif fault == "custom_code":
        config = json.loads((base / "config.json").read_text())
        config["model_file"] = "evil.py"
        (base / "config.json").write_text(json.dumps(config))
    elif fault == "oversize_header":
        path = payload / "reward.safetensors"
        path.write_bytes((4 * 1024 * 1024 + 1).to_bytes(8, "little"))
    elif fault == "directory_symlink":
        token = payload / "tokenizer"
        new = tmp_path / "moved-tokenizer"
        token.rename(new)
        token.symlink_to(new, target_is_directory=True)
    else:
        path = payload / "reward.safetensors"
        new = tmp_path / "moved-weight"
        path.rename(new)
        if fault == "file_symlink":
            path.symlink_to(new)
        else:
            os.link(new, path)
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
    from soup_cli.utils.mlx_reward_bundle import label

    value = label("[bold]\x1b" + "x" * 500)
    assert "[" not in value and "\x1b" not in value and len(value) <= 260


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
    return subprocess.run(
        [sys.executable, "-m", "soup_cli", "train", "--config", str(config_path), "--yes", *args],
        cwd=config_path.parent,
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", COLUMNS="240"),
    )


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--push-as", "org/model"], "--push-as"),
        (["--gate", "missing.yaml"], "--gate"),
        (["--diagnose-gate", "missing.yaml"], "--diagnose-gate"),
        (["--capture-activations", "activations"], "--capture-activations"),
        (["--resume", "missing", "--push-as", "org/model"], "--resume is not supported"),
        (["--hf-resume", "--push-as", "org/model"], "--hf-resume is not supported"),
    ],
)
def test_public_cli_early_refusals(tmp_path, args, expected):
    import yaml

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(reward_config().model_dump(mode="json")))
    result = public_cli(config, *args)
    assert result.returncode == 1, result.stdout + result.stderr
    assert expected in result.stdout, result.stdout + result.stderr
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
