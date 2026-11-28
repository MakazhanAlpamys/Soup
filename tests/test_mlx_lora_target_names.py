"""Explicit native target names must not silently train a subset (#3163)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

if TYPE_CHECKING:
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

TASKS = ["sft", "dpo", "orpo", "simpo", "kto", "grpo"]


@pytest.fixture(scope="module", params=[False, True], ids=["dense", "4bit"])
def native_base(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest,
) -> tuple[Path, bool]:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.utils import save_model
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("target-base")
    mx.random.seed(42)
    args = ModelArgs(
        model_type="llama", hidden_size=128, num_hidden_layers=2,
        intermediate_size=256, num_attention_heads=2, num_key_value_heads=1,
        rms_norm_eps=1e-5, vocab_size=260, max_position_embeddings=2048,
        tie_word_embeddings=True,
    )
    model = Model(args)
    config = {**asdict(args), "eos_token_id": 2, "bos_token_id": 1, "pad_token_id": 3}
    if request.param:
        nn.quantize(model, group_size=32, bits=4)
        config["quantization"] = {"group_size": 32, "bits": 4}
    save_model(path, model)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    vocab = ["<unk>", "<bos>", "<eos>", "<pad>"] + sorted(
        pre_tokenizers.ByteLevel.alphabet()
    )
    raw = Tokenizer(models.BPE(vocab=dict(zip(vocab, range(len(vocab)))), merges=[]))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", bos_token="<bos>",
        eos_token="<eos>", pad_token="<pad>",
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )
    tokenizer.save_pretrained(path)
    return path, request.param


def recipe(
    tmp_path: Path, native_base: tuple[Path, bool], task: str, targets: str | list[str],
) -> tuple[MLXSFTTrainerWrapper, SoupConfig]:
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.mlx_routing import get_mlx_trainer

    base, quantized = native_base
    prompt = "Please name a color and explain the answer briefly."
    if task in {"dpo", "orpo", "simpo"}:
        rows = [{"prompt": prompt, "chosen": "Blue is a color.", "rejected": "A number."}] * 2
    elif task == "kto":
        rows = [{"prompt": prompt, "completion": "Blue is a color.", "label": True},
                {"prompt": prompt, "completion": "A number.", "label": False}]
    elif task == "grpo":
        rows = [{"prompt": [{"role": "user", "content": prompt}]}] * 2
    else:
        rows = [{"text": prompt + " Blue is a color."}] * 2
    training = {
        "quantization": "4bit" if quantized else "none", "epochs": 1,
        "batch_size": 2 if task in {"kto", "grpo"} else 1, "gradient_accumulation_steps": 1,
        "lr": 0.001, "logging_steps": 1, "save_steps": 1, "seed": 42,
        "data_seed": 42, "warmup_ratio": 0, "gradient_checkpointing": False,
        "rewind_log": False, "lora": {
            "r": 4, "alpha": 8, "dropout": 0, "target_modules": targets,
        },
    }
    if task == "grpo":
        training.update(reward_fn="format", num_generations=2, grpo_beta=0.1)
    config = SoupConfig(
        base=str(base), backend="mlx", task=task, output=str(tmp_path / "adapter"),
        data={"train": "unused.jsonl", "max_length": 64 if task == "grpo" else 128,
              "val_split": 0, "train_on_responses_only": False},
        training=training,
    )
    wrapper = get_mlx_trainer(task)(config)
    wrapper.setup({"train": rows})
    return wrapper, config


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("missing_first", [False, True], ids=["valid-first", "missing-first"])
def test_missing_names_refuse_before_model_mutation_or_adapter_publication(
    tmp_path: Path, native_base: tuple[Path, bool], task: str, missing_first: bool,
) -> None:
    import numpy as np
    from mlx.utils import tree_flatten

    names = ["self_attn.q_proj", "self_attn.z_missing", "self_attn.a_missing"]
    if missing_first:
        names.reverse()
    wrapper, config = recipe(tmp_path, native_base, task, names)
    before = {name: np.array(value) for name, value in tree_flatten(wrapper.model.parameters())}
    trainable = {name for name, _ in tree_flatten(wrapper.model.trainable_parameters())}

    with pytest.raises(ValueError, match="unmatched targets") as error:
        wrapper.train()

    assert "self_attn.z_missing" in str(error.value)
    assert "self_attn.a_missing" in str(error.value)
    after = dict(tree_flatten(wrapper.model.parameters()))
    assert before.keys() == after.keys()
    for name, value in before.items():
        np.testing.assert_array_equal(np.array(after[name]), value)
    assert {name for name, _ in tree_flatten(wrapper.model.trainable_parameters())} == trainable
    assert not list(Path(config.output).glob("*.safetensors"))
    assert not (Path(config.output) / "adapter_config.json").exists()


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize(
    "targets", [["self_attn.q_proj"], ["self_attn.q_proj", "self_attn.v_proj"], "auto", "", []],
    ids=["valid-only", "all-valid", "auto", "empty-string", "empty-list"],
)
def test_valid_target_names_train_save_and_strictly_reload(
    tmp_path: Path, native_base: tuple[Path, bool], task: str, targets: str | list[str],
) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from soup_cli.utils.mlx_generation import load_native_model

    wrapper, config = recipe(tmp_path, native_base, task, targets)
    result = wrapper.train()
    assert result["total_steps"] > 0
    assert math.isfinite(result["final_loss"])
    model, _ = load_native_model(config.output)
    actual = dict(tree_flatten(model.trainable_parameters()))
    # The consumer freezes its base and strictly checks the complete adapter.
    assert actual
    expected = (
        ["self_attn.q_proj", "self_attn.v_proj"] if targets in ("auto", "", []) else targets
    )
    for name in actual:
        assert any(key + ".lora_" in name for key in expected), name
    assert all(bool(mx.all(mx.isfinite(value)).item()) for value in actual.values())


@pytest.mark.parametrize("task", TASKS)
def test_public_train_refuses_missing_target_instead_of_publishing(
    tmp_path: Path, native_base: tuple[Path, bool], task: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import subprocess
    import sys

    import yaml

    import soup_cli
    from tests.conftest import strip_ansi

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    _, config = recipe(tmp_path, native_base, task, ["self_attn.q_proj", "self_attn.v_projj"])
    prompt = "Please name a color and explain the answer briefly."
    if task in {"dpo", "orpo", "simpo"}:
        rows = [{"prompt": prompt, "chosen": "Blue is a color.", "rejected": "A number."}] * 2
        fmt = "dpo"
    elif task == "kto":
        rows = [{"prompt": prompt, "completion": "Blue is a color.", "label": True},
                {"prompt": prompt, "completion": "A number.", "label": False}]
        fmt = "kto"
    else:
        rows = [{"instruction": prompt, "output": "Blue is a color."}] * 2
        fmt = "alpaca"
    (tmp_path / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    data = config.model_dump(mode="json", exclude_unset=True)
    data["data"].update(
        train=str(tmp_path / "rows.jsonl"), format=fmt,
    )
    path = tmp_path / "soup.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    response = subprocess.run(
        [sys.executable, "-m", "soup_cli", "--no-telemetry", "--no-audit-log",
         "train", "--config", str(path), "--yes"],
        env={**os.environ, "PYTHONPATH": str(Path(soup_cli.__file__).resolve().parents[1])},
        capture_output=True, text=True, timeout=90, check=False,
    )
    assert response.returncode == 1, response.stdout + response.stderr
    text = " ".join(strip_ansi(response.stdout + response.stderr).split())
    assert "unmatched targets" in text and "self_attn.v_projj" in text
    assert not list((tmp_path / "adapter").glob("*.safetensors"))


def test_ordinary_lora_preserves_supported_root_embedding_conversion(
    tmp_path: Path, native_base: tuple[Path, bool],
) -> None:
    from mlx_lm.tuner.lora import LoRAEmbedding

    wrapper, _ = recipe(tmp_path, native_base, "sft", ["model.embed_tokens"])
    wrapper._apply_lora(wrapper.model)
    assert isinstance(wrapper.model.model.embed_tokens, LoRAEmbedding)


def test_dora_still_refuses_embedding_targets(
    tmp_path: Path, native_base: tuple[Path, bool],
) -> None:
    wrapper, config = recipe(tmp_path, native_base, "sft", ["model.embed_tokens"])
    config.training.lora.use_dora = True
    with pytest.raises(ValueError, match="DoRA.*dense linear targets"):
        wrapper.train()


def test_correcting_a_missing_name_on_the_same_trainer_can_train(
    tmp_path: Path, native_base: tuple[Path, bool],
) -> None:
    from soup_cli.utils.mlx_generation import load_native_model

    wrapper, config = recipe(tmp_path, native_base, "sft", ["self_attn.v_projj"])
    with pytest.raises(ValueError, match="self_attn.v_projj"):
        wrapper.train()
    config.training.lora.target_modules = ["self_attn.q_proj"]
    assert wrapper.train()["total_steps"] > 0
    load_native_model(config.output)


@pytest.mark.parametrize("targets", ["auto", ["auto"], "", []])
def test_auto_keeps_existing_default_matching_on_a_partial_model(
    targets: str | list[str],
) -> None:
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(32, 32)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = [Block()]

    config = SoupConfig(
        base="unused-local", backend="mlx", data={"train": "unused.jsonl"},
        training={"quantization": "none", "lora": {"r": 4, "target_modules": targets}},
    )
    model = Model()
    MLXSFTTrainerWrapper(config)._apply_lora(model)
    assert isinstance(model.layers[0].self_attn.q_proj, LoRALinear)
