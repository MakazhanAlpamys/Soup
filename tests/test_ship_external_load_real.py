"""Offline real HF/PEFT/HFLM qualification; CUDA cells use the project's GPU marker."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.cli_runner import CliRunner


@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.parametrize(
    "device,quantization",
    [
        ("cpu", "none"),
        pytest.param("cuda:0", "4bit", marks=pytest.mark.gpu),
        pytest.param("cuda:0", "8bit", marks=pytest.mark.gpu),
    ],
)
def test_real_local_mixed_suite(tmp_path, monkeypatch, adapter, device, quantization):
    pytest.importorskip("lm_eval")
    import lm_eval.evaluator
    import torch
    import yaml
    from lm_eval.tasks import TaskManager
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from soup_cli.commands import ship
    from soup_cli.utils import live_eval

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(3366)
    vocab = {word: index for index, word in enumerate(
        ["<unk>", "<bos>", "<eos>", "hello", "world", "hi", "bye", "say"]
    )}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>",
        pad_token="<eos>", unk_token="<unk>",
    )
    config = LlamaConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        bos_token_id=1, eos_token_id=2, pad_token_id=2,
    )
    model = LlamaForCausalLM(config)
    # End live generation after its first token; likelihood benchmarks still run real forwards.
    model.generation_config.eos_token_id = list(range(len(vocab)))
    model.save_pretrained("base")
    tokenizer.save_pretrained("base")
    if adapter:
        tuned = get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
        with torch.no_grad():
            for name, parameter in tuned.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.01)
        tuned.save_pretrained("adapter")
        tokenizer.save_pretrained("adapter")
    else:
        model.save_pretrained("tuned")
        tokenizer.save_pretrained("tuned")
    Path("tasks.jsonl").write_text(
        '{"prompt":"say hi","expected":"hi","scoring":"contains"}\n'
    )
    Path("soup.yaml").write_text(
        f"base: base\ndata:\n  train: tasks.jsonl\ntraining:\n  quantization: {quantization}\n"
    )
    Path("benchmark.jsonl").write_text('{"prompt":"hello","answer":1}\n')
    task_path = tmp_path / "local_tasks"
    task_path.mkdir()
    (task_path / "local_probe.yaml").write_text(yaml.safe_dump({
        "task": "local_probe", "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(tmp_path / "benchmark.jsonl")}},
        "test_split": "test", "output_type": "multiple_choice", "doc_to_text": "prompt",
        "doc_to_choice": ["bye", "hi"], "doc_to_target": "answer",
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
        "num_fewshot": 0,
    }))
    # Supply a local task through the evaluator's standard include_path mechanism.
    monkeypatch.setattr(
        lm_eval.evaluator, "TaskManager",
        lambda **kwargs: TaskManager(include_path=str(task_path), include_defaults=False, **kwargs),
    )
    actual_load = live_eval.load_model_and_tokenizer
    loads = []

    def record_load(*args, **kwargs):
        loaded = actual_load(*args, **kwargs)
        model = loaded[0]
        loads.append({
            "device": str(model.device),
            "dtype": next(model.parameters()).dtype,
            "modules": {type(module).__name__ for module in model.modules()},
            "config": getattr(model.config, "quantization_config", None),
        })
        return loaded

    monkeypatch.setattr(live_eval, "load_model_and_tokenizer", record_load)
    result = CliRunner().invoke(ship.app, [
        "--base", "base", "--adapter" if adapter else "--tuned",
        "adapter" if adapter else "tuned", "--task-eval", "tasks.jsonl",
        "--config", "soup.yaml", "--device", device,
        "--general-suite", "mini_mmlu,local_probe", "--emit-evidence", "evidence.json",
    ])
    assert result.exit_code in (0, 2), (result.output, result.exception)
    evidence = json.loads(Path("evidence.json").read_text())
    assert evidence["numerics"] == ("float32" if quantization == "none" else quantization)
    assert set(evidence["benchmarks"]) == {"mini_mmlu", "local_probe"}
    assert all(math.isfinite(value) for value in evidence["benchmarks"]["local_probe"].values())
    assert len(loads) == 4
    for load in loads:
        assert load["device"] == ("cpu" if device == "cpu" else "cuda:0")
        if quantization == "none":
            assert load["dtype"] == torch.float32
            assert load["config"] is None
        else:
            assert ("Linear4bit" if quantization == "4bit" else "Linear8bitLt") in load["modules"]
            if quantization == "4bit":
                assert load["config"].bnb_4bit_quant_type == "nf4"
                assert load["config"].bnb_4bit_use_double_quant is True
