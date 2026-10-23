"""Portable native-consumer contracts; hardware boundaries are replaced explicitly."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def test_native_infer_reject_peft_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "owner/base", "r": 4,
    }))
    (tmp_path / "prompts.jsonl").write_text('{"prompt":"hello"}\n')
    args = ["infer", "--backend", "mlx", "--model", str(adapter)]
    args += ["--input", "prompts.jsonl", "--output", "results.jsonl"]
    result = runner.invoke(app, args, input="/quit\n")
    assert result.exit_code == 1
    assert "native MLX metadata" in result.output
    assert not (tmp_path / "results.jsonl").exists()


def test_native_infer_dispatch_and_preserve_results(tmp_path, monkeypatch):
    from soup_cli.utils import mlx_generation

    monkeypatch.chdir(tmp_path)
    model = tmp_path / "base"
    model.mkdir()
    histories = []
    monkeypatch.setattr(mlx_generation, "load_native_model", lambda *a, **k: (object(), object()))

    def generate(model, tokenizer, messages, **kwargs):
        histories.append([dict(message) for message in messages])
        return f"answer-{len(histories)}", 3

    monkeypatch.setattr(mlx_generation, "generate_native", generate)
    args = ["infer", "--backend", "mlx", "--model", str(model)]
    (tmp_path / "prompts.jsonl").write_text('{"prompt":"one"}\n{"prompt":"two"}\n')
    args += ["--input", "prompts.jsonl", "--output", "results.jsonl"]
    result = runner.invoke(app, args, input="one\ntwo\n/clear\nthree\n/quit\n")
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert rows == [
        {"prompt": "one", "response": "answer-1", "tokens_generated": 3},
        {"prompt": "two", "response": "answer-2", "tokens_generated": 3},
    ]
    assert [h[0]["content"] for h in histories] == ["one", "two"]


def _metadata(tmp_path):
    path = tmp_path / "adapter"
    path.mkdir()
    config = {
        "model": "owner/base", "fine_tune_type": "lora", "num_layers": 1,
        "lora_parameters": {"rank": 2, "scale": 4.0, "dropout": 0.0, "keys": ["q_proj"]},
    }
    (path / "adapters.safetensors").write_bytes(b"boundary not loaded")
    return path, config


@pytest.mark.parametrize("change, diagnostic", [
    ({"model": ""}, "identity"),
    ({"base_model_name_or_path": "other/base"}, "Ambiguous"),
    ({"fine_tune_type": "dora"}, "ordinary linear LoRA"),
    ({"num_layers": True}, "positive integers"),
    ({"lora_parameters": {}}, "positive integers"),
    ({"lora_parameters": {"rank": 2, "scale": float("nan"), "dropout": 0, "keys": ["q"]}},
     "finite"),
    ({"lora_parameters": {"rank": 2, "scale": 1, "dropout": 0, "keys": ["auto"]}},
     "resolved target"),
])
def test_metadata_refusals_have_adjacent_valid_control(tmp_path, change, diagnostic):
    from soup_cli.utils.mlx_generation import read_native_adapter

    path, config = _metadata(tmp_path)
    metadata = path / "adapter_config.json"
    metadata.write_text(json.dumps(config))
    assert read_native_adapter(path) == config
    metadata.write_text(json.dumps(config | change))
    with pytest.raises(ValueError, match=diagnostic):
        read_native_adapter(path)


def test_metadata_missing_weights_and_wrong_base(tmp_path):
    from soup_cli.utils.mlx_generation import read_native_adapter

    path, config = _metadata(tmp_path)
    (path / "adapter_config.json").write_text(json.dumps(config))
    assert read_native_adapter(path, "owner/base") == config
    with pytest.raises(ValueError, match="does not match"):
        read_native_adapter(path, "owner/wrong")
    (path / "adapters.safetensors").unlink()
    with pytest.raises(ValueError, match="missing adapters"):
        read_native_adapter(path)


@pytest.mark.parametrize("options, diagnostic", [
    ({"temperature": float("nan")}, "temperature"),
    ({"temperature": -1}, "temperature"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"device": "cuda"}, "device"),
    ({"hub": "modelscope"}, "hub"),
    ({"task": "asr"}, "text"),
    ({"batch_size": 2}, "batch-size"),
    ({"cuda_graphs": True}, "cuda-graphs"),
])
def test_native_options_refuse_unimplemented_semantics(options, diagnostic):
    from soup_cli.utils.mlx_generation import validate_native_options

    valid = dict(temperature=0.7, max_tokens=10, device=None, hub="hf")
    validate_native_options(**valid)
    with pytest.raises(ValueError, match=diagnostic):
        validate_native_options(**(valid | options))


def test_infer_keeps_completed_rows_on_generation_failure(tmp_path, monkeypatch):
    from soup_cli.utils import mlx_generation

    monkeypatch.chdir(tmp_path)
    (tmp_path / "base").mkdir()
    (tmp_path / "prompts.jsonl").write_text('{"prompt":"one"}\n{"prompt":"two"}\n')
    monkeypatch.setattr(mlx_generation, "load_native_model", lambda *a, **k: (None, None))

    def generate(model, tokenizer, messages, **kwargs):
        if messages[0]["content"] == "two":
            raise ValueError("generation failed")
        return "complete", 2

    monkeypatch.setattr(mlx_generation, "generate_native", generate)
    result = runner.invoke(app, ["infer", "--backend", "mlx", "--model", "base",
                                 "--input", "prompts.jsonl", "--output", "results.jsonl"])
    assert result.exit_code != 0
    assert json.loads((tmp_path / "results.jsonl").read_text()) == {
        "prompt": "one", "response": "complete", "tokens_generated": 2,
    }
    assert "Inference Complete" not in result.output


@pytest.mark.parametrize("interrupted", [False, True])
def test_generation_tokenizes_template_once_counts_tokens_and_closes(monkeypatch, interrupted):
    import sys
    from types import ModuleType, SimpleNamespace

    from soup_cli.utils.mlx_generation import generate_native

    closed = []
    received = []

    class Tokenizer:
        chat_template = "template with BOS"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "é"}]
            assert tokenize is False and add_generation_prompt is True
            return "<bos> user é assistant"

        def encode(self, text, add_special_tokens=True):
            assert text == "<bos> user é assistant"
            assert add_special_tokens is False
            return [1, 3, 7, 4]

    def stream(model, tokenizer, *, prompt, max_tokens, sampler):
        received.append((prompt, max_tokens, sampler))
        try:
            yield SimpleNamespace(text="ré", generation_tokens=1)
            if interrupted:
                raise KeyboardInterrupt
            # A final flush can contain several decoded characters and an EOS,
            # so neither chunk count nor re-tokenized text is a token counter.
            yield SimpleNamespace(text="ponse", generation_tokens=4)
        finally:
            closed.append(True)

    native = ModuleType("mlx_lm")
    native.stream_generate = stream
    samples = ModuleType("mlx_lm.sample_utils")
    samples.make_sampler = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "mlx_lm", native)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", samples)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            generate_native(None, Tokenizer(), [{"role": "user", "content": "é"}],
                            max_tokens=4, temperature=0)
    else:
        assert generate_native(None, Tokenizer(), [{"role": "user", "content": "é"}],
                               max_tokens=4, temperature=0) == ("réponse", 4)
    assert received == [([1, 3, 7, 4], 4, {"temp": 0, "top_p": 1.0})]
    assert closed == [True]


def test_default_backend_refuses_native_artifact_before_transformers(tmp_path):
    path, config = _metadata(tmp_path)
    (path / "adapter_config.json").write_text(json.dumps(config))
    args = ["infer", "--model", str(path)]
    args += ["--input", "unused.jsonl", "--output", "unused-results.jsonl"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "select --backend mlx" in result.output


@pytest.mark.parametrize("option", [["--temperature", "nan"], ["--device", "cuda"]])
def test_public_native_option_refusals_precede_loading(tmp_path, option):
    args = ["infer", "--backend", "mlx", "--model", str(tmp_path)] + option
    args += ["--input", "unused.jsonl", "--output", "unused-results.jsonl"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "MLX" in result.output
    assert ("temperature" if option[0] == "--temperature" else "device") in result.output
    assert "Model loaded" not in result.output
