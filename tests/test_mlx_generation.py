"""Portable native-consumer contracts; hardware boundaries are replaced explicitly."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


@pytest.fixture
def native_boundary(monkeypatch):
    """Replace only the optional engine; exercise Soup's actual loader and guards."""
    class Linear:
        pass

    class QuantizedLinear(Linear):
        bits = 4

    model = SimpleNamespace()
    state = SimpleNamespace(
        config={}, calls=[], loaded=[], target=Linear(),
        weights={"adapter.lora_a": SimpleNamespace(shape=(2, 2), dtype="float")},
    )
    model.layers = [SimpleNamespace(named_modules=lambda: [("q_proj", state.target)])
                    for _ in range(2)]
    model.named_modules = lambda: []
    model.freeze = lambda: None
    model.eval = lambda: None
    model.trainable_parameters = lambda: state.weights
    model.parameters = lambda: state.weights
    model.load_weights = lambda weights, strict: state.loaded.append(weights)

    def load(path, *, tokenizer_config, return_config, trust_remote_code=False):
        state.calls.append((path, tokenizer_config, trust_remote_code))
        assert return_config is True
        tokenizer = SimpleNamespace(encode=lambda *a, **k: [1, 2], eos_token_id=None)
        return model, tokenizer, state.config

    def stream(*args, **kwargs):
        yield SimpleNamespace(text="done", generation_tokens=1)

    modules = {name: ModuleType(name) for name in (
        "mlx", "mlx.core", "mlx.nn", "mlx.utils", "mlx_lm", "mlx_lm.tuner",
        "mlx_lm.tuner.utils", "mlx_lm.sample_utils",
    )}
    modules["mlx.core"].load = lambda path: state.weights
    modules["mlx.core"].eval = lambda weights: None
    modules["mlx.core"].floating = "float"
    modules["mlx.core"].issubdtype = lambda actual, expected: actual == expected
    modules["mlx.core"].isfinite = lambda weight: True
    modules["mlx.core"].all = lambda value: SimpleNamespace(item=lambda: value)
    modules["mlx.nn"].Linear = Linear
    modules["mlx.nn"].QuantizedLinear = QuantizedLinear
    modules["mlx.utils"].tree_flatten = lambda tree: list(tree.items())
    modules["mlx_lm"].load = load
    modules["mlx_lm"].stream_generate = stream
    modules["mlx_lm.sample_utils"].make_sampler = lambda **kwargs: kwargs
    modules["mlx_lm.tuner.utils"].linear_to_lora_layers = lambda *args: None
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    state.modules = modules
    state.model = model
    return state


def _infer_args(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "base").mkdir()
    (tmp_path / "prompts.jsonl").write_text('{"prompt":"one"}\n')
    return ["infer", "--backend", "mlx", "--model", "base", "--input", "prompts.jsonl",
            "--output", "results.jsonl"]


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
    result = runner.invoke(app, args)
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
    result = runner.invoke(app, args)
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
    ({"use_dora": True}, "ordinary linear LoRA"),
    ({"lora_parameters": "x"}, "requires lora_parameters"),
    ({"num_layers": True}, "positive integers"),
    ({"num_layers": 0}, "positive integers"),
    ({"lora_parameters": {}}, "positive integers"),
    ({"lora_parameters": {"rank": 0, "scale": 4, "dropout": 0, "keys": ["q_proj"]}},
     "positive integers"),
    ({"lora_parameters": {"rank": 2, "scale": float("nan"), "dropout": 0, "keys": ["q"]}},
     "finite"),
    ({"lora_parameters": {"rank": 2, "scale": 1, "dropout": 0, "keys": ["auto"]}},
     "resolved target"),
    *[({"lora_parameters": {"rank": 2, "scale": scale, "dropout": 0,
                              "keys": ["q_proj"]}}, "positive scale") for scale in (0, -1)],
    ({"lora_parameters": {"rank": 2, "scale": 4, "dropout": 1, "keys": ["q_proj"]}},
     "dropout"),
    *[({"lora_parameters": {"rank": 2, "scale": 4, "dropout": 0, "keys": keys}},
       "unique resolved target") for keys in ([], ["q_proj", "q_proj"])],
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


def test_metadata_missing_config_and_base_without_adapter(tmp_path):
    from soup_cli.utils.mlx_generation import read_native_adapter

    path, config = _metadata(tmp_path)
    metadata = path / "adapter_config.json"
    metadata.write_text(json.dumps(config))
    assert read_native_adapter(path, "owner/base") == config
    metadata.unlink()
    with pytest.raises(ValueError, match="weights require adapter_config"):
        read_native_adapter(path)
    (path / "adapters.safetensors").unlink()
    assert read_native_adapter(path) is None
    with pytest.raises(ValueError, match="--base is only valid"):
        read_native_adapter(path, "owner/base")


@pytest.mark.parametrize("options, diagnostic", [
    ({"temperature": float("nan")}, "temperature"),
    ({"temperature": -1}, "temperature"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"max_tokens": 16385}, "max_tokens"),
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
    assert isinstance(result.exception, ValueError)
    assert "generation failed" in str(result.exception)
    assert json.loads((tmp_path / "results.jsonl").read_text()) == {
        "prompt": "one", "response": "complete", "tokens_generated": 2,
    }
    assert "Inference Complete" not in result.output


@pytest.mark.parametrize("temperature", [0, 0.7])
@pytest.mark.parametrize("interrupted", [False, True])
def test_generation_tokenizes_template_once_counts_tokens_and_closes(
    monkeypatch, interrupted, temperature,
):
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
            yield SimpleNamespace(text="  ré", generation_tokens=1)
            if interrupted:
                raise KeyboardInterrupt
            # A final flush can contain several decoded characters and an EOS,
            # so neither chunk count nor re-tokenized text is a token counter.
            yield SimpleNamespace(text="ponse \n", generation_tokens=4)
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
                            max_tokens=4, temperature=temperature)
    else:
        assert generate_native(None, Tokenizer(), [{"role": "user", "content": "é"}],
                               max_tokens=4, temperature=temperature) == ("réponse", 4)
    assert received == [([1, 3, 7, 4], 4,
                         {"temp": temperature, "top_p": 0.9 if temperature else 1.0})]
    assert closed == [True]


def test_default_backend_refuses_native_artifact_before_transformers(tmp_path):
    path, config = _metadata(tmp_path)
    (path / "adapter_config.json").write_text(json.dumps(config))
    args = ["infer", "--model", str(path)]
    args += ["--input", "unused.jsonl", "--output", "unused-results.jsonl"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "select --backend mlx" in result.output


@pytest.mark.parametrize("option,diagnostic,code", [
    (["--temperature", "nan"], "temperature", 1),
    (["--device", "cuda"], "device", 1),
    (["--backend", "bogus"], "backend must be transformers or mlx", 2),
    (["--asr-language", "fr"], "ASR options", 1),
    (["--asr-task", "translate"], "ASR options", 1),
    (["--audio-dir", "audio"], "ASR options", 1),
    (["--task", "asr"], "requires --task text", 1),
    (["--batch-size", "2"], "requires --batch-size 1", 1),
    (["--cuda-graphs"], "does not support --cuda-graphs", 1),
    (["--hub", "modelscope"], "--hub hf only", 1),
])
def test_public_native_option_refusals_precede_loading(
    tmp_path, monkeypatch, native_boundary, option, diagnostic, code,
):
    args = _infer_args(tmp_path, monkeypatch)
    control = runner.invoke(app, args)
    assert control.exit_code == 0, control.output
    (tmp_path / "results.jsonl").unlink()
    result = runner.invoke(app, args + option)
    from tests.conftest import strip_ansi

    assert result.exit_code == code, result.output
    assert diagnostic in " ".join(strip_ansi(result.output).split())
    assert not (tmp_path / "results.jsonl").exists()
    assert "Model loaded" not in result.output


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("adapter", [False, True])
def test_infer_remote_code_flag_reaches_native_engine(
    tmp_path, monkeypatch, native_boundary, requested, adapter,
):
    args = _infer_args(tmp_path, monkeypatch)
    if adapter:
        path, config = _metadata(tmp_path)
        config["model"] = str(tmp_path / "base")
        (path / "adapter_config.json").write_text(json.dumps(config))
        args += ["--model", str(path)]
    if requested:
        args += ["--trust-remote-code"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    target, tokenizer_config, architecture_trust = native_boundary.calls[-1]
    assert target == (str(tmp_path / "base") if adapter else "base")
    assert tokenizer_config == {"trust_remote_code": requested}
    assert architecture_trust is requested
    assert ("REMOTE CODE WARNING" in result.output) is requested


@pytest.mark.parametrize("adapter", [False, True])
def test_native_remote_code_policy_checks_actual_base_before_loading(
    tmp_path, native_boundary, adapter, capsys,
):
    from soup_cli.utils.mlx_generation import load_native_model

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"auto_map": {"AutoConfig": "custom.Cfg"}}))
    target = base
    if adapter:
        target, config = _metadata(tmp_path)
        config["model"] = str(base)
        (target / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="requires trust_remote_code"):
        load_native_model(str(target))
    assert native_boundary.calls == []
    load_native_model(str(target), trust_remote_code=True)
    assert native_boundary.calls[-1] == (str(base), {"trust_remote_code": True}, True)
    assert "REMOTE CODE WARNING" in capsys.readouterr().out


def test_infer_base_override_is_forwarded_and_mismatch_writes_nothing(
    tmp_path, monkeypatch, native_boundary,
):
    args = _infer_args(tmp_path, monkeypatch)
    path, config = _metadata(tmp_path)
    (path / "adapter_config.json").write_text(json.dumps(config))
    args += ["--model", str(path), "--base", "owner/base"]
    control = runner.invoke(app, args)
    assert control.exit_code == 0, control.output
    assert native_boundary.calls[-1][0] == "owner/base"
    (tmp_path / "results.jsonl").unlink()
    failed = runner.invoke(app, args + ["--base", "owner/wrong"])
    assert failed.exit_code == 1, failed.output
    assert "does not match" in failed.output
    assert not (tmp_path / "results.jsonl").exists()


@pytest.mark.parametrize("base_config", [
    {"vision_config": {"hidden_size": 4}},
    {"audio_config": {"hidden_size": 4}},
    {"architectures": ["LlamaForSequenceClassification"]},
    {"architectures": ["LlamaEmbeddingModel"]},
])
def test_native_loader_refuses_non_causal_bases(tmp_path, native_boundary, base_config):
    from soup_cli.utils.mlx_generation import load_native_model

    assert load_native_model(str(tmp_path))[0] is native_boundary.model
    native_boundary.config = base_config
    with pytest.raises(ValueError, match="text causal language models only"):
        load_native_model(str(tmp_path))


def test_native_loader_refuses_too_many_adapter_layers(tmp_path, native_boundary):
    from soup_cli.utils.mlx_generation import load_native_model

    path, config = _metadata(tmp_path)
    metadata = path / "adapter_config.json"
    metadata.write_text(json.dumps(config))
    assert load_native_model(str(path))[0] is native_boundary.model
    config["num_layers"] = 3
    metadata.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="num_layers exceeds"):
        load_native_model(str(path))


@pytest.mark.parametrize("kind", ["embedding", "switch", "8bit"])
def test_native_loader_refuses_unsupported_target_modules(tmp_path, native_boundary, kind):
    from soup_cli.utils.mlx_generation import load_native_model

    path, config = _metadata(tmp_path)
    (path / "adapter_config.json").write_text(json.dumps(config))
    assert load_native_model(str(path))[0] is native_boundary.model
    if kind == "8bit":
        native_boundary.target = native_boundary.modules["mlx.nn"].QuantizedLinear()
        native_boundary.target.bits = 8
    else:
        # These are non-linear engine types, not ordinary projections.
        native_boundary.target = type("Embedding" if kind == "embedding" else "SwitchLinear",
                                      (), {})()
    with pytest.raises(ValueError, match="dense or 4-bit linear"):
        load_native_model(str(path))


def test_native_loader_corrupt_weights_have_friendly_cli_error(
    tmp_path, monkeypatch, native_boundary,
):
    args = _infer_args(tmp_path, monkeypatch)
    path, config = _metadata(tmp_path)
    (path / "adapter_config.json").write_text(json.dumps(config))

    def corrupt(path):
        raise RuntimeError("truncated safetensors header")

    monkeypatch.setattr(native_boundary.modules["mlx.core"], "load", corrupt)
    result = runner.invoke(app, args + ["--model", str(path)])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Cannot read native MLX adapter weights" in result.output
    assert not (tmp_path / "results.jsonl").exists()


def test_native_missing_dependency_has_install_hint(tmp_path, monkeypatch):
    import builtins

    from tests.conftest import strip_ansi

    args = _infer_args(tmp_path, monkeypatch)
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "mlx.core":
            raise ModuleNotFoundError("No module named 'mlx'", name="mlx")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "soup-cli[mlx]" in " ".join(strip_ansi(result.output).split())
    assert not (tmp_path / "results.jsonl").exists()


@pytest.mark.parametrize("failure", ["empty", "template", "max_tokens", "temperature"])
def test_generation_refusals_are_reachable(tmp_path, native_boundary, failure):
    from soup_cli.utils.mlx_generation import generate_native

    class Tokenizer:
        chat_template = "template"
        failure = None

        def apply_chat_template(self, *args, **kwargs):
            if self.failure == "template":
                raise ValueError("template is broken")
            return "user alpha assistant"

        def encode(self, *args, **kwargs):
            return [] if self.failure == "empty" else [1, 2]

    options = {"max_tokens": 0} if failure == "max_tokens" else (
        {"temperature": -1} if failure == "temperature" else {}
    )
    diagnostic = {"empty": "nonempty tokenized prompt", "template": "template is broken",
                  "max_tokens": "max_tokens", "temperature": "temperature"}[failure]
    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "alpha"}]
    assert generate_native(None, tokenizer, messages) == ("done", 1)
    tokenizer.failure = failure
    with pytest.raises(ValueError, match=diagnostic):
        generate_native(None, tokenizer, messages, **options)


def test_mocked_native_infer_keeps_heavy_dependencies_unimported(tmp_path):
    (tmp_path / "base").mkdir()
    (tmp_path / "prompts.jsonl").write_text('{"prompt":"hello"}\n')
    script = '''
import json, sys
from typer.testing import CliRunner
from soup_cli.cli import app
from soup_cli.utils import mlx_generation
heavy = {"torch", "peft", "trl", "transformers"}
assert not heavy.intersection(sys.modules)
mlx_generation.load_native_model = lambda *a, **k: (object(), object())
mlx_generation.generate_native = lambda *a, **k: ("answer", 1)
result = CliRunner().invoke(app, ["infer", "--backend", "mlx", "--model", "base",
                                "--input", "prompts.jsonl", "--output", "results.jsonl"])
assert result.exit_code == 0, (result.output, repr(result.exception))
assert json.load(open("results.jsonl")) == {
    "prompt": "hello", "response": "answer", "tokens_generated": 1}
assert not heavy.intersection(sys.modules), sorted(heavy.intersection(sys.modules))
'''
    import os

    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    completed = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                               text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("options, said, not_said", [
    ({"task": "asr"}, "--task text", ("--batch-size", "--cuda-graphs")),
    ({"batch_size": 2}, "--batch-size 1", ("--task", "--cuda-graphs")),
    ({"cuda_graphs": True}, "--cuda-graphs", ("--task", "--batch-size")),
])
def test_native_option_refusal_names_only_the_option_that_failed(options, said, not_said):
    from soup_cli.utils.mlx_generation import validate_native_options

    valid = dict(temperature=0.7, max_tokens=10, device=None, hub="hf")
    with pytest.raises(ValueError) as caught:
        validate_native_options(**(valid | options))
    message = str(caught.value)
    assert said in message
    for other in not_said:
        assert other not in message
