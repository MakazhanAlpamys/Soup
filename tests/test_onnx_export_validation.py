"""An ONNX completion message must not hide a requested numerical gate failure."""

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import typer
from typer.testing import CliRunner

from tests.conftest import strip_ansi


def _checker():
    return importlib.import_module("soup_cli.eval.onnx_check")


def test_help_explains_validation_codes_and_the_tolerance_policy():
    from soup_cli.cli import app

    result = CliRunner().invoke(app, ["export", "--help"], terminal_width=160)
    text = " ".join(strip_ansi(result.output).replace("│", " ").split())
    assert result.exit_code == 0, result.output
    assert "0=PASS" in text
    assert "2=numerical FAIL" in text
    assert "3=unusable validation" in text
    assert "option conflicts also use 2" in text
    assert "1e-4" in text and "FP32 smoke check" in text
    assert "not a quality threshold" in text


def _fake_exporter(monkeypatch, action=None):
    module = ModuleType("optimum.exporters.onnx")
    module.main_export = Mock(side_effect=action)
    monkeypatch.setitem(sys.modules, "optimum.exporters.onnx", module)
    return module.main_export


def _invoke(tmp_path, extra):
    from soup_cli.cli import app

    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    return CliRunner().invoke(app, ["export", "--model", str(model), "--format", "onnx",
                                    *extra])


def test_legacy_export_does_not_load_validation_deps(tmp_path, monkeypatch):
    exporter = _fake_exporter(monkeypatch)
    result = _invoke(tmp_path, [])
    assert result.exit_code == 0, result.output
    assert "ONNX Export Complete" in result.output
    assert exporter.call_args.kwargs == {
        "model_name_or_path": str(tmp_path / "model"),
        "output": str(tmp_path / "model_onnx"), "task": "text-generation",
    }


@pytest.mark.parametrize("passed,code", [(True, 0), (False, 2)])
def test_requested_gate_controls_completion_and_registry(tmp_path, monkeypatch, passed, code):
    export = importlib.import_module("soup_cli.commands.export")
    checker = _checker()
    exporter = _fake_exporter(monkeypatch)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("Hello world\n", encoding="utf-8")
    observed = {}

    def validate(source, output, texts, *, atol, trust_remote_code):
        observed.update(source=source, output=output, texts=texts, atol=atol,
                        trust=trust_remote_code)
        return checker.OnnxValidationResult(0.0 if passed else 1.0, 0.0 if passed else 0.25,
                                            8, 1, atol)

    monkeypatch.setattr(checker, "validate_onnx_export", validate)
    attach = Mock()
    monkeypatch.setattr(export, "_maybe_attach_export", attach)
    result = _invoke(tmp_path, ["--validate-onnx", "--prompts", str(prompts), "--atol", "0.001"])
    assert result.exit_code == code, result.output
    assert ("ONNX Export Complete" in result.output) is passed
    assert ("PASS" if passed else "FAIL") in result.output
    assert "0.001" in strip_ansi(result.output)
    assert "mean" in result.output.lower() and "max" in result.output.lower()
    assert attach.call_count == int(passed)
    assert exporter.call_count == 1
    assert observed == {"source": tmp_path / "model", "output": tmp_path / "model_onnx",
                        "texts": ["Hello world"], "atol": 0.001, "trust": False}


@pytest.mark.parametrize("args", [
    ["--validate-onnx"], ["--atol", "0.1"], ["--prompts", "missing.txt"],
    ["--validate-onnx", "--prompts", "missing.txt"],
    ["--validate-onnx", "--onnx-task", "feature-extraction", "--prompts", "missing.txt"],
])
def test_bad_validation_options_stop_before_export(tmp_path, monkeypatch, args):
    exporter = _fake_exporter(monkeypatch)
    result = _invoke(tmp_path, args)
    assert result.exit_code == 3, result.output
    exporter.assert_not_called()
    assert "ONNX Export Complete" not in result.output


@pytest.mark.parametrize("flag", ["--verbose", "--no-install-deps"])
@pytest.mark.parametrize("validation", [False, True])
def test_legacy_option_conflict_exit_two_is_not_a_numerical_fail(
    tmp_path, monkeypatch, flag, validation,
):
    exporter = _fake_exporter(monkeypatch)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("Hi\n", encoding="utf-8")
    args = [flag]
    if validation:
        args.extend(["--validate-onnx", "--prompts", str(prompts)])
    result = _invoke(tmp_path, args)
    assert result.exit_code == 2, result.output
    assert "require --format gguf" in " ".join(strip_ansi(result.output).split())
    assert "ONNX Logits Validation" not in result.output
    assert "FAIL" not in result.output
    exporter.assert_not_called()


@pytest.mark.parametrize("atol", ["nan", "inf", "-1"])
def test_invalid_tolerance_stops_before_export(tmp_path, monkeypatch, atol):
    exporter = _fake_exporter(monkeypatch)
    path = tmp_path / "prompts.txt"
    path.write_text("Hi\n", encoding="utf-8")
    result = _invoke(tmp_path, ["--validate-onnx", "--prompts", str(path), "--atol", atol])
    assert result.exit_code == 3, result.output
    exporter.assert_not_called()


def test_validation_options_are_onnx_only(tmp_path, monkeypatch):
    from soup_cli.cli import app

    export = importlib.import_module("soup_cli.commands.export")
    backend = Mock()
    monkeypatch.setattr(export, "_export_tensorrt", backend)
    model = tmp_path / "model"
    model.mkdir()
    result = CliRunner().invoke(app, ["export", "--model", str(model), "--format", "tensorrt",
                                      "--validate-onnx", "--prompts", "missing.txt"])
    assert result.exit_code == 3, result.output
    backend.assert_not_called()


@pytest.mark.parametrize("failure", [False, True, "error"])
def test_merged_reference_lives_until_validation_and_is_cleaned(tmp_path, monkeypatch, failure):
    export = importlib.import_module("soup_cli.commands.export")
    checker = _checker()
    _fake_exporter(monkeypatch)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path": "local-base"}')
    merged = tmp_path / "merged"
    merged.mkdir()
    monkeypatch.setattr(export, "_merge_dir_for", lambda _: (merged, True))
    monkeypatch.setattr(export, "_merge_adapter", Mock())

    def validate(source, output, prompts, **kwargs):
        assert source == merged and merged.exists()
        if failure == "error":
            raise checker.OnnxValidationError("unsupported graph")
        return checker.OnnxValidationResult(float(failure), float(failure), 1, 1, kwargs["atol"])

    monkeypatch.setattr(checker, "validate_onnx_export", validate)
    with pytest.raises(typer.Exit) if failure else _no_exception():
        export._export_onnx(adapter, str(tmp_path / "out"), None,
                            validation_prompts=["Hello"], validation_atol=1e-4)
    assert not merged.exists()


def _no_exception():
    from contextlib import nullcontext

    return nullcontext()


def test_error_statistics_are_element_weighted_and_boundary_is_inclusive():
    np = pytest.importorskip("numpy")
    checker = _checker()
    pairs = [(np.zeros(3), np.full(3, 0.125)), (np.zeros(1), np.full(1, 0.5))]
    result = checker.compare_logits(pairs, atol=0.5)
    assert result.max_abs_error == 0.5 and result.passed
    assert result.mean_abs_error == 0.21875
    assert result.elements == 4 and result.prompts == 2
    assert not checker.compare_logits(pairs, atol=0.499).passed


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_cannot_pass(value):
    np = pytest.importorskip("numpy")
    checker = _checker()
    assert not checker.compare_logits([(np.array([value]), np.array([value]))], atol=1).passed


def test_empty_or_mismatched_logits_are_not_a_pass():
    np = pytest.importorskip("numpy")
    checker = _checker()
    for pairs in ([], [(np.zeros(1), np.zeros(2))], [(np.zeros(0), np.zeros(0))]):
        with pytest.raises(checker.OnnxValidationError):
            checker.compare_logits(pairs, atol=1e-4)


def test_prompt_files_preserve_json_text_and_reject_bad_records(tmp_path):
    checker = _checker()
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"prompt": "  Keep spaces  "}) + "\n", encoding="utf-8")
    assert checker.load_validation_prompts(path) == ["  Keep spaces  "]
    for text in ('{"prompt": 42}', '{"wrong": "Hi"}', '{bad', '{"prompt": " "}', ""):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(checker.OnnxValidationError):
            checker.load_validation_prompts(path)


def test_prompt_file_size_and_count_are_bounded(tmp_path):
    checker = _checker()
    path = tmp_path / "prompts.txt"
    path.write_bytes(b"x" * (checker.MAX_PROMPT_BYTES + 1))
    with pytest.raises(checker.OnnxValidationError):
        checker.load_validation_prompts(path)
    path.write_text("Hi\n" * (checker.MAX_PROMPTS + 1), encoding="utf-8")
    with pytest.raises(checker.OnnxValidationError):
        checker.load_validation_prompts(path)


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    """Exercise graph validation and shared token inputs without installing ONNX."""
    import numpy as np
    import torch

    output = tmp_path / "onnx"
    output.mkdir()
    (output / "model.onnx").touch()
    names = ["input_ids", "attention_mask", "position_ids", "token_type_ids"]
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name=n, type="tensor(int64)") for n in names]
    session.get_outputs.return_value = [SimpleNamespace(name="logits", type="tensor(float)")]
    session.run.return_value = [np.zeros((1, 2, 6), dtype=np.float32)]
    runtime = ModuleType("onnxruntime")
    runtime.SessionOptions = SimpleNamespace
    runtime.InferenceSession = Mock(return_value=session)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    tokenizer = Mock(return_value={"input_ids": torch.tensor([[1, 2]]),
                                  "attention_mask": torch.tensor([[1, 1]]),
                                  "token_type_ids": torch.tensor([[0, 0]])})
    model = Mock(return_value=SimpleNamespace(logits=torch.zeros((1, 2, 6))))
    model.cpu.return_value = model.eval.return_value = model
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=Mock(return_value=model))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return output, session, model, tokenizer, runtime, transformers


def test_runtime_uses_cpu_float32_and_identical_tokens(fake_runtime, tmp_path):
    import numpy as np
    import torch

    output, session, model, tokenizer, runtime, transformers = fake_runtime
    source = tmp_path / "source"
    result = _checker().validate_onnx_export(source, output, ["Hello"], atol=1e-4,
                                           trust_remote_code=True)
    assert result.passed and result.elements == 12
    kwargs = transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
    assert kwargs == {"local_files_only": True, "trust_remote_code": True,
                      "dtype": torch.float32, "attn_implementation": "eager"}
    assert runtime.InferenceSession.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    assert tokenizer.call_args.kwargs["truncation"] is False
    assert model.call_args.kwargs["use_cache"] is False
    feed = session.run.call_args.kwargs["input_feed"]
    for name in feed:
        np.testing.assert_array_equal(feed[name], model.call_args.kwargs[name].numpy())


@pytest.mark.parametrize("kind", ["past", "no_ids", "float_input", "no_logits", "fp16_output"])
def test_unknown_graph_layouts_are_refused_before_model_load(fake_runtime, tmp_path, kind):
    output, session, _, _, _, transformers = fake_runtime
    if kind == "past":
        session.get_inputs.return_value.append(SimpleNamespace(name="past_key_values.0",
                                                               type="tensor(float)"))
    elif kind == "no_ids":
        session.get_inputs.return_value = []
    elif kind == "float_input":
        session.get_inputs.return_value[0].type = "tensor(float)"
    elif kind == "no_logits":
        session.get_outputs.return_value = []
    else:
        session.get_outputs.return_value[0].type = "tensor(float16)"
    with pytest.raises(_checker().OnnxValidationError):
        _checker().validate_onnx_export(tmp_path, output, ["Hi"], atol=1e-4)
    transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()


def test_missing_or_multiple_graphs_cannot_pass(fake_runtime, tmp_path):
    output, _, _, _, _, _ = fake_runtime
    checker = _checker()
    (output / "model.onnx").unlink()
    with pytest.raises(checker.OnnxValidationError):
        checker.validate_onnx_export(tmp_path, output, ["Hi"], atol=1e-4)
    (output / "first.onnx").touch()
    (output / "second.onnx").touch()
    with pytest.raises(checker.OnnxValidationError):
        checker.validate_onnx_export(tmp_path, output, ["Hi"], atol=1e-4)


@pytest.mark.parametrize("error", [ImportError("onnxruntime missing"), ValueError("bad graph")])
def test_validation_failure_is_exit_three_and_never_registered(tmp_path, monkeypatch, error):
    checker = _checker()
    export = importlib.import_module("soup_cli.commands.export")
    _fake_exporter(monkeypatch)
    monkeypatch.setattr(checker, "validate_onnx_export", Mock(side_effect=error))
    attach = Mock()
    monkeypatch.setattr(export, "_maybe_attach_export", attach)
    path = tmp_path / "prompts.txt"
    path.write_text("Hi\n", encoding="utf-8")
    result = _invoke(tmp_path, ["--validate-onnx", "--prompts", str(path)])
    assert result.exit_code == 3, result.output
    assert "ONNX Export Complete" not in result.output
    attach.assert_not_called()


@pytest.fixture(scope="module")
def real_export(tmp_path_factory):
    """No network: a tiny local GPT-2, a word tokenizer and a real ONNX graph.

    Use torch's exporter to test the checker independently of optimum's separately
    versioned exporter. Optimum's currently published ONNX extra requires
    transformers<4.58, outside Soup's >=5.16.1 range; do not conceal that conflict
    by monkeypatching its missing transformers functions.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("real-onnx")
    source, output = root / "source", root / "export"
    raw = Tokenizer(WordLevel({"<unk>": 0, "Hello": 1, "world": 2, "a": 3, "test": 4,
                               "<eos>": 5}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>",
                                        eos_token="<eos>", pad_token="<eos>")
    with torch.random.fork_rng():
        torch.manual_seed(2676)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=6, n_positions=1024, n_embd=16,
                                          n_layer=1, n_head=2, use_cache=False,
                                          bos_token_id=5, eos_token_id=5, pad_token_id=5))
        model.save_pretrained(source)
    tokenizer.save_pretrained(source)

    class Logits(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, input_ids, attention_mask, position_ids):
            return self.core(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, use_cache=False).logits

    output.mkdir()
    model.config._attn_implementation = "eager"
    ids = torch.tensor([[1, 2, 3]])
    torch.onnx.export(
        Logits(model.eval()), (ids, torch.ones_like(ids), torch.arange(3).unsqueeze(0)),
        str(output / "model.onnx"), input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["logits"], opset_version=18, dynamo=False,
        dynamic_axes={"input_ids": {1: "sequence"}, "attention_mask": {1: "sequence"},
                      "position_ids": {1: "sequence"}, "logits": {1: "sequence"}},
    )
    return source, output


def test_real_export_matches_source_on_explicit_prompts(real_export):
    source, output = real_export
    result = _checker().validate_onnx_export(source, output, ["Hello world", "a test"], atol=1e-4)
    assert result.passed and result.elements == 24 and result.prompts == 2


def test_real_corrupted_initializer_fails_the_cli_gate(real_export, tmp_path, monkeypatch):
    """Exporter returns successfully, but the on-disk graph is wrong: exit 2, no success."""
    import shutil

    import numpy as np
    import onnx
    from onnx import numpy_helper

    source, output = real_export
    corrupted = tmp_path / "corrupted"
    shutil.copytree(output, corrupted)
    graph_path = next(corrupted.glob("*.onnx"))
    graph = onnx.load(graph_path)
    changed = False
    for tensor in graph.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT:
            values = numpy_helper.to_array(tensor).copy()
            values += np.float32(0.5)
            tensor.CopyFrom(numpy_helper.from_array(values, name=tensor.name))
            changed = True
    assert changed
    onnx.save(graph, graph_path)
    _fake_exporter(monkeypatch)
    export = importlib.import_module("soup_cli.commands.export")
    attach = Mock()
    monkeypatch.setattr(export, "_maybe_attach_export", attach)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("Hello world\na test\n", encoding="utf-8")
    from soup_cli.cli import app

    result = CliRunner().invoke(app, ["export", "--model", str(source), "--format", "onnx",
                                      "--output", str(corrupted), "--validate-onnx",
                                      "--prompts", str(prompts)])
    assert result.exit_code == 2, result.output
    assert "FAIL" in result.output and "ONNX Export Complete" not in result.output
    attach.assert_not_called()


def test_overlong_prompt_is_refused_without_truncation(real_export):
    source, output = real_export
    checker = _checker()
    with pytest.raises(checker.OnnxValidationError, match="512"):
        checker.validate_onnx_export(source, output, ["Hello " * 513], atol=1e-4)
