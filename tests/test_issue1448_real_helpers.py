"""#1448: `soup export --registry-id` with the real helpers; only the converters are stubbed.

The helpers' own ``return output_path`` is what hands the exported path to the registry, so
nothing here replaces a helper: each test stubs the third-party converter one level below it and
checks what the registry recorded.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.registry.store import RegistryStore

runner = CliRunner()


def _write(path, size=10):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _quantizer(module_name, cls_name, saved_name):
    class _Model:
        def quantize(self, *args, **kwargs):
            return None

        def save_quantized(self, path):
            _write(Path(path) / saved_name, 3000)

    return {
        module_name: _module(
            module_name,
            BaseQuantizeConfig=lambda **kwargs: None,
            **{cls_name: types.SimpleNamespace(from_pretrained=lambda *a, **k: _Model())},
        )
    }


def _gptq_modules():
    return _quantizer("auto_gptq", "AutoGPTQForCausalLM", "gptq_model-4bit-128g.safetensors")


def _awq_modules():
    return _quantizer("awq", "AutoAWQForCausalLM", "model.safetensors")


def _onnx_modules():
    def main_export(model_name_or_path, output, task):
        _write(Path(output) / "model.onnx", 100)
        _write(Path(output) / "model.onnx_data", 5000)

    return {
        "optimum": _module("optimum"),
        "optimum.exporters": _module("optimum.exporters"),
        "optimum.exporters.onnx": _module("optimum.exporters.onnx", main_export=main_export),
    }


def _tensorrt_modules():
    names = ("tensorrt_llm", "tensorrt_llm.commands", "tensorrt_llm.commands.convert_checkpoint")
    return {name: _module(name) for name in names}


def _fake_run(argv, **_kwargs):
    # convert_checkpoint fills <out>/checkpoint, trtllm-build fills <out>/engine
    out = Path(argv[argv.index("--output_dir") + 1])
    _write(out / ("rank0.engine" if argv[0] == "trtllm-build" else "rank0.safetensors"), 4000)
    return subprocess.CompletedProcess(argv, 0, "", "")


class _Tokenizer:
    def __call__(self, *args, **kwargs):
        return {}

    def save_pretrained(self, path):
        _write(Path(path) / "tokenizer.json")


def _gguf(*, output_path, **_kwargs):
    _write(output_path, 2000)


def _torchao(*, output_dir, **_kwargs):
    _write(Path(output_dir) / "pytorch_model.bin", 3000)


# format -> (extra CLI args, third-party modules to stub, what the registry must hold)
CASES = {
    "onnx": ([], _onnx_modules,
             [("onnx", "model.onnx"), ("onnx", "model.onnx_data")]),
    "tensorrt": ([], _tensorrt_modules,
                 [("tensorrt", "rank0.engine")]),
    "awq": (["--calibration-data", "calib.jsonl"], _awq_modules,
            [("awq", "model.safetensors")]),
    "gptq": (["--calibration-data", "calib.jsonl"], _gptq_modules,
             [("gptq", "model.safetensors")]),
    "torchao": (["--quant-config", "q.yaml"], dict,
                [("torchao", "pytorch_model.bin")]),
    "gguf-ud": (["--gguf-flavour", "Q4_0_4_4"], dict, [("gguf", "out.gguf")]),
    "bitnet": ([], dict, [("gguf", "out.gguf")]),
    "tq1_0": ([], dict, [("gguf", "out.gguf")]),
}


@pytest.mark.parametrize("fmt", sorted(CASES))
def test_every_format_attaches_what_its_helper_returned(tmp_path, monkeypatch, fmt):
    extra_args, modules, expected = CASES[fmt]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "calib.jsonl").write_text('{"text": "hello"}\n', encoding="utf-8")
    (tmp_path / "q.yaml").write_text("scheme: Int4WeightOnly\n", encoding="utf-8")
    with RegistryStore() as store:
        entry_id = store.push(
            name="m", tag="v1", base_model="org/base", task="sft", run_id=None, config={},
        )
    for name, module in modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    if "transformers" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            _module(
                "transformers",
                AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tokenizer()),
            ),
        )

    with patch.object(subprocess, "run", _fake_run), \
         patch("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: _Tokenizer()), \
         patch("soup_cli.utils.save_formats.export_torchao", _torchao), \
         patch("soup_cli.utils.gguf_quant.export_advanced_gguf", _gguf), \
         patch("soup_cli.utils.bitnet.export_bitnet_gguf", _gguf), \
         patch("soup_cli.commands.export._find_llama_cpp", return_value=Path("llama")):
        output = "out.gguf" if fmt in ("gguf-ud", "bitnet", "tq1_0") else f"out_{fmt}"
        result = runner.invoke(
            app,
            ["export", "--model", "model", "--format", fmt, "-o", output,
             "--registry-id", entry_id, *extra_args],
        )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    with RegistryStore() as store:
        rows = [(row["kind"], Path(row["path"]).name) for row in store.get_artifacts(entry_id)]
    assert sorted(rows) == sorted(expected)
