"""#3366: exercise ship dispatch, external loading, scoring and the existing stamp."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from soup_cli.commands import ship
from soup_cli.utils import live_eval
from tests.cli_runner import CliRunner


@pytest.fixture
def engines(tmp_path, monkeypatch):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    monkeypatch.chdir(tmp_path)
    Path("tasks.jsonl").write_text(
        '{"prompt":"say hi","expected":"hi","scoring":"contains"}\n'
    )
    state = SimpleNamespace(
        loads=[], live=[], wrappers=[], evaluations=[], attachments=[], tokenizers=[],
        version="0.4.13",
    )
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"

    def load(model_id, **options):
        model = MagicMock()
        model.to.return_value = model
        state.loads.append((model_id, options, model))
        return model

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load)

    def load_tokenizer(model_id, **kwargs):
        state.tokenizers.append(model_id)
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    import peft

    def attach(model, adapter, **kwargs):
        state.attachments.append((model, adapter))
        return model

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", attach)

    def generator(model_id, **options):
        state.live.append((model_id, options))
        live_eval.load_model_and_tokenizer(
            model_id, adapter=options.get("adapter"), device=options["device"],
            dtype=options["dtype"], quantization=options["quantization"],
            **({"quantization_config": options["quantization_config"]}
               if "quantization_config" in options else {}),
        )
        return lambda prompt: "hi" if model_id == "tuned" or options.get("adapter") else "bye"

    monkeypatch.setattr(live_eval, "make_generator", generator)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    harness = ModuleType("lm_eval")
    models = ModuleType("lm_eval.models")
    huggingface = ModuleType("lm_eval.models.huggingface")

    class HFLM:
        def __init__(self, **options):
            state.wrappers.append(options)

    def evaluate(**options):
        state.evaluations.append(options)
        return {"results": {"arc_easy": {"acc,none": 0.5}}}

    huggingface.HFLM = HFLM
    harness.simple_evaluate = evaluate
    for name, module in (
        ("lm_eval", harness), ("lm_eval.models", models),
        ("lm_eval.models.huggingface", huggingface),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version",
        lambda name: state.version if name == "lm-eval" else original_version(name),
    )
    return state


def invoke(*, quantization="none", device="cpu", adapter=False, extra=()):
    Path("soup.yaml").write_text(
        f"base: base\ndata:\n  train: tasks.jsonl\ntraining:\n  quantization: {quantization}\n"
    )
    arguments = [
        "--base", "base", "--adapter" if adapter else "--tuned",
        "adapter" if adapter else "tuned", "--task-eval", "tasks.jsonl",
        "--config", "soup.yaml", "--general-suite", "mini_mmlu,arc_easy",
        "--emit-evidence", "evidence.json", "--output", "verdict.json",
    ]
    if device is not None:
        arguments.extend(["--device", device])
    return CliRunner().invoke(ship.app, [*arguments, *extra])


@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.parametrize("quantization", ["none", "4bit", "8bit"])
def test_same_load_reaches_both_engines_and_stamp(engines, quantization, adapter):
    import torch

    result = invoke(quantization=quantization, adapter=adapter)
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(engines.loads) == 4
    assert engines.attachments == (
        [(engines.loads[1][2], "adapter"), (engines.loads[2][2], "adapter")] if adapter else []
    )
    assert [entry[0] for entry in engines.loads] == [
        "base", "base" if adapter else "tuned", "base" if adapter else "tuned", "base"
    ]
    for _, options, _ in engines.loads:
        if quantization == "none":
            assert options["dtype"] == "float32"
            assert "quantization_config" not in options
        else:
            assert "dtype" not in options
            config = options["quantization_config"]
            assert options["device_map"] == "cpu"
            assert config.load_in_4bit == (quantization == "4bit")
            assert config.load_in_8bit == (quantization == "8bit")
            if quantization == "4bit":
                assert config.bnb_4bit_quant_type == "nf4"
                assert config.bnb_4bit_compute_dtype == torch.float32
                assert config.bnb_4bit_use_double_quant is True
    assert len(engines.wrappers) == len(engines.evaluations) == 2
    assert [entry["pretrained"] for entry in engines.wrappers] == [
        engines.loads[2][2], engines.loads[3][2]
    ]
    assert all(entry["tokenizer"] is not None for entry in engines.wrappers)
    assert all(entry["batch_size"] == 1 for entry in engines.wrappers)
    assert all(entry["model"] != "hf" for entry in engines.evaluations)
    assert all(entry["model_args"] is None for entry in engines.evaluations)
    for path in ("evidence.json", "verdict.json"):
        assert json.loads(Path(path).read_text())["numerics"] == (
            "float32" if quantization == "none" else quantization
        )


@pytest.mark.parametrize("device", [None, "auto", "cuda:1"])
def test_device_and_precision_resolve_once_before_both_engines(engines, monkeypatch, device):
    from soup_cli.utils import gpu

    calls = []
    original = live_eval.resolve_device

    def resolve(requested):
        calls.append(requested)
        return "cuda:1" if requested in (None, "auto") else original(requested)

    import torch

    monkeypatch.setattr(live_eval, "resolve_device", resolve)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(gpu, "bf16_fp16_flags", lambda selected: (False, True))
    result = invoke(device=device)
    assert result.exit_code == 0, (result.output, result.exception)
    if device in (None, "auto"):
        assert calls.count(device) == 1
    assert all(selected == "cuda:1" for selected in calls[1:])
    assert all(options["device"] == "cuda:1" for _, options in engines.live)
    assert all(options["dtype"] == "float16" for _, options, _ in engines.loads)
    for _, _, model in engines.loads:
        model.to.assert_called_once_with("cuda:1")
    assert all(options["device"] == "cuda:1" for options in engines.evaluations)
    assert json.loads(Path("evidence.json").read_text())["numerics"] == "float16"


@pytest.mark.parametrize(
    "device,quantization,version,diagnostic",
    [
        ("cpu", "none", "0.4.0", "lm-eval>=0.4.1"),
        ("nonsense", "none", "0.4.13", "device"),
        ("meta", "none", "0.4.13", "device"),
        ("cuda:3", "none", "0.4.13", "device"),
        ("cuda", "none", "0.4.13", "device"),
        ("mps", "4bit", "0.4.13", "quantization"),
        ("mps:1", "none", "0.4.13", "device"),
    ],
)
def test_preflight_refuses_before_either_engine_allocates(
    engines, device, quantization, version, diagnostic,
):
    engines.version = version
    result = invoke(device=device, quantization=quantization)
    assert result.exit_code == 3, (result.output, result.exception)
    assert diagnostic in result.output
    assert engines.loads == engines.live == engines.wrappers == engines.tokenizers == []
    assert not Path("evidence.json").exists()


def test_missing_external_dependency_refuses_before_live_allocation(engines, monkeypatch):
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", None)
    result = invoke()
    assert result.exit_code == 3, (result.output, result.exception)
    assert "lm-eval" in result.output
    assert engines.loads == []


def test_external_binary_import_error_refuses_before_allocation(engines, monkeypatch):
    import builtins

    original = builtins.__import__

    def fail_import(name, *args, **kwargs):
        if name == "lm_eval.models.huggingface":
            raise OSError("lm-eval binary dependency unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    result = invoke()
    assert result.exit_code == 3, (result.output, result.exception)
    assert "lm-eval binary dependency unavailable" in result.output
    assert engines.loads == engines.tokenizers == []


@pytest.mark.parametrize("dependency,quantization,adapter", [
    ("peft", "none", True), ("bitsandbytes", "4bit", False),
])
def test_missing_load_dependency_refuses_before_allocation(
    engines, monkeypatch, dependency, quantization, adapter,
):
    monkeypatch.setitem(sys.modules, dependency, None)
    result = invoke(quantization=quantization, adapter=adapter)
    assert result.exit_code == 3, (result.output, result.exception)
    assert dependency.lower() in result.output.lower()
    assert engines.loads == []


@pytest.mark.parametrize("bf16", [False, True])
def test_indexed_nf4_compute_is_frozen_for_both_engines(engines, monkeypatch, bf16):
    import torch

    from soup_cli.utils import gpu

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(gpu, "get_compute_dtype", lambda: torch.float16 if bf16 else torch.bfloat16)
    probes = []

    def flags(device):
        probes.append(device)
        return bf16, not bf16

    monkeypatch.setattr(gpu, "bf16_fp16_flags", flags)
    result = invoke(quantization="4bit", device="cuda:1")
    assert result.exit_code == 0, (result.output, result.exception)
    configs = [options["quantization_config"] for _, options, _ in engines.loads]
    assert len(configs) == 4 and all(config is configs[0] for config in configs)
    assert configs[0].bnb_4bit_compute_dtype == (torch.bfloat16 if bf16 else torch.float16)
    assert probes == ["cuda:1"]


@pytest.mark.parametrize("requested,kind", [
    ("8bit", "4bit"), ("4bit", "8bit"), ("8bit", "none"), (None, "none"),
    ("4bit", "object"),
])
def test_preflighted_quantization_family_is_validated_before_tokenizer(engines, requested, kind):
    from transformers import BitsAndBytesConfig

    with pytest.raises(ValueError, match="quantization_config must match"):
        live_eval.load_model_and_tokenizer(
            "base", quantization=requested,
            quantization_config=object() if kind == "object" else BitsAndBytesConfig(
                load_in_4bit=kind == "4bit", load_in_8bit=kind == "8bit",
            ),
        )
    assert engines.loads == engines.tokenizers == []


@pytest.mark.parametrize("device", ["cuda:3", "mps"])
def test_unavailable_device_branches_refuse_before_allocation(engines, monkeypatch, device):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    result = invoke(device=device)
    assert result.exit_code == 3, (result.output, result.exception)
    assert "unavailable" in result.output
    assert engines.loads == engines.tokenizers == []


def test_verdict_never_re_resolves_precision_after_loading(engines, monkeypatch):
    calls = []

    def numerics(*args):
        calls.append(args)
        return "float32" if len(calls) == 1 else "float16"

    monkeypatch.setattr(ship, "_live_eval_numerics", numerics)
    result = invoke()
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == [(None, "cpu")]
    assert json.loads(Path("evidence.json").read_text())["numerics"] == "float32"


def test_missing_external_scores_still_refuse(engines, monkeypatch):
    monkeypatch.setattr(sys.modules["lm_eval"], "simple_evaluate", lambda **kwargs: {"results": {}})
    result = invoke()
    assert result.exit_code == 1, (result.output, result.exception)
    assert "could not score benchmark" in result.output
    assert not Path("evidence.json").exists()


def test_model_argument_injection_is_still_refused_before_load(engines):
    result = invoke(extra=("--tuned", "model,dtype=float16"))
    assert result.exit_code == 3, (result.output, result.exception)
    assert "--tuned" in result.output
    assert engines.loads == []


def test_external_base_baseline_skips_only_base_load(engines):
    Path("baseline.json").write_text('{"arc_easy": 0.5}')
    result = invoke(extra=("--baseline", "baseline.json"))
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(engines.loads) == 3
    assert len(engines.wrappers) == len(engines.evaluations) == 1
