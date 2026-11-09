"""Live ship precision follows the requested GPU and survives evidence replay."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi


class _CudaCards:
    """Two different cards; emulated bf16 deliberately gives a false positive."""

    def __init__(self, majors=(8, 7), *, available=True, legacy=False):
        self.majors = majors
        self.available = available
        self.legacy = legacy
        self.current = 0
        self.probes = []
        self.fail_probe = False

    def is_available(self):
        return self.available

    @contextmanager
    def device(self, device):
        previous = self.current
        self.current = int(str(device).split(":")[-1]) if ":" in str(device) else previous
        try:
            yield
        finally:
            self.current = previous

    def is_bf16_supported(self, including_emulation=True):
        self.probes.append((self.current, including_emulation))
        if self.fail_probe:
            raise RuntimeError("unavailable capability probe")
        if self.legacy:
            raise TypeError("unexpected keyword argument 'including_emulation'")
        return including_emulation or self.majors[self.current] >= 8

    def get_device_capability(self, device=None):
        index = self.current if device is None else int(str(device).split(":")[-1])
        return self.majors[index], 0


@pytest.fixture
def cards(monkeypatch):
    import torch

    def install(majors=(8, 7), *, available=True, legacy=False):
        cuda = _CudaCards(majors, available=available, legacy=legacy)
        monkeypatch.setattr(torch, "cuda", cuda)
        return cuda

    return install


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.mark.parametrize("majors, expected", [((8, 7), True), ((7, 8), False)])
def test_unqualified_capability_keeps_current_device_semantics(cards, majors, expected):
    from soup_cli.utils.gpu import cuda_supports_bf16

    cuda = cards(majors)
    assert cuda_supports_bf16() is expected
    assert cuda.probes == [(0, False)]
    assert cuda.current == 0


@pytest.mark.parametrize("legacy", [False, True], ids=["native-probe", "older-torch"])
@pytest.mark.parametrize("majors, expected", [((8, 7), False), ((7, 8), True)])
def test_indexed_capability_uses_requested_card_and_restores_current(
    cards, majors, expected, legacy,
):
    from soup_cli.utils.gpu import bf16_fp16_flags, cuda_supports_bf16

    cuda = cards(majors, legacy=legacy)
    assert cuda_supports_bf16("cuda:1") is expected
    assert cuda.current == 0
    assert bf16_fp16_flags("cuda:1") == (expected, not expected)
    assert cuda.current == 0
    assert all(index == 1 and emulation is False for index, emulation in cuda.probes)


def test_indexed_probe_restores_current_device_after_runtime_error(cards):
    from soup_cli.utils.gpu import cuda_supports_bf16

    cuda = cards()
    cuda.fail_probe = True
    assert cuda_supports_bf16("cuda:1") is False
    assert cuda.current == 0


@pytest.mark.parametrize(
    "device, majors, available, expected",
    [
        ("cuda", (7, 8), True, "float16"),
        ("cuda", (8, 7), True, "bfloat16"),
        ("cuda:1", (8, 7), True, "float16"),
        ("cuda:1", (7, 8), True, "bfloat16"),
        ("cpu", (8, 7), True, "float32"),
        ("mps", (8, 7), True, "float32"),
        ("auto", (7, 8), True, "float16"),
        (None, (8, 7), True, "bfloat16"),
        ("auto", (8, 7), False, "float32"),
        ("cuda", (8, 7), False, "float32"),
        ("cuda:1", (8, 7), False, "float32"),
    ],
)
def test_live_dtype_respects_runtime_and_resolved_card(cards, device, majors, available, expected):
    from soup_cli.commands.ship import _live_eval_numerics

    cuda = cards(majors, available=available)
    assert _live_eval_numerics(None, device) == expected
    assert cuda.current == 0
    if expected == "float32":
        assert cuda.probes == []
    else:
        assert cuda.probes and all(emulation is False for _, emulation in cuda.probes)


@pytest.mark.parametrize("quantization", ["4bit", "8bit"])
def test_quantized_stamp_never_probes_cuda(cards, quantization):
    from soup_cli.commands.ship import _live_eval_numerics

    cuda = cards()
    cuda.fail_probe = True
    assert _live_eval_numerics(quantization, "cuda:1") == quantization
    assert cuda.probes == []


def test_explicit_cuda_without_torch_uses_float32(tmp_path):
    script = """
import sys
sys.modules['torch'] = None
from soup_cli.commands.ship import _live_eval_numerics
from soup_cli.utils.gpu import bf16_fp16_flags, cuda_supports_bf16
assert _live_eval_numerics(None, 'cuda:1') == 'float32'
assert bf16_fp16_flags('cuda:1') == (False, False)
assert cuda_supports_bf16() is False
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("adapter", [False, True], ids=["separate-tuned", "adapter"])
@pytest.mark.parametrize(
    "device, majors, available, expected",
    [
        ("cuda:1", (8, 7), True, "float16"),
        ("cuda:1", (7, 8), True, "bfloat16"),
        ("cpu", (8, 7), True, "float32"),
        ("cuda", (8, 7), False, "float32"),
    ],
)
def test_dtype_reaches_both_from_pretrained_calls(
    cards, monkeypatch, device, majors, available, expected, adapter,
):
    from soup_cli.commands.ship import _resolve_generators

    # Mock only the external model/tokenizer constructors, keeping Soup's
    # make_generator and load_model_and_tokenizer paths real.
    transformers = ModuleType("transformers")
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    model = MagicMock()
    model.to.return_value = model
    load_model = MagicMock(return_value=model)
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer))
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    peft = ModuleType("peft")
    attach_adapter = MagicMock(return_value=model)
    peft.PeftModel = SimpleNamespace(from_pretrained=attach_adapter)
    monkeypatch.setitem(sys.modules, "peft", peft)
    cuda = cards(majors, available=available)

    _resolve_generators(
        "owner/base", None if adapter else "owner/tuned",
        "owner/adapter" if adapter else None, device,
    )

    assert load_model.call_count == 2
    assert [call.kwargs["dtype"] for call in load_model.call_args_list] == [expected, expected]
    assert [call.args[0] for call in load_model.call_args_list] == [
        "owner/base", "owner/base" if adapter else "owner/tuned",
    ]
    assert all("quantization_config" not in call.kwargs for call in load_model.call_args_list)
    assert attach_adapter.call_count == int(adapter)
    assert cuda.current == 0


@pytest.mark.parametrize("color", [False, True], ids=["plain", "ansi"])
@pytest.mark.parametrize(
    "device, majors, available, expected",
    [
        ("cuda:1", (8, 7), True, "float16"),
        ("cuda:1", (7, 8), True, "bfloat16"),
        ("cpu", (8, 7), True, "float32"),
        ("auto", (7, 8), True, "float16"),
        ("cuda", (8, 7), False, "float32"),
    ],
)
def test_console_verdict_emission_and_replay_agree(
    cards, monkeypatch, tmp_path, device, majors, available, expected, color,
):
    from soup_cli.commands import ship
    from soup_cli.utils import live_eval

    cuda = cards(majors, available=available)
    calls = []

    def make_generator(model, **kwargs):
        calls.append((model, kwargs))
        return lambda _: "yes" if model == "owner/tuned" else "no"

    monkeypatch.setattr(live_eval, "make_generator", make_generator)
    monkeypatch.setattr(ship, "console", Console(force_terminal=color, width=80))
    monkeypatch.chdir(tmp_path)
    Path("task.jsonl").write_text(
        '{"prompt":"say yes","expected":"yes","scoring":"contains"}\n', encoding="utf-8",
    )
    Path("soup.yaml").write_text(
        "base: owner/base\ndata: {train: train.jsonl}\ntraining: {quantization: none}\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(ship.app, [
        "--base", "owner/base", "--tuned", "owner/tuned", "--task-eval", "task.jsonl",
        "--general-suite", "mini_mmlu", "--device", device, "--config", "soup.yaml",
        "--emit-evidence", "evidence.json", "--output", "verdict.json",
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    assert f"Live eval: loading base/tuned at full precision ({expected});" in _plain(result.output)
    assert [(model, options["dtype"]) for model, options in calls] == [
        ("owner/base", expected), ("owner/tuned", expected),
    ]
    emitted = json.loads(Path("evidence.json").read_text(encoding="utf-8"))
    verdict = json.loads(Path("verdict.json").read_text(encoding="utf-8"))
    assert emitted["numerics"] == verdict["numerics"] == expected
    replay = runner.invoke(ship.app, [
        "--evidence", "evidence.json", "--config", "soup.yaml", "--output", "replayed.json",
    ])
    assert replay.exit_code == result.exit_code, (replay.output, replay.exception)
    assert json.loads(Path("replayed.json").read_text(encoding="utf-8")) == verdict
    assert len(calls) == 2  # replay never loads either model
    assert cuda.current == 0


@pytest.mark.parametrize("numerics", ["float16", "bfloat16", "float32", "4bit", "8bit"])
def test_evidence_replay_remains_torch_free(tmp_path, numerics):
    from soup_cli.commands.ship import _config_sha_of
    from soup_cli.config.loader import load_config_from_string

    quantization = numerics if numerics in {"4bit", "8bit"} else "none"
    config = (
        f"base: owner/base\ndata: {{train: train.jsonl}}\n"
        f"training: {{quantization: {quantization}}}\n"
    )
    (tmp_path / "soup.yaml").write_text(config, encoding="utf-8")
    (tmp_path / "evidence.json").write_text(json.dumps({
        "task": {"mode": "metric", "base": 0.5, "tuned": 0.7},
        "benchmarks": {"mini_mmlu": {"base": 0.6, "tuned": 0.6}},
        "numerics": numerics,
        "provenance": {"config_sha": _config_sha_of(load_config_from_string(config))},
    }), encoding="utf-8")
    script = """
import importlib.abc
import json
import sys
from pathlib import Path

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'peft'}:
            raise AssertionError('offline replay imported ' + fullname)

sys.meta_path.insert(0, NoTorch())
from typer.testing import CliRunner
from soup_cli.commands.ship import app
result = CliRunner().invoke(app, [
    '--evidence', 'evidence.json', '--config', 'soup.yaml', '--output', 'verdict.json',
])
assert result.exit_code == 0, (result.output, result.exception)
assert json.loads(Path('verdict.json').read_text())['numerics'] == sys.argv[1]
assert 'torch' not in sys.modules
assert 'soup_cli.utils.live_eval' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, numerics], cwd=tmp_path, capture_output=True,
        text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
