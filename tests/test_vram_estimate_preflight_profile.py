"""#2049 acceptance: the train pre-flight and ``soup profile`` return the same number.

The same config through both paths, and through the MCP ``profile`` tool, compared
in bytes before the pre-flight's 10% margin. Before #2049 the pre-flight and
``soup profile`` used two formulas in two units and disagreed on every row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.train import _hardware_fit_preflight
from soup_cli.config.loader import load_config_from_string
from soup_cli.mcp_server.registry import tool_profile
from soup_cli.utils import hardware_fit, model_shape
from soup_cli.utils.vram_estimate import VRAM_SAFETY_MARGIN
from tests._tiny_checkpoint import TINY_LLAMA, write_tiny_model

_ANY_CARD = {"memory_total_bytes": 10**15}

_RUNS = {
    "lora bf16": {"quantization": "none", "batch_size": 2, "lora": {"r": 16}},
    "lora bf16, checkpointing": {
        "quantization": "none",
        "batch_size": 4,
        "gradient_checkpointing": True,
        "lora": {"r": 8, "target_modules": "all-linear"},
    },
    "qlora": {"quantization": "4bit", "batch_size": 1, "lora": {"r": 64}},
    "full fine-tuning": {"quantization": "none", "batch_size": 1, "lora": {"r": 0}},
    "8-bit optimizer": {"quantization": "none", "batch_size": 2, "optimizer": "adamw_bnb_8bit"},
}


def _yaml(base: str, training: dict, max_length: int = 1024) -> str:
    return yaml.safe_dump(
        {
            "base": base,
            "task": "sft",
            "data": {"train": "train.jsonl", "max_length": max_length},
            "training": training,
        }
    )


def _preflight_report(cfg):
    seen = []
    real = hardware_fit.decide_hardware_fit

    def recording(inp, **kwargs):
        seen.append(real(inp, **kwargs))
        return seen[-1]

    hardware_fit.decide_hardware_fit = recording
    try:
        _hardware_fit_preflight(cfg, _ANY_CARD, allow_oom_attempt=False)
    finally:
        hardware_fit.decide_hardware_fit = real
    assert len(seen) == 1, "the pre-flight did not price the run"
    return seen[0]


def _three_ways(tmp_path: Path, text: str) -> tuple[int, int, int, object]:
    (tmp_path / "soup.yaml").write_text(text, encoding="utf-8")
    report = _preflight_report(load_config_from_string(text))
    result = CliRunner().invoke(app, ["profile", "--config", "soup.yaml", "--json"])
    assert result.exit_code == 0, result.output
    profile = json.loads(result.output)["total_memory_bytes"]
    mcp = tool_profile({"config": "soup.yaml"})["total_memory_bytes"]
    return report.breakdown.estimate.total_bytes, profile, mcp, report


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "tiny")
    return tmp_path


@pytest.mark.parametrize("name", sorted(_RUNS))
def test_a_local_model_gets_one_number(workdir, name):
    text = _yaml(str(workdir / "tiny"), _RUNS[name])
    preflight, profile, mcp, report = _three_ways(workdir, text)
    assert preflight == profile == mcp
    # The comparison is before the margin, which only the gate adds.
    assert report.peak_vram_gb == pytest.approx(preflight / 1e9)
    assert report.required_with_margin_gb == pytest.approx(
        report.peak_vram_gb * (1 + VRAM_SAFETY_MARGIN)
    )


@pytest.mark.parametrize("name", sorted(_RUNS))
def test_a_hub_model_gets_one_number(workdir, monkeypatch, name):
    monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kw: (dict(TINY_LLAMA), None))
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: 1_234_567)
    preflight, profile, mcp, _ = _three_ways(workdir, _yaml("someone/tiny-llama", _RUNS[name]))
    assert preflight == profile == mcp


def test_a_model_with_no_readable_config_gets_one_number_too(workdir):
    preflight, profile, mcp, report = _three_ways(
        workdir, _yaml("someone/Unreadable-8B", _RUNS["lora bf16"])
    )
    assert preflight == profile == mcp
    assert report.breakdown.logits_gb == 0  # the vocabulary is unknown, and it says so
    assert report.breakdown.estimate.notes
