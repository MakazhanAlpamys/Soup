"""#830: ``soup profile`` must not present an assumed GPU as a measured fit.

With no device detected and no ``--gpu``, the profile used to report
"OK Fits in 24 GB VRAM" as if it had measured a 24 GB card. The GPU table also
had no RTX 3050, no 50-series card below the 5090, no Blackwell datacenter part
and no laptop variants, so ``--gpu`` rejected exactly the cards where the
memory estimate decides whether a run is possible.
"""

from __future__ import annotations

import json
import re

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.mcp_server import registry as reg
from soup_cli.utils.profiler import GPU_MEMORY, normalize_gpu_key

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Every card #830 added, with its memory. A card dropped from GPU_MEMORY, or
# given the wrong size, fails its own row.
_NEW_IN_ISSUE_830 = {
    "rtx3050": 8,
    "rtx3050_6gb": 6,
    "rtx5060": 8,
    "rtx5060ti": 16,
    "rtx5060ti_8gb": 8,
    "rtx5070": 12,
    "rtx5070ti": 16,
    "rtx5080": 16,
    "rtx4050laptop": 6,
    "rtx4060laptop": 8,
    "rtx4070laptop": 8,
    "rtx4080laptop": 12,
    "rtx4090laptop": 16,
    "rtx5060laptop": 8,
    "rtx5070laptop": 8,
    "rtx5070tilaptop": 12,
    "rtx5080laptop": 16,
    "rtx5090laptop": 24,
    "b200": 180,
}


def _json(result) -> dict:
    # Rich highlights JSON when colour is forced (FORCE_COLOR=1), so strip it.
    return json.loads(_ANSI.sub("", result.output))


_CONFIG = (
    "base: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"
    "task: sft\n"
    "data:\n"
    "  train: ./data/train.jsonl\n"
    "  max_length: 512\n"
    "training:\n"
    "  batch_size: 4\n"
    "  quantization: none\n"
    "output: ./output\n"
)


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "soup.yaml"
    path.write_text(_CONFIG, encoding="utf-8")
    return path


@pytest.fixture
def no_gpu(monkeypatch):
    def _no_device():
        raise RuntimeError("no GPU")

    monkeypatch.setattr("soup_cli.utils.gpu.get_gpu_info", _no_device)


@pytest.fixture
def wide_console(monkeypatch):
    # profile.py builds its Console at import, before CliRunner sets COLUMNS,
    # so a terminal_width argument would not stop the panel wrapping.
    monkeypatch.setattr("soup_cli.commands.profile.console", Console(width=200))


def test_no_detected_gpu_is_not_reported_as_a_measured_fit(config_file, no_gpu, wide_console):
    result = runner.invoke(app, ["profile", "--config", str(config_file)])

    assert result.exit_code == 0, result.output
    assert "Fits in 24 GB VRAM" not in result.output
    assert "No GPU detected" in result.output
    assert "assumed 24 GB" in result.output


def test_json_says_the_gpu_memory_was_assumed(config_file, no_gpu):
    result = runner.invoke(app, ["profile", "--config", str(config_file), "--json"])

    data = _json(result)
    assert data["gpu_memory_gb"] == 24.0
    assert data["gpu_memory_source"] == "assumed"


def test_a_detected_gpu_is_still_a_real_verdict(config_file, monkeypatch):
    monkeypatch.setattr(
        "soup_cli.utils.gpu.get_gpu_info",
        lambda: {"memory_total_bytes": 8 * 1024**3, "memory_total": "8 GB", "gpu_count": 1},
    )

    result = runner.invoke(app, ["profile", "--config", str(config_file), "--json"])

    data = _json(result)
    assert data["gpu_memory_gb"] == 8.0
    assert data["gpu_memory_source"] == "detected"


@pytest.mark.parametrize(
    ("gpu", "memory_gb"),
    [
        ("rtx3050", 8),
        ("rtx5070", 12),
        ("rtx 5070 laptop", 8),
        ("NVIDIA GeForce RTX 5070 Laptop GPU", 8),
        ("rtx4090 laptop", 16),
        ("b200", 180),
    ],
)
def test_newer_and_laptop_gpus_are_accepted(config_file, no_gpu, gpu, memory_gb):
    result = runner.invoke(app, ["profile", "--config", str(config_file), "--gpu", gpu, "--json"])

    assert result.exit_code == 0, result.output
    data = _json(result)
    assert data["gpu_memory_gb"] == memory_gb
    assert data["gpu_memory_source"] == "flag"


def test_an_unknown_gpu_is_still_rejected(config_file):
    result = runner.invoke(app, ["profile", "--config", str(config_file), "--gpu", "rtx9999"])

    assert result.exit_code == 1
    assert "Unknown GPU" in result.output


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("NVIDIA GeForce RTX 5070 Laptop GPU", "rtx5070laptop"),
        ("rtx-4090", "rtx4090"),
        ("Tesla T4", "t4"),
        ("a100_40gb", "a100_40gb"),
    ],
)
def test_normalize_gpu_key(name, key):
    assert normalize_gpu_key(name) == key
    assert key in GPU_MEMORY


@pytest.mark.parametrize(("key", "memory_gb"), sorted(_NEW_IN_ISSUE_830.items()))
def test_every_card_added_for_830_is_in_the_table(key, memory_gb):
    assert GPU_MEMORY[key] == memory_gb


# The MCP ``profile`` tool mirrors the command's GPU resolution, so it is held
# to the same two behaviours.

_MCP_CONFIG = "base: Qwen/Qwen2.5-0.5B\ntask: sft\ndata:\n  train: data.jsonl\n"


@pytest.fixture
def mcp_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(_MCP_CONFIG, encoding="utf-8")
    return "soup.yaml"


def test_mcp_profile_reports_an_assumed_gpu(mcp_config, no_gpu):
    out = reg.tool_profile({"config": mcp_config})

    assert out["gpu_memory_gb"] == 24.0
    assert out["gpu_memory_source"] == "assumed"


def test_mcp_profile_accepts_a_torch_device_name(mcp_config):
    out = reg.tool_profile({"config": mcp_config, "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU"})

    assert out["gpu_memory_gb"] == 8.0
    assert out["gpu_memory_source"] == "flag"
