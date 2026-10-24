"""RTX 30-series laptop GPUs in ``soup profile --gpu`` and the MCP ``profile`` tool.

``GPU_MEMORY`` had the 40- and 50-series laptop parts but none of the 30-series
ones, so ``--gpu "rtx3060 laptop"`` was "Unknown GPU", and the laptop part has
6 GB, not the desktop card's 12. Memory sizes are NVIDIA's own specification
table for the series; device names are the ones NVIDIA's driver reports, which
is what torch prints.
"""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.mcp_server import registry as reg
from soup_cli.utils.profiler import GPU_MEMORY, normalize_gpu_key

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# NVIDIA's specification table for the GeForce RTX 30 Series laptop GPUs
# (nvidia.com/en-us/geforce/laptops/30-series/, "Memory Size"): two parts ship
# in two sizes, "RTX 3050: 4 GB or 6 GB" and "RTX 3080: 8 GB or 16 GB".
_NVIDIA_SPEC = {
    "rtx3050laptop": 4,
    "rtx3050laptop_4gb": 4,
    "rtx3050laptop_6gb": 6,
    "rtx3050tilaptop": 4,
    "rtx3060laptop": 6,
    "rtx3070laptop": 8,
    "rtx3070tilaptop": 8,
    "rtx3080laptop": 16,
    "rtx3080laptop_8gb": 8,
    "rtx3080tilaptop": 16,
}

# Every 30-series laptop name in the NVIDIA driver's supported-GPU list, as
# torch reports it. The 3050 ships under three names, two of which carry their
# size, and the 3050 Ti is listed spelled both ways.
_DRIVER_NAMES = [
    ("NVIDIA GeForce RTX 3050 Laptop GPU", "rtx3050laptop"),
    ("NVIDIA GeForce RTX 3050 4GB Laptop GPU", "rtx3050laptop_4gb"),
    ("NVIDIA GeForce RTX 3050 6GB Laptop GPU", "rtx3050laptop_6gb"),
    ("NVIDIA GeForce RTX 3050 Ti Laptop GPU", "rtx3050tilaptop"),
    ("NVIDIA GeForce RTX 3050Ti Laptop GPU", "rtx3050tilaptop"),
    ("NVIDIA GeForce RTX 3060 Laptop GPU", "rtx3060laptop"),
    ("NVIDIA GeForce RTX 3070 Laptop GPU", "rtx3070laptop"),
    ("NVIDIA GeForce RTX 3070 Ti Laptop GPU", "rtx3070tilaptop"),
    ("NVIDIA GeForce RTX 3080 Laptop GPU", "rtx3080laptop"),
    ("NVIDIA GeForce RTX 3080 Ti Laptop GPU", "rtx3080tilaptop"),
]


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


@pytest.mark.parametrize(("key", "memory_gb"), sorted(_NVIDIA_SPEC.items()))
def test_every_30_series_laptop_part_has_its_nvidia_memory(key, memory_gb):
    assert GPU_MEMORY[key] == memory_gb


@pytest.mark.parametrize(("name", "key"), _DRIVER_NAMES)
def test_the_driver_name_resolves_to_its_own_key(name, key):
    assert normalize_gpu_key(name) == key
    assert GPU_MEMORY[key] == _NVIDIA_SPEC[key]


@pytest.mark.parametrize(
    ("typed", "key"),
    [
        ("rtx3060 laptop", "rtx3060laptop"),
        ("RTX 3080 Ti Laptop", "rtx3080tilaptop"),
        ("rtx 3080 laptop 8gb", "rtx3080laptop_8gb"),
        ("rtx3080 laptop 8 GB", "rtx3080laptop_8gb"),
        ("rtx3080laptop_8gb", "rtx3080laptop_8gb"),
        ("rtx 3050 6gb laptop", "rtx3050laptop_6gb"),
    ],
)
def test_typed_laptop_names_resolve(typed, key):
    assert normalize_gpu_key(typed) == key
    assert key in GPU_MEMORY


@pytest.mark.parametrize(
    ("name", "key", "memory_gb"),
    [
        # Desktop parts keep their own entries: a laptop key must not shadow them.
        ("NVIDIA GeForce RTX 3060", "rtx3060", 12),
        ("NVIDIA GeForce RTX 3080", "rtx3080", 10),
        ("NVIDIA GeForce RTX 3070 Ti", "rtx3070ti", 8),
        # Size suffixes already in the table, written as keys.
        ("rtx3050_6gb", "rtx3050_6gb", 6),
        ("a100_40gb", "a100_40gb", 40),
        ("rtx5060ti_8gb", "rtx5060ti_8gb", 8),
        ("NVIDIA GeForce RTX 5070 Laptop GPU", "rtx5070laptop", 8),
        ("rtx-4090", "rtx4090", 24),
    ],
)
def test_existing_keys_resolve_as_before(name, key, memory_gb):
    assert normalize_gpu_key(name) == key
    assert GPU_MEMORY[key] == memory_gb


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("rtx 3050 6gb", "rtx3050_6gb"),
        ("A100 40GB", "a100_40gb"),
        ("rtx5060ti 8gb", "rtx5060ti_8gb"),
        # A hyphen before the size reads the same as a space.
        ("A100-40GB", "a100_40gb"),
        ("rtx-3050-6gb", "rtx3050_6gb"),
    ],
)
def test_a_size_written_as_a_word_finds_the_suffixed_key(name, key):
    # The driver writes the size as a word ("RTX 3050 6GB Laptop GPU"); the
    # table writes it as a suffix. The desktop keys get the same reading.
    assert normalize_gpu_key(name) == key
    assert key in GPU_MEMORY


@pytest.mark.parametrize(
    ("name", "key", "memory_gb"),
    [
        ("rtx 3080 laptop 16gb", "rtx3080laptop", 16),
        ("A100 80GB", "a100", 80),
        ("rtx 3060 laptop 6gb", "rtx3060laptop", 6),
        ("NVIDIA GeForce RTX 3060 12GB", "rtx3060", 12),
    ],
)
def test_a_size_the_plain_key_already_holds_resolves_to_it(name, key, memory_gb):
    # Only the smaller size of a two-size part has a suffixed key; the plain key
    # is the larger one, so naming that size has to reach the plain key.
    assert normalize_gpu_key(name) == key
    assert GPU_MEMORY[key] == memory_gb


@pytest.mark.parametrize(
    ("name", "key"),
    [
        # A size the plain key does not hold stays unknown, not the plain key's.
        ("rtx 3060 8gb", "rtx3060_8gb"),
        ("A100 20GB", "a100_20gb"),
    ],
)
def test_a_size_no_key_holds_stays_unknown(name, key):
    assert normalize_gpu_key(name) == key
    assert key not in GPU_MEMORY


def test_a_size_too_long_to_read_as_a_number_stays_unknown(config_file):
    # The size word is user text and the regex takes any number of digits, so a
    # 5000-digit size must not be turned into an int: that raises instead of
    # answering, and ``profile`` should still say Unknown GPU.
    typed = "a100 " + "9" * 5000 + "gb"

    key = normalize_gpu_key(typed)
    assert key not in GPU_MEMORY

    result = runner.invoke(app, ["profile", "--config", str(config_file), "--gpu", typed])

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "UnknownGPU:" in re.sub(r"\s+", "", _ANSI.sub("", result.output))


def test_profile_takes_the_laptop_3060_from_the_issue(config_file, no_gpu):
    result = runner.invoke(
        app, ["profile", "--config", str(config_file), "--gpu", "rtx3060 laptop", "--json"]
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    data = _json(result)
    assert data["gpu_memory_gb"] == 6
    assert data["gpu_memory_source"] == "flag"


def test_profile_takes_a_size_named_driver_name(config_file, no_gpu):
    result = runner.invoke(
        app,
        [
            "profile",
            "--config",
            str(config_file),
            "--gpu",
            "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
            "--json",
        ],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert _json(result)["gpu_memory_gb"] == 6


def test_the_unknown_gpu_list_names_the_30_series_laptop_keys(config_file):
    result = runner.invoke(app, ["profile", "--config", str(config_file), "--gpu", "rtx9999"])

    assert result.exit_code == 1, (result.output, repr(result.exception))
    # Rich wraps the list at the terminal width, so drop all whitespace first.
    output = re.sub(r"\s+", "", _ANSI.sub("", result.output))
    assert "UnknownGPU:rtx9999" in output
    listed = set(output.split("Validoptions:", 1)[1].split(","))
    assert set(_NVIDIA_SPEC) <= listed


_MCP_CONFIG = "base: Qwen/Qwen2.5-0.5B\ntask: sft\ndata:\n  train: data.jsonl\n"


@pytest.fixture
def mcp_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(_MCP_CONFIG, encoding="utf-8")
    return "soup.yaml"


def test_mcp_profile_takes_the_laptop_device_name(mcp_config):
    out = reg.tool_profile({"config": mcp_config, "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU"})

    assert out["gpu_memory_gb"] == 6.0
    assert out["gpu_memory_source"] == "flag"
