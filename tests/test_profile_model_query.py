"""#2049 part 2: ``soup profile --model`` without a soup.yaml, and the MLX reading.

``--model`` answers fits / tight / does not fit for a model id or directory with the
same estimate the train pre-flight uses, read offline from a cached ``config.json``.
When the shape cannot be read it stops with a message instead of guessing the shape
from the name. On Apple Silicon the fit is against Metal's recommended working-set
size, not the total unified memory.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.mcp_server import registry as reg
from soup_cli.utils import mlx as mlx_utils
from soup_cli.utils import model_shape
from soup_cli.utils.model_shape import read_model_shape
from soup_cli.utils.profiler import GPU_MEMORY
from soup_cli.utils.vram_estimate import RunSpec, estimate_vram, fit_verdict
from tests._tiny_checkpoint import TINY_LLAMA, tensors_for, write_safetensors

runner = CliRunner()

# The real cache and Hub lookups, taken before the conftest fixture replaces them.
REAL_HUB_LOOKUP = model_shape._hub_lookup
REAL_HUB_PARAM_COUNT = model_shape._hub_param_count

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(result) -> str:
    return " ".join(_ANSI.sub("", result.output).split())


def _json(result) -> dict:
    return json.loads(_ANSI.sub("", result.output))


def _tiny_model(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(TINY_LLAMA), encoding="utf-8")
    write_safetensors(directory / "model.safetensors", tensors_for(TINY_LLAMA))
    return directory


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    return _tiny_model(tmp_path / "tiny-llama")


@pytest.fixture(autouse=True)
def _not_a_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test here may read this machine's GPU; the MLX tests opt back in."""
    monkeypatch.setattr(mlx_utils, "is_apple_silicon", lambda: False)
    from soup_cli.utils import gpu

    monkeypatch.setattr(gpu, "get_gpu_info", lambda backend=None: {"memory_total_bytes": 0})


def _profile(*args: str):
    return runner.invoke(app, ["profile", *args])


# --- the query --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "quant", "peft"),
    [("lora", "none", "lora"), ("qlora", "4bit", "qlora"), ("full", "none", "full")],
)
def test_model_query_prices_the_shared_estimate(tiny, method, quant, peft):
    result = _profile(
        "--model", str(tiny), "--gpu", "rtx3060laptop", "--seq", "256",
        "--batch", "2", "--method", method, "--json",
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    data = _json(result)
    expected = estimate_vram(
        read_model_shape(str(tiny)),
        RunSpec(seq_len=256, batch_size=2, quant=quant, peft=peft),
    )
    assert data["total_memory_bytes"] == expected.total_bytes
    assert data["memory_terms_bytes"] == expected.terms
    assert data["model_name"] == str(tiny)
    assert data["peft"] == peft
    assert data["shape_source"] == "config.json"
    assert data["params_source"] == "safetensors"
    assert data["gpu_memory_source"] == "flag"
    assert data["fit"] == fit_verdict(expected.total_bytes, GPU_MEMORY["rtx3060laptop"] * 1024**3)


@pytest.mark.parametrize(
    ("method", "training_yaml"),
    [
        ("lora", "  quantization: none\n"),
        ("qlora", "  quantization: 4bit\n"),
        ("full", "  quantization: none\n  lora:\n    r: 0\n"),
    ],
)
def test_model_query_and_the_config_give_the_same_number(tmp_path, tiny, method, training_yaml):
    """The query is the config's estimate: same model, same run, same bytes."""
    cfg = tmp_path / "soup.yaml"
    cfg.write_text(
        f"base: {tiny.as_posix()}\n"
        "task: sft\n"
        "data:\n  train: data.jsonl\n  max_length: 512\n"
        "training:\n  batch_size: 3\n  gradient_checkpointing: true\n"
        + training_yaml,
        encoding="utf-8",
    )
    from_config = _profile("--config", str(cfg), "--gpu", "a100", "--json")
    assert from_config.exit_code == 0, (from_config.output, repr(from_config.exception))
    query = _profile(
        "--model", str(tiny), "--gpu", "a100", "--seq", "512", "--batch", "3",
        "--method", method, "--gradient-checkpointing", "full", "--json",
    )
    assert query.exit_code == 0, (query.output, repr(query.exception))
    a, b = _json(from_config), _json(query)
    assert a["peft"] == b["peft"]
    assert b["total_memory_bytes"] == a["total_memory_bytes"]
    assert b["memory_terms_bytes"] == a["memory_terms_bytes"]


def test_model_query_defaults_are_soup_yaml_defaults(tiny):
    """With only --model: seq 2048, batch 4 (how profile prices `auto`), QLoRA r 64."""
    result = _profile("--model", str(tiny), "--gpu", "a100", "--json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    data = _json(result)
    assert (data["seq_len"], data["batch_size"], data["peft"]) == (2048, 4, "qlora")
    assert data["lora_r"] == 64
    assert data["gradient_checkpointing"] is False


def test_model_query_lora_r_and_optimizer_change_the_terms(tiny):
    base = ["--model", str(tiny), "--gpu", "a100", "--method", "lora", "--json"]
    small = _json(_profile(*base, "--lora-r", "8"))
    large = _json(_profile(*base, "--lora-r", "128"))
    assert small["memory_terms_bytes"]["adapters"] * 16 == large["memory_terms_bytes"]["adapters"]
    adam = _json(_profile(*base))
    eight_bit = _json(_profile(*base, "--optimizer", "paged_adamw_8bit"))
    optimizer = adam["memory_terms_bytes"]["optimizer"]
    assert eight_bit["memory_terms_bytes"]["optimizer"] * 4 == optimizer


def test_model_query_renders_the_verdict_and_terms(tiny, monkeypatch):
    # A relative name: the panel would wrap a long temporary path mid-word.
    monkeypatch.chdir(tiny.parent)
    result = _profile(
        "--model", tiny.name, "--gpu", "rtx3060laptop", "--seq", "256", "--method", "lora"
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert "Model: tiny-llama" in text
    assert "Fits in 6 GB VRAM" in text
    assert "Logits and loss (vocab 1,000)" in text
    assert "with LoRA r=64" in text


def test_model_query_says_does_not_fit(tiny, monkeypatch):
    monkeypatch.setitem(GPU_MEMORY, "tinycard", 0.001)
    result = _profile("--model", str(tiny), "--gpu", "tinycard", "--seq", "4096", "--json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert _json(result)["fit"] == "does not fit"


# --- offline from the cache, and a clear stop when the shape cannot be read --------


def _fake_hf_cache(root: Path, repo: str) -> Path:
    """The Hugging Face cache layout for ``repo``: refs/main and one snapshot."""
    folder = root / ("models--" + repo.replace("/", "--"))
    revision = "0" * 40
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(revision, encoding="utf-8")
    return _tiny_model(folder / "snapshots" / revision)


@pytest.fixture
def offline_cache(tmp_path, monkeypatch):
    """The real cache lookup over a temporary cache, offline, with the network calls
    replaced by ones that fail the test."""
    import huggingface_hub
    from huggingface_hub import constants

    monkeypatch.setattr(model_shape, "_hub_lookup", REAL_HUB_LOOKUP)
    monkeypatch.setattr(model_shape, "_hub_param_count", REAL_HUB_PARAM_COUNT)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def _no_network(*_args, **_kwargs):
        raise AssertionError("soup profile --model reached the network while offline")

    for name in ("hf_hub_download", "get_hf_file_metadata", "get_safetensors_metadata"):
        monkeypatch.setattr(huggingface_hub, name, _no_network)
    return tmp_path / "hf-cache"


def test_model_query_works_offline_from_the_cache(offline_cache):
    snapshot = _fake_hf_cache(offline_cache, "acme/tiny-llama")
    result = _profile(
        "--model", "acme/tiny-llama", "--gpu", "a100", "--seq", "256", "--json"
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    data = _json(result)
    assert data["model_name"] == "acme/tiny-llama"
    assert data["shape_source"] == "config.json"
    assert data["params_source"] == "safetensors"
    expected = estimate_vram(
        read_model_shape(str(snapshot)),
        RunSpec(seq_len=256, batch_size=4, quant="4bit", peft="qlora"),
    )
    assert data["total_memory_bytes"] == expected.total_bytes


def test_model_query_not_cached_offline_stops_with_a_clear_message(offline_cache):
    result = _profile("--model", "acme/not-cached", "--gpu", "a100", "--json")
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert not isinstance(result.exception, AssertionError)
    text = _plain(result)
    assert "Cannot read acme/not-cached" in text
    assert "not in the Hugging Face cache" in text
    assert "HF_HUB_OFFLINE" in text
    # Nothing was estimated, so no JSON result and no guessed shape.
    assert "total_memory_bytes" not in result.output


def test_model_query_hub_unreachable_stops_with_a_clear_message():
    # The conftest lookup finds no cache and no Hub, as an unreachable or gated repo.
    result = _profile("--model", "Qwen/Qwen2.5-7B-Instruct", "--gpu", "a100")
    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _plain(result)
    assert "Cannot read Qwen/Qwen2.5-7B-Instruct" in text
    assert "could not be fetched from the Hub" in text
    assert "HF_HUB_OFFLINE" not in text
    # The name says 7B; a guessed 7B estimate must not be printed instead.
    assert "GPU Memory Estimate" not in text


def test_model_query_directory_without_config_stops(tmp_path):
    empty = tmp_path / "no-config"
    empty.mkdir()
    result = _profile("--model", str(empty), "--gpu", "a100")
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "no readable config.json in that directory" in _plain(result)


def test_model_query_rejects_a_value_that_is_neither_a_directory_nor_a_hub_id():
    result = _profile("--model", "../outside/model", "--gpu", "a100")
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "not a local model directory or a Hugging Face model id" in _plain(result)


def test_model_query_escapes_markup_in_the_model_name():
    # No path separator before the bracket: on Windows a backslash would escape it.
    result = _profile("--model", "[bold]x", "--gpu", "a100")
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Cannot read [bold]x" in _plain(result)


# --- refused combinations ----------------------------------------------------------


def test_model_and_config_together_are_refused(tmp_path, tiny):
    result = _profile("--model", str(tiny), "--config", str(tmp_path / "soup.yaml"))
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "--model and --config cannot be used together" in _plain(result)


@pytest.mark.parametrize(
    "flag",
    [
        ("--seq", "512"),
        ("--batch", "2"),
        ("--method", "lora"),
        ("--lora-r", "8"),
        ("--optimizer", "sgd"),
        ("--gradient-checkpointing", "full"),
    ],
)
def test_query_flags_without_model_are_refused(tmp_path, flag):
    result = _profile("--config", str(tmp_path / "soup.yaml"), *flag)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"{flag[0]} only applies with --model" in _plain(result)


@pytest.mark.parametrize(
    ("flag", "value", "choices"),
    [
        ("--method", "dora", "full, lora, qlora"),
        ("--gradient-checkpointing", "on", "auto, full, medium, off, selective"),
        ("--optimizer", "adam", "adafactor"),
    ],
)
def test_unknown_choices_are_refused_with_the_valid_ones(tiny, flag, value, choices):
    result = _profile("--model", str(tiny), flag, value)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Unknown {flag}: {value}" in text
    assert choices in text


@pytest.mark.parametrize(("flag", "value"), [("--seq", "63"), ("--batch", "0"), ("--lora-r", "0")])
def test_out_of_range_numbers_are_usage_errors(tiny, flag, value):
    result = _profile("--model", str(tiny), flag, value)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "is not in the range" in _plain(result)


def test_the_default_config_still_applies_without_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _profile()
    assert result.exit_code == 1
    assert "Config file not found: soup.yaml" in _plain(result)


# --- MLX: Metal's working set, not the unified memory -----------------------------


class _FakeMetal:
    def __init__(self, info):
        self._info = info

    def device_info(self):
        if isinstance(self._info, Exception):
            raise self._info
        return self._info


def _install_fake_mlx(monkeypatch, *, top_level=None, metal=None) -> None:
    core = types.ModuleType("mlx.core")
    if top_level is not None:
        core.device_info = _FakeMetal(top_level).device_info
    core.metal = _FakeMetal(metal if metal is not None else {})
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(mlx_utils, "is_apple_silicon", lambda: True)


_WORKING_SET = 22 * 1024**3
_UNIFIED = 32 * 1024**3


@pytest.mark.parametrize(
    ("top_level", "metal"),
    [
        ({"max_recommended_working_set_size": _WORKING_SET}, None),
        (None, {"max_recommended_working_set_size": _WORKING_SET}),
    ],
)
def test_working_set_is_read_from_either_mlx_api(monkeypatch, top_level, metal):
    _install_fake_mlx(monkeypatch, top_level=top_level, metal=metal)
    assert mlx_utils.get_mlx_working_set_bytes() == _WORKING_SET


@pytest.mark.parametrize(
    "info", [{}, {"max_recommended_working_set_size": 0}, {"max_recommended_working_set_size": "x"},
             {"max_recommended_working_set_size": True}, RuntimeError("no device")],
)
def test_unreadable_working_set_is_none(monkeypatch, info):
    _install_fake_mlx(monkeypatch, metal=info)
    assert mlx_utils.get_mlx_working_set_bytes() is None


def test_no_working_set_off_apple_silicon(monkeypatch):
    _install_fake_mlx(monkeypatch, metal={"max_recommended_working_set_size": _WORKING_SET})
    monkeypatch.setattr(mlx_utils, "is_apple_silicon", lambda: False)
    assert mlx_utils.get_mlx_working_set_bytes() is None


def _mlx_config(tmp_path: Path, tiny: Path) -> Path:
    cfg = tmp_path / "soup.yaml"
    cfg.write_text(
        f"base: {tiny.as_posix()}\n"
        "task: sft\nbackend: mlx\n"
        "data:\n  train: data.jsonl\n  max_length: 256\n"
        "training:\n  batch_size: 1\n  quantization: none\n",
        encoding="utf-8",
    )
    return cfg


@pytest.fixture
def mac(monkeypatch):
    _install_fake_mlx(monkeypatch, metal={"max_recommended_working_set_size": _WORKING_SET})
    monkeypatch.setattr(mlx_utils, "get_unified_memory_bytes", lambda: _UNIFIED)
    from soup_cli.utils import gpu

    # What get_gpu_info reports on a Mac with mlx: the whole unified memory.
    monkeypatch.setattr(
        gpu, "get_gpu_info", lambda backend=None: {"memory_total_bytes": _UNIFIED}
    )


def test_profile_on_apple_silicon_fits_against_the_working_set(tmp_path, tiny, mac):
    result = _profile("--config", str(_mlx_config(tmp_path, tiny)), "--json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    data = _json(result)
    assert data["gpu_memory_source"] == "mlx working set"
    assert data["gpu_memory_gb"] == _WORKING_SET / 1024**3
    assert data["fit"] == fit_verdict(data["total_memory_bytes"], _WORKING_SET)
    assert any("working-set" in note and "not measured on Apple" in note for note in data["notes"])


def test_working_set_decides_the_verdict_not_the_unified_memory(tmp_path, tiny, mac, monkeypatch):
    """A run between the working set and the total memory does not fit."""
    from soup_cli.utils import profiler

    real = profiler.profile_result

    def _sized(breakdown, **kwargs):
        result = real(breakdown, **kwargs)
        result["total_memory_bytes"] = 26 * 1024**3
        return result

    monkeypatch.setattr(profiler, "profile_result", _sized)
    result = _profile("--config", str(_mlx_config(tmp_path, tiny)), "--json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert _json(result)["fit"] == "does not fit"


def test_profile_text_names_the_working_set(tmp_path, tiny, mac):
    result = _profile("--config", str(_mlx_config(tmp_path, tiny)))
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Fits in 22 GB of Metal's working set" in _plain(result)


def test_model_query_on_apple_silicon_uses_the_working_set(tiny, mac):
    result = _profile("--model", str(tiny), "--seq", "256", "--json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert _json(result)["gpu_memory_source"] == "mlx working set"


def test_gpu_flag_still_wins_on_apple_silicon(tmp_path, tiny, mac):
    result = _profile("--config", str(_mlx_config(tmp_path, tiny)), "--gpu", "a100", "--json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert _json(result)["gpu_memory_source"] == "flag"


def test_apple_silicon_without_a_working_set_is_assumed_not_unified(
    tmp_path, tiny, mac, monkeypatch
):
    monkeypatch.setattr(mlx_utils, "get_mlx_working_set_bytes", lambda: None)
    result = _profile("--config", str(_mlx_config(tmp_path, tiny)), "--json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    data = _json(result)
    assert data["gpu_memory_source"] == "assumed"
    assert data["gpu_memory_gb"] != _UNIFIED / 1024**3


def test_mcp_profile_reads_the_working_set_too(tmp_path, tiny, mac, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _mlx_config(tmp_path, tiny)
    result = reg.tool_profile({"config": "soup.yaml"})
    assert result["gpu_memory_source"] == "mlx working set"
    assert result["gpu_memory_gb"] == _WORKING_SET / 1024**3
