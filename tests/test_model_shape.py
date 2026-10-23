"""``utils/model_shape.py``: a model's shape for the VRAM estimate, read without torch (#2049).

The suite's ``_no_hub_model_shape`` fixture keeps the reader off the Hugging Face
cache and the Hub; the tests of those paths patch the two lookups themselves, or
undo the fixture and fence the library calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.utils import model_shape
from soup_cli.utils.model_shape import (
    guessed_shape,
    lora_param_count,
    module_dims,
    params_from_config,
    read_model_shape,
)
from tests._tiny_checkpoint import TINY_LLAMA as _CONFIG
from tests._tiny_checkpoint import write_tiny_model

# The real lookups, taken before the suite's autouse stub replaces them for a test.
_REAL_HUB_LOOKUP = model_shape._hub_lookup
_REAL_HUB_PARAM_COUNT = model_shape._hub_param_count


@pytest.fixture
def local_model(tmp_path: Path) -> tuple[Path, int]:
    model = tmp_path / "tiny-llama"
    count = write_tiny_model(model)
    return model, count


class TestLocalDirectory:
    def test_dims_from_config_and_the_exact_count_from_the_headers(self, local_model):
        model, count = local_model
        shape = read_model_shape(str(model))
        assert shape.params == count
        assert shape.params_source == "safetensors"
        assert shape.shape_source == "config.json"
        assert (shape.hidden_size, shape.intermediate_size, shape.num_layers) == (64, 160, 3)
        assert (shape.num_attention_heads, shape.num_kv_heads, shape.head_dim) == (8, 2, 8)
        assert shape.vocab_size == 1000
        assert shape.tie_word_embeddings is False
        assert shape.model_type == "llama"
        assert not shape.guessed

    def test_without_weights_the_count_comes_from_the_config(self, local_model):
        model, count = local_model
        (model / "model.safetensors").unlink()
        shape = read_model_shape(str(model))
        assert shape.params_source == "config.json"
        # The config count leaves out nothing this model has, so it is exact here.
        assert shape.params == count

    def test_a_multimodal_config_is_read_from_its_text_section(self, tmp_path):
        model = tmp_path / "vlm"
        model.mkdir()
        nested = {"model_type": "some_vlm", "text_config": dict(_CONFIG), "vision_config": {}}
        (model / "config.json").write_text(json.dumps(nested), encoding="utf-8")
        shape = read_model_shape(str(model))
        assert shape.hidden_size == 64 and shape.vocab_size == 1000

    def test_a_directory_without_a_config_is_guessed_from_its_headers(self, local_model):
        model, count = local_model
        (model / "config.json").unlink()
        shape = read_model_shape(str(model))
        assert shape.guessed and shape.vocab_size is None
        assert shape.params == pytest.approx(count, rel=1e-6)

    @pytest.mark.parametrize("payload", ["[1, 2]", "{not json", '{"hidden_size": 64}'])
    def test_an_unusable_config_falls_back_to_a_guess(self, local_model, payload):
        model, _ = local_model
        (model / "config.json").write_text(payload, encoding="utf-8")
        assert read_model_shape(str(model)).guessed


class TestHubIds:
    def test_with_no_config_the_shape_is_guessed_from_the_name(self):
        shape = read_model_shape("someone/Some-Model-7B")
        assert shape.guessed
        assert shape.vocab_size is None
        assert shape.params_source == "name"
        assert shape.params == 7_000_000_000

    def test_a_hub_config_and_header_count_are_used(self, monkeypatch):
        monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kw: (dict(_CONFIG), None))
        monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: 123_456)
        shape = read_model_shape("someone/tiny-llama")
        assert shape.params == 123_456
        assert shape.params_source == "hub safetensors metadata"
        assert shape.vocab_size == 1000

    def test_a_hub_config_without_header_metadata_counts_from_the_config(self, monkeypatch):
        monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kw: (dict(_CONFIG), None))
        shape = read_model_shape("someone/tiny-llama")
        assert shape.params_source == "config.json"
        assert shape.params == params_from_config(_CONFIG)

    def test_a_cached_snapshot_is_counted_from_its_headers(self, monkeypatch, local_model):
        model, count = local_model
        monkeypatch.setattr(
            model_shape, "_hub_lookup", lambda repo, **kw: (dict(_CONFIG), str(model))
        )
        shape = read_model_shape("someone/tiny-llama")
        assert (shape.params, shape.params_source) == (count, "safetensors")

    def test_allow_network_reaches_the_lookups(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            model_shape, "_hub_lookup", lambda repo, **kw: seen.append(kw) or (None, None)
        )
        read_model_shape("someone/x-1B", allow_network=False)
        assert seen == [{"allow_network": False}]

    @pytest.mark.parametrize("base", ["../escape", "a b", ""])
    def test_a_name_that_is_not_a_repo_id_never_reaches_the_lookup(self, monkeypatch, base):
        def boom(repo, **kw):
            raise AssertionError(f"looked up {repo!r}")

        monkeypatch.setattr(model_shape, "_hub_lookup", boom)
        assert read_model_shape(base).guessed


class TestOfflineAndOldHub:
    """The real lookups, with the library calls fenced."""

    @pytest.fixture
    def network(self, monkeypatch):
        """The real lookups, with every network call recorded instead of made.

        The lookups swallow errors on purpose (a gated or missing repo is no config),
        so a fence that raised would pass unseen; the calls are recorded and asserted.
        """
        # Put the real lookups back for these tests only. monkeypatch.undo() would
        # also drop every other autouse fixture's patch for the test.
        monkeypatch.setattr(model_shape, "_hub_lookup", _REAL_HUB_LOOKUP)
        monkeypatch.setattr(model_shape, "_hub_param_count", _REAL_HUB_PARAM_COUNT)
        import huggingface_hub

        calls = []
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
        monkeypatch.setattr(
            huggingface_hub,
            "get_hf_file_metadata",
            lambda *a, **k: calls.append("head") or SimpleNamespace(size=600),
        )
        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda *a, **k: calls.append("download") or "no-such-dir/config.json",
        )
        monkeypatch.setattr(
            "soup_cli.utils.hf.get_hf_api", lambda *a, **k: calls.append("api") or object()
        )
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        return monkeypatch, calls

    @pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
    def test_offline_never_reaches_the_network(self, network, var):
        monkeypatch, calls = network
        monkeypatch.setenv(var, "1")
        assert model_shape._hub_lookup("someone/x", allow_network=True) == (None, None)
        assert model_shape._hub_param_count("someone/x", allow_network=True) is None
        assert calls == []

    def test_allow_network_false_never_reaches_the_network(self, network):
        _, calls = network
        assert model_shape._hub_lookup("someone/x", allow_network=False) == (None, None)
        assert model_shape._hub_param_count("someone/x", allow_network=False) is None
        assert calls == []

    def test_online_the_fence_sees_the_calls(self, network):
        """Control: the recording fence sees both calls when the network is allowed."""
        _, calls = network
        model_shape._hub_lookup("someone/x", allow_network=True)
        model_shape._hub_param_count("someone/x", allow_network=True)
        assert calls == ["head", "download", "api"]

    def test_a_hub_without_the_header_call_counts_from_the_config(self, network):
        # huggingface-hub < 0.20 has no get_safetensors_metadata; Soup's floor is 0.16.
        _, calls = network
        assert model_shape._hub_param_count("someone/x", allow_network=True) is None
        assert calls == ["api"]

    def test_a_failing_download_is_no_config(self, network):
        import huggingface_hub

        monkeypatch, _ = network

        def gated(*args, **kwargs):
            raise RuntimeError("401 token hf_secret")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", gated)
        assert model_shape._hub_lookup("someone/x", allow_network=True) == (None, None)

    @pytest.mark.parametrize("size", [5 * 1024 * 1024, None])
    def test_a_config_over_the_cap_or_of_unknown_size_is_never_downloaded(self, network, size):
        import huggingface_hub

        monkeypatch, calls = network
        monkeypatch.setattr(
            huggingface_hub,
            "get_hf_file_metadata",
            lambda *a, **k: calls.append("head") or SimpleNamespace(size=size),
        )
        assert model_shape._hub_lookup("someone/x", allow_network=True) == (None, None)
        assert calls == ["head"]

    def test_the_head_request_has_the_timeout(self, network):
        import huggingface_hub

        monkeypatch, _ = network
        seen = {}
        monkeypatch.setattr(
            huggingface_hub,
            "get_hf_file_metadata",
            lambda *a, **k: seen.update(k) or SimpleNamespace(size=600),
        )
        model_shape._hub_lookup("someone/x", allow_network=True)
        assert seen["timeout"] == model_shape.HUB_TIMEOUT_S

    def _api(self, monkeypatch, getter):
        monkeypatch.setattr(
            "soup_cli.utils.hf.get_hf_api",
            lambda *a, **k: SimpleNamespace(get_safetensors_metadata=getter),
        )

    def test_a_hub_that_takes_a_timeout_gets_one(self, network):
        monkeypatch, _ = network
        seen = {}

        def getter(repo, *, timeout=None):
            seen["timeout"] = timeout
            return SimpleNamespace(parameter_count={"BF16": 123})

        self._api(monkeypatch, getter)
        assert model_shape._hub_param_count("someone/x", allow_network=True) == 123
        assert seen["timeout"] == model_shape.HUB_TIMEOUT_S

    def test_a_hub_without_the_timeout_argument_still_counts(self, network):
        monkeypatch, _ = network
        self._api(monkeypatch, lambda repo: SimpleNamespace(parameter_count={"BF16": 123}))
        assert model_shape._hub_param_count("someone/x", allow_network=True) == 123

    def test_a_stalled_metadata_lookup_gives_up_at_the_deadline(self, network):
        monkeypatch, _ = network
        monkeypatch.setattr(model_shape, "HUB_METADATA_DEADLINE_S", 0.2)
        release = threading.Event()

        def stalled(repo):
            release.wait(10)
            return SimpleNamespace(parameter_count={"BF16": 123})

        self._api(monkeypatch, stalled)
        started = time.monotonic()
        try:
            assert model_shape._hub_param_count("someone/x", allow_network=True) is None
            assert time.monotonic() - started < 2
        finally:
            release.set()


class TestLoraCount:
    def _shape(self):
        return model_shape.shape_from_config(_CONFIG, params=1, params_source="safetensors")

    def test_gqa_projections_have_their_own_sizes(self):
        shape = self._shape()
        assert module_dims(shape, "q_proj") == (64, 64)
        assert module_dims(shape, "k_proj") == (64, 16)
        assert module_dims(shape, "down_proj") == (160, 64)

    def test_auto_is_peft_s_default_for_the_architecture(self):
        # llama -> q_proj and v_proj: r * (64 + 64) + r * (64 + 16) per layer.
        assert lora_param_count(self._shape(), r=8, target_modules="auto") == 3 * 8 * (128 + 80)

    def test_all_linear_counts_every_projection(self):
        per_layer = 8 * ((64 + 64) + (64 + 16) * 2 + (64 + 64) + (64 + 160) * 2 + (160 + 64))
        got = lora_param_count(self._shape(), r=8, target_modules="all-linear")
        assert got == 3 * per_layer

    def test_rank_zero_is_no_adapter(self):
        assert lora_param_count(self._shape(), r=0, target_modules="auto") == 0


def test_a_guessed_shape_keeps_the_old_size_table():
    shape = guessed_shape(8.0)
    assert (shape.hidden_size, shape.num_layers) == (4096, 32)
    assert shape.vocab_size is None


def test_reading_a_shape_and_estimating_never_imports_torch(local_model):
    model, _ = local_model
    code = (
        "import sys\n"
        "from soup_cli.utils.model_shape import read_model_shape\n"
        "from soup_cli.utils.vram_estimate import RunSpec, estimate_vram\n"
        "from soup_cli.utils import hardware_fit, profiler\n"
        f"shape = read_model_shape({str(model)!r}, allow_network=False)\n"
        "run = RunSpec(seq_len=512, batch_size=1, quant='4bit', peft='qlora')\n"
        "est = estimate_vram(shape, run)\n"
        "assert est.total_bytes > 0 and est.terms['logits'] > 0\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr
