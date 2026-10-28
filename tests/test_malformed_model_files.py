"""A malformed model file is refused or ignored by name.

The readers behind ``soup train``'s VRAM pre-flight, ``soup plan``, ``soup apply``,
``soup profile`` and ``soup infer --backend mlx`` read a model's ``config.json``, the
header of a ``.safetensors`` file and a native adapter's ``adapter_config.json``. A
malformed file or an out-of-range figure gives the same result as a missing or
unreadable one: the shape is guessed and the output names ``config.json``, or the
adapter is refused with a message naming ``adapter_config.json``.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import train as train_mod
from soup_cli.config.loader import load_config_from_string
from soup_cli.mcp_server.registry import tool_profile
from soup_cli.utils import hardware_fit, model_shape
from soup_cli.utils.gpu import param_count_from_local_safetensors
from soup_cli.utils.mlx_generation import read_native_adapter, reject_native_adapter
from soup_cli.utils.model_shape import read_model_shape
from tests._tiny_checkpoint import TINY_LLAMA, write_safetensors, write_tiny_model
from tests.conftest import strip_ansi

SIZE = 200_000
BEYOND = 10**400
OUT_OF_RANGE = pytest.param(BEYOND, id="out-of-range")


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _malformed_json() -> bytes:
    return b"[" * SIZE + b"]" * SIZE


def _model(tmp_path: Path, *, config=None, config_bytes: bytes | None = None, name="model"):
    """A model directory whose config.json is ``config`` (or the raw ``config_bytes``)."""
    path = tmp_path / name
    path.mkdir()
    if config_bytes is not None:
        (path / "config.json").write_bytes(config_bytes)
    else:
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return path


def _header_file(path: Path, header: bytes) -> None:
    path.write_bytes(struct.pack("<Q", len(header)) + header)


class TestConfigJson:
    def test_a_malformed_config_is_no_config(self, tmp_path):
        model = _model(tmp_path, config_bytes=_malformed_json())
        shape = read_model_shape(str(model), allow_network=False)
        assert shape.guessed
        assert shape.vocab_size is None

    @pytest.mark.parametrize(
        "field",
        [
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ],
    )
    def test_an_out_of_range_dimension_is_no_config(self, tmp_path, field):
        model = _model(tmp_path, config={**TINY_LLAMA, field: BEYOND})
        shape = read_model_shape(str(model), allow_network=False)
        assert shape.guessed
        assert shape.vocab_size is None

    @pytest.mark.parametrize(
        ("field", "limit"),
        [
            ("hidden_size", model_shape._MAX_DIM),
            ("intermediate_size", model_shape._MAX_DIM),
            ("vocab_size", model_shape._MAX_DIM),
            ("num_hidden_layers", model_shape._MAX_LAYERS),
        ],
    )
    def test_the_limit_is_inclusive(self, tmp_path, field, limit):
        at = _model(tmp_path, config={**TINY_LLAMA, field: limit}, name="at")
        over = _model(tmp_path, config={**TINY_LLAMA, field: limit + 1}, name="over")
        assert not read_model_shape(str(at), allow_network=False).guessed
        assert read_model_shape(str(over), allow_network=False).guessed

    def test_a_real_model_is_untouched_by_the_limits(self, tmp_path):
        model = tmp_path / "tiny"
        count = write_tiny_model(model)
        shape = read_model_shape(str(model), allow_network=False)
        assert not shape.guessed
        assert shape.params == count

    def test_an_out_of_range_dimension_in_the_text_section_is_no_config(self, tmp_path):
        multimodal = {"model_type": "x", "text_config": {**TINY_LLAMA, "hidden_size": BEYOND}}
        model = _model(tmp_path, config=multimodal)
        assert read_model_shape(str(model), allow_network=False).guessed

    def test_the_json_reader_returns_none_for_a_malformed_file(self, tmp_path):
        # The Hub-cache and Hub paths read their config.json through the same function.
        cached = tmp_path / "snapshot"
        cached.mkdir()
        (cached / "config.json").write_bytes(_malformed_json())
        assert model_shape._read_json_file(str(cached / "config.json")) is None


class TestSafetensorsHeader:
    def _model(self, tmp_path, header: bytes):
        model = _model(tmp_path, config=TINY_LLAMA)
        _header_file(model / "model.safetensors", header)
        return model

    def test_a_malformed_header_is_no_count(self, tmp_path):
        model = self._model(tmp_path, _malformed_json())
        assert param_count_from_local_safetensors(str(model)) is None

    def test_an_out_of_range_dimension_is_no_count(self, tmp_path):
        header = json.dumps(
            {"w": {"dtype": "F32", "shape": [BEYOND, 2], "data_offsets": [0, 0]}}
        ).encode()
        model = self._model(tmp_path, header)
        assert param_count_from_local_safetensors(str(model)) is None

    def test_one_tensor_with_more_elements_than_any_model_has_is_no_count(self, tmp_path):
        header = json.dumps(
            {"w": {"dtype": "F32", "shape": [2**21, 2**20], "data_offsets": [0, 0]}}
        ).encode()
        model = self._model(tmp_path, header)
        assert param_count_from_local_safetensors(str(model)) is None

    def test_tensors_that_add_up_to_more_than_any_model_has_are_no_count(self, tmp_path):
        tensor = {"dtype": "F32", "shape": [2**20, 2**20], "data_offsets": [0, 0]}
        header = json.dumps({f"w{i}": tensor for i in range(2048)}).encode()
        model = self._model(tmp_path, header)
        assert param_count_from_local_safetensors(str(model)) is None

    def test_the_shape_falls_back_to_the_config_count(self, tmp_path):
        model = self._model(tmp_path, _malformed_json())
        shape = read_model_shape(str(model), allow_network=False)
        assert not shape.guessed
        assert shape.params_source == "config.json"

    def test_a_malformed_index_file_is_no_count(self, tmp_path):
        model = self._model(tmp_path, b"{}")
        (model / "model.safetensors.index.json").write_bytes(_malformed_json())
        assert param_count_from_local_safetensors(str(model)) is None

    def test_a_real_header_still_counts(self, tmp_path):
        model = tmp_path / "ok"
        model.mkdir()
        count = write_safetensors(model / "model.safetensors", {"w": [3, 5], "b": [7]})
        assert param_count_from_local_safetensors(str(model)) == count == 22


class TestTheEstimateNeverRaises:
    def test_an_out_of_range_shape_is_not_priced(self, tmp_path, monkeypatch):
        shape = model_shape.shape_from_config(
            TINY_LLAMA, params=10, params_source="safetensors"
        )
        monkeypatch.setattr(
            model_shape,
            "read_model_shape",
            lambda base, **kw: model_shape.ModelShape(**{**shape.__dict__, "params": BEYOND}),
        )
        cfg = load_config_from_string(
            yaml.safe_dump(
                {
                    "base": "x",
                    "task": "sft",
                    "data": {"train": "t.jsonl", "max_length": 512},
                    "training": {"batch_size": 2},
                }
            )
        )
        assert hardware_fit.hardware_fit_input_from_config(cfg) is None


def _config_file(workdir: Path, base: Path) -> None:
    document = {
        "base": str(base),
        "task": "sft",
        "data": {"train": "train.jsonl", "max_length": 512},
        "training": {"batch_size": 2, "quantization": "none"},
    }
    (workdir / "soup.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")


MALFORMED = {
    "malformed": {"config_bytes": _malformed_json()},
    "out-of-range dimension": {
        "config": {**TINY_LLAMA, "hidden_size": BEYOND, "vocab_size": BEYOND}
    },
}


@pytest.fixture(params=sorted(MALFORMED))
def malformed_model(request, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _model(tmp_path, **MALFORMED[request.param])
    _config_file(tmp_path, model)
    return model


class TestEveryEntryPoint:
    def test_soup_plan_records_that_it_could_not_price_the_run(self, malformed_model):
        result = CliRunner().invoke(app, ["plan", "--config", "soup.yaml"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        plan = json.loads(Path("soup.tfstate").read_text(encoding="utf-8"))["plan"]
        assert plan["peak_vram_gb"] is None
        assert "config.json" in plan["peak_vram_reason"]

    def test_soup_apply_is_unaffected(self, malformed_model):
        runner = CliRunner()
        assert runner.invoke(app, ["plan", "--config", "soup.yaml"]).exit_code == 0
        for args in (["apply", "--dry-run"], ["apply"]):
            result = runner.invoke(app, [*args, "--config", "soup.yaml"])
            assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_soup_profile_prices_a_guessed_shape_and_names_the_file(self, malformed_model):
        result = CliRunner().invoke(app, ["profile", "--config", "soup.yaml", "--json"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        profile = json.loads(result.output)
        assert profile["shape_source"] == "guessed"
        assert any("config.json" in note for note in profile["notes"])

    def test_the_mcp_profile_tool_answers_the_same_way(self, malformed_model):
        profile = tool_profile({"config": "soup.yaml"})
        assert profile["shape_source"] == "guessed"
        assert any("config.json" in note for note in profile["notes"])

    def test_the_train_pre_flight_prices_it_without_raising(self, malformed_model, monkeypatch):
        printed = []

        class Screen:
            def print(self, renderable):
                printed.append(renderable)

        monkeypatch.setattr(train_mod, "console", Screen())
        text = (malformed_model.parent / "soup.yaml").read_text(encoding="utf-8")
        cfg = load_config_from_string(text)
        # A card too small for the guessed 7B model: the answer is a verdict, not a traceback.
        try:
            train_mod._hardware_fit_preflight(
                cfg, {"memory_total_bytes": int(2e9)}, allow_oom_attempt=False
            )
        except typer.Exit as exc:
            assert exc.exit_code in (0, 1)
        assert printed, "the pre-flight printed no verdict"


class TestNativeAdapterMetadata:
    @staticmethod
    def _adapter(tmp_path: Path, config=None, raw: bytes | None = None) -> Path:
        path = tmp_path / "adapter"
        path.mkdir()
        (path / "adapters.safetensors").write_bytes(b"boundary not loaded")
        if raw is not None:
            (path / "adapter_config.json").write_bytes(raw)
        else:
            (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        return path

    @staticmethod
    def _valid(**params) -> dict:
        return {
            "model": "owner/base",
            "fine_tune_type": "lora",
            "num_layers": 1,
            "lora_parameters": {
                "rank": 2, "scale": 4.0, "dropout": 0.0, "keys": ["q_proj"], **params,
            },
        }

    def test_the_control_is_accepted(self, tmp_path):
        path = self._adapter(tmp_path, self._valid())
        assert read_native_adapter(path)["model"] == "owner/base"

    def test_a_malformed_file_is_refused_naming_the_file(self, tmp_path):
        path = self._adapter(tmp_path, raw=_malformed_json())
        with pytest.raises(ValueError, match="adapter_config.json"):
            read_native_adapter(path)

    def test_an_oversized_file_is_refused_naming_the_file(self, tmp_path):
        padding = "x" * (5 * 1024 * 1024)
        path = self._adapter(tmp_path, {**self._valid(), "note": padding})
        with pytest.raises(ValueError, match="adapter_config.json"):
            read_native_adapter(path)

    @pytest.mark.parametrize(
        "scale", [OUT_OF_RANGE, pytest.param(-BEYOND, id="negative-out-of-range")]
    )
    def test_an_out_of_range_scale_is_refused(self, tmp_path, scale):
        path = self._adapter(tmp_path, self._valid(scale=scale))
        with pytest.raises(ValueError, match="scale"):
            read_native_adapter(path)

    def test_an_out_of_range_dropout_is_refused(self, tmp_path):
        path = self._adapter(tmp_path, self._valid(dropout=BEYOND))
        with pytest.raises(ValueError, match="dropout"):
            read_native_adapter(path)

    @pytest.mark.parametrize("rank", [4097, 10**9, OUT_OF_RANGE])
    def test_an_out_of_range_rank_is_refused(self, tmp_path, rank):
        path = self._adapter(tmp_path, self._valid(rank=rank))
        with pytest.raises(ValueError, match="rank"):
            read_native_adapter(path)

    def test_the_largest_rank_is_accepted(self, tmp_path):
        path = self._adapter(tmp_path, self._valid(rank=4096))
        assert read_native_adapter(path)["lora_parameters"]["rank"] == 4096

    @pytest.mark.parametrize("layers", [2**16 + 1, OUT_OF_RANGE])
    def test_an_out_of_range_layer_count_is_refused(self, tmp_path, layers):
        config = {**self._valid(), "num_layers": layers}
        path = self._adapter(tmp_path, config)
        with pytest.raises(ValueError, match="num_layers"):
            read_native_adapter(path)

    def test_too_many_target_keys_are_refused(self, tmp_path):
        keys = [f"layers.{i}.q_proj" for i in range(1025)]
        path = self._adapter(tmp_path, self._valid(keys=keys))
        with pytest.raises(ValueError, match="keys"):
            read_native_adapter(path)

    def test_a_very_long_target_key_is_refused(self, tmp_path):
        path = self._adapter(tmp_path, self._valid(keys=["q" * 257]))
        with pytest.raises(ValueError, match="keys"):
            read_native_adapter(path)

    def test_the_transformers_path_leaves_a_malformed_file_to_peft(self, tmp_path):
        path = self._adapter(tmp_path, raw=_malformed_json())
        reject_native_adapter(str(path))  # PEFT reports its own error for this file later

    def test_the_transformers_path_still_spots_a_native_adapter(self, tmp_path):
        path = self._adapter(tmp_path, self._valid())
        with pytest.raises(ValueError, match="select --backend mlx"):
            reject_native_adapter(str(path))
