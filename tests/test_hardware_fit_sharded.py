"""The train pre-flight prices a sharded run (``--fsdp``, ``--deepspeed``) per card (#2300).

Under ``--fsdp full_shard`` / ``full_offload`` or DeepSpeed ZeRO-3 each card holds a share
of the weights and of the adapters; under those and ``--fsdp shard_grad`` / ZeRO-2 a share
of the gradients and the optimizer state. The pre-flight compared the unsharded total with
one card, so FSDP-QLoRA of a 70B model in 4-bit on two 24 GB cards was refused although its
weights are about 22 GB a card once sharded.

The floor (weights, CUDA context, the trained parameters' adapters, gradients and optimizer
state) is now per card for a run sharded across more than one process. The estimate is
fitted on single-card runs, so for such a run its peak only warns, as outside the calibrated
range, and the panel says the run is sharded. Data-parallel ``--gpus N`` without sharding
and every single-process run are priced as before.

No GPU and no distributed launch: the card is a fake ``gpu_info``, the model's
``config.json`` is the dict a Hub lookup would return, a launched rank is its
``RANK`` / ``WORLD_SIZE`` environment, and the command stops right after the pre-flight.
"""

from __future__ import annotations

import io
import json
import tempfile

import pytest
import typer
import yaml
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import train as train_mod
from soup_cli.commands.train import _hardware_fit_preflight
from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import deepspeed as ds
from soup_cli.utils import hardware_fit, model_shape, topology
from soup_cli.utils.vram_estimate import GB, OVERHEAD_FIXED_BYTES
from tests.conftest import strip_ansi

# Llama-3.1-70B: 70.55B parameters; its 4-bit weights are about 45 GB with the bf16
# embeddings and output head, 22 GB a card across two.
_LLAMA_70B = {
    "model_type": "llama",
    "hidden_size": 8192,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "intermediate_size": 28672,
    "vocab_size": 128256,
}
_LLAMA_70B_PARAMS = 70_553_706_496

# Llama-2-7B: 6.74B parameters; a full fine-tune holds 27 GB of fp32 weights, as much again
# in gradients and twice that in AdamW state.
_LLAMA_7B = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "intermediate_size": 11008,
    "vocab_size": 32000,
}
_LLAMA_7B_PARAMS = 6_738_415_616

_CARD_24_GB = {"memory_total_bytes": int(24e9)}
_CARD_40_GB = {"memory_total_bytes": int(40e9)}
_FULL = {"quantization": "none", "lora": {"r": 0}}

#: Unset for every test, and restored after it: a launched rank's ``soup train`` exports
#: the NCCL hints with ``os.environ.setdefault``.
_DISTRIBUTED_ENV = (
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "ACCELERATE_MIXED_PRECISION",
    "ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_FSDP",
    "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_NVLS_ENABLE",
)

GATE = "Hardware-fit gate"
WARNING = "Hardware-fit warning"
NOT_EXTRAPOLATED = "not an extrapolation"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _yaml(**training) -> str:
    return yaml.safe_dump({
        "base": "someone/model",
        "task": "sft",
        "data": {"train": "train.jsonl", "format": "alpaca", "max_length": 2048},
        "training": {"quantization": "4bit", "batch_size": 4, **training},
    })


@pytest.fixture(autouse=True)
def _single_process(monkeypatch, tmp_path):
    """No launcher environment unless a test sets one; preset temp files under tmp_path."""
    for key in _DISTRIBUTED_ENV:
        # setenv first, so the teardown restores the key even when it was unset.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


@pytest.fixture
def shape_of(monkeypatch):
    def pick(config: dict, params: int) -> None:
        monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kw: (dict(config), None))
        monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: params)

    return pick


@pytest.fixture
def screen(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(
        train_mod, "console", Console(file=buffer, width=400, force_terminal=False)
    )
    return lambda: _plain(buffer.getvalue())


def _run(cfg_text: str, *, card, **sharding) -> int | None:
    """Run the pre-flight; the exit code if it refused, else ``None``."""
    try:
        _hardware_fit_preflight(
            load_config_from_string(cfg_text), card, allow_oom_attempt=False, **sharding
        )
    except typer.Exit as exc:
        return exc.exit_code
    return None


# --- soup train, stopped right after the pre-flight -----------------------------------


PASSED = 99


@pytest.fixture
def train_cli(tmp_path, monkeypatch, shape_of, screen):
    """``soup train`` in ``tmp_path`` on a fake 24 GB card; exit 99 = the pre-flight passed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"instruction": "hi", "output": "hello"}) + "\n", encoding="utf-8"
    )
    shape_of(_LLAMA_70B, _LLAMA_70B_PARAMS)
    monkeypatch.setattr(topology, "_detected_gpu_count", lambda: 2)
    monkeypatch.setattr(topology, "detect_topology", lambda: {
        "gpu_count": 2, "interconnect": "pcie",
    })
    monkeypatch.setattr(train_mod, "detect_device", lambda **kw: ("cuda", "fake 24 GB card"))
    monkeypatch.setattr(train_mod, "get_gpu_info", lambda **kw: dict(_CARD_24_GB))
    calls = []
    real = train_mod._hardware_fit_preflight

    def stop_after_preflight(*args, **kwargs):
        calls.append(kwargs)
        real(*args, **kwargs)
        raise typer.Exit(PASSED)

    monkeypatch.setattr(train_mod, "_hardware_fit_preflight", stop_after_preflight)

    def invoke(*args: str, training: dict | None = None, world_size: int | None = None):
        (tmp_path / "soup.yaml").write_text(_yaml(**(training or {})), encoding="utf-8")
        if world_size is not None:
            monkeypatch.setenv("RANK", "0")
            monkeypatch.setenv("WORLD_SIZE", str(world_size))
        result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--yes", *args])
        assert calls, (result.output, repr(result.exception))
        return result.exit_code, screen(), calls[-1]

    return invoke


class TestSeventyBOnTwo24GBCards:
    """The issue's example: FSDP-QLoRA of a 70B model in 4-bit across two 24 GB cards."""

    def test_full_shard_in_a_launched_rank_is_not_refused_and_says_sharded(self, train_cli):
        code, text, kwargs = train_cli("--fsdp", "full_shard", world_size=2)
        assert code == PASSED, text
        assert (kwargs["sharding"], kwargs["num_processes"]) == ("full_shard", 2)
        assert WARNING in text
        assert GATE not in text
        assert (
            "Sharded across 2 processes (--fsdp full_shard): the floor is per card; "
            "the other figures are for one unsharded card." in text
        )
        assert "70.6B parameters, sharded across 2 processes" in text
        assert "Warning only: launching anyway." in text

    def test_full_shard_from_the_launcher_side_counts_gpus_times_nodes(self, train_cli):
        # Not yet launched: --dry-run skips the re-exec and reaches the pre-flight.
        code, text, kwargs = train_cli("--fsdp", "full_shard", "--gpus", "2", "--dry-run")
        assert code == PASSED, text
        assert (kwargs["sharding"], kwargs["num_processes"]) == ("full_shard", 2)
        assert "Sharded across 2 processes (--fsdp full_shard)" in text

    def test_without_fsdp_it_is_still_refused(self, train_cli):
        code, text, kwargs = train_cli(world_size=2)
        assert code == 1
        assert kwargs.get("sharding") is None
        assert GATE in text
        assert NOT_EXTRAPOLATED in text
        assert "Sharded across" not in text

    def test_data_parallel_gpus_2_is_unchanged(self, train_cli):
        _, single, _ = train_cli()
        code, text, kwargs = train_cli("--gpus", "2", "--no-reexec", world_size=2)
        assert code == 1
        assert kwargs.get("sharding") is None
        assert GATE in text
        assert "Sharded across" not in text
        # The same panel as the single-process run, after that run's own output.
        assert text.endswith(single[single.index(GATE):])

    def test_full_shard_in_one_process_is_priced_as_today(self, train_cli):
        _, single, _ = train_cli()
        code, text, _ = train_cli("--fsdp", "full_shard", world_size=1)
        assert code == 1
        assert "Sharded across" not in text
        assert text.endswith(single[single.index(GATE):])

    def test_shard_grad_does_not_divide_the_weights(self, train_cli):
        code, text, kwargs = train_cli("--fsdp", "shard_grad", world_size=2)
        assert code == 1
        assert kwargs["sharding"] == "shard_grad"
        assert GATE in text
        assert NOT_EXTRAPOLATED in text
        assert "Sharded across 2 processes (--fsdp shard_grad)" in text

    def test_zero3_preset_divides_the_weights(self, train_cli):
        code, text, kwargs = train_cli("--deepspeed", "zero3", world_size=2)
        assert code == PASSED, text
        assert (kwargs["sharding"], kwargs["num_processes"]) == ("zero3", 2)
        assert "Sharded across 2 processes (DeepSpeed ZeRO-3)" in text

    def test_zero2_preset_does_not_divide_the_weights(self, train_cli):
        code, text, kwargs = train_cli("--deepspeed", "zero2", world_size=2)
        assert code == 1
        assert kwargs["sharding"] == "zero2"
        assert GATE in text
        assert "Sharded across 2 processes (DeepSpeed ZeRO-2)" in text

    @pytest.mark.parametrize(("stage", "passed"), [(3, True), (2, False)])
    def test_the_stage_is_read_from_a_user_json(self, train_cli, tmp_path, stage, passed):
        config = ds.get_deepspeed_config("zero3" if stage == 3 else "zero2")
        (tmp_path / "my_ds.json").write_text(json.dumps(config), encoding="utf-8")
        code, text, kwargs = train_cli("--deepspeed", "my_ds.json", world_size=2)
        assert kwargs["sharding"] == f"zero{stage}"
        assert (code == PASSED) is passed, text
        assert f"Sharded across 2 processes (DeepSpeed ZeRO-{stage})" in text


# --- where the mode and the process count come from ------------------------------------


def _sharding(*args):
    # Looked up on the module at call time, so the file collects on a tree without it.
    return train_mod._preflight_sharding(*args)


class TestPreflightSharding:
    @pytest.mark.parametrize("preset", ["full_shard", "shard_grad", "full_offload"])
    def test_the_fsdp_preset_is_the_mode(self, preset):
        assert _sharding(preset, None, 2, 1) == (preset, 2)

    @pytest.mark.parametrize(
        ("preset", "mode"),
        [
            ("zero2", "zero2"), ("zero3", "zero3"),
            # Offload variants are priced as their stage without offload.
            ("zero2_offload", "zero2"), ("zero3_offload", "zero3"),
            ("zero++", "zero3"),
        ],
    )
    def test_a_deepspeed_preset_is_read_from_the_file_it_writes(self, tmp_path, preset, mode):
        path = train_mod._resolve_deepspeed(preset)
        assert path.startswith(str(tmp_path))
        assert _sharding(None, path, 4, 1) == (mode, 4)

    @pytest.mark.parametrize(
        ("body", "mode"),
        [
            ({"zero_optimization": {"stage": 3}}, "zero3"),
            ({"zero_optimization": {"stage": 2}}, "zero2"),
            ({"zero_optimization": {"stage": 1}}, None),
            ({"zero_optimization": {"stage": True}}, None),
            ({"zero_optimization": {"stage": "3"}}, None),
            ({"train_batch_size": 8}, None),
            ([], None),
        ],
    )
    def test_a_user_json_is_read_by_its_stage(self, tmp_path, monkeypatch, body, mode):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "user.json").write_text(json.dumps(body), encoding="utf-8")
        path = train_mod._resolve_deepspeed("user.json")
        assert _sharding(None, path, 2, 1) == (mode, 2)

    def test_an_unreadable_deepspeed_file_is_no_sharding(self, tmp_path):
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        assert _sharding(None, str(tmp_path / "broken.json"), 2, 1) == (None, 2)
        assert _sharding(None, str(tmp_path / "missing.json"), 2, 1) == (None, 2)

    @pytest.mark.parametrize(
        ("num_gpus", "nodes", "processes"), [(2, 1, 2), (4, 2, 8), (None, 1, 1), (0, 1, 1)]
    )
    def test_outside_a_launcher_it_is_gpus_times_nodes(self, num_gpus, nodes, processes):
        assert _sharding("full_shard", None, num_gpus, nodes) == (
            "full_shard", processes,
        )

    def test_inside_a_launched_rank_world_size_wins(self, monkeypatch):
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "16")
        assert _sharding("full_shard", None, None, 1) == ("full_shard", 16)
        assert _sharding("full_shard", None, 2, 2) == ("full_shard", 16)

    def test_a_stray_world_size_without_a_launcher_is_ignored(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "16")
        assert _sharding("full_shard", None, None, 1) == ("full_shard", 1)

    def test_a_junk_world_size_falls_back_to_the_launcher_count(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "two")
        assert _sharding("full_shard", None, 2, 1) == ("full_shard", 2)


# --- the floor per card ----------------------------------------------------------------


def _parts(shape_of, model, params, **training):
    shape_of(model, params)
    cfg = load_config_from_string(_yaml(**training))
    inp = train_mod._build_hardware_fit_input(cfg)
    return cfg, inp, hardware_fit.estimate_peak_vram_gb(inp)


_CONTEXT = OVERHEAD_FIXED_BYTES / GB


class TestFloorPerCard:
    """The exact arithmetic: which terms each mode divides by the process count."""

    @pytest.mark.parametrize("mode", ["full_shard", "full_offload", "zero3"])
    def test_parameter_sharding_divides_every_term_but_the_context(self, shape_of, mode):
        cfg, inp, b = _parts(shape_of, _LLAMA_70B, _LLAMA_70B_PARAMS)
        floor, terms = hardware_fit.unextrapolated_floor(
            cfg, inp, b, sharding=mode, num_processes=4
        )
        assert floor == pytest.approx(
            b.weights_gb / 4 + _CONTEXT + b.adapters_gb / 4
            + b.gradients_gb / 4 + b.optimizer_gb / 4
        )
        assert terms == ("weights", "CUDA context", "adapters", "gradients", "optimizer state")

    @pytest.mark.parametrize("mode", ["shard_grad", "zero2"])
    def test_state_sharding_divides_only_the_gradients_and_optimizer(self, shape_of, mode):
        cfg, inp, b = _parts(shape_of, _LLAMA_70B, _LLAMA_70B_PARAMS)
        floor, _ = hardware_fit.unextrapolated_floor(cfg, inp, b, sharding=mode, num_processes=4)
        assert floor == pytest.approx(
            b.weights_gb + _CONTEXT + b.adapters_gb + b.gradients_gb / 4 + b.optimizer_gb / 4
        )

    @pytest.mark.parametrize(
        ("mode", "processes"), [(None, 4), ("zero1", 4), ("full_shard", 1), ("zero3", 0)]
    )
    def test_anything_else_is_one_unsharded_card(self, shape_of, mode, processes):
        cfg, inp, b = _parts(shape_of, _LLAMA_70B, _LLAMA_70B_PARAMS)
        assert hardware_fit.unextrapolated_floor(
            cfg, inp, b, sharding=mode, num_processes=processes
        ) == hardware_fit.unextrapolated_floor(cfg, inp, b)

    def test_the_seventy_b_case_fits_24_gb_only_with_every_term_divided(self, shape_of):
        # Each divided term matters on its own: undividing any one is more than 24 GB.
        cfg, inp, b = _parts(shape_of, _LLAMA_70B, _LLAMA_70B_PARAMS)
        floor, _ = hardware_fit.unextrapolated_floor(
            cfg, inp, b, sharding="full_shard", num_processes=2
        )
        assert floor < 24
        for term in (b.weights_gb, b.adapters_gb, b.gradients_gb, b.optimizer_gb):
            assert floor + term / 2 > 24


class TestFullFineTuneThroughTheGate:
    """A 7B full fine-tune with AdamW: 27 GB weights, 27 GB gradients, 54 GB state."""

    @pytest.mark.parametrize("mode", ["shard_grad", "zero2"])
    def test_state_sharding_across_8_lets_it_onto_40_gb(self, shape_of, screen, mode):
        shape_of(_LLAMA_7B, _LLAMA_7B_PARAMS)
        assert _run(_yaml(**_FULL), card=_CARD_40_GB) == 1
        assert _run(_yaml(**_FULL), card=_CARD_40_GB, sharding=mode, num_processes=8) is None
        text = screen()
        assert WARNING in text
        assert "Sharded across 8 processes" in text

    @pytest.mark.parametrize("mode", ["shard_grad", "zero2"])
    def test_state_sharding_keeps_the_weights_whole(self, shape_of, screen, mode):
        # 27.5 GB of weights and context plus 81 / 4 = 20.2 GB of state: 47.7 GB.
        shape_of(_LLAMA_7B, _LLAMA_7B_PARAMS)
        assert _run(_yaml(**_FULL), card=_CARD_40_GB, sharding=mode, num_processes=4) == 1
        text = screen()
        assert GATE in text
        assert "alone take 47.7 GB" in text

    def test_zero3_across_4_divides_the_weights_too(self, shape_of, screen):
        # 6.7 + 0.5 + 20.2 = 27.5 GB a card.
        shape_of(_LLAMA_7B, _LLAMA_7B_PARAMS)
        assert _run(_yaml(**_FULL), card=_CARD_40_GB, sharding="zero3", num_processes=4) is None
        assert "Sharded across 4 processes (DeepSpeed ZeRO-3)" in screen()

    def test_the_peak_inside_the_calibrated_range_only_warns_when_sharded(
        self, shape_of, screen
    ):
        # A model inside the range is refused on its estimate alone; sharded, it is not.
        small = {
            "model_type": "qwen2", "hidden_size": 896, "num_hidden_layers": 24,
            "num_attention_heads": 14, "num_key_value_heads": 2, "intermediate_size": 4864,
            "vocab_size": 151936, "tie_word_embeddings": True,
        }
        shape_of(small, 494_032_768)
        card = {"memory_total_bytes": int(6e9)}
        assert _run(_yaml(), card=card) == 1
        assert "inside the calibrated range" in screen()
        assert _run(_yaml(), card=card, sharding="full_shard", num_processes=2) is None
        text = screen()
        assert "sharded across 2 processes; the estimate may overstate the peak" in text
