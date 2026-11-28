"""``training.l2l`` on a MoE model streamed with the per_expert layout.

Four seams. The config keys (``load_config_from_string``). The launch rules
(``refuse_unsupported_l2l``, ``resolve_l2l_recipe``). The expert-buffer pre-flight
(``_expert_buffer_requirements`` and ``_stream_budget_lines``). And, on CPU, the shipped L2L
step on a tiny DeepSeek-V3 streamed through the expert buffer pool, which must give the same
loss and the same LoRA gradients, ``torch.equal``, as the ordinary accumulation step on the same
model, in fp32 and under bf16 autocast, with the expert levers off and on.

Adapter values come from integer arithmetic, so no comparison depends on ``torch.randn``.
"""

from __future__ import annotations

import re
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string

EXPERT_LEVERS = ("l2l_two_read_experts", "l2l_grouped_expert_routes")


def _yaml(
    *, base: str = "hf-internal-testing/tiny-random-LlamaForCausalLM", **training: Any
) -> str:
    fields = {
        "batch_size": 1,
        "quantization": "none",
        "stream_layers": True,
        "gradient_accumulation_steps": 2,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
    }
    fields.update(training)
    return yaml.safe_dump(
        {
            "base": base,
            "task": "sft",
            "data": {"train": "data.jsonl"},
            "training": fields,
        }
    )


# --------------------------------------------------------------------------
# config keys
# --------------------------------------------------------------------------
def test_the_expert_levers_are_off_by_default():
    cfg = load_config_from_string(_yaml())
    for name in EXPERT_LEVERS:
        assert getattr(cfg.training, name) is False


@pytest.mark.parametrize("name", EXPERT_LEVERS)
def test_a_lever_set_while_l2l_is_off_is_refused_with_the_field_named(name):
    pattern = rf"training\.{name} is set but training\.l2l is false.*Set l2l=true or drop {name}"
    with pytest.raises(ValueError, match=pattern):
        load_config_from_string(_yaml(**{name: True, "stream_expert_layout": "per_expert"}))


@pytest.mark.parametrize("name", EXPERT_LEVERS)
@pytest.mark.parametrize("value", [1, 0, "yes", "true"])
def test_a_lever_takes_a_bool_only(name, value):
    with pytest.raises(ValueError, match=rf"training\.{name} must be true or false"):
        load_config_from_string(
            _yaml(l2l=True, stream_expert_layout="per_expert", **{name: value})
        )


@pytest.mark.parametrize("name", EXPERT_LEVERS)
def test_a_lever_on_the_fused_layout_is_refused_with_the_layout_named(name):
    pattern = rf"training\.{name} acts on the routed experts of a per_expert cache.*per_expert"
    with pytest.raises(ValueError, match=pattern):
        load_config_from_string(_yaml(l2l=True, **{name: True}))


@pytest.mark.parametrize("name", EXPERT_LEVERS)
def test_a_lever_loads_with_l2l_and_the_per_expert_layout(name):
    cfg = load_config_from_string(
        _yaml(l2l=True, stream_expert_layout="per_expert", **{name: True})
    )
    assert getattr(cfg.training, name) is True
    other = [lever for lever in EXPERT_LEVERS if lever != name][0]
    assert getattr(cfg.training, other) is False


def test_l2l_with_the_per_expert_layout_loads_without_a_lever():
    cfg = load_config_from_string(_yaml(l2l=True, stream_expert_layout="per_expert"))
    assert cfg.training.l2l is True
    assert cfg.training.stream_expert_layout == "per_expert"


# --------------------------------------------------------------------------
# launch rules
# --------------------------------------------------------------------------
def _model_config(arch: str) -> Any:
    return SimpleNamespace(model_type=arch)


def test_the_launch_admits_deepseek_v3_with_the_per_expert_layout():
    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l

    refuse_unsupported_l2l(_model_config("deepseek_v3"), n_gpu=1, expert_layout="per_expert")


@pytest.mark.parametrize("layout", ["fused", ""])
def test_the_launch_refuses_deepseek_v3_on_another_layout_naming_the_layout(layout):
    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l

    with pytest.raises(ValueError) as caught:
        refuse_unsupported_l2l(_model_config("deepseek_v3"), n_gpu=1, expert_layout=layout)
    message = str(caught.value)
    assert "training.stream_expert_layout='per_expert'" in message
    assert f"got {layout!r}" in message


def test_the_launch_still_refuses_an_unlisted_architecture_and_many_gpus():
    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l

    with pytest.raises(ValueError, match="does not support model_type 'qwen3_moe'"):
        refuse_unsupported_l2l(_model_config("qwen3_moe"), n_gpu=1, expert_layout="per_expert")
    with pytest.raises(ValueError, match="single-GPU"):
        refuse_unsupported_l2l(_model_config("deepseek_v3"), n_gpu=2, expert_layout="per_expert")


def test_the_moe_refusal_of_the_earlier_version_is_gone():
    from soup_cli.utils import l2l_step

    for name in ("MOE_L2L_REFUSAL", "MOE_L2L_ENABLED", "refuse_moe_l2l", "MOE_ARCHS"):
        assert not hasattr(l2l_step, name), name


def _tcfg(**overrides: Any) -> Any:
    values = dict(
        l2l=True,
        l2l_activation_codec="none",
        l2l_auto_k=False,
        l2l_spill_dir=None,
        gradient_accumulation_steps=4,
        batch_size=1,
        stream_expert_layout="per_expert",
        l2l_two_read_experts=False,
        l2l_grouped_expert_routes=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_the_auto_k_planner_stays_refused_on_a_moe_run():
    from soup_cli.utils.l2l_recipe import validate_l2l_options

    with pytest.raises(ValueError, match="training.l2l_auto_k is not available yet"):
        validate_l2l_options(_tcfg(l2l_auto_k=True))


def test_the_recipe_records_the_expert_levers():
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe

    plain = resolve_l2l_recipe(_tcfg())
    assert (plain.two_read_experts, plain.grouped_expert_routes) == (False, False)
    both = resolve_l2l_recipe(_tcfg(l2l_two_read_experts=True, l2l_grouped_expert_routes=True))
    assert (both.two_read_experts, both.grouped_expert_routes) == (True, True)


@pytest.mark.parametrize("name", EXPERT_LEVERS)
def test_a_resume_with_another_expert_lever_is_refused_with_the_lever_named(
    tmp_path, monkeypatch, name
):
    monkeypatch.chdir(tmp_path)
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe, validate_resume, write_recipe

    write_recipe(tmp_path, resolve_l2l_recipe(_tcfg()))
    validate_resume(tmp_path, resolve_l2l_recipe(_tcfg()))
    key = name.removeprefix("l2l_")
    pattern = rf"cannot resume.*{key}=False.*{key}=True"
    with pytest.raises(ValueError, match=pattern):
        validate_resume(tmp_path, resolve_l2l_recipe(_tcfg(**{name: True})))


def test_a_recipe_written_before_the_levers_existed_resumes_as_off(tmp_path):
    import json

    from soup_cli.utils.l2l_recipe import RECIPE_NAME, load_recipe

    (tmp_path / RECIPE_NAME).write_text(
        json.dumps(
            {"requested_k": 4, "effective_k": 4, "activation_codec": "none", "auto_k": False}
        ),
        encoding="utf-8",
    )
    recipe = load_recipe(tmp_path)
    assert (recipe.two_read_experts, recipe.grouped_expert_routes) == (False, False)


class _ReachedCheckpointLoadError(Exception):
    """The wrapper passed validation and handed the checkpoint to the trainer."""


def _resume_wrapper(destination: Path, *, l2l: bool = True, **training: Any) -> Any:
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe

    cfg = load_config_from_string(
        _yaml(l2l=l2l, stream_expert_layout="per_expert", gradient_accumulation_steps=4, **training)
    )
    wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
    wrapper.config = cfg
    wrapper._output_dir = str(destination)
    wrapper._quest_metadata = None
    wrapper._stream_loss_chunk_hook = None
    wrapper._l2l_recipe = resolve_l2l_recipe(cfg.training) if l2l else None
    wrapper.trainer = SimpleNamespace(
        model=SimpleNamespace(named_parameters=lambda: ()),
        args=SimpleNamespace(),
        train=Mock(side_effect=_ReachedCheckpointLoadError),
        add_callback=Mock(),
        remove_callback=Mock(),
    )
    return wrapper


@pytest.fixture
def resume_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "source" / "checkpoint-4"
    checkpoint.mkdir(parents=True)
    destination = tmp_path / "destination"
    destination.mkdir()
    return checkpoint, destination


@pytest.mark.parametrize("name", EXPERT_LEVERS)
def test_resume_refuses_a_source_expert_mismatch_before_writing_destination(resume_dirs, name):
    from soup_cli.utils.l2l_recipe import RECIPE_NAME, resolve_l2l_recipe, write_recipe

    checkpoint, destination = resume_dirs
    write_recipe(checkpoint.parent, resolve_l2l_recipe(_tcfg()))
    wrapper = _resume_wrapper(destination, **{name: True})
    with pytest.raises(ValueError) as caught:
        wrapper.train(resume_from_checkpoint=str(checkpoint))
    field = name.removeprefix("l2l_")
    assert str(caught.value) == (
        f"cannot resume: the checkpoint was trained with {field}=False, "
        f"but this run has {field}=True. Restore the old value in the config."
    )
    wrapper.trainer.train.assert_not_called()
    assert not (destination / RECIPE_NAME).exists()


def test_resume_uses_checkpoint_recipe_over_parent_and_destination_recipes(resume_dirs):
    from soup_cli.utils.l2l_recipe import load_recipe, resolve_l2l_recipe, write_recipe

    checkpoint, destination = resume_dirs
    recipe = resolve_l2l_recipe(_tcfg(l2l_two_read_experts=True))
    unrelated = resolve_l2l_recipe(_tcfg(l2l_grouped_expert_routes=True))
    write_recipe(checkpoint.parent, unrelated)
    write_recipe(destination, unrelated)
    write_recipe(checkpoint, recipe)
    wrapper = _resume_wrapper(destination, l2l_two_read_experts=True)

    with pytest.raises(_ReachedCheckpointLoadError):
        wrapper.train(resume_from_checkpoint=str(checkpoint))
    wrapper.trainer.train.assert_called_once_with(resume_from_checkpoint=str(checkpoint))
    assert load_recipe(destination) == recipe
    assert load_recipe(checkpoint) == recipe
    assert load_recipe(checkpoint.parent) == unrelated


def test_resume_refuses_a_source_l2l_recipe_when_l2l_is_off(resume_dirs):
    from soup_cli.utils.l2l_recipe import RECIPE_NAME, resolve_l2l_recipe, write_recipe

    checkpoint, destination = resume_dirs
    write_recipe(checkpoint.parent, resolve_l2l_recipe(_tcfg()))
    wrapper = _resume_wrapper(destination, l2l=False)
    with pytest.raises(ValueError) as caught:
        wrapper.train(resume_from_checkpoint=str(checkpoint))
    assert str(caught.value) == (
        "cannot resume: the checkpoint was trained with training.l2l "
        "(effective_k=4, activation_codec='none'), but this run has training.l2l off. "
        "Set training.l2l to true with the same settings."
    )
    wrapper.trainer.train.assert_not_called()
    assert not (destination / RECIPE_NAME).exists()


@pytest.mark.parametrize("l2l", [False, True])
def test_resume_accepts_a_source_without_a_recipe_even_with_a_destination_recipe(resume_dirs, l2l):
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe, write_recipe

    checkpoint, destination = resume_dirs
    write_recipe(destination, resolve_l2l_recipe(_tcfg(l2l_grouped_expert_routes=True)))
    wrapper = _resume_wrapper(destination, l2l=l2l)
    with pytest.raises(_ReachedCheckpointLoadError):
        wrapper.train(resume_from_checkpoint=str(checkpoint))
    wrapper.trainer.train.assert_called_once_with(resume_from_checkpoint=str(checkpoint))


def test_a_resume_into_the_same_directory_does_not_relabel_the_legacy_checkpoint(resume_dirs):
    checkpoint, _destination = resume_dirs
    first = _resume_wrapper(checkpoint.parent, l2l_two_read_experts=True)
    with pytest.raises(_ReachedCheckpointLoadError):
        first.train(resume_from_checkpoint=str(checkpoint))

    retry = _resume_wrapper(checkpoint.parent, l2l=False)
    with pytest.raises(_ReachedCheckpointLoadError):
        retry.train(resume_from_checkpoint=str(checkpoint))
    retry.trainer.train.assert_called_once_with(resume_from_checkpoint=str(checkpoint))


@pytest.mark.parametrize("l2l", [False, True])
def test_new_checkpoint_keeps_its_recipe_when_the_run_root_recipe_changes(resume_dirs, l2l):
    from dataclasses import replace

    from soup_cli.utils.l2l_recipe import RECIPE_NAME, load_recipe, resolve_l2l_recipe, write_recipe

    checkpoint, destination = resume_dirs
    root = checkpoint.parent
    # The older source checkpoint predates the L2L run recorded in its parent.
    write_recipe(root, replace(resolve_l2l_recipe(_tcfg()), start_step=4))
    wrapper = _resume_wrapper(root, l2l=l2l, l2l_two_read_experts=l2l)
    new_checkpoint = root / "checkpoint-5"

    def save_checkpoint(**_kwargs: Any) -> None:
        new_checkpoint.mkdir()
        args = SimpleNamespace(output_dir=str(root), should_save=True)
        state = SimpleNamespace(global_step=5)
        control = SimpleNamespace()
        for call in wrapper.trainer.add_callback.call_args_list:
            call.args[0].on_save(args, state, control)
        raise _ReachedCheckpointLoadError

    wrapper.trainer.train.side_effect = save_checkpoint
    with pytest.raises(_ReachedCheckpointLoadError):
        wrapper.train(resume_from_checkpoint=str(checkpoint))
    assert (new_checkpoint / RECIPE_NAME).is_file()
    expected = replace(wrapper._l2l_recipe, start_step=4) if l2l else None
    assert load_recipe(new_checkpoint) == expected

    # The root now describes a different run. The newly saved checkpoint remains authoritative.
    write_recipe(root, resolve_l2l_recipe(_tcfg(l2l_grouped_expert_routes=True)))
    retry = _resume_wrapper(destination, l2l=l2l, l2l_two_read_experts=l2l)
    with pytest.raises(_ReachedCheckpointLoadError):
        retry.train(resume_from_checkpoint=str(new_checkpoint))
    retry.trainer.train.assert_called_once_with(resume_from_checkpoint=str(new_checkpoint))


# --------------------------------------------------------------------------
# expert-buffer pre-flight
# --------------------------------------------------------------------------
def test_a_fused_cache_needs_no_expert_buffer(tmp_path):
    from soup_cli.trainer.stream_setup import _expert_buffer_requirements

    index = SimpleNamespace(expert_layout="")
    assert _expert_buffer_requirements(str(tmp_path), index, device="cpu") == (0, 0)


def test_a_per_expert_cache_without_expert_files_is_refused(tmp_path, monkeypatch):
    from soup_cli.trainer.stream_setup import _expert_buffer_requirements
    from soup_cli.utils import layer_shard

    monkeypatch.setattr(layer_shard, "expert_paths", lambda shard_dir, index: {})
    index = SimpleNamespace(expert_layout="per_expert")
    with pytest.raises(ValueError, match="no expert files"):
        _expert_buffer_requirements(str(tmp_path), index, device="cpu")


def _tiny_moe(tmp_path: Any) -> Any:
    pytest.importorskip("peft")
    from tests.test_expert_stream import _shard, _tiny_deepseek_v3

    weights, resident = _tiny_deepseek_v3(tmp_path)
    return weights, _shard(tmp_path, weights, expert_layout="per_expert")


def test_the_expert_buffer_requirements_are_the_header_plan_of_the_largest_layer(tmp_path):
    from soup_cli.trainer.stream_setup import _expert_buffer_requirements
    from soup_cli.utils.expert_shard import ExpertShardReader
    from soup_cli.utils.expert_stream import expert_allocation_budget, plan_expert_buffer
    from soup_cli.utils.layer_shard import expert_paths

    _weights, index = _tiny_moe(tmp_path)
    arena, staging = _expert_buffer_requirements(str(tmp_path / "shards"), index, device="cpu")
    assert arena > 0 and staging > 0
    wanted_arena = wanted_staging = 0
    for path in expert_paths(str(tmp_path / "shards"), index).values():
        reader = ExpertShardReader(
            path, count=index.expert_count, shapes=index.expert_shapes,
            quant_specs=index.expert_quant_specs, dtype=index.dtype,
        )
        try:
            plan = plan_expert_buffer(reader, dtype=index.dtype)
        finally:
            reader.close()
        wanted_arena = max(wanted_arena, expert_allocation_budget(plan, device="cpu"))
        wanted_staging = max(wanted_staging, plan.host_staging_bytes)
    assert (arena, staging) == (wanted_arena, wanted_staging)
    cuda_arena, _ = _expert_buffer_requirements(str(tmp_path / "shards"), index, device="cuda")
    assert cuda_arena == arena + 32 * 2**20


def test_the_preflight_charges_the_pool_arena_instead_of_the_expert_in_flight(monkeypatch):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from soup_cli.utils import layer_stream
    from soup_cli.utils.expert_stream import expert_stream_vram_bytes
    from tests.test_stream_expert_layout import (
        _SMALL_SHAPES,
        _Config,
        _index,
        _ModelConfig,
        _TrainConfig,
    )

    real = layer_stream.estimate_stream_peak_vram
    seen: Dict[str, Any] = {}

    def capture(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(layer_stream, "estimate_stream_peak_vram", capture)

    def forecast(pool_bytes: int) -> str:
        lines, _ = StreamingSetupMixin()._stream_budget_lines(
            _Config(), _TrainConfig(), model_config=_ModelConfig(), layer_bytes=1000,
            embed_bytes=0, index=_index("per_expert"), on_cuda=False,
            expert_pool_bytes=pool_bytes,
        )
        return "\n".join(lines)

    in_flight = expert_stream_vram_bytes(_SMALL_SHAPES, dtype_bytes=2, quantised=False)
    forecast(0)
    assert seen["persistent_bytes"] == in_flight
    text = forecast(123_456_789)
    assert seen["persistent_bytes"] == 123_456_789
    assert "experts      0.123 GB for the expert buffer pool arena (training.l2l)" in text
    assert "routed expert in flight" not in text


def test_a_run_without_a_pool_keeps_the_in_flight_line(monkeypatch):
    from tests.test_stream_expert_layout import _charged

    _charged_bytes, text = _charged(monkeypatch, "per_expert")
    assert "for the routed expert in flight" in text
    assert "expert buffer pool" not in text


# --------------------------------------------------------------------------
# the setup wiring: levers and pool only under training.l2l
# --------------------------------------------------------------------------
def test_the_setup_reads_the_levers_and_builds_the_pool_only_under_l2l_on_per_expert():
    source = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "trainer"
    text = (source / "stream_setup.py").read_text(encoding="utf-8")
    assert "l2l_experts = bool(getattr(tcfg, \"l2l\", False)) and bool(expert_layout)" in text
    call = re.search(
        r"\*\*\(\s*\{\s*\"expert_pool\": True,.*?\}\s*if l2l_experts\s*else \{\}\s*\)", text, re.S
    )
    assert call is not None
    assert "if l2l_experts" in call.group(0)
    assert "two_read_experts" in call.group(0) and "grouped_expert_routes" in call.group(0)


class _ReachedRuntimeBuilderError(Exception):
    """Stop streaming setup before allocating the runtime."""


@pytest.fixture
def expert_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Mock]:
    pytest.importorskip("peft")
    from transformers import AutoTokenizer

    from soup_cli.trainer import stream_setup
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import kimi_k2, layer_stream, layer_stream_runtime, spectrum_scan
    from tests.test_expert_stream import _tiny_deepseek_v3

    weights, model = _tiny_deepseek_v3(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SOUP_LAYER_STREAM_STRIPE_DIRS", raising=False)
    monkeypatch.setattr(kimi_k2, "load_stream_model_config", lambda *_a, **_k: model.config)
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *_a, **_k: SimpleNamespace(pad_token="<pad>")
    )
    monkeypatch.setattr(spectrum_scan, "resolve_model_weights", lambda *_a, **_k: weights)
    monkeypatch.setattr(layer_stream, "free_ram_bytes", lambda: 10_000_000_000)
    monkeypatch.setattr(layer_stream, "total_ram_bytes", lambda: 100_000_000_000)
    monkeypatch.setattr(
        layer_stream, "classify_disk_kind",
        lambda *_a, **_k: layer_stream.DiskClassification("nvme"),
    )
    monkeypatch.setattr(SFTTrainerWrapper, "_stream_budget_lines", lambda *_a, **_k: ([], None))
    builder = Mock(side_effect=_ReachedRuntimeBuilderError)
    monkeypatch.setattr(layer_stream_runtime, "build_streamed_model", builder)

    def setup(*, staging_bytes: int | None = None, **training: Any) -> None:
        if staging_bytes is not None:
            # Synthetic header sizes, with no matching allocations. A 6 GB decoder store
            # fits 7 GB of headroom by itself. Disk staging uses two 2 GB decoder layers.
            monkeypatch.setattr(
                layer_stream_runtime.RamSource, "layer_specs_from_paths",
                lambda _paths: [{"weight": ((500_000_000,), "float32")}] * 3,
            )
            monkeypatch.setattr(layer_stream_runtime, "extras_resident_bytes", lambda _dir: 0)
            monkeypatch.setattr(layer_stream_runtime, "large_layer_store_bytes", lambda *_a: 0)
            monkeypatch.setattr(
                stream_setup, "_expert_buffer_requirements", lambda *_a, **_k: (0, staging_bytes)
            )
        cfg = load_config_from_string(
            _yaml(base=weights, stream_expert_layout="per_expert", **training)
        )
        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper._trust_remote_code = False
        wrapper.device = "cpu"
        wrapper._setup_streaming_transformers(cfg, cfg.training)

    return setup, builder


@pytest.mark.parametrize(
    ("training", "expected"),
    [
        ({"l2l": True}, (True, False, False)),
        ({"l2l": True, "l2l_two_read_experts": True}, (True, True, False)),
        ({"l2l": True, "l2l_grouped_expert_routes": True}, (True, False, True)),
        ({}, (None, None, None)),
    ],
    ids=["defaults", "two-reads", "grouped-routes", "ordinary-per-expert"],
)
def test_streaming_setup_passes_expert_pool_and_lever_values(expert_setup, training, expected):
    setup, builder = expert_setup
    with pytest.raises(_ReachedRuntimeBuilderError):
        setup(**training)
    builder.assert_called_once()
    kwargs = builder.call_args.kwargs
    assert kwargs["index"].expert_layout == "per_expert"
    for name, value in zip(("expert_pool", "two_read_experts", "grouped_expert_routes"), expected):
        if value is None:
            assert name not in kwargs
        else:
            assert kwargs[name] is value


@pytest.mark.parametrize("limit", ["available", "physical"])
def test_forced_ram_refuses_when_the_expert_staging_exceeds_headroom(
    expert_setup, monkeypatch, limit
):
    from soup_cli.utils import layer_stream

    setup, builder = expert_setup
    if limit == "physical":
        monkeypatch.setattr(layer_stream, "free_ram_bytes", lambda: 20_000_000_000)
        monkeypatch.setattr(layer_stream, "total_ram_bytes", lambda: 12_000_000_000)
    with pytest.raises(ValueError) as caught:
        setup(l2l=True, stream_source="ram", staging_bytes=2_000_000_000)
    reason = {
        "available": "needs 8.0 GB and only 10.0 GB of RAM is free.",
        "physical": "needs 8.0 GB, which exceeds 55% of physical RAM (12.0 GB).",
    }[limit]
    assert str(caught.value) == (
        "training.stream_source='ram' but the base plus resident extras and expert pool "
        f"host staging {reason} Set stream_source='auto' to allow the disk tier, "
        "free RAM, or pick a smaller base."
    )
    builder.assert_not_called()


@pytest.mark.parametrize(
    ("staging_bytes", "tier"), [(500_000_000, "ram"), (2_000_000_000, "disk")]
)
def test_auto_tier_charges_expert_staging_on_top_of_the_decoder_store(
    expert_setup, staging_bytes, tier
):
    setup, builder = expert_setup
    with pytest.raises(_ReachedRuntimeBuilderError):
        setup(l2l=True, stream_source="auto", staging_bytes=staging_bytes)
    builder.assert_called_once()
    assert builder.call_args.kwargs["tier"] == tier


def test_disk_tier_still_charges_expert_staging_on_top_of_decoder_staging(expert_setup):
    setup, builder = expert_setup
    with pytest.raises(ValueError) as caught:
        setup(l2l=True, stream_source="disk", staging_bytes=4_000_000_000)
    assert str(caught.value) == (
        "layer streaming's disk tier would hold 8.00 GB of host staging (page-locked when the "
        "box allows) at training.stream_read_ahead=2, and with 0.00 GB of resident extras "
        "that needs 8.00 GB — more than the 7.00 GB safety headroom on 10.0 GB of free RAM. "
        "The reader stages whole layers, and the embedding and lm_head take one slot each at "
        "ANY depth because they are one layer each. Lower training.stream_read_ahead "
        "(currently 2), free RAM, or use a smaller base."
    )
    builder.assert_not_called()


# --------------------------------------------------------------------------
# the L2L step on a tiny DeepSeek-V3: equal to the ordinary accumulation step
# --------------------------------------------------------------------------
SEQ = 8
VOCAB = 128
HIDDEN = 64
LEVERS = {
    "pool": dict(expert_pool=True),
    "pool-two-reads": dict(expert_pool=True, two_read_experts=True),
    "pool-all-levers": dict(
        expert_pool=True, two_read_experts=True, grouped_expert_routes=True
    ),
}


def _micro_batches(count: int) -> List[Any]:
    import torch

    return [
        ((torch.arange(SEQ, dtype=torch.int64) * (3 + mb) + 5 * mb) % VOCAB).reshape(1, SEQ)
        for mb in range(count)
    ]


def _build(tmp_path: Any, *, device: str = "cpu", **levers: Any) -> Any:
    pytest.importorskip("peft")
    import torch

    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_expert_stream import _lora, _shard, _tiny_deepseek_v3

    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=str(tmp_path / "shards"), index=index,
        lora_config=_lora(), device=device, dtype="float32", buffers=2, pin=False, seed=3,
        **levers,
    )
    with torch.no_grad():  # integer-arithmetic adapter values: every lora_B is non-zero
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                raw = (torch.arange(parameter.numel(), dtype=torch.int64) * 7 + 3) % 11 - 5
                parameter.copy_(raw.to(torch.float32).reshape(parameter.shape) / 256)
    return model, runtime


def _store(count: int) -> Any:
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore

    return ActivationStore(
        n_layers=3, k=count, chunk_shape=(1, SEQ, HIDDEN), dtype=torch.float32, device="cpu",
        pin=False,
    )


def _grads(model: Any) -> Dict[str, Any]:
    return {
        name.replace(".inner.", "."): parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.grad is not None
    }


def _accumulate(model: Any, batches: List[Any], bf16: bool) -> Any:
    """HF's accumulation: forward under autocast, backward after the block has exited."""
    import torch

    total = sum(int(ids.shape[0]) * (int(ids.shape[1]) - 1) for ids in batches)
    model.zero_grad(set_to_none=True)
    losses = []
    for ids in batches:
        context = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
        with context:
            out = model(input_ids=ids, labels=ids, num_items_in_batch=total)
        out.loss.backward()
        losses.append(out.loss.detach())
    return losses, _grads(model)


def _l2l(model: Any, runtime: Any, batches: List[Any], bf16: bool) -> Any:
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    step = L2LStep(model, runtime, capture_layer_call(model, batches[0]))
    model.zero_grad(set_to_none=True)
    context = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
    with context:
        losses = step.run(batches, _store(len(batches)))
    return losses, _grads(model), step


def _assert_same(got: Any, want: Any) -> None:
    import torch

    got_losses, got_grads = got
    want_losses, want_grads = want
    assert len(got_losses) == len(want_losses)
    for left, right in zip(got_losses, want_losses):
        assert torch.equal(left.cpu(), right.cpu())
    assert want_grads and set(got_grads) == set(want_grads)
    assert any(float(grad.abs().sum()) > 0.0 for grad in want_grads.values())
    for name, grad in want_grads.items():
        assert torch.equal(got_grads[name], grad), name


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16-autocast"])
@pytest.mark.parametrize("lever", sorted(LEVERS))
def test_the_l2l_step_equals_accumulation_on_a_streamed_moe(tmp_path, lever, bf16):
    model, runtime = _build(tmp_path, **LEVERS[lever])
    try:
        batches = _micro_batches(3)
        want = _accumulate(model, batches, bf16)
        losses, grads, _step = _l2l(model, runtime, batches, bf16)
        _assert_same((losses, grads), want)
    finally:
        runtime.close()


def test_the_moe_step_reads_each_hit_expert_in_forward_recompute_and_backward(tmp_path):
    model, runtime = _build(tmp_path, expert_pool=True)
    try:
        batches = _micro_batches(2)
        _losses, _grads_, step = _l2l(model, runtime, batches, False)
        reads = step.counters.expert_fetches
        assert reads > 0
        assert step.counters.expert_bytes_read > 0
    finally:
        runtime.close()
    (tmp_path / "two").mkdir()
    two_model, two_runtime = _build(tmp_path / "two", expert_pool=True, two_read_experts=True)
    try:
        _l, _g, two_step = _l2l(two_model, two_runtime, batches, False)
        assert 0 < two_step.counters.expert_fetches < reads
    finally:
        two_runtime.close()


# --------------------------------------------------------------------------
# the routing guard
# --------------------------------------------------------------------------
def test_the_routing_guard_stops_a_recompute_routed_differently_from_its_forward(tmp_path):
    import torch

    from soup_cli.utils.l2l_step import RoutingGuard
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    model, runtime = _build(tmp_path, expert_pool=True)
    try:
        layers = list(decoder_owner(model).layers)
        moe_index = 1
        guard = RoutingGuard([layers[moe_index]])
        ids = _micro_batches(1)[0]
        try:
            with torch.no_grad(), guard.record(moe_index, 0):
                model(input_ids=ids)
            gate = getattr(layers[moe_index], "inner", layers[moe_index]).mlp.gate
            with torch.no_grad():
                gate.e_score_correction_bias.copy_(-gate.e_score_correction_bias)
            with pytest.raises(RuntimeError, match=r"layer 1 micro-batch 0 was routed differently"):
                with torch.no_grad(), guard.check(moe_index, 0):
                    model(input_ids=ids)
        finally:
            guard.close()
    finally:
        runtime.close()


def test_the_routing_guard_passes_an_unchanged_recompute_and_frees_the_record(tmp_path):
    import torch

    from soup_cli.utils.l2l_step import RoutingGuard
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    model, runtime = _build(tmp_path, expert_pool=True)
    try:
        layers = list(decoder_owner(model).layers)
        guard = RoutingGuard([layers[1]])
        ids = _micro_batches(1)[0]
        try:
            with torch.no_grad(), guard.record(1, 0):
                model(input_ids=ids)
            with torch.no_grad(), guard.check(1, 0):
                model(input_ids=ids)
            assert guard._records == {}
        finally:
            guard.close()
    finally:
        runtime.close()
