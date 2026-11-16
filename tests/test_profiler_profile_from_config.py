"""``profiler.profile_from_config``: the entry point outside programs call for the estimate.

It is what ``soup profile --config`` and the MCP ``profile`` tool call, and it builds the
train pre-flight's input (one exception: the pre-flight passes ``kbit_fp32`` on a card without
bf16 hardware, which this function cannot take), and the one function a program that has no
``soup train`` to run can use for "will this recipe fit". These tests pin what such a program
reads: the result keys, the two optional arguments that describe the run when the config
cannot (``seq_len``, ``shape``), and the ``FitUnavailable`` answer. ``estimate_total``, removed
in 0.77.2 (#2369), must stay removed: it priced a run from a parameter count alone, with no
vocabulary and so no logits.

``seq_len`` is honoured only for a run that pads each micro-batch to its longest row (text
SFT without packing). A run that packs, generates, pairs rows or carries image or audio tokens
fills ``data.max_length``, so a shorter length from the caller would under-price it: it is
priced at ``data.max_length`` and the result says why.
"""

from __future__ import annotations

import typing

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
from soup_cli.utils import model_shape, profiler, vram_estimate
from soup_cli.utils.hardware_fit import (
    FitUnavailable,
    estimate_peak_vram_gb,
    hardware_fit_input_or_reason,
)
from soup_cli.utils.profiler import profile_from_config, profile_result
from tests._tiny_checkpoint import TINY_LLAMA, tensors_for, write_safetensors, write_tiny_model

#: The keys of the dict, as ``soup profile --json`` prints them. A program that reads one of
#: them breaks if it is renamed or dropped; a new key is added here on purpose.
RESULT_KEYS = frozenset(
    {
        "activation_memory_gb",
        "batch_size",
        "dequant_memory_gb",
        "gradient_checkpointing",
        "gradient_checkpointing_tier",
        "gradients_memory_gb",
        "hidden_size",
        "logits_memory_gb",
        "lora_memory_gb",
        "lora_r",
        "lora_targets",
        "memory_terms_bytes",
        "model_memory_gb",
        "model_name",
        "model_params_b",
        "notes",
        "num_layers",
        "optimizer_memory_gb",
        "optimizer_step_memory_gb",
        "overhead_gb",
        "params_source",
        "peft",
        "quantization",
        "seq_len",
        "shape_source",
        "total_memory_bytes",
        "total_memory_gb",
        "trainable_params",
        "vocab_size",
    }
)
#: Added only when the caller passed ``seq_len``.
SEQ_LEN_KEYS = frozenset({"seq_len_requested", "seq_len_source"})

MAX_LENGTH = 1024
UNREADABLE = "someone/Unreadable-3B"


def _text(base: str, *, batch_size=2, max_length=MAX_LENGTH, quantization="none") -> str:
    return yaml.safe_dump(
        {
            "base": base,
            "task": "sft",
            "data": {"train": "train.jsonl", "format": "alpaca", "max_length": max_length},
            "training": {
                "epochs": 1,
                "batch_size": batch_size,
                "quantization": quantization,
                "lora": {"r": 16, "alpha": 32, "target_modules": "all-linear"},
            },
        }
    )


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A readable local model, and its parameter count."""
    monkeypatch.chdir(tmp_path)
    params = write_tiny_model(tmp_path / "tiny")
    return tmp_path / "tiny", params


@pytest.fixture
def cfg(tiny):
    return load_config_from_string(_text(str(tiny[0])))


@pytest.fixture
def unreadable(monkeypatch):
    """A model id with no config.json anywhere: neither the cache nor the Hub has it."""
    monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kwargs: (None, None))
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kwargs: None)
    return load_config_from_string(_text(UNREADABLE))


# --- the result, as outside programs read it --------------------------------------------


def test_the_result_keys_are_pinned(cfg):
    result = profile_from_config(cfg, allow_network=False)
    assert isinstance(result, dict)
    assert set(result) == RESULT_KEYS, (
        sorted(set(result) - RESULT_KEYS),
        sorted(RESULT_KEYS - set(result)),
    )


def test_the_result_prices_the_length_the_config_names(cfg):
    result = profile_from_config(cfg, allow_network=False)
    assert result["seq_len"] == MAX_LENGTH
    assert result["batch_size"] == 2
    assert result["total_memory_bytes"] == sum(result["memory_terms_bytes"].values())


def test_estimate_total_is_gone_and_stays_gone():
    # It priced a run from a parameter count alone: no vocabulary, so no logits term.
    assert not hasattr(profiler, "estimate_total")


# --- seq_len -----------------------------------------------------------------------------


def _profile_from_config_at_0_77_2(cfg, **kwargs):
    """The body of ``profile_from_config`` before ``seq_len`` and ``shape`` existed, frozen."""
    inp = hardware_fit_input_or_reason(cfg, profile_defaults=True, **kwargs)
    if isinstance(inp, FitUnavailable):
        return inp
    result = profile_result(
        estimate_peak_vram_gb(inp),
        model_name=cfg.base,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
    )
    result["quantization"] = cfg.training.quantization
    return result


def test_without_the_new_arguments_the_result_is_what_0_77_2_returned(cfg, unreadable):
    for config in (cfg, unreadable):
        frozen = _profile_from_config_at_0_77_2(config, allow_network=False)
        assert profile_from_config(config, allow_network=False) == frozen
        assert profile_from_config(config, allow_network=False, seq_len=None, shape=None) == frozen
        assert not SEQ_LEN_KEYS & set(frozen)


def test_a_4_bit_run_is_the_same_as_0_77_2_too(tiny):
    config = load_config_from_string(_text(str(tiny[0]), quantization="4bit"))
    frozen = _profile_from_config_at_0_77_2(config, allow_network=False)
    assert profile_from_config(config, allow_network=False) == frozen


def _with(config, **changes):
    """``config`` with fields replaced, nested ones through ``training=`` and ``data=``.

    ``model_copy`` does not re-run the schema's validators, so a combination the schema refuses
    (a task with the wrong data format, packing with a task that never packs) still reaches the
    function, which reads only these attributes.
    """
    training, data = changes.pop("training", None), changes.pop("data", None)
    if training:
        changes["training"] = config.training.model_copy(update=training)
    if data:
        changes["data"] = config.data.model_copy(update=data)
    return config.model_copy(update=changes)


def _literal(model, field):
    return set(typing.get_args(model.model_fields[field].annotation))


def test_the_fields_the_length_rule_reads_exist_in_the_schema():
    # A renamed field would make the rule read a default and honour seq_len silently.
    assert {"packing", "multipack", "stream_layers"} <= set(TrainingConfig.model_fields)
    assert "streaming" in DataConfig.model_fields
    assert {"task", "modality", "backend"} <= set(SoupConfig.model_fields)
    assert {"sft"} <= _literal(SoupConfig, "task")
    assert {"text"} <= _literal(SoupConfig, "modality")
    assert {"transformers"} <= _literal(SoupConfig, "backend")


def test_plain_text_sft_honours_seq_len(cfg):
    result = profile_from_config(cfg, allow_network=False, seq_len=79)
    assert result["seq_len"] == 79
    assert result["seq_len_source"] == "caller"
    assert result["seq_len_requested"] == 79


#: The runs whose micro-batches are not padded to the longest row, one reason each. Every task
#: other than sft is added below from the schema, so a new task is priced at max_length until
#: someone measures it.
_NOT_PADDED_TO_THE_LONGEST_ROW = {
    "packing": (dict(training={"packing": True}), "training.packing is on"),
    "multipack": (dict(training={"multipack": True}), "training.multipack is on"),
    "streamed data": (dict(data={"streaming": True}), "data.streaming is on"),
    "layer streaming": (dict(training={"stream_layers": True}), "training.stream_layers is on"),
    **{
        f"modality {name}": (dict(modality=name), f"modality {name!r}")
        for name in sorted(_literal(SoupConfig, "modality") - {"text"})
    },
    **{
        f"backend {name}": (dict(backend=name), f"backend {name!r}")
        for name in sorted(_literal(SoupConfig, "backend") - {"transformers"})
    },
    **{
        f"task {name}": (dict(task=name), f"task {name!r}")
        for name in sorted(_literal(SoupConfig, "task") - {"sft"})
    },
}


@pytest.mark.parametrize("case", sorted(_NOT_PADDED_TO_THE_LONGEST_ROW))
def test_a_run_that_does_not_pad_to_its_longest_row_is_priced_at_max_length(cfg, case):
    changes, reason = _NOT_PADDED_TO_THE_LONGEST_ROW[case]
    config = _with(cfg, **changes)
    plain = profile_from_config(config, allow_network=False)
    asked = profile_from_config(config, allow_network=False, seq_len=79)
    assert asked["seq_len"] == MAX_LENGTH, case
    assert asked["total_memory_bytes"] == plain["total_memory_bytes"], case
    assert asked["seq_len_source"] == "max_length"
    assert asked["seq_len_requested"] == 79
    note = " ".join(asked["notes"])
    assert reason in note, (case, note)
    assert "seq_len 79 was supplied by the caller" in note
    assert f"priced at data.max_length ({MAX_LENGTH})" in note


def test_every_task_but_sft_is_in_the_exclusion_table():
    tasks = {case for case in _NOT_PADDED_TO_THE_LONGEST_ROW if case.startswith("task ")}
    assert len(tasks) == len(_literal(SoupConfig, "task")) - 1 >= 20
    assert {"task grpo", "task ppo", "task online_dpo", "task dpo", "task pretrain"} <= tasks


def test_a_packed_run_is_not_told_it_fits_at_a_short_length(cfg):
    # The failure this guards: packing fills every micro-batch to max_length, so 79 tokens
    # would price a Qwen3-0.6B run at 2.9 GB where it needs 10.2 (test_vram_estimate_realised_
    # length.py has the real model); here the tiny model shows the direction.
    packed = _with(cfg, training={"packing": True})
    asked = profile_from_config(packed, allow_network=False, seq_len=79)
    honoured = profile_from_config(cfg, allow_network=False, seq_len=79)
    assert asked["total_memory_bytes"] > honoured["total_memory_bytes"]
    assert asked["total_memory_bytes"] == profile_from_config(packed, allow_network=False)[
        "total_memory_bytes"
    ]


def test_seq_len_prices_that_many_tokens_per_row(cfg):
    full = profile_from_config(cfg, allow_network=False)
    short = profile_from_config(cfg, allow_network=False, seq_len=128)
    assert short["seq_len"] == 128
    # tokens: the two token terms scale with them, the weights do not.
    assert short["memory_terms_bytes"]["weights"] == full["memory_terms_bytes"]["weights"]
    assert short["memory_terms_bytes"]["logits"] * (MAX_LENGTH // 128) == pytest.approx(
        full["memory_terms_bytes"]["logits"], rel=1e-3
    )
    assert short["total_memory_bytes"] < full["total_memory_bytes"]


def test_seq_len_says_where_the_length_came_from(cfg):
    short = profile_from_config(cfg, allow_network=False, seq_len=128)
    assert short["seq_len_source"] == "caller"
    assert short["seq_len_requested"] == 128
    note = " ".join(short["notes"])
    assert "seq_len 128 was supplied by the caller" in note
    assert f"data.max_length is {MAX_LENGTH}" in note
    assert "rows that fill 128 tokens" in note


def test_seq_len_above_max_length_is_priced_at_max_length(cfg):
    full = profile_from_config(cfg, allow_network=False)
    over = profile_from_config(cfg, allow_network=False, seq_len=10 * MAX_LENGTH)
    assert over["seq_len"] == MAX_LENGTH
    assert over["total_memory_bytes"] == full["total_memory_bytes"]
    assert over["seq_len_requested"] == 10 * MAX_LENGTH
    # The length priced came from the config, not the caller.
    assert over["seq_len_source"] == "max_length"
    assert f"clamped to data.max_length ({MAX_LENGTH})" in " ".join(over["notes"])


@pytest.mark.parametrize("value", [1, 63])
def test_seq_len_below_the_floor_is_priced_at_64(cfg, value):
    result = profile_from_config(cfg, allow_network=False, seq_len=value)
    assert result["seq_len"] == 64
    assert result["seq_len_requested"] == value
    assert result["seq_len_source"] == "caller"
    assert "clamped to 64" in " ".join(result["notes"])


@pytest.mark.parametrize("value", [0, -1, -5])
def test_seq_len_below_one_is_refused_by_name(cfg, value):
    # An empty dataset has no longest row; it must not get a silent 64-token estimate.
    with pytest.raises(ValueError, match=r"seq_len must be at least 1, got " + str(value)):
        profile_from_config(cfg, allow_network=False, seq_len=value)


def test_seq_len_at_the_floor_and_at_max_length_are_not_clamped(cfg):
    for value in (64, MAX_LENGTH):
        result = profile_from_config(cfg, allow_network=False, seq_len=value)
        assert result["seq_len"] == value
        assert "clamped" not in " ".join(result["notes"])


@pytest.mark.parametrize(
    "value, shown",
    [(True, "bool"), (False, "bool"), ("128", "str"), (128.0, "float"), (b"128", "bytes")],
)
def test_seq_len_that_is_not_an_int_is_refused_by_name(cfg, value, shown):
    with pytest.raises(TypeError, match=rf"seq_len must be an int or None, got {shown}"):
        profile_from_config(cfg, allow_network=False, seq_len=value)


def test_a_bad_seq_len_is_refused_before_the_model_is_read(monkeypatch, cfg):
    def explode(*args, **kwargs):
        raise AssertionError("the model was read before seq_len was checked")

    monkeypatch.setattr(model_shape, "read_model_shape", explode)
    with pytest.raises(TypeError, match="seq_len"):
        profile_from_config(cfg, allow_network=False, seq_len=True)


def test_the_priced_length_does_not_change_the_card_independent_terms(cfg):
    low = profile_from_config(cfg, allow_network=False, seq_len=64)
    high = profile_from_config(cfg, allow_network=False, seq_len=512)
    for term in ("weights", "adapters", "gradients", "optimizer"):
        assert low["memory_terms_bytes"][term] == high["memory_terms_bytes"][term]


# --- FitUnavailable ----------------------------------------------------------------------


def test_a_run_the_estimate_cannot_price_returns_its_reason(tiny):
    over = load_config_from_string(_text(str(tiny[0]), batch_size=2000))
    result = profile_from_config(over, allow_network=False)
    assert isinstance(result, FitUnavailable)
    assert result.field == "training.batch_size"
    assert "outside 1 to 1,024" in result.reason


def test_seq_len_does_not_hide_a_run_the_estimate_cannot_price(tiny):
    over = load_config_from_string(_text(str(tiny[0]), batch_size=2000))
    result = profile_from_config(over, allow_network=False, seq_len=128)
    assert isinstance(result, FitUnavailable)
    assert result.field == "training.batch_size"


# --- a model whose shape cannot be read ----------------------------------------------------

_TINY_SHAPE = {
    "hidden_size": TINY_LLAMA["hidden_size"],
    "num_layers": TINY_LLAMA["num_hidden_layers"],
    "intermediate_size": TINY_LLAMA["intermediate_size"],
    "vocab_size": TINY_LLAMA["vocab_size"],
}


_OPTIONAL = ("num_attention_heads", "num_key_value_heads", "head_dim")


def _tiny_shape(params: int, **extra) -> dict:
    return {
        **_TINY_SHAPE,
        "params": params,
        "num_attention_heads": TINY_LLAMA["num_attention_heads"],
        "num_key_value_heads": TINY_LLAMA["num_key_value_heads"],
        "head_dim": TINY_LLAMA["hidden_size"] // TINY_LLAMA["num_attention_heads"],
        **extra,
    }


def test_an_unreadable_model_is_guessed_and_prices_no_logits_today(unreadable):
    # The blind spot the shape argument exists to close: a shape guessed from the name has
    # no vocabulary, so nothing is charged for the logits, 74% of a small model's estimate.
    result = profile_from_config(unreadable, allow_network=False)
    assert isinstance(result, dict), "an unreadable model is priced, not FitUnavailable"
    assert result["shape_source"] == "guessed"
    assert result["params_source"] == "name"
    assert result["vocab_size"] is None
    assert result["logits_memory_gb"] == 0
    assert "logits term is not counted" in " ".join(result["notes"])


def test_a_supplied_shape_prices_what_a_readable_config_would(tiny, unreadable):
    directory, params = tiny
    readable = profile_from_config(
        load_config_from_string(_text(str(directory))), allow_network=False
    )
    supplied = profile_from_config(unreadable, allow_network=False, shape=_tiny_shape(params))
    assert supplied["shape_source"] == "caller"
    assert supplied["params_source"] == "caller"
    assert supplied["vocab_size"] == TINY_LLAMA["vocab_size"]
    assert supplied["logits_memory_gb"] > 0
    assert supplied["memory_terms_bytes"] == readable["memory_terms_bytes"]
    assert supplied["total_memory_bytes"] == readable["total_memory_bytes"]
    assert supplied["model_params_b"] == readable["model_params_b"]


def test_a_supplied_shape_says_so(tiny, unreadable):
    result = profile_from_config(unreadable, allow_network=False, shape=_tiny_shape(tiny[1]))
    note = " ".join(result["notes"])
    assert "shape was supplied by the caller" in note
    assert "No config.json could be read" not in note


def test_a_shape_without_the_optional_dims_is_still_priced(tiny, unreadable):
    # The heads default to hidden // 128 as the name-guessed shape does; the logits term,
    # the point of the argument, does not depend on them.
    shape = {**_TINY_SHAPE, "params": tiny[1]}
    result = profile_from_config(unreadable, allow_network=False, shape=shape)
    assert result["shape_source"] == "caller"
    assert result["logits_memory_gb"] > 0


def test_a_parameter_count_alone_is_not_a_shape(unreadable):
    # It is the pre-#2049 blind spot: no vocabulary, no logits.
    with pytest.raises(ValueError) as caught:
        profile_from_config(unreadable, allow_network=False, shape={"params": 3_000_000_000})
    message = str(caught.value)
    for name in ("hidden_size", "num_layers", "intermediate_size", "vocab_size"):
        assert name in message
    assert "logits" in message


@pytest.mark.parametrize("missing", sorted(_TINY_SHAPE))
def test_every_required_field_is_named_when_missing(tiny, unreadable, missing):
    shape = _tiny_shape(tiny[1])
    del shape[missing]
    with pytest.raises(ValueError, match=rf"shape is missing {missing}"):
        profile_from_config(unreadable, allow_network=False, shape=shape)


def test_a_shape_without_params_is_refused(unreadable):
    with pytest.raises(ValueError, match="shape is missing params"):
        profile_from_config(unreadable, allow_network=False, shape=dict(_TINY_SHAPE))


@pytest.mark.parametrize("field", ["params", *sorted(_TINY_SHAPE), *_OPTIONAL])
@pytest.mark.parametrize("value", [True, False])
def test_a_bool_shape_field_is_refused_by_name(tiny, unreadable, field, value):
    shape = _tiny_shape(tiny[1])
    shape[field] = value
    with pytest.raises(TypeError, match=rf"shape\[{field!r}\] must be an int, not bool"):
        profile_from_config(unreadable, allow_network=False, shape=shape)


@pytest.mark.parametrize("field", ["params", *sorted(_TINY_SHAPE), *_OPTIONAL])
@pytest.mark.parametrize("value", ["64", 64.0, None])
def test_a_non_int_shape_field_is_refused_by_name(tiny, unreadable, field, value):
    shape = _tiny_shape(tiny[1])
    shape[field] = value
    with pytest.raises(TypeError, match=rf"shape\[{field!r}\] must be an int, got"):
        profile_from_config(unreadable, allow_network=False, shape=shape)


@pytest.mark.parametrize("field", ["params", *sorted(_TINY_SHAPE), *_OPTIONAL])
@pytest.mark.parametrize("value", [0, -1])
def test_a_zero_or_negative_shape_field_is_refused_by_name(tiny, unreadable, field, value):
    shape = _tiny_shape(tiny[1])
    shape[field] = value
    with pytest.raises(ValueError, match=rf"shape\[{field!r}\] must be positive"):
        profile_from_config(unreadable, allow_network=False, shape=shape)


@pytest.mark.parametrize(
    "field, value",
    [
        ("params", 2_000 * 10**9),
        ("vocab_size", 2**25),
        ("hidden_size", 2**25),
        ("intermediate_size", 2**25),
        ("num_layers", 2**17),
        ("num_attention_heads", 2**25),
        ("num_key_value_heads", 2**25),
        ("head_dim", 2**25),
    ],
)
def test_a_shape_beyond_what_a_model_has_is_refused(tiny, unreadable, field, value):
    shape = _tiny_shape(tiny[1])
    shape[field] = value
    with pytest.raises(ValueError, match=rf"shape\[{field!r}\] is larger than"):
        profile_from_config(unreadable, allow_network=False, shape=shape)


def test_an_unknown_shape_key_is_refused_by_name(tiny, unreadable):
    with pytest.raises(ValueError, match="unknown shape field 'vocab'"):
        profile_from_config(
            unreadable, allow_network=False, shape={**_tiny_shape(tiny[1]), "vocab": 1}
        )


def test_a_shape_that_is_not_a_mapping_is_refused(unreadable):
    with pytest.raises(TypeError, match="shape must be a mapping"):
        profile_from_config(unreadable, allow_network=False, shape=[1, 2, 3])


def test_tie_word_embeddings_must_be_a_bool(tiny, unreadable):
    shape = _tiny_shape(tiny[1], tie_word_embeddings=1)
    with pytest.raises(TypeError, match=r"shape\['tie_word_embeddings'\] must be a bool"):
        profile_from_config(unreadable, allow_network=False, shape=shape)


def test_a_readable_model_wins_over_a_supplied_shape(tiny):
    directory, params = tiny
    readable_cfg = load_config_from_string(_text(str(directory)))
    plain = profile_from_config(readable_cfg, allow_network=False)
    wrong = {**_tiny_shape(params), "num_layers": 40, "intermediate_size": 9_999}
    result = profile_from_config(readable_cfg, allow_network=False, shape=wrong)
    assert result["shape_source"] == "config.json"
    assert result["num_layers"] == TINY_LLAMA["num_hidden_layers"]
    assert result["total_memory_bytes"] == plain["total_memory_bytes"]
    assert "shape was not used" in " ".join(result["notes"])


def test_without_shape_the_result_has_no_shape_note(unreadable):
    notes = " ".join(profile_from_config(unreadable, allow_network=False)["notes"])
    assert "supplied by the caller" not in notes


def test_seq_len_and_shape_compose(tiny, unreadable):
    result = profile_from_config(
        unreadable, allow_network=False, seq_len=128, shape=_tiny_shape(tiny[1])
    )
    assert result["seq_len"] == 128
    assert result["shape_source"] == "caller"
    assert result["seq_len_source"] == "caller"


# --- the parameter count --------------------------------------------------------------------


def test_params_below_the_embedding_table_alone_is_refused_naming_both_fields(unreadable):
    floor = _TINY_SHAPE["vocab_size"] * _TINY_SHAPE["hidden_size"]
    with pytest.raises(ValueError) as caught:
        profile_from_config(
            unreadable, allow_network=False, shape={**_TINY_SHAPE, "params": floor - 1}
        )
    message = str(caught.value)
    assert "shape['params']" in message
    assert f"{floor - 1:,}" in message
    assert "vocab_size x hidden_size" in message
    assert f"{floor:,}" in message


def test_params_equal_to_the_embedding_table_is_accepted(unreadable):
    floor = _TINY_SHAPE["vocab_size"] * _TINY_SHAPE["hidden_size"]
    result = profile_from_config(
        unreadable, allow_network=False, shape={**_TINY_SHAPE, "params": floor}
    )
    assert result["shape_source"] == "caller"


_QWEN3_0_6B = {
    "hidden_size": 1024,
    "num_layers": 28,
    "intermediate_size": 3072,
    "vocab_size": 151936,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
}
#: The safetensors header sums both ``embed_tokens`` and the tied ``lm_head`` that this
#: checkpoint stores. transformers loads 596,049,920, which the estimate now uses for a model
#: it reads itself.
_QWEN3_STORED, _QWEN3_LOADED = 751_632_384, 596_049_920


def test_the_stored_and_the_loaded_count_of_a_tied_model_price_apart_by_the_table(unreadable):
    # A supplied count is taken as given: the estimate corrects a stored-twice tied head only
    # in a count it reads itself (test_vram_estimate_measured_overhead.py), so the docs say to
    # pass the loaded count. The stored one is higher by the table, 2 bytes an element of
    # weights; a bf16 LoRA run's overhead has no per-allocated-byte slack to add to that.
    stored = profile_from_config(
        unreadable, allow_network=False, shape={**_QWEN3_0_6B, "params": _QWEN3_STORED}
    )
    loaded = profile_from_config(
        unreadable, allow_network=False, shape={**_QWEN3_0_6B, "params": _QWEN3_LOADED}
    )
    assert _QWEN3_STORED - _QWEN3_LOADED == 151936 * 1024
    gap = stored["total_memory_bytes"] - loaded["total_memory_bytes"]
    assert gap == 2 * 151936 * 1024


def test_tie_word_embeddings_changes_the_k_bit_weights(tiny, monkeypatch):
    # The flag reaches the estimate: a tied head is one table under 4-bit loading, an untied
    # one is two. Marking an untied model tied under-prices it, so the docs say to omit it.
    monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kwargs: (None, None))
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kwargs: None)
    config = load_config_from_string(_text(UNREADABLE, quantization="4bit"))
    base = _tiny_shape(tiny[1])
    flags = {"tie_word_embeddings": False}, {"tie_word_embeddings": True}
    untied, tied = (
        profile_from_config(config, allow_network=False, shape={**base, **flag}) for flag in flags
    )
    omitted = profile_from_config(config, allow_network=False, shape=base)
    table = _TINY_SHAPE["vocab_size"] * _TINY_SHAPE["hidden_size"]
    expected = table * (vram_estimate.NF4_BYTES_PER_PARAM - vram_estimate.KBIT_UNQUANTIZED_BYTES)
    assert tied["memory_terms_bytes"]["weights"] < untied["memory_terms_bytes"]["weights"]
    assert (
        tied["memory_terms_bytes"]["weights"] - untied["memory_terms_bytes"]["weights"]
        == pytest.approx(expected, abs=2)
    )
    assert omitted["memory_terms_bytes"] == untied["memory_terms_bytes"]


def test_a_bad_shape_is_refused_before_the_model_is_read(monkeypatch, unreadable):
    def explode(*args, **kwargs):
        raise AssertionError("the model was read before shape was checked")

    monkeypatch.setattr(model_shape, "read_model_shape", explode)
    with pytest.raises(ValueError, match="shape is missing"):
        profile_from_config(unreadable, allow_network=False, shape={"params": 3_000_000_000})
    with pytest.raises(TypeError, match="shape must be a mapping"):
        profile_from_config(unreadable, allow_network=False, shape="Qwen/Qwen3-0.6B")


def test_a_count_read_from_the_weights_wins_over_the_supplied_one(tmp_path, monkeypatch):
    # Weights on disk and no config.json: the shape is still guessed, but the parameter count is
    # measured from the safetensors headers, and that is not replaced by the caller's.
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "weights-only"
    directory.mkdir()
    measured = write_safetensors(directory / "model.safetensors", tensors_for(TINY_LLAMA))
    config = load_config_from_string(_text(str(directory)))
    guessed = profile_from_config(config, allow_network=False)
    assert guessed["shape_source"] == "guessed" and guessed["params_source"] == "safetensors"
    result = profile_from_config(
        config, allow_network=False, shape=_tiny_shape(measured + 12_345)
    )
    assert result["shape_source"] == "caller"
    assert result["params_source"] == "safetensors"
    assert result["memory_terms_bytes"]["weights"] == measured * 2
    note = " ".join(result["notes"])
    assert f"{measured:,}" in note and f"{measured + 12_345:,}" in note
    assert "safetensors" in note
