"""``profiler.profile_from_config``: the entry point outside programs call for the estimate.

It is what ``soup profile --config``, the MCP ``profile`` tool and the train pre-flight's
input share, and the one function a program that has no ``soup train`` to run can use for
"will this recipe fit". These tests pin what such a program reads: the result keys, the two
optional arguments that describe the run when the config cannot (``seq_len``, ``shape``),
and the ``FitUnavailable`` answer. ``estimate_total``, removed in 0.77.2 (#2369), must stay
removed: it priced a run from a parameter count alone, with no vocabulary and so no logits.
"""

from __future__ import annotations

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import model_shape, profiler
from soup_cli.utils.hardware_fit import FitUnavailable
from soup_cli.utils.profiler import profile_from_config
from tests._tiny_checkpoint import TINY_LLAMA, write_tiny_model

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


def test_seq_len_none_is_the_result_without_the_argument(cfg):
    plain = profile_from_config(cfg, allow_network=False)
    explicit = profile_from_config(cfg, allow_network=False, seq_len=None)
    assert explicit == plain
    assert not SEQ_LEN_KEYS & set(plain), "a call without seq_len must not grow new keys"


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
    assert f"clamped to data.max_length ({MAX_LENGTH})" in " ".join(over["notes"])


@pytest.mark.parametrize("value", [1, 0, -5, 63])
def test_seq_len_below_the_floor_is_priced_at_64(cfg, value):
    result = profile_from_config(cfg, allow_network=False, seq_len=value)
    assert result["seq_len"] == 64
    assert result["seq_len_requested"] == value
    assert "clamped to 64" in " ".join(result["notes"])


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


@pytest.mark.parametrize("field", ["params", *sorted(_TINY_SHAPE), "head_dim"])
@pytest.mark.parametrize("value", [True, False])
def test_a_bool_shape_field_is_refused_by_name(tiny, unreadable, field, value):
    shape = _tiny_shape(tiny[1])
    shape[field] = value
    with pytest.raises(TypeError, match=rf"shape\[{field!r}\] must be an int, not bool"):
        profile_from_config(unreadable, allow_network=False, shape=shape)


@pytest.mark.parametrize("field", ["params", *sorted(_TINY_SHAPE)])
@pytest.mark.parametrize("value", ["64", 64.0, None])
def test_a_non_int_shape_field_is_refused_by_name(tiny, unreadable, field, value):
    shape = _tiny_shape(tiny[1])
    shape[field] = value
    with pytest.raises(TypeError, match=rf"shape\[{field!r}\] must be an int, got"):
        profile_from_config(unreadable, allow_network=False, shape=shape)


@pytest.mark.parametrize("field", ["params", *sorted(_TINY_SHAPE)])
@pytest.mark.parametrize("value", [0, -1])
def test_a_zero_or_negative_shape_field_is_refused_by_name(tiny, unreadable, field, value):
    shape = _tiny_shape(tiny[1])
    shape[field] = value
    with pytest.raises(ValueError, match=rf"shape\[{field!r}\] must be positive"):
        profile_from_config(unreadable, allow_network=False, shape=shape)


def test_a_shape_beyond_what_a_model_has_is_refused(tiny, unreadable):
    for field, value in (("params", 2_000 * 10**9), ("vocab_size", 2**25), ("num_layers", 2**17)):
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
    wrong = {**_tiny_shape(params), "vocab_size": 500_000}
    result = profile_from_config(readable_cfg, allow_network=False, shape=wrong)
    assert result["shape_source"] == "config.json"
    assert result["vocab_size"] == TINY_LLAMA["vocab_size"]
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
