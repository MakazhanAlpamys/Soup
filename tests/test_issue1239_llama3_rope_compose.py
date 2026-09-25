"""#1239 — extending a checkpoint whose RoPE block is already scaled.

Llama-3.1 checkpoints ship ``max_position_embeddings: 131072`` with a native
``llama3`` block: factor 8 over ``original_max_position_embeddings`` 8192,
``low_freq_factor`` 1, ``high_freq_factor`` 4. ``rope_scaling_type: llama3``
used to REPLACE that block with factor 2 over 131072, so 35 of the 64 frequency
pairs of an 8B-shaped head rotated up to 8x faster than in pretraining.

It now COMPOSES: the checkpoint's original length and band factors are kept
and its factor is multiplied by ``target / max_position_embeddings``. Every
other extension of a checkpoint whose RoPE block is not ``default`` is refused
before the model is built, and the message names the checkpoint's
``rope_type`` and ``factor``. Checkpoints without RoPE scaling extend exactly as
before.

``rope_scaling_type: longrope`` is refused at config load, in every spelling:
its per-dimension ``short_factor`` / ``long_factor`` vectors exist only on
checkpoints already scaled with LongRoPE, and extending those is refused, so it
cannot extend anything; the refusal points at the types that can. Fine-tuning a
LongRoPE checkpoint at its native length with ``rope_scaling_type`` unset keeps
working, and a control trains one for real.

Reference inverse frequencies come from transformers' own llama3 init
(``ROPE_INIT_FUNCTIONS["llama3"]``, i.e. ``_compute_llama3_parameters``), never
from a re-implementation. The tiny models are config-built (hidden 64, one
layer) with ``head_dim`` 128, so they have the 64 frequency pairs of the 8B.
"""

import json
import math
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string

# Llama-3.1-8B model-card values.
NATIVE_MAX = 131072
NATIVE_LLAMA31 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_theta": 500000.0,
}
TARGET = 262144
# The composition #1239 asks for: same original length and bands, factor x2.
COMPOSED_LLAMA31 = {
    "rope_theta": 500000.0,
    "rope_type": "llama3",
    "factor": 16.0,
    "original_max_position_embeddings": 8192,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
}

# Native blocks that are already scaled and have no defined composition.
NATIVE_YARN = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
}
NATIVE_LONGROPE = {  # Phi-3-mini-128k style: learned vectors, no ``factor`` key
    "rope_type": "longrope",
    "short_factor": [1.0] * 4,
    "long_factor": [2.0] * 4,
    "original_max_position_embeddings": 4096,
    "rope_theta": 10000.0,
}
NATIVE_LINEAR = {"rope_type": "linear", "factor": 4.0, "rope_theta": 10000.0}
NATIVE_DYNAMIC = {"rope_type": "dynamic", "factor": 2.0, "rope_theta": 10000.0}

REQUESTED_TYPES = ("llama3", None, "linear", "dynamic", "yarn", "longrope")

# The refusal on a native llama3 block points at the one composition that exists.
_COMPOSE_HINT = "rope_scaling_type='llama3', which composes with the checkpoint's own llama3 block"


def _requires_train_extra() -> None:
    for module in ("torch", "transformers", "peft"):
        pytest.importorskip(module, reason=f"{module} is only in the [train] extra")


def _tiny_llama_config(*, max_position_embeddings: int, rope_parameters: dict):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=max_position_embeddings,
        rope_parameters=deepcopy(rope_parameters),
    )


def _transformers_llama3_inv_freq(*, max_position_embeddings: int, rope_parameters: dict):
    """The inverse frequencies transformers' own llama3 init computes."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    config = _tiny_llama_config(
        max_position_embeddings=max_position_embeddings,
        rope_parameters=rope_parameters,
    )
    inv_freq, _attention_factor = ROPE_INIT_FUNCTIONS["llama3"](config, None)
    return inv_freq


def _native_inv_freq():
    return _transformers_llama3_inv_freq(
        max_position_embeddings=NATIVE_MAX, rope_parameters=NATIVE_LLAMA31
    )


def _bands():
    """Masks of the high / low frequency bands for Llama-3.1's native block."""
    import torch

    base = 1.0 / (
        NATIVE_LLAMA31["rope_theta"] ** (torch.arange(0, 128, 2, dtype=torch.int64).float() / 128)
    )
    wavelen = 2 * math.pi / base
    high = wavelen < 8192 / NATIVE_LLAMA31["high_freq_factor"]
    low = wavelen > 8192 / NATIVE_LLAMA31["low_freq_factor"]
    return high, low


def _snapshot(config) -> dict:
    return deepcopy(vars(config))


def _message_names(exc_info, *parts: str) -> None:
    message = str(exc_info.value)
    for part in parts:
        assert part in message, (part, message)


# ---------------------------------------------------------------------------
# The composition itself
# ---------------------------------------------------------------------------


class TestComposeLlama3RopeParameters:
    def test_keeps_the_native_bands_and_multiplies_the_factor(self) -> None:
        from soup_cli.utils.long_context import compose_llama3_rope_parameters

        block = compose_llama3_rope_parameters(
            NATIVE_LLAMA31, max_position_embeddings=NATIVE_MAX, target_length=TARGET
        )

        assert block == {
            "rope_type": "llama3",
            "factor": 16.0,
            "original_max_position_embeddings": 8192,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        }

    def test_reduces_to_the_native_inv_freq_when_target_equals_max(self) -> None:
        _requires_train_extra()
        import torch

        from soup_cli.utils.long_context import compose_llama3_rope_parameters

        block = compose_llama3_rope_parameters(
            NATIVE_LLAMA31, max_position_embeddings=NATIVE_MAX, target_length=NATIVE_MAX
        )
        composed = _transformers_llama3_inv_freq(
            max_position_embeddings=NATIVE_MAX,
            rope_parameters={**block, "rope_theta": NATIVE_LLAMA31["rope_theta"]},
        )

        assert block["factor"] == NATIVE_LLAMA31["factor"]
        assert torch.equal(composed, _native_inv_freq())

    @pytest.mark.parametrize("target", [131073, 200000, TARGET, 1048576])
    def test_no_pair_rotates_faster_than_the_checkpoint(self, target: int) -> None:
        _requires_train_extra()
        import torch

        from soup_cli.utils.long_context import compose_llama3_rope_parameters

        block = compose_llama3_rope_parameters(
            NATIVE_LLAMA31, max_position_embeddings=NATIVE_MAX, target_length=target
        )
        composed = _transformers_llama3_inv_freq(
            max_position_embeddings=target,
            rope_parameters={**block, "rope_theta": NATIVE_LLAMA31["rope_theta"]},
        )
        native = _native_inv_freq()
        high, low = _bands()

        assert composed.shape == native.shape == (64,)
        faster = [int(k) for k in torch.nonzero(composed > native).flatten()]
        assert faster == [], f"pairs rotating faster than pretrained: {faster}"
        assert torch.equal(composed[high], native[high])
        assert bool((composed[low] < native[low]).all())

    def test_the_issue_target_slows_the_same_35_pairs_the_bug_sped_up(self) -> None:
        _requires_train_extra()
        import torch

        from soup_cli.utils.long_context import compose_llama3_rope_parameters

        block = compose_llama3_rope_parameters(
            NATIVE_LLAMA31, max_position_embeddings=NATIVE_MAX, target_length=TARGET
        )
        composed = _transformers_llama3_inv_freq(
            max_position_embeddings=TARGET,
            rope_parameters={**block, "rope_theta": NATIVE_LLAMA31["rope_theta"]},
        )
        native = _native_inv_freq()
        high, low = _bands()
        medium = ~high & ~low

        assert (int(high.sum()), int(low.sum()), int(medium.sum())) == (29, 29, 6)
        assert int((composed != native).sum()) == 35
        # Factor 8 -> 16: the low band rotates exactly half as fast as pretrained.
        assert torch.equal(composed[low], native[low] / 2)
        assert bool((composed[medium] < native[medium]).all())
        assert bool((composed[medium] > native[medium] / 2).all())

    def test_a_missing_original_length_falls_back_like_transformers(self) -> None:
        """transformers reads a llama3 block without ``original_max_position_embeddings``
        as if it were ``max_position_embeddings``; the composition keeps that value."""
        _requires_train_extra()
        import torch

        from soup_cli.utils.long_context import compose_llama3_rope_parameters

        native = {
            key: value
            for key, value in NATIVE_LLAMA31.items()
            if key != "original_max_position_embeddings"
        }
        block = compose_llama3_rope_parameters(
            native, max_position_embeddings=8192, target_length=8192
        )
        assert block["original_max_position_embeddings"] == 8192
        assert torch.equal(
            _transformers_llama3_inv_freq(
                max_position_embeddings=8192,
                rope_parameters={**block, "rope_theta": native["rope_theta"]},
            ),
            _transformers_llama3_inv_freq(max_position_embeddings=8192, rope_parameters=native),
        )

        extended = compose_llama3_rope_parameters(
            native, max_position_embeddings=8192, target_length=32768
        )
        assert extended["original_max_position_embeddings"] == 8192
        assert extended["factor"] == 32.0

    def test_refuses_to_shrink_the_factor(self) -> None:
        from soup_cli.utils.long_context import compose_llama3_rope_parameters

        with pytest.raises(ValueError, match=r"#1239") as exc_info:
            compose_llama3_rope_parameters(
                NATIVE_LLAMA31, max_position_embeddings=NATIVE_MAX, target_length=65536
            )
        _message_names(exc_info, "65536", "131072")

    def test_refuses_a_block_that_is_not_llama3(self) -> None:
        from soup_cli.utils.long_context import compose_llama3_rope_parameters

        with pytest.raises(ValueError, match=r"rope_type='yarn'"):
            compose_llama3_rope_parameters(
                NATIVE_YARN, max_position_embeddings=NATIVE_MAX, target_length=TARGET
            )

    def test_equal_band_factors_compose_too(self) -> None:
        """Llama 4 Scout's text block ships ``low == high == 1``; transformers only
        warns about that, and the composition must keep it rather than refuse it."""
        _requires_train_extra()
        import torch

        from soup_cli.utils.long_context import compose_llama3_rope_parameters

        scout = {
            "rope_type": "llama3",
            "factor": 16.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_theta": 500000.0,
        }
        scout_max = 10485760
        block = compose_llama3_rope_parameters(
            scout, max_position_embeddings=scout_max, target_length=2 * scout_max
        )
        composed = _transformers_llama3_inv_freq(
            max_position_embeddings=2 * scout_max,
            rope_parameters={**block, "rope_theta": scout["rope_theta"]},
        )
        native = _transformers_llama3_inv_freq(
            max_position_embeddings=scout_max, rope_parameters=scout
        )

        assert block["factor"] == 32.0
        assert (block["low_freq_factor"], block["high_freq_factor"]) == (1.0, 1.0)
        assert bool(torch.isfinite(composed).all())
        assert bool((composed <= native).all())
        assert bool((composed < native).any())

    @pytest.mark.parametrize(
        ("override", "match"),
        [
            ({"factor": None}, "factor"),
            ({"factor": float("nan")}, "factor"),
            ({"low_freq_factor": True}, "low_freq_factor"),
            ({"high_freq_factor": -4.0}, "high_freq_factor"),
            ({"original_max_position_embeddings": 0}, "original_max_position_embeddings"),
            ({"original_max_position_embeddings": 8192.5}, "original_max_position_embeddings"),
            ({"original_max_position_embeddings": None}, "original_max_position_embeddings"),
        ],
        ids=[
            "no-factor",
            "nan-factor",
            "bool-low",
            "negative-high",
            "zero-original",
            "float-original",
            "null-original",
        ],
    )
    def test_refuses_a_malformed_native_block(self, override: dict, match: str) -> None:
        from soup_cli.utils.long_context import compose_llama3_rope_parameters

        native = {**NATIVE_LLAMA31, **override}
        with pytest.raises(ValueError, match=match):
            compose_llama3_rope_parameters(
                native, max_position_embeddings=NATIVE_MAX, target_length=TARGET
            )

    def test_an_integral_float_original_length_is_kept_as_an_int(self) -> None:
        """transformers accepts ``8192.0`` there (it only warns), so this does too."""
        from soup_cli.utils.long_context import compose_llama3_rope_parameters

        block = compose_llama3_rope_parameters(
            {**NATIVE_LLAMA31, "original_max_position_embeddings": 8192.0},
            max_position_embeddings=NATIVE_MAX,
            target_length=TARGET,
        )

        assert block["original_max_position_embeddings"] == 8192
        assert type(block["original_max_position_embeddings"]) is int


# ---------------------------------------------------------------------------
# apply_long_context_config — compose or refuse, before any mutation
# ---------------------------------------------------------------------------


def _llama31_namespace(spelling: str = "rope_parameters") -> SimpleNamespace:
    """The native block in each spelling a config object can carry."""
    if spelling == "rope_parameters":
        return SimpleNamespace(
            max_position_embeddings=NATIVE_MAX, rope_parameters=dict(NATIVE_LLAMA31)
        )
    if spelling == "rope_scaling+type":
        legacy = {k: v for k, v in NATIVE_LLAMA31.items() if k != "rope_type"}
        return SimpleNamespace(
            max_position_embeddings=NATIVE_MAX, rope_scaling={**legacy, "type": "llama3"}
        )
    assert spelling == "rope_scaling+rope_type"
    return SimpleNamespace(max_position_embeddings=NATIVE_MAX, rope_scaling=dict(NATIVE_LLAMA31))


class TestApplyLongContextConfigOnScaledCheckpoints:
    @pytest.mark.parametrize(
        "spelling", ["rope_parameters", "rope_scaling+type", "rope_scaling+rope_type"]
    )
    @pytest.mark.parametrize("requested", ["llama3", None])
    def test_llama3_on_a_native_llama3_block_composes(self, spelling: str, requested) -> None:
        from soup_cli.utils.long_context import apply_long_context_config

        config = _llama31_namespace(spelling)
        applied = apply_long_context_config(
            config, target_length=TARGET, rope_scaling_type=requested
        )

        assert applied == COMPOSED_LLAMA31
        assert config.rope_parameters == COMPOSED_LLAMA31
        assert config.max_position_embeddings == TARGET

    def test_a_real_llama_config_composes(self) -> None:
        _requires_train_extra()
        from soup_cli.utils.long_context import apply_long_context_config

        config = _tiny_llama_config(
            max_position_embeddings=NATIVE_MAX, rope_parameters=NATIVE_LLAMA31
        )
        applied = apply_long_context_config(config, TARGET, "llama3")

        assert applied == COMPOSED_LLAMA31
        assert config.rope_parameters == COMPOSED_LLAMA31
        assert config.max_position_embeddings == TARGET

    def test_composition_keeps_partial_rotary_factor(self) -> None:
        from soup_cli.utils.long_context import apply_long_context_config

        config = SimpleNamespace(
            max_position_embeddings=NATIVE_MAX,
            rope_parameters={**NATIVE_LLAMA31, "partial_rotary_factor": 0.5},
        )
        applied = apply_long_context_config(config, TARGET, "llama3")

        assert applied == {**COMPOSED_LLAMA31, "partial_rotary_factor": 0.5}

    @pytest.mark.parametrize("requested", ["linear", "dynamic", "yarn", "longrope"])
    @pytest.mark.parametrize(
        "spelling", ["rope_parameters", "rope_scaling+type", "rope_scaling+rope_type"]
    )
    def test_other_types_on_a_native_llama3_block_are_refused_before_any_mutation(
        self, requested: str, spelling: str
    ) -> None:
        from soup_cli.utils.long_context import apply_long_context_config

        config = _llama31_namespace(spelling)
        before = _snapshot(config)

        with pytest.raises(ValueError, match=r"#1239") as exc_info:
            apply_long_context_config(config, TARGET, requested)

        _message_names(
            exc_info,
            f"rope_scaling_type={requested!r}",
            "rope_type='llama3', factor=8.0",
            _COMPOSE_HINT,
            f"at or below {NATIVE_MAX}",
        )
        assert _snapshot(config) == before

    @pytest.mark.parametrize("requested", REQUESTED_TYPES)
    @pytest.mark.parametrize(
        ("native", "shown"),
        [
            (NATIVE_YARN, "rope_type='yarn', factor=4.0"),
            (NATIVE_LONGROPE, "rope_type='longrope', factor=None"),
            (NATIVE_LINEAR, "rope_type='linear', factor=4.0"),
            (NATIVE_DYNAMIC, "rope_type='dynamic', factor=2.0"),
        ],
        ids=["yarn", "longrope", "linear", "dynamic"],
    )
    def test_extending_any_other_scaled_checkpoint_is_refused(
        self, native: dict, shown: str, requested
    ) -> None:
        from soup_cli.utils.long_context import apply_long_context_config

        config = SimpleNamespace(max_position_embeddings=NATIVE_MAX, rope_parameters=dict(native))
        before = _snapshot(config)

        with pytest.raises(ValueError, match=r"#1239") as exc_info:
            apply_long_context_config(config, TARGET, requested)

        _message_names(exc_info, shown, f"at or below {NATIVE_MAX}")
        if requested is None:
            _message_names(exc_info, "rope_scaling_type=None (auto-detected 'dynamic')")
        else:
            _message_names(exc_info, f"rope_scaling_type={requested!r}")
        # llama3 composes only with a llama3 block, so it is not offered here.
        assert _COMPOSE_HINT not in str(exc_info.value)
        assert _snapshot(config) == before

    def test_a_legacy_type_key_on_a_scaled_block_is_refused_too(self) -> None:
        from soup_cli.utils.long_context import apply_long_context_config

        legacy = {k: v for k, v in NATIVE_YARN.items() if k != "rope_type"}
        config = SimpleNamespace(
            max_position_embeddings=NATIVE_MAX, rope_scaling={**legacy, "type": "yarn"}
        )
        before = _snapshot(config)

        with pytest.raises(ValueError, match=re.escape("rope_type='yarn', factor=4.0")):
            apply_long_context_config(config, TARGET, "yarn")
        assert _snapshot(config) == before

    def test_autodetect_reads_the_type_the_way_transformers_does(self) -> None:
        """With both keys present transformers builds from ``rope_type``; the None
        auto-detect follows it, so a stale legacy ``type`` cannot turn a llama3
        block into a refused ``dynamic`` request."""
        from soup_cli.utils.long_context import apply_long_context_config

        config = SimpleNamespace(
            max_position_embeddings=NATIVE_MAX,
            rope_parameters={**NATIVE_LLAMA31, "type": "linear"},
        )

        assert apply_long_context_config(config, TARGET, None) == COMPOSED_LLAMA31

    @pytest.mark.parametrize(
        "override",
        [{"factor": None}, {"low_freq_factor": True}, {"original_max_position_embeddings": 0}],
        ids=["no-factor", "bool-low", "zero-original"],
    )
    @pytest.mark.parametrize("requested", ["llama3", None])
    def test_a_malformed_native_llama3_block_is_refused_with_the_setting_and_a_way_out(
        self, override: dict, requested
    ) -> None:
        from soup_cli.utils.long_context import apply_long_context_config

        config = SimpleNamespace(
            max_position_embeddings=NATIVE_MAX,
            rope_parameters={**NATIVE_LLAMA31, **override},
        )
        before = _snapshot(config)

        with pytest.raises(ValueError) as exc_info:
            apply_long_context_config(config, TARGET, requested)

        shown = "'llama3'" if requested else "None (auto-detected 'llama3')"
        _message_names(
            exc_info,
            f"training.rope_scaling_type={shown}",
            "#1239",
            next(iter(override)),
            f"at or below {NATIVE_MAX}",
        )
        assert _snapshot(config) == before


# ---------------------------------------------------------------------------
# Controls — behaviour that must NOT change
# ---------------------------------------------------------------------------


def _default_rope_configs():
    """Checkpoints without RoPE scaling, in the shapes the loader can hand over."""
    configs = {"no-rope-attributes": SimpleNamespace(max_position_embeddings=4096)}
    try:
        from transformers import LlamaConfig, MistralConfig
    except ImportError:  # pragma: no cover - the [train] extra is absent
        return configs
    configs["llama2-style"] = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        rope_theta=10000.0,
    )
    configs["mistral-style"] = MistralConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        rope_theta=1000000.0,
    )
    return configs


# What every rope_scaling_type produced for 4096 -> 16384 before #1239 (the
# control tests pass unchanged on the unfixed module, which is the point).
_EXPECTED_DEFAULT = {
    "linear": {"rope_type": "linear", "factor": 4.0},
    "dynamic": {"rope_type": "dynamic", "factor": 4.0},
    None: {"rope_type": "dynamic", "factor": 4.0},
    "yarn": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096},
    "llama3": {
        "rope_type": "llama3",
        "factor": 4.0,
        "original_max_position_embeddings": 4096,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
    },
}
_THETA = {"no-rope-attributes": None, "llama2-style": 10000.0, "mistral-style": 1000000.0}


class TestControls:
    @pytest.mark.parametrize("requested", ["linear", "dynamic", None, "yarn", "llama3"])
    @pytest.mark.parametrize("shape", ["no-rope-attributes", "llama2-style", "mistral-style"])
    def test_a_checkpoint_without_rope_scaling_extends_exactly_as_before(
        self, shape: str, requested
    ) -> None:
        from soup_cli.utils.long_context import apply_long_context_config

        configs = _default_rope_configs()
        if shape not in configs:
            pytest.skip("transformers is only in the [train] extra")
        config = configs[shape]
        expected = dict(_EXPECTED_DEFAULT[requested])
        if _THETA[shape] is not None:
            expected = {"rope_theta": _THETA[shape], **expected}

        applied = apply_long_context_config(config, 16384, requested)

        assert applied == expected
        assert list(applied) == list(expected)
        assert config.rope_parameters == expected
        assert config.max_position_embeddings == 16384

    @pytest.mark.parametrize("requested", REQUESTED_TYPES)
    @pytest.mark.parametrize("target", [NATIVE_MAX, 8192])
    @pytest.mark.parametrize(
        "native",
        [NATIVE_LLAMA31, NATIVE_YARN, NATIVE_LONGROPE],
        ids=["llama3", "yarn", "longrope"],
    )
    def test_a_target_at_or_below_max_is_still_a_no_op(
        self, native: dict, target: int, requested
    ) -> None:
        from soup_cli.utils.long_context import apply_long_context_config

        config = SimpleNamespace(max_position_embeddings=NATIVE_MAX, rope_parameters=dict(native))
        before = _snapshot(config)

        assert apply_long_context_config(config, target, requested) is None
        assert _snapshot(config) == before


# ---------------------------------------------------------------------------
# longrope — refused at config load
# ---------------------------------------------------------------------------

# What the refusal must say: why longrope can extend nothing, the issue, how to
# extend a checkpoint instead, and how to fine-tune a LongRoPE one.
_LONGROPE_REASONS = (
    "#1239",
    "cannot extend any checkpoint",
    "exist only on checkpoints already scaled with LongRoPE",
    "extending an already-scaled checkpoint is refused",
    "To extend a checkpoint without RoPE scaling, use linear, dynamic, yarn or llama3",
    "leave rope_scaling_type unset",
)


def _yaml_config(
    training: dict, *, task: str = "sft", backend: str = "transformers", max_length: int = 2048
) -> str:
    return yaml.safe_dump(
        {
            "base": "microsoft/Phi-3-mini-128k-instruct",
            "task": task,
            "backend": backend,
            "data": {"train": "train.jsonl", "max_length": max_length},
            "training": training,
        }
    )


class TestLongropeIsRefusedAtConfigLoad:
    @pytest.mark.parametrize(
        "spelling",
        ["longrope", "LongRoPE", "LONGROPE", " longrope ", "long_rope", "long-rope", "Long RoPE"],
    )
    def test_every_spelling_is_refused_with_the_reason(self, spelling: str) -> None:
        with pytest.raises(ValueError) as exc_info:
            load_config_from_string(_yaml_config({"rope_scaling_type": spelling}))

        _message_names(exc_info, "rope_scaling_type", *_LONGROPE_REASONS)

    @pytest.mark.parametrize(
        ("task", "backend", "max_length"),
        [
            ("sft", "transformers", 2048),
            ("sft", "transformers", 262144),
            ("pretrain", "transformers", 4096),
            ("dpo", "transformers", 2048),
            ("sft", "unsloth", 2048),
        ],
        ids=["sft-short", "sft-long", "pretrain", "dpo", "unsloth"],
    )
    def test_the_refusal_is_unconditional(self, task: str, backend: str, max_length: int) -> None:
        """Not tied to a task, a backend, or whether max_length extends anything."""
        with pytest.raises(ValueError) as exc_info:
            load_config_from_string(
                _yaml_config(
                    {"rope_scaling_type": "longrope"},
                    task=task,
                    backend=backend,
                    max_length=max_length,
                )
            )

        _message_names(exc_info, *_LONGROPE_REASONS)

    def test_the_training_model_refuses_it_directly(self) -> None:
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(rope_scaling_type="longrope")

        _message_names(exc_info, *_LONGROPE_REASONS)

    def test_an_unknown_type_is_no_longer_offered_longrope(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            load_config_from_string(_yaml_config({"rope_scaling_type": "ntk"}))

        message = str(exc_info.value)
        for offered in ("'linear'", "'dynamic'", "'yarn'", "'llama3'"):
            assert offered in message, (offered, message)
        assert "longrope" not in message.lower()

    @pytest.mark.parametrize("kind", ["linear", "dynamic", "yarn", "llama3", None])
    def test_the_other_types_and_unset_still_load(self, kind) -> None:
        """Control: only longrope is refused."""
        training = {} if kind is None else {"rope_scaling_type": kind}
        config = load_config_from_string(_yaml_config(training))

        assert config.training.rope_scaling_type == kind

    @pytest.mark.parametrize("shape", ["no-rope-attributes", "llama2-style", "mistral-style"])
    def test_the_runtime_helper_gives_the_same_answer(self, shape: str) -> None:
        """A direct call on a checkpoint without scaling no longer advises choosing
        a LongRoPE checkpoint, which the #1239 rule refuses to extend."""
        from soup_cli.utils.long_context import apply_long_context_config

        configs = _default_rope_configs()
        if shape not in configs:
            pytest.skip("transformers is only in the [train] extra")
        config = configs[shape]
        before = _snapshot(config)

        with pytest.raises(ValueError) as exc_info:
            apply_long_context_config(config, 16384, "longrope")

        _message_names(exc_info, *_LONGROPE_REASONS)
        assert "choose a checkpoint that ships LongRoPE factors" not in str(exc_info.value)
        assert _snapshot(config) == before


# ---------------------------------------------------------------------------
# End to end — the SFT and pretrain trainers
# ---------------------------------------------------------------------------


def _write_tiny_tokenizer(directory) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "yo"):
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    with open(directory / "tokenizer_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "model_max_length": TARGET,
                "clean_up_tokenization_spaces": False,
            },
            handle,
        )


def _save_tiny_checkpoint(directory, *, max_position_embeddings: int, rope_parameters: dict):
    import torch
    from transformers import LlamaForCausalLM

    torch.manual_seed(0)
    config = _tiny_llama_config(
        max_position_embeddings=max_position_embeddings, rope_parameters=rope_parameters
    )
    LlamaForCausalLM(config).save_pretrained(directory)
    _write_tiny_tokenizer(directory)
    return str(directory)


@pytest.fixture(scope="module")
def llama31_checkpoint(tmp_path_factory) -> str:
    _requires_train_extra()
    return _save_tiny_checkpoint(
        tmp_path_factory.mktemp("tiny-llama31"),
        max_position_embeddings=NATIVE_MAX,
        rope_parameters=NATIVE_LLAMA31,
    )


@pytest.fixture(scope="module")
def yarn_checkpoint(tmp_path_factory) -> str:
    _requires_train_extra()
    return _save_tiny_checkpoint(
        tmp_path_factory.mktemp("tiny-yarn"),
        max_position_embeddings=NATIVE_MAX,
        rope_parameters=NATIVE_YARN,
    )


@pytest.fixture(scope="module")
def longrope_checkpoint(tmp_path_factory) -> str:
    """A config-built Phi-3 shaped like Phi-3-mini-128k: LongRoPE over 4096, window 131072."""
    _requires_train_extra()
    import torch
    from transformers import Phi3Config, Phi3ForCausalLM

    half = 32  # head_dim 64 -> 32 frequency pairs, one factor each
    torch.manual_seed(0)
    config = Phi3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=NATIVE_MAX,
        original_max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "longrope",
            "short_factor": [1.0] * half,
            "long_factor": [1.0 + 15.0 * k / (half - 1) for k in range(half)],
            "rope_theta": 10000.0,
        },
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
    )
    directory = tmp_path_factory.mktemp("tiny-phi3-longrope")
    Phi3ForCausalLM(config).save_pretrained(directory)
    _write_tiny_tokenizer(directory)
    return str(directory)


@pytest.fixture(scope="module")
def default_rope_checkpoint(tmp_path_factory) -> str:
    _requires_train_extra()
    return _save_tiny_checkpoint(
        tmp_path_factory.mktemp("tiny-default"),
        max_position_embeddings=8192,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
    )


def _config(
    base: str,
    output,
    *,
    task: str = "sft",
    rope_type: str | None = "llama3",
    max_length: int = TARGET,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
):
    lora = {"r": 4, "alpha": 8, "target_modules": list(target_modules)}
    data = {"train": "unused.jsonl", "max_length": max_length}
    if task == "sft":
        data["chat_template"] = "chatml"
    training = {"quantization": "none", "batch_size": 1, "epochs": 1, "lora": lora}
    if rope_type is not None:  # None leaves the key out: the setting is unset
        training["rope_scaling_type"] = rope_type
    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": task,
                "backend": "transformers",
                "data": data,
                "training": training,
                "output": str(output),
            }
        )
    )


_CHAT_ROWS = {
    "train": [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    ]
    * 2
}
_TEXT_ROWS = {"train": [{"text": "hello world hi yo"}] * 2}


def _rotary(model):
    matches = [module.rotary_emb for module in model.modules() if hasattr(module, "rotary_emb")]
    assert matches
    return matches[0]


def _assert_built_with_the_composed_block(model) -> None:
    import torch

    built = _rotary(model).inv_freq
    expected = _transformers_llama3_inv_freq(
        max_position_embeddings=TARGET, rope_parameters=COMPOSED_LLAMA31
    )
    native = _native_inv_freq()

    assert model.config.rope_parameters == COMPOSED_LLAMA31
    assert model.config.max_position_embeddings == TARGET
    assert torch.equal(built, expected)
    faster = [int(k) for k in torch.nonzero(built > native).flatten()]
    assert faster == [], f"pairs rotating faster than pretrained: {faster}"
    assert int((built != native).sum()) == 35


class TestTrainers:
    def test_sft_setup_composes_the_native_llama3_block(self, llama31_checkpoint, tmp_path) -> None:
        from soup_cli.trainer.sft import SFTTrainerWrapper

        config = _config(llama31_checkpoint, tmp_path / "out")
        wrapper = SFTTrainerWrapper(config, device="cpu")
        wrapper.setup(deepcopy(_CHAT_ROWS))

        _assert_built_with_the_composed_block(wrapper.model)

    # longrope never reaches setup: config load refuses it (TestLongropeIsRefusedAtConfigLoad).
    @pytest.mark.parametrize("requested", ["linear", "dynamic", "yarn"])
    def test_sft_setup_refuses_before_the_model_is_built(
        self, llama31_checkpoint, tmp_path, requested: str
    ) -> None:
        from soup_cli.trainer.sft import SFTTrainerWrapper

        config = _config(llama31_checkpoint, tmp_path / "out", rope_type=requested)
        wrapper = SFTTrainerWrapper(config, device="cpu")

        with pytest.raises(ValueError, match=r"#1239") as exc_info:
            wrapper.setup(deepcopy(_CHAT_ROWS))

        _message_names(
            exc_info, f"rope_scaling_type={requested!r}", "rope_type='llama3', factor=8.0"
        )
        assert wrapper.model is None

    def test_sft_refuses_to_extend_a_native_yarn_checkpoint_again(
        self, yarn_checkpoint, tmp_path
    ) -> None:
        from soup_cli.trainer.sft import SFTTrainerWrapper

        config = _config(yarn_checkpoint, tmp_path / "out", rope_type="yarn")
        wrapper = SFTTrainerWrapper(config, device="cpu")

        with pytest.raises(ValueError, match=re.escape("rope_type='yarn', factor=4.0")):
            wrapper._setup_transformers(config, config.training)
        assert wrapper.model is None

    def test_pretrain_composes_the_native_llama3_block(self, llama31_checkpoint, tmp_path) -> None:
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        config = _config(llama31_checkpoint, tmp_path / "out", task="pretrain")
        wrapper = PretrainTrainerWrapper(config, device="cpu")
        wrapper._setup_transformers(config, config.training)

        _assert_built_with_the_composed_block(wrapper.model)

    def test_pretrain_refuses_before_the_model_is_built(self, llama31_checkpoint, tmp_path) -> None:
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        config = _config(llama31_checkpoint, tmp_path / "out", task="pretrain", rope_type="linear")
        wrapper = PretrainTrainerWrapper(config, device="cpu")

        with pytest.raises(ValueError, match=re.escape("rope_type='llama3', factor=8.0")):
            wrapper._setup_transformers(config, config.training)
        assert wrapper.model is None

    def test_sft_target_within_the_native_window_leaves_the_block_alone(
        self, llama31_checkpoint, tmp_path
    ) -> None:
        """Control: nothing is extended, so nothing is refused or rebuilt."""
        import torch

        from soup_cli.trainer.sft import SFTTrainerWrapper

        config = _config(llama31_checkpoint, tmp_path / "out", rope_type="linear", max_length=4096)
        wrapper = SFTTrainerWrapper(config, device="cpu")
        wrapper._setup_transformers(config, config.training)

        assert wrapper.model.config.max_position_embeddings == NATIVE_MAX
        assert wrapper.model.config.rope_parameters == NATIVE_LLAMA31
        assert torch.equal(_rotary(wrapper.model).inv_freq, _native_inv_freq())

    @pytest.mark.parametrize("task", ["sft", "pretrain"])
    def test_a_longrope_checkpoint_fine_tunes_at_its_native_length(
        self, longrope_checkpoint, tmp_path, task: str
    ) -> None:
        """Control: refusing rope_scaling_type=longrope must not refuse the checkpoint.

        Unset rope_scaling_type, set data.max_length to the full native window and
        run the real setup() and train(): optimizer steps happen with a finite loss,
        and the model keeps the checkpoint's own LongRoPE block. The rows are short,
        so the short-factor half of LongRoPE is the one exercised.
        """
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        config = _config(
            longrope_checkpoint,
            tmp_path / "out",
            task=task,
            rope_type=None,
            max_length=NATIVE_MAX,
            target_modules=("qkv_proj",),
        )
        assert config.training.rope_scaling_type is None
        if task == "sft":
            from soup_cli.trainer.sft import SFTTrainerWrapper

            wrapper = SFTTrainerWrapper(config, device="cpu")
            wrapper.setup(deepcopy(_CHAT_ROWS))
        else:
            from soup_cli.trainer.pretrain import PretrainTrainerWrapper

            wrapper = PretrainTrainerWrapper(config, device="cpu")
            wrapper.setup(deepcopy(_TEXT_ROWS))
        result = wrapper.train()

        assert result["total_steps"] >= 1, result
        assert result["loss_summary_kind"] != "unavailable", result
        assert math.isfinite(result["final_loss"]) and result["final_loss"] > 0, result

        native_config = AutoConfig.from_pretrained(longrope_checkpoint)
        native_rotary = _rotary(AutoModelForCausalLM.from_pretrained(longrope_checkpoint))
        built_rotary = _rotary(wrapper.model)

        assert native_config.rope_parameters["rope_type"] == "longrope"
        assert wrapper.model.config.rope_parameters == native_config.rope_parameters
        assert wrapper.model.config.max_position_embeddings == NATIVE_MAX
        assert torch.equal(built_rotary.inv_freq, native_rotary.inv_freq)
        assert built_rotary.attention_scaling == native_rotary.attention_scaling

    def test_sft_default_rope_checkpoint_extends_exactly_as_before(
        self, default_rope_checkpoint, tmp_path
    ) -> None:
        """Control: a checkpoint without scaling still gets the block it got before."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        config = _config(default_rope_checkpoint, tmp_path / "out", max_length=32768)
        wrapper = SFTTrainerWrapper(config, device="cpu")
        wrapper._setup_transformers(config, config.training)

        assert wrapper.model.config.max_position_embeddings == 32768
        assert wrapper.model.config.rope_parameters == {
            "rope_theta": 500000.0,
            "rope_type": "llama3",
            "factor": 4.0,
            "original_max_position_embeddings": 8192,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        }
