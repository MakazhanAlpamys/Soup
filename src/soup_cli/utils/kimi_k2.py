"""Kimi K2 on layer streaming: its ``config.json`` loaded into transformers' own
``DeepseekV3Config``, behind a guard.

Kimi-K2-Base and -Instruct (moonshotai) declare ``model_type: "kimi_k2"`` and an ``auto_map``
onto their repository's own ``configuration_deepseek`` / ``modeling_deepseek``. The decoder is
DeepSeek-V3's: the configuration class there declares ``model_type = "deepseek_v3"``, the
checkpoints declare ``DeepseekV3ForCausalLM``, and transformers itself maps a ``kimi_k2`` text
config onto ``deepseek_v3`` (``Kimi_K25Config``). transformers registers no ``kimi_k2`` config,
though, so ``AutoConfig.from_pretrained(trust_remote_code=False)`` refuses the repository as
custom code, and layer streaming never runs a repository's modelling code
(``StreamingSetupMixin._STREAM_NATIVE_MODEL_CODE``, #1933).

:func:`load_stream_model_config` is the one config loader of the streamed model (the trainer
setup and :func:`~soup_cli.utils.layer_stream_runtime.build_meta_skeleton`). Any config
``AutoConfig`` loads is returned exactly as before. A ``kimi_k2`` config that ``AutoConfig``
refuses is read as a plain dict (no code runs), checked by :func:`kimi_k2_guard_failures`, and
loaded into ``DeepseekV3Config`` with ``model_type`` rewritten to ``deepseek_v3`` and
``auto_map`` dropped: the rewrite ``AutoConfig`` would do if the file said ``deepseek_v3``.

The guard admits a config only where the native class computes what K2's own
``modeling_deepseek.py`` computes; each row names the difference it rules out. Every key of the
published file must be present (``auto_map`` and ``quantization_config`` may be absent), and no
other key may be. A config that fails any row is refused with :class:`UnsupportedKimiK2Error`,
naming every failing field and its value, and saying that other K2 releases are out of scope for
now. Kimi-K2-Thinking (compressed-tensors INT4) and Kimi-K2-Instruct-0905 (serialised defaults,
``ep_size``, ``pad_token_id``) are refused this way. The evidence is
``benchmarks/gate-kimi-k2-admission.md``.

Known numeric differences from K2's own code, which the guard does not and cannot remove: the
MoE combine (K2 sums the weighted expert outputs in fp32 and rounds once; the native class
rounds each weighted output to the model dtype before ``index_add_``), eager versus sdpa
attention, and the router bias held as a Parameter (K2) versus a float32 buffer (native).

No top-level torch or transformers: the trainer imports this on its setup path.
"""

from __future__ import annotations

import copy
import importlib.util
from typing import Any, Mapping, NamedTuple, Optional, Tuple

KIMI_K2_MODEL_TYPE = "kimi_k2"
#: The ``model_type`` the rewritten config carries: transformers' native DeepSeek-V3.
NATIVE_MODEL_TYPE = "deepseek_v3"
#: Keys the rewrite drops: ``auto_map`` points at the repository's own code.
_DROPPED_KEYS = ("auto_map",)
_ARCHITECTURES = ["DeepseekV3ForCausalLM"]
#: Every top-level key of the published Kimi-K2-Base and Kimi-K2-Instruct ``config.json``
#: (byte-identical files; revisions in ``benchmarks/gate-kimi-k2-admission.md``). The guard is an
#: allowlist: a key outside this set and :data:`_METADATA_KEYS` is refused by name, because the
#: native class may read a key K2's own code ignores (``head_dim``, ``partial_rotary_factor``, a
#: transformers-5 ``rope_parameters`` ...) and then compute differently or crash.
_PUBLISHED_KEYS = frozenset(
    {
        "architectures", "attention_bias", "attention_dropout", "auto_map", "aux_loss_alpha",
        "bos_token_id", "eos_token_id", "first_k_dense_replace", "hidden_act", "hidden_size",
        "initializer_range", "intermediate_size", "kv_lora_rank", "max_position_embeddings",
        "model_type", "moe_intermediate_size", "moe_layer_freq", "n_group", "n_routed_experts",
        "n_shared_experts", "norm_topk_prob", "num_attention_heads", "num_experts_per_tok",
        "num_hidden_layers", "num_key_value_heads", "num_nextn_predict_layers", "pretraining_tp",
        "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "quantization_config",
        "rms_norm_eps", "rope_theta", "routed_scaling_factor", "rope_scaling", "scoring_func",
        "seq_aux", "tie_word_embeddings", "topk_group", "topk_method", "torch_dtype",
        "transformers_version", "use_cache", "v_head_dim", "vocab_size",
    }
)
#: Keys outside the published file, admitted because neither class computes with them.
_METADATA_KEYS = frozenset(
    {
        # Added by transformers' own reader (``get_config_dict``) on 5.17, to every dict it
        # returns (a Hub snapshot's commit, or None); 5.19 adds it to none.
        # ``PreTrainedConfig`` pops it as the Hub revision of the file, not a model setting.
        "_commit_hash",
        # Written by ``save_pretrained`` as the source path: a provenance stamp only.
        "_name_or_path",
        # transformers 5's spelling of the published ``torch_dtype``: the checkpoint's storage
        # dtype, which the streamed model does not use (it is built at the stream dtype).
        "dtype",
    }
)
_ALLOWED_KEYS = _PUBLISHED_KEYS | _METADATA_KEYS
_UNEXPECTED_REASON = (
    "unexpected field {name}: training.stream_layers admits kimi_k2 only with the published K2 "
    "config fields"
)
#: Published keys a config may leave out: the rewrite drops ``auto_map`` anyway, and a copy
#: dequantised to bf16 carries no ``quantization_config``. Every other published key must be
#: present: a missing one would not be refused by the allowlist, and the native class would
#: compute with its own default for it (``rope_theta`` 10000.0 where K2 publishes 50000.0,
#: ``routed_scaling_factor`` 2.5 where K2 publishes 2.827, ...).
_OPTIONAL_PUBLISHED_KEYS = frozenset({"auto_map", "quantization_config"})
_REQUIRED_KEYS = tuple(sorted(_PUBLISHED_KEYS - _OPTIONAL_PUBLISHED_KEYS))
#: A required key either spelling satisfies: transformers 5 writes ``torch_dtype`` as ``dtype``.
_KEY_SPELLINGS = {"torch_dtype": ("torch_dtype", "dtype")}
_MISSING_REASON = (
    "missing field: the published Kimi-K2-Base / -Instruct config sets it, and without it the "
    "native DeepseekV3Config would use its own default, which need not be K2's value"
)
#: Said in every refusal, before the field list (which a release's file can make long).
_OUT_OF_SCOPE = (
    "Other Kimi K2 releases (Kimi-K2-Instruct-0905, Kimi-K2-Thinking) and copies whose config "
    "fields differ from the published Base / Instruct file are out of scope for now: layer "
    "streaming does not admit them yet."
)
#: The ``rope_scaling`` keys K2's yarn reads. The native yarn reads more (``attention_factor``,
#: ``truncate``, ``partial_rotary_factor``), each of which K2 ignores.
_ROPE_KEYS = frozenset(
    {
        "type",
        "rope_type",
        "factor",
        "original_max_position_embeddings",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
    }
)
#: Longest rendering of a config value or key in a message; both come from an untrusted file.
_BRIEF_LIMIT = 80
#: Fields a refusal lists before "and N more": a file can hold any number of unexpected keys.
_MAX_LISTED = 10
_ABSENT = object()


class UnsupportedKimiK2Error(ValueError):
    """A ``kimi_k2`` config the native DeepSeek-V3 class would not compute like K2's own code."""


class GuardFailure(NamedTuple):
    """One field the guard refuses: its name, its value as shown, and what would differ."""

    field: str
    value: str
    reason: str


def _bounded(text: str) -> str:
    return text if len(text) <= _BRIEF_LIMIT else text[: _BRIEF_LIMIT - 3] + "..."


def _brief(value: Any) -> str:
    """``ascii`` of an untrusted value, truncated; ``absent`` for a missing one.

    ``ascii`` escapes every control and non-ASCII character (ESC, BEL, U+202E, a Cyrillic
    homoglyph ...), so a value is shown as the bytes it is; the CLI's error printer then applies
    ``for_terminal`` to the whole message (``utils/errors.py``).
    """
    if value is _ABSENT:
        return "absent"
    return _bounded(ascii(value))


def _key_name(key: Any) -> str:
    """A top-level key as a refusal names it: a plain ASCII identifier as it is, anything else
    through :func:`_brief`; both bounded."""
    if isinstance(key, str) and key.isascii() and key.isidentifier():
        return _bounded(key)
    return _brief(key)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _rope_failures(rope: Any) -> list:
    """The ``rope_scaling`` rows: a yarn dict K2's yarn and the native yarn compute alike."""
    if rope is _ABSENT or rope is None:
        return []  # both use their plain rotary embedding (G1c "no-scaling")
    if not isinstance(rope, Mapping):
        return [
            GuardFailure(
                "rope_scaling", _brief(rope), "K2 reads rope_scaling as a dict (or none at all)"
            )
        ]
    if not rope:
        return [
            GuardFailure(
                "rope_scaling", _brief(dict(rope)),
                "K2 indexes rope_scaling['type'] whenever rope_scaling is not null, so an empty "
                "dict fails there, while the native class treats it as no scaling",
            )
        ]
    failures = []
    rope_kind = rope.get("type", _ABSENT)
    if rope_kind != "yarn":
        failures.append(
            GuardFailure(
                "rope_scaling.type", _brief(rope_kind),
                "only K2's yarn is admitted (K2 reads 'type'; its linear and dynamic NTK "
                "embeddings are not the native ones)",
            )
        )
    if rope.get("rope_type", "yarn") != "yarn":
        failures.append(
            GuardFailure(
                "rope_scaling.rope_type", _brief(rope.get("rope_type")),
                "the native class reads rope_type, K2 reads type: both must say yarn",
            )
        )
    for key in rope:
        if key not in _ROPE_KEYS:
            failures.append(
                GuardFailure(
                    f"rope_scaling[{_brief(key)}]", _brief(rope[key]),
                    "K2's yarn ignores this key and the native yarn may read it",
                )
            )
    if not _is_number(rope.get("factor", _ABSENT)):
        failures.append(
            GuardFailure(
                "rope_scaling.factor", _brief(rope.get("factor", _ABSENT)),
                "K2's yarn needs a numeric factor",
            )
        )
    if "original_max_position_embeddings" not in rope:
        failures.append(
            GuardFailure(
                "rope_scaling.original_max_position_embeddings", "absent",
                "K2 defaults it to 4096, the native class to max_position_embeddings",
            )
        )
    for key, default in (("beta_fast", 32), ("beta_slow", 1)):
        if key in rope and not rope[key]:
            failures.append(
                GuardFailure(
                    f"rope_scaling.{key}", _brief(rope[key]),
                    f"the native yarn replaces a zero with {default}; K2 divides by it",
                )
            )
    mscale = rope.get("mscale", _ABSENT)
    if rope.get("mscale_all_dim"):
        consistent = mscale is not _ABSENT and bool(mscale)
    else:
        consistent = mscale is _ABSENT or mscale == 1
    if not consistent:
        failures.append(
            GuardFailure(
                "rope_scaling.mscale", _brief(mscale),
                "the native yarn applies mscale only beside a non-zero mscale_all_dim (and "
                "assumes 1 otherwise); K2 always applies both, so their attention factors "
                "differ",
            )
        )
    return failures


def _quantization_failures(quant: Any) -> list:
    """Unquantised, or the FP8 block spelling the sharder dequantises (``utils/fp8_block``)."""
    from soup_cli.utils.fp8_block import UnsupportedFP8Error, fp8_block_size

    if quant is _ABSENT or quant is None:
        return []
    try:
        block = fp8_block_size(quant, arch=NATIVE_MODEL_TYPE)
    except UnsupportedFP8Error as exc:
        return [GuardFailure("quantization_config", _brief(quant), str(exc))]
    if block is not None:
        return []
    if isinstance(quant, Mapping) and "quant_method" in quant:
        return [
            GuardFailure(
                "quantization_config.quant_method", _brief(quant["quant_method"]),
                "layer streaming reads Kimi K2 unquantised or in DeepSeek-V3's block FP8 (the "
                "format K2-Base and K2-Instruct are published in); it does not dequantise this "
                "one (Kimi-K2-Thinking's is compressed-tensors INT4)",
            )
        ]
    return [
        GuardFailure(
            "quantization_config", _brief(quant),
            "not a quantization_config layer streaming can read",
        )
    ]


def kimi_k2_guard_failures(raw: Mapping[str, Any]) -> Tuple[GuardFailure, ...]:
    """Every field of a ``kimi_k2`` config the native class would not compute like K2, in
    guard order; ``()`` when the config is admitted.

    The order is the table in ``benchmarks/gate-kimi-k2-admission.md`` (G1b), then every
    published key that is absent (second review; ``auto_map`` and ``quantization_config`` may
    be), then the allowlist of top-level keys added in the review round. Its ``ep_size`` and
    ``rope_interleave`` rows are now the allowlist's: neither key is in the published file, so
    either is refused whenever present.
    """

    def value(key: str) -> Any:
        return raw.get(key, _ABSENT)

    failures = []

    def check(key: str, admitted: bool, reason: str) -> None:
        if not admitted:
            failures.append(GuardFailure(key, _brief(value(key)), reason))

    check(
        "architectures", value("architectures") == _ARCHITECTURES,
        "the published checkpoints declare exactly ['DeepseekV3ForCausalLM'], the decoder "
        "this mapping was checked on",
    )
    check(
        "topk_method", value("topk_method") == "noaux_tc",
        "K2's router implements only 'noaux_tc'; the native router always runs it",
    )
    check(
        "scoring_func", value("scoring_func") == "sigmoid",
        "K2's router implements only 'sigmoid'; the native router always uses it",
    )
    group_reason = (
        "K2 fills unselected expert groups with 0.0 and the native router with -inf; with one "
        "group (and one group picked) nothing is masked, so the two pick the same experts"
    )
    check("n_group", _is_int(value("n_group")) and value("n_group") == 1, group_reason)
    check("topk_group", _is_int(value("topk_group")) and value("topk_group") == 1, group_reason)
    top_k = value("num_experts_per_tok")
    renormalised = raw.get("norm_topk_prob", True)
    check(
        "num_experts_per_tok", _is_int(top_k) and (top_k >= 2 or not renormalised),
        "K2 renormalises the top-k weights only when top-k > 1; the native router always does",
    )
    check(
        "moe_layer_freq", value("moe_layer_freq") is _ABSENT or (
            _is_int(value("moe_layer_freq")) and value("moe_layer_freq") == 1
        ),
        "K2 makes only every n-th layer an MoE layer; the native class makes every layer past "
        "first_k_dense_replace one",
    )
    check(
        "n_routed_experts",
        _is_int(value("n_routed_experts")) and value("n_routed_experts") >= 1,
        "without routed experts K2 builds dense layers; the native class cannot",
    )
    check(
        "n_shared_experts",
        _is_int(value("n_shared_experts")) and value("n_shared_experts") >= 0,
        "without it K2 drops the shared expert; the native class always builds one",
    )
    check(
        "num_key_value_heads",
        _is_int(value("num_key_value_heads"))
        and value("num_key_value_heads") == value("num_attention_heads"),
        "K2's MLA ignores it, while the native class divides num_attention_heads by it (and "
        "defaults it to 128 when absent): only equal to num_attention_heads, as published, do "
        "the two compute alike",
    )
    failures.extend(_rope_failures(value("rope_scaling")))
    failures.extend(_quantization_failures(value("quantization_config")))
    # Every published key present (second review): absent, the native class would compute
    # with its own default for it. A key whose row above already refused it is named once.
    named = {failure.field for failure in failures}
    for key in _REQUIRED_KEYS:
        if key not in named and not any(
            spelling in raw for spelling in _KEY_SPELLINGS.get(key, (key,))
        ):
            failures.append(GuardFailure(key, _brief(_ABSENT), _MISSING_REASON))
    # Last, so the registered rows above keep their order (added in the review round).
    for key in raw:
        if key not in _ALLOWED_KEYS:  # whatever the value, null included
            name = _key_name(key)
            quoted = name if name.startswith(("'", '"')) else f"'{name}'"
            failures.append(
                GuardFailure(name, _brief(raw[key]), _UNEXPECTED_REASON.format(name=quoted))
            )
    return tuple(failures)


def _refusal(failures: Tuple[GuardFailure, ...]) -> UnsupportedKimiK2Error:
    listed = "; ".join(
        f"{item.field}={item.value} ({item.reason})" for item in failures[:_MAX_LISTED]
    )
    if len(failures) > _MAX_LISTED:
        listed += f"; and {len(failures) - _MAX_LISTED} more"
    return UnsupportedKimiK2Error(
        f"Kimi K2 (model_type '{KIMI_K2_MODEL_TYPE}') streams through transformers' own "
        f"DeepseekV3 class, admitted only for a config it computes like K2's own modelling "
        f"code: Kimi-K2-Base and Kimi-K2-Instruct as published (FP8 block, or dequantised to "
        f"bf16). {_OUT_OF_SCOPE} This config differs: {listed}."
    )


def kimi_k2_native_config_dict(raw: Any) -> dict:
    """The guarded rewrite: a deep copy of ``raw`` with ``model_type`` set to ``deepseek_v3``
    and ``auto_map`` dropped. Raises :class:`UnsupportedKimiK2Error` naming every field the
    guard refuses. ``raw`` itself is never changed (transformers' rope standardisation writes
    into the ``rope_scaling`` dict it is handed, hence the deep copy).
    """
    if not isinstance(raw, Mapping):
        raise UnsupportedKimiK2Error(
            f"a Kimi K2 config.json must hold a JSON object; got {_brief(raw)}, "
            f"which is not a JSON object"
        )
    failures = kimi_k2_guard_failures(raw)
    if failures:
        raise _refusal(failures)
    native = {key: value for key, value in copy.deepcopy(dict(raw)).items()
              if key not in _DROPPED_KEYS}
    native["model_type"] = NATIVE_MODEL_TYPE
    return native


def _raw_config_dict(model_id: str, kwargs: Mapping[str, Any]) -> Optional[tuple]:
    """``(config dict, unused kwargs)`` exactly as ``AutoConfig.from_pretrained`` reads them,
    or ``None`` when the file cannot be read. Reads JSON only: no code is imported or run."""
    try:
        from transformers import PretrainedConfig

        # AutoConfig hands name_or_path through get_config_dict to from_dict the same way.
        config_dict, unused = PretrainedConfig.get_config_dict(
            model_id, **dict(kwargs, name_or_path=model_id)
        )
    except Exception:  # noqa: BLE001 -- unreadable means "not a K2 we can admit"
        return None
    if not isinstance(config_dict, dict):
        return None
    return config_dict, unused


def load_stream_model_config(model_id: str, *, trust_remote_code: bool = False, **kwargs: Any):
    """The streamed model's config: ``AutoConfig``'s, or a guarded native one for Kimi K2.

    ``kwargs`` (a pinned ``revision``) reach both reads. ``trust_remote_code=True`` is handed
    to ``AutoConfig`` unchanged, with no K2 handling: that is the repository's own config.
    Otherwise every config ``AutoConfig`` loads is returned as it loads it. When ``AutoConfig``
    refuses (a ``ValueError``: K2 is "custom code" to it), the raw ``config.json`` is read;
    anything but ``model_type: kimi_k2`` gets ``AutoConfig``'s own error back, and a K2 config
    goes through :func:`kimi_k2_native_config_dict` into ``DeepseekV3Config``.
    """
    from transformers import AutoConfig

    if trust_remote_code:
        return AutoConfig.from_pretrained(model_id, trust_remote_code=True, **kwargs)
    try:
        return AutoConfig.from_pretrained(model_id, trust_remote_code=False, **kwargs)
    except ValueError:
        read = _raw_config_dict(model_id, kwargs)
        if read is None or read[0].get("model_type") != KIMI_K2_MODEL_TYPE:
            raise
    config_dict, unused = read
    native = kimi_k2_native_config_dict(config_dict)

    from transformers import DeepseekV3Config

    return DeepseekV3Config.from_dict(native, **unused)


def _tiktoken_available() -> bool:
    return importlib.util.find_spec("tiktoken") is not None


def kimi_k2_tokenizer_hint(model_id: str, **kwargs: Any) -> Optional[str]:
    """What to tell a user whose tokenizer load failed on a Kimi K2 base, or ``None``.

    Called only on that failure path. ``None`` unless the base's ``config.json`` declares
    ``model_type: kimi_k2``.
    """
    read = _raw_config_dict(model_id, kwargs)
    if read is None or read[0].get("model_type") != KIMI_K2_MODEL_TYPE:
        return None
    hint = (
        "Kimi K2's tokenizer exists only as code in its repository (tokenization_kimi.py), so "
        "it needs --trust-remote-code. Under layer streaming that flag reaches the tokenizer "
        "only: the config and the model are still transformers' own DeepseekV3 classes, never "
        "the repository's code."
    )
    if not _tiktoken_available():
        hint += (
            " That tokenizer also imports the tiktoken package, which Soup does not install: "
            "pip install tiktoken."
        )
    return hint


def print_kimi_k2_tokenizer_hint(model_id: str, console: Any, **kwargs: Any) -> None:
    """Print :func:`kimi_k2_tokenizer_hint` after a failed tokenizer load, if there is one.

    Never raises: the caller re-raises the tokenizer's own error, which a hint must not hide.
    """
    try:
        hint = kimi_k2_tokenizer_hint(model_id, **kwargs)
        if hint:
            console.print(hint, style="yellow", markup=False, highlight=False)
    except Exception:  # noqa: BLE001 -- a hint never replaces the real error
        return
