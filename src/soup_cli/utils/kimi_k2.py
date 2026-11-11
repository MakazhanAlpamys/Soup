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
import os
import re
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
#: The resident route's wording (ruling D15): no streamed-route naming anywhere in a
#: resident refusal, and the resident loader dequantises the published FP8 block spelling.
_UNEXPECTED_REASON_RESIDENT = (
    "unexpected field {name}: the resident route admits kimi_k2 only with the published K2 "
    "config fields"
)
#: The resident route's FP8 support wording (ruling D15): replaces ``fp8_block``'s
#: streamed tail in a resident refusal.
_FP8_RESIDENT_SUPPORT = (
    "The resident route dequantises one FP8 format: DeepSeek-V3 FP8 block "
    "(weight_block_size [128,128], e4m3, float scales) - see "
    "docs/performance-and-quantization.md."
)
_UNEXPECTED_REASONS = {"stream": _UNEXPECTED_REASON, "resident": _UNEXPECTED_REASON_RESIDENT}
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


def _quantization_failures(quant: Any, route: str = "stream") -> list:
    """Unquantised, or the FP8 block spelling the sharder dequantises (``utils/fp8_block``)."""
    from soup_cli.utils.fp8_block import SUPPORTED_FP8_FORMAT, UnsupportedFP8Error, fp8_block_size

    if quant is _ABSENT or quant is None:
        return []
    try:
        block = fp8_block_size(quant, arch=NATIVE_MODEL_TYPE)
    except UnsupportedFP8Error as exc:
        detail = str(exc)
        if route == "resident":
            # fp8_block's tail names layer streaming (its other callers are the streamed
            # sharder); a resident refusal must not (D15). The problem sentence is
            # route-neutral; swap the support tail for the resident route's own.
            detail = detail.removesuffix(SUPPORTED_FP8_FORMAT).rstrip()
            detail = f"{detail} {_FP8_RESIDENT_SUPPORT}"
        return [GuardFailure("quantization_config", _brief(quant), detail)]
    if block is not None:
        return []
    if isinstance(quant, Mapping) and "quant_method" in quant:
        if route == "resident":
            detail = (
                "the resident route reads Kimi K2 unquantised or dequantises DeepSeek-V3's "
                "block FP8 (the format K2-Base and K2-Instruct are published in); it does not "
                "handle this one (Kimi-K2-Thinking's is compressed-tensors INT4)"
            )
        else:
            detail = (
                "layer streaming reads Kimi K2 unquantised or in DeepSeek-V3's block FP8 (the "
                "format K2-Base and K2-Instruct are published in); it does not dequantise this "
                "one (Kimi-K2-Thinking's is compressed-tensors INT4)"
            )
        return [
            GuardFailure("quantization_config.quant_method", _brief(quant["quant_method"]), detail)
        ]
    if route == "resident":
        detail = "not a quantization_config the resident route can read"
    else:
        detail = "not a quantization_config layer streaming can read"
    return [GuardFailure("quantization_config", _brief(quant), detail)]


def kimi_k2_guard_failures(
    raw: Mapping[str, Any], route: str = "stream"
) -> Tuple[GuardFailure, ...]:
    """Every field of a ``kimi_k2`` config the native class would not compute like K2, in
    guard order; ``()`` when the config is admitted.

    ``route`` words the refusal reasons: ``"stream"`` (the default, the streamed loader's
    wording, pinned byte-identical by its suite) or ``"resident"`` (no streamed-route
    naming anywhere, ruling D15).

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
    failures.extend(_quantization_failures(value("quantization_config"), route))
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
                GuardFailure(
                    name, _brief(raw[key]),
                    _UNEXPECTED_REASONS[route].format(name=quoted),
                )
            )
    return tuple(failures)


def _refusal(failures: Tuple[GuardFailure, ...], route: str = "stream") -> UnsupportedKimiK2Error:
    listed = "; ".join(
        f"{item.field}={item.value} ({item.reason})" for item in failures[:_MAX_LISTED]
    )
    if len(failures) > _MAX_LISTED:
        listed += f"; and {len(failures) - _MAX_LISTED} more"
    if route == "stream":
        lead = (
            f"Kimi K2 (model_type '{KIMI_K2_MODEL_TYPE}') streams through transformers' own "
            f"DeepseekV3 class"
        )
        scope = _OUT_OF_SCOPE
    else:
        lead = (
            f"Kimi K2 (model_type '{KIMI_K2_MODEL_TYPE}') runs resident through transformers' "
            f"own DeepseekV3 class"
        )
        scope = _OUT_OF_SCOPE.replace(
            "layer streaming does not admit them yet", "the resident route does not admit them yet"
        )
    return UnsupportedKimiK2Error(
        f"{lead}, admitted only for a config it computes like K2's own modelling "
        f"code: Kimi-K2-Base and Kimi-K2-Instruct as published (FP8 block, or dequantised to "
        f"bf16). {scope} This config differs: {listed}."
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


def load_resident_model_config(
    model_id: str, *, revision: str | None = None, **kwargs: Any
):
    """The RESIDENT (non-streamed) route's config loader (#2104), the twin of
    :func:`load_stream_model_config`.

    ``AutoConfig``'s config for every model it loads; when it refuses (a ``ValueError``: K2
    is "custom code" to it), the raw ``config.json`` is read — no remote code is executed —
    and a ``model_type: kimi_k2`` config goes through :func:`kimi_k2_guard_failures` into
    :func:`kimi_k2_native_config_dict` and ``DeepseekV3Config``. The checkpoint itself is
    never modified: the rewrite is in-memory only. Any other model_type re-raises
    ``AutoConfig``'s own error, and a refusal names every field the guard rejects, worded
    for the resident route rather than the streamed one.
    """
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model_id, revision=revision, **kwargs)
    except ValueError:
        read = _raw_config_dict(model_id, dict(kwargs, revision=revision))
        if read is None or read[0].get("model_type") != KIMI_K2_MODEL_TYPE:
            raise
    config_dict, unused = read
    failures = kimi_k2_guard_failures(config_dict, route="resident")
    if failures:
        raise _refusal(failures, route="resident")

    from transformers import DeepseekV3Config

    return DeepseekV3Config.from_dict(kimi_k2_native_config_dict(config_dict), **unused)


#: Tensor key suffixes the resident load ignores entirely: the published K2 checkpoint
#: carries 61 stale per-layer rotary buffers (BF16, shape [7]) that neither K2's own code
#: (a non-persistent buffer) nor the native class (no per-layer rotary module) reads.
_RESIDENT_IGNORED_SUFFIXES = (".rotary_emb.inv_freq",)


def _local_kimi_k2_dir(base: str) -> bool:
    """A local directory whose ``config.json`` declares ``model_type: kimi_k2`` (JSON only)."""
    if not os.path.isdir(os.path.realpath(os.path.expanduser(base))):
        return False
    read = _raw_config_dict(base, {})
    return read is not None and read[0].get("model_type") == KIMI_K2_MODEL_TYPE


def _resolve_resident_device(device_map: Any, torch: Any):
    """The single device an accelerate-style ``device_map`` asks the whole model to live on.

    ``None`` loads on CPU; ``"auto"`` is accelerate's "one device" plan, resolved to the
    current CUDA device when one is available and to CPU otherwise; a plain device string
    or accelerate's whole-model entry (a single-entry dict ``{"": <rank|device>}``,
    what a per-rank map reduces to) names the target. Multi-device maps
    (``"balanced"`` and friends, dicts with more than the ``""`` entry) are refused: v1
    moves one whole model to one device, never shards it.

    An explicit CUDA target (``"cuda:N"``, ``{"": N}``) is validated here, before a single
    weight is read: the host must have the device, else a ``ValueError`` names the
    request and refuses — not a raw ``RuntimeError`` from ``model.to`` after the whole
    checkpoint is already in RAM.
    """
    device = _resident_device_target(device_map, torch)
    if device.type == "cuda":
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if count == 0 or (device.index or 0) >= count:
            raise ValueError(
                f"resident Kimi K2 route: device_map asks for {device}, but this host "
                f"serves {count} CUDA device(s) "
                f"(torch.cuda.is_available()={torch.cuda.is_available()}); refusing "
                "before loading the checkpoint. Pass a device this host serves, 'auto', "
                "or leave device_map unset (loads on CPU)."
            )
    return device


def _resident_device_target(device_map: Any, torch: Any):
    """The raw device a ``device_map`` names, with multi-device plans refused. ``"auto"``
    resolves to the current CUDA device when available, else CPU."""
    if device_map is None:
        return torch.device("cpu")
    if isinstance(device_map, str):
        if device_map == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda", torch.cuda.current_device())
            return torch.device("cpu")
        if device_map in ("balanced", "balanced_low_0", "sequential"):
            raise NotImplementedError(
                f"resident Kimi K2 route: device_map {device_map!r} is a multi-device "
                "accelerate plan the v1 loader does not implement; pass 'auto', a plain "
                "device string (e.g. 'cuda:0'), or leave device_map unset (loads on CPU)."
            )
        return torch.device(device_map)
    if isinstance(device_map, dict) and len(device_map) == 1 and "" in device_map:
        target = device_map[""]
        if isinstance(target, bool):
            target = None  # refused below
        if isinstance(target, int):
            return torch.device("cuda", target)
        if isinstance(target, (str, torch.device)):
            return torch.device(target)
    raise NotImplementedError(
        f"resident Kimi K2 route: device_map {device_map!r} is a multi-device accelerate "
        "map the v1 loader does not resolve; pass 'auto', a plain device string "
        "(e.g. 'cuda:0'), or leave device_map unset (loads on CPU)."
    )


def load_resident_model(base: str, *, model_kwargs: dict, console: Any = None):
    """The RESIDENT (non-streamed) route's model loader (#2104): a guarded Kimi K2
    checkpoint loaded natively into ``DeepseekV3ForCausalLM``, FP8-block weights
    dequantised per-tensor on load.

    Every base that is not Kimi K2 (a hub id :func:`~soup_cli.utils.kimi.is_resident_kimi_k2_base`
    admits, or a local directory whose ``config.json`` declares ``model_type: kimi_k2``)
    passes straight through to ``AutoModelForCausalLM.from_pretrained(base,
    **model_kwargs)``, so call sites may call this unconditionally.

    For a K2 base the config goes through :func:`load_resident_model_config` (the guard
    refuses Thinking/0905/bad configs, naming every failing field), the safetensors are
    read tensor by tensor — each FP8 block weight is dequantised to ``dtype`` against its
    float32 ``_scale_inv`` exactly as the streamed sharder does
    (:func:`~soup_cli.utils.fp8_block.dequantize_fp8_block`), raw FP8 bytes released
    before the next tensor — and the state dict is assigned onto a meta skeleton built
    under ``init_empty_weights``.

    Honoured ``model_kwargs``: ``dtype`` (default ``torch.bfloat16``), ``revision``, and
    ``attn_implementation`` (set on the config). ``trust_remote_code`` is accepted and
    ignored: this route never runs a repository's code. ``quantization_config`` is
    refused — the resident route loads bf16 only; train on a dequantised copy or use the
    streamed route for 4-bit. A caller-supplied ``config`` (e.g. the long-context rope
    override) is refused too: per-config rope overrides are not supported on the resident
    K2 route yet (follow-up), and a foreign config would bypass the guard — drop the
    kwarg or use the streamed route. ``device_map``: v1 loads on CPU and then moves to
    one device — ``"auto"`` resolves to the current CUDA device when one is available,
    else CPU; a plain device string (``"cpu"``, ``"cuda:0"``, ...) or accelerate's
    whole-model entry (a single-entry dict like ``{"": 0}``) names it directly; any
    multi-device accelerate map is refused rather than guessed.

    Parameters load at ``dtype``; float buffers keep their stored dtype (the checkpoint
    and ``from_pretrained`` keep ``e_score_correction_bias`` float32), and integer
    buffers keep their exact dtype.

    A hub id's weights resolve through ``huggingface_hub.snapshot_download`` with the
    safetensors/json allow-pattern — normal cache behaviour, same as transformers would
    use; on first real use that is the user's explicit request to train K2 and may fetch
    the full multi-TB repository.

    The model is returned on the requested device in the requested dtype; train/eval mode
    is the caller's to set. No layer streaming is involved on this route: nothing here
    reads or builds a streamed shard cache.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.kimi import is_resident_kimi_k2_base

    if not is_resident_kimi_k2_base(base) and not _local_kimi_k2_dir(base):
        return AutoModelForCausalLM.from_pretrained(base, **model_kwargs)

    model_kwargs = dict(model_kwargs)
    revision = model_kwargs.pop("revision", None)
    if model_kwargs.pop("quantization_config", None) is not None:
        raise ValueError(
            "resident Kimi K2 route: model_kwargs['quantization_config'] is set, but the "
            "resident route loads bf16 only. Remove it and load the FP8-block checkpoint "
            "dequantised (this loader does that), train on a dequantised bf16 copy, or "
            "use the streamed route for 4-bit training."
        )
    dtype = model_kwargs.pop("dtype", torch.bfloat16)
    device_map = model_kwargs.pop("device_map", None)
    model_kwargs.pop("trust_remote_code", None)  # never used: no remote code runs here
    attn_implementation = model_kwargs.pop("attn_implementation", None)
    if model_kwargs.pop("config", None) is not None:
        # Ruling D17: a caller-supplied config would bypass the guard, and with
        # trust_remote_code resolved it would even be code AutoConfig already ran.
        raise UnsupportedKimiK2Error(
            "resident Kimi K2 route: model_kwargs['config'] (a pre-built config, e.g. the "
            "long-context rope override) is not supported yet — the guarded published K2 "
            "config is the only admitted one on this route, and a foreign config would "
            "bypass that guard. Remove the config kwarg (drop the rope override), or use "
            "the streamed route."
        )
    if model_kwargs:
        raise ValueError(
            "resident Kimi K2 route: unsupported model_kwargs "
            f"{sorted(model_kwargs)}; supported: dtype, revision, "
            "attn_implementation, device_map, trust_remote_code."
        )

    device = _resolve_resident_device(device_map, torch)

    config = load_resident_model_config(base, revision=revision)

    if os.path.isdir(os.path.realpath(os.path.expanduser(base))):
        weights_dir = os.path.realpath(os.path.expanduser(base))
    else:
        from huggingface_hub import snapshot_download

        weights_dir = snapshot_download(
            repo_id=base, revision=revision, allow_patterns=["*.safetensors", "*.json"],
        )

    from soup_cli.utils.fp8_block import (
        SUPPORTED_FP8_FORMAT,
        FP8BlockPlan,
        UnsupportedFP8Error,
        dequantize_fp8_block,
        plan_fp8_block,
        quantization_config_of,
    )
    from soup_cli.utils.layer_shard import (
        _canonical_stream_key,
        _discover_safetensors,
        _read_source_tensors,
    )
    from soup_cli.utils.safetensors_reader import read_header, read_into

    shard_paths = _discover_safetensors(weights_dir)
    tensors = _read_source_tensors(shard_paths)
    try:
        plan = plan_fp8_block(quantization_config_of(config), tensors, arch="deepseek_v3")
    except UnsupportedFP8Error as exc:
        # plan_fp8_block's tail names the streamed route; this is the resident one.
        message = str(exc)
        if message.endswith(SUPPORTED_FP8_FORMAT):
            message = message[: -len(SUPPORTED_FP8_FORMAT)] + _FP8_RESIDENT_SUPPORT
        raise ValueError(message) from exc

    _torch_dtypes = {
        "F8_E4M3": torch.float8_e4m3fn,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }

    # Every tensor's (file, byte range), canonical key, so an FP8 weight's scale is found
    # whether or not it sits in the same shard.
    entries = {}
    for path in shard_paths:
        for key, entry in read_header(path).items():
            entries[_canonical_stream_key(key)] = (path, entry)

    from accelerate import init_empty_weights
    from transformers import DeepseekV3ForCausalLM

    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    with init_empty_weights():
        model = DeepseekV3ForCausalLM(config)
    model.tie_weights()
    # Only the model's own buffers keep their stored dtype (e_score_correction_bias is
    # float32 in the checkpoint and in from_pretrained's own load); everything else —
    # parameters and per-expert tensors awaiting the fuse — takes the requested dtype.
    buffer_keys = {name for name, _buffer in model.named_buffers()}

    state = {}
    handles = {}
    dequantised = 0
    try:
        for key in sorted(entries):
            path, entry = entries[key]
            raw_dtype, shape = tensors[key]
            if plan is not None and key in plan.weights:
                scale_key = FP8BlockPlan.scale_key(key)
                scale_path, scale_entry = entries[scale_key]
                raw_weight = torch.empty(shape, dtype=torch.float8_e4m3fn)
                read_into(_shard_handle(handles, path), entry, raw_weight)
                raw_scale = torch.empty(tensors[scale_key][1], dtype=torch.float32)
                read_into(_shard_handle(handles, scale_path), scale_entry, raw_scale)
                state[key] = dequantize_fp8_block(
                    raw_weight, raw_scale, plan.block, name=key, dtype=dtype
                )
                dequantised += 1
                del raw_weight, raw_scale
            elif plan is not None and key in plan.scales:
                continue  # consumed by its weight's dequantisation, never a parameter
            elif key.endswith(_RESIDENT_IGNORED_SUFFIXES):
                continue  # stale per-layer rotary buffer; the native class has none
            else:
                stored_dtype = _torch_dtypes[raw_dtype]
                tensor = torch.empty(shape, dtype=stored_dtype)
                read_into(_shard_handle(handles, path), entry, tensor)
                if key in buffer_keys:
                    # A float buffer keeps its stored dtype; only parameters (and the
                    # per-expert tensors the fuse turns into parameters) take ``dtype``.
                    state[key] = tensor
                else:
                    state[key] = tensor if stored_dtype == dtype else tensor.to(dtype)
    finally:
        for handle in handles.values():
            handle.close()

    _fuse_per_expert_state(state, config)

    model_state = model.state_dict()
    expected = set(model_state)
    if getattr(config, "tie_word_embeddings", False):
        # K2 publishes tie_word_embeddings=true: no separate lm_head tensor in the shards.
        expected.discard("lm_head.weight")
    unexpected = sorted(set(state) - expected)
    if unexpected:
        raise ValueError(
            "checkpoint holds tensors the native DeepseekV3 model has no parameters for: "
            + ", ".join(unexpected[:_MAX_LISTED])
            + (
                f" and {len(unexpected) - _MAX_LISTED} more"
                if len(unexpected) > _MAX_LISTED
                else ""
            )
        )
    missing = sorted(expected - set(state))
    if missing:
        raise ValueError(
            "checkpoint is missing tensors the native DeepseekV3 model needs: "
            + ", ".join(missing[:_MAX_LISTED])
            + (f" and {len(missing) - _MAX_LISTED} more" if len(missing) > _MAX_LISTED else "")
        )
    mismatched = sorted(
        key for key in state
        if tuple(int(dim) for dim in model_state[key].shape)
        != tuple(int(dim) for dim in state[key].shape)
    )
    if mismatched:
        raise ValueError(
            "checkpoint tensors whose shapes differ from the native model's parameters: "
            + ", ".join(mismatched[:_MAX_LISTED])
        )
    model.load_state_dict(state, assign=True, strict=False)
    if getattr(config, "tie_word_embeddings", False):
        # assign=True replaced embed_tokens; re-point the tied head at the loaded tensor.
        model.lm_head.weight = model.get_input_embeddings().weight

    if console is not None:
        detail = f", {dequantised} FP8 block weights dequantised" if dequantised else ""
        console.print(
            f"[green]Kimi K2 resident load:[/] {len(state)} tensors{detail} "
            f"({dtype}, device {device})"
        )
    return model.to(device)


def _shard_handle(handles: dict, path: str):
    """The cached read handle for one shard file, opened on first use."""
    handle = handles.get(path)
    if handle is None:
        handle = open(path, "rb")
        handles[path] = handle
    return handle


#: Per-expert checkpoint spelling (the published K2 index, and what ``save_pretrained``
#: writes for the fused native model) -> the fused 3-D parameters the native model holds
#: in transformers 5, the layout :func:`transformers.AutoModelForCausalLM.from_pretrained`
#: itself produces via its qwen2_moe conversion (MergeModulelist + Concatenate).
_EXPERT_KEY_RE = re.compile(r"^(.*)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def _fuse_per_expert_state(state: dict, config: Any) -> None:
    """Merge ``mlp.experts.<i>.*.weight`` tensors into the fused ``gate_up_proj`` /
    ``down_proj`` parameters, in place, one preallocated copy per layer.

    Per-expert tensors stay per-expert on disk and in the read loop; this is the single
    materialisation the fused native module tree needs. Any expert index missing, or any
    of the three projections missing for one expert, is refused by name.
    """
    import torch

    grouped = {}
    for key in list(state):
        match = _EXPERT_KEY_RE.match(key)
        if match is not None:
            grouped.setdefault(match.group(1), {}).setdefault(
                int(match.group(2)), {}
            )[match.group(3)] = state.pop(key)
    if not grouped:
        return
    n_routed = int(config.n_routed_experts)
    for prefix, experts in sorted(grouped.items()):
        indices = sorted(experts)
        incomplete = [
            index
            for index in indices
            if set(experts[index]) != {"gate_proj", "up_proj", "down_proj"}
        ]
        if incomplete or indices != list(range(n_routed)):
            raise ValueError(
                f"checkpoint layer {prefix} holds per-expert tensors for experts "
                f"{indices}, not every index 0..{n_routed - 1} with gate_proj, up_proj "
                f"and down_proj (incomplete: {incomplete}). The checkpoint is damaged or "
                "not a complete Kimi K2 copy."
            )
        for projection in ("gate_proj", "up_proj", "down_proj"):
            reference_shape = tuple(int(dim) for dim in experts[0][projection].shape)
            for index in range(1, n_routed):
                shape = tuple(int(dim) for dim in experts[index][projection].shape)
                if shape != reference_shape:
                    raise ValueError(
                        f"checkpoint layer {prefix} holds {projection} tensors of "
                        f"differing shapes: expert 0 has {list(reference_shape)}, but "
                        f"{prefix}.mlp.experts.{index}.{projection}.weight has "
                        f"{list(shape)}. The checkpoint is damaged."
                    )
        gate_shape = experts[0]["gate_proj"].shape
        down_shape = experts[0]["down_proj"].shape
        gate_up = torch.empty(
            (n_routed, 2 * int(gate_shape[0]), int(gate_shape[1])),
            dtype=experts[0]["gate_proj"].dtype,
        )
        down = torch.empty(
            (n_routed, int(down_shape[0]), int(down_shape[1])),
            dtype=experts[0]["down_proj"].dtype,
        )
        for index in range(n_routed):
            parts = experts[index]
            rows = int(parts["gate_proj"].shape[0])
            gate_up[index, :rows].copy_(parts["gate_proj"])
            gate_up[index, rows:].copy_(parts["up_proj"])
            down[index].copy_(parts["down_proj"])
            del parts["gate_proj"], parts["up_proj"], parts["down_proj"]
        state[prefix + ".mlp.experts.gate_up_proj"] = gate_up
        state[prefix + ".mlp.experts.down_proj"] = down


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
