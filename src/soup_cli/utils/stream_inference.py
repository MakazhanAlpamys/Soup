"""Forward-only inference over the layer-streaming runtime.

A model whose weights do not fit in VRAM can still be RUN: the decoder layers are streamed from
RAM or disk through the same buffer pool, prefetch and NF4 rebuild the trainer uses. Every
forward reads every layer once, so the read is paid per forward and amortised over all the tokens
of that forward: this is for prefill, scoring and batch forward (eval, judge, reward scoring),
not for token-by-token generation. A decode step would re-read the whole model per token, which
is why per-token generation stays a permanent streaming exclusion
(``utils/config_bounds.py``). There is no KV cache either: the model is built with
``use_cache=False`` and every call is a full forward over the sequences given.

Typical use::

    from soup_cli.utils.stream_inference import build_streamed_inference_model

    with build_streamed_inference_model(model_id="Qwen/Qwen2.5-7B", shard_dir=shards) as stream:
        logits = stream.logits(input_ids, attention_mask)          # (batch, seq, vocab)
        log_probs = stream.score_sequences(input_ids, attention_mask)  # (batch, seq - 1)

``shard_dir`` is a layer-stream shard cache (the one ``soup train`` builds with
``training.stream_layers``, or ``utils.layer_shard.shard_checkpoint``). Leaving the ``with``
block closes the runtime exactly the way the trainer does (``StreamRuntime.close``): the prefetch
hook is detached and, on the disk tier, the reader thread and its staging buffers are released.

A LoRA adapter trained on a streamed base (the directory ``save_pretrained`` wrote) is loaded
back with ``adapter=``: the model is then the streamed base with that adapter placed on it, still
forward-only (every parameter frozen, no checkpointing, ``use_cache=False``), and its outputs are
the trained model's outputs. See :func:`build_streamed_inference_model`.

No heavy module is imported at module level: torch, transformers, safetensors and peft (peft
only when an ``adapter`` is given) are reached from inside the functions.
"""

from __future__ import annotations

import json
import os
import re
import stat
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

from soup_cli.utils.config_bounds import (
    DEFAULT_STREAM_READ_AHEAD,
    MAX_STREAM_BUFFERS,
    MIN_STREAM_BUFFERS,
)

_ADAPTER_CONFIG = "adapter_config.json"
_ADAPTER_WEIGHTS = "adapter_model.safetensors"
_ADAPTER_PICKLE = "adapter_model.bin"
_MAX_ADAPTER_CONFIG_BYTES = 256 * 1024
#: What PEFT saves beside the LoRA of an adapted boundary module (``lm_head``, ``embed_tokens``):
#: a copy of that module's frozen base weight, under ``<module>.base_layer.weight``.
_BASE_COPY_SUFFIX = ".base_layer.weight"
#: The adapter PEFT names its first adapter (and ``save_pretrained`` strips from the saved keys).
_ADAPTER_NAME = "default"
_ADAPTER_NAME_SEGMENT = re.compile(r"\.default(?=\.|$)")
#: ``init_lora_weights`` values whose adapter is valid on an unrewritten base.
_PLAIN_LORA_INITS = (None, True, False, "gaussian")

#: ``adapter_config.json`` fields a streamed base cannot honour: ``(field, is_set, reason)``.
#: Each reason says why, and every one contains "streamed" so the refusal names the runtime.
_UNSUPPORTED_ADAPTER_FIELDS = (
    (
        "modules_to_save",
        bool,
        "a streamed base keeps its embedding and output head frozen in shared slots, so a "
        "trained copy of a module has nowhere to live; adapt it with a LoRA target instead",
    ),
    (
        "use_dora",
        bool,
        "DoRA reads the whole base weight when the adapter is built, and a streamed base "
        "holds none of it resident",
    ),
    (
        "target_parameters",
        bool,
        "an adapter on raw expert parameters is not verified on a streamed base",
    ),
    (
        "trainable_token_indices",
        bool,
        "trained token rows would have to live in the frozen streamed embedding",
    ),
    (
        "layer_replication",
        bool,
        "it duplicates decoder layers, and the streamed shards hold one per index",
    ),
    (
        "bias",
        lambda value: value not in (None, "none"),
        "the saved adapter would carry base-model bias tensors that the streamed base holds "
        "in its shards",
    ),
    (
        "init_lora_weights",
        lambda value: value not in _PLAIN_LORA_INITS,
        "an init that rewrites the base weights (PiSSA, OLoRA, LoftQ, ...) is only valid on "
        "that rewritten base, and a streamed base is the original checkpoint",
    ),
)


def _model_device(model: Any) -> Any:
    """The device the resident (non-streamed) weights live on."""
    for param in model.parameters():
        if not param.is_meta:
            return param.device
    for buffer in model.buffers():
        if not buffer.is_meta:
            return buffer.device
    raise RuntimeError("the streamed model has no materialised weight to place inputs next to")


def _vocab_size(model: Any) -> Optional[int]:
    config = getattr(model, "config", None)
    size = getattr(config, "vocab_size", None)
    return int(size) if isinstance(size, int) and not isinstance(size, bool) else None


def _check_batch(
    model: Any, input_ids: Any, attention_mask: Any, *, min_tokens: int = 1
) -> tuple[Any, Any]:
    """Validate and place one batch. Returns ``(input_ids, attention_mask)`` on the device."""
    import torch

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(
            f"input_ids must be a torch.Tensor of token ids, got {type(input_ids).__name__}"
        )
    if input_ids.dim() != 2:
        raise ValueError(
            "input_ids must be a 2-D (batch, seq_len) tensor, "
            f"got shape {tuple(input_ids.shape)}"
        )
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError(
            f"input_ids is empty (shape {tuple(input_ids.shape)}); "
            "give at least one sequence with at least one token"
        )
    if input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex():
        raise TypeError(
            f"input_ids must hold integer token ids, got dtype {str(input_ids.dtype)[6:]}"
        )
    if input_ids.shape[1] < min_tokens:
        raise ValueError(
            f"this call needs at least {min_tokens} tokens per sequence, "
            f"got {input_ids.shape[1]}"
        )
    device = _model_device(model)
    input_ids = input_ids.to(device=device, dtype=torch.long)
    vocab = _vocab_size(model)
    if vocab is not None:
        low, high = int(input_ids.min()), int(input_ids.max())
        if low < 0 or high >= vocab:
            bad = low if low < 0 else high
            raise ValueError(
                f"input_ids holds token id {bad} outside the vocabulary range [0, {vocab})"
            )
    if attention_mask is None:
        return input_ids, None
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError(
            "attention_mask must be a torch.Tensor of 0/1, "
            f"got {type(attention_mask).__name__}"
        )
    if tuple(attention_mask.shape) != tuple(input_ids.shape):
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} does not match "
            f"input_ids shape {tuple(input_ids.shape)}"
        )
    mask = attention_mask.to(device=device)
    if bool(((mask != 0) & (mask != 1)).any()):
        raise ValueError("attention_mask must hold only 0 and 1")
    mask = mask.to(dtype=torch.long)
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("attention_mask has a row with no real token (all zeros)")
    return input_ids, mask


def streamed_forward_logits(model: Any, input_ids: Any, attention_mask: Any = None) -> Any:
    """Logits ``(batch, seq_len, vocab)`` of one full forward, under ``torch.inference_mode()``.

    ``model`` is the model of :func:`build_streamed_inference_model`. The result is an
    inference tensor: read it, copy it, reduce it, but do not feed it to autograd. It comes back
    in the model's compute dtype and is ``batch * seq_len * vocab`` elements, so budget the
    batch for the vocabulary. Padding follows the Hugging Face forward: pass the mask, and use
    left or right padding as the model expects (position ids come from the position in the
    padded row, as in a plain ``model(input_ids, attention_mask)`` call). ``use_cache`` is False
    (no KV cache).
    """
    import torch

    input_ids, mask = _check_batch(model, input_ids, attention_mask)
    with torch.inference_mode():
        return model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits


def score_sequences(model: Any, input_ids: Any, attention_mask: Any = None) -> Any:
    """Per-token log-probabilities of the given sequences, ``(batch, seq_len - 1)`` float32.

    Entry ``[row, j]`` is ``log p(input_ids[row, j + 1] | input_ids[row, : j + 1])``: the first
    token has no context and is not scored, so a sequence needs at least 2 tokens. Where
    ``attention_mask[row, j + 1]`` is 0 (a padded target) the entry is 0.0; sum a row for the
    sequence log-likelihood. The softmax is taken in float32 one row at a time, so the extra
    memory is one ``(seq_len, vocab)`` float32 matrix. Runs under ``torch.inference_mode()`` and
    returns an inference tensor (see :func:`streamed_forward_logits`).
    """
    import torch

    input_ids, mask = _check_batch(model, input_ids, attention_mask, min_tokens=2)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits
        scores = torch.zeros(
            input_ids.shape[0], input_ids.shape[1] - 1, dtype=torch.float32, device=logits.device
        )
        for row in range(input_ids.shape[0]):
            log_probs = torch.log_softmax(logits[row, :-1].float(), dim=-1)
            scores[row] = log_probs.gather(-1, input_ids[row, 1:].unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask[:, 1:] == 0, 0.0)
    return scores


class StreamedInferenceModel:
    """A streamed model plus its runtime, with the forward helpers and a context manager.

    Build one with :func:`build_streamed_inference_model`. ``model`` is the plain Hugging Face
    model (eval mode, every parameter frozen, no PEFT wrapper), or, when an adapter was loaded,
    the ``PeftModel`` that wraps it (still eval mode, still every parameter frozen);
    ``adapter`` is the absolute path of that adapter directory, or ``None``. ``runtime`` is the
    ``StreamRuntime`` that feeds the model. ``close()`` (or leaving the ``with`` block)
    releases the runtime and is idempotent; a closed model refuses to run.
    """

    def __init__(self, model: Any, runtime: Any, adapter: Optional[str] = None) -> None:
        self.model = model
        self.runtime = runtime
        self.adapter = adapter
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "this streamed inference model is closed; build a new one with "
                "build_streamed_inference_model()"
            )

    def logits(self, input_ids: Any, attention_mask: Any = None) -> Any:
        """See :func:`streamed_forward_logits`."""
        self._require_open()
        return streamed_forward_logits(self.model, input_ids, attention_mask)

    def score_sequences(self, input_ids: Any, attention_mask: Any = None) -> Any:
        """See :func:`score_sequences`."""
        self._require_open()
        return score_sequences(self.model, input_ids, attention_mask)

    def stats(self) -> Dict[str, Any]:
        """``StreamRuntime.stats()``: layer loads, buffer bytes, tier, store bytes."""
        self._require_open()
        return self.runtime.stats()

    def close(self) -> None:
        """Release the weight source and detach the prefetch hook (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self.runtime.close()

    def __enter__(self) -> "StreamedInferenceModel":
        self._require_open()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


# ==========================================================================
# loading a saved LoRA adapter
# ==========================================================================
class _AdapterSource(NamedTuple):
    """An adapter directory that passed every check that needs no model."""

    directory: str
    weights_path: str
    lora_config: Any
    header: Dict[str, Tuple[Tuple[int, ...], str]]


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _require_plain_file(path: str, name: str, directory: str) -> None:
    """``path`` must be a regular file: no symlink, and on Windows no reparse point."""
    from soup_cli.utils.terminal import for_terminal

    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"no {name} in {for_terminal(directory)}") from None
    except OSError as exc:
        raise ValueError(f"{name} is unreadable: {type(exc).__name__}") from exc
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(info.st_mode) or (
        os.name == "nt" and getattr(info, "st_file_attributes", 0) & reparse
    ):
        raise ValueError(f"{name} must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} is not a regular file")


def _resolve_adapter_dir(adapter: Any) -> str:
    """The absolute adapter directory: under cwd, not a link, and existing."""
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink
    from soup_cli.utils.terminal import for_terminal

    if not isinstance(adapter, (str, os.PathLike)):
        raise TypeError(
            f"adapter must be a path (str or os.PathLike), got {type(adapter).__name__}"
        )
    path = os.fspath(adapter)
    if not isinstance(path, str):
        raise TypeError("adapter must be a text path, not bytes")
    enforce_under_cwd_and_no_symlink(path, "adapter")
    shown = for_terminal(path)
    if not os.path.lexists(path):
        raise FileNotFoundError(f"adapter directory not found: {shown}")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"adapter {shown} is not a directory")
    return os.path.abspath(path)


def _read_adapter_config(directory: str) -> Dict[str, Any]:
    path = os.path.join(directory, _ADAPTER_CONFIG)
    _require_plain_file(path, _ADAPTER_CONFIG, directory)
    if os.path.getsize(path) > _MAX_ADAPTER_CONFIG_BYTES:
        raise ValueError(
            f"{_ADAPTER_CONFIG} is larger than {_MAX_ADAPTER_CONFIG_BYTES} bytes; "
            "refusing to read it"
        )
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        from soup_cli.utils.terminal import for_terminal

        raise ValueError(
            f"{_ADAPTER_CONFIG} is not valid JSON: {for_terminal(str(exc))}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{_ADAPTER_CONFIG} must hold a JSON object")
    return payload


def _check_adapter_kind(config: Dict[str, Any]) -> None:
    """Only a LoRA adapter, and only one whose every feature a streamed base can hold."""
    from soup_cli.utils.terminal import for_terminal

    peft_type = config.get("peft_type")
    if not (isinstance(peft_type, str) and peft_type.upper() == "LORA"):
        raise ValueError(
            f"{_ADAPTER_CONFIG} is not a LoRA adapter (peft_type={for_terminal(repr(peft_type))}); "
            "only LoRA adapters can be loaded into a streamed base"
        )
    for field, is_set, reason in _UNSUPPORTED_ADAPTER_FIELDS:
        value = config.get(field)
        if is_set(value):
            raise ValueError(
                f"{_ADAPTER_CONFIG} sets {field}={for_terminal(repr(value))}, which a streamed "
                f"base cannot load: {reason}"
            )


#: ``LoraConfig``'s own defaults for the two numbers a config may leave out.
_PEFT_DEFAULT_RANK = 8
_PEFT_DEFAULT_ALPHA = 8


def _shown(value: Any) -> str:
    """A value from the file, quoted, cut short and made safe for a terminal."""
    from soup_cli.utils.terminal import for_terminal

    return for_terminal(repr(value)[:80])


def _check_module_names(field: str, value: Any, *, required: bool) -> None:
    """``target_modules`` / ``exclude_modules``: peft reads a string as a regex, a list as names."""
    from soup_cli.config import schema
    from soup_cli.utils.safe_regex import check_config_regex
    from soup_cli.utils.terminal import for_terminal

    max_chars = schema._MAX_LORA_PATTERN_KEY_LEN
    max_names = schema._MAX_LORA_RANK_PATTERN_KEYS
    if value is None or value == "" or value == []:
        if required:
            raise ValueError(f"{_ADAPTER_CONFIG} {field} names no module to adapt")
        return
    if isinstance(value, str):
        if len(value) > max_chars:
            raise ValueError(
                f"{_ADAPTER_CONFIG} {field} is {len(value)} characters, over the "
                f"{max_chars}-character cap"
            )
        try:
            re.compile(value)
            check_config_regex(value, field)
        except (re.error, ValueError) as exc:
            raise ValueError(
                f"{_ADAPTER_CONFIG} {field} is not a usable pattern: {for_terminal(str(exc))}"
            ) from None
        return
    if not isinstance(value, list) or not all(isinstance(name, str) and name for name in value):
        raise ValueError(
            f"{_ADAPTER_CONFIG} {field} must be a string or a list of non-empty strings, "
            f"got {_shown(value)}"
        )
    if len(value) > max_names:
        raise ValueError(
            f"{_ADAPTER_CONFIG} {field} lists {len(value)} names, over the {max_names} cap"
        )
    if any(len(name) > max_chars for name in value):
        raise ValueError(
            f"{_ADAPTER_CONFIG} {field} has a name over the {max_chars}-character cap"
        )


def _check_adapter_limits(config: Dict[str, Any]) -> None:
    """Bound every config value peft acts on, before anything is built or allocated.

    peft allocates an adapter from ``r`` / ``rank_pattern`` the moment the model is wrapped, and
    matches ``rank_pattern`` keys and a string ``target_modules`` as regular expressions, so
    these come from a file nobody vouches for and get the training schema's own caps
    (``config/schema.py``: ranks to 1024, 256 keys of up to 512 characters, and the same
    ``check_config_regex`` complexity test).
    """
    import math

    from pydantic import ValidationError

    from soup_cli.config import schema
    from soup_cli.utils.terminal import for_terminal

    max_rank = schema._MAX_LORA_RANK_PATTERN_VALUE
    rank = config.get("r", _PEFT_DEFAULT_RANK)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= max_rank:
        raise ValueError(
            f"{_ADAPTER_CONFIG} r must be an integer from 1 to {max_rank}, got {_shown(rank)}"
        )
    alpha = config.get("lora_alpha", _PEFT_DEFAULT_ALPHA)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError(
            f"{_ADAPTER_CONFIG} lora_alpha must be a positive finite number, got {_shown(alpha)}"
        )
    try:
        schema.LoraConfig(
            rank_pattern=config.get("rank_pattern"), alpha_pattern=config.get("alpha_pattern")
        )
    except ValidationError as exc:
        detail = "; ".join(str(error["msg"]) for error in exc.errors())
        raise ValueError(
            f"{_ADAPTER_CONFIG} has an unusable rank_pattern or alpha_pattern: "
            f"{for_terminal(detail[:300])}"
        ) from None
    _check_module_names("target_modules", config.get("target_modules"), required=True)
    _check_module_names("exclude_modules", config.get("exclude_modules"), required=False)


def _check_adapter_ranks(
    config: Dict[str, Any], header: Dict[str, Tuple[Tuple[int, ...], str]]
) -> None:
    """Every LoRA A tensor in the file must have a rank the config can have produced.

    A module's rank is ``r`` or, when a ``rank_pattern`` key matches it, that key's value. The
    file's header holds the real ranks, so a config and a weights file that are not from the
    same adapter are refused here, with nothing built (``r`` itself may go unused when the
    patterns cover every module, so it is not required to appear in the file).
    """
    from soup_cli.utils.terminal import for_terminal

    ranks = {
        key: shape[0]
        for key, (shape, _dtype) in header.items()
        if (".lora_A." in key or ".lora_embedding_A" in key) and shape
    }
    if not ranks:
        raise ValueError(f"{_ADAPTER_WEIGHTS} holds no LoRA A tensor, so it is not a LoRA adapter")
    allowed = {config.get("r", _PEFT_DEFAULT_RANK), *(config.get("rank_pattern") or {}).values()}
    for key in sorted(ranks):
        if ranks[key] not in allowed:
            raise ValueError(
                f"adapter tensor {for_terminal(key)} has rank {ranks[key]}, but "
                f"{_ADAPTER_CONFIG} allows only rank {sorted(allowed)}: the weights and the "
                "config are not from the same adapter"
            )


def _same_base(recorded: str, model_id: str) -> bool:
    """Whether the adapter's recorded base is the base being streamed.

    Two existing directories are the same base only if they are the same directory. Otherwise
    the final path segments are compared, so a Hub id and a local copy of that model
    (``org/Model`` and ``D:/models/Model``) match. The weights themselves are checked later,
    by shape, so this is the cheap early check and not the safety net.

    ``recorded`` comes out of ``adapter_config.json``, so it is never handed to the filesystem
    when it names a network share or a device (a UNC path, a device-namespace prefix, ``NUL``):
    on Windows a stat of such a path opens a connection to the host. It is then compared as
    text only, like a Hub id.
    """
    from soup_cli.utils.paths import is_network_or_device_path

    if (
        os.path.isdir(model_id)
        and not is_network_or_device_path(recorded)
        and os.path.isdir(recorded)
    ):
        return os.path.samefile(recorded, model_id)

    def last_segment(text: str) -> str:
        return text.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()

    return last_segment(recorded) == last_segment(model_id)


def _check_adapter_base(config: Dict[str, Any], model_id: str, allow_mismatch: bool) -> None:
    from soup_cli.utils.terminal import for_terminal

    recorded = config.get("base_model_name_or_path")
    if recorded is None or recorded == "":
        return
    if not isinstance(recorded, str):
        raise ValueError(
            f"{_ADAPTER_CONFIG} base_model_name_or_path must be text, "
            f"got {type(recorded).__name__}"
        )
    if allow_mismatch or _same_base(recorded, str(model_id)):
        return
    raise ValueError(
        f"the adapter was trained on base '{for_terminal(recorded)}' but this streamed base is "
        f"'{for_terminal(model_id)}'; pass allow_adapter_base_mismatch=True if they are the same "
        "checkpoint under another name (the tensor shapes are checked either way)"
    )


def _lora_config_of(directory: str) -> Any:
    from peft import LoraConfig

    from soup_cli.utils.terminal import for_terminal

    try:
        return LoraConfig.from_pretrained(directory)
    except Exception as exc:  # peft raises TypeError / ValueError / OSError for a bad config
        raise ValueError(
            f"{_ADAPTER_CONFIG} cannot be read as a LoRA config: {for_terminal(str(exc))}"
        ) from exc


def _read_adapter_header(path: str) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """``{key: (shape, dtype)}`` from the safetensors header alone: no tensor is read."""
    from safetensors import safe_open

    from soup_cli.utils.terminal import for_terminal

    header: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    try:
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                piece = handle.get_slice(key)
                header[key] = (tuple(piece.get_shape()), str(piece.get_dtype()))
    except Exception as exc:  # safetensors raises its own SafetensorError for a bad file
        raise ValueError(
            f"{_ADAPTER_WEIGHTS} cannot be read as safetensors: {for_terminal(str(exc))}"
        ) from exc
    for key, (_shape, dtype) in header.items():
        if not dtype.startswith(("F", "BF")):
            raise ValueError(
                f"{_ADAPTER_WEIGHTS} holds the non-floating tensor {for_terminal(key)} "
                f"({dtype}); a LoRA adapter is floating point"
            )
    return header


def _prepare_adapter(adapter: Any, model_id: str, allow_base_mismatch: bool) -> _AdapterSource:
    """Every check on an adapter directory that needs no model, before one is built."""
    directory = _resolve_adapter_dir(adapter)
    config = _read_adapter_config(directory)
    _check_adapter_kind(config)
    _check_adapter_limits(config)
    _check_adapter_base(config, model_id, allow_base_mismatch)
    weights_path = os.path.join(directory, _ADAPTER_WEIGHTS)
    try:
        _require_plain_file(weights_path, _ADAPTER_WEIGHTS, directory)
    except FileNotFoundError as exc:
        if os.path.lexists(os.path.join(directory, _ADAPTER_PICKLE)):
            raise FileNotFoundError(
                f"{exc} (found {_ADAPTER_PICKLE}: a pickle file is never loaded; "
                "re-save the adapter as safetensors)"
            ) from None
        raise
    header = _read_adapter_header(weights_path)
    _check_adapter_ranks(config, header)
    lora_config = _lora_config_of(directory)
    # The recorded name was checked above. Left in place, PEFT would warn that it renames it to
    # the streamed base's name; cleared, PEFT records the streamed base there, as in training.
    lora_config.base_model_name_or_path = None
    return _AdapterSource(directory, weights_path, lora_config, header)


def _adapter_slots(model: Any) -> Dict[str, Any]:
    """The model's LoRA parameters under the key ``save_pretrained`` writes for each."""
    return {
        _ADAPTER_NAME_SEGMENT.sub("", name): param
        for name, param in model.named_parameters()
        if "lora_" in name
    }


def _frozen_base_copies(header: Dict[str, Any], slots: Dict[str, Any]) -> frozenset:
    """Keys of the frozen base weight PEFT saved next to the LoRA of an adapted module."""
    adapted = {key.rsplit(".lora_", 1)[0] for key in slots}
    return frozenset(
        key
        for key in header
        if key.endswith(_BASE_COPY_SUFFIX) and key[: -len(_BASE_COPY_SUFFIX)] in adapted
    )


def _load_adapter_weights(
    model: Any, weights_path: str, header: Dict[str, Tuple[Tuple[int, ...], str]]
) -> None:
    """Place the saved adapter on the built model, or refuse by name.

    The files are compared with the model's own adapter slots first (partial adapter, foreign
    tensor, wrong shape), so a mismatch is named here and does not surface as a size error
    inside PEFT. The tensors are then loaded with ``set_peft_model_state_dict`` straight into
    the adapters that are already placed (``PeftModel.load_adapter`` would re-plan the device
    map and cannot move this runtime's pools), the result is checked, and finally every
    adapter tensor is compared with the file, because a load that lands nothing is only a
    ``UserWarning`` in PEFT (#1011).

    The frozen base weight PEFT saves beside a LoRA on ``lm_head`` / ``embed_tokens`` is
    shape-checked but neither read nor loaded: the shards are the base.
    """
    import peft
    import torch
    from safetensors import safe_open

    from soup_cli.utils.terminal import for_terminal

    slots = _adapter_slots(model)
    copies = _frozen_base_copies(header, slots)
    missing = sorted(set(slots) - set(header))
    unexpected = sorted(set(header) - set(slots) - copies)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(
                f"it is missing {_plural(len(missing), 'LoRA tensor')} "
                f"(first: {for_terminal(missing[0])})"
            )
        if unexpected:
            parts.append(
                f"it holds {_plural(len(unexpected), 'tensor')} this streamed base has no slot "
                f"for (first: {for_terminal(unexpected[0])})"
            )
        if any(".inner." in key for key in unexpected):
            parts.append(
                "a '.inner.' key is the streaming wrapper's own spelling (v0.72.0), which "
                "reloads as nothing: re-save the adapter"
            )
        raise ValueError(
            f"the adapter does not fit this streamed base ({'; '.join(parts)}); "
            "a partial adapter would reload as a silently different model"
        )
    expected = {key: tuple(param.shape) for key, param in slots.items()}
    live = dict(model.named_parameters())
    expected.update({key: tuple(live[key].shape) for key in copies if key in live})
    wrong = [
        (key, header[key][0], want)
        for key, want in sorted(expected.items())
        if header[key][0] != want
    ]
    if wrong:
        key, got, want = wrong[0]
        raise ValueError(
            f"adapter tensor {for_terminal(key)} has shape {got} but the streamed base expects "
            f"{want}: the adapter was trained on a different base model or with a different "
            f"LoRA rank ({_plural(len(wrong), 'tensor')} differ)"
        )
    with safe_open(weights_path, framework="pt") as handle:
        stored = {key: handle.get_tensor(key) for key in sorted(slots)}
    result = peft.set_peft_model_state_dict(model, stored, adapter_name=_ADAPTER_NAME)
    lost = [key for key in (getattr(result, "missing_keys", None) or []) if "lora_" in key]
    stray = list(getattr(result, "unexpected_keys", None) or [])
    if lost:
        raise RuntimeError(
            f"set_peft_model_state_dict reported {_plural(len(lost), 'missing LoRA tensor')} "
            f"(first: {for_terminal(lost[0])}); the adapter did not load"
        )
    if stray:
        raise RuntimeError(
            f"set_peft_model_state_dict reported {_plural(len(stray), 'unexpected key')} "
            f"(first: {for_terminal(stray[0])}); the adapter did not load"
        )
    slots = _adapter_slots(model)
    unplaced = [
        key
        for key, value in stored.items()
        if not torch.equal(slots[key].detach().cpu(), value.to(dtype=slots[key].dtype))
    ]
    if unplaced:
        raise RuntimeError(
            f"{len(unplaced)} of {len(stored)} adapter tensors did not land in the streamed "
            f"model (first: {for_terminal(unplaced[0])}); refusing to run a model whose adapter "
            "is not the one on disk"
        )


def build_streamed_inference_model(
    *,
    model_id: str,
    shard_dir: str,
    index: Any = None,
    device: str = "cuda",
    dtype: Optional[str] = None,
    buffers: int = 2,
    pin: Optional[bool] = None,
    require_pin: bool = False,
    tier: str = "ram",
    read_ahead: int = DEFAULT_STREAM_READ_AHEAD,
    weights_dir: Optional[str] = None,
    ngram_source: str = "disk",
    nf4_scale_cache: bool = False,
    trust_remote_code: bool = False,
    console: Any = None,
    adapter: Optional[Union[str, "os.PathLike[str]"]] = None,
    allow_adapter_base_mismatch: bool = False,
) -> StreamedInferenceModel:
    """Build a forward-only streamed model from an existing layer-stream shard cache.

    ``model_id`` is the model the cache was sharded from (a Hub id or a local directory: its
    ``config.json`` builds the meta skeleton; no weight is read from it). ``shard_dir`` is the
    cache; ``index`` defaults to the one stored in it. The quantisation (``none`` or NF4,
    double-quant or not) and, when ``dtype`` is omitted, the compute dtype come from the index,
    so they cannot disagree with the shards. ``pin`` defaults to page-locked host memory on a
    CUDA device and off elsewhere. ``tier`` is ``"ram"`` or ``"disk"``; ``buffers`` is the
    number of per-layer slots on the device (2 to 8; two lets the next layer load while the
    current one computes).

    Without ``adapter`` this is the inference counterpart of
    ``build_streamed_model(lora_config=None)``: no PEFT, no adapter, no trainable parameter, no
    gradient checkpointing, ``use_cache=False`` (there is no KV cache). Use the result in a
    ``with`` block, or call ``close()``.

    ``adapter`` is a directory ``PeftModel.save_pretrained`` wrote (``adapter_config.json`` plus
    ``adapter_model.safetensors``, as ``soup train`` leaves it), for a LoRA trained on this same
    streamed base. The streamed model is built with that adapter's LoRA config through the
    training path's own placement (so the adapters sit where the trainer put them), the saved
    tensors are loaded into them, and then everything is frozen: the result is forward-only
    like one built without an adapter, except that ``model`` is the ``PeftModel`` and its logits
    are those of the trained model. The path must stay under the current directory and neither
    it nor its files may be links. Refused, by name: a directory, config or weights file that is
    missing or unreadable; an adapter that is not LoRA, or that uses ``modules_to_save``, DoRA,
    ``target_parameters``, ``trainable_token_indices``, ``layer_replication``, a ``bias`` mode
    or a base-rewriting init (a streamed base cannot hold them); an adapter whose recorded
    ``base_model_name_or_path`` is not ``model_id`` (waive with ``allow_adapter_base_mismatch``
    when one checkpoint goes by two names); and an adapter with missing, extra or wrongly shaped
    tensors, or tensors that did not land. The tensor checks run once the streamed model is
    built (the shapes to compare with come from it), so a wrong adapter costs that build; a
    refusal after it releases the runtime. The saved tensors are cast to the adapter dtype
    (float32, as the trainer keeps it): an adapter saved in a lower precision loads, but is then
    only as exact as that precision.
    """
    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model, freeze_for_inference

    if index is None:
        try:
            index = read_shard_index(shard_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"no layer-stream shard index in {os.fspath(shard_dir)!r}; shard the checkpoint "
                "first (utils.layer_shard.shard_checkpoint, or run a streamed `soup train`)"
            ) from exc
    if (
        isinstance(buffers, bool)
        or not isinstance(buffers, int)
        or not MIN_STREAM_BUFFERS <= buffers <= MAX_STREAM_BUFFERS
    ):
        raise ValueError(
            f"buffers must be an integer between {MIN_STREAM_BUFFERS} and "
            f"{MAX_STREAM_BUFFERS} (the prefetch loads the next layer while the current one "
            f"computes, which needs two slots), got {buffers!r}"
        )
    source = None
    if adapter is not None:
        source = _prepare_adapter(adapter, model_id, allow_adapter_base_mismatch)
    on_cuda = str(device).startswith("cuda")
    model, runtime = build_streamed_model(
        model_id=model_id,
        shard_dir=shard_dir,
        index=index,
        lora_config=None if source is None else source.lora_config,
        device=device,
        dtype=dtype if dtype is not None else str(index.dtype),
        buffers=buffers,
        pin=on_cuda if pin is None else bool(pin),
        require_pin=require_pin,
        trust_remote_code=trust_remote_code,
        console=console,
        quant=str(getattr(index, "quant", "none")),
        double_quant=bool(getattr(index, "double_quant", True)),
        tier=tier,
        read_ahead=read_ahead,
        weights_dir=weights_dir,
        ngram_source=ngram_source,
        nf4_scale_cache=nf4_scale_cache,
    )
    if source is None:
        return StreamedInferenceModel(model, runtime)
    try:
        _load_adapter_weights(model, source.weights_path, source.header)
        # The build with a LoRA config leaves the adapters trainable and the layers checkpointing.
        freeze_for_inference(model)
    except BaseException:
        runtime.close()
        raise
    return StreamedInferenceModel(model, runtime, source.directory)
