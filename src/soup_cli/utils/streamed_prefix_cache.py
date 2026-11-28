"""Bounded, per-row activation storage for a frozen streamed decoder prefix."""

from __future__ import annotations

import hashlib
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from soup_cli.utils.frozen_prefix_cache import fingerprint_prefix_configuration
from soup_cli.utils.layer_stream_runtime import decoder_owner

OWNER_ATTR = "_soup_streamed_prefix_cache"
SUPPORTED_DECODERS = ("llama", "mistral", "qwen2", "qwen3", "qwen3_5", "qwen3_5_text")
_DECODER_INPUTS = {
    "input_ids", "inputs_embeds", "attention_mask", "position_ids", "cache_position",
    "past_key_values", "use_cache", "output_attentions", "output_hidden_states",
    "output_router_logits", "return_dict", "num_items_in_batch",
}


def resolve_cache_bytes(limit_gib: float | None) -> int:
    """Cap the per-run store and leave host RAM for streaming and L2L allocations."""
    from soup_cli.utils.layer_stream import free_ram_bytes

    configured = int((8 if limit_gib is None else limit_gib) * (1 << 30))
    free = free_ram_bytes()
    return configured if free is None else min(configured, max(0, free // 4))


def streamed_cache_decoder(model: Any) -> Any:
    decoder = decoder_owner(model)
    if (
        getattr(decoder.layers[0], "prefetcher", None) is None
        or decoder.config.model_type not in SUPPORTED_DECODERS
    ):
        raise ValueError(
            "training.frozen_prefix_cache with training.stream_layers requires a "
            "streamed Llama, Mistral, Qwen2, Qwen3 or Qwen3.5 decoder."
        )
    return decoder


def _hash_value(digest: Any, value: Any) -> bool:
    """Include type, shape and bytes. Unknown layer metadata makes the input uncacheable."""
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_meta or value.layout != torch.strided:
            return False
        tensor = value.detach().cpu().contiguous()
        digest.update(repr((str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    elif value is None or type(value) in (bool, int, float, str):
        data = repr(value).encode()
        digest.update(str(len(data)).encode() + b":" + data)
    elif isinstance(value, (tuple, list)):
        digest.update(repr((type(value).__name__, len(value))).encode())
        return all(_hash_value(digest, item) for item in value)
    elif isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return _hash_value(digest, tuple(sorted(value.items())))
    else:
        return False
    return True


def _execution_context(device: Any) -> tuple[Any, ...]:
    import torch

    autocast = torch.is_autocast_enabled(device.type)
    return (
        str(device), autocast, str(torch.get_autocast_dtype(device.type)) if autocast else None,
        torch.get_float32_matmul_precision(),
    )


@dataclass
class PrefixCacheStats:
    cutoff: int
    hits: int = 0
    misses: int = 0
    partial: int = 0
    l2l_hits: int = 0
    l2l_misses: int = 0
    l2l_partial: int = 0
    rows_stored: int = 0
    rows_not_stored: int = 0
    bytes_stored: int = 0


@dataclass(frozen=True)
class _DecoderBatch:
    keys: list[bytes]
    shape: tuple[int, int]
    start: int


class StreamedPrefixCache:
    """Advance a cutoff while keeping valid entries from older, shorter prefixes."""

    def __init__(self, model: Any, cutoff: int, max_bytes: int) -> None:
        self.decoder = streamed_cache_decoder(model)
        self.prefetcher = self.decoder.layers[0].prefetcher
        self.max_bytes = max_bytes
        self.stats = PrefixCacheStats(cutoff)
        self._rows: dict[bytes, tuple[int, Any]] = {}
        self._batch: _DecoderBatch | None = None
        self._handles: list[Any] = []
        self._forwards: list[tuple[Any, Any]] = []
        self._signature = self._inspect_prefix(cutoff)
        self._floor = self.prefetcher.backward_floor
        self._suspended = False

    @property
    def cutoff(self) -> int:
        return self.stats.cutoff

    @property
    def enabled(self) -> bool:
        return self.decoder.training and not self._suspended

    @contextmanager
    def bypass(self) -> Iterator[None]:
        previous = self._suspended
        self._suspended = True
        try:
            yield
        finally:
            self._suspended = previous

    def _inspect_prefix(self, cutoff: int) -> tuple[Any, ...]:
        import torch.nn as nn

        if not 0 < cutoff < len(self.decoder.layers):
            raise ValueError(
                "training.frozen_prefix_cache needs a nonempty frozen prefix and "
                "a nonempty trainable suffix."
            )
        modules = [self.decoder.embed_tokens, *self.decoder.layers[:cutoff]]
        parameters = [parameter for module in modules for parameter in module.parameters()]
        if any(parameter.requires_grad for parameter in parameters):
            raise ValueError(
                "training.frozen_prefix_cache requires frozen input embeddings and "
                "all prefix parameters frozen."
            )
        children = [sub for module in modules for sub in module.modules()]
        if getattr(self.decoder.config, "attention_dropout", 0) or any(
            (isinstance(sub, nn.modules.dropout._DropoutNd) and sub.p > 0)
            or getattr(sub, "attention_dropout", 0)
            for sub in children
        ):
            raise ValueError(
                "training.frozen_prefix_cache requires zero dropout in the frozen prefix "
                "(including attention_dropout and training.lora.dropout)."
            )
        adapters = (
            (
                id(sub), tuple(sub.active_adapters), sub.disable_adapters,
                tuple(sub.merged_adapters),
            )
            for sub in children if hasattr(sub, "lora_A")
        )
        return (
            tuple(
                (
                    id(parameter), parameter._version, parameter.dtype, parameter.device,
                    parameter.data_ptr(),
                )
                for parameter in parameters
            ),
            tuple(adapters),
        )

    def _check_prefix(self) -> None:
        if self._inspect_prefix(self.cutoff) != self._signature:
            raise RuntimeError(
                "training.frozen_prefix_cache detected a changed frozen parameter or adapter "
                "state. Restart training to rebuild the streamed cache."
            )

    def install(self) -> StreamedPrefixCache:
        if getattr(self.decoder, OWNER_ATTR, None) is not None:
            raise ValueError("training.frozen_prefix_cache is already installed on the decoder.")
        self._handles.append(self.decoder.register_forward_pre_hook(
            self._before_decoder, with_kwargs=True, prepend=True,
        ))
        self._handles.append(self.decoder.register_forward_hook(
            self._after_decoder, always_call=True,
        ))
        self._patch_layers(self.cutoff)
        setattr(self.decoder, OWNER_ATTR, self)
        return self

    def raise_cutoff(self, cutoff: int) -> None:
        if cutoff <= self.cutoff:
            raise ValueError("training.frozen_prefix_cache can only advance the streamed cutoff.")
        self._check_prefix()
        signature = self._inspect_prefix(cutoff)
        self._patch_layers(cutoff)
        self._signature = signature
        self.stats.cutoff = cutoff

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for layer, original in self._forwards:
            if original is None:
                vars(layer).pop("forward", None)
            else:
                layer.forward = original
        self._forwards.clear()
        if getattr(self.decoder, OWNER_ATTR, None) is self:
            delattr(self.decoder, OWNER_ATTR)
            self._after_decoder()
            self.prefetcher.backward_floor = self._floor
        self._rows.clear()
        self.stats.bytes_stored = 0

    def _patch_layers(self, cutoff: int) -> None:
        for index in range(len(self._forwards), cutoff):
            layer = self.decoder.layers[index]
            original = layer.forward
            self._forwards.append((layer, vars(layer).get("forward")))
            layer.forward = types.MethodType(self._wrap(index, original), layer)

    def _wrap(self, index: int, original: Any) -> Any:
        def forward(_layer: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            import torch

            batch = self._batch
            if batch is not None and index < batch.start:
                if index == 0:
                    if tuple(hidden_states.shape[:2]) != batch.shape:
                        raise RuntimeError("training.frozen_prefix_cache input shape changed.")
                    return torch.stack([
                        self._rows[key][1].to(hidden_states.device) for key in batch.keys
                    ])
                return hidden_states
            out = original(hidden_states, *args, **kwargs)
            if batch is not None and index == self.cutoff - 1:
                if out.ndim != 3 or tuple(out.shape[:2]) != batch.shape:
                    raise RuntimeError("training.frozen_prefix_cache output shape changed.")
                for key, row in zip(batch.keys, out, strict=True):
                    self._keep(key, row)
            return out

        return forward

    def _after_decoder(self, *_args: Any) -> None:
        self._batch = None
        self.prefetcher.prime_start = 0

    def _decoder_keys(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[bytes]:
        import torch

        if len(args) > 1 or kwargs.keys() - _DECODER_INPUTS:
            return []
        if kwargs.get("past_key_values") is not None or any(
            kwargs.get(name) or getattr(self.decoder.config, name, False) for name in (
                "use_cache", "output_attentions", "output_hidden_states", "output_router_logits",
            )
        ):
            return []
        ids = args[0] if args else kwargs.get("input_ids")
        embeds = kwargs.get("inputs_embeds")
        if embeds is None:
            payload, ndim, kind = ids, 2, "ids"
        elif ids is None:
            payload, ndim, kind = embeds, 3, "embeds"
        else:
            return []
        if not isinstance(payload, torch.Tensor) or payload.ndim != ndim or payload.requires_grad:
            return []
        rows, width = payload.shape[:2]
        if not rows or not width:
            return []
        mask, positions = kwargs.get("attention_mask"), kwargs.get("position_ids")
        if mask is not None and (
            not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (rows, width)
        ):
            return []
        if positions is not None and (
            not isinstance(positions, torch.Tensor)
            or positions.ndim < 2
            or positions.shape[-1] != width
            or positions.shape[-2] not in (1, rows)
        ):
            return []
        digest = hashlib.sha256()
        # Use a fixed configuration fingerprint scope across progressive cutoffs.
        _hash_value(digest, (
            kind, rows, width, _execution_context(payload.device),
            getattr(self.decoder.config, "_attn_implementation", None),
            fingerprint_prefix_configuration(self.decoder, 1),
        ))
        if not _hash_value(digest, kwargs.get("cache_position")):
            return []
        content = payload.detach().cpu()
        mask = mask.detach().cpu() if mask is not None else None
        positions = positions.detach().cpu() if positions is not None else None
        _hash_value(digest, None if mask is None else bool(mask.all()))
        keys = []
        for idx, row in enumerate(content):
            key = digest.copy()
            pos = None
            if positions is not None:
                pos = positions[..., idx if positions.shape[-2] > 1 else 0, :]
            if not _hash_value(key, (row, None if mask is None else mask[idx], pos)):
                return []
            keys.append(key.digest())
        return keys

    def _before_decoder(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        import torch

        self._after_decoder()
        self.prefetcher.backward_floor = self._floor
        if not self.enabled or not torch.is_grad_enabled():
            return
        keys = self._decoder_keys(args, kwargs)
        if not keys:
            return
        self._check_prefix()
        levels = {self._level(key) for key in keys}
        # A streamed decoder still builds its batch's masks and rotary arguments. The batch
        # can skip together if all rows share a level. L2L can resume each row independently.
        start = next(iter(levels)) if len(levels) == 1 else 0
        if start == self.cutoff:
            self.stats.hits += 1
        elif start:
            self.stats.partial += 1
        else:
            self.stats.misses += 1
        payload = args[0] if args else kwargs.get("input_ids")
        if payload is None:
            payload = kwargs["inputs_embeds"]
        self._batch = _DecoderBatch(keys, tuple(payload.shape[:2]), start)
        self.prefetcher.prime_start = start
        self.prefetcher.backward_floor = self.cutoff

    def _level(self, key: bytes) -> int:
        entry = self._rows.get(key)
        return 0 if entry is None else entry[0]

    def _keep(self, key: bytes, value: Any) -> None:
        old = self._rows.get(key)
        if old is not None and old[0] == self.cutoff:
            return
        size = value.numel() * value.element_size()
        freed = 0 if old is None else old[1].numel() * old[1].element_size()
        total = self.stats.bytes_stored - freed + size
        if total > self.max_bytes:
            self.stats.rows_not_stored += 1
            return
        self._rows[key] = (self.cutoff, value.detach().to("cpu", copy=True))
        self.stats.rows_stored += 1
        self.stats.bytes_stored = total

    def l2l_lookup(
        self, rows: Sequence[Any], calls: Sequence[Any],
    ) -> tuple[list[bytes], list[int], list[Any]] | None:
        self._check_prefix()
        keys = []
        for ids, call in zip(rows, calls, strict=True):
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
                return None
            digest = hashlib.sha256(b"l2l")
            context = (
                _execution_context(ids.device),
                getattr(self.decoder.config, "_attn_implementation", None),
                ids, call.args, call.kwargs,
            )
            if not _hash_value(digest, context):
                return None
            keys.append(digest.digest())
        levels = [self._level(key) for key in keys]
        kept = [
            self._rows[key][1].to(ids.device, copy=True).unsqueeze(0) if level else None
            for key, ids, level in zip(keys, rows, levels, strict=True)
        ]
        if min(levels) == self.cutoff:
            self.stats.l2l_hits += 1
        elif min(levels) > 0:
            self.stats.l2l_partial += 1
        else:
            self.stats.l2l_misses += 1
        return keys, levels, kept

    def l2l_store(self, keys: Sequence[bytes], ids: Sequence[Any], hidden: Sequence[Any]) -> None:
        for key, row, value in zip(keys, ids, hidden, strict=True):
            if value.ndim != 3 or tuple(value.shape[:2]) != tuple(row.shape):
                raise RuntimeError("training.frozen_prefix_cache L2L output shape changed.")
            self._keep(key, value[0])
