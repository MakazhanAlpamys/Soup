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

No heavy module is imported at module level: torch, transformers and peft (never needed here)
are reached from inside the functions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from soup_cli.utils.config_bounds import (
    DEFAULT_STREAM_READ_AHEAD,
    MAX_STREAM_BUFFERS,
    MIN_STREAM_BUFFERS,
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
    model (eval mode, every parameter frozen, no PEFT wrapper); ``runtime`` is the
    ``StreamRuntime`` that feeds it. ``close()`` (or leaving the ``with`` block) releases the
    runtime and is idempotent; a closed model refuses to run.
    """

    def __init__(self, model: Any, runtime: Any) -> None:
        self.model = model
        self.runtime = runtime
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

    This is the inference counterpart of ``build_streamed_model(lora_config=None)``: no PEFT, no
    adapter, no trainable parameter, no gradient checkpointing, ``use_cache=False`` (there is
    no KV cache). Use the result in a ``with`` block, or call ``close()``.
    """
    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

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
    on_cuda = str(device).startswith("cuda")
    model, runtime = build_streamed_model(
        model_id=model_id,
        shard_dir=shard_dir,
        index=index,
        lora_config=None,
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
    return StreamedInferenceModel(model, runtime)
