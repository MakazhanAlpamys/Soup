"""Frozen-prefix forward execution.

Scope:
- Mistral and OLMoE decoders
- Resident model (no layer streaming)
- No KV cache
- Deterministic frozen prefix
"""

from __future__ import annotations

import contextlib
import types
from pathlib import Path
from typing import Any

from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
    fingerprint_frozen_prefix,
    fingerprint_prefix_configuration,
    resolve_cache_directory,
)

_SUPPORTED = "Mistral and OLMoE"


def hooked_modules(modules: Any) -> list[str]:
    """Names of the modules (and their submodules) that carry a forward hook.

    A forward or forward-pre hook on the embeddings or a frozen layer can change
    what that part computes (NEFTune noise is one), so a cached activation could
    differ from a fresh one.
    """
    found = []
    for prefix, module in modules:
        for name, submodule in module.named_modules(prefix=prefix):
            if getattr(submodule, "_forward_hooks", None) or getattr(
                submodule, "_forward_pre_hooks", None
            ):
                found.append(name)
    return found


def execution_signature(embeddings: Any, config: Any) -> dict[str, Any]:
    """What besides the weights and the inputs shapes the numbers of the frozen prefix.

    The activation dtype, the autocast state, the device type and the attention
    implementation each change the result, so each is part of the cache key.
    """
    import torch

    device_type = embeddings.device.type
    autocast_dtype = None
    try:
        if torch.is_autocast_enabled(device_type):
            autocast_dtype = str(torch.get_autocast_dtype(device_type))
    except RuntimeError:  # a device type autocast does not know
        pass
    return {
        "dtype": str(embeddings.dtype),
        "autocast": autocast_dtype,
        "device_type": device_type,
        "attn_implementation": str(getattr(config, "_attn_implementation", None)),
    }


def estimate_cache_bytes(rows: int, max_length: int, hidden_size: int, element_size: int) -> int:
    """Upper bound on the entries of one epoch: rows x tokens x hidden x bytes.

    The cache stores one activation per distinct batch at the cutoff, so the
    total does not depend on the batch size; padding to the longest row in each
    batch makes the real figure smaller.
    """
    return rows * max_length * hidden_size * element_size


class FrozenPrefixRunner:
    """Execute a decoder with reusable frozen-prefix activations."""

    def __init__(
        self,
        decoder: Any,
        cache: FrozenPrefixCache,
        cutoff: int,
        metadata_factory: Any,
    ):
        if not 0 < cutoff < len(decoder.layers):
            raise ValueError("Invalid frozen-prefix cutoff")

        self.decoder = decoder
        self.cache = cache
        self.cutoff = cutoff
        self.metadata_factory = metadata_factory
        self.frozen_prefix_fingerprint = fingerprint_frozen_prefix(decoder, cutoff)
        self._frozen_parameters = tuple(
            [
                *decoder.embed_tokens.parameters(),
                *(
                    parameter
                    for layer in decoder.layers[:cutoff]
                    for parameter in layer.parameters()
                ),
            ]
        )

        self._validate_deterministic_prefix()
        self._refuse_hooks()

        if any(param.requires_grad for param in self._frozen_parameters):
            raise ValueError("The frozen-prefix cache requires a fully frozen prefix")

        self._frozen_versions = tuple(param._version for param in self._frozen_parameters)
        self.hits = 0
        self.misses = 0

    def _validate_deterministic_prefix(self) -> None:
        """Reject stochastic operations in the frozen prefix."""
        import torch.nn as nn

        decoder = self.decoder
        frozen_modules = [decoder.embed_tokens, *decoder.layers[: self.cutoff]]

        for module in frozen_modules:
            for submodule in module.modules():
                if (
                    isinstance(submodule, nn.modules.dropout._DropoutNd)
                    and submodule.training
                    and submodule.p > 0
                ):
                    raise ValueError(
                        "The frozen-prefix cache requires a deterministic frozen "
                        "prefix: active dropout is not supported"
                    )

        attention_dropout = getattr(decoder.config, "attention_dropout", 0.0)

        if attention_dropout > 0 and any(
            layer.training for layer in decoder.layers[: self.cutoff]
        ):
            raise ValueError(
                "The frozen-prefix cache requires a deterministic frozen prefix: "
                "attention_dropout must be zero during training"
            )

    def _refuse_hooks(self) -> None:
        """Refuse a forward hook on the embeddings or the frozen layers."""
        decoder = self.decoder
        found = hooked_modules(
            [
                ("embed_tokens", decoder.embed_tokens),
                *((f"layers.{index}", decoder.layers[index]) for index in range(self.cutoff)),
            ]
        )
        if found:
            raise ValueError(
                "training.frozen_prefix_cache cannot run with a forward hook on the input "
                f"embeddings or a frozen layer (found on {found[0]}): a hook can change the "
                "cached activation, and NEFTune installs one. Remove the hook or turn the "
                "cache off."
            )

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Any:
        import torch
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
        from transformers.modeling_outputs import (
            BaseModelOutputWithPast,
            MoeModelOutputWithPast,
        )

        if past_key_values is not None or use_cache:
            raise ValueError("The frozen-prefix cache does not support KV caching")

        if inputs_embeds is not None or input_ids is None:
            raise ValueError("The frozen-prefix cache requires input_ids")

        kwargs.pop("num_items_in_batch", None)
        output_attentions = kwargs.pop("output_attentions", None)
        output_hidden_states = kwargs.pop("output_hidden_states", None)
        router_logits = kwargs.pop("output_router_logits", None)

        if router_logits not in (None, False):
            raise ValueError(
                "The frozen-prefix cache does not support output_router_logits=True: "
                "the cached prefix has no router logits. Set "
                "training.moe_aux_loss_coeff to 0."
            )

        if output_attentions not in (None, False):
            raise ValueError("The frozen-prefix cache does not support output_attentions=True")

        if output_hidden_states not in (None, False):
            raise ValueError(
                "The frozen-prefix cache does not support output_hidden_states=True"
            )

        if kwargs:
            raise ValueError(
                f"The frozen-prefix cache got unsupported forward arguments: {sorted(kwargs)}"
            )
        self._validate_deterministic_prefix()
        # NEFTune and similar hooks are installed after this runner, when training
        # starts, so they are looked for on every forward and not only at install.
        self._refuse_hooks()
        decoder = self.decoder

        embeddings = decoder.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                embeddings.shape[1], device=embeddings.device, dtype=input_ids.dtype
            ).unsqueeze(0)

        mask_function = (
            create_causal_mask
            if getattr(decoder.config, "sliding_window", None) is None
            else create_sliding_window_causal_mask
        )

        causal_mask = mask_function(
            config=decoder.config,
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

        position_embeddings = decoder.rotary_emb(embeddings, position_ids=position_ids)

        # The key covers the input ids, the attention mask and the position ids
        # the layers actually see, on top of the weights and the configuration
        # (RoPE included) fingerprinted at install time. Any difference is a miss.
        metadata = dict(
            self.metadata_factory(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        )
        metadata["frozen_prefix_fingerprint"] = self.frozen_prefix_fingerprint
        metadata["execution"] = execution_signature(embeddings, decoder.config)

        current_versions = tuple(param._version for param in self._frozen_parameters)
        if current_versions != self._frozen_versions:
            raise RuntimeError(
                "The frozen-prefix weights changed during training. "
                "Restart the run to rebuild the cache fingerprint."
            )

        hidden = self.cache.load(metadata)

        if hidden is None:
            self.misses += 1
            hidden = embeddings

            with torch.no_grad():
                for layer in decoder.layers[: self.cutoff]:
                    hidden = _layer_output(
                        layer(
                            hidden,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_values=None,
                            use_cache=False,
                            position_embeddings=position_embeddings,
                        )
                    )

            self.cache.save(hidden, metadata)
        else:
            self.hits += 1
            # Keep the stored dtype: it is what the layers produced when the
            # entry was written, so a hit and a miss feed the upper layers alike.
            hidden = hidden.to(device=embeddings.device)

        for layer in decoder.layers[self.cutoff :]:
            hidden = _layer_output(
                layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
            )

        hidden = decoder.norm(hidden)

        # The MoE head reads outputs.router_logits unconditionally; the cached prefix has
        # none, so MoE runs set moe_aux_loss_coeff: 0 and output_router_logits stays off.
        output_class = (
            MoeModelOutputWithPast
            if type(decoder).__name__ == "OlmoeModel"
            else BaseModelOutputWithPast
        )
        return output_class(last_hidden_state=hidden, past_key_values=None)


def _layer_output(output: Any) -> Any:
    """A decoder layer returns a tensor, or a tuple that starts with it."""
    return output[0] if isinstance(output, tuple) else output


@contextlib.contextmanager
def install_frozen_prefix_cache(
    model: Any,
    cache: FrozenPrefixCache,
    cutoff: int,
    metadata_factory: Any,
):
    """Temporarily install the cached forward on a resident decoder."""
    from transformers import MistralModel, OlmoeModel

    from soup_cli.utils.layer_stream_runtime import decoder_owner

    decoder = decoder_owner(model)

    if not isinstance(decoder, (MistralModel, OlmoeModel)):
        raise ValueError(
            f"training.frozen_prefix_cache currently supports {_SUPPORTED} models only; "
            f"this model's decoder is {type(decoder).__name__}."
        )

    if not 0 < cutoff < len(decoder.layers):
        raise ValueError(
            "training.frozen_prefix_cache needs a nonempty frozen prefix and a nonempty "
            f"upper stack; got cutoff={cutoff} of {len(decoder.layers)} layers."
        )

    if any(param.requires_grad for param in decoder.embed_tokens.parameters()):
        raise ValueError("training.frozen_prefix_cache requires frozen input embeddings")

    for layer in decoder.layers[:cutoff]:
        if any(param.requires_grad for param in layer.parameters()):
            raise ValueError("training.frozen_prefix_cache requires all prefix parameters frozen")

    runner = FrozenPrefixRunner(
        decoder=decoder,
        cache=cache,
        cutoff=cutoff,
        metadata_factory=metadata_factory,
    )

    original_forward = decoder.forward

    def cached_forward(self, *args, **kwargs):
        if self is not runner.decoder:
            raise RuntimeError(
                "training.frozen_prefix_cache does not support torch.nn.DataParallel "
                "replicas. Use a single GPU or a validated model-sharding setup."
            )
        return runner.forward(*args, **kwargs)

    decoder.forward = types.MethodType(cached_forward, decoder)

    try:
        yield runner
    finally:
        decoder.forward = original_forward


def announce_cache(
    console: Any,
    cache: FrozenPrefixCache,
    decoder: Any,
    *,
    rows: int | None,
    max_length: int | None,
) -> None:
    """Print where the cache lives, how large it can get and how much room there is."""
    import shutil

    from soup_cli.utils.frozen_prefix_cache import free_space_reserve
    from soup_cli.utils.terminal import for_terminal

    try:
        usage = shutil.disk_usage(cache.directory)
    except OSError:
        usage = None
    free = "unknown" if usage is None else f"{usage.free / 1e9:.1f} GB"
    reserve = "" if usage is None else (
        f" Writing stops when less than {free_space_reserve(usage.total) / 1e9:.1f} GB "
        "would stay free."
    )
    estimate = None
    if rows is None or max_length is None:
        size = "The size depends on the training set"
    else:
        element_size = decoder.embed_tokens.weight.element_size()
        estimate = estimate_cache_bytes(rows, max_length, decoder.config.hidden_size, element_size)
        size = (
            f"Up to about {estimate / 1e9:.1f} GB for {rows} rows of {max_length} tokens "
            "(an upper bound: batches are padded to their longest row)"
        )
    console.print(
        f"[green]Frozen-prefix cache directory:[/] {for_terminal(str(cache.directory))}. "
        f"{size}; {free} free on that volume.{reserve} "
        "The folder is kept after the run."
    )
    if (
        estimate is not None
        and usage is not None
        and estimate > usage.free - free_space_reserve(usage.total)
    ):
        console.print(
            "[yellow]The estimate is larger than the free space above the reserve: the "
            "cache will stop writing before the first epoch ends, and later epochs will "
            "reuse only the batches already stored.[/]"
        )


@contextlib.contextmanager
def frozen_prefix_session(
    model: Any,
    *,
    cutoff: int,
    output_dir: str | Path,
    model_revision: str,
    n_gpu: int = 1,
    quantized: bool = False,
    rows: int | None = None,
    max_length: int | None = None,
    console: Any = None,
):
    """Install the cache for one training run, with its directory under ``output_dir``.

    ``cutoff`` is the number of frozen bottom layers. Yields the runner, whose
    ``hits`` and ``misses`` count the lookups. With ``console`` it prints the
    directory, an upper bound on the cache size (when ``rows`` and ``max_length``
    are known), the free space, and any later stop of writing.
    """
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    if cutoff <= 0:
        raise ValueError(
            "training.frozen_prefix_cache needs at least one frozen decoder layer: "
            "training.freeze_layers / training.freeze_ratio froze none."
        )

    if n_gpu > 1 and not quantized:
        # The cache is attached to one decoder instance and cannot serve the
        # replicas torch.nn.DataParallel makes.
        raise RuntimeError(
            "training.frozen_prefix_cache does not support torch.nn.DataParallel. "
            "Use a single visible GPU or a validated model-sharding configuration."
        )

    decoder = decoder_owner(model)
    cache = FrozenPrefixCache(
        resolve_cache_directory(output_dir),
        notify=(lambda message: console.print(f"[yellow]{message}[/]")) if console else None,
    )
    if console is not None:
        announce_cache(console, cache, decoder, rows=rows, max_length=max_length)
    # Computed once per installation: no per-step host synchronisation for the key.
    configuration = fingerprint_prefix_configuration(decoder, cutoff)

    def metadata_factory(*, input_ids, attention_mask, position_ids):
        return build_cache_metadata(
            model_revision=model_revision,
            frozen_prefix_fingerprint="runner-computed",
            config_fingerprint=configuration,
            cutoff=cutoff,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    with install_frozen_prefix_cache(
        model=model,
        cache=cache,
        cutoff=cutoff,
        metadata_factory=metadata_factory,
    ) as runner:
        yield runner
