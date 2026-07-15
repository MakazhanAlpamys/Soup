"""LISA — Layerwise Importance Sampled AdamW (v0.71.34 #267).

LISA (arXiv:2403.17919) gives full-FT quality at LoRA-like memory: every N
steps it freezes all decoder layers except a small randomly-sampled set
(embeddings + language-model head always trainable). The dynamic cousin of
Spectrum's static ``unfrozen_parameters`` selection.

Correctness invariant (see also ``trainer/sft.py``): the SFT trainer leaves the
model **fully trainable at prep time** so HF's ``create_optimizer`` (called
*before* ``on_train_begin``) includes every decoder parameter in its param
groups. This callback then toggles ``requires_grad`` — frozen parameters produce
``grad=None`` and AdamW skips them, and their optimizer state is cleared on
re-freeze — so peak optimizer memory ≈ (embed + head + ``num_layers`` active)
rather than the whole model.

No top-level torch/transformers — the callback is duck-typed (mirrors
``ReLoRACallback``); torch is never imported here at all (pure ``requires_grad``
toggling + optimizer ``.state`` dict operations).
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Parameter-name substrings that must stay trainable for every LISA interval:
# input embeddings, the LM head, and the final norm (LISA keeps them always on).
_ALWAYS_ON = (
    "embed_tokens",
    "embed_out",
    "wte",
    "wpe",
    "lm_head",
    "model.norm.",
    "ln_f.",
    "final_layernorm",
)

_LAYER_RE = re.compile(r"(?:layers|h)\.(\d+)\.")


@dataclass(frozen=True)
class LisaPolicy:
    """Immutable LISA schedule."""

    num_layers: int
    interval_steps: int
    reset_optimizer: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("num_layers", "interval_steps", "seed"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"LisaPolicy.{name} must be int, not {type(val).__name__}")
        if self.num_layers < 1:
            raise ValueError("LisaPolicy.num_layers must be >= 1")
        if self.interval_steps < 1:
            raise ValueError("LisaPolicy.interval_steps must be >= 1")
        if self.seed < 0:
            raise ValueError("LisaPolicy.seed must be >= 0")
        if not isinstance(self.reset_optimizer, bool):
            raise TypeError("LisaPolicy.reset_optimizer must be bool")


def _is_always_on(name: str) -> bool:
    return any(sub in name for sub in _ALWAYS_ON)


def locate_decoder_layer_indices(model: Any) -> list[int]:
    """Sorted-unique decoder-layer indices from parameter names."""
    seen: set[int] = set()
    for name, _ in model.named_parameters():
        m = _LAYER_RE.search(name)
        if m:
            seen.add(int(m.group(1)))
    return sorted(seen)


class LisaCallback:
    """Duck-typed HF ``TrainerCallback`` implementing LISA layer sampling.

    Not a ``transformers.TrainerCallback`` subclass so importing this module
    never loads transformers — the Trainer's callback dispatch is structural.
    """

    def __init__(self, policy: LisaPolicy, console: Any = None) -> None:
        self.policy = policy
        self.console = console
        self._rng = random.Random(policy.seed)
        # decoder param objects currently trainable — tracked so we can clear
        # optimizer state for the ones that get re-frozen on the next sample.
        self._active_decoder_params: list[Any] = []
        self.fire_count = 0

    # -- HF callback hooks --------------------------------------------------
    def on_train_begin(self, args: Any, state: Any, control: Any = None, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        if model is not None:
            self._resample(model, optimizer=None)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any = None, **kwargs: Any) -> Any:
        global_step = int(getattr(state, "global_step", 0) or 0)
        if global_step <= 0 or global_step % self.policy.interval_steps != 0:
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        self._resample(model, optimizer=kwargs.get("optimizer"))
        self.fire_count += 1
        if self.console is not None:
            try:
                self.console.print(
                    f"[cyan]LISA[/cyan] re-sampled {self.policy.num_layers} "
                    f"decoder layer(s) at step {global_step}"
                )
            except Exception:  # noqa: BLE001 — console.print is best-effort
                pass
        return control

    # -- core ---------------------------------------------------------------
    def _resample(self, model: Any, optimizer: Any) -> None:
        indices = locate_decoder_layer_indices(model)
        if not indices:
            logger.warning(
                "LISA: could not detect numbered decoder layers "
                "('layers.N.' / 'h.N.') — no layer sampling applied."
            )
            return
        k = min(self.policy.num_layers, len(indices))
        chosen = set(self._rng.sample(indices, k))

        prev_active = self._active_decoder_params
        new_active: list[Any] = []
        skipped_non_float: list[str] = []

        for name, param in model.named_parameters():
            if _is_always_on(name):
                param.requires_grad = True
                continue
            m = _LAYER_RE.search(name)
            if m is None:
                # non-layer, non-always-on params (rare) stay frozen
                param.requires_grad = False
                continue
            in_chosen = int(m.group(1)) in chosen
            if in_chosen and not (param.is_floating_point() or param.is_complex()):
                skipped_non_float.append(name)
                param.requires_grad = False
                continue
            param.requires_grad = in_chosen
            if in_chosen:
                new_active.append(param)

        # Clear optimizer state for decoder params that were active and are now
        # frozen (avoids stale Adam moments; mirrors ReLoRACallback reset).
        if optimizer is not None and self.policy.reset_optimizer:
            new_ids = {id(p) for p in new_active}
            re_frozen = [p for p in prev_active if id(p) not in new_ids]
            self._clear_optimizer_state(optimizer, re_frozen)

        self._active_decoder_params = new_active

        if skipped_non_float:
            logger.warning(
                "LISA: %d matched parameter(s) are non-float (e.g. quantized) "
                "and cannot be trained — skipped. LISA requires "
                "quantization: none. First skipped: %s",
                len(skipped_non_float),
                skipped_non_float[0],
            )

    @staticmethod
    def _clear_optimizer_state(optimizer: Any, params: list[Any]) -> None:
        try:
            state = getattr(optimizer, "state", None)
            if state is None:
                return
            for param in params:
                if param in state:
                    state[param] = {}
        except Exception:  # noqa: BLE001 — optimizer state shape varies (DS/FSDP)
            return
