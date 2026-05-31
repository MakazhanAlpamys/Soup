"""ReLoRA — periodic LoRA adapter magnitude-prune + optimizer reset.

Mirrors the technique described in the ReLoRA paper / Axolotl
``monkeypatch/relora.py``: every N steps, magnitude-prune the LoRA
adapter weights and reset the optimizer state. Useful for very long
training runs where the LoRA capacity saturates.

The callback is a no-op when ``policy`` is ``None``. Pass a
:class:`ReLoRAPolicy` to enable it. Lazy imports keep the module CLI-fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Cap relora_steps to a sane upper bound.
MAX_RELORA_STEPS = 10**7


@dataclass(frozen=True)
class ReLoRAPolicy:
    """Frozen policy for the ReLoRA callback.

    Attributes:
        steps: fire every N global steps (must be > 0).
        warmup_ratio: fraction of total steps to skip at the start ([0, 1]).
        reset_optimizer: if True, clears optimizer state for the pruned
            LoRA parameters after pruning so momentum doesn't fight the
            new sparse weights.
        prune_ratio: fraction of LoRA weights to zero out, by magnitude
            (0 < x <= 1; e.g. 0.9 keeps the top 10%).
    """

    steps: int
    warmup_ratio: float = 0.1
    reset_optimizer: bool = True
    prune_ratio: float = 0.9

    def __post_init__(self) -> None:
        if not isinstance(self.steps, int) or isinstance(self.steps, bool):
            raise ValueError("ReLoRAPolicy.steps must be int")
        if self.steps <= 0 or self.steps > MAX_RELORA_STEPS:
            raise ValueError(
                f"ReLoRAPolicy.steps must be in (0, {MAX_RELORA_STEPS}], got {self.steps}"
            )
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError(
                f"ReLoRAPolicy.warmup_ratio must be in [0, 1], got {self.warmup_ratio}"
            )
        # Mirror magnitude_prune_tensor's strict (0, 1) bound. prune_ratio=1.0
        # would zero every weight on first fire — that's a footgun, not a feature.
        if not (0.0 < self.prune_ratio < 1.0):
            raise ValueError(
                f"ReLoRAPolicy.prune_ratio must be in (0, 1), got {self.prune_ratio}"
            )

    def should_fire(self, global_step: int, total_steps: Optional[int] = None) -> bool:
        if global_step <= 0:
            return False
        if global_step % self.steps != 0:
            return False
        if total_steps is not None and total_steps > 0:
            warmup_cutoff = int(total_steps * self.warmup_ratio)
            if global_step < warmup_cutoff:
                return False
        return True


def magnitude_prune_tensor(tensor: Any, prune_ratio: float) -> Any:
    """Zero out the smallest-magnitude entries of ``tensor`` in place.

    ``prune_ratio=0.9`` keeps the top 10% of weights by absolute value.
    ``prune_ratio`` of exactly 0.0 or 1.0 is rejected to avoid silent
    no-ops or wholesale zeroing.
    """
    if not (0.0 < prune_ratio < 1.0):
        raise ValueError(
            f"magnitude_prune_tensor prune_ratio must be in (0, 1), got {prune_ratio}"
        )
    import torch  # lazy

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"magnitude_prune_tensor expects torch.Tensor, got {type(tensor)}")

    flat = tensor.detach().abs().reshape(-1)
    # Empty / 1-element tensor: nothing meaningful to prune.
    if flat.numel() <= 1:
        return tensor
    k = max(1, int(flat.numel() * prune_ratio))
    if k >= flat.numel():
        # keep the single largest
        k = flat.numel() - 1
    # The k-th smallest absolute value: anything <= it gets zeroed
    threshold = torch.kthvalue(flat, k).values
    mask = tensor.detach().abs() > threshold
    tensor.detach().mul_(mask.to(tensor.dtype))
    return tensor


def _is_lora_param_name(name: str) -> bool:
    """Match PEFT's lora_A / lora_B parameter naming."""
    return ("lora_A" in name) or ("lora_B" in name)


class ReLoRACallback:
    """HF TrainerCallback that magnitude-prunes LoRA weights every N steps.

    We don't subclass ``transformers.TrainerCallback`` here so importing
    this module never loads transformers. The Trainer's callback dispatch
    works structurally — any object with the right method names is fine.
    """

    def __init__(self, policy: Optional[ReLoRAPolicy], console: Any = None) -> None:
        self.policy = policy
        self.console = console
        self.fire_count = 0

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        if self.policy is None:
            return control
        global_step = int(getattr(state, "global_step", 0) or 0)
        total_steps = getattr(state, "max_steps", None)
        try:
            total_steps_int = int(total_steps) if total_steps else None
        except (TypeError, ValueError):
            total_steps_int = None
        if not self.policy.should_fire(global_step, total_steps_int):
            return control
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        if model is None:
            return control
        self._prune_and_reset(model, optimizer)
        self.fire_count += 1
        if self.console is not None:
            try:
                self.console.print(
                    f"[yellow]ReLoRA[/yellow] fired at step {global_step} "
                    f"(prune_ratio={self.policy.prune_ratio})"
                )
            except Exception:  # noqa: BLE001 — console.print is best-effort in callback
                pass
        return control

    def _prune_and_reset(self, model: Any, optimizer: Any) -> None:
        import torch  # lazy

        pruned_params = []
        for name, param in model.named_parameters():
            if not _is_lora_param_name(name):
                continue
            if param.requires_grad and param.numel() > 0:
                with torch.no_grad():
                    magnitude_prune_tensor(param.data, self.policy.prune_ratio)
                pruned_params.append(param)

        if optimizer is None or not self.policy.reset_optimizer:
            return
        # Reset optimizer state only for the pruned parameters.
        try:
            state = getattr(optimizer, "state", None)
            if state is None:
                return
            for param in pruned_params:
                if param in state:
                    state[param] = type(state[param])() if state[param] else {}
        except Exception:
            # Optimizer state structure varies (DeepSpeed / FSDP wrap it);
            # silent best-effort is the documented Axolotl behaviour too.
            return
