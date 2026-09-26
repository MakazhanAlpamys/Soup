"""ReLoRA — periodic merge-into-base restarts for LoRA training.

Implements the restart loop from the ReLoRA paper (arXiv:2307.05695):
every N steps, merge the low-rank update into the frozen base weight,
reinitialize ``lora_A`` / ``lora_B``, clear optimizer state for those
adapters, and run a short learning-rate warm-up.

The callback is a no-op when ``policy`` is ``None``. Pass a
:class:`ReLoRAPolicy` to enable it. Lazy imports keep the module CLI-fast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Tuple

MAX_RELORA_STEPS = 10**7


@dataclass(frozen=True)
class ReLoRAPolicy:
    """Frozen policy for the ReLoRA callback."""

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

    def lr_warmup_steps(self) -> int:
        return min(100, max(1, self.steps // 10))


def magnitude_prune_tensor(tensor: Any, prune_ratio: float) -> Any:
    """Zero smallest-magnitude entries in place with exact keep-count (#693).

    Unused by the restart callback (optimizer moments are fully cleared, not
    pruned); retained so the exact keep-count under ties stays tested.
    """
    if not (0.0 < prune_ratio < 1.0):
        raise ValueError(
            f"magnitude_prune_tensor prune_ratio must be in (0, 1), got {prune_ratio}"
        )
    import torch  # lazy

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"magnitude_prune_tensor expects torch.Tensor, got {type(tensor)}")

    flat = tensor.detach().abs().reshape(-1)
    if flat.numel() <= 1:
        return tensor
    num_prune = max(1, int(flat.numel() * prune_ratio))
    if num_prune >= flat.numel():
        num_prune = flat.numel() - 1
    num_keep = flat.numel() - num_prune
    _, keep_idx = torch.topk(flat, num_keep, largest=True, sorted=False)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[keep_idx] = True
    tensor.detach().mul_(mask.reshape(tensor.shape).to(tensor.dtype))
    return tensor


def _is_sharded_or_quantized_tensor(tensor: Any) -> bool:
    import torch  # lazy

    if not isinstance(tensor, torch.Tensor):
        return True
    type_name = type(tensor).__name__
    if type_name in {"DTensor", "Params4bit", "Int8Params"}:
        return True
    if getattr(tensor, "quant_state", None) is not None:
        return True
    data = getattr(tensor, "data", None)
    if data is not None and data is not tensor and not isinstance(data, torch.Tensor):
        return True
    return False


def _require_writable_base(base_weight: Any) -> Any:
    import torch  # lazy

    if _is_sharded_or_quantized_tensor(base_weight):
        raise RuntimeError(
            "ReLoRA restart cannot merge into a quantized or sharded base weight. "
            "Use quantization='none' on a resident float LoRA run."
        )
    if not isinstance(base_weight, torch.Tensor):
        raise RuntimeError(
            f"ReLoRA restart expected a torch.Tensor base weight, got {type(base_weight)}"
        )
    if not base_weight.is_floating_point():
        raise RuntimeError(
            "ReLoRA restart requires a floating-point base weight "
            f"(got dtype={base_weight.dtype})."
        )
    return base_weight


def _resolve_adapter_key(module: Any) -> str:
    active = getattr(module, "active_adapter", None)
    if active is None:
        return "default"
    if isinstance(active, str):
        return active
    if isinstance(active, list):
        return active[0]
    return "default"


def _resolve_lora_weight(module: Any, attr: str, adapter: str) -> Any:
    import torch.nn as nn  # lazy

    container = getattr(module, attr, None)
    if container is None:
        raise RuntimeError(f"ReLoRA module missing {attr}")
    if isinstance(container, nn.ModuleDict):
        if adapter not in container:
            raise RuntimeError(
                f"ReLoRA adapter {adapter!r} not found in {attr} "
                f"(available: {list(container.keys())})"
            )
        return container[adapter].weight
    if hasattr(container, "weight"):
        return container.weight
    raise RuntimeError(f"ReLoRA could not resolve {attr} weight on {type(module)}")


def _resolve_base_weight(module: Any) -> Any:
    if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
        return module.base_layer.weight
    if hasattr(module, "base") and hasattr(module.base, "weight"):
        return module.base.weight
    if hasattr(module, "weight") and not hasattr(module, "lora_embedding_A"):
        return module.weight
    raise RuntimeError(
        f"ReLoRA could not locate a base weight on module {type(module).__name__}"
    )


def _lora_scaling(module: Any, adapter: str) -> float:
    scaling = getattr(module, "lora_scaling", None)
    if isinstance(scaling, dict):
        return float(scaling.get(adapter, 1.0))
    if scaling is not None:
        return float(scaling)
    alpha = getattr(module, "lora_alpha", None)
    r = getattr(module, "r", None)
    if alpha is not None and r:
        return float(alpha) / float(r)
    return 1.0


def _compute_delta(module: Any, adapter: str, weight_a: Any, weight_b: Any) -> Any:
    if hasattr(module, "get_delta_weight"):
        return module.get_delta_weight(adapter)
    scaling = _lora_scaling(module, adapter)
    delta = weight_b @ weight_a
    return delta * scaling


def _iter_lora_modules(model: Any) -> Iterator[Tuple[Any, str]]:
    import torch.nn as nn  # lazy

    seen: set[int] = set()
    for module in model.modules():
        if not isinstance(module, nn.Module):
            continue
        if id(module) in seen:
            continue
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        # peft embedding adapters keep an empty lora_A ModuleDict. Linear LoRA
        # is still merged: peft puts empty embedding dicts on every LoraLayer.
        lora_a = getattr(module, "lora_A", None)
        if isinstance(lora_a, nn.ModuleDict) and len(lora_a) == 0:
            continue
        seen.add(id(module))
        yield module, _resolve_adapter_key(module)


def _reinit_lora_weights(weight_a: Any, weight_b: Any) -> None:
    import torch.nn as nn  # lazy

    nn.init.kaiming_uniform_(weight_a, a=math.sqrt(5))
    nn.init.zeros_(weight_b)


def merge_reinit_lora_module(
    module: Any,
    adapter: str,
    *,
    merge: bool = True,
) -> Tuple[Any, Any]:
    """Merge one LoRA module into its base and reinitialize A/B."""
    import torch  # lazy

    weight_a = _resolve_lora_weight(module, "lora_A", adapter)
    weight_b = _resolve_lora_weight(module, "lora_B", adapter)
    base_weight = _require_writable_base(_resolve_base_weight(module))

    if merge:
        delta = _compute_delta(module, adapter, weight_a, weight_b)
        if delta.shape != base_weight.shape:
            raise RuntimeError(
                f"ReLoRA delta shape {tuple(delta.shape)} does not match base "
                f"{tuple(base_weight.shape)} on {type(module).__name__}"
            )
        with torch.no_grad():
            base_weight.add_(delta.to(device=base_weight.device, dtype=base_weight.dtype))

    _reinit_lora_weights(weight_a, weight_b)
    return weight_a, weight_b


def _preflight_writable_bases(model: Any) -> None:
    """Refuse a restart if any LoRA module's base cannot be merged in-place."""
    for module, _adapter in _iter_lora_modules(model):
        _require_writable_base(_resolve_base_weight(module))


def _assert_optimizer_resettable(optimizer: Any) -> None:
    state = getattr(optimizer, "state", None)
    if state is None:
        raise RuntimeError(
            "ReLoRA restart could not clear optimizer state: optimizer has no "
            "`.state` dict (wrapped DeepSpeed/FSDP optimizers are unsupported)."
        )


def _clear_optimizer_state(optimizer: Any, lora_params: list[Any]) -> None:
    state = optimizer.state
    for param in lora_params:
        if param in state:
            state[param] = type(state[param])() if state[param] else {}


def _merge_reinit_and_reset(
    model: Any,
    optimizer: Any,
    *,
    reset_optimizer: bool,
    merge: bool = True,
) -> None:
    if optimizer is not None:
        _assert_optimizer_resettable(optimizer)

    _preflight_writable_bases(model)

    lora_params = []
    for module, adapter in _iter_lora_modules(model):
        weight_a, weight_b = merge_reinit_lora_module(module, adapter, merge=merge)
        if weight_a.requires_grad:
            lora_params.append(weight_a)
        if weight_b.requires_grad:
            lora_params.append(weight_b)

    if optimizer is None or not reset_optimizer or not lora_params:
        return

    _clear_optimizer_state(optimizer, lora_params)


def _try_import_callback_base():
    try:
        from transformers import TrainerCallback  # noqa: PLC0415

        return TrainerCallback
    except Exception:  # noqa: BLE001
        return object


class _ReLoRACallback_body:  # type: ignore[misc]  # noqa: N801
    """HF TrainerCallback that runs ReLoRA restarts every N steps.

    Subclasses the lazily-resolved ``TrainerCallback`` so it inherits the no-op
    default for every Trainer event — HF's ``CallbackHandler.call_event``
    dispatches every event via ``getattr(cb, event)`` with no ``hasattr`` guard,
    so a bare duck-typed callback crashes on ``on_epoch_begin`` (#308). Only
    ``on_step_end`` is overridden. The lazy factory keeps the module import free
    of transformers.
    """

    def __init__(self, policy: Optional[ReLoRAPolicy], console: Any = None) -> None:
        self.policy = policy
        self.console = console
        self.fire_count = 0
        self._warmup_steps_total: Optional[int] = None
        self._warmup_steps_done = 0
        self._warmup_target_lrs: Optional[list[float]] = None

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        if self.policy is None:
            return control
        optimizer = kwargs.get("optimizer")
        self._maybe_advance_lr_warmup(optimizer)

        global_step = int(getattr(state, "global_step", 0) or 0)
        total_steps = getattr(state, "max_steps", None)
        try:
            total_steps_int = int(total_steps) if total_steps else None
        except (TypeError, ValueError):
            total_steps_int = None
        if not self.policy.should_fire(global_step, total_steps_int):
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        self._merge_reinit_and_reset(model, optimizer)
        self._start_lr_warmup(optimizer)
        self.fire_count += 1
        if self.console is not None:
            try:
                self.console.print(
                    f"[yellow]ReLoRA[/yellow] restart at step {global_step}"
                )
            except Exception:  # noqa: BLE001
                pass
        return control

    def _merge_reinit_and_reset(self, model: Any, optimizer: Any) -> None:
        if self.policy is None:
            return
        _merge_reinit_and_reset(
            model,
            optimizer,
            reset_optimizer=self.policy.reset_optimizer,
        )

    def _start_lr_warmup(self, optimizer: Any) -> None:
        if optimizer is None or self.policy is None:
            return
        self._warmup_steps_total = self.policy.lr_warmup_steps()
        self._warmup_steps_done = 0
        self._warmup_target_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
        for pg in optimizer.param_groups:
            pg["lr"] = 0.0

    def _maybe_advance_lr_warmup(self, optimizer: Any) -> None:
        if (
            optimizer is None
            or self._warmup_steps_total is None
            or self._warmup_target_lrs is None
        ):
            return
        if self._warmup_steps_done >= self._warmup_steps_total:
            return
        self._warmup_steps_done += 1
        frac = self._warmup_steps_done / self._warmup_steps_total
        for pg, target in zip(optimizer.param_groups, self._warmup_target_lrs):
            pg["lr"] = target * frac


_LAZY_CALLBACKS = {
    "ReLoRACallback": _ReLoRACallback_body,
}
_BODY_SKIP = frozenset(("__dict__", "__weakref__"))


def __getattr__(name: str):  # PEP 562
    body = _LAZY_CALLBACKS.get(name)
    if body is not None:
        base = _try_import_callback_base()
        ns = {k: v for k, v in vars(body).items() if k not in _BODY_SKIP}
        cls = type(name, (base,), ns)
        cls.__module__ = __name__
        cls.__qualname__ = name
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
