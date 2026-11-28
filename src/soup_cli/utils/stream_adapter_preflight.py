"""Budget streamed adapter tensors from a meta model before opening its weights."""

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

from soup_cli.utils.adapter_budget import AdapterBudget, AdapterTensor, estimate_adapter_budget


@dataclass(frozen=True)
class StreamAdapterPlan:
    tensors: tuple[AdapterTensor, ...]
    budget: AdapterBudget
    lora_config: Any
    quant_suffixes: tuple[str, ...]
    target_modules: Any


def _optimizer_profile(tcfg: Any, *, optimizer_backend: str | None = None) -> str:
    """Select a profile for the effective optimizer used by Trainer."""
    if optimizer_backend is not None:
        return f"unsupported: {optimizer_backend}"
    if getattr(tcfg, "use_galore", False):
        return "unsupported: GaLore"
    if getattr(tcfg, "loraplus_lr_ratio", None) is not None:
        return "unsupported: LoRA+"
    name = getattr(tcfg, "optimizer", "adamw_torch")
    if getattr(tcfg, "use_lorafa", False):
        if name not in (None, "adamw_torch", "adamw", "adamw_torch_fused"):
            return "unsupported: LoRA-FA optimizer conflict"
        return "peft_lorafa_fp32"
    profiles = {
        "adamw_torch": "torch_adamw_fp32",
        "sgd": "torch_sgd_fp32",
        "adagrad": "torch_adagrad_fp32",
        "rmsprop": "torch_rmsprop_fp32",
        "adafactor": "hf_adafactor_fp32",
    }
    if name in ("adamw_bnb_8bit", "adamw_8bit"):
        try:
            if version("bitsandbytes") == "0.50.2":
                return "bnb_0.50.2_adamw8bit_fp32"
        except PackageNotFoundError:
            pass
    return profiles.get(name, f"unsupported: {name}")


def build_stream_adapter_plan(
    base: str,
    model_config: Any,
    tcfg: Any,
    *,
    dtype: str,
    quant: str,
    double_quant: bool,
    trust_remote_code: bool,
    on_cuda: bool,
    optimizer_backend: str | None = None,
    console: Any = None,
) -> StreamAdapterPlan:
    """Size the adapter that the real build attaches with ``get_peft_model``."""
    from accelerate import init_empty_weights
    from peft import TaskType, get_peft_model
    from torch.random import fork_rng

    from soup_cli.utils.layer_shard import QUANT_NF4
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes
    from soup_cli.utils.moe import resolve_moe_lora_targets
    from soup_cli.utils.peft_wiring import (
        apply_pre_lora_patches,
        build_lora_config,
        resolve_lora_target_modules,
    )

    use_lorafa = getattr(tcfg, "use_lorafa", False)
    with fork_rng(devices=[]):
        # The skeleton builder keeps model buffers real when they carry constructor values.
        model = build_meta_skeleton(
            base, dtype=dtype, quant=quant, double_quant=double_quant,
            trust_remote_code=trust_remote_code,
        )
        targets = resolve_lora_target_modules(model_config, tcfg.lora.target_modules, console)
        targets = resolve_moe_lora_targets(model, tcfg, targets, console=console)
        suffixes = tuple(sorted(quantised_layer_suffixes(model))) if quant == QUANT_NF4 else ()
        config = build_lora_config(tcfg.lora, target_modules=targets, task_type=TaskType.CAUSAL_LM)
        for param in model.parameters():
            param.requires_grad = False
        apply_pre_lora_patches(model, base)
        # PEFT mutates the config during target conversion. The real build needs the original.
        probe_config = deepcopy(config)
        with init_empty_weights(include_buffers=True):
            model = get_peft_model(model, probe_config)

    tensors = []
    for name, param in model.named_parameters():
        if not param.is_meta:
            raise ValueError(
                f"stream adapter preflight unexpectedly materialized parameter {name!r}"
            )
        if "lora_" not in name:
            if param.requires_grad:
                raise ValueError(f"stream adapter preflight cannot account for {name!r}")
            continue
        trainable = param.requires_grad and not (use_lorafa and ".lora_A." in name)
        # materialize_meta_adapters creates float32 adapters regardless of the base dtype.
        tensors.append(AdapterTensor(name, tuple(param.shape), "float32", trainable))

    profile = _optimizer_profile(tcfg, optimizer_backend=optimizer_backend)
    if profile == "peft_lorafa_fp32" and probe_config.target_parameters:
        profile = "unsupported: LoRA-FA on fused expert parameters"
    budget = estimate_adapter_budget(tensors, profile=profile, device="cuda" if on_cuda else "cpu")
    return StreamAdapterPlan(tuple(tensors), budget, config, suffixes, targets)


def try_build_stream_adapter_plan(
    base: str,
    model_config: Any,
    tcfg: Any,
    *,
    console: Any,
    **build_kwargs: Any,
) -> StreamAdapterPlan | None:
    """Build the plan, or return None after one notice when the probe itself fails.

    The preflight may refuse only on a computed "adapter does not fit" result. A
    probe error (skeleton, pre-LoRA patches, owner resolution) for any architecture
    skips the preflight; setup then continues exactly as it did without it.
    """
    try:
        return build_stream_adapter_plan(base, model_config, tcfg, console=console, **build_kwargs)
    except Exception as exc:  # noqa: BLE001 - fail open on any probe failure
        from soup_cli.utils.terminal import for_terminal

        console.print(
            "[yellow]Adapter memory preflight skipped: "
            f"{for_terminal(f'{type(exc).__name__}: {exc}')}[/]"
        )
        return None


def _validate_cuda_capacity(available_cuda_bytes: int) -> None:
    if type(available_cuda_bytes) is not int or available_cuda_bytes < 0:
        raise ValueError(
            "available CUDA memory must be a non-negative integer byte count; "
            f"got {available_cuda_bytes!r}"
        )


def check_stream_adapter_budget(
    plan: StreamAdapterPlan,
    *,
    available_cuda_bytes: int | None,
    console: Any,
    recommendations: Callable[[], str] | None = None,
) -> None:
    """Refuse an oversized persistent adapter budget. Passing leaves setup unchanged."""
    if available_cuda_bytes is None:
        return
    _validate_cuda_capacity(available_cuda_bytes)

    budget = plan.budget
    total = budget.total
    if total is None:
        # Only storage is unconditional when an unverified optimizer may change trainability.
        needed = budget.weights.cuda_bytes
        detail = f"adapter weights alone={needed} bytes; {budget.unsupported_reason}"
        console.print(
            "[yellow]Adapter optimizer budget unavailable: "
            f"{budget.unsupported_reason}. No adapter-fit verdict is available.[/]"
        )
    else:
        needed = total.cuda_bytes
        detail = (
            f"weights={budget.weights.cuda_bytes}, gradients={budget.gradients.cuda_bytes}, "
            f"optimizer={budget.optimizer.cuda_bytes} CUDA bytes; "
            f"CPU tensors={total.cpu_bytes} bytes"
        )
    if needed > available_cuda_bytes:
        alternatives = recommendations() if recommendations is not None else ""
        raise ValueError(
            "Adapter memory budget exceeded before checkpoint loading: "
            f"{budget.stored_parameters:,} adapter parameters; {detail}; "
            f"required={needed} bytes ({needed / 1e9:.3f} GB), "
            f"available={available_cuda_bytes} bytes ({available_cuda_bytes / 1e9:.3f} GB). "
            "Reduce LoRA targets or rank. The full training step needs additional memory."
            + alternatives
        )


def recommend_stream_adapter_targets(
    original: StreamAdapterPlan,
    base: str,
    model_config: Any,
    tcfg: Any,
    *,
    available_cuda_bytes: int,
    **build_kwargs: Any,
) -> str:
    """Return verified narrower target patterns with ranks and optimizer held fixed."""
    _validate_cuda_capacity(available_cuda_bytes)
    original_total = original.budget.total
    if original_total is None:
        return " No verified target alternative: optimizer budget is unknown."
    attention = (
        r".*\.(?:self_attn|attention)\."
        r"(?:q_proj|k_proj|v_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)"
    )
    shared = r".*\.(?:shared_expert|shared_experts)\.(?:gate_proj|up_proj|down_proj)"
    candidates = (
        ("attention only", attention),
        ("attention + shared expert", f"(?:{attention}|{shared})"),
    )
    messages = []
    for label, targets in candidates:
        settings = deepcopy(tcfg)
        settings.moe_lora = False
        settings.lora.target_modules = targets
        try:
            candidate = build_stream_adapter_plan(base, model_config, settings, **build_kwargs)
        except (ValueError, RuntimeError, TypeError, ImportError):
            # An unsupported alternative must leave the original refusal intact.
            continue
        owners = {tensor.name.rsplit(".lora_", 1)[0] for tensor in candidate.tensors}
        if not all(re.fullmatch(targets, owner) for owner in owners):
            continue
        if not any(re.fullmatch(attention, owner) for owner in owners):
            continue
        if targets != attention and not any(re.fullmatch(shared, owner) for owner in owners):
            continue
        budget = candidate.budget
        total = budget.total
        if (
            total is None
            or total.cuda_bytes > available_cuda_bytes
            or total.cuda_bytes >= original_total.cuda_bytes
        ):
            continue
        messages.append(
            f"\n{label}: {budget.stored_parameters:,} adapter parameters; "
            f"weights={budget.weights.cuda_bytes}, gradients={budget.gradients.cuda_bytes}, "
            f"optimizer={budget.optimizer.cuda_bytes}, total={total.cuda_bytes} CUDA bytes; "
            f"CPU tensors={total.cpu_bytes} bytes.\n"
            "Set training.moe_lora: false and training.lora.target_modules: "
            f"{json.dumps(targets)}; keep other settings unchanged."
        )
    if not messages:
        return " No smaller verified attention/shared-expert alternative fits the adapter budget."
    return "\nVerified adapter-budget alternatives (persistent adapter tensors only):" + "".join(
        messages
    )
