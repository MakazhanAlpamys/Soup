"""Resolve real streamed adapter shapes and refuse impossible persistent budgets."""

import json
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Optional

from soup_cli.utils.adapter_budget import AdapterBudget, AdapterTensor, estimate_adapter_budget


@dataclass(frozen=True)
class StreamAdapterPlan:
    tensors: tuple[AdapterTensor, ...]
    budget: AdapterBudget
    lora_config: Any
    quant_suffixes: tuple[str, ...]


def _optimizer_profile(tcfg: Any) -> str:
    """Resolve only settings consumed by Soup's ordinary Trainer setup.

    Overrides are not silently treated as the YAML optimizer. Unsupported profiles
    still allow an adapter-weight lower-bound refusal, but never report a fit.
    """
    if getattr(tcfg, "use_galore", False):
        return "unsupported: GaLore"
    if getattr(tcfg, "loraplus_lr_ratio", None) is not None:
        return "unsupported: LoRA+"
    name = getattr(tcfg, "optimizer", "adamw_torch")
    if getattr(tcfg, "use_lorafa", False):
        # The full setup validates the allowed AdamW names. Do not bless a conflict.
        if name not in ("adamw_torch", "adamw_hf", "adamw_torch_fused"):
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
    console: Any = None,
) -> StreamAdapterPlan:
    """Use the same resolvers/patches as the real build, without materializing weights."""
    from accelerate import init_empty_weights
    from peft import TaskType, get_peft_model

    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes
    from soup_cli.utils.moe import resolve_moe_lora_targets
    from soup_cli.utils.peft_wiring import (
        apply_pre_lora_patches,
        build_lora_config,
        resolve_lora_target_modules,
    )

    with init_empty_weights(include_buffers=True):
        model = build_meta_skeleton(
            base, dtype=dtype, quant=quant, double_quant=double_quant,
            trust_remote_code=trust_remote_code,
        )
        suffixes = tuple(sorted(quantised_layer_suffixes(model))) if quant == "nf4" else ()
        moe_targets = resolve_moe_lora_targets(model, tcfg, None, console=console)
        targets = resolve_lora_target_modules(model_config, tcfg.lora.target_modules, console)
        if moe_targets:
            targets = moe_targets
        config = build_lora_config(tcfg.lora, target_modules=targets, task_type=TaskType.CAUSAL_LM)
        for param in model.parameters():
            param.requires_grad = False
        apply_pre_lora_patches(model, base)
        # PEFT mutates config during fused-MoE conversion. Keep the real build pristine.
        probe_config = deepcopy(config)
        model = get_peft_model(model, probe_config)
        if any(not param.is_meta for param in model.parameters()):
            raise ValueError("stream adapter preflight unexpectedly materialized model parameters")
        tensors = []
        for name, param in model.named_parameters():
            if "lora_" not in name:
                if param.requires_grad:
                    raise ValueError(f"stream adapter preflight cannot account for {name!r}")
                continue
            trainable = param.requires_grad
            if getattr(tcfg, "use_lorafa", False) and ".lora_A." in name:
                trainable = False
            # materialize_meta_adapters creates float32, regardless of the base dtype.
            tensors.append(AdapterTensor(name, tuple(param.shape), "float32", trainable))
    profile = _optimizer_profile(tcfg)
    if getattr(tcfg, "use_lorafa", False) and probe_config.target_parameters:
        profile = "unsupported: LoRA-FA on fused expert parameters"
    budget = estimate_adapter_budget(tensors, profile=profile, device="cuda" if on_cuda else "cpu")
    return StreamAdapterPlan(tuple(tensors), budget, config, suffixes)


def check_stream_adapter_budget(
    plan: StreamAdapterPlan, *, available_cuda_bytes: Optional[int], console: Any,
    recommendations: Optional[Callable[[], str]] = None,
) -> None:
    """Passing this check is not a promise that the entire training step fits."""
    budget = plan.budget
    total = budget.total
    if total is None:
        # Only storage is unconditional when an optimizer may alter trainability.
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
        console.print(
            f"[dim]Adapter persistent tensors: {budget.stored_parameters:,} parameters; "
            f"{detail}. Excludes base buffers, activations and optimizer temporaries.[/]"
        )
    if available_cuda_bytes is not None:
        if available_cuda_bytes < 0:
            raise ValueError("available CUDA memory must be non-negative")
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
    original: StreamAdapterPlan, base: str, model_config: Any, tcfg: Any,
    *, available_cuda_bytes: int, **build_kwargs: Any,
) -> str:
    """Rebuild two narrow meta candidates, keeping ranks and optimizer unchanged.

    Regexes constrain owners as well as projection names: bare gate_proj would
    also select every routed expert. Never change the user's configuration.
    """
    if original.budget.total is None:
        return " No verified target alternative: optimizer budget is unknown."
    attention = (
        r".*\.(?:self_attn|attention)\."
        r"(?:q_proj|k_proj|v_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)"
    )
    shared = r".*\.(?:shared_expert|shared_experts)\.(?:gate_proj|up_proj|down_proj)"
    candidates = (("attention only", attention),
                  ("attention + shared expert", f"(?:{attention}|{shared})"))
    messages = []
    for label, targets in candidates:
        settings = deepcopy(tcfg)
        settings.moe_lora = False  # Otherwise the resolver overrides explicit targets.
        settings.lora.target_modules = targets
        try:
            candidate = build_stream_adapter_plan(base, model_config, settings, **build_kwargs)
        except (ValueError, RuntimeError, TypeError, ImportError):
            # An unsupported candidate must not obscure the original budget refusal.
            continue
        if not any(".self_attn." in t.name or ".attention." in t.name
                   for t in candidate.tensors):
            continue
        if label == "attention + shared expert" and not any(
            ".shared_expert." in t.name or ".shared_experts." in t.name
            for t in candidate.tensors
        ):
            continue
        budget = candidate.budget
        if (budget.total is None or budget.total.cuda_bytes > available_cuda_bytes
                or budget.total.cuda_bytes >= original.budget.total.cuda_bytes):
            continue
        messages.append(
            f"\n{label}: {budget.stored_parameters:,} adapter parameters; "
            f"weights={budget.weights.cuda_bytes}, gradients={budget.gradients.cuda_bytes}, "
            f"optimizer={budget.optimizer.cuda_bytes}, total={budget.total.cuda_bytes} CUDA bytes; "
            f"CPU tensors={budget.total.cpu_bytes} bytes.\n"
            "Set training.moe_lora: false and training.lora.target_modules: "
            f"{json.dumps(targets)}; keep other settings unchanged."
        )
    if not messages:
        return " No smaller verified attention/shared-expert alternative fits the adapter budget."
    return ("\nVerified adapter-budget alternatives (not a full-training fit or quality guarantee):"
            + "".join(messages))
