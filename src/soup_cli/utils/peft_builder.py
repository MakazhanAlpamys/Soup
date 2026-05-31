"""PEFT config builder — unifies LoRA / DoRA / VeRA / OLoRA construction.

Returns an intermediate spec dict so trainers can import the right class and
instantiate it without every trainer needing to replicate the branching logic.
"""

from __future__ import annotations

from typing import Any

from soup_cli.config.schema import LoraConfig as SchemaLoraConfig


def build_peft_config(
    lora_cfg: SchemaLoraConfig,
    target_modules: "str | list[str]",
    task_type: str,
) -> dict[str, Any]:
    """Build a peft config spec from Soup's schema LoraConfig.

    Returns:
        Dict with keys:
        - ``peft_cls``: str — class name to import from ``peft`` (``LoraConfig``
          or ``VeraConfig``).
        - ``init_kwargs``: dict — kwargs to pass to the constructor.

    Trainers can use this spec to instantiate the right peft config without
    duplicating the branching logic for DoRA / VeRA / OLoRA.
    """
    if lora_cfg.use_vera:
        return {
            "peft_cls": "VeraConfig",
            "init_kwargs": {
                "r": lora_cfg.r,
                "target_modules": target_modules,
                "task_type": task_type,
                "vera_dropout": lora_cfg.dropout,
                "bias": "none",
            },
        }

    init_kwargs: dict = {
        "r": lora_cfg.r,
        "lora_alpha": lora_cfg.alpha,
        "lora_dropout": lora_cfg.dropout,
        "target_modules": target_modules,
        "task_type": task_type,
        "bias": "none",
        "use_dora": lora_cfg.use_dora,
        "use_rslora": lora_cfg.use_rslora,
    }

    # v0.39.0 Part C — per-pattern rank/alpha (peft natively supports these)
    if lora_cfg.rank_pattern:
        init_kwargs["rank_pattern"] = dict(lora_cfg.rank_pattern)
    if lora_cfg.alpha_pattern:
        init_kwargs["alpha_pattern"] = dict(lora_cfg.alpha_pattern)

    # init_strategy is the canonical source; use_olora is back-compat (validator
    # aligns init_strategy='olora' when use_olora=True).
    if lora_cfg.init_strategy in ("pissa", "olora"):
        init_kwargs["init_lora_weights"] = lora_cfg.init_strategy
    elif lora_cfg.use_olora:
        # Defensive fallback if init_strategy alignment was bypassed.
        init_kwargs["init_lora_weights"] = "olora"

    return {
        "peft_cls": "LoraConfig",
        "init_kwargs": init_kwargs,
    }


def instantiate_peft_config(spec: dict[str, Any]) -> Any:
    """Instantiate the peft config from a spec dict (lazy import).

    Returns ``peft.PeftConfig`` (return type is ``Any`` because ``peft`` is a
    lazy import and cannot be referenced at module scope).
    """
    import peft  # lazy

    cls = getattr(peft, spec["peft_cls"])
    return cls(**spec["init_kwargs"])
