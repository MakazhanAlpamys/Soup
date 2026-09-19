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
    from soup_cli.utils.peft_wiring import build_peft_config_spec

    return build_peft_config_spec(
        lora_cfg,
        target_modules=target_modules,
        task_type=task_type,
    )


def instantiate_peft_config(spec: dict[str, Any]) -> Any:
    """Instantiate the peft config from a spec dict (lazy import).

    Returns ``peft.PeftConfig`` (return type is ``Any`` because ``peft`` is a
    lazy import and cannot be referenced at module scope).
    """
    import peft  # lazy

    cls = getattr(peft, spec["peft_cls"])
    return cls(**spec["init_kwargs"])
