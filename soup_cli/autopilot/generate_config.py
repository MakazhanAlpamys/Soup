"""Autopilot YAML config generator."""

from __future__ import annotations

from pathlib import Path

import yaml

from soup_cli.autopilot.analyzer import (
    analyze_dataset,
    analyze_hardware,
    analyze_model,
)
from soup_cli.autopilot.decisions import (
    decide_batch_size,
    decide_epochs,
    decide_lr,
    decide_max_length,
    decide_peft,
    decide_performance_flags,
    decide_quantization,
    decide_task,
)
from soup_cli.config.schema import (
    DataConfig,
    LoraConfig,
    SoupConfig,
    TrainingConfig,
)


def build_soup_config(
    model: str,
    data_path: str,
    goal: str,
    vram_gb: float | None = None,
) -> SoupConfig:
    """Run the full autopilot decision pipeline and return a ``SoupConfig``."""
    dataset_profile = analyze_dataset(data_path)
    model_profile = analyze_model(model)
    hardware_profile = analyze_hardware()

    target_vram = vram_gb if vram_gb is not None else max(hardware_profile.vram_gb, 8.0)

    task = decide_task(goal, dataset_profile)
    quantization = decide_quantization(
        model_params_b=model_profile.params_b, vram_gb=target_vram
    )
    peft = decide_peft(
        data_size=dataset_profile.samples,
        model_size_b=model_profile.params_b,
        vram_gb=target_vram,
    )
    max_length = decide_max_length(
        p95_tokens=dataset_profile.p95_tokens,
        model_context=model_profile.context,
    )
    batch_size, grad_accum = decide_batch_size(
        model_size_b=model_profile.params_b,
        vram_gb=target_vram,
        max_length=max_length,
        quantization=quantization,
    )
    lr = decide_lr(rank=peft["r"], quantization=quantization)
    epochs = decide_epochs(dataset_profile.samples)
    vram_headroom_gb = max(0.0, target_vram - model_profile.params_b * 2.0)
    perf_flags = decide_performance_flags(
        gpu_name=hardware_profile.gpu_name,
        compute_capability=hardware_profile.compute_capability,
        max_length=max_length,
        vram_headroom_gb=vram_headroom_gb,
    )

    data_format = dataset_profile.format
    if data_format == "unknown":
        data_format = "auto"

    return SoupConfig(
        base=model,
        task=task,
        data=DataConfig(
            train=data_path,
            format=data_format,
            max_length=max_length,
        ),
        training=TrainingConfig(
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            quantization=quantization,
            lora=LoraConfig(
                r=peft["r"],
                alpha=peft["alpha"],
                target_modules="auto",
                use_dora=peft["use_dora"],
            ),
            use_flash_attn=perf_flags["use_flash_attn"],
            use_liger=perf_flags["use_liger"],
            gradient_checkpointing=perf_flags["gradient_checkpointing"],
            forgetting_detection=True,
            checkpoint_intelligence=True,
            early_stop_on_regression=True,
        ),
        output="./output",
    )


def write_yaml(config: SoupConfig, path: Path) -> None:
    """Dump a ``SoupConfig`` to YAML."""
    path = Path(path)
    data = config.model_dump(mode="json", exclude_none=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def generate_config(
    base: str,
    data_path: str,
    decisions: dict,
    output_path: Path | str,
) -> Path:
    """Render an autopilot decisions dict to a YAML config file.

    The ``decisions`` dict mirrors what ``build_soup_config`` produces but
    with a flat shape — useful for testing and for v0.32.0 callers that
    pre-compute decisions outside the analyzer pipeline.
    """
    from soup_cli.utils.paths import is_under_cwd

    output = Path(output_path)
    if not is_under_cwd(output):
        raise ValueError(f"output_path must stay under cwd: {output}")
    output_field = decisions.get("output", "./output")
    if not is_under_cwd(Path(output_field)):
        raise ValueError(
            f"decisions['output'] must stay under cwd: {output_field}"
        )
    perf = decisions.get("perf", {})
    lora = decisions.get("lora", {})
    training_kwargs = {
        "epochs": decisions["epochs"],
        "lr": decisions["lr"],
        "batch_size": decisions["batch_size"],
        "gradient_accumulation_steps": decisions["grad_accum"],
        "quantization": decisions["quantization"],
        "lora": LoraConfig(
            r=lora.get("r", 16),
            alpha=lora.get("alpha", 32),
            target_modules="auto",
            use_dora=lora.get("use_dora", False),
        ),
        "use_flash_attn": perf.get("use_flash_attn", False),
        "use_liger": perf.get("use_liger", False),
        "gradient_checkpointing": perf.get("gradient_checkpointing", False),
        "warmup_auto": bool(decisions.get("warmup_auto", False)),
        "auto_mixed_precision": bool(decisions.get("mixed_precision") is not None),
    }
    cfg = SoupConfig(
        base=base,
        task=decisions["task"],
        data=DataConfig(
            train=data_path,
            format=decisions.get("format", "auto"),
            max_length=decisions["max_length"],
        ),
        training=TrainingConfig(**training_kwargs),
        output=decisions.get("output", "./output"),
    )
    write_yaml(cfg, output)
    return output
