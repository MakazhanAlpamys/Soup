"""DeepSpeed configuration templates for multi-GPU training."""

import copy
import json
import os
import tempfile
from typing import Any

# ZeRO Stage 2: splits optimizer states + gradients across GPUs
ZERO_STAGE_2 = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}

# ZeRO Stage 3: splits model params + optimizer + gradients across GPUs
ZERO_STAGE_3 = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}

# ZeRO Stage 2 with CPU offload (for memory-constrained setups)
ZERO_STAGE_2_OFFLOAD = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


# ZeRO Stage 3 with CPU parameter offload — the configuration a user whose
# weights do not fit in VRAM actually wants. Until this preset existed the only
# offload config here was stage 2, optimizer-only, so the DeepSpeed comparison in
# benchmarks/gate-h100-validation.md (STEP 3) had to be run from hand-written
# JSON. Measured there on one H100 (Llama-3.1-8B, bf16, LoRA r=8, 256 steps):
# 21.65 tok/s at a 38,135 MiB peak.
#
# `offload_optimizer` stays "none" deliberately. Turning it on makes DeepSpeed
# JIT-build its `cpu_adam` op, which needs a matching CUDA toolkit; on a box with
# no `nvcc` that fails with `CUDAMismatchException` and then
# `'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'`. Users who do have a
# toolkit can flip it in a copy of this JSON.
#
# The stage3_* knobs are left at the stage-3 defaults. Tightening all four
# (max_live_parameters/max_reuse_distance 1e9 -> 1e7, param_persistence_threshold
# auto -> 0, prefetch_bucket_size auto -> 5e6) was measured on the same box at
# 5.6% less peak VRAM for 12% less throughput — a trade worth making by hand, not
# by default.
ZERO_STAGE_3_OFFLOAD = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": int(1e9),
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": int(1e9),
        "stage3_max_reuse_distance": int(1e9),
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


# ZeRO++ (v0.27.0): stage-3 base + hierarchical partitioning + quantized
# weights/gradients. Reduces inter-node communication 4-8x on 8+ GPUs.
#
# Two of these values are placeholders that :func:`resolve_deepspeed_config`
# rewrites before the config reaches DeepSpeed, and both were measured wrong on
# real hardware in #336:
#
# * ``zero_hpz_partition_size: 8`` is a hardcoded guess. DeepSpeed requires the
#   world size to be divisible by it, so on the 4xH100 box it was invalid.
#   The resolver sets it to the actual world size.
# * ``zero_quantized_weights`` / ``zero_quantized_gradients`` are the fp16 CUDA
#   quantiser. Against the ``bf16`` this same file enables, the dequantised
#   all-gather comes back ``c10::Half`` and meets a ``c10::BFloat16``
#   activation inside ``deepspeed/runtime/zero/linear.py``, which raises
#   ``expected mat1 and mat2 to have the same dtype``. The resolver turns them
#   off for a bf16 run and says so; hierarchical partitioning — the half of
#   ZeRO++ that is dtype-agnostic — stays on.
ZERO_PLUS_PLUS = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": int(1e9),
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": int(1e9),
        "stage3_max_reuse_distance": int(1e9),
        "stage3_gather_16bit_weights_on_model_save": True,
        # ZeRO++ specifics
        "zero_hpz_partition_size": 8,
        "zero_quantized_weights": True,
        "zero_quantized_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


CONFIGS = {
    "zero2": ZERO_STAGE_2,
    "zero3": ZERO_STAGE_3,
    "zero2_offload": ZERO_STAGE_2_OFFLOAD,
    "zero3_offload": ZERO_STAGE_3_OFFLOAD,
    "zero++": ZERO_PLUS_PLUS,
    "zero_pp": ZERO_PLUS_PLUS,
}


def get_deepspeed_config(stage: str = "zero2") -> dict:
    """Get a DeepSpeed config dict by name."""
    if stage not in CONFIGS:
        raise ValueError(f"Unknown DeepSpeed config: {stage}. Options: {', '.join(CONFIGS.keys())}")
    return copy.deepcopy(CONFIGS[stage])


def resolve_world_size() -> int:
    """How many processes this run will actually have.

    Inside an ``accelerate launch`` / ``torchrun`` rank ``WORLD_SIZE`` is the
    authority and is what DeepSpeed will divide by, so it wins over the visible
    device count: ``--gpus 2`` on an 8-GPU box must not produce a partition
    size of 8. Outside a launcher there is no world yet, so the visible device
    count is the best available estimate.

    Returns 0 when neither is known, which the resolver reads as "unknown".
    """
    world = _env_int("WORLD_SIZE")
    if world > 0:
        return world
    return int(detect_multi_gpu().get("gpu_count", 0) or 0)


def _env_int(name: str) -> int:
    """Read a positive integer from the environment; 0 when absent or junk."""
    raw = os.environ.get(name)
    if raw is None:
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def resolve_hpz_partition_size(world_size: int) -> int:
    """Pick a ``zero_hpz_partition_size`` that DeepSpeed will accept.

    DeepSpeed requires ``world_size %% hpz == 0``. The *useful* value is the
    number of ranks per node — hierarchical partitioning exists to keep the
    secondary shard inside one node's fast interconnect — so ``LOCAL_WORLD_SIZE``
    is preferred when the launcher exported it and it divides the world.
    Otherwise the world size itself, which always divides and on a single node
    is the same number. When the world is unknown, 1: it is valid for every
    world, and disables hierarchical partitioning rather than failing
    DeepSpeed's divisibility check with a guess (#336).
    """
    world = max(int(world_size or 0), 0)
    if world <= 0:
        return 1
    local = _env_int("LOCAL_WORLD_SIZE")
    if 0 < local <= world and world % local == 0:
        return local
    return world


def resolve_deepspeed_config(
    config: dict, *, gpu_count: int | None = None
) -> tuple[dict, list[str]]:
    """Adapt a preset to the run it is about to be used for.

    Returns a *new* config plus human-readable notes describing every change,
    so the caller can print them rather than silently altering the run.

    Only ZeRO++ carries run-dependent keys today; every other preset comes back
    unchanged with no notes (#336).
    """
    resolved = copy.deepcopy(config)
    zero: dict[str, Any] = resolved.get("zero_optimization") or {}
    notes: list[str] = []

    if "zero_hpz_partition_size" in zero:
        world = gpu_count if gpu_count is not None else resolve_world_size()
        world = max(int(world or 0), 0)
        partition = resolve_hpz_partition_size(world)
        if partition != zero["zero_hpz_partition_size"]:
            notes.append(
                f"zero_hpz_partition_size {zero['zero_hpz_partition_size']} -> {partition} "
                f"(the preset hardcoded a value; DeepSpeed requires the world size "
                f"[{world or 'unknown'}] to be divisible by it)"
            )
        zero["zero_hpz_partition_size"] = partition

    quant_keys = [
        key
        for key in ("zero_quantized_weights", "zero_quantized_gradients")
        if zero.get(key)
    ]
    if quant_keys:
        bf16_on = bool((resolved.get("bf16") or {}).get("enabled"))
        if bf16_on:
            for key in quant_keys:
                zero[key] = False
            notes.append(
                "zero_quantized_weights/zero_quantized_gradients disabled: DeepSpeed's "
                "quantiser is the fp16 kernel and this run is bf16, so the dequantised "
                "all-gather returns Half and meets a BFloat16 activation "
                "(`expected mat1 and mat2 to have the same dtype`). Hierarchical "
                "partitioning stays enabled."
            )

    if zero:
        resolved["zero_optimization"] = zero
    return resolved, notes


def write_deepspeed_config(stage: str = "zero2", gpu_count: int | None = None) -> str:
    """Write a DeepSpeed config to a temp file and return the path.

    The preset is resolved against the actual run first (#336); any adjustment
    is printed rather than applied silently.
    """
    config, notes = resolve_deepspeed_config(get_deepspeed_config(stage), gpu_count=gpu_count)
    if notes:
        from rich.console import Console
        from rich.markup import escape

        console = Console()
        for note in notes:
            console.print(f"[yellow]DeepSpeed {stage}:[/] {escape(note)}")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="ds_config_", delete=False
    )
    json.dump(config, tmp, indent=2)
    tmp.close()
    return tmp.name


def prune_empty_param_groups(optimizer) -> int:
    """Drop parameter groups that hold no parameters. Returns how many went.

    HF's ``create_optimizer`` always emits two groups — decay and no-decay.
    With LoRA every trainable tensor is a 2-D ``lora_A``/``lora_B`` weight, so
    the no-decay group is **empty** (measured ``[192, 0]``). DeepSpeed drops
    that group when it builds its own partitions, but the LR scheduler was
    already constructed over both, so the first ``lr_scheduler.step()`` zips 1
    param group against 2 ``base_lrs`` and torch >= 2.13's ``strict=True``
    raises ``zip() argument 2 is longer than argument 1``.

    Dropping the empty group is mathematically a no-op — it owns no tensors —
    and it has to happen before the scheduler is built, which is why this runs
    inside ``create_optimizer`` rather than after.

    Mutates ``optimizer.param_groups`` in place: the scheduler holds a
    reference to that same list.

    If *every* group is empty the list is left alone. Pruning to zero groups
    would replace a clear "nothing is trainable" failure with an obscure one
    inside DeepSpeed.
    """
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        return 0
    kept = [group for group in groups if group.get("params")]
    if not kept or len(kept) == len(groups):
        return 0
    dropped = len(groups) - len(kept)
    groups[:] = kept
    return dropped


def attach_empty_param_group_guard(trainer) -> bool:
    """Make ``trainer.create_optimizer`` hand DeepSpeed only non-empty groups.

    Returns ``True`` when the guard was installed, ``False`` when this trainer
    already carries one (idempotent — ``setup()`` may run more than once).

    See :func:`prune_empty_param_groups` for why this is needed and why it must
    happen at optimizer-construction time. Full fine-tuning populates both
    groups, so nothing is pruned there and the path that already worked is
    untouched (#336).

    Scope: currently wired in ``trainer/sft.py`` only, which is where #336 was
    measured. Every other wrapper that accepts a ``deepspeed_config`` has the
    same exposure and needs the same two-line call after its trainer is built.
    """
    if getattr(trainer, "_soup_empty_group_guard", False):
        return False

    original = trainer.create_optimizer

    def create_optimizer():
        optimizer = original()
        dropped = prune_empty_param_groups(optimizer)
        if dropped:
            from rich.console import Console

            Console().print(
                f"[green]DeepSpeed:[/] dropped {dropped} empty optimizer parameter "
                "group(s) before handing the optimizer over (LoRA leaves the "
                "no-decay group empty; DeepSpeed and the LR scheduler would "
                "otherwise disagree on how many groups there are)"
            )
        return optimizer

    trainer.create_optimizer = create_optimizer
    trainer._soup_empty_group_guard = True
    return True


def detect_multi_gpu() -> dict:
    """Detect multiple GPUs and return info."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu_count": 0, "gpus": []}

        gpu_count = torch.cuda.device_count()
        gpus = []
        for idx in range(gpu_count):
            props = torch.cuda.get_device_properties(idx)
            gpus.append({
                "index": idx,
                "name": props.name,
                "memory_gb": props.total_memory / (1024 ** 3),
            })

        return {"gpu_count": gpu_count, "gpus": gpus}
    except ImportError:
        return {"gpu_count": 0, "gpus": []}
