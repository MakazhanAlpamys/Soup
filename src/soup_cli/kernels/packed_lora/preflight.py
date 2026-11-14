# SPDX-License-Identifier: Apache-2.0
#
# Written for Soup. The memory plan it prices a run with is ``budget.py``; the checks it runs are
# the functions of ``support.py``.
"""What ``soup train`` checks, before anything is loaded, for a run with ``training.packed_lora``.

:func:`read_machine` reads the facts of this machine once (platform, Triton, the GPU's compute
capability and free memory, the free host RAM, the world size); :func:`preflight` takes them with
the model's ``transformers`` config and the run's ``SoupConfig`` and returns every reason the
kernels cannot take the run, and the memory plan they would run it with. Both are cheap and need
neither a model nor a dataset. The generic hardware-fit estimate (``utils/hardware_fit.py``)
prices a run that checkpoints its activations and builds its logits; the kernels do neither, so
for a run that uses them this plan, a fit to measurements of the kernels, is what is checked
against the card.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any, Optional

from . import budget, loader, support
from .policy import Policy

_MIB = 1 << 20


@dataclass(frozen=True)
class Machine:
    """What the run's machine offers. ``None`` where a figure could not be read (that check is
    skipped, not failed)."""

    system: str
    has_triton: bool
    compute_capability: Optional[tuple[int, int]]
    bf16_supported: bool
    world_size: int = 1
    free_ram_bytes: Optional[int] = None
    free_vram_bytes: Optional[int] = None
    total_vram_bytes: Optional[int] = None


@dataclass
class Preflight:
    """The verdict. ``problems``: why the kernels cannot take the run at all (the machine, the
    model, the settings; an unsupported run, which ``on_unsupported`` decides). ``fit_problems``:
    why they cannot take it *right now* (pinned host memory or GPU memory that is not free; the
    way out is to free memory, and ``soup train --allow-oom-attempt`` goes on anyway). Both empty:
    the run is taken. Also the ``plan`` (``None`` without a ``gpu_budget_mib``), the tokens a step
    can hold and the estimated peak of the step in bytes (the plan's, or the floor of a run
    without a budget)."""

    problems: list[str] = field(default_factory=list)
    fit_problems: list[str] = field(default_factory=list)
    plan: Optional[budget.Plan] = None
    tokens: int = 0
    estimate_bytes: int = 0
    host_bytes: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not self.fit_problems


def read_machine() -> Machine:
    """The facts of this machine. Imports torch (and psutil, when it is installed) here, not at
    module import."""
    capability = bf16 = free_vram = total_vram = None
    try:
        import torch

        if torch.cuda.is_available():
            capability = tuple(torch.cuda.get_device_capability(0))
            bf16 = bool(torch.cuda.is_bf16_supported())
            free_vram, total_vram = (int(v) for v in torch.cuda.mem_get_info(0))
    except (ImportError, RuntimeError, OSError):
        pass
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    except ValueError:
        world_size = 1
    return Machine(
        system=platform.system(),
        has_triton=loader.triton_available(),
        compute_capability=capability,
        bf16_supported=bool(bf16),
        world_size=world_size,
        free_ram_bytes=_free_ram_bytes(),
        free_vram_bytes=free_vram,
        total_vram_bytes=total_vram,
    )


def _free_ram_bytes() -> Optional[int]:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def read_model_config(base: str, trust_remote_code: bool = False) -> Any:
    """The base model's ``transformers`` config, or ``None`` when it cannot be read here (offline,
    gated, not a model): the model checks then wait for the load."""
    try:
        from transformers import AutoConfig

        from soup_cli.utils.run_provenance import base_revision_kwargs

        return AutoConfig.from_pretrained(
            base, trust_remote_code=trust_remote_code, **base_revision_kwargs(base))
    except Exception:  # noqa: BLE001 - every way a config can fail to load means "not readable"
        return None


def tokens_per_step(config: Any) -> Optional[int]:
    """The most tokens one step can hold (micro-batch x ``data.max_length``), or ``None`` for a
    batch size that is not a number yet (``auto``)."""
    batch = config.training.batch_size
    if isinstance(batch, bool) or not isinstance(batch, int):
        return None
    return batch * config.data.max_length


def _named(reasons: list[str]) -> list[str]:
    """Each reason starts with the setting it is about, as the schema's refusals do."""
    return [reason if reason.startswith("training.packed_lora") else
            f"training.packed_lora {reason}" for reason in reasons]


def preflight(config: Any, model_config: Any, machine: Machine, *,
              sharding: Optional[str] = None) -> Preflight:
    """Every reason the kernels cannot take this run on this machine, and the plan they would use.

    ``config``: the ``SoupConfig`` (its ``training.packed_lora`` block must be on);
    ``model_config``: the base model's ``transformers`` config, or ``None`` when it could not be
    read (the model checks then wait for the load). ``sharding`` names the FSDP or DeepSpeed
    setting of the run, if it has one (the kernels run on one GPU)."""
    policy = Policy.from_config(config)
    result = Preflight()
    problems = result.problems
    problems += _named(support.environment_problems(
        system=machine.system, has_triton=machine.has_triton,
        compute_capability=machine.compute_capability,
        bf16_supported=machine.bf16_supported, world_size=machine.world_size,
        sharding=sharding))
    if not problems:
        problems += _named(support.attention_signature_problems())
    if model_config is None:
        result.notes.append("the model's config could not be read; it is checked at the load")
        return result
    config_problems = _named(support.model_config_problems(model_config))
    problems += config_problems
    tokens = tokens_per_step(config)
    if tokens is None:
        problems += _named([
            "needs an explicit training.batch_size; 'auto' cannot be planned, because the "
            "memory plan is made for the batch."])
    if config_problems or tokens is None:
        return result
    result.tokens = tokens
    shape = budget.Shape.of_config(model_config, rank=max(1, config.training.lora.r))
    int8 = policy.int8_switch().enabled()
    acts = policy.activations if policy.activations != "auto" else None
    layer_bytes = shape.layer(tokens, int8)
    if policy.gpu_budget_bytes is not None:
        try:
            result.plan = budget.Plan(shape, policy.gpu_budget_bytes, tokens, int8, acts)
        except budget.BudgetTooSmallError as exc:
            problems.append(str(exc))
            return result
        result.estimate_bytes = int(result.plan.est_peak)
        result.host_bytes = result.plan.host_bytes()
        if machine.total_vram_bytes is not None and (
                policy.gpu_budget_bytes > machine.total_vram_bytes):
            problems.append(
                f"training.packed_lora.gpu_budget_mib={policy.gpu_budget_bytes // _MIB} is more "
                f"than the GPU has ({machine.total_vram_bytes // _MIB} MiB).")
        elif machine.free_vram_bytes is not None and (
                result.estimate_bytes > machine.free_vram_bytes):
            result.fit_problems.append(
                f"training.packed_lora.gpu_budget_mib={policy.gpu_budget_bytes // _MIB} plans a "
                f"peak of about {result.estimate_bytes // _MIB} MiB, but only "
                f"{machine.free_vram_bytes // _MIB} MiB of GPU memory is free right now. Lower "
                f"the budget or free the GPU.")
    else:
        # No budget: every table stays on the GPU; the activations that fit stay resident and
        # the rest go to pinned host memory (all layers' buffers are page-locked up front unless
        # activations: resident)
        floor = shape.peak(True, True, 0, tokens, int8)
        result.estimate_bytes = int(floor)
        if policy.activations != "resident":
            result.host_bytes = shape.layers * layer_bytes
    fits = result.fit_problems
    fits += _named(support.host_memory_problems(result.host_bytes, machine.free_ram_bytes))
    if result.plan is None:
        fits += _named(support.gpu_memory_problems(int(floor), machine.free_vram_bytes))
    return result
