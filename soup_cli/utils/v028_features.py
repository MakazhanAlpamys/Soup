"""v0.28.0 speed/memory feature application — extracted for multi-trainer reuse.

The original v0.28.0 release wired Cut Cross-Entropy, FP8, kernel-auto-compose
into ``SFTTrainerWrapper`` only and gated other trainers via a
``model_validator`` to fail-fast at config-load. v0.33.0 (#43) drops that
gate and extracts the apply logic here so any trainer wrapper can call it
in two lines.

Activation-offloading is NOT included here — its scope is the entire
``trainer.train()`` call (it wraps in a context manager), so each trainer
wires it inline. CCE / FP8 / kernel-pick are pre-train one-shots and fit
this single helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from rich.console import Console

    from soup_cli.config.schema import TrainingConfig


def apply_v028_speed_memory(
    *,
    model: Any,
    tcfg: "TrainingConfig",
    base_model: str,
    console: Optional["Console"] = None,
) -> dict[str, bool]:
    """Apply Cut-CE / FP8 / kernel-auto-compose features to ``model``.

    Returns a dict ``{feature_name: applied}`` so the caller can log the
    decisions for the run record. Each feature degrades silently to a
    yellow advisory if the underlying lib isn't available — never crashes
    the training kick-off.
    """
    applied: dict[str, bool] = {
        "cut_ce": False,
        "fp8": False,
        "kernel_auto_compose": False,
    }

    def _say(text: str, style: str = "green") -> None:
        if console is None:
            return
        console.print(f"[{style}]{text}[/]")

    # --- Cut Cross-Entropy ---------------------------------------------------
    if getattr(tcfg, "use_cut_ce", False):
        try:
            from soup_cli.utils.cut_ce import apply_cut_ce
            ok = bool(apply_cut_ce(base_model))
        except Exception:  # noqa: BLE001 — degrade gracefully
            ok = False
        applied["cut_ce"] = ok
        if ok:
            _say("Cut Cross-Entropy enabled (chunked CCE kernel)")
        else:
            _say(
                "Cut Cross-Entropy: no matching architecture or "
                "cut_cross_entropy not installed", style="yellow",
            )

    # --- FP8 training --------------------------------------------------------
    if getattr(tcfg, "quantization_aware", None) == "fp8":
        try:
            from soup_cli.utils.fp8 import apply_fp8_training
            ok = bool(apply_fp8_training(model))
        except Exception:  # noqa: BLE001
            ok = False
        applied["fp8"] = ok
        if ok:
            _say("FP8 training enabled (Float8Linear)")
        else:
            _say(
                "FP8 training: torchao.float8 unavailable or no "
                "compatible linears", style="yellow",
            )

    # --- Kernel auto-compose -------------------------------------------------
    if getattr(tcfg, "kernel_auto_compose", False):
        try:
            from soup_cli.utils.kernel_picker import (
                enumerate_candidates,
                pick_best_kernel,
            )
            candidates = list(enumerate_candidates())
            picked = pick_best_kernel(candidates)
            applied["kernel_auto_compose"] = True
            _say(f"Kernel auto-compose picked: {picked.name}")
        except Exception:  # noqa: BLE001 — picker may be benchmark-blocked
            _say(
                "Kernel auto-compose: benchmarking unavailable on this host",
                style="yellow",
            )

    return applied


def supports_v028_features(task: str) -> bool:
    """Tasks where v0.28.0 speed/memory wiring has been ported.

    Every task that calls :func:`apply_v028_speed_memory` should be listed
    here so config validation can advise users on tasks that would silently
    no-op.
    """
    return task in {"sft", "dpo", "pretrain"}


def warn_unsupported_features(
    tcfg: "TrainingConfig", task: str,
) -> Optional[str]:
    """Return a human warning if non-v0.28.0-wired tasks set v0.28.0 flags.

    Returns None when nothing to warn about.
    """
    if supports_v028_features(task):
        return None
    issues: list[str] = []
    if getattr(tcfg, "use_cut_ce", False):
        issues.append("use_cut_ce")
    if getattr(tcfg, "quantization_aware", None) == "fp8":
        issues.append('quantization_aware="fp8"')
    if getattr(tcfg, "kernel_auto_compose", False):
        issues.append("kernel_auto_compose")
    if getattr(tcfg, "activation_offloading", None) is not None:
        issues.append("activation_offloading")
    if not issues:
        return None
    return (
        f"v0.28.0 speed/memory features {issues} are not yet wired for "
        f"task={task!r} (live in: sft, dpo, pretrain). Flags will be "
        "silently ignored. Multi-trainer expansion is tracked in "
        "release notes."
    )
