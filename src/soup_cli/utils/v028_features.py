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

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from rich.console import Console

    from soup_cli.config.schema import TrainingConfig


def apply_v028_speed_memory(
    *,
    model: Any,
    tcfg: "TrainingConfig",
    base_model: str,
    console: Optional["Console"] = None,
    device: str = "cpu",
    backend: str = "transformers",
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
        recipe = getattr(tcfg, "fp8_recipe", "tensorwise")
        try:
            from soup_cli.utils.fp8 import apply_fp8_training
            ok = bool(apply_fp8_training(model, recipe=recipe))
        except Exception:  # noqa: BLE001
            ok = False
        applied["fp8"] = ok
        if ok:
            _say(f"FP8 training enabled (Float8Linear, recipe={recipe})")
        else:
            _say(
                "FP8 training: torchao.float8 unavailable or no "
                "compatible linears", style="yellow",
            )

    # --- Kernel auto-compose -------------------------------------------------
    if getattr(tcfg, "kernel_auto_compose", False):
        picked_name = _bench_and_pick_kernel(
            model=model, device=device, backend=backend,
        )
        if picked_name is None:
            _say(
                "Kernel auto-compose: benchmarking unavailable on this host",
                style="yellow",
            )
        else:
            applied["kernel_auto_compose"] = True
            _say(f"Kernel auto-compose picked: {picked_name}")

    return applied


def _bench_and_pick_kernel(
    *, model: Any, device: str, backend: str,
) -> Optional[str]:
    """v0.35.0 #45 — benchmark candidate kernel combos and return the
    picked combo's name. Returns ``None`` on any benchmark / pick failure
    so the caller can degrade gracefully (no kernel_auto_compose flag set).
    """
    try:
        from soup_cli.utils.kernel_picker import (
            benchmark_kernel_combos,
            enumerate_kernel_combos,
            pick_best_kernel,
        )
        candidates = enumerate_kernel_combos(backend=backend, device=device)
        # On CPU / unsloth / mlx the candidate list is just [baseline]; no
        # benchmark needed — picker would raise on all-None times. Skip.
        if len(candidates) <= 1:
            return None
        timed = benchmark_kernel_combos(
            model=model, candidates=candidates, device=device,
        )
        picked = pick_best_kernel(timed)
        # Picker returns either a dict (current shape) or an object with
        # ``.name`` (legacy / namespace shape) — handle both defensively.
        if isinstance(picked, dict):
            return str(picked.get("name", "unknown"))
        return str(getattr(picked, "name", "unknown"))
    except Exception:  # noqa: BLE001 — picker is best-effort
        return None


@contextlib.contextmanager
def activation_offloading_context(
    tcfg: "TrainingConfig", output_dir: str,
) -> Iterator[None]:
    """Wrap a trainer's ``trainer.train()`` call with activation offloading.

    Centralises the cwd-containment guard for ``activation_offloading="disk"``
    (defence-in-depth — caller's ``cfg.output`` is also validated upstream)
    and the ``offload_context`` setup. ``None`` and ``"cpu"`` modes pass
    through to ``offload_context`` directly.

    Raises ``ValueError`` when ``activation_offloading="disk"`` and
    ``output_dir`` is outside the current working directory.
    """
    from soup_cli.utils.activation_offload import offload_context
    from soup_cli.utils.paths import is_under_cwd

    save_dir: Optional[str] = None
    mode = getattr(tcfg, "activation_offloading", None)
    if mode == "disk":
        if not is_under_cwd(output_dir):
            # Reduce to basename so $HOME / absolute paths don't leak (matches
            # the v0.34.0 crash.py policy).
            import os as _os

            raise ValueError(
                "activation_offloading='disk' requires the training output "
                "dir to be under the current working directory; got "
                f"basename={_os.path.basename(output_dir)!r}"
            )
        save_dir = str(Path(output_dir) / "_activation_offload")
    with offload_context(mode, save_dir=save_dir):
        yield


def supports_v028_features(task: str) -> bool:
    """Tasks where v0.28.0 speed/memory wiring has been ported.

    Every task that calls :func:`apply_v028_speed_memory` should be listed
    here so config validation can advise users on tasks that would silently
    no-op. v0.35.0 (#60) extends coverage from {sft, dpo, pretrain} to
    every transformer-backend trainer.
    """
    return task in {
        "sft",
        "dpo",
        "pretrain",
        "grpo",
        "kto",
        "orpo",
        "simpo",
        "ipo",
        "bco",
        "preference",
        "ppo",
        "reward_model",
        "embedding",
    }


def warn_unsupported_features(
    tcfg: "TrainingConfig", task: str,
) -> Optional[str]:
    """Return a human warning if non-v0.28.0-wired tasks set v0.28.0 flags.

    Returns None when nothing to warn about. v0.35.0 #60 expanded coverage
    to every transformer-backend trainer; this helper now only fires for
    truly-unsupported tasks (e.g. a hypothetical future task or an MLX
    backend route).
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
        f"v0.28.0 speed/memory features {issues} are not wired for "
        f"task={task!r}. Flags will be silently ignored. See "
        "supports_v028_features() for the current supported task list."
    )
