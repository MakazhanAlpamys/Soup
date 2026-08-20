"""Accelerate / torchrun launcher wrapper helpers (v0.27.0).

v0.27.0 is advisory only: when the user passes --gpus N>1 but soup is not
running under a launcher, we print the exact `accelerate launch` command to
run. Auto-reexec is deferred to v0.27.1 to keep the blast radius small.
"""

from __future__ import annotations

import os
import shlex
import sys
from typing import Sequence

VALID_MIXED_PRECISION = ("no", "fp16", "bf16", "fp8")
MAX_NUM_MACHINES = 256  # sanity cap, consistent with --gpus MAX_GPU_COUNT=128


def is_in_distributed() -> bool:
    """Return True if the current process was launched by torchrun / accelerate.

    Detection is based on the presence of either the torch.distributed standard
    variables (RANK + WORLD_SIZE) or any Accelerate-specific marker.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return True
    return any(
        key in os.environ
        for key in (
            "ACCELERATE_MIXED_PRECISION",
            "ACCELERATE_USE_DEEPSPEED",
            "ACCELERATE_USE_FSDP",
        )
    )


def build_accelerate_argv(
    num_processes: int,
    script_args: Sequence[str],
    mixed_precision: str | None = None,
    num_machines: int = 1,
) -> list[str]:
    """Build argv that wraps ``script_args`` with ``accelerate launch``.

    When ``num_processes == 1`` the launcher wrapper is skipped and
    ``script_args`` is returned unchanged.

    Args:
        num_processes: Total number of processes. Must be >= 1.
        script_args: The command to run (e.g. ``["soup", "train"]``).
        mixed_precision: One of ``no``, ``fp16``, ``bf16``, ``fp8``.
        num_machines: Number of nodes. Defaults to 1.

    Raises:
        ValueError: On invalid ``num_processes`` or ``mixed_precision``.
    """
    if not isinstance(num_processes, int) or num_processes < 1:
        raise ValueError(
            f"num_processes must be a positive integer (got {num_processes!r})."
        )
    if mixed_precision is not None and mixed_precision not in VALID_MIXED_PRECISION:
        raise ValueError(
            f"Invalid mixed_precision: {mixed_precision!r}. "
            f"Options: {', '.join(VALID_MIXED_PRECISION)}."
        )
    if num_machines < 1 or num_machines > MAX_NUM_MACHINES:
        raise ValueError(
            f"num_machines must be in [1, {MAX_NUM_MACHINES}] (got {num_machines})."
        )

    script_list = list(script_args)
    if num_processes == 1:
        return script_list

    argv: list[str] = ["accelerate", "launch", "--num_processes", str(num_processes)]
    if num_machines > 1:
        argv.extend(["--num_machines", str(num_machines)])
    if mixed_precision is not None:
        argv.extend(["--mixed_precision", mixed_precision])
    argv.extend(_as_accelerate_target(script_list))
    return argv


def _is_python_interpreter(path: str) -> bool:
    name = os.path.basename(path).lower()
    if name.endswith(".exe"):
        name = name[: -len(".exe")]
    return path == sys.executable or name.startswith("python") or name.startswith("pypy")


def _as_accelerate_target(script_list: list[str]) -> list[str]:
    """Turn ``[python, "-m", "pkg.mod", ...]`` into ``["--module", "pkg.mod", ...]``.

    #77 — ``accelerate launch`` takes a **script path** positionally, or a module
    behind ``--module``. Soup passed ``sys.executable``, so accelerate opened the
    Python ELF binary and parsed it as source::

        File "/root/venv/bin/python", line 1
          ELF
        SyntaxError: source code cannot contain null bytes

    Every rank died before the trainer existed, i.e. ``soup train --gpus N`` — the
    documented multi-GPU entry point — never ran at all. Measured on 4xH100; every
    arm of the #77 matrix had to be launched by hand.

    A real script path is returned untouched, because that form is valid and
    translating it would break it.
    """
    if (
        len(script_list) >= 3
        and script_list[1] == "-m"
        and _is_python_interpreter(script_list[0])
    ):
        return ["--module", script_list[2], *script_list[3:]]
    return list(script_list)


def collect_reexec_passthrough(
    *,
    name: str | None = None,
    fsdp: str | None = None,
    deepspeed: str | None = None,
    resume: str | None = None,
    wandb: bool = False,
    tensorboard: bool = False,
    echo_trap_tokenizer_aware: bool = False,
    reward_hack_detector: str | None = None,
    reward_hack_halt: bool = False,
    reward_hack_mitigation: str | None = None,
    gate: str | None = None,
    push_as: str | None = None,
    hf_resume: bool = False,
    trust_remote_code: bool = False,
    tracker: str | None = None,
    diagnose_gate: str | None = None,
    annex_xi: str | None = None,
    repro_receipt: str | None = None,
    profile_run: bool = False,
    allow_oom_attempt: bool = False,
    track_energy: bool = False,
    energy_country: str | None = None,
    energy_out: str | None = None,
    yes: bool = False,
    minillm_on_policy: bool = False,
    capture_activations: str | None = None,
    capture_prompts: str | None = None,
    replay: str | None = None,
    replay_ratio: float | None = None,
    replay_seed: int | None = None,
) -> list[str]:
    """User-flag tail shared by auto-reexec and the ``--no-reexec`` hint (#372).

    Keep this as the only place that decides which ``soup train`` flags survive
    a multi-GPU launch. ``build_train_reexec_argv`` wraps the result; the hint
    is derived from that same argv, so the two cannot drift.

    Flags that are deliberately *not* parameters here, because they must not
    survive onto the launched command:

    * ``config`` / ``no_reexec`` — injected by ``build_train_reexec_argv``
      (the child is already under a launcher and must not re-exec).
    * ``gpus`` — becomes ``accelerate launch --num_processes``; repeating
      ``--gpus`` under the launcher would double-count.
    * ``dry_run`` — plan-only; the multi-GPU path skips re-exec entirely.
    * ``find_lr`` and its range/output knobs — early-return before launch.
    * ``cloud`` / ``gpu`` / ``cloud_submit`` — a different launch path
      (Modal stub / submit). They return before the accelerate re-exec, and
      ``--cloud`` with ``--gpus`` is two launchers; they are incompatible.
    """
    extra: list[str] = []

    def _opt(flag: str, value: object) -> None:
        if value is not None and value != "":
            extra.extend([flag, str(value)])

    def _switch(flag: str, on: bool) -> None:
        if on:
            extra.append(flag)

    _opt("--name", name)
    _opt("--fsdp", fsdp)
    _opt("--deepspeed", deepspeed)
    _opt("--resume", resume)
    _switch("--wandb", wandb)
    _switch("--tensorboard", tensorboard)
    _switch("--echo-trap-tokenizer-aware", echo_trap_tokenizer_aware)
    _opt("--reward-hack-detector", reward_hack_detector)
    _switch("--reward-hack-halt", reward_hack_halt)
    _opt("--reward-hack-mitigation", reward_hack_mitigation)
    _opt("--gate", gate)
    _opt("--push-as", push_as)
    _switch("--hf-resume", hf_resume)
    _switch("--trust-remote-code", trust_remote_code)
    _opt("--tracker", tracker)
    _opt("--diagnose-gate", diagnose_gate)
    _opt("--annex-xi", annex_xi)
    _opt("--repro-receipt", repro_receipt)
    _switch("--profile", profile_run)
    _switch("--allow-oom-attempt", allow_oom_attempt)
    if track_energy:
        extra.append("--track-energy")
        _opt("--energy-country", energy_country)
        _opt("--energy-out", energy_out)
    _switch("--yes", yes)
    _switch("--minillm-on-policy", minillm_on_policy)
    _opt("--capture-activations", capture_activations)
    _opt("--capture-prompts", capture_prompts)
    _opt("--replay", replay)
    _opt("--replay-ratio", replay_ratio)
    _opt("--replay-seed", replay_seed)
    return extra


def build_train_reexec_argv(
    config: str,
    extra_flags: Sequence[str] | None = None,
) -> list[str]:
    """Argv soup would pass through ``accelerate launch`` on auto-reexec.

    This is the single source for "what the user typed" (#372). Both
    ``os.execvp`` and the ``--no-reexec`` printed hint derive from this list, so
    they cannot drift. ``--no-reexec`` is always present because the child is
    already under a launcher and must not re-exec again.
    """
    if not isinstance(config, str) or not config:
        raise ValueError("config must be a non-empty str")
    extra = list(extra_flags or ())
    return [
        sys.executable,
        "-m",
        "soup_cli.cli",
        "train",
        "--config",
        config,
        "--no-reexec",
        *extra,
    ]


def hint_argv_from_reexec(script_args: Sequence[str]) -> list[str]:
    """``soup train ...`` form of a re-exec argv, for the ``--no-reexec`` hint.

    Drops ``--no-reexec``: under ``accelerate launch`` the run is already
    distributed and never re-execs, so repeating the flag would be noise.
    """
    args = list(script_args)
    if (
        len(args) < 4
        or args[1] != "-m"
        or args[3] != "train"
    ):
        raise ValueError(
            "re-exec argv must look like "
            "[python, '-m', 'soup_cli.cli', 'train', ...]"
        )
    rest = [item for item in args[4:] if item != "--no-reexec"]
    return ["soup", "train", *rest]


def format_advice(num_processes: int, script_args: Sequence[str]) -> str:
    """Human-readable hint telling the user the exact command to re-run."""
    cmd = build_accelerate_argv(num_processes=num_processes, script_args=script_args)
    quoted = " ".join(shlex.quote(arg) for arg in cmd)
    return (
        f"To train on {num_processes} GPUs, re-run under accelerate:\n\n"
        f"    {quoted}\n\n"
        f"(soup does not auto-re-exec in v0.27.0 — it prints this hint so you "
        f"stay in control of env vars and stdio.)"
    )
