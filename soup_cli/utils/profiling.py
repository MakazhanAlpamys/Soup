"""Auto-profiling helper (v0.34.0 Part F).

Wraps torch.profiler in a context manager that records a trace + writes a
Chrome trace JSON. Designed to be opt-in (``soup train --profile``) and to
no-op gracefully when torch is unavailable so unit tests can exercise the
glue code on a CPU-only / torch-less CI runner.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from soup_cli.utils.paths import is_under_cwd

# Default warm-up + active windows. Profiling for an entire training run
# would explode disk; default profiles the early steady-state.
DEFAULT_WAIT_STEPS = 1
DEFAULT_WARMUP_STEPS = 1
DEFAULT_ACTIVE_STEPS = 5
DEFAULT_REPEAT = 1

# Bounded so users can't accidentally make a multi-GB trace.
MAX_ACTIVE_STEPS = 50


@dataclass(frozen=True)
class ProfilerSchedule:
    wait: int
    warmup: int
    active: int
    repeat: int

    @classmethod
    def default(cls) -> "ProfilerSchedule":
        return cls(
            DEFAULT_WAIT_STEPS,
            DEFAULT_WARMUP_STEPS,
            DEFAULT_ACTIVE_STEPS,
            DEFAULT_REPEAT,
        )

    def validate(self) -> None:
        for name, value in (
            ("wait", self.wait),
            ("warmup", self.warmup),
            ("active", self.active),
            ("repeat", self.repeat),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"profiler {name} must be a non-negative int")
        if self.active == 0:
            raise ValueError("profiler active must be > 0 (nothing would be recorded)")
        if self.active > MAX_ACTIVE_STEPS:
            raise ValueError(
                f"profiler active {self.active} exceeds cap {MAX_ACTIVE_STEPS}"
            )


def resolve_trace_path(output_dir: Path, run_id: str) -> Path:
    """Compute the full Chrome trace path under ``output_dir/profiles``.

    Validation of ``run_id`` rejects path separators, null bytes, and bare
    parent / current directory references so a crafted id can't escape
    the profiles dir on path composition. Containment is checked via
    ``os.path.realpath + commonpath`` (project convention).
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if run_id in (".", "..") or "/" in run_id or "\\" in run_id or "\x00" in run_id:
        raise ValueError("run_id contains path separator, null byte, or '..'")
    target = Path(os.path.realpath(str(Path(output_dir) / "profiles")))
    if not is_under_cwd(target):
        raise ValueError(f"profile dir {target} is not under cwd")
    return target / f"{run_id}.trace.json"


@contextlib.contextmanager
def profile_training(
    *,
    output_dir: Path,
    run_id: str,
    schedule: Optional[ProfilerSchedule] = None,
) -> Iterator[Optional[object]]:
    """Yield a torch.profiler.profile, or None when torch is missing.

    Caller invokes ``profiler.step()`` on each training step (when not None)
    so torch.profiler can advance through the wait → warmup → active phases.
    """
    schedule = schedule or ProfilerSchedule.default()
    schedule.validate()
    # Validate run_id / output_dir up-front so the caller fails fast even
    # without torch installed.
    trace_path = resolve_trace_path(output_dir, run_id)

    try:
        from torch import profiler as torch_profiler
    except ImportError:
        yield None
        return

    # Only create the profiles dir once we know torch.profiler is usable —
    # avoids leaving stale empty dirs on torch-less CI runs.
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    activities = [torch_profiler.ProfilerActivity.CPU]
    try:
        import torch

        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
    except Exception:
        pass

    def _on_trace_ready(prof) -> None:
        try:
            prof.export_chrome_trace(str(trace_path))
        except Exception:
            # Profiling must never crash a real training run.
            pass

    with torch_profiler.profile(
        activities=activities,
        schedule=torch_profiler.schedule(
            wait=schedule.wait,
            warmup=schedule.warmup,
            active=schedule.active,
            repeat=schedule.repeat,
        ),
        on_trace_ready=_on_trace_ready,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        yield profiler
