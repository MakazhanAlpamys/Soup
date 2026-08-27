"""Evaluate code outputs against test cases in an isolated subprocess.

Mirrors the v0.25.0 RLVR code_exec_reward sandbox:
  - 5-second hard timeout
  - 512MB memory limit on POSIX (RLIMIT_AS)
  - 10KB output cap
  - Python socket monkey-patch (SANDBOX_NETWORK_GUARD)
  - Linux unshare / macOS sandbox-exec when available

Output is parsed as JSON. Structured outcomes:
  - 'ok'           : non-zero exit code 0 AND parseable output
  - 'tool_error'   : non-zero exit code OR unparseable output
  - 'timeout'      : exceeded 5s wall-clock timeout
"""

from __future__ import annotations

import base64
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Optional

from rich.console import Console

console = Console()


@dataclass
class EvalResult:
    score: float
    passed: bool
    feedback: str
    details: dict


def run_eval_in_sandbox(code: str) -> "tuple[Optional[int], str, bool]":
    """Run ``code`` in the v0.25.0 RLVR sandbox; return ``(rc, stdout, timed)``.

    Reuses the exact v0.33.0 isolation strategy (RLIMIT / namespaces /
    sandbox-exec on POSIX) and the shared ``SANDBOX_NETWORK_GUARD``. The
    subprocess + 5 s timeout + 10 KB output cap apply on every platform; the
    strong isolation primitives are POSIX-only (preexec is skipped on
    Windows). Returns ``(returncode, stdout, timed_out)``.
    """
    from soup_cli.trainer.rewards import (
        SANDBOX_NETWORK_GUARD,
        _apply_rlimit,
        _run_sandboxed_subprocess,
    )

    wrapped = SANDBOX_NETWORK_GUARD + "\n" + code
    preexec = _apply_rlimit if sys.platform != "win32" else None
    argv: list[str] = [sys.executable, "-I", "-S", "-c", wrapped]

    result = _run_sandboxed_subprocess(argv, preexec)

    if result.output_exceeded or result.launch_failed:
        # Oversize output or launch failure is treated as a failure (matches v0.25.0 cap policy).
        return result.returncode if result.returncode is not None else 1, "", False
    return result.returncode, result.stdout, result.timed_out


def classify_sandbox_outcome(
    returncode: "Optional[int]", stdout: str, timed_out: bool
) -> str:
    """Map a sandbox run to ``ok`` / ``tool_error`` / ``timeout``.

    ``ok`` requires returncode 0 AND non-empty parseable JSON output (the
    issue's "returned 0 + parseable output"). A zero return with empty or
    non-parseable output is a ``tool_error``.
    """
    if timed_out:
        return "timeout"
    if returncode != 0:
        return "tool_error"
    if not stdout or not stdout.strip():
        return "tool_error"
    try:
        json.loads(stdout.strip())
        return "ok"
    except (json.JSONDecodeError, ValueError):
        return "tool_error"
