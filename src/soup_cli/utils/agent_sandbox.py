"""Sandboxed tool-call scoring for ``soup agent eval`` (v0.71.18 #110).

v0.46.0 ``soup agent eval`` scores predicted tool-calls with a *heuristic*:
the predicted ``tool`` must exist in the spec catalog and its ``arguments``
may only reference declared parameters. That catches malformed predictions
but never *executes* the call.

This module wires the v0.25.0 RLVR ``code_exec`` sandbox (5 s timeout, POSIX
``RLIMIT_AS`` / ``RLIMIT_CPU``, best-effort Linux namespaces / macOS
``sandbox-exec``, ephemeral cwd, network monkey-patch, ``python -I -S``) so
each heuristic-passing prediction is *run* against a generated mock of the
endpoint. The mock derives the endpoint's required path parameters from the
spec ``path`` (``/users/{user_id}`` -> ``user_id`` is required) and builds
the would-be request URL; a prediction that omits a required path parameter
fails to construct the URL and surfaces as a ``tool_error``.

Classifications (4-way, matching the issue):

* ``ok``        — the sandbox returned 0 with parseable JSON output.
* ``tool_error``— the sandbox returned non-zero (the mock raised).
* ``timeout``   — the sandbox hit the 5 s wall-clock cap.
* ``arg_error`` — the heuristic rejected the row before the sandbox
                  (unknown tool / unknown parameter key).

Security:
- No code interpolation. The tool / params / path / arguments are
  base64-encoded JSON and decoded *as data* inside the stub; the base64
  alphabet (``[A-Za-z0-9+/=]``) carries zero injection surface.
- Output truncated at 10 KB (the stub itself + the v0.25.0 output cap).
- Strong isolation (RLIMIT / namespaces / sandbox-exec) is POSIX-only; on
  Windows the subprocess + timeout + output cap + network guard still apply
  (reduced isolation — the CLI prints a friendly advisory).
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Optional

# Closed set of the 4 outcome buckets.
SANDBOX_CLASSIFICATIONS = ("ok", "tool_error", "timeout", "arg_error")

# Matches ``{name}`` path placeholders that don't contain ``/`` or braces.
_PATH_PARAM_RE = re.compile(r"\{([^{}/]+)\}")

# In-stub output cap (defence-in-depth on top of the v0.25.0 10 KB cap).
_STUB_OUTPUT_CHARS = 10_000

# Built via string concatenation (not a triple-quoted literal) so there is no
# brace / backslash escaping to get wrong. ``__PAYLOAD__`` is replaced with the
# base64 JSON blob; the stub decodes it as DATA and never evals it.
_STUB_TEMPLATE = (
    "import base64, json\n"
    '_blob = base64.b64decode("__PAYLOAD__".encode("ascii")).decode("utf-8")\n'
    "_p = json.loads(_blob)\n"
    '_path = _p.get("path") or ""\n'
    '_args = _p.get("args") or {}\n'
    '_required = _p.get("required") or []\n'
    "if not isinstance(_args, dict):\n"
    '    raise SystemExit("arguments must be an object")\n'
    "_missing = [x for x in _required if x not in _args]\n"
    "if _missing:\n"
    '    raise SystemExit("missing required path params: " + ",".join(_missing))\n'
    "_url = _path\n"
    "for _x in _required:\n"
    '    _url = _url.replace("{" + _x + "}", str(_args.get(_x, "")))\n'
    "_extra = {k: v for k, v in _args.items() if k not in _required}\n"
    '_out = json.dumps({"url": _url, "params": _extra}, default=str)\n'
    f"print(_out[:{_STUB_OUTPUT_CHARS}])\n"
)

# Test / advanced-operator seam: when set, the sandbox runner is replaced.
# Signature mirrors :func:`run_eval_in_sandbox` -> ``(returncode, stdout,
# timed_out)``. Defaults to the real sandbox executor.
_AGENT_SANDBOX_RUN_OVERRIDE: Optional[
    Callable[[str], "tuple[Optional[int], str, bool]"]
] = None


@dataclass(frozen=True)
class SandboxScorecard:
    """Aggregated per-classification counts over a predictions file."""

    ok: int = 0
    tool_error: int = 0
    timeout: int = 0
    arg_error: int = 0

    @property
    def total(self) -> int:
        return self.ok + self.tool_error + self.timeout + self.arg_error


def _required_path_params(path: object) -> list[str]:
    """Extract ``{name}`` path placeholders from a spec path string."""
    if not isinstance(path, str):
        return []
    return _PATH_PARAM_RE.findall(path)


def build_eval_stub(
    *,
    tool: str,
    parameters: Sequence[str],
    path: str,
    arguments: Mapping,
) -> str:
    """Generate the sandbox stub for one prediction (v0.71.18 #110).

    Embeds the endpoint path, its required path parameters, and the
    predicted arguments as a base64 JSON blob (no code interpolation). The
    emitted program validates the required path params, builds the would-be
    URL, and prints a JSON result. Raises ``ValueError`` on a
    non-serialisable / oversize payload.

    ``tool`` / ``parameters`` are accepted for caller symmetry (the CLI passes
    ``ep.tool`` / ``ep.parameters``) but the stub only needs path / required /
    args, so they are not embedded — keeps the base64 blob lean.
    """
    if not isinstance(arguments, Mapping):
        raise ValueError("arguments must be a mapping")
    del tool, parameters  # accepted for symmetry; not embedded in the stub
    payload = {
        "path": str(path) if path is not None else "",
        "required": _required_path_params(path),
        "args": dict(arguments),
    }
    try:
        blob = json.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"arguments not JSON-serialisable: {exc}") from exc
    b64 = base64.b64encode(blob.encode("utf-8")).decode("ascii")
    return _STUB_TEMPLATE.replace("__PAYLOAD__", b64)


def run_eval_in_sandbox(code: str) -> "tuple[Optional[int], str, bool]":
    """Run ``code`` in the v0.25.0 RLVR sandbox; return ``(rc, stdout, timed)``.

    Reuses the exact v0.33.0 isolation strategy (RLIMIT / namespaces /
    sandbox-exec on POSIX) and the shared ``SANDBOX_NETWORK_GUARD``. The
    subprocess + 5 s timeout + 10 KB output cap apply on every platform; the
    strong isolation primitives are POSIX-only (preexec is skipped on
    Windows). Returns ``(returncode, stdout, timed_out)``.
    """
    from soup_cli.trainer.rewards import (
        CODE_EXEC_TIMEOUT_SECONDS,
        MACOS_SANDBOX_PROFILE,
        MAX_CODE_OUTPUT_BYTES,
        SANDBOX_NETWORK_GUARD,
        _apply_rlimit,
        _get_isolation_strategy,
    )

    wrapped = SANDBOX_NETWORK_GUARD + "\n" + code
    preexec = _apply_rlimit if sys.platform != "win32" else None
    argv: list[str] = [sys.executable, "-I", "-S", "-c", wrapped]
    if _get_isolation_strategy() == "sandbox-exec":
        import shutil

        sandbox_bin = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
        argv = [sandbox_bin, "-p", MACOS_SANDBOX_PROFILE, *argv]

    with tempfile.TemporaryDirectory(prefix="soup-agent-eval-") as tmpdir:
        try:
            proc = subprocess.run(  # noqa: S603 — list args, trusted interpreter
                argv,
                capture_output=True,
                text=True,
                timeout=CODE_EXEC_TIMEOUT_SECONDS,
                check=False,
                cwd=tmpdir,
                preexec_fn=preexec,  # noqa: PLW1509 — intentional RLIMIT application
            )
        except subprocess.TimeoutExpired:
            return None, "", True
        except (OSError, ValueError):
            return None, "", False

    out = proc.stdout or ""
    # Oversize output is treated as a failure (matches v0.25.0 cap policy).
    if len(out.encode("utf-8", errors="replace")) > MAX_CODE_OUTPUT_BYTES:
        return proc.returncode, "", False
    return proc.returncode, out.strip(), False


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
    if not stdout:
        return "tool_error"
    try:
        json.loads(stdout)
    except (ValueError, TypeError):
        return "tool_error"
    return "ok"


def score_sandbox(
    *,
    tool: str,
    parameters: Sequence[str],
    path: str,
    arguments: Mapping,
) -> str:
    """Build the stub, run it sandboxed, return the classification.

    Only called for heuristic-passing rows (unknown tool / param keys are
    ``arg_error``, handled by the caller before this). Returns one of
    ``ok`` / ``tool_error`` / ``timeout``.
    """
    stub = build_eval_stub(
        tool=tool, parameters=parameters, path=path, arguments=arguments
    )
    runner = _AGENT_SANDBOX_RUN_OVERRIDE or run_eval_in_sandbox
    returncode, stdout, timed_out = runner(stub)
    return classify_sandbox_outcome(returncode, stdout, timed_out)
