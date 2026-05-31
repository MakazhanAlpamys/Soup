"""Reward functions for GRPO training.

Built-in reward functions:
  - accuracy: checks if the model answer matches the expected answer
  - format: checks if the response follows a structured format (e.g., <think>...</think>)
  - verifiable: RLVR — deterministic reward via math_verify / code_exec / json_schema

Custom reward functions can be loaded from a Python file with a
`reward_fn(completions, **kwargs)` callable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil as _shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()

MAX_CODE_OUTPUT_BYTES = 10_000
CODE_EXEC_TIMEOUT_SECONDS = 5
# Concurrency cap — at most this many code_exec subprocesses run in parallel
# per reward batch. Prevents fork-storms on large num_generations values.
CODE_EXEC_MAX_PARALLEL = 4
CODE_EXEC_MAX_MEMORY_BYTES = 512 * 1024 * 1024  # 512 MB per run

_CODE_EXEC_WARNING_SHOWN = False

# Cached isolation strategy — recomputed on demand when tests reset to None
_ISOLATION_STRATEGY_CACHE: "str | None" = None

# macOS sandbox-exec profile: default-deny, allow narrow process needs, block
# network and writes outside /tmp. Defence-in-depth on top of RLIMIT + socket
# patch + ephemeral cwd. See sandbox-exec(1) and Apple's seatbelt SBPL.
MACOS_SANDBOX_PROFILE = (
    "(version 1)"
    "(deny default)"
    "(allow process-fork)"
    "(allow process-exec)"
    "(allow signal (target self))"
    "(allow file-read*)"
    '(allow file-write* (subpath "/tmp") (subpath "/private/tmp") (subpath "/var/folders"))'
    "(allow sysctl-read)"
    # Narrow mach-lookup allowlist — broad ``(allow mach-lookup)`` permits
    # DNS / NSURLSession via launchd-brokered Mach IPC and effectively
    # bypasses ``(deny network*)``. The names below are required for the
    # interpreter to boot (entitlement / system-services lookup) but do
    # NOT include ``com.apple.SystemConfiguration`` or ``com.apple.dnssd``.
    '(allow mach-lookup'
    ' (global-name "com.apple.SecurityServer")'
    ' (global-name "com.apple.system.notification_center")'
    ' (global-name "com.apple.system.opendirectoryd.libinfo"))'
    "(deny network*)"
)


def _compute_isolation_strategy() -> str:
    """Detect best-available OS-level sandbox isolation for code_exec_reward.

    Returns one of:
      - "namespaces" : Linux with `os.unshare` available (Python 3.12+) — we
        will best-effort `unshare(CLONE_NEWUSER|CLONE_NEWNET|CLONE_NEWPID)` in
        the child preexec_fn. Falls back at runtime if unprivileged user
        namespaces are disabled (EPERM/ENOSYS).
      - "sandbox-exec" : macOS with `sandbox-exec` binary on PATH — we wrap
        argv with `sandbox-exec -p <profile>`.
      - "best-effort" : everything else (Windows, restricted Linux). Existing
        RLIMIT + socket-patch + ephemeral-cwd guards still apply.

    The result is cached after first call. Tests reset
    ``_ISOLATION_STRATEGY_CACHE`` to None to re-probe.
    """
    if sys.platform == "linux" and hasattr(os, "unshare"):
        return "namespaces"
    if sys.platform == "darwin" and _shutil.which("sandbox-exec") is not None:
        return "sandbox-exec"
    return "best-effort"


def _get_isolation_strategy() -> str:
    """Cached wrapper for ``_compute_isolation_strategy``."""
    global _ISOLATION_STRATEGY_CACHE
    if _ISOLATION_STRATEGY_CACHE is None:
        _ISOLATION_STRATEGY_CACHE = _compute_isolation_strategy()
    return _ISOLATION_STRATEGY_CACHE


# Linux unshare flags — matches kernel uapi/linux/sched.h. Hard-coded so we
# don't depend on a runtime constant import.
_CLONE_NEWUSER = 0x10000000
_CLONE_NEWNET = 0x40000000
_CLONE_NEWPID = 0x20000000


def _try_unshare_namespaces() -> None:
    """Best-effort: unshare into new user/net/pid namespaces. Silent on failure.

    Called from the POSIX preexec_fn after RLIMITs are set. If the kernel
    rejects the unshare (unprivileged user namespaces disabled, common on
    hardened distros), we silently fall back to RLIMIT + socket patch alone.
    """
    unshare = getattr(os, "unshare", None)
    if unshare is None:
        return
    try:
        unshare(_CLONE_NEWUSER | _CLONE_NEWNET | _CLONE_NEWPID)
    except (OSError, ValueError):
        # EPERM / ENOSYS / EINVAL — unprivileged unshare not allowed.
        # Continue with weaker isolation rather than failing the run.
        pass


def _show_code_exec_warning_once() -> None:
    """Display a one-time warning panel when code_exec_reward is first used."""
    global _CODE_EXEC_WARNING_SHOWN
    if _CODE_EXEC_WARNING_SHOWN:
        return
    _CODE_EXEC_WARNING_SHOWN = True
    console.print(
        Panel(
            "[bold yellow]RLVR code_exec_reward is a BEST-EFFORT sandbox.[/]\n\n"
            "Model-generated code runs in a subprocess with:\n"
            "  - 5s wall-clock timeout\n"
            "  - 512MB RLIMIT_AS on POSIX (Linux/macOS)\n"
            "  - Restricted temporary working directory\n"
            "  - A Python-level socket monkey-patch\n\n"
            "[bold]The socket patch can be bypassed[/] by generated code "
            "invoking os.system / subprocess / ctypes. Network isolation is "
            "NOT enforced. Do not enable code_exec_reward on hosts that "
            "hold secrets or run alongside trusted services. Prefer running "
            "training inside a container/VM with no network interface.",
            title="code_exec_reward — security notice",
            border_style="yellow",
        )
    )


def accuracy_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward based on whether the final answer matches the expected answer.

    Looks for the answer after the last '####' or in a \\boxed{} block.
    Falls back to checking if the expected answer appears anywhere in the response.

    Args:
        completions: list of message lists, each containing a completion with 'content'.
        **kwargs: must contain 'answer' — the expected answer for each prompt.

    Returns:
        List of float rewards (1.0 for correct, 0.0 for incorrect).
    """
    answers = kwargs.get("answer", [])
    rewards = []
    for completion, expected in zip(completions, answers):
        content = completion[-1]["content"] if completion else ""
        predicted = _extract_answer(content)
        if predicted is not None and predicted.strip() == str(expected).strip():
            rewards.append(1.0)
        elif str(expected).strip().lower() in content.lower():
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward based on whether the response follows a structured reasoning format.

    Checks for:
      - <think>...</think> block (chain-of-thought)
      - A final answer section after the thinking block

    Args:
        completions: list of message lists.
        **kwargs: unused.

    Returns:
        List of float rewards (0.0 to 1.0).
    """
    rewards = []
    for completion in completions:
        content = completion[-1]["content"] if completion else ""
        score = 0.0
        # Check for <think> block
        if re.search(r"<think>.*?</think>", content, re.DOTALL):
            score += 0.5
        # Check for content after </think>
        after_think = re.split(r"</think>", content)
        if len(after_think) > 1 and after_think[-1].strip():
            score += 0.5
        rewards.append(score)
    return rewards


def _extract_answer(text: str) -> str | None:
    """Extract the final answer from model output.

    Supports:
      - #### <answer> format (GSM8K style)
      - \\boxed{<answer>} format (math style)
    """
    # Try #### format
    parts = text.split("####")
    if len(parts) > 1:
        return parts[-1].strip()
    # Try \\boxed{} format
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


# --- RLVR: verifiable rewards (Part C of v0.25.0) ---

_MATH_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_numeric_answer(text: str) -> "float | None":
    """Extract a numeric answer using safe regex (never calls eval())."""
    answer_str = _extract_answer(text)
    if answer_str is None:
        # Fallback: find the last number in text
        nums = _MATH_NUM_RE.findall(text)
        if not nums:
            return None
        answer_str = nums[-1]

    # Accept only simple numeric literals — reject anything else
    match = _MATH_NUM_RE.fullmatch(answer_str.strip())
    if match is None:
        return None
    try:
        return float(match.group(0))
    except (ValueError, TypeError):
        return None


def math_verify_reward(
    completions: list[list[dict]],
    tolerance: float = 1e-4,
    **kwargs,
) -> list[float]:
    """RLVR math reward: compare extracted numeric answer to expected.

    Security: never uses ``eval()``. Only numeric literals that match a strict
    regex are accepted. Non-numeric answers score 0.0.
    """
    answers = kwargs.get("answer", [])
    rewards: list[float] = []
    for completion, expected in zip(completions, answers):
        content = completion[-1]["content"] if completion else ""
        predicted = _extract_numeric_answer(content)
        try:
            expected_num = float(str(expected).strip())
        except (ValueError, TypeError):
            expected_num = None

        if predicted is None or expected_num is None:
            rewards.append(0.0)
            continue

        if abs(predicted - expected_num) <= tolerance:
            rewards.append(1.0)
        elif abs(predicted - expected_num) <= max(tolerance * 100, 1e-2):
            rewards.append(0.6)
        else:
            rewards.append(0.0)
    return rewards


def _extract_code_block(content: str) -> str:
    """Extract a Python code block from content. Strips markdown fences."""
    match = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


def _apply_rlimit() -> None:
    """POSIX only: set resource limits for sandboxed subprocess.

    Called via ``preexec_fn`` before the child runs user code. On Windows this
    is never invoked because ``preexec_fn`` is POSIX-only and the caller skips
    it there.
    """
    try:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS,
            (CODE_EXEC_MAX_MEMORY_BYTES, CODE_EXEC_MAX_MEMORY_BYTES),
        )
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (CODE_EXEC_TIMEOUT_SECONDS, CODE_EXEC_TIMEOUT_SECONDS),
        )
    except (ImportError, ValueError, OSError):
        pass
    # Linux defence-in-depth: best-effort unshare into private namespaces.
    if _get_isolation_strategy() == "namespaces":
        _try_unshare_namespaces()


def _run_code_sandbox(code: str) -> "str | None":
    """Run code in a subprocess sandbox with timeout, rlimits, and output caps.

    Security posture (best-effort, NOT a strong sandbox):
    - Hard wall-clock timeout via subprocess.run(timeout=...).
    - POSIX ``RLIMIT_AS`` (address space) and ``RLIMIT_CPU`` via preexec_fn.
    - Output truncated to ``MAX_CODE_OUTPUT_BYTES``.
    - Python-level socket monkey-patch (bypassable via os.system / ctypes).
    - Subprocess cwd is a freshly created temporary directory per run, so
      the child's default relative writes land in an ephemeral sandbox dir.
    - Uses ``python -I -S`` to disable site packages and user customization.

    Returns stdout string or None on failure.
    """
    _show_code_exec_warning_once()

    guard = (
        "import socket\n"
        "def _blocked(*a, **k):\n"
        "    raise OSError('network disabled in sandbox')\n"
        "socket.socket = _blocked\n"
        "socket.create_connection = _blocked\n"
    )
    wrapped = guard + "\n" + code

    preexec = _apply_rlimit if sys.platform != "win32" else None

    argv: list[str] = [sys.executable, "-I", "-S", "-c", wrapped]
    if _get_isolation_strategy() == "sandbox-exec":
        # macOS: prefix with sandbox-exec + inline profile. The profile denies
        # all by default and only re-allows what an interpreter must do to
        # boot; network is explicitly denied.
        sandbox_bin = _shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
        argv = [sandbox_bin, "-p", MACOS_SANDBOX_PROFILE, *argv]

    with tempfile.TemporaryDirectory(prefix="soup-code-exec-") as tmpdir:
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
            return None
        except (OSError, ValueError):
            return None

    if proc.returncode != 0:
        return None

    out = proc.stdout or ""
    if len(out.encode("utf-8", errors="replace")) > MAX_CODE_OUTPUT_BYTES:
        return None
    return out.strip()


def code_exec_reward(
    completions: list[list[dict]],
    **kwargs,
) -> list[float]:
    """RLVR code reward: execute completion code, compare output to expected.

    Security: runs every completion in a subprocess sandbox with a 5s timeout
    and a 10KB output cap. Network access is disabled via socket monkey-patch
    injected before user code runs. Per-batch parallelism is capped at
    ``CODE_EXEC_MAX_PARALLEL`` to prevent fork storms on large batches.
    """
    from concurrent.futures import ThreadPoolExecutor

    expected_outputs = kwargs.get("expected", kwargs.get("answer", []))
    items: list[tuple[str, str]] = []
    for completion, expected in zip(completions, expected_outputs):
        content = completion[-1]["content"] if completion else ""
        code = _extract_code_block(content)
        items.append((code, str(expected).strip()))

    def _score(item: tuple[str, str]) -> float:
        code, expected = item
        if not code:
            return 0.0
        output = _run_code_sandbox(code)
        if output is None:
            return 0.0
        return 1.0 if output.strip() == expected else 0.0

    max_workers = max(1, min(CODE_EXEC_MAX_PARALLEL, len(items)))
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_score, items))


def _score_against_schema(data: object, schema: dict) -> float:
    """Lightweight JSON-schema completeness score (no jsonschema dep).

    - 1.0 if all required fields are present with matching primitive types.
    - Partial credit proportional to required fields satisfied.
    - 0.0 if schema validation fails completely.
    """
    if not isinstance(schema, dict):
        return 0.0
    if schema.get("type") == "object":
        if not isinstance(data, dict):
            return 0.0
        required = schema.get("required") or list((schema.get("properties") or {}).keys())
        if not required:
            return 1.0
        hit = 0
        properties = schema.get("properties") or {}
        for field_name in required:
            if field_name not in data:
                continue
            field_schema = properties.get(field_name, {})
            if _type_matches(data[field_name], field_schema.get("type")):
                hit += 1
        return hit / len(required)
    return 1.0 if _type_matches(data, schema.get("type")) else 0.0


def _type_matches(value: object, type_name: "str | None") -> bool:
    if type_name is None:
        return True
    mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    expected_type = mapping.get(type_name)
    if expected_type is None:
        return True
    if type_name == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, expected_type)


def json_schema_reward(
    completions: list[list[dict]],
    **kwargs,
) -> list[float]:
    """RLVR JSON schema reward: parse completion as JSON, score schema conformance."""
    schemas = kwargs.get("schema", [])
    rewards: list[float] = []
    for completion, schema in zip(completions, schemas):
        content = completion[-1]["content"] if completion else ""
        # Strip markdown fences first
        fenced = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if fenced:
            content = fenced.group(1)
        try:
            data = json.loads(content.strip())
        except (json.JSONDecodeError, ValueError):
            rewards.append(0.0)
            continue
        rewards.append(_score_against_schema(data, schema))
    return rewards


# Registry of built-in reward functions
BUILTIN_REWARDS: dict[str, Callable] = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

VERIFIABLE_DOMAINS: dict[str, Callable] = {
    "math": math_verify_reward,
    "code": code_exec_reward,
    "json_schema": json_schema_reward,
}


def load_reward_fn(
    reward_fn_spec: str, verifiable_domain: "str | None" = None,
) -> Callable:
    """Load a reward function by name or from a custom Python file.

    Args:
        reward_fn_spec: Either a built-in name ('accuracy', 'format', 'verifiable')
            or a path to a .py file containing a `reward_fn` callable.
        verifiable_domain: Required when ``reward_fn_spec == 'verifiable'``.
            One of ``"math"``, ``"code"``, ``"json_schema"``.

    Returns:
        A callable reward function with signature:
        (completions: list[list[dict]], **kwargs) -> list[float]
    """
    # RLVR: verifiable reward routing
    if reward_fn_spec == "verifiable":
        if verifiable_domain is None:
            raise ValueError(
                "reward_fn='verifiable' requires verifiable_domain "
                "(one of: math, code, json_schema)"
            )
        if verifiable_domain not in VERIFIABLE_DOMAINS:
            raise ValueError(
                f"Unknown verifiable_domain: '{verifiable_domain}'. "
                f"Options: {', '.join(VERIFIABLE_DOMAINS.keys())}"
            )
        console.print(
            f"[dim]Using verifiable reward: domain={verifiable_domain}[/]"
        )
        return VERIFIABLE_DOMAINS[verifiable_domain]

    # Built-in reward function
    if reward_fn_spec in BUILTIN_REWARDS:
        console.print(f"[dim]Using built-in reward function: {reward_fn_spec}[/]")
        return BUILTIN_REWARDS[reward_fn_spec]

    # Custom Python file
    reward_path = Path(reward_fn_spec)
    if reward_path.exists() and reward_path.suffix == ".py":
        console.print(
            f"[bold yellow]Warning:[/] Loading custom reward function from: "
            f"[bold]{reward_path.resolve()}[/]\n"
            f"[yellow]This will execute arbitrary Python code. "
            f"Only use reward files you trust.[/]"
        )
        spec = importlib.util.spec_from_file_location("custom_reward", reward_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "reward_fn"):
            raise ValueError(
                f"Custom reward file {reward_path} must define a 'reward_fn' callable.\n"
                f"Example:\n"
                f"  def reward_fn(completions, **kwargs):\n"
                f"      return [1.0] * len(completions)"
            )
        return module.reward_fn

    raise ValueError(
        f"Unknown reward function: '{reward_fn_spec}'\n"
        f"Options: {', '.join(BUILTIN_REWARDS.keys())}, 'verifiable', "
        f"or path to a .py file"
    )
