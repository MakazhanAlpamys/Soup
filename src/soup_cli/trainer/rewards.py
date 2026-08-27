"""Reward functions for GRPO training.

Built-in reward functions:
  - accuracy: checks if the model answer matches the expected answer
  - format: checks if the response follows a structured format (e.g., <think>...</think>)
  - verifiable: RLVR — deterministic reward via math_verify / code_exec / json_schema

Custom reward functions can be loaded from a Python file with a
`reward_fn(completions, **kwargs)` callable.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import os
import re
import shutil as _shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

console = Console()

MAX_CODE_OUTPUT_BYTES = 10_000
CODE_EXEC_TIMEOUT_SECONDS = 5
# Concurrency cap — at most this many code_exec subprocesses run in parallel
# per reward batch. Prevents fork-storms on large num_generations values.
CODE_EXEC_MAX_PARALLEL = 4
CODE_EXEC_MAX_MEMORY_BYTES = 512 * 1024 * 1024  # 512 MB per run

_SANDBOX_SEMAPHORE = threading.Semaphore(CODE_EXEC_MAX_PARALLEL)

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

# Python-level socket monkey-patch prepended before any sandboxed user code.
# Best-effort (bypassable via os.system / ctypes) — defence-in-depth on top of
# the RLIMIT + namespace / sandbox-exec isolation. Shared so the v0.71.18 #110
# agent-eval sandbox reuses the exact same network guard.
SANDBOX_NETWORK_GUARD = (
    "import socket\n"
    "def _blocked(*a, **k):\n"
    "    raise OSError('network disabled in sandbox')\n"
    "socket.socket = _blocked\n"
    "socket.create_connection = _blocked\n"
)


def _get_safe_sandbox_env() -> dict[str, str]:
    """Construct a minimal, secret-free environment for sandboxed processes."""
    safe_keys = {"PATH", "LANG", "LC_ALL", "TMPDIR", "TERM", "TZ"}
    env = {k: os.environ[k] for k in safe_keys if k in os.environ}
    if "PATH" not in env:
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    env["LANG"] = env.get("LANG", "C.UTF-8")
    env["LC_ALL"] = env.get("LC_ALL", "C.UTF-8")
    env["HOME"] = "/tmp"
    return env


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


def _try_unshare_namespaces(strict: bool = False) -> None:
    """Best-effort: unshare into new user/net/pid namespaces. Silent on failure.

    Called from the POSIX preexec_fn after RLIMITs are set. If the kernel
    rejects the unshare (unprivileged user namespaces disabled, common on
    hardened distros), we silently fall back to RLIMIT + socket patch alone.
    When ``strict=True``, failures raise PermissionError / OSError instead of falling back.
    """
    unshare = getattr(os, "unshare", None)
    if unshare is None:
        if strict:
            raise PermissionError("os.unshare not available")
        return
    try:
        unshare(_CLONE_NEWUSER | _CLONE_NEWNET | _CLONE_NEWPID)
    except (OSError, ValueError):
        # EPERM / ENOSYS / EINVAL — unprivileged unshare not allowed.
        if strict:
            raise
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
        expected_text = "" if expected is None else str(expected).strip()
        if not expected_text:
            rewards.append(0.0)
            continue
        predicted = _extract_answer(content)
        if predicted is not None and predicted.strip() == expected_text:
            rewards.append(1.0)
        elif expected_text.lower() in content.lower():
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


def _apply_rlimit(strict_namespaces: bool = False) -> None:
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
        if hasattr(resource, "RLIMIT_FSIZE"):
            # Limit file size created by subprocess (10 MB max)
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024)
            )
        if hasattr(resource, "RLIMIT_CORE"):
            # Disable core dumps
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if hasattr(resource, "RLIMIT_NOFILE"):
            # Cap open file descriptors
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        if hasattr(resource, "RLIMIT_NPROC"):
            # Cap child processes / threads
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except (ImportError, ValueError, OSError):
        pass
    # Linux defence-in-depth: best-effort unshare into private namespaces.
    if _get_isolation_strategy() == "namespaces":
        _try_unshare_namespaces(strict=strict_namespaces)


@dataclass
class SandboxProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    output_exceeded: bool = False
    launch_failed: bool = False


def _run_sandboxed_subprocess(
    argv: list[str],
    preexec_fn: Callable[[], None] | None = None,
    max_output_bytes: int = MAX_CODE_OUTPUT_BYTES,
    timeout_seconds: int = CODE_EXEC_TIMEOUT_SECONDS,
    env: dict[str, str] | None = None,
) -> SandboxProcessResult:
    """Execute a sandboxed subprocess under concurrency limit with streaming kill-on-overflow."""
    if _get_isolation_strategy() == "sandbox-exec" and sys.platform == "darwin":
        sandbox_bin = _shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
        argv = [sandbox_bin, "-p", MACOS_SANDBOX_PROFILE, *argv]

    if env is None:
        env = _get_safe_sandbox_env()

    with _SANDBOX_SEMAPHORE, tempfile.TemporaryDirectory(prefix="soup-sandbox-") as tmpdir:
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
                preexec_fn=preexec_fn,
                env=env,
            )
        except (PermissionError, OSError) as exc:
            return SandboxProcessResult(
                returncode=None,
                stdout="",
                stderr=str(exc),
                launch_failed=True,
            )

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        output_exceeded = False
        lock = threading.Lock()

        def _reader(stream, chunks: list[bytes]) -> None:
            nonlocal output_exceeded
            try:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    with lock:
                        chunks.append(chunk)
                        total = sum(len(c) for c in stdout_chunks) + sum(len(c) for c in stderr_chunks)
                        if total > max_output_bytes:
                            output_exceeded = True
                            with contextlib.suppress(Exception):
                                proc.kill()
                            break
            finally:
                stream.close()

        t_out = threading.Thread(target=_reader, args=(proc.stdout, stdout_chunks))
        t_err = threading.Thread(target=_reader, args=(proc.stderr, stderr_chunks))
        t_out.daemon = True
        t_err.daemon = True
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=1.0)

        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)

        if timed_out:
            return SandboxProcessResult(None, "", "", timed_out=True)

        if output_exceeded:
            return SandboxProcessResult(
                proc.returncode if proc.returncode is not None else 1,
                "",
                "sandbox output exceeded limit",
                output_exceeded=True,
            )

        raw_out = b"".join(stdout_chunks)
        raw_err = b"".join(stderr_chunks)
        if len(raw_out) + len(raw_err) > max_output_bytes:
            return SandboxProcessResult(
                proc.returncode if proc.returncode is not None else 1,
                "",
                "sandbox output exceeded limit",
                output_exceeded=True,
            )

        stdout = raw_out.decode("utf-8", errors="replace").strip()
        stderr = raw_err.decode("utf-8", errors="replace").strip()
        return SandboxProcessResult(proc.returncode, stdout, stderr)


def _run_code_sandbox(code: str) -> "str | None":
    """Run code in a subprocess sandbox with timeout, rlimits, and output caps.

    Security posture (best-effort, NOT a strong sandbox):
    - Hard wall-clock timeout via subprocess.
    - POSIX ``RLIMIT_AS`` (address space) and ``RLIMIT_CPU`` via preexec_fn.
    - Output truncated to ``MAX_CODE_OUTPUT_BYTES``.
    - Python-level socket monkey-patch (bypassable via os.system / ctypes).
    - Subprocess cwd is a freshly created temporary directory per run, so
      the child's default relative writes land in an ephemeral sandbox dir.
    - Uses ``python -I -S`` to disable site packages and user customization.

    Returns stdout string or None on failure.
    """
    _show_code_exec_warning_once()

    wrapped = SANDBOX_NETWORK_GUARD + "\n" + code

    preexec = _apply_rlimit if sys.platform != "win32" else None

    argv: list[str] = [sys.executable, "-I", "-S", "-c", wrapped]

    result = _run_sandboxed_subprocess(argv, preexec)
    if result.returncode != 0 or result.timed_out or result.output_exceeded or result.launch_failed:
        return None
    return result.stdout


def _run_bash_sandbox(command: str) -> SandboxProcessResult:
    """Run bash command in a subprocess sandbox with timeout, rlimits, and output caps.

    Security posture: mirrors _run_code_sandbox, but does not use Python-level
    socket monkey-patching. Relies entirely on OS-level isolation (unshare on Linux,
    sandbox-exec on macOS).
    Not supported on Windows.
    """
    if sys.platform == "win32":
        raise NotImplementedError("bash sandbox not supported on Windows")

    _show_code_exec_warning_once()

    def preexec() -> None:
        _apply_rlimit(strict_namespaces=True)

    argv: list[str] = ["bash", "-c", command]

    result = _run_sandboxed_subprocess(argv, preexec)
    if result.launch_failed:
        raise PermissionError(result.stderr)
    return result


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


def _validated_reward_fn(reward_fn: Callable) -> Callable:
    """Wrap one reward function with TRL's one-score-per-completion contract."""
    reward_name = getattr(reward_fn, "__name__", "reward_fn")

    @wraps(reward_fn)
    def checked(*args: Any, **kwargs: Any) -> Any:
        rewards = reward_fn(*args, **kwargs)
        completions = kwargs.get("completions")
        if completions is None:
            if len(args) >= 2:
                completions = args[1]
            elif args:
                completions = args[0]
        if completions is None:
            raise ValueError(
                f"Reward function {reward_name!r} was called without completions"
            )
        try:
            reward_count = len(rewards)
        except TypeError as exc:
            raise ValueError(
                f"Reward function {reward_name!r} must return one finite score "
                "per completion"
            ) from exc
        completion_count = len(completions)
        if reward_count != completion_count:
            raise ValueError(
                f"Reward function {reward_name!r} returned {reward_count} scores "
                f"for {completion_count} completions; return exactly one finite "
                "score per completion and verify the required dataset columns"
            )
        for index, reward in enumerate(rewards):
            if isinstance(reward, bool):
                raise ValueError(
                    f"Reward function {reward_name!r} returned boolean score at "
                    f"index {index}; scores must be finite numbers"
                )
            try:
                finite = math.isfinite(float(reward))
            except (TypeError, ValueError, OverflowError):
                finite = False
            if not finite:
                raise ValueError(
                    f"Reward function {reward_name!r} returned non-finite or "
                    f"non-numeric score {reward!r} at index {index}"
                )
        return rewards

    return checked


def validate_reward_funcs(reward_funcs: Any) -> Any:
    """Validate one reward callable or an ensemble without changing its shape."""
    if isinstance(reward_funcs, (list, tuple)):
        return [_validated_reward_fn(reward_fn) for reward_fn in reward_funcs]
    return _validated_reward_fn(reward_funcs)


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


def load_reward_fns(
    reward_fn_spec: str, verifiable_domain: "str | None" = None,
) -> list[Callable]:
    """Load one OR MORE reward functions from a comma-separated spec (v0.71.40 #311).

    ``reward_fn`` may name several rewards, e.g. ``"accuracy,format"`` — each is
    resolved via :func:`load_reward_fn` and returned as a list, which TRL's
    ``GRPOTrainer(reward_funcs=[...])`` accepts and which the ``rm_ensemble``
    reward-hack detector (needs >= 2 reward fns) unlocks. A single name still
    returns a one-element list, so callers can always treat the result uniformly.

    Splitting is on ``,``; blank / empty segments raise (``"accuracy,"`` is a typo,
    not "accuracy plus nothing"), and duplicate segments raise (they would collide
    by ``__name__`` in the ``rm_ensemble`` capture buffer, silently shrinking the
    ensemble). A ``.py`` path containing a literal comma is not supported — rename
    the file.
    """
    if not isinstance(reward_fn_spec, str):
        raise ValueError(
            "reward_fn must be a string (a name, a .py path, or a comma-separated "
            f"list), got {type(reward_fn_spec).__name__}"
        )
    if not reward_fn_spec.strip():
        raise ValueError("reward_fn must not be blank")
    segments = [seg.strip() for seg in reward_fn_spec.split(",")]
    if any(not seg for seg in segments):
        raise ValueError(
            f"reward_fn {reward_fn_spec!r} has an empty comma segment — "
            "remove the stray comma"
        )
    seen: set[str] = set()
    for seg in segments:
        if seg in seen:
            raise ValueError(
                f"reward_fn {reward_fn_spec!r} lists {seg!r} twice — "
                "duplicate rewards collide in the ensemble"
            )
        seen.add(seg)
    return [load_reward_fn(seg, verifiable_domain=verifiable_domain) for seg in segments]
