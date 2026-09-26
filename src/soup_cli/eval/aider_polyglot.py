"""Safe orchestration and result parsing for Aider's Polyglot benchmark.

The PyPI ``aider-chat`` wheel does not ship Aider's benchmark harness.  The
upstream-supported runner lives in the Aider source tree and is executed in
the locally built ``aider-benchmark`` Docker image.  This module wraps that
documented container contract and converts its per-exercise JSON files into a
single Soup evaluation row.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

from soup_cli.utils.paths import (
    atomic_write_text,
    enforce_under_cwd_and_no_symlink,
    is_under,
    is_under_cwd,
)

TASK_NAME = "aider_polyglot"
RESULT_FILENAME = "soup_result.json"
MAX_RESULT_FILES = 1_000
MAX_RESULT_FILE_BYTES = 1 * 1024 * 1024
MAX_RESULT_TOTAL_BYTES = 32 * 1024 * 1024
DOCKER_PREFLIGHT_TIMEOUT_SECONDS = 10

# Docker's ``--env NAME`` form inherits the value without putting the secret in
# argv.  Keep this list deliberately narrow and aligned with common Aider
# providers plus OpenAI-compatible local servers.
_FORWARDED_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "AZURE_API_BASE",
    "AZURE_API_KEY",
    "AZURE_API_VERSION",
    "GEMINI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
)

_COUNT_FIELDS = (
    "num_error_outputs",
    "test_timeouts",
    "num_malformed_responses",
    "syntax_errors",
    "indentation_errors",
    "num_exhausted_context_windows",
)
_DOCKER_IMAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]*")


class AiderEvalError(ValueError):
    """An actionable, user-facing Aider evaluation failure."""


def _validate_image_name(image: str) -> None:
    """Reject values Docker could parse as options instead of an image."""
    if (
        not isinstance(image, str)
        or not image
        or _DOCKER_IMAGE_RE.fullmatch(image) is None
    ):
        raise AiderEvalError(
            "--image must be a valid Docker image reference and cannot begin with '-'"
        )


def _process_message(process: object) -> str:
    """Return a short subprocess diagnostic without flooding the terminal."""
    stderr = str(getattr(process, "stderr", "") or "").strip()
    stdout = str(getattr(process, "stdout", "") or "").strip()
    message = stderr or stdout
    if len(message) > 500:
        message = message[:497] + "..."
    return message


def preflight_docker(image: str) -> str:
    """Verify the Docker client, daemon, and official benchmark image."""
    _validate_image_name(image)

    docker = shutil.which("docker")
    if docker is None:
        raise AiderEvalError(
            "Docker CLI was not found. Install Docker, start it, and retry."
        )

    try:
        daemon = subprocess.run(  # noqa: S603 -- trusted executable, fixed argv
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=DOCKER_PREFLIGHT_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AiderEvalError("Docker daemon preflight timed out after 10 seconds") from exc
    except OSError as exc:
        raise AiderEvalError(f"Docker preflight failed: {type(exc).__name__}") from exc

    if daemon.returncode != 0:
        detail = _process_message(daemon)
        suffix = f" ({detail})" if detail else ""
        raise AiderEvalError(
            "Docker is installed, but its daemon is unavailable. Start Docker "
            f"and retry{suffix}"
        )

    try:
        image_check = subprocess.run(  # noqa: S603 -- trusted executable, fixed argv
            [docker, "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=DOCKER_PREFLIGHT_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AiderEvalError("Docker image preflight timed out after 10 seconds") from exc
    except OSError as exc:
        raise AiderEvalError(f"Docker image preflight failed: {type(exc).__name__}") from exc

    if image_check.returncode != 0:
        raise AiderEvalError(
            f"Docker image {image!r} was not found. In an Aider source checkout, "
            "run ./benchmark/docker_build.sh first."
        )
    return docker


def prepare_output_dir(output_dir: str | Path) -> Path:
    """Create a contained, non-symlink output directory and return its real path."""
    raw = str(output_dir)
    if not raw or "\x00" in raw:
        raise AiderEvalError("--output must be a non-empty directory path")
    if not is_under_cwd(raw):
        raise AiderEvalError("--output must stay under the current working directory")
    try:
        enforce_under_cwd_and_no_symlink(raw, "--output")
    except (OSError, TypeError, ValueError) as exc:
        raise AiderEvalError(str(exc)) from exc

    path = Path(raw)
    if path.exists() and not path.is_dir():
        raise AiderEvalError("--output exists but is not a directory")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AiderEvalError(f"Could not create --output: {type(exc).__name__}") from exc

    # Recheck after creation so a raced parent symlink cannot escape cwd.
    if not is_under_cwd(path):
        raise AiderEvalError("--output escaped the current working directory")
    return Path(os.path.realpath(path))


def validate_exercises_dir(exercises_dir: str | Path) -> Path:
    """Validate the prepared Aider-AI/polyglot-benchmark checkout."""
    raw = str(exercises_dir)
    if not raw or "\x00" in raw:
        raise AiderEvalError("--exercises-dir must be a non-empty directory path")
    path = Path(os.path.realpath(raw))
    if not path.is_dir():
        raise AiderEvalError(
            "Polyglot exercises directory was not found. Clone "
            "https://github.com/Aider-AI/polyglot-benchmark first."
        )
    return path


def _mount(source: Path, target: str, *, readonly: bool = False) -> str:
    source_text = str(source)
    if "," in source_text:
        raise AiderEvalError("Docker bind-mount paths must not contain commas")
    value = f"type=bind,source={source_text},target={target}"
    if readonly:
        value += ",readonly"
    return value


def build_docker_command(
    *,
    docker: str,
    image: str,
    model: str,
    exercises_dir: Path,
    output_dir: Path,
    threads: int,
    num_tests: int,
    allow_host_services: bool = False,
) -> list[str]:
    """Build the fixed-argv invocation of Aider's upstream benchmark harness."""
    _validate_image_name(image)
    if not model or "\x00" in model:
        raise AiderEvalError("--model must be a non-empty Aider model identifier")
    if not 1 <= threads <= 64:
        raise AiderEvalError("--threads must be between 1 and 64")
    if num_tests == 0 or num_tests < -1:
        raise AiderEvalError("--num-tests must be -1 (all) or a positive integer")

    command = [
        docker,
        "run",
        "--rm",
        "--memory=12g",
        "--memory-swap=12g",
    ]
    if allow_host_services:
        command.append("--add-host=host.docker.internal:host-gateway")
    command.extend(
        [
            "--env",
            "AIDER_DOCKER=1",
            "--env",
            "AIDER_BENCHMARK_DIR=/benchmarks",
            "--mount",
            _mount(exercises_dir, "/benchmarks/polyglot-benchmark", readonly=True),
            "--mount",
            _mount(output_dir, "/results"),
        ]
    )
    for name in _FORWARDED_ENV_NAMES:
        if os.environ.get(name):
            command.extend(["--env", name])
    command.extend(
        [
            image,
            "python3",
            "/aider/benchmark/benchmark.py",
            "/results",
            "--model",
            model,
            "--threads",
            str(threads),
            "--num-tests",
            str(num_tests),
        ]
    )
    return command


def _read_result(path: Path) -> dict[str, Any]:
    """Read one bounded, regular JSON file without following a final symlink."""
    if path.is_symlink():
        raise AiderEvalError(f"Refusing symlinked Aider result: {path}")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AiderEvalError(f"Could not open Aider result {path.name}: {exc}") from exc

    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise AiderEvalError(f"Aider result is not a regular file: {path}")
        if file_stat.st_size > MAX_RESULT_FILE_BYTES:
            raise AiderEvalError(
                f"Aider result is too large ({file_stat.st_size} bytes): {path}"
            )
        with os.fdopen(descriptor, encoding="utf-8") as handle:
            descriptor = -1
            raw = handle.read(MAX_RESULT_FILE_BYTES + 1)
    except (OSError, UnicodeError) as exc:
        raise AiderEvalError(f"Could not read Aider result {path.name}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if len(raw.encode("utf-8")) > MAX_RESULT_FILE_BYTES:
        raise AiderEvalError(f"Aider result is too large: {path}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AiderEvalError(f"Malformed Aider result JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AiderEvalError(f"Aider result JSON must be an object: {path}")
    return payload


def _count(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        or value < 0
        or int(value) != value
    ):
        raise AiderEvalError(f"Invalid {key!r} count in Aider result")
    return int(value)


def parse_aider_results(output_dir: str | Path, *, model: str) -> dict[str, Any]:
    """Aggregate upstream ``.aider.results.json`` files into one Soup row."""
    root = Path(os.path.realpath(output_dir))
    paths = sorted(root.glob("*/exercises/practice/*/.aider.results.json"))
    if not paths:
        raise AiderEvalError(f"No Aider result files were found under {root}")
    if len(paths) > MAX_RESULT_FILES:
        raise AiderEvalError(
            f"Too many Aider result files ({len(paths)}; maximum {MAX_RESULT_FILES})"
        )

    totals = {key: 0 for key in _COUNT_FIELDS}
    passed = 0
    total_bytes = 0
    for path in paths:
        if path.is_symlink():
            raise AiderEvalError(f"Refusing symlinked Aider result: {path}")
        if not is_under(path, root):
            raise AiderEvalError(f"Aider result escaped --output: {path}")
        try:
            total_bytes += path.lstat().st_size
        except OSError as exc:
            raise AiderEvalError(f"Could not stat Aider result: {path}") from exc
        if total_bytes > MAX_RESULT_TOTAL_BYTES:
            raise AiderEvalError("Aider result set is too large to parse safely")

        payload = _read_result(path)
        outcomes = payload.get("tests_outcomes", [])
        if not isinstance(outcomes, list):
            raise AiderEvalError("Invalid 'tests_outcomes' in Aider result")
        if outcomes and outcomes[-1] is True:
            passed += 1
        for key in totals:
            totals[key] += _count(payload, key)

    completed = len(paths)
    details = {
        "completed_tests": completed,
        "passed_tests": passed,
        "failed_tests": completed - passed,
        "error_outputs": totals["num_error_outputs"],
        "test_timeouts": totals["test_timeouts"],
        "malformed_responses": totals["num_malformed_responses"],
        "syntax_errors": totals["syntax_errors"],
        "indentation_errors": totals["indentation_errors"],
        "exhausted_context_windows": totals["num_exhausted_context_windows"],
    }
    return {
        "model": model,
        "task": TASK_NAME,
        "score": passed / completed,
        "errors": completed - passed,
        "details": details,
    }


def write_soup_result(output_dir: Path, row: dict[str, Any]) -> Path:
    """Atomically write the aggregate Soup result inside the output directory."""
    result_path = output_dir / RESULT_FILENAME
    try:
        written = atomic_write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n",
            str(result_path),
            field="Aider result output",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise AiderEvalError(f"Could not write Soup result: {exc}") from exc
    return Path(written)
