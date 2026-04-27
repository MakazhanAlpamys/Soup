"""Run a ``.can`` end-to-end: extract, train, optionally deploy (v0.33.0 #34).

Orchestrator wired to existing CLI primitives via subprocess. Keeps the
training entry point a single source of truth (``soup train``) instead of
re-implementing the trainer dispatch.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from soup_cli.cans.schema import DeployTarget
from soup_cli.cans.unpack import extract_can, inspect_can
from soup_cli.utils.paths import is_under_cwd

_RUN_TIMEOUT_SECONDS = 60 * 60 * 24  # 24h max — generous; user kills with ^C


@dataclass(frozen=True)
class CanRunResult:
    extract_dir: Path
    train_returncode: int
    deploy_returncode: Optional[int] = None
    env_path: Optional[Path] = None


def capture_env(out_path: Path) -> Path:
    """Write Python / pip / accelerator details to ``out_path``.

    Captures: ``python --version``, ``pip freeze``, and a coarse accelerator
    summary derived from ``utils/gpu.detect_device``. Failures fall back to
    a stub line so the can-run flow is never blocked by env capture.
    """
    lines: list[str] = [
        "# soup-cli can env capture",
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
    ]
    try:
        from soup_cli.utils.gpu import detect_device
        device, info = detect_device()
        lines.append(f"device={device}")
        lines.append(f"device_info={info}")
    except Exception as exc:  # noqa: BLE001 — env capture is best-effort
        lines.append(f"device=unknown ({exc})")

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if proc.returncode == 0:
            lines.append("# pip freeze:")
            lines.append(proc.stdout.strip())
        else:
            lines.append(f"# pip freeze failed: rc={proc.returncode}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        lines.append(f"# pip freeze failed: {exc}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _validate_can_path(can_path: str) -> Path:
    path = Path(can_path)
    if not is_under_cwd(path):
        raise ValueError(f"can path '{can_path}' is outside cwd - refusing")
    if not path.exists():
        raise FileNotFoundError(f"can not found: {can_path}")
    return path


def _run_subprocess(argv: list[str], *, cwd: Optional[Path] = None) -> int:
    """Run a child subprocess and return its returncode. Re-raises on
    OSError so the orchestrator surfaces an actionable message instead of
    hiding behind a generic non-zero exit.
    """
    proc = subprocess.run(
        argv, cwd=str(cwd) if cwd else None,
        timeout=_RUN_TIMEOUT_SECONDS, check=False,
    )
    return proc.returncode


def _deploy_target(target: DeployTarget, extract_dir: Path) -> int:
    """Run the deploy step for a single DeployTarget. Returns rc."""
    if target.kind == "ollama":
        if not target.name:
            raise ValueError("deploy_target kind=ollama requires 'name'")
        # Find the GGUF artifact in the extract dir
        gguf_files = list(extract_dir.rglob("*.gguf"))
        if not gguf_files:
            raise ValueError(
                "deploy_target kind=ollama requires a *.gguf in the can"
            )
        return _run_subprocess([
            sys.executable, "-m", "soup_cli.cli",
            "deploy", "ollama",
            "--gguf", str(gguf_files[0]),
            "--name", target.name,
        ])
    if target.kind == "gguf":
        # Already extracted — just confirm presence.
        if not target.path:
            raise ValueError("deploy_target kind=gguf requires 'path'")
        gguf_path = (extract_dir / target.path).resolve()
        if not gguf_path.exists():
            raise FileNotFoundError(f"deploy gguf path not found: {gguf_path}")
        return 0
    if target.kind == "vllm":
        # Live serve needs a port; orchestrator only validates here so users
        # opt in to long-running serve commands explicitly.
        return 0
    raise ValueError(f"unknown deploy target kind: {target.kind}")


def run_can(
    can_path: str, *,
    yes: bool = False,
    deploy: bool = False,
    extract_dir: Optional[str] = None,
    capture_env_to: Optional[str] = None,
    train_argv_extra: Optional[list[str]] = None,
    confirm_callback=None,
) -> CanRunResult:
    """Extract a can, run ``soup train`` against the embedded config, optionally
    run the embedded deploy targets.

    Args:
        can_path: path to the ``.can`` file (must stay under cwd).
        yes: skip the confirmation prompt (matches the ``--yes`` flag in
            ``soup train`` / ``soup autopilot``).
        deploy: also run any ``deploy_targets`` from the manifest.
        extract_dir: where to extract the can. Defaults to a fresh temp dir.
        capture_env_to: write env summary to this path before training.
        train_argv_extra: appended to the train invocation (e.g. ``--gpus 1``).
        confirm_callback: called with the manifest if ``yes`` is False; should
            return True to proceed, False to abort. Defaults to a no-op
            confirm so tests don't deadlock.

    Returns:
        ``CanRunResult`` with the extract dir, train rc, and (optional) deploy rc.

    Security:
        Confirmation panel is mandatory for non-``yes`` runs because this
        auto-downloads data and auto-trains. A crafted can could otherwise
        pull arbitrary HF datasets or HTTPS payloads.
    """
    src = _validate_can_path(can_path)

    # Read manifest first — surface schema errors before any side effects.
    manifest = inspect_can(str(src))

    if not yes:
        if confirm_callback is None:
            raise PermissionError(
                "soup can run requires --yes or an explicit confirm_callback "
                "(this command auto-downloads data + auto-trains)"
            )
        if not confirm_callback(manifest):
            raise PermissionError("user declined to run the can")

    # Resolve extraction destination.
    if extract_dir is None:
        owned_dir = Path(tempfile.mkdtemp(prefix="soup-can-run-"))
    else:
        candidate = Path(extract_dir)
        if not is_under_cwd(candidate):
            raise ValueError(
                f"extract_dir '{extract_dir}' is outside cwd - refusing"
            )
        candidate.mkdir(parents=True, exist_ok=True)
        owned_dir = candidate

    extract_can(str(src), str(owned_dir))

    # Optional env capture.
    env_path: Optional[Path] = None
    if capture_env_to:
        env_target = Path(capture_env_to)
        if not is_under_cwd(env_target):
            raise ValueError(
                f"capture_env_to '{capture_env_to}' is outside cwd - refusing"
            )
        env_path = capture_env(env_target)

    # Train via subprocess against the embedded config.
    train_argv = [
        sys.executable, "-m", "soup_cli.cli", "train",
        "--config", str(owned_dir / "config.yaml"),
        "--yes",  # the can-run wrapper already confirmed
    ]
    if train_argv_extra:
        train_argv.extend(train_argv_extra)
    train_rc = _run_subprocess(train_argv)

    # Optional deploy after training.
    deploy_rc: Optional[int] = None
    if deploy and train_rc == 0 and manifest.deploy_targets:
        for target in manifest.deploy_targets:
            deploy_rc = _deploy_target(target, owned_dir)
            if deploy_rc != 0:
                break

    return CanRunResult(
        extract_dir=owned_dir,
        train_returncode=train_rc,
        deploy_returncode=deploy_rc,
        env_path=env_path,
    )


def cleanup_extract_dir(extract_dir: Path) -> None:
    """Remove an extraction directory created by :func:`run_can`.

    Safety: only deletes if the path is under the system tempdir or under
    cwd, never an absolute path elsewhere. Mirrors the pattern in
    ``commands/export.py`` which guards merge_dir cleanup.
    """
    real = os.path.realpath(str(extract_dir))
    tmp_real = os.path.realpath(tempfile.gettempdir())
    cwd_real = os.path.realpath(os.getcwd())
    if not (real.startswith(tmp_real + os.sep) or real.startswith(cwd_real + os.sep)):
        return
    shutil.rmtree(extract_dir, ignore_errors=True)
