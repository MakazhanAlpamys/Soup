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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from soup_cli.cans.schema import DeployTarget, Manifest
from soup_cli.cans.unpack import extract_can, inspect_can, read_config
from soup_cli.utils.paths import is_under_cwd, quote_path
from soup_cli.utils.redaction import find_redacted_paths
from soup_cli.utils.yaml_limits import MAX_CONFIG_BYTES, load_yaml_file_limited

_RUN_TIMEOUT_SECONDS = 60 * 60 * 24  # 24h max — generous; user kills with ^C
#: Most ``data.train`` entries the confirmation panel lists before it says "+N more".
_MAX_SHOWN_DATA_ENTRIES = 20


@dataclass(frozen=True)
class CanRunResult:
    extract_dir: Path
    train_returncode: int
    deploy_returncode: Optional[int] = None
    env_path: Optional[Path] = None


@dataclass(frozen=True)
class CanTargets:
    """Where a can's config says ``soup train`` will write and read, for the panel.

    ``output`` is the text to show; ``output_problem`` is the refusal message
    :func:`check_can_output` raises for it (``None`` when ``soup can run`` accepts it).
    ``data_train`` holds each ``data.train`` entry with whether it is a local path outside
    the working directory (``soup can run`` shows those with a warning and does not
    refuse them: reading local data is the user's choice).
    """

    output: str
    output_problem: Optional[str]
    data_train: tuple[tuple[str, bool], ...]
    data_train_more: int = 0


def _output_of(config: Mapping[str, Any]) -> object:
    """The ``output`` value ``soup train`` will use for a can's config mapping."""
    if "output" not in config:
        from soup_cli.config.schema import SoupConfig

        return SoupConfig.model_fields["output"].default
    return config["output"]


def _output_problem(output: object) -> Optional[str]:
    """Why ``soup can run`` refuses this ``output``, or ``None`` when it accepts it.

    ``soup train`` creates the checkpoints, logs and final model under ``output``, resolved
    against the directory ``soup can run`` was started in. The can's config is written by
    whoever packed the can, so a path that resolves outside the working directory (an
    absolute path, ``..``, a symlink or a Windows drive-relative or UNC spelling) would
    put those files, and the rotation of ``checkpoint-*`` directories already there, in
    any directory the user can write to. ``is_under_cwd`` resolves symlinks and compares
    with ``os.path.commonpath``, which also treats a different drive or UNC share as
    outside. A value that is not text is refused too: the loader would turn a YAML
    ``!!binary`` value into text, so it cannot be left for the loader to judge.
    """
    if not isinstance(output, str):
        return (
            "this can's config sets output to a value that is not a text path "
            f"({type(output).__name__}); soup can run needs a plain path under the "
            "working directory"
        )
    if "\x00" in output:
        # POSIX path resolution raises on it, Windows resolution keeps it: say so here
        # rather than leave the answer to the platform.
        return (
            f"this can's config sets output to {quote_path(output)}, which contains a NUL "
            "byte; soup can run needs a plain path under the working directory"
        )
    if is_under_cwd(output):
        return None
    return (
        f"this can's config sets output to {quote_path(output)}, which resolves outside "
        f"the working directory ({quote_path(os.getcwd())}), and soup can run does not "
        "write training output there. Run it from the directory the output should go "
        "to, or take config.yaml out of the can (a .tar.gz), set output to a path under "
        "your working directory and run soup train on it"
    )


def _refuse_redacted(config: Mapping[str, Any]) -> None:
    """Refuse a config that holds a ``<redacted>`` placeholder (``ValueError``)."""
    holes = find_redacted_paths(config)
    if holes:
        raise ValueError(
            "this can holds redacted value(s) at: " + ", ".join(holes[:20])
            + (" (+more)" if len(holes) > 20 else "")
            + ". They were removed when the can was packed; supply them in the can's "
            "config.yaml (a .tar.gz) and run soup train on it instead of soup can run"
        )


def check_can_output(config: Mapping[str, Any]) -> None:
    """Refuse a can whose config ``output`` resolves outside the working directory.

    Raises ``ValueError`` naming the value; see :func:`_output_problem`.
    """
    problem = _output_problem(_output_of(config))
    if problem is not None:
        raise ValueError(problem)


def describe_can_targets(config: Mapping[str, Any]) -> CanTargets:
    """What the confirmation panel shows: ``output`` and every ``data.train`` entry."""
    output = _output_of(config)
    data = config.get("data")
    raw_train = data.get("train") if isinstance(data, Mapping) else None
    if isinstance(raw_train, str):
        entries: list[object] = [raw_train]
    elif isinstance(raw_train, list):
        entries = list(raw_train)
    else:
        entries = []
    shown = []
    for entry in entries[:_MAX_SHOWN_DATA_ENTRIES]:
        text = entry if isinstance(entry, str) else str(entry)
        # An entry with "://" names a remote source and anything else without a data
        # file suffix may be a Hub dataset id; neither is a path the warning is about,
        # and a relative id resolves inside the working directory anyway.
        outside = "://" not in text and not is_under_cwd(text)
        shown.append((text, outside))
    return CanTargets(
        output=output if isinstance(output, str) else repr(output),
        output_problem=_output_problem(output),
        data_train=tuple(shown),
        data_train_more=max(0, len(entries) - _MAX_SHOWN_DATA_ENTRIES),
    )


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


_TIMEOUT_RC = 124  # `coreutils timeout` convention — surfaces in CanRunResult


def _run_subprocess(argv: list[str], *, cwd: Optional[Path] = None) -> int:
    """Run a child subprocess and return its returncode.

    Re-raises ``OSError`` (e.g. binary not on PATH) so the orchestrator
    surfaces an actionable message. ``subprocess.TimeoutExpired`` after the
    24h cap is converted to ``_TIMEOUT_RC`` (124) so callers can tell
    "timed out" from "exited cleanly" without unhandled tracebacks.
    """
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd) if cwd else None,
            timeout=_RUN_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired:
        return _TIMEOUT_RC
    return proc.returncode


def _deploy_target(target: DeployTarget, extract_dir: Path) -> int:
    """Run the deploy step for a single DeployTarget. Returns rc."""
    if target.kind == "ollama":
        if not target.name:
            raise ValueError("deploy_target kind=ollama requires 'name'")
        # Find the GGUF artifact in the extract dir, then verify it really
        # lives under extract_dir (a crafted can with a symlink could
        # otherwise point rglob at an arbitrary path on disk).
        extract_real = os.path.realpath(str(extract_dir))
        gguf_path: Optional[Path] = None
        for candidate in extract_dir.rglob("*.gguf"):
            cand_real = os.path.realpath(str(candidate))
            try:
                common = os.path.commonpath([extract_real, cand_real])
            except ValueError:
                continue
            if common != extract_real:
                continue
            gguf_path = Path(cand_real)
            break
        if gguf_path is None:
            raise ValueError(
                "deploy_target kind=ollama requires a *.gguf inside the can"
            )
        return _run_subprocess([
            sys.executable, "-m", "soup_cli.cli",
            "deploy", "ollama",
            "--gguf", str(gguf_path),
            "--name", target.name,
        ])
    if target.kind == "gguf":
        # Already extracted — just confirm presence, but verify the manifest's
        # path really resolves inside extract_dir first. ``target.path`` comes
        # from the can, so an absolute path (pathlib drops the left operand
        # when the right one is absolute), a ``..`` segment, or a symlink
        # planted in the can would otherwise reach an arbitrary file on disk.
        # Same containment pattern as the kind=ollama branch above; the
        # DeployTarget field validator is the first line, this is the second.
        if not target.path:
            raise ValueError("deploy_target kind=gguf requires 'path'")
        extract_real = os.path.realpath(str(extract_dir))
        gguf_real = os.path.realpath(str(extract_dir / target.path))
        try:
            common = os.path.commonpath([extract_real, gguf_real])
        except ValueError:  # different drives on Windows — never contained
            common = ""
        if common != extract_real or gguf_real == extract_real:
            raise ValueError(
                f"deploy gguf path '{target.path}' escapes the can extract "
                f"dir - refusing"
            )
        gguf_path = Path(gguf_real)
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
    confirm_callback: Optional[Callable[[Manifest], bool]] = None,
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
        pull arbitrary HF datasets or HTTPS payloads. A can whose config ``output``
        resolves outside the working directory is refused (``ValueError``) before
        anything is extracted, and again for the extracted ``config.yaml``.
    """
    src = _validate_can_path(can_path)

    # Read manifest first — surface schema errors before any side effects.
    manifest = inspect_can(str(src))
    # The extracted config.yaml is handed to ``soup train``, whose loader has
    # no size or expansion limit of its own; read it through the capped reader
    # first so an oversized or alias-amplified config is refused here.
    config = read_config(str(src))
    # A value that ``soup can pack`` redacted is not in the can: running it would send the
    # placeholder to a server. Refuse before anything is extracted, fetched or trained.
    _refuse_redacted(config)
    # Refuse an output outside the working directory before anything is extracted or
    # asked: the confirmation would otherwise approve a run that is then refused.
    check_can_output(config)

    if not yes:
        if confirm_callback is None:
            # ValueError (not PermissionError) so a caller wrapping in a
            # broad ``except OSError`` cannot silently swallow the gate —
            # PermissionError is an OSError subclass. The CLI handler
            # surfaces the message verbatim.
            raise ValueError(
                "soup can run requires --yes or an explicit confirm_callback "
                "(this command auto-downloads data + auto-trains)"
            )
        if not confirm_callback(manifest):
            raise ValueError("user declined to run the can")

    # Pre-validate capture_env_to before creating temporary extract dir.
    env_target: Optional[Path] = None
    if capture_env_to:
        env_target = Path(capture_env_to)
        if not is_under_cwd(env_target):
            raise ValueError(
                f"capture_env_to '{capture_env_to}' is outside cwd - refusing"
            )

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

    try:
        extract_can(str(src), str(owned_dir))

        # The file ``soup train`` is about to read is the one extraction left at
        # ``config.yaml``, not the archive member read for the checks above, so the
        # extracted file is checked too, with the loader ``soup train`` itself uses.
        extracted_config = load_yaml_file_limited(
            owned_dir / "config.yaml", "config.yaml", max_bytes=MAX_CONFIG_BYTES
        ) or {}
        if not isinstance(extracted_config, dict):
            raise ValueError("config.yaml must deserialise to a mapping")
        _refuse_redacted(extracted_config)
        check_can_output(extracted_config)

        # Optional env capture.
        env_path: Optional[Path] = None
        if env_target is not None:
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
    except Exception:
        # If any later step fails after we created an owned tmp dir, clean up
        # to prevent leaks of extracted (potentially large) data.
        if extract_dir is None:
            cleanup_extract_dir(owned_dir)
        raise

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

    def _under(base: str) -> bool:
        try:
            return os.path.commonpath([real, base]) == base and real != base
        except ValueError:
            return False

    if not (_under(tmp_real) or _under(cwd_real)):
        return
    shutil.rmtree(extract_dir, ignore_errors=True)
