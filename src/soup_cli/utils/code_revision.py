"""Best-effort, local Git identity of the directory that starts a training run."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from soup_cli.utils.tool_paths import find_tool, tool_environment

_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_PROBE_TIMEOUT_SECONDS = 2.0
_ROUTING_ENV = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE",
)


@dataclass(frozen=True)
class CodeRevision:
    """A startup observation, not an archive or integrity attestation."""

    commit: str | None = None
    dirty: bool | None = None

    def __post_init__(self) -> None:
        if self.commit is not None and (
            not isinstance(self.commit, str) or _COMMIT.fullmatch(self.commit) is None
        ):
            raise ValueError("code commit must be a full lowercase Git object ID or None")
        if self.dirty is not None and type(self.dirty) is not bool:
            raise ValueError("code dirty state must be bool or None")
        if self.commit is None and self.dirty is not None:
            raise ValueError("code dirty state must be unknown without a commit")


def _bounded_git_output(argv: list[str], env: dict[str, str], deadline: float, limit: int) -> bytes:
    """Read a small prefix, cancelling status as soon as any change is observed.

    A reader thread permits a deadline on pipe reads on Windows as well as Unix;
    it never reads more than ``limit`` bytes. The child is stopped and reaped even
    on timeout. HEAD needs at most 64 hex digits and a newline; status and filter
    names need one byte, never changed paths or configuration values.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired("git", _PROBE_TIMEOUT_SECONDS)
    process = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=0, env=env,
    )
    finished = threading.Event()
    output = bytearray()
    failures: list[OSError] = []

    def read_prefix() -> None:
        try:
            while len(output) < limit:
                part = process.stdout.read(limit - len(output))
                if not part:
                    break
                output.extend(part)
        except OSError as exc:
            failures.append(exc)
        finally:
            finished.set()

    reader = threading.Thread(target=read_prefix, name="soup-git-observation", daemon=True)
    try:
        reader.start()
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not finished.wait(remaining):
            raise subprocess.TimeoutExpired("git", _PROBE_TIMEOUT_SECONDS)
        if failures:
            raise failures[0]
        if len(output) == limit:
            if limit != 1:
                raise ValueError("Git object ID output exceeds its bound")
            # A status prefix is sufficient. Do not drain the rest of the pipe.
            return bytes(output)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("git", _PROBE_TIMEOUT_SECONDS)
        code = process.wait(timeout=remaining)
        if code:
            raise subprocess.CalledProcessError(code, argv)
        return bytes(output)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        reader.join(timeout=0.1)
        process.stdout.close()


def capture_code_revision(cwd: str | Path | None = None) -> CodeRevision:
    """Observe HEAD and tracked/untracked changes with a two-second total deadline.

    Git is optional. Missing Git, an unborn HEAD, a non-repository or a failed
    observation stays unknown. Re-read HEAD to reject a concurrent commit rather
    than pair one commit's ID with another commit's status. This cannot freeze
    edits made after the observation. Repository fsmonitor hooks are disabled,
    Git's ownership checks are preserved, and no paths or file contents are stored.
    Local content-filter drivers leave the commit known but skip status, so the
    observation does not execute their repository-configured commands.
    """
    unknown = CodeRevision()
    deadline = time.monotonic() + _PROBE_TIMEOUT_SECONDS
    env = tool_environment(drop=_ROUTING_ENV)
    # `git status` otherwise may refresh/write the index as an optimization.
    env["GIT_OPTIONAL_LOCKS"] = "0"

    try:
        directory = os.fspath(cwd) if cwd is not None else os.getcwd()
        program = find_tool("git")
        if program is None:
            return unknown

        def git(*args: str) -> str:
            try:
                output = _bounded_git_output(
                    [program, "-c", "core.fsmonitor=false", "-C", directory, *args],
                    env, deadline, 1 if args[0] == "status" or "--get-regexp" in args else 66,
                )
            except subprocess.CalledProcessError as exc:
                # Config lookups exit 1 when the requested key/name is absent.
                if args[0] != "config" or exc.returncode != 1:
                    raise
                output = b""
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("git", _PROBE_TIMEOUT_SECONDS)
            return output.decode("ascii")

        commit = git("rev-parse", "--verify", "HEAD").strip()
        if _COMMIT.fullmatch(commit) is None:
            return unknown
        # Read names only, including local includes; never remote URLs or values.
        has_filters = bool(git(
            "config", "--local", "--includes", "--name-only", "--get-regexp", r"^filter\.",
        ))
        # Older Git refuses --worktree on linked trees without this extension.
        if not has_filters and git(
            "config", "--local", "--includes", "--bool", "--get", "extensions.worktreeConfig",
        ).strip() == "true":
            has_filters = bool(git(
                "config", "--worktree", "--includes", "--name-only", "--get-regexp", r"^filter\.",
            ))
        dirty = None if has_filters else bool(git(
            "status", "--porcelain=v1", "--untracked-files=normal", "--ignore-submodules=none",
        ))
        if git("rev-parse", "--verify", "HEAD").strip() != commit:
            return unknown
        return CodeRevision(commit=commit, dirty=dirty)
    except (OSError, ValueError, subprocess.SubprocessError):
        return unknown
