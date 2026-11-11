"""Best-effort, local Git identity of the directory that starts a training run."""

from __future__ import annotations

import os
import re
import subprocess
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


def capture_code_revision(cwd: str | Path | None = None) -> CodeRevision:
    """Observe HEAD and tracked/untracked changes with a two-second total deadline.

    Git is optional. Missing Git, an unborn HEAD, a non-repository or a failed
    observation stays unknown. Re-read HEAD to reject a concurrent commit rather
    than pair one commit's ID with another commit's status. This cannot freeze
    edits made after the observation. No paths, file contents or diff are stored.
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
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("git", _PROBE_TIMEOUT_SECONDS)
            result = subprocess.run(
                [program, "-C", directory, *args],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", timeout=remaining, env=env,
                check=True,
            )
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("git", _PROBE_TIMEOUT_SECONDS)
            return result.stdout.strip()

        commit = git("rev-parse", "--verify", "HEAD")
        if _COMMIT.fullmatch(commit) is None:
            return unknown
        status = git("status", "--porcelain=v1", "--untracked-files=normal",
                     "--ignore-submodules=none")
        if git("rev-parse", "--verify", "HEAD") != commit:
            return unknown
        return CodeRevision(commit=commit, dirty=bool(status))
    except (OSError, ValueError, subprocess.SubprocessError):
        return unknown
