"""Issue #3209 — macOS: sandboxed code cannot write in its own working directory.

On macOS the per-user temporary directory resolves to
``/private/var/folders/...``, so the static profile's
``(subpath "/var/folders")`` entry never matches and a sandboxed child
that creates a file in its own working directory fails with
``PermissionError: [Errno 1] Operation not permitted``. The fix names
the run's own resolved directory in a per-run profile.
"""

from __future__ import annotations

import os
import sys

import pytest


def test_per_run_profile_names_the_resolved_run_directory():
    """The per-run profile allows writes in exactly the run's directory."""
    from soup_cli.trainer.rewards import (
        MACOS_SANDBOX_PROFILE,
        _macos_sandbox_profile_for_dir,
    )

    run_dir = "/private/var/folders/xx/soup-sandbox-abc123"
    profile = _macos_sandbox_profile_for_dir(run_dir)

    assert f'(subpath "{run_dir}")' in profile
    # Everything else in the static profile is kept verbatim.
    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    # Narrower than the alternative fix: the static entry is NOT widened
    # to /private/var/folders, which would allow every other temp dir too.
    assert '(subpath "/private/var/folders")' not in profile
    assert profile != MACOS_SANDBOX_PROFILE


def test_per_run_profile_resolves_symlinked_paths(monkeypatch, tmp_path):
    """A symlinked tmpdir is named by its real path, not the symlink spelling."""
    from soup_cli.trainer.rewards import _macos_sandbox_profile_for_dir

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    profile = _macos_sandbox_profile_for_dir(str(link))

    assert f'(subpath "{real}")' in profile
    assert f'(subpath "{link}")' not in profile


def test_sandboxed_subprocess_uses_per_run_profile_on_macos(monkeypatch):
    """On darwin + sandbox-exec, argv carries the per-run profile for the run's cwd."""
    from soup_cli.trainer import rewards

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(rewards, "_get_isolation_strategy", lambda: "sandbox-exec")
    monkeypatch.setattr(rewards, "_ISOLATION_STRATEGY_CACHE", "sandbox-exec")
    started: list[dict] = []

    def popen(argv, **kwargs):
        started.append({"argv": list(argv), "cwd": kwargs.get("cwd")})
        raise OSError("not started in this test")

    monkeypatch.setattr(rewards.subprocess, "Popen", popen)

    result = rewards._run_sandboxed_subprocess([sys.executable, "-c", "pass"])

    assert result.launch_failed
    assert started[0]["argv"][0] == rewards._MACOS_SANDBOX_EXEC
    profile = started[0]["argv"][2]
    # The profile names the exact cwd the child was launched with.
    assert f'(subpath "{os.path.realpath(started[0]["cwd"])}")' in profile
    assert '(subpath "/private/var/folders")' not in profile


@pytest.mark.skipif(
    sys.platform != "darwin" or not os.path.isfile("/usr/bin/sandbox-exec"),
    reason="macOS-only: needs a real sandbox-exec (issue #3209)",
)
def test_macos_child_writes_file_in_its_working_directory():
    """Real macOS row: a sandboxed child writes a file in its cwd and reads it back.

    Fails on main (PermissionError: Operation not permitted). Runs only in
    the macOS cell, so the pull request needs the full matrix.
    """
    from soup_cli.trainer.rewards import _run_sandboxed_subprocess

    result = _run_sandboxed_subprocess(
        [
            sys.executable,
            "-c",
            "open('probe.txt', 'w').write('ok'); print(open('probe.txt').read())",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok"
