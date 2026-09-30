"""Which tree a benchmark row measured (#1387).

``soup-cli`` reports a release version that does not move between commits, so a
harness records the commit of the ``soup_cli`` it imported. Resolved from the
package's own location, not the harness file's: the two can come from different
checkouts (a ``PYTHONPATH`` override, an editable install elsewhere).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout


def source_sha(*, dirty_pathspec: Optional[str] = None) -> str:
    """The commit of the tree ``soup_cli`` was imported from, or ``unknown``.

    ``-dirty`` marks uncommitted changes on top of that commit; ``dirty_pathspec``
    limits which ones count (``"src"`` keeps a harness's own output file at the
    repo root from marking a clean tree dirty).

    A ``soup-cli`` wheel installed into a gitignored ``.venv/`` inside the checkout
    resolves to the checkout too, but the code measured is the wheel's, so
    ``soup_cli/__init__.py`` must be a file the checkout tracks.
    """
    import soup_cli

    package_file = getattr(soup_cli, "__file__", None)
    if package_file is None:
        return "unknown"
    try:
        package_path = Path(package_file).resolve()
        root = _git(["rev-parse", "--show-toplevel"], package_path.parent).strip()
        if not root:
            return "unknown"
        _git(["ls-files", "--error-unmatch", str(package_path)], Path(root))
        commit = _git(["rev-parse", "HEAD"], Path(root)).strip().lower()
        status = ["status", "--porcelain", "--untracked-files=normal"]
        if dirty_pathspec is not None:
            status += ["--", dirty_pathspec]
        dirty = _git(status, Path(root))
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit
