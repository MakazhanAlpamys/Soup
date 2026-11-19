"""The README Docker example passes --yes (#2229).

Since v0.77.0 a non-interactive `soup train` (stdin is not a terminal, as under
`docker run` without `-it`) stops before training unless `--yes` is given, so the
front-door example must carry it in README.md and in every translation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
READMES = sorted([ROOT / "README.md", *ROOT.glob("README.*.md")])
FENCE = re.compile(r"^```")


def _docker_train_lines(path: Path) -> list[str]:
    lines: list[str] = []
    in_code = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if FENCE.match(raw.strip()):
            in_code = not in_code
            continue
        line = raw.strip()
        if in_code and line.startswith("docker run") and "soup train" in line:
            lines.append(line)
    return lines


def test_every_readme_is_checked() -> None:
    names = {path.name for path in READMES}
    assert "README.md" in names
    assert len(names) >= 5, names


@pytest.mark.parametrize("readme", READMES, ids=lambda path: path.name)
def test_docker_train_example_passes_yes(readme: Path) -> None:
    lines = _docker_train_lines(readme)
    assert lines, f"{readme.name}: no `docker run ... soup train` line in a code block"
    for line in lines:
        tokens = line.split()
        assert "--yes" in tokens or "-y" in tokens, f"{readme.name}: {line}"
