"""Tests for issue #1817: ``soup recipes verify --config`` refuses a missing or
unparseable config before torch is imported.

Each case runs in a FRESH interpreter, as ``test_cli_startup_is_light.py``
does: the pytest process has already imported torch via other modules, so an
in-process ``sys.modules`` assertion would be vacuous.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HEAVY = ("torch", "transformers", "accelerate", "peft", "trl", "datasets", "bitsandbytes")


def _probe(argv: list[str], cwd: Path) -> tuple[str, list[str], str]:
    """Invoke the CLI in a clean interpreter; return (exit code, leaked heavy deps, output)."""
    script = (
        "import sys\n"
        "from typer.testing import CliRunner\n"
        "from soup_cli.cli import app\n"
        f"r = CliRunner().invoke(app, {argv!r})\n"
        "try:\n"
        "    err = r.stderr\n"
        "except Exception:\n"
        "    err = ''\n"
        "print('SOUPPROBE|%s|%s|%s' % ("
        "r.exit_code, "
        f"','.join(sorted(m for m in {HEAVY!r} if m in sys.modules)), "
        "(r.output + err).replace('\\n', ' ')))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(cwd),
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    marked = [ln for ln in proc.stdout.splitlines() if ln.startswith("SOUPPROBE|")]
    assert len(marked) == 1, (proc.stdout, proc.stderr)
    _, exit_code, leaked_line, output = marked[0].split("|", 3)
    return exit_code, [m for m in leaked_line.split(",") if m], output


def test_recipes_verify_missing_config_refuses_without_torch(tmp_path: Path) -> None:
    exit_code, leaked, output = _probe(["recipes", "verify", "-c", "nope.yaml"], tmp_path)
    assert exit_code == "3", output
    assert "Config not found" in output
    assert not leaked, f"Refusal pulled in heavy modules: {leaked}"


def test_recipes_verify_unparseable_config_refuses_without_torch(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("suite: [\n", encoding="utf-8")
    exit_code, leaked, output = _probe(["recipes", "verify", "-c", "bad.yaml"], tmp_path)
    assert exit_code == "3", output
    assert "Config does not parse" in output
    assert not leaked, f"Refusal pulled in heavy modules: {leaked}"
