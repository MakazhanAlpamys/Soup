"""Tests for issue #1817: soup bench train --warmup >= --steps refusal is light."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HEAVY = ("torch", "transformers", "accelerate", "peft", "trl", "datasets", "bitsandbytes")


def test_bench_train_warmup_leaves_nothing_refuses_without_torch(tmp_path: Path):
    """Refuse --warmup >= --steps before torch or model configuration is loaded."""
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(
        "model: test-model\ntask: sft\ndata:\n  train: data.jsonl\n",
        encoding="utf-8",
    )
    argv = ["bench", "train", "-c", str(config_file), "--steps", "2", "--warmup", "5"]
    probe = (
        "import sys\n"
        "from typer.testing import CliRunner\n"
        "from soup_cli.cli import app\n"
        f"r = CliRunner().invoke(app, {argv!r})\n"
        "print('SOUPPROBE|%s|%s|%s' % ("
        "r.exit_code, "
        "','.join(sorted(m for m in {heavy!r} if m in sys.modules)), "
        "r.output.replace('\\n', ' ')))\n"
    ).format(heavy=HEAVY)

    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    marked = [ln for ln in proc.stdout.splitlines() if ln.startswith("SOUPPROBE|")]
    assert len(marked) == 1, (proc.stdout, proc.stderr)
    _, exit_code, leaked_line, output = marked[0].split("|", 3)
    assert exit_code == "1", f"Expected exit code 1, got {exit_code}: {output}"
    assert "--steps 2 leaves nothing after --warmup 5" in output
    leaked = [m for m in leaked_line.split(",") if m]
    assert not leaked, f"Refusal pulled in heavy modules: {leaked}"
