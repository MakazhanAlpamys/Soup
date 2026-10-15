"""Tests for issue #1817: ``--cuda-graphs`` on a non-CUDA device is refused before
the model loads, and for ``soup infer --device cpu`` before torch is imported.

Each case runs in a FRESH interpreter, as ``test_cli_startup_is_light.py``
does, so the ``sys.modules`` assertion is not vacuous.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HEAVY = ("torch", "transformers", "accelerate", "peft", "trl", "datasets", "bitsandbytes")
MSG = "CUDA graphs require every model tensor on a single CUDA device"


def _probe(argv: list[str], cwd: Path, prelude: str = "") -> tuple[str, list[str], str]:
    """Invoke the CLI in a clean interpreter; return (exit code, leaked heavy deps, output).

    ``prelude`` is Python run after the CLI import and before the invocation
    (used to stub the device detector and the model loader).
    """
    script = (
        "import sys\n"
        "from typer.testing import CliRunner\n"
        "from soup_cli.cli import app\n"
        f"{prelude}"
        f"r = CliRunner().invoke(app, {argv!r})\n"
        "print('SOUPPROBE|%s|%s|%s' % ("
        "r.exit_code, "
        f"','.join(sorted(m for m in {HEAVY!r} if m in sys.modules)), "
        "r.output.replace('\\n', ' ')))\n"
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


def test_infer_cuda_graphs_on_cpu_refuses_without_torch(tmp_path: Path) -> None:
    (tmp_path / "prompts.jsonl").write_text(json.dumps({"prompt": "hi"}) + "\n", encoding="utf-8")
    argv = [
        "infer",
        "--model",
        "owner/repo-name",
        "--input",
        "prompts.jsonl",
        "--output",
        "out.jsonl",
        "--device",
        "cpu",
        "--cuda-graphs",
    ]
    exit_code, leaked, output = _probe(argv, tmp_path)
    assert exit_code == "1", output
    assert MSG in output
    assert "Loading model" not in output
    assert not leaked, f"Refusal pulled in heavy modules: {leaked}"


def test_bench_infer_cuda_graphs_on_cpu_refuses_before_model_load(tmp_path: Path) -> None:
    # ``soup bench`` has no --device: it always auto-detects, and detection itself
    # imports torch. Stub it to "cpu" so the probe isolates the --cuda-graphs path,
    # and stub the loader so a refusal that arrives after it would be unmistakable.
    prelude = (
        "import soup_cli.utils.gpu as _gpu\n"
        "_gpu.detect_device = lambda backend=None: ('cpu', 'cpu')\n"
        "import soup_cli.commands.infer as _infer\n"
        "def _boom(*a, **k):\n"
        "    raise AssertionError('LOADER CALLED')\n"
        "_infer._load_model = _boom\n"
    )
    argv = ["bench", "owner/repo-name", "--backend", "transformers", "--cuda-graphs"]
    exit_code, leaked, output = _probe(argv, tmp_path, prelude)
    assert exit_code == "1", output
    assert MSG in output
    assert "LOADER CALLED" not in output
    assert "Loading model" not in output
    assert not leaked, f"Refusal pulled in heavy modules: {leaked}"
