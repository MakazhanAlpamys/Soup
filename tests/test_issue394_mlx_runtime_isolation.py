"""Runtime regression coverage for the standalone ``soup-cli[mlx]`` path (#394)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TRAIN_EXTRA_ROOTS = (
    "torch",
    "datasets",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
)


def test_mlx_cli_route_does_not_touch_transformers_training_stack(tmp_path: Path) -> None:
    """The real CLI must reach the MLX wrapper without importing ``[train]``.

    The two handled torch probes in ``utils.gpu`` are tracked separately by #423.
    Blocking them here models an MLX-only environment and lets this test pin the
    #394 boundary: local data loading and backend dispatch must reach ``_require_mlx``
    without attempting datasets, TRL, PEFT, accelerate, or bitsandbytes.
    """
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Say hi"},
                    {"role": "assistant", "content": "Hi"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "soup.yaml"
    config_path.write_text(
        "base: mlx-community/tiny\n"
        "task: sft\n"
        "backend: mlx\n"
        "data:\n"
        f"  train: {json.dumps(str(data_path))}\n"
        "  format: chatml\n"
        "  val_split: 0\n"
        "training:\n"
        "  epochs: 1\n"
        "  lr: 0.0002\n"
        "  batch_size: 1\n"
        "  quantization: none\n"
        f"output: {json.dumps(str(tmp_path / 'out'))}\n",
        encoding="utf-8",
    )

    probe = f"""
import importlib.abc
import json
import sys
from pathlib import Path
from unittest.mock import patch

blocked = []
blocked_roots = {TRAIN_EXTRA_ROOTS!r}

class BlockTrainingStack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition('.')[0]
        if root in blocked_roots or root == 'mlx':
            blocked.append(root)
            raise ModuleNotFoundError(f'blocked by #394 probe: {{fullname}}', name=fullname)
        return None

sys.meta_path.insert(0, BlockTrainingStack())

from typer.testing import CliRunner
from soup_cli.cli import app

with patch('pathlib.Path.home', return_value=Path({str(tmp_path)!r})):
    result = CliRunner().invoke(
        app,
        ['train', '--config', {str(config_path)!r}, '--yes'],
    )
payload = {{
    'exit_code': result.exit_code,
    'exception': str(result.exception) if result.exception else '',
    'blocked': blocked,
    'sft_loaded': 'soup_cli.trainer.sft' in sys.modules,
}}
print('SOUP394|' + json.dumps(payload, sort_keys=True))
"""
    env = os.environ.copy()
    source_root = str(Path(__file__).parents[1] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
        env=env,
    )

    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    marked = [line for line in proc.stdout.splitlines() if line.startswith("SOUP394|")]
    assert len(marked) == 1, (proc.stdout, proc.stderr)
    payload = json.loads(marked[0].removeprefix("SOUP394|"))
    assert payload["exit_code"] == 1
    assert "MLX backend requires" in payload["exception"]
    assert set(payload["blocked"]) <= {"torch", "mlx"}
    assert "mlx" in payload["blocked"]
    assert payload["sft_loaded"] is False
