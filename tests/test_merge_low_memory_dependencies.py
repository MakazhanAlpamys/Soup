"""`soup merge --low-memory` must report a missing dependency, not crash (#2227).

On a core install (no ``transformers``), ``_merge_low_memory`` used to bind
``LowMemoryMergeError`` *after* the failing ``transformers`` import. Python still
evaluated the ``except LowMemoryMergeError`` clause against a name that was
never assigned, so the user got ``UnboundLocalError`` instead of the
``Missing dependency`` line and the ``pip install`` hint the regular merge
prints. This test pins the message, not the traceback.
"""

import json
import re
import sys
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()


def _squash(text: str) -> str:
    """Strip SGR codes and collapse the wrapping rich adds at 80 columns."""
    return re.sub(r"\s+", " ", strip_ansi(text))


def _adapter_tree(root: Path) -> tuple[Path, Path]:
    """An adapter directory and a local base directory, as the CLI expects them."""
    adapter = root / "ad"
    base = root / "base"
    adapter.mkdir()
    base.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "base_model_name_or_path": "./base",
            }
        )
    )
    return adapter, base


def test_low_memory_merge_without_transformers_prints_install_hint(
    tmp_path: Path, monkeypatch
) -> None:
    adapter, base = _adapter_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    # A None entry makes ``import transformers`` raise ImportError, which is what
    # a core install looks like.
    monkeypatch.setitem(sys.modules, "transformers", None)

    result = runner.invoke(
        app,
        [
            "merge",
            "--adapter",
            str(adapter),
            "--base",
            str(base),
            "--output",
            "out",
            "--low-memory",
        ],
    )

    output = _squash(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Missing dependency" in output, output
    assert "pip install torch transformers safetensors" in output, output
    assert "UnboundLocalError" not in output, output
