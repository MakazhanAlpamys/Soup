"""`soup merge --low-memory` on an install without `transformers` names the missing package."""

import json
import sys

from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


def test_missing_transformers_prints_install_hint(tmp_path, monkeypatch):
    adapter = tmp_path / "ad"
    base = tmp_path / "base"
    adapter.mkdir()
    base.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8, "lora_alpha": 16}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    # A None entry makes `import transformers` raise ImportError, as on a core install.
    monkeypatch.setitem(sys.modules, "transformers", None)

    result = CliRunner().invoke(
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

    output = strip_ansi(result.output)
    assert not isinstance(result.exception, UnboundLocalError), repr(result.exception)
    assert result.exit_code == 1, (output, repr(result.exception))
    assert "UnboundLocalError" not in output
    assert "Missing dependency" in output
    assert "pip install torch transformers safetensors" in output
