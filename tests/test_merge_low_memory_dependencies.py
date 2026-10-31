"""`soup merge --low-memory` on an install without `transformers` must report a
missing dependency instead of crashing with `UnboundLocalError`.

See issue #2227: the `LowMemoryMergeError` import must bind before the
`transformers` import so the matching `except` clause is resolvable when the
`transformers` import raises `ImportError`.
"""

import json
import sys

from tests.conftest import strip_ansi


def test_low_memory_merge_without_transformers_reports_missing_dependency(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    adapter = tmp_path / "ad"
    adapter.mkdir()
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
    (tmp_path / "base").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "transformers", None)

    result = CliRunner().invoke(
        app,
        [
            "merge",
            "--adapter",
            "ad",
            "--base",
            "base",
            "--output",
            "out",
            "--low-memory",
        ],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    output = " ".join(strip_ansi(result.output).split())
    assert "Missing dependency" in output
    assert "pip install torch transformers safetensors" in output
    assert "UnboundLocalError" not in output
