"""The expanded-vocabulary refusal text points at the open tracker (#3089).

#1986 and #1988 are closed; the two refusal messages in
``utils/mlx_generation.py`` and ``commands/merge.py``, and the docs bullet, now
cite #3089, the open issue that tracks lifting the refusal. The metadata is read
before any model or MLX import, so the whole sentence is asserted here without
MLX.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

_GENERATION = (
    "Native MLX generation does not currently support artifacts with expanded "
    "vocabularies (#3089)"
)
_MERGE = "soup merge does not support adapters with expanded vocabularies (#3089)."


def _text(output: str | None) -> str:
    return " ".join(strip_ansi(output).split())


def _expanded_adapter(tmp_path: Path) -> Path:
    # A PEFT-style config, so `soup merge` (transformers) does not stop earlier
    # at `reject_native_adapter`; the `added_tokens.safetensors` file is what
    # every path reads first.
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "added_tokens.safetensors").write_text("stub", encoding="utf-8")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base"}), encoding="utf-8"
    )
    return adapter


def _metadata_only_adapter(tmp_path: Path) -> Path:
    # No `added_tokens.safetensors`: the `vocab_expansion` key in
    # `adapter_config.json` is the branch `read_native_adapter` reaches next
    # (`mlx_generation.py:115-119`), so its message is asserted too.
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"model": "org/base", "vocab_expansion": True}), encoding="utf-8"
    )
    return adapter


def test_chat_names_the_open_issue(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = _expanded_adapter(tmp_path)

    result = CliRunner().invoke(app, ["chat", "--backend", "mlx", "--model", str(adapter)])

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert _GENERATION in _text(result.output)


def test_infer_names_the_open_issue(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = _expanded_adapter(tmp_path)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hello"}\n', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "infer",
            "--backend",
            "mlx",
            "--model",
            str(adapter),
            "--input",
            str(prompts),
            "--output",
            str(tmp_path / "out.jsonl"),
        ],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert _GENERATION in _text(result.output)


def test_merge_names_the_open_issue(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = _expanded_adapter(tmp_path)

    result = CliRunner().invoke(
        app, ["merge", "--adapter", str(adapter), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert _MERGE in _text(result.output)


def test_merge_backend_mlx_names_the_open_issue(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = _expanded_adapter(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "merge",
            "--backend",
            "mlx",
            "--adapter",
            str(adapter),
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert _GENERATION in _text(result.output)


def test_chat_names_the_open_issue_for_the_metadata_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = _metadata_only_adapter(tmp_path)

    result = CliRunner().invoke(app, ["chat", "--backend", "mlx", "--model", str(adapter)])

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert _GENERATION in _text(result.output)


def test_infer_names_the_open_issue_for_the_metadata_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = _metadata_only_adapter(tmp_path)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hello"}\n', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "infer",
            "--backend",
            "mlx",
            "--model",
            str(adapter),
            "--input",
            str(prompts),
            "--output",
            str(tmp_path / "out.jsonl"),
        ],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert _GENERATION in _text(result.output)
