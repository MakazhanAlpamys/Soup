"""An axolotl dataset whose ``type:`` is a custom prompt format migrates with a warning.

axolotl documents ``type:`` as either a prompt-strategy name or a mapping of field
names and a template ("How to add custom prompt format", docs/dataset-formats/
inst_tune.qmd). Before, the mapping went straight into a dict lookup, so
``soup migrate --from axolotl`` ended in ``Error: TypeError: unhashable type: 'dict'``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.migrate.axolotl import migrate_axolotl
from soup_cli.migrate.common import config_to_yaml

# axolotl's documented example, under a minimal LoRA config.
_CUSTOM = (
    "base_model: NousResearch/Llama-3.2-1B\nadapter: lora\nlora_r: 16\nlora_alpha: 32\n"
    "datasets:\n  - path: data.jsonl\n    type:\n      system_prompt: ''\n"
    "      field_system: system\n      field_instruction: input\n      field_output: output\n"
    "      format: '[INST] {instruction} [/INST]'\n"
    "      no_input_format: '[INST] {instruction} [/INST]'\n"
)


def _migrate(tmp_path: Path, text: str):
    path = tmp_path / "ax.yaml"
    path.write_text(text, encoding="utf-8")
    result = migrate_axolotl(path)
    return result, load_config_from_string(config_to_yaml(result))


def test_a_custom_prompt_format_migrates_to_auto_and_says_so(tmp_path: Path) -> None:
    result, cfg = _migrate(tmp_path, _CUSTOM)
    assert cfg.data.format == "auto"
    assert cfg.data.train == "data.jsonl"
    notes = [w for w in result["_warnings"] if "custom prompt format" in w]
    assert len(notes) == 1, result["_warnings"]
    for key in ("field_instruction", "field_output", "format", "no_input_format"):
        assert key in notes[0]


def test_the_cli_writes_the_config_instead_of_a_type_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ax.yaml").write_text(_CUSTOM, encoding="utf-8")
    run = CliRunner().invoke(app, ["migrate", "--from", "axolotl", "ax.yaml", "--yes"])
    assert run.exit_code == 0, run.output
    assert load_config_from_string(
        (tmp_path / "soup.yaml").read_text(encoding="utf-8")
    ).data.format == "auto"


@pytest.mark.parametrize("ds_type,expected", [("alpaca", "alpaca"), ("chat_template", "chatml")])
def test_a_named_type_is_still_mapped(tmp_path: Path, ds_type: str, expected: str) -> None:
    result, cfg = _migrate(tmp_path, _CUSTOM.split("    type:")[0] + f"    type: {ds_type}\n")
    assert cfg.data.format == expected
    assert not any("custom prompt format" in w for w in result["_warnings"])
