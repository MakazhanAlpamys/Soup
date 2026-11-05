"""The data-mix apply snippet must parse back into the original mixture."""

import pytest
import yaml
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import data_mix
from tests.conftest import make_cli_runner, strip_ansi


@pytest.mark.parametrize(
    ("paths", "width"),
    [
        pytest.param(["a.jsonl", "b.jsonl"], 80, id="plain"),
        pytest.param(["odd: name.jsonl", "#dataset", "on"], 80, id="yaml-syntax"),
        pytest.param(["[red]:warning:", "snowman-☃-😀.jsonl", "raw\x9b\u202e"], 80,
                     id="literal-terminal-text"),
        pytest.param([r"C:\datasets\[train\rows.jsonl", "x" * 160 + ".jsonl"], 48,
                     id="backslashes-and-narrow-terminal"),
    ],
)
def test_apply_preserves_list_paths_in_spliceable_yaml(paths, width, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data_mix, "console", Console(width=width, height=100))
    weights = [0.25] * (len(paths) - 1)
    weights.append(1 - sum(weights))
    block = {
        "data": {
            "train": paths,
            "interleave": {"strategy": "probs", "probs": weights},
        }
    }
    recipe = tmp_path / "recipe.yaml"
    original = yaml.safe_dump(block)
    recipe.write_text(original, encoding="utf-8")

    result = make_cli_runner().invoke(app, ["data", "mix", "--apply", recipe.name])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x9b" not in result.stdout
    assert "\u202e" not in result.stdout
    _, marker, snippet = strip_ansi(result.stdout).partition("\ndata:\n")
    assert marker, result.stdout
    assert yaml.safe_load("data:\n" + snippet) == block
    assert recipe.read_text(encoding="utf-8") == original
