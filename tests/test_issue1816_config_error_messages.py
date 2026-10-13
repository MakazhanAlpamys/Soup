"""#1816 -- config load errors name a location and, for an unmatched key, a next step.

A ``SoupConfig``-level validator reports an empty pydantic ``loc``, so the loader
printed ``": Value error, backend='unsloth' is not applied ..."``. An unknown key
with no close field name printed only its verdict. Every string is asserted
verbatim, through ``strip_ansi`` and with ``FORCE_COLOR=1`` for the CLI paths.
"""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.unknown_keys import NO_SUGGESTION_REMEDY
from tests.conftest import strip_ansi

_REWARD_ON_UNSLOTH = (
    "config: backend='unsloth' is not applied by task='reward_model': that trainer has "
    "no unsloth setup and would train on plain transformers. Use backend: transformers."
)
_ENV = {"COLUMNS": "400", "FORCE_COLOR": "1"}


def _yaml(backend="transformers", **training):
    raw = {
        "base": "org/m",
        "task": "reward_model",
        "backend": backend,
        "data": {"train": "./x.jsonl"},
        "training": {"epochs": 1, **training},
    }
    return yaml.safe_dump(raw)


def _refusal(text):
    with pytest.raises(ValueError) as info:
        load_config_from_string(text)
    return str(info.value)


def _lines(output):
    return [line.strip() for line in strip_ansi(output).splitlines()]


class TestCrossFieldValidatorLocation:
    def test_string_loader(self):
        assert _refusal(_yaml(backend="unsloth")) == _REWARD_ON_UNSLOTH

    @pytest.mark.parametrize("command", [["train", "--config"], ["doctor", "--config"]])
    def test_cli(self, tmp_path, command):
        path = tmp_path / "soup.yaml"
        path.write_text(_yaml(backend="unsloth"), encoding="utf-8")

        result = CliRunner().invoke(app, [*command, str(path)], env=_ENV)

        assert result.exit_code != 0, result.output
        lines = _lines(result.output)
        assert _REWARD_ON_UNSLOTH in lines, result.output
        assert not any(line.startswith(":") for line in lines), result.output
        assert "Value error, " not in strip_ansi(result.output), result.output


class TestUnknownKeyRemedy:
    def test_no_suggestion_names_a_next_step(self):
        message = _refusal(_yaml(zzzzzzzzqq=1))
        assert message == (
            "unknown config key 'training.zzzzzzzzqq' - refused.\n" + NO_SUGGESTION_REMEDY
        )

    def test_a_suggestion_is_unchanged(self):
        message = _refusal(_yaml(quantizaton="4bit"))
        assert message == (
            "unknown config key 'training.quantizaton' - did you mean 'quantization' "
            "or 'quantization_aware'? Refused."
        )

    def test_cli(self, tmp_path):
        path = tmp_path / "soup.yaml"
        path.write_text(_yaml(zzzzzzzzqq=1), encoding="utf-8")

        result = CliRunner().invoke(app, ["train", "--config", str(path)], env=_ENV)

        assert result.exit_code == 1, result.output
        lines = _lines(result.output)
        assert "unknown config key 'training.zzzzzzzzqq' - refused." in lines, result.output
        assert NO_SUGGESTION_REMEDY in lines, result.output
