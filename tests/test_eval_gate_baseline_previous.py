"""#2523 - ``training.eval_gate.baseline: previous`` is refused, not read as a file.

The field was documented as ``registry://<id> | 'previous' | file path``, but
``resolve_baseline`` had no ``previous`` branch: the value was read as a file named
``previous``, and the epoch-end gate logged one warning and went on with an EMPTY
baseline, so a run meant to guard against regression had nothing to regress against.

``previous`` has no defined meaning (Soup keeps no version chain), so it is refused at
config load with a message naming the field, the value and the forms that work.
``resolve_baseline`` refuses it too, which covers ``soup eval gate --baseline`` and
``soup ship --baseline``.

The spellings are enumerated here rather than testing the one in the issue: any case,
surrounding whitespace (including a no-break space) and one pair of quotes, which is
how the old docs wrote it (``'previous'``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config, load_config_from_string
from soup_cli.config.schema import EvalGateConfig
from soup_cli.eval.gate import previous_baseline_refusal, resolve_baseline
from soup_cli.utils.exit_codes import EXIT_USAGE_ERROR
from tests.conftest import strip_ansi

runner = CliRunner()

FIELD = "training.eval_gate.baseline"
NBSP = chr(0xA0)
TAB = chr(9)
NEWLINE = chr(10)

#: Every spelling of ``previous`` that is refused.
REFUSED_SPELLINGS = [
    "previous",
    "Previous",
    "PREVIOUS",
    "pReViOuS",
    " previous",
    "previous ",
    "  previous  ",
    TAB + "previous" + NEWLINE,
    NBSP + "previous" + NBSP,
    "'previous'",
    '"previous"',
    " 'previous' ",
    "' previous '",
    "'Previous'",
]
REFUSED_IDS = [
    "lower",
    "capitalised",
    "upper",
    "mixed-case",
    "leading-space",
    "trailing-space",
    "padded",
    "tab-newline",
    "no-break-space",
    "single-quoted",
    "double-quoted",
    "quoted-padded-outside",
    "quoted-padded-inside",
    "quoted-capitalised",
]

#: Values that merely contain the word, or are other documented forms: all still load.
ALLOWED_VALUES = [
    None,
    "",
    "registry://x",
    "registry://previous",
    "./previous",
    "previous.json",
    "baselines/previous.json",
    "previous-run.json",
    "previously",
    "prev",
    "my previous run",
    "'previous",
    "previous'",
    "pre vious",
    "   ",
    "'",
    "''",
]
ALLOWED_IDS = [
    "none",
    "empty",
    "registry-ref",
    "registry-ref-named-previous",
    "dot-slash-previous",
    "previous-json",
    "nested-previous-json",
    "previous-run-json",
    "previously",
    "prev",
    "phrase",
    "unbalanced-leading-quote",
    "unbalanced-trailing-quote",
    "split-word",
    "whitespace-only",
    "lone-quote",
    "empty-quotes",
]


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed: Rich colours and wraps CLI output."""
    return " ".join(strip_ansi(text).split())


def _config_text(baseline: object) -> str:
    return yaml.safe_dump({
        "base": "org/model",
        "task": "sft",
        "data": {"train": "./x.jsonl"},
        "training": {
            "eval_gate": {"enabled": True, "suite": "./evals/gate.yaml", "baseline": baseline},
        },
        "output": "./o",
    })


def _suite_and_tasks(tmp_path: Path) -> tuple[Path, Path]:
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        json.dumps({"prompt": "control", "expected": "gold", "scoring": "exact"}) + "\n",
        encoding="utf-8",
    )
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(
        yaml.safe_dump({
            "suite": "previous-baseline-probe",
            "tasks": [{
                "type": "custom",
                "name": "control",
                "threshold": 0.0,
                "tasks": str(tasks_file),
                "scorer": "exact",
            }],
        }),
        encoding="utf-8",
    )
    return suite_file, tasks_file


@pytest.fixture
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Config load
# ---------------------------------------------------------------------------


class TestConfigRefusesPrevious:
    @pytest.mark.parametrize("spelling", REFUSED_SPELLINGS, ids=REFUSED_IDS)
    def test_field_refuses_every_spelling(self, spelling: str) -> None:
        with pytest.raises(ValidationError) as caught:
            EvalGateConfig(baseline=spelling)
        errors = caught.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("baseline",)
        message = errors[0]["msg"]
        assert message.startswith(
            f"Value error, {FIELD}: {spelling!r} is not a supported baseline. "
            "'previous' has no defined meaning"
        )

    def test_message_names_the_forms_that_work_and_how_to_get_one(self) -> None:
        with pytest.raises(ValidationError) as caught:
            EvalGateConfig(baseline="previous")
        message = _plain(caught.value.errors()[0]["msg"])
        assert "Use registry://<id> or the path to a scores JSON file" in message
        assert "`soup eval gate --suite <suite> --model <model> --write-baseline <file>`" in message
        assert "To use a file named previous, write ./previous." in message
        assert "the gate would run with no baseline" in message

    def test_the_write_baseline_flag_the_message_names_exists(self) -> None:
        import typer.main

        root = typer.main.get_command(app)
        gate_command = root.commands["eval"].commands["gate"]
        flags = {opt for param in gate_command.params for opt in getattr(param, "opts", [])}
        assert {"--suite", "--model", "--write-baseline"} <= flags

    def test_refused_with_the_gate_enabled_and_with_it_off(self) -> None:
        for enabled in (True, False):
            with pytest.raises(ValidationError, match="is not a supported baseline"):
                EvalGateConfig(enabled=enabled, suite="gate.yaml", baseline="previous")

    @pytest.mark.parametrize("spelling", REFUSED_SPELLINGS, ids=REFUSED_IDS)
    def test_load_config_from_string_names_the_field(self, spelling: str) -> None:
        with pytest.raises(ValueError) as caught:
            load_config_from_string(_config_text(spelling))
        assert (
            f"training -> eval_gate -> baseline: Value error, {FIELD}: {spelling!r} "
            "is not a supported baseline."
        ) in str(caught.value)

    @pytest.mark.parametrize(
        "spelling", ["previous", "Previous", " previous ", "'previous'"],
        ids=["lower", "capitalised", "padded", "single-quoted"],
    )
    def test_load_config_prints_the_refusal_and_exits_1(
        self, in_tmp: Path, capsys: pytest.CaptureFixture[str], spelling: str,
    ) -> None:
        (in_tmp / "soup.yaml").write_text(_config_text(spelling), encoding="utf-8")
        with pytest.raises(SystemExit) as raised:
            load_config("soup.yaml")
        assert raised.value.code == 1
        output = _plain(capsys.readouterr().out)
        assert "Config validation error:" in output
        assert "training -> eval_gate -> baseline:" in output
        assert f"{FIELD}: {spelling!r} is not a supported baseline." in output
        assert "Use registry://<id> or the path to a scores JSON file" in output

    @pytest.mark.parametrize("value", ALLOWED_VALUES, ids=ALLOWED_IDS)
    def test_every_other_value_still_loads_unchanged(self, value: str | None) -> None:
        assert EvalGateConfig(baseline=value).baseline == value

    @pytest.mark.parametrize(
        "value", ["registry://llama31-chat-v1", "./baselines/prior.json", "prior.json"],
    )
    def test_the_documented_forms_load_through_the_config_loader(self, value: str) -> None:
        config = load_config_from_string(_config_text(value))
        assert config.training.eval_gate is not None
        assert config.training.eval_gate.baseline == value

    def test_a_config_without_a_baseline_still_loads(self) -> None:
        text = yaml.safe_dump({
            "base": "org/model",
            "task": "sft",
            "data": {"train": "./x.jsonl"},
            "training": {"eval_gate": {"enabled": True, "suite": "./evals/gate.yaml"}},
            "output": "./o",
        })
        config = load_config_from_string(text)
        assert config.training.eval_gate is not None
        assert config.training.eval_gate.baseline is None


# ---------------------------------------------------------------------------
# The shared check and resolve_baseline (defence in depth, and the CLI flags)
# ---------------------------------------------------------------------------


class TestResolveBaselineRefusesPrevious:
    @pytest.mark.parametrize("spelling", REFUSED_SPELLINGS, ids=REFUSED_IDS)
    def test_every_spelling_raises_a_clear_value_error(self, spelling: str) -> None:
        with pytest.raises(ValueError) as caught:
            resolve_baseline(spelling)
        message = str(caught.value)
        assert message.startswith(f"{spelling!r} is not a supported baseline. ")
        assert "Use registry://<id> or the path to a scores JSON file" in _plain(message)
        # No field prefix: a CLI flag adds its own.
        assert FIELD not in message

    def test_previous_is_not_read_as_a_file_named_previous(
        self, in_tmp: Path,
    ) -> None:
        """The reported bug: a file called ``previous`` made the value 'work'."""
        (in_tmp / "previous").write_text('{"control": 0.9}', encoding="utf-8")
        with pytest.raises(ValueError, match="is not a supported baseline"):
            resolve_baseline("previous")

    def test_a_file_really_named_previous_is_reached_as_dot_slash_previous(
        self, in_tmp: Path,
    ) -> None:
        (in_tmp / "previous").write_text('{"control": 0.9}', encoding="utf-8")
        emitted: list[str] = []
        assert resolve_baseline("./previous", warn=emitted.append) == {"control": 0.9}

    def test_previous_with_no_such_file_is_a_refusal_not_a_missing_file(
        self, in_tmp: Path,
    ) -> None:
        with pytest.raises(ValueError) as caught:
            resolve_baseline("previous")
        assert "baseline file not found" not in str(caught.value)

    def test_none_and_empty_still_mean_no_baseline(self) -> None:
        assert resolve_baseline(None) == {}
        assert resolve_baseline("") == {}

    @pytest.mark.parametrize("value", [v for v in ALLOWED_VALUES if v], ids=[
        i for v, i in zip(ALLOWED_VALUES, ALLOWED_IDS) if v
    ])
    def test_the_check_ignores_everything_that_is_not_previous(self, value: str) -> None:
        assert previous_baseline_refusal(value) is None

    @pytest.mark.parametrize("value", [None, 5, 1.5, True, ["previous"], {"previous": 1}])
    def test_the_check_ignores_non_strings(self, value: object) -> None:
        assert previous_baseline_refusal(value) is None

    def test_the_field_prefix_is_optional(self) -> None:
        assert previous_baseline_refusal("previous").startswith("'previous' is not a supported")
        assert previous_baseline_refusal("previous", field="x.y").startswith(
            "x.y: 'previous' is not a supported"
        )


class TestCliFlagsRefusePrevious:
    @pytest.mark.parametrize("spelling", ["previous", "Previous", "'previous'"])
    def test_eval_gate_baseline_flag_exits_with_the_usage_code(
        self, in_tmp: Path, spelling: str,
    ) -> None:
        suite_file, _ = _suite_and_tasks(in_tmp)
        result = runner.invoke(
            app, ["eval", "gate", "--suite", suite_file.name, "--baseline", spelling],
        )
        assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert "Cannot resolve baseline:" in output
        assert f"{spelling!r} is not a supported baseline." in output
        assert "Use registry://<id> or the path to a scores JSON file" in output
        assert "GATE PASSED" not in output

    def test_ship_baseline_flag_exits_with_the_usage_code(self, in_tmp: Path) -> None:
        tasks = in_tmp / "tasks.jsonl"
        tasks.write_text(
            '{"prompt": "x", "expected": "y", "scoring": "exact"}\n', encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "ship",
                "--base", "m",
                "--tuned", "m",
                "--task-eval", tasks.name,
                "--baseline", "previous",
            ],
        )
        assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert "--baseline: 'previous' is not a supported baseline." in output


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------


class TestDocsNoLongerListPrevious:
    @staticmethod
    def _baseline_row() -> str:
        docs = Path(__file__).resolve().parent.parent / "docs" / "config-reference.md"
        rows = [
            line
            for line in docs.read_text(encoding="utf-8").splitlines()
            if line.startswith("| `training.eval_gate.baseline` |")
        ]
        assert len(rows) == 1, rows
        return rows[0]

    def test_the_field_description_lists_only_the_forms_that_work(self) -> None:
        description = EvalGateConfig.model_fields["baseline"].description
        assert description is not None
        assert "previous" not in description
        assert "registry://<id>" in description
        assert "file path" in description

    def test_the_config_reference_row_lists_only_the_forms_that_work(self) -> None:
        row = self._baseline_row()
        assert "previous" not in row
        assert "registry://&lt;id>" in row
        assert "file path" in row

    def test_the_config_reference_row_matches_the_field_description(self) -> None:
        row = self._baseline_row()
        description = EvalGateConfig.model_fields["baseline"].description
        assert description is not None
        assert "scores to compare against" in row
        assert "`soup eval gate --write-baseline` writes a file." in row
        assert "`soup eval gate --write-baseline` writes a file." in description
