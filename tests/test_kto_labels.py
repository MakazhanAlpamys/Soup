"""KTO labels have one strict meaning in conversion, validation and training."""

import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from soup_cli.data.formats import _convert_kto, format_to_messages_with_reason
from tests.conftest import strip_ansi

ACCEPTED = [
    (True, True), (False, False), (1, True), (0, False), (-1, False),
    ("true", True), ("1", True), ("yes", True),
    ("false", False), ("0", False), ("no", False),
    (" TRUE ", True), (" YeS ", True), (" FALSE ", False), (" No ", False),
]
REFUSED = [None, 0.5, 1.0, 0.0, -1.0, float("nan"), float("inf"),
           float("-inf"), 2, -2, [], ["no"], {}, {"label": True}, "maybe", "-1"]


@pytest.mark.parametrize("value,expected", ACCEPTED)
def test_accepted_kto_label_spellings(value, expected):
    converted = _convert_kto({"prompt": "Q", "completion": "A", "label": value})
    assert converted == {
        "prompt": "Q", "completion": "A", "label": expected,
    }
    assert converted["label"] is expected


@pytest.mark.parametrize("value", REFUSED)
def test_kto_refuses_non_labels_with_the_value(value):
    row = {"prompt": "Q", "completion": "A", "label": value}
    with pytest.raises(ValueError, match="label") as error:
        _convert_kto(row)
    assert repr(value) in str(error.value)
    converted, reason = format_to_messages_with_reason(row, "kto")
    assert converted is None
    assert "label" in reason and repr(value) in reason


@pytest.mark.parametrize("value", REFUSED)
@pytest.mark.parametrize("command", ["validate", "dry-run"])
def test_cli_refuses_invalid_kto_label_before_model_load(tmp_path, monkeypatch, value, command):
    from soup_cli.cli import app
    from soup_cli.trainer import dispatch

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"prompt": "Q", "completion": "A", "label": value}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "soup.yaml").write_text(
        "base: public/model\ntask: kto\n"
        "data: {train: data.jsonl, format: kto, val_split: 0}\n"
        "training: {quantization: none}\noutput: out\n", encoding="utf-8",
    )
    build = Mock(side_effect=AssertionError("invalid label reached a trainer"))
    monkeypatch.setattr(dispatch, "build_trainer", build)
    args = (["data", "validate", "data.jsonl", "--format", "kto"] if command == "validate"
            else ["train", "--config", "soup.yaml", "--dry-run"])
    result = CliRunner().invoke(app, args)
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == (2 if command == "validate" else 1), (output, result.exception)
    assert "label" in output and repr(value) in output, output
    build.assert_not_called()


@pytest.mark.parametrize("value", [False, -1, " No "])
def test_valid_undesirable_label_passes_cli_preflight(tmp_path, monkeypatch, value):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"prompt": "Q", "completion": "A", "label": value}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "soup.yaml").write_text(
        "base: public/model\ntask: kto\n"
        "data: {train: data.jsonl, format: kto, val_split: 0}\n"
        "training: {quantization: none}\noutput: out\n", encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["data", "validate", "data.jsonl", "--format", "kto"])
    assert result.exit_code == 0, strip_ansi(result.output)
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])
    assert result.exit_code == 0, strip_ansi(result.output)
    assert "Ready to train" in strip_ansi(result.output)
