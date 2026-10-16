"""Tests for issue #818: strict validation of models and probes in `soup edit diff`."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import live_eval

from .conftest import strip_ansi

runner = CliRunner()


def _clean(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def test_edit_diff_probes_without_models_exits_2(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
        probes_file = Path(fs) / "probes.jsonl"
        probes_file.write_text(
            json.dumps({"prompt": "The Eiffel Tower is located in"}) + "\n"
            + json.dumps({"prompt": "The capital of France is"}) + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, [
            "edit", "diff", "runA", "runB",
            "--probes", str(probes_file),
        ])
        assert result.exit_code == 2, result.output
        assert "both --before-model and --after-model are required" in _clean(result.output)


def test_edit_diff_only_before_model_exits_2_and_names_after_model(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
        probes_file = Path(fs) / "probes.jsonl"
        probes_file.write_text(
            json.dumps({"prompt": "The Eiffel Tower is located in"}) + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, [
            "edit", "diff", "runA", "runB",
            "--probes", str(probes_file),
            "--before-model", "some-before-model",
        ])
        assert result.exit_code == 2, result.output
        assert "--after-model is required" in _clean(result.output)


def test_edit_diff_only_after_model_exits_2_and_names_before_model(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
        probes_file = Path(fs) / "probes.jsonl"
        probes_file.write_text(
            json.dumps({"prompt": "The Eiffel Tower is located in"}) + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, [
            "edit", "diff", "runA", "runB",
            "--probes", str(probes_file),
            "--after-model", "some-after-model",
        ])
        assert result.exit_code == 2, result.output
        assert "--before-model is required" in _clean(result.output)


def test_edit_diff_asymmetric_models_without_probes_exits_2(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, [
            "edit", "diff", "runA", "runB",
            "--before-model", "some-before-model",
        ])
        assert result.exit_code == 2, result.output
        assert "--after-model is required" in _clean(result.output)


def test_edit_diff_probes_lacking_prompt_key_exits_2(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
        probes_file = Path(fs) / "probes_text.jsonl"
        probes_file.write_text(
            json.dumps({"text": "The Eiffel Tower is located in"}) + "\n"
            + json.dumps({"question": "The capital of France is"}) + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, [
            "edit", "diff", "runA", "runB",
            "--probes", str(probes_file),
            "--before-model", "modelA",
            "--after-model", "modelB",
        ])
        assert result.exit_code == 2, result.output
        assert "prompt" in _clean(result.output).lower()
        assert "missing 'prompt' key" in _clean(result.output)


def test_edit_diff_empty_probe_file_exits_2(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
        probes_file = Path(fs) / "empty.jsonl"
        probes_file.write_text("\n   \n\n", encoding="utf-8")
        result = runner.invoke(app, [
            "edit", "diff", "runA", "runB",
            "--probes", str(probes_file),
            "--before-model", "modelA",
            "--after-model", "modelB",
        ])
        assert result.exit_code == 2, result.output
        assert "empty" in _clean(result.output).lower()


def test_edit_diff_probes_with_null_byte_exits_2(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
        probes_file = Path(fs) / "null_probes.jsonl"
        probes_file.write_text(
            json.dumps({"prompt": "The Eiffel Tower is located in \x00 Paris"}) + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, [
            "edit", "diff", "runA", "runB",
            "--probes", str(probes_file),
            "--before-model", "modelA",
            "--after-model", "modelB",
        ])
        assert result.exit_code == 2, result.output
        assert "prompt contains null byte" in _clean(result.output)



def test_no_probe_receives_unmeasured_changed_false(monkeypatch, tmp_path: Path) -> None:
    """Verifies that changed verdicts are only produced by live evaluation of both models."""
    def _mock_make_gen(model_id: str, **kwargs):
        if model_id == "before":
            return lambda p: "Paris" if "France" in p else "4"
        return lambda p: "Lyon" if "France" in p else "4"

    monkeypatch.setattr(live_eval, "make_generator", _mock_make_gen)

    with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
        probes_file = Path(fs) / "probes.jsonl"
        probes_file.write_text(
            json.dumps({"prompt": "The capital of France is"}) + "\n"
            + json.dumps({"prompt": "2 + 2 ="}) + "\n",
            encoding="utf-8",
        )
        out_json = Path(fs) / "diff.json"
        result = runner.invoke(app, [
            "edit", "diff", "before-run", "after-run",
            "--probes", str(probes_file),
            "--before-model", "before",
            "--after-model", "after",
            "--output", str(out_json),
        ])
        assert result.exit_code == 0, result.output
        assert out_json.exists()
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["total_probes"] == 2
        # France: Paris vs Lyon -> changed = True
        france_change = next(c for c in data["changes"] if "France" in c["prompt"])
        assert france_change["changed"] is True
        assert france_change["before"] == "Paris"
        assert france_change["after"] == "Lyon"
        # Math: 4 vs 4 -> changed = False (measured genuine unchanged)
        math_change = next(c for c in data["changes"] if "2 + 2" in c["prompt"])
        assert math_change["changed"] is False
        assert math_change["before"] == "4"
        assert math_change["after"] == "4"


def test_edit_diff_without_probes_produces_empty_changes(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
        out_json = Path(fs) / "diff.json"
        result = runner.invoke(app, [
            "edit", "diff", "before-run", "after-run",
            "--output", str(out_json),
        ])
        assert result.exit_code == 0, result.output
        assert "no probes supplied" in result.output
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["total_probes"] == 0
        assert data["changes"] == []
