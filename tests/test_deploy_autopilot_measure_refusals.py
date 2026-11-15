"""Tests for #2827: ``soup deploy autopilot`` pre-write validation.

All four ``--measure`` refusals and both output-path checks (``--recipe-out``,
``--script-out``) must run before ``write_recipe`` or ``write_deploy_script``
touches existing ``deploy_autopilot.yaml`` / ``deploy_autopilot.sh`` files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from soup_cli.commands.deploy import autopilot
from tests.conftest import strip_ansi


def _write_tasks(tmp_path: Path) -> Path:
    """Write a 2-row JSONL eval task file."""
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"prompt": "say hello", "expected": "hello", "scoring": "exact"}\n'
        '{"prompt": "say world", "expected": "world", "scoring": "exact"}\n',
        encoding="utf-8",
    )
    return path


class TestDeployAutopilotMeasureRefusals:
    @pytest.mark.parametrize(
        ("extra_args", "expected_msg", "create_tasks"),
        [
            ([], "--measure requires --tasks", False),
            (["--tasks", "../outside.jsonl"], "must stay under cwd", False),
            (["--tasks", "missing.jsonl"], "Tasks file not found", False),
            (
                ["--tasks", "tasks.jsonl", "--measure-candidates", " , "],
                "parsed to empty list",
                True,
            ),
        ],
    )
    def test_measure_refusals_leave_existing_outputs_untouched(
        self, tmp_path, monkeypatch, extra_args, expected_msg, create_tasks
    ):
        """All 4 --measure refusals exit 2 before writing either default file."""
        monkeypatch.chdir(tmp_path)
        recipe = tmp_path / "deploy_autopilot.yaml"
        script = tmp_path / "deploy_autopilot.sh"
        sentinel_recipe = b"# my hand-tuned recipe\nbase: my-org/my-model\n"
        sentinel_script = b"#!/bin/sh\n# my edited script\n"
        recipe.write_bytes(sentinel_recipe)
        script.write_bytes(sentinel_script)
        if create_tasks:
            _write_tasks(tmp_path)

        app = typer.Typer()
        app.command()(autopilot)
        result = CliRunner().invoke(
            app,
            [
                "--target",
                "rtx-4090-24gb",
                "--base",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "--measure",
                *extra_args,
            ],
        )
        assert result.exit_code == 2, result.output
        assert expected_msg in strip_ansi(result.output)
        assert recipe.read_bytes() == sentinel_recipe
        assert script.read_bytes() == sentinel_script

    def test_bad_script_out_leaves_existing_recipe_untouched(
        self, tmp_path, monkeypatch
    ):
        """Invalid --script-out fails before write_recipe overwrites default YAML."""
        monkeypatch.chdir(tmp_path)
        recipe = tmp_path / "deploy_autopilot.yaml"
        script = tmp_path / "deploy_autopilot.sh"
        sentinel_recipe = b"# my hand-tuned recipe\nbase: my-org/my-model\n"
        sentinel_script = b"#!/bin/sh\n# my edited script\n"
        recipe.write_bytes(sentinel_recipe)
        script.write_bytes(sentinel_script)

        app = typer.Typer()
        app.command()(autopilot)
        result = CliRunner().invoke(
            app,
            [
                "--target",
                "rtx-4090-24gb",
                "--base",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "--script-out",
                "../outside.sh",
            ],
        )
        assert result.exit_code == 1, result.output
        assert "script_path must stay under cwd" in strip_ansi(result.output)
        assert recipe.read_bytes() == sentinel_recipe
        assert script.read_bytes() == sentinel_script

    def test_valid_autopilot_overwrites_default_outputs(self, tmp_path, monkeypatch):
        """Control: valid autopilot invocation renders and writes both files."""
        monkeypatch.chdir(tmp_path)
        recipe = tmp_path / "deploy_autopilot.yaml"
        script = tmp_path / "deploy_autopilot.sh"
        recipe.write_bytes(b"# stale recipe\n")
        script.write_bytes(b"#!/bin/sh\n# stale script\n")

        app = typer.Typer()
        app.command()(autopilot)
        result = CliRunner().invoke(
            app,
            [
                "--target",
                "rtx-4090-24gb",
                "--base",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ],
        )
        assert result.exit_code == 0, result.output
        recipe_text = recipe.read_text(encoding="utf-8")
        assert "base: TinyLlama/TinyLlama-1.1B-Chat-v1.0" in recipe_text
        assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n")
