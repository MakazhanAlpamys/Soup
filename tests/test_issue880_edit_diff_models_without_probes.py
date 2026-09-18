"""Issue #880 — both models given and `--probes` forgotten succeeds silently.

#863 closed three of the four combinations: probes without models, and either
model without the other. The fourth was left open — both models supplied and
no probes — and it exits 0 with `Total probes: 0; 0 changed.`

A script that runs `soup edit set && soup edit diff --before-model X
--after-model Y -o d.json` and drops `--probes` gets a confident success and an
empty diff, which reads as "the edit changed nothing" rather than "nothing was
measured".

Smaller than #818 was, and worth closing anyway: #863 establishes the
symmetric rule everywhere else, and this is the only surviving hole in that
shape. Exit 2 follows the nine `raise typer.Exit` sites in
`commands/edit.py`, not the repo-wide 1-for-usage rule — that file's
inconsistency is pre-existing and tracked in #813, and one odd guard left
behind would make that harder to fix, not easier.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app

from .conftest import strip_ansi

runner = CliRunner()


def _clean(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _probes(fs: str) -> Path:
    path = Path(fs) / "probes.jsonl"
    path.write_text(json.dumps({"prompt": "The capital of France is"}) + "\n", encoding="utf-8")
    return path


class TestBothModelsWithoutProbes:
    def test_it_is_refused_rather_than_reporting_an_empty_diff(self, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, [
                "edit", "diff", "runA", "runB",
                "--before-model", "./base",
                "--after-model", "./edited",
            ])
            assert result.exit_code == 2, result.output

    def test_the_message_names_the_missing_flag(self, tmp_path):
        """Asserted on the message, not only the code, so this cannot pass
        because some *other* guard fired — which is exactly how a new guard
        that can never be reached would look green."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, [
                "edit", "diff", "runA", "runB",
                "--before-model", "./base",
                "--after-model", "./edited",
            ])
            assert "--probes" in _clean(result.output), result.output

    def test_no_diff_file_is_written(self, tmp_path):
        """The shape the issue describes: a script reads `d.json` and believes
        it. Refusing must not leave a file that says "nothing changed"."""
        with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
            out = Path(fs) / "d.json"
            result = runner.invoke(app, [
                "edit", "diff", "runA", "runB",
                "--before-model", "./base",
                "--after-model", "./edited",
                "-o", str(out),
            ])
            assert result.exit_code == 2, result.output
            assert not out.exists(), "an empty diff was written despite the refusal"


class TestTheGuardsAlreadyLandedStillFire:
    """#863's three refusals, each with its own message.

    A new guard placed too early could swallow these and still look green,
    because every case would exit 2 either way. The messages are what tell
    them apart.
    """

    def test_probes_without_either_model(self, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
            result = runner.invoke(app, [
                "edit", "diff", "runA", "runB", "--probes", str(_probes(fs)),
            ])
            assert result.exit_code == 2, result.output
            assert "both --before-model and --after-model are required" in _clean(result.output)

    def test_only_before_model(self, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
            result = runner.invoke(app, [
                "edit", "diff", "runA", "runB",
                "--probes", str(_probes(fs)), "--before-model", "./base",
            ])
            assert result.exit_code == 2, result.output
            assert "--after-model is required" in _clean(result.output)

    def test_only_after_model(self, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
            result = runner.invoke(app, [
                "edit", "diff", "runA", "runB",
                "--probes", str(_probes(fs)), "--after-model", "./edited",
            ])
            assert result.exit_code == 2, result.output
            assert "--before-model is required" in _clean(result.output)


class TestTheValidInvocationIsUnchanged:
    def test_neither_model_and_no_probes_still_succeeds(self, tmp_path):
        """Acceptance criterion 3. A placeholder report is a legitimate
        non-live invocation and must keep exiting 0 — the guard is about a
        *contradictory* combination, not about probes being mandatory."""
        with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
            out = Path(fs) / "d.json"
            result = runner.invoke(app, [
                "edit", "diff", "before-run", "after-run", "--output", str(out),
            ])
            assert result.exit_code == 0, result.output
            data = json.loads(out.read_text(encoding="utf-8"))
            assert data["total_probes"] == 0
            assert data["changes"] == []
