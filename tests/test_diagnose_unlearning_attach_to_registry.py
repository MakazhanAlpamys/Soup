"""``--attach-to-registry`` on the two report commands attaches, or fails loudly.

``soup diagnose`` and ``soup eval unlearning`` write a JSON report and can link
it to a Model Registry entry. Both called the shared helper
``soup_cli.registry.attach.attach_artifact`` with arguments it does not take
(three positionals in one command, an ``artifact_path=`` keyword in the other),
printed the resulting ``TypeError`` as a warning and exited 0: the report was
never attached, and the shell was told the command had succeeded.

A requested attach is part of the command's success, as it already is for
``soup bom emit``, ``soup attest emit`` and ``soup eval custom``:

* the report is stored on the entry;
* an attach that fails exits 1 and names the problem, with the report on disk;
* ``--attach-to-registry`` without ``--output`` is refused before the run.

Every test drives the real command line against a real registry store in a
throwaway SQLite file (``SOUP_REGISTRY_DB_PATH``) and asserts on the rows the
store holds, with the real ``attach_artifact`` in between: a stand-in with a
looser signature is what let the wrong call through.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.utils.diagnose.live as live_mod
import soup_cli.utils.unlearning_eval as unlearning_eval
from soup_cli.cli import app
from soup_cli.registry.store import RegistryStore
from soup_cli.utils.diagnose.report import FAILURE_MODES
from soup_cli.utils.diagnose.runner import build_report, neutral_score
from soup_cli.utils.exit_codes import EXIT_GATE_FAILED, EXIT_USAGE_ERROR
from tests.conftest import strip_ansi

runner = CliRunner()

DB_NAME = "registry.db"


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match.

    Rich colours flag names, quoted strings and numbers on a colour terminal
    and wraps at the terminal width, so a raw substring check is flaky.
    """
    return " ".join(strip_ansi(text).split())


def _ok(result) -> None:
    assert result.exit_code == 0, (result.output, repr(result.exception))


def _exits(result, code: int) -> None:
    assert result.exit_code == code, (result.output, repr(result.exception))


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty cwd whose registry is ``./registry.db`` (never ``~/.soup``)."""
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / DB_NAME))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _push(workdir: Path, name: str = "demo", tag: str = "v1") -> str:
    with RegistryStore(db_path=workdir / DB_NAME) as store:
        return store.push(
            name=name,
            tag=tag,
            base_model="tiny-base",
            task="sft",
            run_id="run-1",
            config={"base": "tiny-base"},
        )


def _artifacts(workdir: Path, entry_id: str) -> list[dict]:
    with RegistryStore(db_path=workdir / DB_NAME) as store:
        return store.get_artifacts(entry_id)


def _assert_stored(workdir: Path, entry_id: str, kind: str, written: Path) -> None:
    """The entry holds exactly one artifact: ``written``, as ``kind``."""
    rows = _artifacts(workdir, entry_id)
    assert [row["kind"] for row in rows] == [kind], rows
    assert os.path.samefile(rows[0]["path"], written), rows
    assert rows[0]["sha256"] == hashlib.sha256(written.read_bytes()).hexdigest()
    assert rows[0]["size_bytes"] == written.stat().st_size


def _major_evidence(workdir: Path) -> str:
    (workdir / "ev.json").write_text(
        json.dumps({"scores": {"forgetting": {"score": 0.1, "evidence": "regressed"}}}),
        encoding="utf-8",
    )
    return "ev.json"


# ---------------------------------------------------------------------------
# soup diagnose
# ---------------------------------------------------------------------------


class TestDiagnoseAttachesTheReport:
    @pytest.mark.parametrize(
        "output",
        ["report.json", "reports/nested/diag.json", "ABSOLUTE"],
        ids=["relative", "nested", "absolute"],
    )
    def test_the_report_is_stored_on_the_entry(self, workdir, output):
        entry_id = _push(workdir)
        target = workdir / "abs.json" if output == "ABSOLUTE" else workdir / output
        output_arg = str(target) if output == "ABSOLUTE" else output

        result = runner.invoke(
            app,
            ["diagnose", "run-1", "--output", output_arg, "--attach-to-registry", entry_id],
        )

        _ok(result)
        _assert_stored(workdir, entry_id, "diagnose_report", target)
        out = _plain(result.output)
        assert f"Attached diagnose_report to registry entry {entry_id}" in out
        assert "could not attach" not in out

    @pytest.mark.parametrize("spelling", ["full-id", "id-prefix", "registry-url", "name-tag"])
    def test_every_spelling_of_the_entry_reference_attaches(self, workdir, spelling):
        entry_id = _push(workdir, name="demo", tag="v1")
        reference = {
            "full-id": entry_id,
            "id-prefix": entry_id[:-4],
            "registry-url": f"registry://{entry_id}",
            "name-tag": "demo:v1",
        }[spelling]

        result = runner.invoke(
            app,
            ["diagnose", "run-1", "--output", "report.json", "--attach-to-registry", reference],
        )

        _ok(result)
        _assert_stored(workdir, entry_id, "diagnose_report", workdir / "report.json")

    def test_a_live_report_is_attached_too(self, workdir, monkeypatch):
        entry_id = _push(workdir)

        def _live(**kwargs):
            return build_report(
                run_id=kwargs["run_id"],
                base=kwargs["base"],
                adapter="",
                scores={mode: neutral_score(mode, "stub") for mode in FAILURE_MODES},
                soup_version="0",
            )

        monkeypatch.setattr(live_mod, "run_live_diagnose", _live)
        result = runner.invoke(
            app,
            [
                "diagnose", "run-1", "--base-model", "org/m",
                "--output", "live.json", "--attach-to-registry", entry_id,
            ],
        )

        _ok(result)
        _assert_stored(workdir, entry_id, "diagnose_report", workdir / "live.json")

    def test_a_major_report_is_attached_and_still_exits_two(self, workdir):
        """The verdict exit code is unchanged, and a MAJOR report is still linked."""
        entry_id = _push(workdir)

        result = runner.invoke(
            app,
            [
                "diagnose", "run-1", "--evidence", _major_evidence(workdir),
                "--output", "report.json", "--attach-to-registry", entry_id,
            ],
        )

        _exits(result, EXIT_GATE_FAILED)
        _assert_stored(workdir, entry_id, "diagnose_report", workdir / "report.json")


class TestDiagnoseAttachFailureIsNotSuccess:
    def test_an_unknown_entry_exits_one_and_names_it(self, workdir):
        entry_id = _push(workdir)

        result = runner.invoke(
            app,
            [
                "diagnose", "run-1", "--output", "report.json",
                "--attach-to-registry", "no-such-entry",
            ],
        )

        _exits(result, 1)
        out = _plain(result.output)
        assert (
            "Error: could not attach to registry: ValueError: "
            "registry entry not found: no-such-entry" in out
        ), out
        assert "Attached" not in out
        assert _artifacts(workdir, entry_id) == []
        # The report itself was written before the attach was tried.
        written = json.loads((workdir / "report.json").read_text(encoding="utf-8"))
        assert written["run_id"] == "run-1"

    def test_an_ambiguous_prefix_exits_one_and_attaches_to_neither(self, workdir):
        first = _push(workdir, name="first")
        second = _push(workdir, name="second")

        result = runner.invoke(
            app,
            ["diagnose", "run-1", "--output", "report.json", "--attach-to-registry", "reg_"],
        )

        _exits(result, 1)
        out = _plain(result.output)
        assert "Error: could not attach to registry: AmbiguousRefError:" in out, out
        assert "matches 2 entries" in out, out
        assert _artifacts(workdir, first) == []
        assert _artifacts(workdir, second) == []

    def test_an_empty_entry_id_exits_one(self, workdir):
        """``--attach-to-registry "$ID"`` with ``ID`` unset must not pass silently."""
        entry_id = _push(workdir)

        result = runner.invoke(
            app,
            ["diagnose", "run-1", "--output", "report.json", "--attach-to-registry", ""],
        )

        _exits(result, 1)
        out = _plain(result.output)
        assert "Error: could not attach to registry: ValueError: registry entry not found:" in out
        assert _artifacts(workdir, entry_id) == []

    def test_an_unreadable_registry_exits_one(self, tmp_path, monkeypatch):
        """A failure that is not a lookup miss is reported the same way."""
        (tmp_path / "not-a-database").mkdir()
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "not-a-database"))
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            ["diagnose", "run-1", "--output", "report.json", "--attach-to-registry", "any"],
        )

        _exits(result, 1)
        out = _plain(result.output)
        assert "Error: could not attach to registry: OperationalError:" in out, out
        assert (tmp_path / "report.json").is_file()

    def test_a_failed_attach_outranks_the_verdict(self, workdir):
        """Like a failed ``--output`` write: exit 1, not the MAJOR exit 2."""
        _push(workdir)

        result = runner.invoke(
            app,
            [
                "diagnose", "run-1", "--evidence", _major_evidence(workdir),
                "--output", "report.json", "--attach-to-registry", "no-such-entry",
            ],
        )

        _exits(result, 1)
        assert "could not attach to registry" in _plain(result.output)


class TestDiagnoseAttachNeedsOutput:
    @pytest.mark.parametrize("output_args", [[], ["--output", ""]], ids=["absent", "empty"])
    def test_it_is_refused_before_the_report_is_built(self, workdir, output_args):
        entry_id = _push(workdir)

        result = runner.invoke(
            app, ["diagnose", "run-1", *output_args, "--attach-to-registry", entry_id]
        )

        _exits(result, EXIT_USAGE_ERROR)
        out = _plain(result.output)
        assert "Error: --attach-to-registry needs --output (nothing written to attach)." in out
        assert "soup diagnose:" not in out, out
        assert _artifacts(workdir, entry_id) == []

    def test_it_is_refused_before_a_live_run_loads_the_model(self, workdir, monkeypatch):
        entry_id = _push(workdir)
        monkeypatch.setattr(
            live_mod,
            "run_live_diagnose",
            lambda **kwargs: pytest.fail("the model must not load when nothing can be attached"),
        )

        result = runner.invoke(
            app,
            ["diagnose", "run-1", "--base-model", "org/m", "--attach-to-registry", entry_id],
        )

        _exits(result, EXIT_USAGE_ERROR)
        assert "--attach-to-registry needs --output" in _plain(result.output)


class TestDiagnoseWithoutTheFlag:
    @pytest.mark.parametrize(
        "output_args", [[], ["--output", "report.json"]], ids=["bare", "output"]
    )
    def test_the_registry_is_never_opened(self, workdir, output_args):
        result = runner.invoke(app, ["diagnose", "run-1", *output_args])

        _ok(result)
        out = _plain(result.output)
        assert "registry" not in out.lower(), out
        assert not (workdir / DB_NAME).exists()
        if output_args:
            assert "Wrote report.json" in out
            assert (workdir / "report.json").is_file()

    def test_an_existing_entry_gains_no_artifact(self, workdir):
        entry_id = _push(workdir)

        result = runner.invoke(app, ["diagnose", "run-1", "--output", "report.json"])

        _ok(result)
        assert _artifacts(workdir, entry_id) == []


# ---------------------------------------------------------------------------
# soup eval unlearning: the same defect, in its own copy of the call
# ---------------------------------------------------------------------------


class TestEvalUnlearningAttach:
    def test_the_report_is_stored_on_the_entry(self, workdir):
        entry_id = _push(workdir)

        result = runner.invoke(
            app,
            [
                "eval", "unlearning", "run-1",
                "--output", "unlearn.json", "--attach-to-registry", entry_id,
            ],
        )

        _ok(result)
        _assert_stored(workdir, entry_id, "eval_results", workdir / "unlearn.json")
        out = _plain(result.output)
        assert f"Attached eval_results -> registry {entry_id}" in out
        assert "Registry attach failed" not in out

    def test_an_unknown_entry_exits_one_and_names_it(self, workdir):
        entry_id = _push(workdir)

        result = runner.invoke(
            app,
            [
                "eval", "unlearning", "run-1",
                "--output", "unlearn.json", "--attach-to-registry", "no-such-entry",
            ],
        )

        _exits(result, 1)
        out = _plain(result.output)
        assert "Registry attach failed: registry entry not found: no-such-entry" in out, out
        assert "Attached" not in out
        assert _artifacts(workdir, entry_id) == []
        assert (workdir / "unlearn.json").is_file()

    def test_without_output_it_is_refused_before_the_eval_runs(self, workdir, monkeypatch):
        entry_id = _push(workdir)
        monkeypatch.setattr(
            unlearning_eval,
            "run_unlearn_eval",
            lambda **kwargs: pytest.fail("the eval must not run when nothing can be attached"),
        )

        result = runner.invoke(
            app, ["eval", "unlearning", "run-1", "--attach-to-registry", entry_id]
        )

        _exits(result, 2)
        out = _plain(result.output)
        assert "--attach-to-registry requires --output (nothing written to attach)." in out, out
        assert _artifacts(workdir, entry_id) == []

    def test_without_the_flag_the_registry_is_never_opened(self, workdir):
        result = runner.invoke(app, ["eval", "unlearning", "run-1", "--output", "unlearn.json"])

        _ok(result)
        out = _plain(result.output)
        assert "Wrote report -> unlearn.json" in out
        assert "registry" not in out.lower(), out
        assert not (workdir / DB_NAME).exists()
