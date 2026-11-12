"""Issue #2726: ``soup export`` without ``--registry-id`` attaches to the run's entry.

``lookup_entry_by_output_dir`` read an ``output`` key that no registry row has, so the
documented auto-match ("by source --model output dir") never matched and an export
without ``--registry-id`` attached nothing and said nothing. An entry is now matched
through its ``run_id``: the experiment tracker's ``output_dir`` for that run is compared
with ``--model`` by realpath. When several entries match, the newest one wins, in the
order ``RegistryStore.list`` returns them. A registry or tracker database that cannot
be read turns into a warning: the export itself has already succeeded.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.commands.export as export_mod
from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.registry.attach import lookup_entry_by_output_dir
from soup_cli.registry.store import RegistryStore
from tests.conftest import strip_ansi

runner = CliRunner()


def _clean(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Scratch tracker and registry databases, with the working directory at tmp_path."""
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _completed_run(output_dir: str, *, config_output: str = "./out") -> str:
    """Start and finish a tracker run that wrote ``output_dir``; return its run id."""
    tracker = ExperimentTracker()
    try:
        run_id = tracker.start_run(
            {"base": "sshleifer/tiny-gpt2", "task": "sft", "output": config_output},
            "cpu", "cpu", {},
        )
        tracker.finish_run(run_id, 1.0, 0.5, 10, 1.0, output_dir)
    finally:
        tracker.close()
    return run_id


def _register(run_id: str, *, name: str = "chat", tag: str = "v1") -> str:
    tracker = ExperimentTracker()
    try:
        with RegistryStore() as store:
            return store.register_from_run(tracker, run_id, name=name, tag=tag)
    finally:
        tracker.close()


def _newest_entry_id() -> str:
    with RegistryStore() as store:
        return store.list(limit=1)[0]["id"]


def _corrupt(path: Path) -> Path:
    path.write_bytes(b"this is not a sqlite database\n" * 64)
    return path


def _model_dir(root: Path, name: str) -> Path:
    model_dir = root / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.onnx").write_bytes(b"fake onnx")
    return model_dir


class TestLookupEntryByOutputDir:
    @pytest.mark.parametrize("stored", ["absolute", "relative"])
    @pytest.mark.parametrize("query", ["absolute", "dot-relative"])
    def test_matches_the_entry_of_the_run_that_wrote_the_dir(self, workdir, stored, query):
        out = workdir / "out"
        out.mkdir()
        eid = _register(_completed_run(str(out) if stored == "absolute" else "out"))

        target = str(out) if query == "absolute" else "./out"
        assert lookup_entry_by_output_dir(target) == eid

    def test_matches_the_tracked_output_dir_not_the_config_output(self, workdir):
        # The trainer appends experiment_name to the config's ``output``; the tracker
        # records the directory it actually wrote.
        (workdir / "runs" / "exp1").mkdir(parents=True)
        eid = _register(_completed_run("runs/exp1", config_output="./runs"))

        assert lookup_entry_by_output_dir("runs/exp1") == eid
        assert lookup_entry_by_output_dir("runs") is None

    def test_a_dir_no_run_wrote_returns_none(self, workdir):
        (workdir / "out").mkdir()
        (workdir / "other").mkdir()
        _register(_completed_run(str(workdir / "out")))

        assert lookup_entry_by_output_dir(str(workdir / "other")) is None

    def test_an_entry_without_run_id_is_skipped(self, workdir):
        out = workdir / "out"
        out.mkdir()
        eid = _register(_completed_run(str(out)))
        # Newer than the run's entry, so it is scanned first; its config names the dir.
        with RegistryStore() as store:
            store.push(
                name="manual", tag="v1", base_model="org/base", task="sft",
                run_id=None, config={"output": str(out)},
            )

        assert lookup_entry_by_output_dir(str(out)) == eid

    def test_only_entries_without_run_id_return_none_without_opening_tracker(
        self, workdir,
    ):
        with RegistryStore() as store:
            store.push(
                name="manual", tag="v1", base_model="org/base", task="sft",
                run_id=None, config={"output": "./out"},
            )

        assert lookup_entry_by_output_dir("./out") is None
        assert not (workdir / "experiments.db").exists()

    def test_a_run_missing_from_the_tracker_is_skipped(self, workdir):
        out = workdir / "out"
        out.mkdir()
        run_id = _completed_run(str(out))
        _register(run_id)
        tracker = ExperimentTracker()
        try:
            assert tracker.delete_run(run_id)
        finally:
            tracker.close()

        assert lookup_entry_by_output_dir(str(out)) is None

    def test_an_unreadable_tracker_db_raises(self, workdir, monkeypatch):
        out = workdir / "out"
        out.mkdir()
        _register(_completed_run(str(out)))
        monkeypatch.setenv("SOUP_DB_PATH", str(_corrupt(workdir / "broken.db")))

        with pytest.raises(sqlite3.Error):
            lookup_entry_by_output_dir(str(out))

    def test_one_run_registered_twice_newest_entry_wins(self, workdir):
        out = workdir / "out"
        out.mkdir()
        run_id = _completed_run(str(out))
        ids = {_register(run_id, tag="v1"), _register(run_id, tag="v2")}

        newest = _newest_entry_id()
        assert newest in ids
        assert lookup_entry_by_output_dir(str(out)) == newest

    def test_two_runs_into_the_same_dir_newest_entry_wins(self, workdir):
        out = workdir / "out"
        out.mkdir()
        ids = {
            _register(_completed_run(str(out)), tag="v1"),
            _register(_completed_run(str(out)), tag="v2"),
        }

        newest = _newest_entry_id()
        assert newest in ids
        assert lookup_entry_by_output_dir(str(out)) == newest


class TestExportAutoAttach:
    def test_export_without_registry_id_attaches_to_the_run_entry(
        self, workdir, monkeypatch,
    ):
        _model_dir(workdir, "out")
        eid = _register(_completed_run(str(workdir / "out")))
        monkeypatch.setattr(
            export_mod, "_export_onnx", lambda *a, **k: Path("out/model.onnx"),
        )

        result = runner.invoke(app, ["export", "--model", "out", "--format", "onnx"])

        assert result.exit_code == 0, result.output
        assert (
            f"Attached export to registry entry '{eid}' as onnx (model.onnx)."
            in _clean(result.output)
        )
        with RegistryStore() as store:
            artifacts = store.get_artifacts(eid)
        assert [(a["kind"], os.path.basename(a["path"])) for a in artifacts] == [
            ("onnx", "model.onnx"),
        ]

    def test_export_of_a_dir_no_run_wrote_attaches_nothing(self, workdir, monkeypatch):
        _model_dir(workdir, "out")
        _model_dir(workdir, "other")
        eid = _register(_completed_run(str(workdir / "out")))
        monkeypatch.setattr(
            export_mod, "_export_onnx", lambda *a, **k: Path("other/model.onnx"),
        )

        result = runner.invoke(app, ["export", "--model", "other", "--format", "onnx"])

        assert result.exit_code == 0, result.output
        assert "Attached export" not in _clean(result.output)
        with RegistryStore() as store:
            assert store.get_artifacts(eid) == []

    def test_an_explicit_registry_id_overrides_the_auto_match(self, workdir, monkeypatch):
        _model_dir(workdir, "out")
        auto = _register(_completed_run(str(workdir / "out")))
        with RegistryStore() as store:
            explicit = store.push(
                name="manual", tag="v1", base_model="org/base", task="sft",
                run_id=None, config={},
            )
        monkeypatch.setattr(
            export_mod, "_export_onnx", lambda *a, **k: Path("out/model.onnx"),
        )

        result = runner.invoke(
            app,
            ["export", "--model", "out", "--format", "onnx", "--registry-id", explicit],
        )

        assert result.exit_code == 0, result.output
        assert f"Attached export to registry entry '{explicit}'" in _clean(result.output)
        with RegistryStore() as store:
            assert [a["kind"] for a in store.get_artifacts(explicit)] == ["onnx"]
            assert store.get_artifacts(auto) == []

    def test_the_newest_matching_entry_gets_the_export(self, workdir, monkeypatch):
        _model_dir(workdir, "out")
        run_id = _completed_run(str(workdir / "out"))
        first = _register(run_id, tag="v1")
        second = _register(run_id, tag="v2")
        newest = _newest_entry_id()
        older = first if newest == second else second
        monkeypatch.setattr(
            export_mod, "_export_onnx", lambda *a, **k: Path("out/model.onnx"),
        )

        result = runner.invoke(app, ["export", "--model", "out", "--format", "onnx"])

        assert result.exit_code == 0, result.output
        assert f"Attached export to registry entry '{newest}'" in _clean(result.output)
        with RegistryStore() as store:
            assert store.get_artifacts(older) == []

    @pytest.mark.parametrize("broken", ["SOUP_REGISTRY_DB_PATH", "SOUP_DB_PATH"])
    def test_an_unreadable_database_warns_and_the_export_still_succeeds(
        self, workdir, monkeypatch, broken,
    ):
        _model_dir(workdir, "out")
        _register(_completed_run(str(workdir / "out")))
        monkeypatch.setenv(broken, str(_corrupt(workdir / "broken.db")))
        calls = []

        def fake_export(*args, **kwargs):
            calls.append(args)
            return Path("out/model.onnx")

        monkeypatch.setattr(export_mod, "_export_onnx", fake_export)

        result = runner.invoke(app, ["export", "--model", "out", "--format", "onnx"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(calls) == 1
        output = _clean(result.output)
        assert "Could not look up a registry entry for this export" in output
        assert "--registry-id" in output
        assert "Attached export" not in output
