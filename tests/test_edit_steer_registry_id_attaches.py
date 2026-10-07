"""``--registry-id`` on ``soup steer train`` / ``soup edit set`` records the result, or fails.

Both commands save a DIRECTORY (a steering vector plus its config, an edited
model, a GRACE codebook) and passed that directory to the registry helper. The
registry stores one hashed FILE per artifact row and refuses anything else, so
every run printed ``Could not attach to Registry: artifact not found: <dir>``
as a warning and exited 0: nothing was recorded, and the shell was told the
command had succeeded. ``soup edit set --registry-id`` without ``--output``
skipped the attach without a word.

A requested attach is part of the command's success, as it already is for
``soup bom emit``, ``soup attest emit`` and ``soup eval custom``:

* the file(s) that make up the result are stored on the entry;
* an attach that fails exits 1 and names the problem, with the result on disk;
* ``soup edit set --registry-id`` without ``--output`` is refused before the run.

Every test drives the real command line against a real registry store in a
throwaway SQLite file (``SOUP_REGISTRY_DB_PATH``) and asserts on the rows the
store holds, with the real ``attach_artifact`` in between. Only the model work
is replaced: the stand-ins write the same files the real fit / edit writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.registry.attach as attach_mod
import soup_cli.utils.knowledge_edit as knowledge_edit
import soup_cli.utils.steering as steering
from soup_cli.cli import app
from soup_cli.registry.store import RegistryStore
from tests.conftest import strip_ansi

runner = CliRunner()

DB_NAME = "registry.db"
VECTOR_NAME = "steering_vector.safetensors"
CONFIG_NAME = "steering_config.json"
CODEBOOK_NAME = "grace_codebook.json"


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
    monkeypatch.setenv("COLUMNS", "200")
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


def _assert_row(row: dict, kind: str, written: Path) -> None:
    assert row["kind"] == kind, row
    assert os.path.samefile(row["path"], written), row
    assert row["sha256"] == hashlib.sha256(written.read_bytes()).hexdigest()
    assert row["size_bytes"] == written.stat().st_size


def _assert_stored(workdir: Path, entry_id: str, kind: str, written: list[Path]) -> None:
    """The entry holds exactly ``written``, in order, each as ``kind``."""
    rows = _artifacts(workdir, entry_id)
    assert len(rows) == len(written), rows
    for row, path in zip(rows, written):
        _assert_row(row, kind, path)


# ---------------------------------------------------------------------------
# soup steer train
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace the live fit with one that writes the two files a real fit writes."""
    calls: list[dict] = []

    def _fit(*, method, name, pairs_path, base, layer=None, output_dir=None, **kwargs):
        import numpy as np
        from safetensors.numpy import save_file

        out_dir = output_dir or os.path.join("steering", name)
        os.makedirs(out_dir, exist_ok=True)
        save_file(
            {"vector": np.asarray([0.5, -0.25, 0.125, 1.0], dtype=np.float32)},
            os.path.join(out_dir, VECTOR_NAME),
        )
        config = {
            "method": method,
            "name": name,
            "layer": 1,
            "hidden_dim": 4,
            "intervention_point": "residual",
            "base": base,
            "num_pairs": 2,
            "default_strength": 1.0,
        }
        Path(out_dir, CONFIG_NAME).write_text(json.dumps(config), encoding="utf-8")
        calls.append({"output_dir": out_dir})
        return steering.SteeringArtifact(
            method=method,
            name=name,
            layer=1,
            hidden_dim=4,
            intervention_point="residual",
            output_dir=out_dir,
            base=base,
            num_pairs=2,
        )

    monkeypatch.setattr(steering, "build_steering_vector", _fit)
    return calls


def _steer(workdir: Path, *extra: str, name: str = "polite"):
    (workdir / "pairs.jsonl").write_text(
        json.dumps({"positive": "Thank you.", "negative": "Go away."}) + "\n",
        encoding="utf-8",
    )
    return runner.invoke(
        app,
        ["steer", "train", "--base", "tiny-base", "--method", "caa", "--name", name,
         "--pairs", "pairs.jsonl", *extra],
    )


class TestSteerTrainAttachesTheVector:
    @pytest.mark.parametrize(
        "output",
        [None, "vectors/polite-v2", "ABSOLUTE"],
        ids=["default-dir", "custom-dir", "absolute-dir"],
    )
    def test_the_vector_file_is_stored_on_the_entry(self, workdir, fitted, output):
        entry_id = _push(workdir)
        if output is None:
            out_dir, extra = workdir / "steering" / "polite", []
        elif output == "ABSOLUTE":
            out_dir = workdir / "abs-out"
            extra = ["--output", str(out_dir)]
        else:
            out_dir, extra = workdir / output, ["--output", output]

        result = _steer(workdir, *extra, "--registry-id", entry_id)

        _ok(result)
        _assert_stored(workdir, entry_id, "steering_vector", [out_dir / VECTOR_NAME])
        out = _plain(result.output)
        assert f"Attached steering_vector to registry entry {entry_id}" in out
        assert "ould not attach" not in out

    @pytest.mark.parametrize("spelling", ["full-id", "id-prefix", "registry-url", "name-tag"])
    def test_every_spelling_of_the_entry_reference_attaches(self, workdir, fitted, spelling):
        entry_id = _push(workdir, name="demo", tag="v1")
        reference = {
            "full-id": entry_id,
            "id-prefix": entry_id[:-4],
            "registry-url": f"registry://{entry_id}",
            "name-tag": "demo:v1",
        }[spelling]

        result = _steer(workdir, "--registry-id", reference)

        _ok(result)
        _assert_stored(
            workdir, entry_id, "steering_vector", [workdir / "steering" / "polite" / VECTOR_NAME]
        )

    def test_an_unknown_entry_exits_1_and_keeps_the_vector(self, workdir, fitted):
        entry_id = _push(workdir)

        result = _steer(workdir, "--registry-id", "no-such-entry")

        _exits(result, 1)
        out = _plain(result.output)
        assert "could not attach to registry: registry entry not found: no-such-entry" in out
        assert "Attached steering_vector" not in out
        assert _artifacts(workdir, entry_id) == []
        # The fit is not thrown away: both files are still where they were written.
        assert (workdir / "steering" / "polite" / VECTOR_NAME).is_file()
        assert (workdir / "steering" / "polite" / CONFIG_NAME).is_file()

    def test_an_unreadable_registry_exits_1_with_a_message(self, workdir, fitted, monkeypatch):
        """Not only the two exception types the old ``except`` listed."""
        entry_id = _push(workdir)

        def _locked(entry, *, path, kind, enforce_cwd=True):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(attach_mod, "attach_artifact", _locked)

        result = _steer(workdir, "--registry-id", entry_id)

        _exits(result, 1)
        assert "could not attach to registry: database is locked" in _plain(result.output)
        assert not isinstance(result.exception, sqlite3.OperationalError)

    def test_a_missing_vector_file_exits_1_and_names_it(self, workdir, fitted, monkeypatch):
        entry_id = _push(workdir)
        real_fit = steering.build_steering_vector

        def _fit_without_vector(**kwargs):
            artifact = real_fit(**kwargs)
            os.remove(os.path.join(artifact.output_dir, VECTOR_NAME))
            return artifact

        monkeypatch.setattr(steering, "build_steering_vector", _fit_without_vector)

        result = _steer(workdir, "--registry-id", entry_id)

        _exits(result, 1)
        out = _plain(result.output)
        assert "could not attach to registry: artifact not found:" in out
        assert VECTOR_NAME in out
        assert _artifacts(workdir, entry_id) == []

    def test_plan_only_attaches_nothing(self, workdir, fitted):
        entry_id = _push(workdir)

        result = _steer(workdir, "--plan-only", "--registry-id", entry_id)

        _ok(result)
        assert fitted == []
        assert _artifacts(workdir, entry_id) == []

    def test_without_the_flag_nothing_is_attached(self, workdir, fitted):
        entry_id = _push(workdir)

        result = _steer(workdir)

        _ok(result)
        assert _artifacts(workdir, entry_id) == []
        assert "registry" not in _plain(result.output).lower()


class TestARegisteredVectorResolvesByName:
    """The row is what ``soup steer apply`` / ``soup serve --steer`` look a name up by."""

    def test_a_vector_trained_outside_the_default_dir_is_found_through_the_registry(
        self, workdir, fitted
    ):
        entry_id = _push(workdir, name="polite", tag="v1")
        _ok(_steer(workdir, "--output", "vectors/polite-v2", "--registry-id", entry_id))
        assert not (workdir / "steering" / "polite").exists()

        resolved = steering.resolve_steering_dir("polite")

        assert os.path.samefile(resolved, workdir / "vectors" / "polite-v2")
        loaded = steering.load_steering_artifact(resolved)
        assert loaded.name == "polite"
        assert loaded.vector.tolist() == [0.5, -0.25, 0.125, 1.0]

    def test_a_name_with_no_row_and_no_default_dir_is_still_refused(self, workdir, fitted):
        _push(workdir, name="polite", tag="v1")

        with pytest.raises(ValueError, match="no steering vector named 'polite'"):
            steering.resolve_steering_dir("polite")

    def test_a_row_whose_directory_lost_its_config_is_not_returned(self, workdir, fitted):
        entry_id = _push(workdir, name="polite", tag="v1")
        _ok(_steer(workdir, "--output", "vectors/polite-v2", "--registry-id", entry_id))
        (workdir / "vectors" / "polite-v2" / CONFIG_NAME).unlink()

        with pytest.raises(ValueError, match="no steering vector named 'polite'"):
            steering.resolve_steering_dir("polite")

    def test_steer_list_shows_the_attached_vector(self, workdir, fitted):
        entry_id = _push(workdir, name="polite", tag="v1")
        _ok(_steer(workdir, "--registry-id", entry_id))

        result = runner.invoke(app, ["steer", "list"])

        _ok(result)
        out = _plain(result.output)
        assert "No steering_vector artifacts registered" not in out
        assert "polite" in out


# ---------------------------------------------------------------------------
# soup edit set
# ---------------------------------------------------------------------------

# What ``save_pretrained`` writes next to the weights; none of it is the edit.
_SIDE_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "training_args.bin",
    "adapter_model.safetensors",
    "model.safetensors.index.json",
)

# Every weight layout ``save_pretrained`` produces, in the order it is attached.
_WEIGHT_LAYOUTS = {
    "one-safetensors": ["model.safetensors"],
    "safetensors-shards": [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ],
    "one-bin": ["pytorch_model.bin"],
    "bin-shards": ["pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"],
}


@pytest.fixture
def edited(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the live edit with one that writes what a real edit writes."""
    state: dict = {"calls": 0, "weights": ["model.safetensors"]}

    def _apply(plan, *, output_dir=None, **kwargs):
        state["calls"] += 1
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            if plan.method == "grace":
                Path(output_dir, CODEBOOK_NAME).write_text('{"entries": []}', encoding="utf-8")
            else:
                for index, weight in enumerate(state["weights"]):
                    Path(output_dir, weight).write_bytes(b"weights-" + bytes([65 + index]) * 64)
                for side in _SIDE_FILES:
                    Path(output_dir, side).write_bytes(b"{}")
        return knowledge_edit.EditResult(
            method=plan.method,
            layer=1,
            norm_delta=0.25,
            layers_edited=(1,),
            output_dir=output_dir,
            target_prob_before=0.1,
            target_prob_after=0.9,
            governed=False,
        )

    monkeypatch.setattr(knowledge_edit, "apply_edit", _apply)
    return state


def _edit(*extra: str, method: str = "rome"):
    return runner.invoke(
        app,
        ["edit", "set", "--base", "tiny-base", "--method", method, "--subject",
         "Paris is the capital of France", "--target", "Lyon", "--no-governor", *extra],
    )


class TestEditSetAttachesTheResult:
    @pytest.mark.parametrize("layout", sorted(_WEIGHT_LAYOUTS))
    def test_the_weight_files_are_stored_on_the_entry(self, workdir, edited, layout):
        entry_id = _push(workdir)
        edited["weights"] = _WEIGHT_LAYOUTS[layout]

        result = _edit("--output", "edited", "--registry-id", entry_id)

        _ok(result)
        _assert_stored(
            workdir,
            entry_id,
            "edited_model",
            [workdir / "edited" / name for name in _WEIGHT_LAYOUTS[layout]],
        )
        out = _plain(result.output)
        assert f"Attached edited_model to registry entry {entry_id}" in out
        assert "ould not attach" not in out

    @pytest.mark.parametrize("method", ["rome", "memit", "alphaedit"])
    def test_every_weight_edit_method_attaches_an_edited_model(self, workdir, edited, method):
        entry_id = _push(workdir)

        result = _edit("--output", "edited", "--registry-id", entry_id, method=method)

        _ok(result)
        _assert_stored(
            workdir, entry_id, "edited_model", [workdir / "edited" / "model.safetensors"]
        )

    def test_a_grace_edit_attaches_the_codebook_file(self, workdir, edited):
        entry_id = _push(workdir)

        result = _edit("--output", "book", "--registry-id", entry_id, method="grace")

        _ok(result)
        _assert_stored(workdir, entry_id, "grace_codebook", [workdir / "book" / CODEBOOK_NAME])
        assert f"Attached grace_codebook to registry entry {entry_id}" in _plain(result.output)

    @pytest.mark.parametrize("plan_only", [False, True], ids=["apply", "plan-only"])
    def test_without_output_the_flag_is_refused_before_the_edit(self, workdir, edited, plan_only):
        entry_id = _push(workdir)
        extra = ["--plan-only"] if plan_only else []

        result = _edit("--registry-id", entry_id, *extra)

        _exits(result, 2)
        assert "--registry-id needs --output (nothing saved to attach)" in _plain(result.output)
        assert edited["calls"] == 0
        assert _artifacts(workdir, entry_id) == []

    def test_an_unknown_entry_exits_1_and_keeps_the_edited_model(self, workdir, edited):
        entry_id = _push(workdir)

        result = _edit("--output", "edited", "--registry-id", "no-such-entry")

        _exits(result, 1)
        out = _plain(result.output)
        assert "could not attach to registry: registry entry not found: no-such-entry" in out
        assert "Attached edited_model" not in out
        assert _artifacts(workdir, entry_id) == []
        assert (workdir / "edited" / "model.safetensors").is_file()

    def test_an_output_dir_with_no_weight_file_exits_1_and_says_so(self, workdir, edited):
        entry_id = _push(workdir)
        edited["weights"] = []

        result = _edit("--output", "edited", "--registry-id", entry_id)

        _exits(result, 1)
        out = _plain(result.output)
        assert "could not attach to registry: no model weight file" in out
        assert "edited" in out
        assert _artifacts(workdir, entry_id) == []

    def test_an_unreadable_registry_exits_1_with_a_message(self, workdir, edited, monkeypatch):
        entry_id = _push(workdir)

        def _locked(entry, *, path, kind, enforce_cwd=True):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(attach_mod, "attach_artifact", _locked)

        result = _edit("--output", "edited", "--registry-id", entry_id)

        _exits(result, 1)
        assert "could not attach to registry: database is locked" in _plain(result.output)
        assert not isinstance(result.exception, sqlite3.OperationalError)

    def test_without_the_flag_nothing_is_attached(self, workdir, edited):
        entry_id = _push(workdir)

        result = _edit("--output", "edited")

        _ok(result)
        assert _artifacts(workdir, entry_id) == []
        assert "registry" not in _plain(result.output).lower()

    def test_plan_only_with_output_attaches_nothing(self, workdir, edited):
        entry_id = _push(workdir)

        result = _edit("--output", "edited", "--plan-only", "--registry-id", entry_id)

        _ok(result)
        assert edited["calls"] == 0
        assert _artifacts(workdir, entry_id) == []


class TestTheHelpSaysWhatIsAttached:
    @pytest.mark.parametrize(
        "command, needle",
        [
            (["steer", "train"], "steering_vector"),
            (["edit", "set"], "needs --output"),
        ],
    )
    def test_help_names_the_artifact_and_the_failure_code(self, workdir, command, needle):
        result = runner.invoke(app, [*command, "--help"])

        _ok(result)
        # The help table wraps a long description; drop its borders before joining.
        out = _plain(result.output.replace("│", " "))
        assert needle in out
        assert "Exits 1 when the attach fails" in out
