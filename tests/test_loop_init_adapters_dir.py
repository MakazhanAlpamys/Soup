"""``soup loop init`` creates the adapters directory and counts in singular.

The loop deploys through ``soup serve --adapter-dir .soup-loops/adapters``, but
serve refuses a missing directory and init did not create it, so the documented
first command failed until the user made the directory by hand. Init also
printed ``training starts at 1 new edits`` when the minimum was 1.

Init now creates ``.soup-loops/adapters`` owner-only on POSIX (mode 0o700)
before writing loop.yaml, so a failure there leaves no half-initialized loop
behind; the panel says ``1 new edit`` / ``1 new rated answer``. A planted
link, a file, or another owner's folder at the adapters path is refused with
the fix named, and a folder this account owns is kept and tightened to
owner-only, with one line saying what changed and why — including one the
train stage made group-writable before this change. On Windows a junction at
the path is refused the same way. A re-init refused for an existing loop
leaves a loose adapters folder exactly as it was. The train stage creates
the folder through the same helper, so the loop does not make a
group-writable one again.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
from soup_cli.commands.loop import app

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

_INIT = ("init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--base-model", "org/base")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def _panel_text(text: str) -> str:
    """Output with the panel's box characters removed, so a wrapped sentence reads whole."""
    return _clean(text.replace("│", " ").replace("|", " "))


def _cli(*args):
    result = CliRunner().invoke(app, list(args))
    return result, _clean(result.output), _panel_text(result.output)


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in (
        "SOUP_LOOP_TRACE_DIR",
        "SOUP_LOOP_THUMBS_DB",
        "SOUP_LOOP_SERVE_ENDPOINT",
        "SOUP_LOOP_FEEDBACK_LOG",
    ):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    return tmp_path


def _mode(path: str) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def _loop_state(**kwargs):
    from soup_cli.utils.loop_state import LoopState

    # A registry:// served model needs a known base to reach training.
    kwargs.setdefault("base_model", "hf-internal-testing/tiny-random-gpt2")
    return LoopState(served_model="registry://dummy", eval_suite="e", baseline="b", **kwargs)


class _TrainRunner:
    """Stand-in for ``subprocess.run`` recording every training invocation."""

    def __init__(self):
        self.argvs: list = []
        self.returncode = 0

    def __call__(self, argv, **kwargs):
        self.argvs.append(list(argv))
        return self


def _train_once(root, monkeypatch):
    """One real train stage on ``root``; returns (outcome, runner)."""
    rows = []
    for i in range(1):
        rows.append({"id": f"{i}a", "prompt": f"Q{i}", "output": "good", "signal": "thumbs_up"})
        rows.append({"id": f"{i}b", "prompt": f"Q{i}", "output": "bad", "signal": "thumbs_down"})
    (root / "traces").mkdir(exist_ok=True)
    (root / "traces" / "t.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", str(root / "traces"))
    harvested = ls.harvest_from_traces(_loop_state())
    runner = _TrainRunner()
    ls._TRAIN_RUNNER = runner
    try:
        out = ls.train_dpo_from_pairs(_loop_state(), harvested)
    finally:
        ls._TRAIN_RUNNER = None
    return out, runner


class TestInitCreatesAdaptersDir:
    @pytest.mark.skipif(os.name == "nt", reason="owner-only mode is POSIX-only")
    def test_init_creates_adapters_dir_owner_only(self, cwd):
        result, _, _ = _cli(*_INIT)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert os.path.isdir(".soup-loops/adapters")
        assert _mode(".soup-loops/adapters") == 0o700
        assert os.path.isfile(".soup/loop.yaml")

    @pytest.mark.skipif(os.name == "nt", reason="owner-only mode is POSIX-only")
    def test_reinit_keeps_the_adapters_dir(self, cwd):
        _cli(*_INIT)
        assert _mode(".soup-loops/adapters") == 0o700

        result, _, _ = _cli(*_INIT, "--force")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert os.path.isdir(".soup-loops/adapters")
        assert _mode(".soup-loops/adapters") == 0o700

    @pytest.mark.skipif(os.name == "nt", reason="owner-only mode is POSIX-only")
    def test_group_writable_parent_does_not_block_init(self, cwd):
        os.makedirs(".soup-loops", exist_ok=True)
        os.chmod(".soup-loops", 0o777)

        result, _, _ = _cli(*_INIT)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert os.path.isdir(".soup-loops/adapters")
        assert _mode(".soup-loops/adapters") == 0o700
        assert os.path.isfile(".soup/loop.yaml")

    @pytest.mark.skipif(os.name == "nt", reason="owner-only mode is POSIX-only")
    def test_init_tightens_a_loose_adapters_dir_it_owns(self, cwd):
        os.makedirs(".soup-loops/adapters", exist_ok=True)
        os.chmod(".soup-loops/adapters", 0o775)

        result, text, _ = _cli(*_INIT)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert _mode(".soup-loops/adapters") == 0o700
        assert "tightened .soup-loops/adapters to owner-only (was mode 775)" in text
        assert "only the loop owner may write the adapters it serves" in text
        assert os.path.isfile(".soup/loop.yaml")

    @pytest.mark.skipif(os.name == "nt", reason="owner-only mode is POSIX-only")
    def test_init_refuses_an_adapters_dir_owned_by_another_user(self, cwd):
        os.makedirs(".soup-loops/adapters", exist_ok=True)
        try:
            os.chown(".soup-loops/adapters", 65534, -1)
        except PermissionError:
            pytest.skip("this account cannot chown a folder to another user")

        result, text, _ = _cli(*_INIT)
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "init failed" in text
        assert "belongs to another user" in text
        assert "remove it or change its owner, then re-run init" in text
        assert not os.path.exists(".soup/loop.yaml")

    @pytest.mark.skipif(os.name == "nt", reason="owner-only mode is POSIX-only")
    def test_refused_reinit_leaves_a_loose_adapters_dir_alone(self, cwd):
        result, _, _ = _cli(*_INIT)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        os.chmod(".soup-loops/adapters", 0o775)
        before = Path(".soup/loop.yaml").read_bytes()

        result, text, _ = _cli(*_INIT)
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "init failed" in text
        assert "already exists" in text
        assert "tightened" not in text
        assert _mode(".soup-loops/adapters") == 0o775
        assert Path(".soup/loop.yaml").read_bytes() == before

    @pytest.mark.skipif(os.name != "nt", reason="Windows junctions need mklink")
    def test_init_refuses_a_planted_adapters_junction(self, cwd, tmp_path):
        target = tmp_path / "elsewhere"
        target.mkdir()
        os.makedirs(".soup-loops", exist_ok=True)
        link = cwd / ".soup-loops" / "adapters"
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
        )

        result, text, _ = _cli(*_INIT)
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "init failed" in text
        assert "is a link" in text
        assert "remove the link and re-run init" in text
        assert not os.path.exists(".soup/loop.yaml")

    @pytest.mark.requires_symlink
    def test_init_refuses_a_planted_adapters_link(self, cwd, tmp_path):
        target = tmp_path / "elsewhere"
        target.mkdir()
        os.makedirs(".soup-loops", exist_ok=True)
        os.symlink(target, ".soup-loops/adapters")

        result, text, _ = _cli(*_INIT)
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "init failed" in text
        assert "is a link" in text
        assert "remove the link and re-run init" in text
        assert not os.path.exists(".soup/loop.yaml")

    def test_init_refuses_a_file_at_the_adapters_path(self, cwd):
        os.makedirs(".soup-loops", exist_ok=True)
        Path(".soup-loops/adapters").write_text("not a directory", encoding="utf-8")

        result, text, _ = _cli(*_INIT)
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "init failed" in text
        assert "is a file, not a directory" in text
        assert "remove it and re-run init" in text
        assert not os.path.exists(".soup/loop.yaml")

    def test_init_fails_cleanly_when_the_adapters_dir_cannot_be_created(
        self, cwd, monkeypatch
    ):
        def _boom():
            raise PermissionError("read-only .soup-loops")

        monkeypatch.setattr(ls, "ensure_adapters_dir", _boom)

        result, text, _ = _cli(*_INIT)
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "init failed" in text
        assert "Traceback" not in result.output
        assert not os.path.exists(".soup/loop.yaml")

    @pytest.mark.skipif(os.name == "nt", reason="owner-only mode is POSIX-only")
    def test_force_reinit_tightens_a_loop_that_already_trained(self, cwd, monkeypatch):
        _cli(*_INIT)
        out, runner = _train_once(cwd, monkeypatch)
        assert len(runner.argvs) == 1 and out["skipped"] is False
        assert _mode(".soup-loops/adapters") == 0o700
        # v0.78.0's train stage made the folder with plain mkdir: group-writable
        # under umask 002, with a run's yaml inside.
        assert list(Path(".soup-loops/adapters").glob("*.yaml"))
        os.chmod(".soup-loops/adapters", 0o775)

        result, text, _ = _cli(*_INIT, "--force")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert _mode(".soup-loops/adapters") == 0o700
        assert "tightened .soup-loops/adapters to owner-only (was mode 775)" in text
        assert list(Path(".soup-loops/adapters").glob("*.yaml"))


class TestTrainStageCreatesAdaptersDir:
    @pytest.mark.skipif(os.name == "nt", reason="owner-only mode is POSIX-only")
    def test_train_stage_on_a_fresh_tree_makes_it_owner_only(self, cwd, monkeypatch):
        out, runner = _train_once(cwd, monkeypatch)
        assert len(runner.argvs) == 1 and out["skipped"] is False

        assert os.path.isdir(".soup-loops/adapters")
        assert _mode(".soup-loops/adapters") == 0o700

    @pytest.mark.skipif(os.name == "nt", reason="owner-only mode is POSIX-only")
    def test_train_stage_tightens_a_loose_adapters_dir(self, cwd, monkeypatch, caplog):
        _train_once(cwd, monkeypatch)
        os.chmod(".soup-loops/adapters", 0o775)

        with caplog.at_level("INFO"):
            out, runner = _train_once(cwd, monkeypatch)
        assert len(runner.argvs) == 1 and out["skipped"] is False
        assert _mode(".soup-loops/adapters") == 0o700
        assert "tightened .soup-loops/adapters to owner-only (was mode 775)" in caplog.text


class TestInitPanelCounts:
    def test_one_new_edit_is_singular(self, cwd):
        result, _, panel = _cli(*_INIT, "--min-new-edits", "1", "--min-new-rated", "1")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert "training starts at 1 new edit or 1 new rated answer" in panel

    def test_default_thresholds_stay_plural(self, cwd):
        result, _, panel = _cli(*_INIT)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert "training starts at 8 new edits or 30 new rated answers" in panel
