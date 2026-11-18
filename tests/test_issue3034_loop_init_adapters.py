"""#3034 — `soup loop init` creates the `--adapter-dir` and counts in singular.

The loop deploys through ``soup serve --adapter-dir .soup-loops/adapters``, but
serve refuses a missing directory and init did not create it, so the documented
first command failed until the user made the directory by hand. Init also
printed ``training starts at 1 new edits`` when the minimum was 1.

Init now creates ``.soup-loops/adapters`` owner-only on POSIX (mode 0o700)
before writing loop.yaml, so a failure there leaves no half-initialized loop
behind; the panel says ``1 new edit`` / ``1 new rated answer``. A planted
link at the adapters path is refused, and a pre-existing group-writable
``.soup-loops`` parent (plain ``mkdir`` from an earlier run) does not block
init — only the adapters directory itself is owner-only.
"""

from __future__ import annotations

import os
import re
import stat

import pytest
from typer.testing import CliRunner

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

    def test_init_refuses_a_planted_adapters_link(self, cwd, tmp_path):
        target = tmp_path / "elsewhere"
        target.mkdir()
        os.makedirs(".soup-loops", exist_ok=True)
        os.symlink(target, ".soup-loops/adapters")

        result, text, _ = _cli(*_INIT)
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "init failed" in text
        assert not os.path.exists(".soup/loop.yaml")


class TestInitPanelCounts:
    def test_one_new_edit_is_singular(self, cwd):
        result, _, panel = _cli(*_INIT, "--min-new-edits", "1", "--min-new-rated", "1")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert "training starts at 1 new edit or 1 new rated answer" in panel

    def test_default_thresholds_stay_plural(self, cwd):
        result, _, panel = _cli(*_INIT)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert "training starts at 8 new edits or 30 new rated answers" in panel
