"""`soup init` refuses a symbolic link, junction or other reparse point at the starter config path.

The other config writers reach the file through the shared helpers in
``soup_cli.utils.paths``, which refuse a link at the write target. ``soup init``
wrote with ``Path.write_text``, which follows one, so a ``soup.yaml`` link sent
the starter config wherever the link pointed (a dangling link even skipped the
overwrite prompt). Every test here drives the real command through
``CliRunner``. Ordinary targets, including ones outside the working directory
and ones reached through a linked parent directory, keep their previous
behaviour.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.templates import load_template
from tests.conftest import strip_ansi

runner = CliRunner()

ORIGINAL = "content that soup init must leave alone\n"

REFUSAL = "is a symbolic link, junction or other reparse point"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Wide enough that Rich never folds a temp-directory path across lines.

    The console itself is replaced, because ``COLUMNS`` for the call does not
    reach it: ``commands/init.py`` builds its console at import, and Rich keeps
    the width a console was built with when ``COLUMNS`` was set at that moment.
    A pytest-xdist worker on Linux starts with ``COLUMNS=80`` (its controller
    imports GNU readline, which exports the terminal size to child processes),
    so there the refusal folded the link path at column 80.
    """
    import soup_cli.commands.init as init_module

    monkeypatch.setattr(init_module, "console", Console(width=1000))


def _init(*args: str, **kwargs):
    return runner.invoke(app, ["init", *args], **kwargs)


def _control(tmp_path: Path, template: str) -> Path:
    """A file written the way ``soup init`` wrote before: ``Path.write_text``."""
    control = tmp_path / "control.yaml"
    control.write_text(load_template(template), encoding="utf-8")
    return control


def _just_before_writing(monkeypatch, path: Path, change) -> list:
    """Call ``change()`` in the last moment before ``path`` is created or opened.

    A new file is created with ``os.open`` on POSIX and by renaming a staging
    file onto ``path`` on Windows; an existing file is opened with ``os.open``.
    Both calls are hooked, so the change lands right before whichever one runs.
    The returned list is non-empty once ``change()`` has run.
    """
    changed = []
    real_open = os.open
    real_rename = os.rename

    def change_once(target):
        if os.fspath(target) == str(path) and not changed:
            changed.append(True)
            change()

    def open_hook(file, flags, *args, **kwargs):
        change_once(file)
        return real_open(file, flags, *args, **kwargs)

    def rename_hook(src, dst, *args, **kwargs):
        change_once(dst)
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_hook)
    monkeypatch.setattr(os, "rename", rename_hook)
    return changed


class TestALinkAtTheOutputPath:
    @pytest.mark.requires_symlink
    def test_force_does_not_write_through_a_file_symlink(self, tmp_path):
        victim = tmp_path / "victim.txt"
        victim.write_text(ORIGINAL, encoding="utf-8")
        link = tmp_path / "soup.yaml"
        os.symlink(str(victim), str(link))

        result = _init("--template", "chat", "--output", str(link), "--force")

        assert result.exit_code == 1, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert str(link) in out
        assert REFUSAL in out
        assert victim.read_text(encoding="utf-8") == ORIGINAL
        assert os.path.islink(link)

    @pytest.mark.requires_symlink
    def test_refused_before_the_overwrite_prompt(self, tmp_path):
        victim = tmp_path / "victim.txt"
        victim.write_text(ORIGINAL, encoding="utf-8")
        link = tmp_path / "soup.yaml"
        os.symlink(str(victim), str(link))

        result = _init("--template", "chat", "--output", str(link), input="y\n")

        assert result.exit_code == 1, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert REFUSAL in out
        assert "Overwrite?" not in out
        assert victim.read_text(encoding="utf-8") == ORIGINAL

    @pytest.mark.requires_symlink
    def test_refused_before_the_wizard_runs(self, tmp_path):
        victim = tmp_path / "victim.txt"
        victim.write_text(ORIGINAL, encoding="utf-8")
        link = tmp_path / "soup.yaml"
        os.symlink(str(victim), str(link))

        result = _init("--output", str(link), "--force")

        assert result.exit_code == 1, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert REFUSAL in out
        assert "Soup Config Wizard" not in out
        assert victim.read_text(encoding="utf-8") == ORIGINAL

    @pytest.mark.requires_symlink
    def test_dangling_symlink_creates_nothing(self, tmp_path):
        missing = tmp_path / "created-through-the-link.yaml"
        link = tmp_path / "soup.yaml"
        os.symlink(str(missing), str(link))

        result = _init("--template", "chat", "--output", str(link))

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert REFUSAL in _plain(result.output)
        assert not os.path.lexists(missing)

    @pytest.mark.skipif(os.name != "nt", reason="Windows junctions only exist on Windows")
    def test_junction_at_the_output_path_is_refused(self, tmp_path):
        import _winapi

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        junction = tmp_path / "soup.yaml"
        _winapi.CreateJunction(str(elsewhere), str(junction))

        result = _init("--template", "chat", "--output", str(junction), "--force")

        assert result.exit_code == 1, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert str(junction) in out
        assert REFUSAL in out
        assert not any(elsewhere.iterdir())

    @pytest.mark.requires_symlink
    def test_link_placed_after_the_first_check_is_refused(self, tmp_path, monkeypatch):
        """The write itself does not follow a link, whenever the link appeared."""
        import soup_cli.commands.init as init_module

        victim = tmp_path / "victim.txt"
        victim.write_text(ORIGINAL, encoding="utf-8")
        link = tmp_path / "soup.yaml"
        real_load_template = init_module.load_template

        def load_then_link(name):
            text = real_load_template(name)
            os.symlink(str(victim), str(link))
            return text

        monkeypatch.setattr(init_module, "load_template", load_then_link)
        result = _init("--template", "chat", "--output", str(link))

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert REFUSAL in _plain(result.output)
        assert victim.read_text(encoding="utf-8") == ORIGINAL

    @pytest.mark.requires_symlink
    def test_link_placed_just_before_the_file_is_created_leaves_its_target_intact(
        self, tmp_path, monkeypatch
    ):
        victim = tmp_path / "victim.txt"
        victim.write_text(ORIGINAL, encoding="utf-8")
        link = tmp_path / "soup.yaml"

        changed = _just_before_writing(
            monkeypatch, link, lambda: os.symlink(str(victim), str(link))
        )
        result = _init("--template", "chat", "--output", str(link))
        monkeypatch.undo()

        assert changed, "the write never reached the call that creates the file"
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert REFUSAL in _plain(result.output)
        assert victim.read_text(encoding="utf-8") == ORIGINAL

    @pytest.mark.requires_symlink
    def test_dangling_link_placed_just_before_the_file_is_created_creates_nothing(
        self, tmp_path, monkeypatch
    ):
        """An exclusive create still follows a dangling link on Windows; this one must not."""
        missing = tmp_path / "created-through-the-link.yaml"
        link = tmp_path / "soup.yaml"

        changed = _just_before_writing(
            monkeypatch, link, lambda: os.symlink(str(missing), str(link))
        )
        result = _init("--template", "chat", "--output", str(link))
        monkeypatch.undo()

        assert changed, "the write never reached the call that creates the file"
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert REFUSAL in _plain(result.output)
        assert not os.path.lexists(missing)
        assert os.path.islink(link)

    @pytest.mark.requires_symlink
    def test_existing_file_swapped_for_a_link_just_before_the_open_leaves_its_target_intact(
        self, tmp_path, monkeypatch
    ):
        """No truncation happens before the open-time link check has passed."""
        victim = tmp_path / "victim.txt"
        victim.write_text(ORIGINAL, encoding="utf-8")
        output = tmp_path / "soup.yaml"
        output.write_text("# the user's own config\n", encoding="utf-8")

        def swap():
            os.unlink(output)
            os.symlink(str(victim), str(output))

        changed = _just_before_writing(monkeypatch, output, swap)
        result = _init("--template", "chat", "--output", str(output), "--force")
        monkeypatch.undo()

        assert changed, "the write never reached the call that opens the file"
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert REFUSAL in _plain(result.output)
        assert victim.read_text(encoding="utf-8") == ORIGINAL

    @pytest.mark.requires_symlink
    def test_existing_file_swapped_for_a_dangling_link_just_before_the_open_creates_nothing(
        self, tmp_path, monkeypatch
    ):
        """An existing file is opened without O_CREAT, so a dangling link's target is never made."""
        missing = tmp_path / "created-through-the-link.yaml"
        output = tmp_path / "soup.yaml"
        output.write_text("# the user's own config\n", encoding="utf-8")

        def swap():
            os.unlink(output)
            os.symlink(str(missing), str(output))

        changed = _just_before_writing(monkeypatch, output, swap)
        result = _init("--template", "chat", "--output", str(output), "--force")
        monkeypatch.undo()

        assert changed, "the write never reached the call that opens the file"
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert REFUSAL in _plain(result.output)
        assert not os.path.lexists(missing)


class TestOtherErrorsAreReportedAsThemselves:
    def test_missing_parent_directory_is_not_reported_as_a_link(self, tmp_path):
        output = tmp_path / "missing" / "soup.yaml"

        result = _init("--template", "chat", "--output", str(output))

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert isinstance(result.exception, FileNotFoundError), repr(result.exception)
        out = _plain(result.output)
        assert REFUSAL not in out
        assert "symbolic link" not in out
        assert not (tmp_path / "missing").exists()

    def test_an_error_inspecting_the_target_is_not_reported_as_a_link(
        self, tmp_path, monkeypatch
    ):
        import soup_cli.commands.init as init_module

        def unreadable(path, *, stop_at=None):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))

        monkeypatch.setattr(init_module, "refuse_linked_dirs", unreadable)
        result = _init("--template", "chat", "--output", str(tmp_path / "soup.yaml"))

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert isinstance(result.exception, PermissionError), repr(result.exception)
        assert REFUSAL not in _plain(result.output)
        assert not (tmp_path / "soup.yaml").exists()


class TestOrdinaryPathsAreUnchanged:
    def test_new_file_is_byte_identical_to_the_previous_writer(self, tmp_path):
        output = tmp_path / "soup.yaml"

        result = _init("--template", "chat", "--output", str(output))

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert output.read_bytes() == _control(tmp_path, "chat").read_bytes()
        assert not os.path.islink(output)
        assert "Config saved to" in _plain(result.output)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["control.yaml", "soup.yaml"]

    def test_new_file_gets_the_same_permissions_as_before(self, tmp_path):
        output = tmp_path / "soup.yaml"

        result = _init("--template", "chat", "--output", str(output))

        assert result.exit_code == 0, (result.output, repr(result.exception))
        control = _control(tmp_path, "chat")
        assert stat.S_IMODE(output.stat().st_mode) == stat.S_IMODE(control.stat().st_mode)

    def test_force_replaces_a_longer_regular_file_completely(self, tmp_path):
        output = tmp_path / "soup.yaml"
        output.write_text("# old\n" * 20_000, encoding="utf-8")

        result = _init("--template", "chat", "--output", str(output), "--force")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert output.read_bytes() == _control(tmp_path, "chat").read_bytes()

    def test_null_device_is_still_a_valid_output(self):
        """Only a regular file is truncated, as O_TRUNC did; a device is written as before."""
        result = _init("--template", "chat", "--output", os.devnull, "--force")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "Config saved to" in _plain(result.output)

    def test_confirmed_overwrite_of_a_regular_file_still_works(self, tmp_path):
        output = tmp_path / "soup.yaml"
        output.write_text(ORIGINAL, encoding="utf-8")

        result = _init("--template", "chat", "--output", str(output), input="y\n")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "Overwrite?" in _plain(result.output)
        assert output.read_bytes() == _control(tmp_path, "chat").read_bytes()

    @pytest.mark.requires_symlink
    def test_a_symlinked_parent_directory_is_not_refused(self, tmp_path):
        """Only the target is inspected: a linked directory above it is the user's layout."""
        real = tmp_path / "real"
        real.mkdir()
        linked = tmp_path / "linked"
        os.symlink(str(real), str(linked), target_is_directory=True)

        result = _init("--template", "chat", "--output", str(linked / "soup.yaml"))

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (real / "soup.yaml").read_bytes() == _control(tmp_path, "chat").read_bytes()

    @pytest.mark.skipif(os.name != "nt", reason="Windows junctions only exist on Windows")
    def test_a_junctioned_parent_directory_is_not_refused(self, tmp_path):
        import _winapi

        real = tmp_path / "real"
        real.mkdir()
        junction = tmp_path / "junction"
        _winapi.CreateJunction(str(real), str(junction))

        result = _init("--template", "chat", "--output", str(junction / "soup.yaml"))

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (real / "soup.yaml").read_bytes() == _control(tmp_path, "chat").read_bytes()
