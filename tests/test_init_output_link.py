"""`soup init` refuses a symbolic link or junction at the starter config path.

The other config writers reach the file through the shared helpers in
``soup_cli.utils.paths``, which refuse a link at the write target. ``soup init``
wrote with ``Path.write_text``, which follows one, so a ``soup.yaml`` link sent
the starter config wherever the link pointed (a dangling link even skipped the
overwrite prompt). Every test here drives the real command through
``CliRunner``. Ordinary targets, including ones outside the working directory,
keep their previous behaviour.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.templates import load_template
from tests.conftest import strip_ansi

runner = CliRunner()

#: Wide enough that Rich never folds a temp-directory path across lines.
WIDE = {"COLUMNS": "1000"}

ORIGINAL = "content that soup init must leave alone\n"

REFUSAL = "is a symbolic link or junction"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _init(*args: str, **kwargs):
    return runner.invoke(app, ["init", *args], env=WIDE, **kwargs)


def _control(tmp_path: Path, template: str) -> Path:
    """A file written the way ``soup init`` wrote before: ``Path.write_text``."""
    control = tmp_path / "control.yaml"
    control.write_text(load_template(template), encoding="utf-8")
    return control


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
    def test_link_placed_just_before_the_open_leaves_its_target_intact(
        self, tmp_path, monkeypatch
    ):
        """No truncation happens before the open-time link checks have passed."""
        victim = tmp_path / "victim.txt"
        victim.write_text(ORIGINAL, encoding="utf-8")
        link = tmp_path / "soup.yaml"
        real_open = os.open
        planted = []

        def open_after_linking(path, flags, *args, **kwargs):
            if os.fspath(path) == str(link) and not planted:
                planted.append(True)
                os.symlink(str(victim), str(link))
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", open_after_linking)
        result = _init("--template", "chat", "--output", str(link))
        monkeypatch.undo()

        assert planted, "the write never reached os.open"
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert REFUSAL in _plain(result.output)
        assert victim.read_text(encoding="utf-8") == ORIGINAL


class TestOrdinaryPathsAreUnchanged:
    def test_new_file_is_byte_identical_to_the_previous_writer(self, tmp_path):
        output = tmp_path / "soup.yaml"

        result = _init("--template", "chat", "--output", str(output))

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert output.read_bytes() == _control(tmp_path, "chat").read_bytes()
        assert not os.path.islink(output)
        assert "Config saved to" in _plain(result.output)

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
