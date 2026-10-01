"""Terminal output strips the C1 control range, U+0080-U+009F.

``soup_cli.utils.terminal`` holds the one table in the tree of characters that
must never reach the terminal. It covered C0 (minus TAB / LF / CR) and DEL, but
not C1: U+0080-U+009F passed through ``strip_control`` and ``for_terminal``
unchanged, and some terminals act on a C1 character in UTF-8 output the way
they act on ESC (U+009B is a one-character CSI). Every printer built on the two
helpers inherited that, so the range is pinned here at both edges and through
two real printers.

The ``.can`` manifest ``author`` field now refuses control characters outright
(``unicodedata.category == "Cc"``: C0, DEL and C1) instead of only NUL / CR /
LF -- an author handle has no use for any of them.

Every non-ASCII character below is built with ``chr()`` so this file stays
ASCII.
"""

from __future__ import annotations

import io
import tarfile
from io import StringIO

import pytest
from rich.console import Console
from typer.testing import CliRunner

runner = CliRunner()

BACKSLASH = chr(92)
C1_CODES = tuple(range(0x80, 0xA0))


def _c1_found(text: str) -> list[str]:
    return [hex(ord(ch)) for ch in text if 0x80 <= ord(ch) <= 0x9F]


def _plain_console(buffer: StringIO) -> Console:
    return Console(file=buffer, color_system=None, width=400)


class TestTheHelpersStripC1:
    @pytest.mark.parametrize("code", C1_CODES, ids=[f"{code:#04x}" for code in C1_CODES])
    def test_every_c1_code_point_is_stripped(self, code: int) -> None:
        from soup_cli.utils.terminal import for_terminal, strip_control

        assert strip_control(f"a{chr(code)}b") == "ab"
        assert for_terminal(f"a{chr(code)}b") == "ab"

    @pytest.mark.parametrize(
        "code, kept",
        [(0x7E, True), (0x7F, False), (0x80, False), (0x9F, False), (0xA0, True), (0xA1, True)],
        ids=["0x7e-tilde", "0x7f-del", "0x80-first-c1", "0x9f-last-c1", "0xa0-nbsp", "0xa1"],
    )
    def test_the_range_edges(self, code: int, kept: bool) -> None:
        from soup_cli.utils.terminal import for_terminal, strip_control

        expected = f"a{chr(code)}b" if kept else "ab"
        assert strip_control(f"a{chr(code)}b") == expected
        assert for_terminal(f"a{chr(code)}b") == expected

    def test_printable_text_outside_ascii_is_left_alone(self) -> None:
        from soup_cli.utils.terminal import for_terminal, strip_control

        text = "caf" + chr(0xE9) + " " + chr(0x65E5) + chr(0x672C) + " " + chr(0x2014) + " x"
        assert strip_control(text) == text
        assert for_terminal(text) == text

    def test_tab_newline_and_carriage_return_are_still_kept(self) -> None:
        from soup_cli.utils.terminal import strip_control

        assert strip_control("a\tb\nc\rd") == "a\tb\nc\rd"

    def test_the_strip_runs_before_the_escape(self) -> None:
        """A C1 character between a backslash and a tag must not end up live markup.

        Escaping first would see ``backslash, C1, [bold]`` (no backslash
        directly before the bracket), add one backslash, and the later strip
        would leave two -- which Rich reads as a literal backslash followed by
        a LIVE tag. Stripping first lets the escape see the real neighbours.
        """
        from soup_cli.utils.terminal import for_terminal

        value = BACKSLASH + chr(0x9B) + "[bold]x[/bold]"
        buffer = StringIO()
        _plain_console(buffer).print(for_terminal(value))
        assert buffer.getvalue().rstrip("\n") == BACKSLASH + "[bold]x[/bold]"


def _write_can(path: str, manifest: str, config: str = "base: m\n") -> str:
    with tarfile.open(path, "w:gz") as archive:
        for name, body in (("manifest.yaml", manifest), ("config.yaml", config)):
            payload = body.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _yaml_escape(code: int) -> str:
    """The YAML double-quoted spelling of one code point, e.g. ``\\x9b``."""
    return f"{BACKSLASH}x{code:02x}"


class TestCanInspectPrintsNoC1:
    MANIFEST = (
        "can_format_version: 1\n"
        'name: "n"\n'
        'author: "a"\n'
        'created_at: "2026-01-01"\n'
        'base_hash: "h"\n'
        f'tags: ["t{_yaml_escape(0x9B)}31m"]\n'
        f'description: "{_yaml_escape(0x9D)}0;TITLE{_yaml_escape(0x9C)}d"\n'
    )

    @pytest.mark.parametrize("force_terminal", [False, True], ids=["plain", "terminal"])
    def test_description_and_tags(self, tmp_path, monkeypatch, force_terminal: bool) -> None:
        from soup_cli.cans.unpack import inspect_can
        from soup_cli.cli import app
        from soup_cli.commands import can as can_cmd

        monkeypatch.chdir(tmp_path)
        _write_can("c.can", self.MANIFEST)
        manifest = inspect_can("c.can")
        assert _c1_found(manifest.description), "fixture must carry C1 characters"
        assert _c1_found(manifest.tags[0]), "fixture must carry C1 characters"

        buffer = StringIO()
        console = (
            Console(file=buffer, force_terminal=True, width=400)
            if force_terminal
            else _plain_console(buffer)
        )
        monkeypatch.setattr(can_cmd, "console", console)
        result = runner.invoke(app, ["can", "inspect", "c.can"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = buffer.getvalue()
        assert "TITLE" in out and "31m" in out, out
        assert _c1_found(out) == [], repr(out)


class TestTheConfigLoaderPrintsNoC1:
    def test_an_unknown_key_carrying_c1_is_named_without_it(self, tmp_path, monkeypatch) -> None:
        from soup_cli.config import loader

        buffer = StringIO()
        monkeypatch.setattr(loader, "console", _plain_console(buffer))
        path = tmp_path / "soup.yaml"
        path.write_text(
            "base: test-model\n"
            "data:\n"
            "  train: ./data.jsonl\n"
            "training:\n"
            f'  "{_yaml_escape(0x9B)}2Jquantizaton": 4bit\n',
            encoding="utf-8",
        )
        with pytest.raises(SystemExit):
            loader.load_config(path)
        out = buffer.getvalue()
        assert "2Jquantizaton" in out, out
        assert _c1_found(out) == [], repr(out)


class TestManifestAuthorRefusesControlCharacters:
    @staticmethod
    def _manifest(author: str):
        from soup_cli.cans.schema import Manifest

        return Manifest(
            can_format_version=1,
            name="x",
            author=author,
            created_at="2026-04-20",
            base_hash="abc",
        )

    @pytest.mark.parametrize(
        "code",
        [0x00, 0x09, 0x0A, 0x0D, 0x1B, 0x7F, 0x80, 0x85, 0x9B, 0x9F],
        ids=["nul", "tab", "lf", "cr", "esc", "del", "0x80", "nel", "csi", "0x9f"],
    )
    def test_a_control_character_is_refused_naming_the_field(self, code: int) -> None:
        with pytest.raises(ValueError, match="author") as excinfo:
            self._manifest(f"alice{chr(code)}x")
        assert "control" in str(excinfo.value)

    @pytest.mark.parametrize(
        "author",
        ["alice", "J" + chr(0xF6) + "rg", "[team] alice", "a" + chr(0xA0) + "b"],
        ids=["ascii", "latin1-letter", "brackets", "nbsp"],
    )
    def test_printable_authors_are_still_accepted(self, author: str) -> None:
        assert self._manifest(author).author == author

    def test_can_inspect_names_the_refused_field(self, tmp_path, monkeypatch) -> None:
        from soup_cli.cli import app
        from soup_cli.commands import can as can_cmd

        monkeypatch.chdir(tmp_path)
        _write_can(
            "c.can",
            "can_format_version: 1\n"
            'name: "n"\n'
            f'author: "a{_yaml_escape(0x9B)}2K"\n'
            'created_at: "2026-01-01"\n'
            'base_hash: "h"\n',
        )
        buffer = StringIO()
        monkeypatch.setattr(can_cmd, "console", _plain_console(buffer))
        result = runner.invoke(app, ["can", "inspect", "c.can"])
        assert result.exit_code == 1, (result.output, repr(result.exception))
        out = buffer.getvalue()
        assert "Cannot inspect can" in out and "author" in out, out
        assert _c1_found(out) == [], repr(out)
