"""A can cannot hold two members that resolve to the same name.

A can holds one entry per name, so the member a command reads is the file it extracts and
uses. Every reader of a can (inspect, verify, run, fork, publish, extract) refuses an archive
that holds two members for one name, a link, a device or a member below a file, with one
message and the exit status the ``soup can`` commands already use for a refused can (1).

The cases below are the spellings that make two member names the same entry on at least one
platform; the ones that must keep working are listed too.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import yaml

from soup_cli.cans import run as can_run
from soup_cli.cans import unpack as unpack_module
from soup_cli.cans.pack import fork_can
from soup_cli.cans.publish import publish_can
from soup_cli.cans.run import run_can
from soup_cli.cans.unpack import extract_can, inspect_can, read_config
from soup_cli.cans.verify import verify_can
from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

_SAME_NAME = "resolve to the same name"

_MANIFEST = yaml.safe_dump(
    {
        "can_format_version": 1,
        "name": "demo-recipe",
        "author": "tester",
        "created_at": "2026-01-01",
        "base_hash": "0123456789abcdef",
    }
).encode("utf-8")
_CONFIG = b"base: test\ntask: sft\n"
_OTHER = b"base: test\ntask: sft\ntraining:\n  epochs: 2\n"


def _write_can(path: Path, members: list[tuple]) -> Path:
    """``members``: ``(name, payload)`` for a file, or ``(name, kind, target)`` for the rest."""
    with tarfile.open(path, "w:gz") as archive:
        for entry in members:
            name = entry[0]
            info = tarfile.TarInfo(name)
            if len(entry) == 2:
                payload = entry[1]
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
                continue
            kind, target = entry[1], entry[2]
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = target
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = target
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
            archive.addfile(info)
    return path


def _can_with(tmp_path: Path, first: str, second: str, name: str = "dup.can") -> Path:
    return _write_can(
        tmp_path / name,
        [("manifest.yaml", _MANIFEST), (first, _CONFIG), (second, _OTHER)],
    )


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


# Pairs of names that are the same entry on at least one platform's extraction.
_COLLIDING = [
    ("config.yaml", "config.yaml"),
    ("config.yaml", "./config.yaml"),
    ("config.yaml", "././config.yaml"),
    ("config.yaml", ".//config.yaml"),
    ("config.yaml", "sub/../config.yaml"),
    ("config.yaml", "/config.yaml"),
    ("config.yaml", ".\\config.yaml"),
    ("config.yaml", "Config.yaml"),
    ("config.yaml", "CONFIG.YAML"),
    ("config.yaml", "config.yaml."),
    ("config.yaml", "config.yaml "),
    ("config.yaml", "config.yaml:stream"),
    ("config.yaml", "config.yaml::$DATA"),
    ("sub/data.txt", "sub\\data.txt"),
    ("sub/data.txt", "SUB/Data.TXT"),
    ("sub/data.txt", "sub//data.txt"),
    ("sub/data.txt", "./sub/./data.txt"),
    ("café/data.txt", "café/data.txt"),
]


class TestTwoMembersWithTheSameName:
    @pytest.mark.parametrize(("first", "second"), _COLLIDING)
    @pytest.mark.parametrize("reverse", [False, True], ids=["in-order", "reversed"])
    def test_read_config_refuses(self, workdir, first, second, reverse):
        pair = (second, first) if reverse else (first, second)
        members = [("manifest.yaml", _MANIFEST)]
        if first != "config.yaml":
            # the colliding pair is some other file: the can still needs its own config
            members.append(("config.yaml", _CONFIG))
        members += [(pair[0], _CONFIG), (pair[1], _OTHER)]
        can = _write_can(workdir / "dup.can", members)

        with pytest.raises(ValueError, match=_SAME_NAME):
            read_config(str(can))

    def test_the_message_names_both_members(self, workdir):
        can = _can_with(workdir, "config.yaml", "./config.yaml")

        with pytest.raises(ValueError) as raised:
            read_config(str(can))

        message = str(raised.value)
        assert "'config.yaml'" in message
        assert "'./config.yaml'" in message
        assert "refusing" in message

    @pytest.mark.parametrize(
        "opener",
        [
            pytest.param(lambda can, work: inspect_can(str(can)), id="inspect_can"),
            pytest.param(lambda can, work: read_config(str(can)), id="read_config"),
            pytest.param(
                lambda can, work: extract_can(str(can), str(work / "out")), id="extract_can"
            ),
            pytest.param(
                lambda can, work: fork_can(
                    source=str(can), out_path=str(work / "forked.can"), modifications=[]
                ),
                id="fork_can",
            ),
            pytest.param(
                lambda can, work: publish_can(str(can), repo_id="me/some-can", token="t"),
                id="publish_can",
            ),
            pytest.param(lambda can, work: run_can(str(can), yes=True), id="run_can"),
        ],
    )
    def test_every_reader_of_a_can_refuses(self, workdir, monkeypatch, opener):
        monkeypatch.setattr(can_run, "_run_subprocess", lambda argv, **kwargs: 0)
        can = _can_with(workdir, "config.yaml", "./config.yaml")

        with pytest.raises(ValueError, match=_SAME_NAME):
            opener(can, workdir)

        assert not (workdir / "forked.can").exists()
        assert not (workdir / "out").exists()

    def test_verify_reports_it(self, workdir):
        can = _can_with(workdir, "config.yaml", "./config.yaml")

        report = verify_can(str(can))

        assert report.manifest_ok is False
        assert _SAME_NAME in report.message

    def test_the_manifest_member_is_covered_too(self, workdir):
        can = _write_can(
            workdir / "dup.can",
            [
                ("manifest.yaml", _MANIFEST),
                ("./manifest.yaml", _MANIFEST),
                ("config.yaml", _CONFIG),
            ],
        )

        with pytest.raises(ValueError, match=_SAME_NAME):
            inspect_can(str(can))

    def test_a_directory_entry_with_the_name_of_a_file(self, workdir):
        can = _write_can(
            workdir / "dup.can",
            [
                ("manifest.yaml", _MANIFEST),
                ("config.yaml", _CONFIG),
                ("config.yaml", "dir", None),
            ],
        )

        with pytest.raises(ValueError, match=_SAME_NAME):
            read_config(str(can))

    def test_a_directory_entry_with_a_trailing_separator_and_the_name_of_a_file(self, workdir):
        can = _write_can(
            workdir / "dup.can",
            [
                ("manifest.yaml", _MANIFEST),
                ("config.yaml", _CONFIG),
                ("config.yaml/", "dir", None),
            ],
        )

        with pytest.raises(ValueError, match=_SAME_NAME):
            read_config(str(can))

    def test_a_member_under_a_file_member(self, workdir):
        can = _write_can(
            workdir / "dup.can",
            [
                ("manifest.yaml", _MANIFEST),
                ("config.yaml", _CONFIG),
                ("config.yaml/inner.txt", b"x"),
            ],
        )

        with pytest.raises(ValueError, match="lies under the file member"):
            read_config(str(can))

    def test_a_name_that_leaves_the_can(self, workdir):
        can = _write_can(
            workdir / "dup.can",
            [("manifest.yaml", _MANIFEST), ("config.yaml", _CONFIG), ("../escape.txt", b"x")],
        )

        with pytest.raises(ValueError, match="leaves the can"):
            read_config(str(can))


class TestMemberCount:
    def test_too_many_members_is_refused_by_every_reader(self, workdir, monkeypatch):
        monkeypatch.setattr(unpack_module, "MAX_EXTRACT_MEMBERS", 5)
        can = _write_can(
            workdir / "many.can",
            [
                ("manifest.yaml", _MANIFEST),
                ("config.yaml", _CONFIG),
                *[(f"data/f{index}.txt", b"x") for index in range(4)],
            ],
        )

        for opener in (
            lambda: inspect_can(str(can)),
            lambda: read_config(str(can)),
            lambda: extract_can(str(can), str(workdir / "out")),
        ):
            with pytest.raises(ValueError, match="too many members"):
                opener()
        assert not (workdir / "out").exists()


class TestLinksAndSpecialMembers:
    @pytest.mark.parametrize("kind", ["symlink", "hardlink"])
    def test_a_link_member_is_refused_by_every_reader(self, workdir, kind):
        can = _write_can(
            workdir / "link.can",
            [
                ("manifest.yaml", _MANIFEST),
                ("config.yaml", _CONFIG),
                ("alias.yaml", kind, "config.yaml"),
            ],
        )

        for opener in (
            lambda: inspect_can(str(can)),
            lambda: read_config(str(can)),
            lambda: extract_can(str(can), str(workdir / "out")),
        ):
            with pytest.raises(ValueError, match="links are not allowed"):
                opener()
        assert not (workdir / "out").exists()

    def test_the_config_member_itself_a_link_to_another_member(self, workdir):
        can = _write_can(
            workdir / "link.can",
            [
                ("manifest.yaml", _MANIFEST),
                ("other.yaml", _OTHER),
                ("config.yaml", "symlink", "other.yaml"),
            ],
        )

        with pytest.raises(ValueError, match="links are not allowed"):
            read_config(str(can))

    def test_a_link_with_a_second_spelling_of_a_real_member(self, workdir):
        can = _write_can(
            workdir / "link.can",
            [
                ("manifest.yaml", _MANIFEST),
                ("config.yaml", _CONFIG),
                ("./config.yaml", "symlink", "other.yaml"),
                ("other.yaml", _OTHER),
            ],
        )

        with pytest.raises(ValueError, match="links are not allowed|resolve to the same name"):
            read_config(str(can))

    def test_a_device_or_fifo_member_is_refused(self, workdir):
        can = _write_can(
            workdir / "fifo.can",
            [("manifest.yaml", _MANIFEST), ("config.yaml", _CONFIG), ("pipe", "fifo", None)],
        )

        with pytest.raises(ValueError, match="neither a regular file nor a directory"):
            read_config(str(can))


class TestWhatStillWorks:
    def test_a_normal_can(self, workdir):
        can = _write_can(
            workdir / "ok.can",
            [
                ("manifest.yaml", _MANIFEST),
                ("config.yaml", _CONFIG),
                ("data_ref.yaml", b"kind: local\n"),
                ("attestations", "dir", None),
                ("attestations/one.json", b"{}"),
            ],
        )

        assert inspect_can(str(can)).name == "demo-recipe"
        assert read_config(str(can))["task"] == "sft"
        extracted = extract_can(str(can), str(workdir / "out"))
        assert (extracted / "config.yaml").read_bytes() == _CONFIG
        assert verify_can(str(can)).manifest_ok is True

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("config.yaml", "config2.yaml"),
            ("config.yaml", "config.yml"),
            ("config.yaml", "sub/config.yaml"),
            ("sub/data.txt", "other/data.txt"),
            ("data.txt", "data.txt.bak"),
            ("config.yaml", ".config.yaml"),
        ],
    )
    def test_names_that_really_differ(self, workdir, first, second):
        can = _write_can(
            workdir / "ok.can",
            [
                ("manifest.yaml", _MANIFEST),
                ("config.yaml", _CONFIG),
                *[(name, b"x") for name in (first, second) if name != "config.yaml"],
            ],
        )

        assert read_config(str(can))["task"] == "sft"
        extract_can(str(can), str(workdir / "out"))

    def test_the_root_directory_entry_is_not_a_collision(self, workdir):
        can = _write_can(
            workdir / "ok.can",
            [
                (".", "dir", None),
                ("./", "dir", None),
                ("manifest.yaml", _MANIFEST),
                ("config.yaml", _CONFIG),
            ],
        )

        assert read_config(str(can))["task"] == "sft"

    def test_a_directory_and_a_file_inside_it(self, workdir):
        can = _write_can(
            workdir / "ok.can",
            [
                ("manifest.yaml", _MANIFEST),
                ("config.yaml", _CONFIG),
                ("models", "dir", None),
                ("models/a.gguf", b"x"),
            ],
        )

        extract_can(str(can), str(workdir / "out"))
        assert (workdir / "out" / "models" / "a.gguf").read_bytes() == b"x"


def _plain(output: str) -> str:
    return " ".join(strip_ansi(output).split())


class TestCommands:
    @pytest.mark.parametrize(
        "argv",
        [
            ["inspect", "dup.can"],
            ["verify", "dup.can"],
            ["run", "dup.can", "--yes"],
            ["fork", "dup.can", "--out", "forked.can"],
            ["publish", "dup.can", "--hf-hub", "me/some-can"],
        ],
        ids=["inspect", "verify", "run", "fork", "publish"],
    )
    def test_every_command_exits_1_and_says_why(self, workdir, monkeypatch, argv):
        monkeypatch.setattr(can_run, "_run_subprocess", lambda argv, **kwargs: 0)
        _can_with(workdir, "config.yaml", "./config.yaml")

        result = make_cli_runner().invoke(app, ["can", *argv])

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert _SAME_NAME in _plain(result.output)
        assert not (workdir / "forked.can").exists()


class TestTheRedactedValueCheckReadsTheFileThatIsUsed:
    """Residual aliasing: whatever the archive layout, ``soup can run`` also checks the
    config file that extraction wrote, which is the one ``soup train`` reads."""

    def test_a_redacted_value_in_the_extracted_file_is_refused(self, workdir, monkeypatch):
        calls: list = []
        monkeypatch.setattr(can_run, "_run_subprocess", lambda argv, **kwargs: calls.append(argv))
        can = _write_can(
            workdir / "ok.can", [("manifest.yaml", _MANIFEST), ("config.yaml", _CONFIG)]
        )
        real_extract = can_run.extract_can

        def extract_with_other_config(path, dest):
            result = real_extract(path, dest)
            (Path(dest) / "config.yaml").write_text(
                "base: test\ntask: sft\ntraining:\n  grpo_judge: <redacted>\n", encoding="utf-8"
            )
            return result

        monkeypatch.setattr(can_run, "extract_can", extract_with_other_config)

        with pytest.raises(ValueError, match="redacted value"):
            run_can(str(can), yes=True)

        assert calls == []

    def test_a_clean_extracted_file_still_runs(self, workdir, monkeypatch):
        calls: list = []
        monkeypatch.setattr(
            can_run, "_run_subprocess", lambda argv, **kwargs: calls.append(argv) or 0
        )
        can = _write_can(
            workdir / "ok.can", [("manifest.yaml", _MANIFEST), ("config.yaml", _CONFIG)]
        )

        run_can(str(can), yes=True)

        assert len(calls) == 1
