"""``soup can run`` keeps a can's training output under the working directory.

A can's ``config.yaml`` is written by whoever packed the can. ``soup can run`` hands it to
``soup train`` unchanged, and ``soup train`` resolves ``output`` against the directory the
command was started in, so an ``output`` that resolves elsewhere made the run write its
checkpoints, logs and final model (and rotate ``checkpoint-*`` directories already there)
anywhere the user can write.

The decision (``.claude/plan.md``): a can whose ``output`` resolves outside the working
directory is refused with a named message; the confirmation panel shows ``output`` and
``data.train``; ``data.train`` is NOT refused, and a local path outside the working
directory is shown with a warning line.
"""

from __future__ import annotations

import io
import os
import re
import tarfile
import tempfile
from pathlib import Path

import pytest
import yaml

from soup_cli.cans import run as can_run
from soup_cli.cans.run import CanRunResult, check_can_output, describe_can_targets, run_can
from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

_REFUSAL = "resolves outside the working directory"


def _members(config_text: str | bytes) -> list[tuple[str, bytes]]:
    manifest = {
        "can_format_version": 1,
        "name": "demo-recipe",
        "author": "tester",
        "created_at": "2026-01-01",
        "base_hash": "0123456789abcdef",
    }
    config = config_text if isinstance(config_text, bytes) else config_text.encode("utf-8")
    return [("manifest.yaml", yaml.safe_dump(manifest).encode("utf-8")), ("config.yaml", config)]


def _write_can(path: Path, members: list[tuple[str, bytes]]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _config(**overrides) -> str:
    config: dict = {"base": "test", "task": "sft", "data": {"train": "data.jsonl"}}
    config.update(overrides)
    return yaml.safe_dump(config)


class _Env:
    def __init__(self, tmp_path: Path, calls: list, systmp: Path):
        self.root = tmp_path
        self.work = tmp_path / "work"
        self.outside = tmp_path / "elsewhere"
        self.calls = calls
        self.systmp = systmp

    def can(self, config_text: str | bytes, name: str = "test.can") -> str:
        _write_can(self.work / name, _members(config_text))
        return name

    def leftovers(self) -> list[Path]:
        return list(self.systmp.glob("soup-can-run-*"))


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Env:
    work = tmp_path / "work"
    work.mkdir()
    systmp = tmp_path / "systmp"
    systmp.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(tempfile, "tempdir", str(systmp))
    calls: list = []

    def fake_train(argv, **kwargs):
        calls.append(argv)
        return 0

    monkeypatch.setattr(can_run, "_run_subprocess", fake_train)
    return _Env(tmp_path, calls, systmp)


def _refused(env: _Env, output: object, **run_kwargs) -> str:
    """Run a can with this ``output``; assert it is refused with nothing left behind."""
    can = env.can(_config(output=output))
    with pytest.raises(ValueError) as excinfo:
        run_can(can, yes=True, **run_kwargs)
    assert env.calls == []
    assert env.leftovers() == []
    assert not env.outside.exists()
    return str(excinfo.value)


class TestOutputOutsideTheWorkingDirectoryIsRefused:
    def test_absolute_path_outside(self, env):
        message = _refused(env, str(env.outside))

        assert _REFUSAL in message

    @pytest.mark.parametrize(
        "relative",
        [
            "../elsewhere",
            "./../elsewhere",
            "a/../../elsewhere",
            "a/b/../../../elsewhere/x",
            "..",
            "../",
            "../work/../elsewhere",
        ],
    )
    def test_dot_dot_spellings(self, env, relative):
        message = _refused(env, relative)

        assert _REFUSAL in message

    def test_parent_of_the_working_directory_by_absolute_path(self, env):
        message = _refused(env, str(env.root))

        assert _REFUSAL in message

    def test_a_sibling_whose_name_starts_like_the_working_directory(self, env):
        # "work-extra" shares the textual prefix "work" with the working directory; the
        # comparison is by path component, not by string prefix.
        message = _refused(env, str(env.root / "work-extra"))

        assert _REFUSAL in message

    @pytest.mark.requires_symlink
    def test_symlink_inside_the_working_directory_that_points_outside(self, env):
        env.outside.mkdir()
        os.symlink(str(env.outside), str(env.work / "link"), target_is_directory=True)
        can = env.can(_config(output="link/run"))

        with pytest.raises(ValueError) as excinfo:
            run_can(can, yes=True)

        assert _REFUSAL in str(excinfo.value)
        assert env.calls == []
        assert list(env.outside.iterdir()) == []

    @pytest.mark.requires_symlink
    def test_symlink_spellings_that_resolve_outside(self, env):
        env.outside.mkdir()
        os.symlink(str(env.outside), str(env.work / "link"), target_is_directory=True)

        for output in ("link", "./link/", "link/a/b"):
            can = env.can(_config(output=output))
            with pytest.raises(ValueError, match=_REFUSAL):
                run_can(can, yes=True)
        assert env.calls == []

    @pytest.mark.skipif(os.name != "nt", reason="drive letters are a Windows path form")
    def test_drive_relative_and_drive_absolute_on_another_drive(self, env):
        free = next(
            letter for letter in "ZYXWVUTSRQ" if not os.path.exists(f"{letter}:\\")
        )

        for output in (f"{free}:out", f"{free}:\\out", f"{free}:/out", f"\\\\?\\{free}:\\out"):
            _refused(env, output)

    @pytest.mark.parametrize("output", ["//server/share/out", "\\\\server\\share\\out"])
    def test_unc_share_is_outside(self, env, monkeypatch, output):
        # A real UNC lookup would touch the network, so the resolver is replaced for this
        # spelling only; what is under test is that a path on another mount (the
        # os.path.commonpath ValueError) counts as outside.
        real_realpath = os.path.realpath

        def fake_realpath(path, *args, **kwargs):
            if str(path) == output:
                return "\\\\server\\share\\out"
            return real_realpath(path, *args, **kwargs)

        monkeypatch.setattr(os.path, "realpath", fake_realpath)

        with pytest.raises(ValueError, match=_REFUSAL):
            check_can_output({"output": output})

    @pytest.mark.parametrize("output", ["out\x00x", "\x00", "../elsewhere\x00"])
    def test_nul_byte_is_refused_on_every_platform(self, env, output):
        # POSIX path resolution raises on a NUL, Windows resolution keeps it and would
        # call the path inside; the refusal does not depend on that.
        with pytest.raises(ValueError, match="contains a NUL byte"):
            check_can_output({"output": output})


class TestValuesThatAreNotATextPath:
    @pytest.mark.parametrize("value", ["5", "[a, b]", "{a: b}", "null", "true", "1.5"])
    def test_non_text_output_is_refused(self, env, value):
        can = env.can(f"base: test\ntask: sft\noutput: {value}\n")

        with pytest.raises(ValueError, match="not a text path"):
            run_can(can, yes=True)

        assert env.calls == []
        assert env.leftovers() == []

    def test_binary_tagged_output_is_refused_not_decoded_by_the_loader(self, env):
        # ``!!binary`` loads as bytes, which the config model would accept as text.
        can = env.can("base: test\ntask: sft\noutput: !!binary Li4vZWxzZXdoZXJl\n")

        with pytest.raises(ValueError, match="not a text path"):
            run_can(can, yes=True)

        assert env.calls == []


class TestNothingHappensBeforeTheRefusal:
    def test_nothing_is_extracted_and_no_confirmation_is_asked(self, env, monkeypatch):
        extracted: list = []
        monkeypatch.setattr(
            can_run, "extract_can", lambda *args, **kwargs: extracted.append(args)
        )
        asked: list = []
        can = env.can(_config(output=str(env.outside)))

        with pytest.raises(ValueError, match=_REFUSAL):
            run_can(can, yes=False, confirm_callback=lambda manifest: asked.append(1) or True)

        assert extracted == []
        assert asked == []
        assert env.leftovers() == []

    def test_user_chosen_extract_dir_is_not_created(self, env):
        can = env.can(_config(output="../elsewhere"))

        with pytest.raises(ValueError, match=_REFUSAL):
            run_can(can, yes=True, extract_dir="extracted")

        assert not (env.work / "extracted").exists()


class TestTheExtractedFileIsTheOneThatIsChecked:
    def _can_with_two_configs(self, env, first: str, second: str) -> str:
        members = _members(first)
        members.append(("./config.yaml", second.encode("utf-8")))
        return _write_can(env.work / "twice.can", members).name

    def test_a_config_that_differs_from_the_one_read_first_is_refused(self, env):
        can = self._can_with_two_configs(env, _config(), _config(output=str(env.outside)))

        with pytest.raises(ValueError, match=_REFUSAL):
            run_can(can, yes=True)

        assert env.calls == []
        assert env.leftovers() == []
        assert not env.outside.exists()

    def test_a_user_chosen_extract_dir_is_kept_but_nothing_is_trained(self, env):
        can = self._can_with_two_configs(env, _config(), _config(output="../elsewhere"))

        with pytest.raises(ValueError, match=_REFUSAL):
            run_can(can, yes=True, extract_dir="extracted")

        assert env.calls == []
        assert (env.work / "extracted").is_dir()

    def test_train_is_handed_the_file_that_was_checked(self, env):
        can = self._can_with_two_configs(env, _config(), _config(output="inside/run"))

        result = run_can(can, yes=True)

        assert isinstance(result, CanRunResult)
        (argv,) = env.calls
        config_path = Path(argv[argv.index("--config") + 1])
        assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["output"] == "inside/run"

    def test_a_config_that_is_not_a_mapping_after_extraction_is_refused(self, env):
        members = _members(_config())
        members.append(("./config.yaml", b"- just\n- a list\n"))
        can = _write_can(env.work / "listy.can", members).name

        with pytest.raises(ValueError, match="must deserialise to a mapping"):
            run_can(can, yes=True)

        assert env.calls == []
        assert env.leftovers() == []

    def test_an_oversized_extracted_config_is_refused_by_the_capped_reader(self, env):
        from soup_cli.utils.yaml_limits import MAX_CONFIG_BYTES

        members = _members(_config())
        members.append(("./config.yaml", b"#" + b"x" * MAX_CONFIG_BYTES + b"\n"))
        can = _write_can(env.work / "big.can", members).name

        with pytest.raises(ValueError, match="larger than"):
            run_can(can, yes=True)

        assert env.calls == []
        assert env.leftovers() == []


class TestWhatStillRuns:
    @pytest.mark.parametrize(
        "output",
        [
            "./output",
            "output",
            "models/run-1",
            "out/sub/../x",
            ".",
            "./",
            "sub dir/out",
            "résultats/été",
            "..foo",
            "a..b/c",
        ],
    )
    def test_output_inside_the_working_directory(self, env, output):
        can = env.can(_config(output=output))

        result = run_can(can, yes=True)

        assert result.train_returncode == 0
        assert len(env.calls) == 1
        assert env.leftovers() == [result.extract_dir]

    def test_output_that_is_absolute_but_inside(self, env):
        can = env.can(_config(output=str(env.work / "abs-out")))

        run_can(can, yes=True)

        assert len(env.calls) == 1

    def test_missing_output_uses_the_default_inside_the_working_directory(self, env):
        can = env.can("base: test\ntask: sft\n")

        run_can(can, yes=True)

        assert len(env.calls) == 1

    @pytest.mark.requires_symlink
    def test_symlink_that_points_to_another_place_inside(self, env):
        (env.work / "real").mkdir()
        os.symlink(str(env.work / "real"), str(env.work / "alias"), target_is_directory=True)
        can = env.can(_config(output="alias/run"))

        run_can(can, yes=True)

        assert len(env.calls) == 1

    @pytest.mark.parametrize(
        "train",
        ["/srv/shared/data.jsonl", "../data.jsonl", "https://example.com/data.jsonl", "org/name"],
    )
    def test_data_train_outside_is_not_refused(self, env, train):
        can = env.can(yaml.safe_dump({"base": "test", "task": "sft", "data": {"train": train}}))

        run_can(can, yes=True)

        assert len(env.calls) == 1


class TestTheRefusalMessage:
    def test_names_the_value_the_directory_and_what_to_do(self, env):
        message = _refused(env, "../elsewhere")

        assert "output to '../elsewhere'" in message
        assert os.getcwd() in message
        assert "set output to a path under your working directory" in message

    def test_control_characters_in_the_value_are_escaped(self, env):
        message = _refused(env, "../x\x1b[2J\u009b")

        assert "\x1b" not in message
        assert "\u009b" not in message
        assert "\\x1b" in message


class TestDescribeTargets:
    def test_output_and_single_data_entry(self, env):
        targets = describe_can_targets(
            {"output": "./out", "data": {"train": "data.jsonl"}}
        )

        assert targets.output == "./out"
        assert targets.output_problem is None
        assert targets.data_train == (("data.jsonl", False),)

    def test_default_output_is_shown_when_the_can_sets_none(self, env):
        assert describe_can_targets({"base": "x"}).output == "./output"

    def test_output_outside_carries_the_refusal_message(self, env):
        targets = describe_can_targets({"output": str(env.outside)})

        assert targets.output == str(env.outside)
        assert _REFUSAL in targets.output_problem

    def test_local_data_path_outside_is_flagged_and_other_entries_are_not(self, env):
        targets = describe_can_targets(
            {
                "data": {
                    "train": [
                        str(env.outside / "a.jsonl"),
                        "../b.jsonl",
                        "inside/c.jsonl",
                        "s3://bucket/d.jsonl",
                        "org/hub-dataset",
                    ]
                }
            }
        )

        assert dict(targets.data_train) == {
            str(env.outside / "a.jsonl"): True,
            "../b.jsonl": True,
            "inside/c.jsonl": False,
            "s3://bucket/d.jsonl": False,
            "org/hub-dataset": False,
        }

    def test_long_list_is_capped_with_a_count(self, env):
        entries = [f"d{index}.jsonl" for index in range(25)]

        targets = describe_can_targets({"data": {"train": entries}})

        assert len(targets.data_train) == 20
        assert targets.data_train_more == 5

    @pytest.mark.parametrize("data", [None, "text", [], {"train": None}, {"train": 5}, {}])
    def test_odd_data_sections_give_no_entries(self, env, data):
        assert describe_can_targets({"data": data}).data_train == ()

    def test_non_text_output_is_shown_by_repr(self, env):
        targets = describe_can_targets({"output": b"../x"})

        assert targets.output == "b'../x'"
        assert "not a text path" in targets.output_problem


def _plain(output: str) -> str:
    return " ".join(strip_ansi(output).split())


def _run_cli(*args: str):
    return make_cli_runner().invoke(app, ["can", "run", *args])


class TestCommand:
    def test_panel_shows_output_and_data_train(self, env):
        can = env.can(_config(output="models/run-1"))

        result = _run_cli(can)

        assert result.exit_code == 1, (result.output, repr(result.exception))
        text = _plain(result.output)
        assert "Output: models/run-1" in text
        assert "Training data: data.jsonl" in text
        assert "warning" not in text
        assert "Pass --yes to confirm." in text

    def test_panel_marks_an_output_outside_as_refused(self, env):
        can = env.can(_config(output="../elsewhere"))

        result = _run_cli(can)

        assert result.exit_code == 1, (result.output, repr(result.exception))
        text = _plain(result.output)
        assert "Output: ../elsewhere" in text
        assert "warning: outside the working directory - soup can run will refuse" in text

    def test_panel_marks_a_data_path_outside_with_a_warning_and_lists_every_entry(self, env):
        data = {"train": ["../a.jsonl", "inside.jsonl", "s3://bucket/c.jsonl"]}
        can = env.can(yaml.safe_dump({"base": "test", "task": "sft", "data": data}))

        result = _run_cli(can)

        text = _plain(result.output)
        assert "Training data: ../a.jsonl" in text
        assert "Training data: inside.jsonl" in text
        assert "Training data: s3://bucket/c.jsonl" in text
        assert text.count("this path is outside the working directory") == 1
        assert "will refuse" not in text

    def test_panel_never_prints_raw_control_characters_from_the_can(self, env):
        can = env.can(_config(output="out\x1b[2Jx", data={"train": "d\x1b]0;t\x07.jsonl"}))

        result = _run_cli(can)

        assert "\x1b[2J" not in result.output
        assert "\x1b]0;" not in result.output
        assert "\x07" not in result.output

    def test_with_yes_an_output_outside_is_refused_with_the_message_and_exit_1(self, env):
        can = env.can(_config(output="../elsewhere"))

        result = _run_cli(can, "--yes")

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert _REFUSAL in _plain(result.output)
        assert "this can's config sets output to '../elsewhere'" in _plain(result.output)
        assert env.calls == []
        assert env.leftovers() == []

    def test_with_yes_an_output_inside_still_runs(self, env):
        can = env.can(_config(output="models/run-1"))

        result = _run_cli(can, "--yes")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(env.calls) == 1
        assert re.search(r"Can run complete", strip_ansi(result.output))
