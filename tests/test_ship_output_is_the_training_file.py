"""``soup ship`` refuses an output that is the training file a ``--config`` names (exit 3) (#3020).

``--output`` and ``--emit-evidence`` were compared with every input file the run was given, but
not with ``data.train`` of the parsed ``--config``, the file the provenance hashes; the atomic
writer would replace it when the run ended. ``data.train`` now joins the compared inputs: a path,
or each entry of the list form, named ``data.train of --config`` (``data.train[i] of --config``
for a list entry) with the path as the config writes it. A relative path is taken relative to the
working directory, as the hashing does. An entry that is not a file (a Hub dataset id, a path
that does not exist) is not compared.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.paths import quote_path

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

TRAIN = '{"prompt": "2+2", "response": "4"}\n'
TASKS = '{"prompt": "2+2", "expected": "4"}\n'
OFFLINE = ["--evidence", "ev.json"]
LIVE = ["--base", "org/model", "--adapter", "adapter", "--task-eval", "t.jsonl"]
OUTPUT_FLAGS = ("--output", "--emit-evidence")


def _plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def _config(train: str) -> str:
    return f"base: org/model\ntask: sft\ndata:\n  train: {train}\noutput: ./out\n"


def _refusal(out_flag: str, out_path: str, in_flag: str, in_path: str) -> str:
    return (
        f"{out_flag} {quote_path(out_path)} is the same file as {in_flag} {quote_path(in_path)}; "
        f"choose a different {out_flag}"
    )


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ev.json").write_text("{}", encoding="utf-8")
    (tmp_path / "t.jsonl").write_text(TASKS, encoding="utf-8")
    (tmp_path / "train.jsonl").write_text(TRAIN, encoding="utf-8")
    return tmp_path


def _ship(*args: str):
    return runner.invoke(app, ["ship", *args])


def _assert_refused(result, line: str, root: Path, name: str = "train.jsonl") -> None:
    out = _plain(result.output)
    assert result.exit_code == 3, (out, repr(result.exception))
    # Rich may fold an absolute path in the middle of a component, so the whole line is
    # compared with its whitespace removed.
    assert "".join(line.split()) in "".join(out.split()), out
    assert (root / name).read_text(encoding="utf-8") == TRAIN  # the training file is untouched


@pytest.mark.parametrize("out_flag", OUTPUT_FLAGS)
def test_the_training_file_named_relatively_is_refused(root, out_flag):
    (root / "soup.yaml").write_text(_config("train.jsonl"), encoding="utf-8")

    result = _ship(*OFFLINE, "--config", "soup.yaml", out_flag, "train.jsonl")

    _assert_refused(
        result, _refusal(out_flag, "train.jsonl", "data.train of --config", "train.jsonl"), root
    )


@pytest.mark.parametrize("out_flag", OUTPUT_FLAGS)
def test_the_training_file_named_absolutely_in_the_config_is_refused(root, out_flag):
    absolute = str(root / "train.jsonl")
    (root / "soup.yaml").write_text(_config(f"'{absolute}'"), encoding="utf-8")

    result = _ship(*OFFLINE, "--config", "soup.yaml", out_flag, "train.jsonl")

    _assert_refused(
        result, _refusal(out_flag, "train.jsonl", "data.train of --config", absolute), root
    )


@pytest.mark.parametrize("out_flag", OUTPUT_FLAGS)
def test_an_absolute_output_for_a_relative_training_file_is_refused(root, out_flag):
    (root / "soup.yaml").write_text(_config("./train.jsonl"), encoding="utf-8")
    absolute = str(root / "train.jsonl")

    result = _ship(*OFFLINE, "--config", "soup.yaml", out_flag, absolute)

    _assert_refused(
        result, _refusal(out_flag, absolute, "data.train of --config", "./train.jsonl"), root
    )


@pytest.mark.requires_symlink
@pytest.mark.parametrize("out_flag", OUTPUT_FLAGS)
def test_a_symlink_to_the_training_file_is_refused(root, out_flag):
    (root / "soup.yaml").write_text(_config("train.jsonl"), encoding="utf-8")
    os.symlink(root / "train.jsonl", root / "alias.jsonl")

    result = _ship(*OFFLINE, "--config", "soup.yaml", out_flag, "alias.jsonl")

    _assert_refused(
        result, _refusal(out_flag, "alias.jsonl", "data.train of --config", "train.jsonl"), root
    )


@pytest.mark.parametrize("out_flag", OUTPUT_FLAGS)
def test_a_hard_link_to_the_training_file_is_refused(root, out_flag):
    (root / "soup.yaml").write_text(_config("train.jsonl"), encoding="utf-8")
    try:
        os.link(root / "train.jsonl", root / "alias.jsonl")
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("this file system has no hard links")

    result = _ship(*OFFLINE, "--config", "soup.yaml", out_flag, "alias.jsonl")

    _assert_refused(
        result, _refusal(out_flag, "alias.jsonl", "data.train of --config", "train.jsonl"), root
    )


@pytest.mark.skipif(sys.platform != "win32", reason="a case-insensitive file system is Windows")
@pytest.mark.parametrize("out_flag", OUTPUT_FLAGS)
def test_the_training_file_in_another_case_is_refused_on_windows(root, out_flag):
    (root / "soup.yaml").write_text(_config("train.jsonl"), encoding="utf-8")

    result = _ship(*OFFLINE, "--config", "soup.yaml", out_flag, "TRAIN.JSONL")

    _assert_refused(
        result, _refusal(out_flag, "TRAIN.JSONL", "data.train of --config", "train.jsonl"), root
    )


@pytest.mark.parametrize("out_flag", OUTPUT_FLAGS)
def test_an_entry_of_a_training_list_is_refused_by_its_index(root, out_flag):
    (root / "second.jsonl").write_text(TRAIN, encoding="utf-8")
    (root / "soup.yaml").write_text(
        "base: org/model\ntask: sft\ndata:\n  train:\n    - train.jsonl\n    - second.jsonl\n"
        "  interleave: concat\noutput: ./out\n",
        encoding="utf-8",
    )

    result = _ship(*OFFLINE, "--config", "soup.yaml", out_flag, "second.jsonl")

    _assert_refused(
        result,
        _refusal(out_flag, "second.jsonl", "data.train[1] of --config", "second.jsonl"),
        root,
        "second.jsonl",
    )


def test_a_live_run_is_refused_before_any_model_loads(root, monkeypatch):
    import soup_cli.commands.ship as ship_module

    def _no_model(*_args, **_kwargs):
        raise AssertionError("a model was loaded before the output check")

    monkeypatch.setattr(ship_module, "_verdict_live", _no_model)
    (root / "soup.yaml").write_text(_config("train.jsonl"), encoding="utf-8")

    result = _ship(*LIVE, "--config", "soup.yaml", "--output", "train.jsonl")

    _assert_refused(
        result, _refusal("--output", "train.jsonl", "data.train of --config", "train.jsonl"), root
    )


def test_a_relative_training_file_is_taken_relative_to_the_working_directory(root):
    (root / "conf").mkdir()
    (root / "conf" / "soup.yaml").write_text(_config("train.jsonl"), encoding="utf-8")

    result = _ship(*OFFLINE, "--config", "conf/soup.yaml", "--output", "train.jsonl")

    _assert_refused(
        result, _refusal("--output", "train.jsonl", "data.train of --config", "train.jsonl"), root
    )


# --- controls: not refused by this check --------------------------------------------------------


def test_a_hub_dataset_id_is_not_a_file_and_is_not_compared(root):
    (root / "soup.yaml").write_text(_config("org/dataset"), encoding="utf-8")

    result = _ship(*OFFLINE, "--config", "soup.yaml", "--output", "org/dataset")

    assert "is the same file as" not in _plain(result.output)


def test_a_training_path_that_does_not_exist_is_not_compared(root):
    (root / "soup.yaml").write_text(_config("missing.jsonl"), encoding="utf-8")

    result = _ship(*OFFLINE, "--config", "soup.yaml", "--output", "missing.jsonl")

    assert "is the same file as" not in _plain(result.output)


def test_an_output_that_is_another_file_is_not_refused(root):
    (root / "soup.yaml").write_text(_config("train.jsonl"), encoding="utf-8")

    result = _ship(*OFFLINE, "--config", "soup.yaml", "--output", "verdict.json")

    assert "is the same file as" not in _plain(result.output)
    assert (root / "train.jsonl").read_text(encoding="utf-8") == TRAIN


def test_without_a_config_there_is_no_training_file_to_compare(root):
    result = _ship(*OFFLINE, "--output", "train.jsonl")

    assert "is the same file as" not in _plain(result.output)
