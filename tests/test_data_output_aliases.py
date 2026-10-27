"""#2226: a ``soup data`` command must not write its output over one of its inputs.

``lint``, ``doctor`` and ``topics`` replaced the dataset with their JSON report,
``from-traces`` emptied the trace file, ``canary insert`` left only the
canary-carrying copy, and ``sample``, ``dedup``, ``filter``, ``convert``,
``merge``, ``active-sample``, ``augment`` and ``evolve`` rewrote the input with
their result. Every one of them exited 0, and none of those writes can be undone.

Each command now calls ``ensure_distinct_paths`` before it reads anything, so an
output that names an input under any spelling (the same string, ``./x``,
``sub/../x``, the absolute path, a case variant on a case-insensitive file
system, a hard link, a symlink) or a default output name the input already
carries is refused with a message that names both paths and says what to change.
The input bytes are compared before and after; a distinct output still works.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.exit_codes import EXIT_USAGE_ERROR
from soup_cli.utils.paths import quote_path
from tests.conftest import strip_ansi

runner = CliRunner()

_ALPACA = tuple(
    {
        "instruction": f"Explain topic number {idx} in plain words",
        "input": "",
        "output": f"Topic {idx} is explained here with a few distinct words {idx * 7}",
    }
    for idx in range(4)
)
_DPO = (
    {"prompt": "Name a colour.", "chosen": "Blue is a colour.", "rejected": "Seven is a number."},
    {"prompt": "Name a fruit.", "chosen": "A pear is a fruit.", "rejected": "A brick is a block."},
)
_CHAT = tuple(
    {
        "messages": [
            {"role": "user", "content": f"hi number {idx}"},
            {"role": "assistant", "content": f"hello number {idx}"},
        ]
    }
    for idx in range(3)
)
_TRACES = tuple({"id": f"t{idx}", "note": "no choices here"} for idx in range(2))
_SCORED = tuple(
    {"id": f"r{idx}", "prompt": f"q{idx}", "output": f"a{idx}", "rm_scores": [0.1 * idx, 0.9]}
    for idx in range(4)
)
_SEEDS = ({"prompt": "Write a short poem about the sea"},)


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    # Soup refuses paths outside the working directory, so every test runs in
    # its own temporary directory and never touches the repository.
    monkeypatch.chdir(tmp_path)


def _write_rows(name: str, rows) -> bytes:
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    with open(name, "wb") as fh:
        fh.write(payload)
    return payload


def _read_bytes(name: str) -> bytes:
    with open(name, "rb") as fh:
        return fh.read()


def _plain(output: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output."""
    return " ".join(strip_ansi(output).split())


def _squeezed(text: str) -> str:
    # Rich folds a long path that has no space in it across lines, so the
    # comparison of a message that quotes an absolute path ignores whitespace.
    return "".join(_plain(text).split())


def _refusal(out_field: str, out_path: str, in_field: str, in_path: str) -> str:
    return (
        f"{out_field} {quote_path(out_path)} is the same file as {in_field} "
        f"{quote_path(in_path)}; choose a different {out_field}"
    )


def _fail_if_called(target: str) -> Callable[..., None]:
    def _boom(*_args, **_kwargs):
        raise AssertionError(f"{target} ran before the output/input check")

    return _boom


def _patch_doctor(monkeypatch) -> None:
    from soup_cli.commands import data_doctor

    class _Report:
        overall = "OK"

        def to_dict(self) -> dict:
            return {"overall": "OK", "checks": []}

    monkeypatch.setattr(data_doctor, "_load_tokenizer", lambda *_a, **_k: object())
    monkeypatch.setattr(data_doctor.engine, "run_doctor", lambda *_a, **_k: _Report())
    monkeypatch.setattr(data_doctor, "_render_doctor_report", lambda _report: None)


def _patch_topics(monkeypatch) -> None:
    from soup_cli.commands import data_topics

    vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    monkeypatch.setattr(data_topics, "embed_texts", lambda *_a, **_k: vectors)


def _patch_augment(monkeypatch) -> None:
    from soup_cli.commands import data as data_cmd
    from soup_cli.data import augment

    monkeypatch.setattr(data_cmd, "_load_augment_provider", lambda *_a, **_k: object())
    monkeypatch.setitem(augment.STRATEGIES, "rephrase", lambda data, **_kwargs: [])


def _patch_evolve(monkeypatch) -> None:
    from soup_cli.utils import magpie

    monkeypatch.setattr(
        magpie,
        "make_magpie_generate_fn",
        lambda *_a, **_k: (lambda _prompt: "Describe the tides of a distant ocean in detail"),
    )


@dataclass(frozen=True)
class Case:
    command: str
    input_name: str
    rows: tuple
    argv: Callable[[str, str], list[str]]
    input_field: str
    exit_code: int
    #: Everything that reads the input (or loads a model for it); each must not run.
    readers: tuple[str, ...]
    patch_for_success: Callable | None = None
    extra_files: dict = field(default_factory=dict)


_CASES = (
    Case(
        "lint", "d.jsonl", _DPO,
        lambda inp, out: ["data", "lint", inp, "-o", out],
        "the input", EXIT_USAGE_ERROR,
        ("soup_cli.commands.data_doctor.load_raw_data",),
    ),
    Case(
        "doctor", "d.jsonl", _CHAT,
        lambda inp, out: ["data", "doctor", inp, "--model", "fake/model", "-o", out],
        "the input", 1,
        (
            "soup_cli.commands.data_doctor.load_raw_data",
            "soup_cli.commands.data_doctor._load_tokenizer",
        ),
        _patch_doctor,
    ),
    Case(
        "topics", "d.jsonl", _ALPACA,
        lambda inp, out: ["data", "topics", inp, "--clusters", "2", "-o", out],
        "the input", 1,
        (
            "soup_cli.commands.data_topics.load_raw_data",
            "soup_cli.commands.data_topics.embed_texts",
        ),
        _patch_topics,
    ),
    Case(
        "canary-insert", "d.jsonl", _ALPACA,
        lambda inp, out: ["data", "canary", "insert", inp, "-o", out, "--manifest", "m.json"],
        "the input", 1,
        (
            "soup_cli.commands.data_canary.load_raw_data",
            "soup_cli.commands.data_canary.write_manifest",
        ),
    ),
    Case(
        "from-traces", "d.jsonl", _TRACES,
        lambda inp, out: ["data", "from-traces", "--logs", inp, "--format", "openai", "-o", out],
        "--logs", 1,
        ("pathlib.Path.open",),
    ),
    Case(
        "sample", "d.jsonl", _ALPACA,
        lambda inp, out: ["data", "sample", inp, "-n", "2", "-o", out],
        "the input", 1,
        ("soup_cli.commands.data.load_raw_data",),
    ),
    Case(
        "dedup", "d.jsonl", _ALPACA,
        lambda inp, out: ["data", "dedup", inp, "-o", out],
        "the input", 1,
        ("soup_cli.commands.data.load_raw_data",),
    ),
    Case(
        "filter", "d.jsonl", _ALPACA,
        lambda inp, out: ["data", "filter", inp, "--coherence", "0.3", "-o", out],
        "the input", 1,
        ("soup_cli.commands.data.load_raw_data",),
    ),
    Case(
        "convert", "d.jsonl", _ALPACA,
        lambda inp, out: ["data", "convert", inp, "--to", "sharegpt", "-o", out],
        "the input", 1,
        ("soup_cli.commands.data.load_raw_data",),
    ),
    Case(
        "merge", "d.jsonl", _ALPACA,
        lambda inp, out: ["data", "merge", inp, "e.jsonl", "-o", out],
        "the input", 1,
        ("soup_cli.commands.data.load_raw_data",),
        extra_files={"e.jsonl": _ALPACA[:2]},
    ),
    Case(
        "active-sample", "d.jsonl", _SCORED,
        lambda inp, out: ["data", "active-sample", "-i", inp, "-o", out],
        "--input", 1,
        ("soup_cli.commands.active_sample.sample_uncertain_rows",),
    ),
    Case(
        "augment", "d.jsonl", _ALPACA,
        lambda inp, out: ["data", "augment", "-i", inp, "-o", out],
        "--input", 2,
        (
            "soup_cli.commands.data.load_raw_data",
            "soup_cli.commands.data._load_augment_provider",
        ),
        _patch_augment,
    ),
    Case(
        "evolve", "d.jsonl", _SEEDS,
        lambda inp, out: [
            "data", "evolve", "--input", inp, "--provider", "ollama", "--model", "m",
            "-o", out,
        ],
        "--input", 2,
        (
            "soup_cli.commands.data._evolve_load_prompts",
            "soup_cli.utils.magpie.make_magpie_generate_fn",
        ),
        _patch_evolve,
    ),
)
_BY_NAME = {case.command: case for case in _CASES}


def _setup(case: Case) -> dict[str, bytes]:
    """Write the case's input files; return each one's bytes."""
    before = {case.input_name: _write_rows(case.input_name, case.rows)}
    for name, rows in case.extra_files.items():
        before[name] = _write_rows(name, rows)
    return before


def _spell(kind: str, name: str) -> str:
    """Another way to name the input file ``name``."""
    if kind == "same":
        return name
    if kind == "dot":
        return f"./{name}"
    if kind == "parent":
        os.mkdir("sub")
        return f"sub/../{name}"
    if kind == "absolute":
        return os.path.abspath(name)
    if kind == "case":
        variant = name.upper()
        if not os.path.exists(variant):
            pytest.skip("this file system is case-sensitive: a case variant is another file")
        return variant
    if kind == "hardlink":
        os.link(name, "linked.jsonl")
        return "linked.jsonl"
    if kind == "symlink":
        os.symlink(name, "linked.jsonl")
        return "linked.jsonl"
    raise AssertionError(kind)


_SPELLINGS = (
    "same",
    "dot",
    "parent",
    "absolute",
    "case",
    pytest.param("hardlink", marks=pytest.mark.requires_hardlink),
    pytest.param("symlink", marks=pytest.mark.requires_symlink),
)


def _invoke_refused(monkeypatch, case: Case, argv: list[str]):
    for target in case.readers:
        monkeypatch.setattr(target, _fail_if_called(target))
    result = runner.invoke(app, argv)
    # A reader that ran raises AssertionError, which CliRunner turns into exit 1;
    # only a refusal through typer.Exit leaves a SystemExit here.
    assert isinstance(result.exception, SystemExit), (result.output, repr(result.exception))
    assert result.exit_code == case.exit_code, (result.output, repr(result.exception))
    return result


@pytest.mark.parametrize("spelling", _SPELLINGS)
@pytest.mark.parametrize("command", list(_BY_NAME))
def test_output_naming_the_input_is_refused_before_any_read(monkeypatch, command, spelling):
    case = _BY_NAME[command]
    before = _setup(case)
    out = _spell(spelling, case.input_name)
    result = _invoke_refused(monkeypatch, case, case.argv(case.input_name, out))
    assert _squeezed(_refusal("--output", out, case.input_field, case.input_name)) in _squeezed(
        result.output
    ), result.output
    for name, payload in before.items():
        assert _read_bytes(name) == payload, f"{name} was modified"


def test_merge_refuses_an_output_that_is_its_second_input(monkeypatch):
    case = _BY_NAME["merge"]
    before = _setup(case)
    result = _invoke_refused(monkeypatch, case, case.argv(case.input_name, "./e.jsonl"))
    assert _squeezed(_refusal("--output", "./e.jsonl", "the input", "e.jsonl")) in _squeezed(
        result.output
    ), result.output
    for name, payload in before.items():
        assert _read_bytes(name) == payload, f"{name} was modified"


@pytest.mark.parametrize(
    ("command", "default_name", "argv"),
    [
        ("merge", "merged.jsonl", ["data", "merge", "merged.jsonl", "e.jsonl"]),
        (
            "from-traces",
            "prefs.jsonl",
            ["data", "from-traces", "--logs", "prefs.jsonl", "--format", "openai"],
        ),
        (
            "active-sample",
            "active_samples.jsonl",
            ["data", "active-sample", "-i", "active_samples.jsonl"],
        ),
        ("augment", "augmented.jsonl", ["data", "augment", "-i", "augmented.jsonl"]),
    ],
)
def test_an_input_carrying_the_default_output_name_is_refused(
    monkeypatch, command, default_name, argv
):
    case = _BY_NAME[command]
    payload = _write_rows(default_name, case.rows)
    for name, rows in case.extra_files.items():
        _write_rows(name, rows)
    result = _invoke_refused(monkeypatch, case, argv)
    expected = _refusal("--output", default_name, case.input_field, default_name)
    assert _squeezed(expected) in _squeezed(result.output), result.output
    assert _read_bytes(default_name) == payload


def test_canary_insert_refuses_a_manifest_that_is_the_dataset(monkeypatch):
    case = _BY_NAME["canary-insert"]
    before = _setup(case)
    argv = ["data", "canary", "insert", "d.jsonl", "-o", "out.jsonl", "--manifest", "./d.jsonl"]
    result = _invoke_refused(monkeypatch, case, argv)
    expected = _refusal("--manifest", "./d.jsonl", "the input", "d.jsonl")
    assert _squeezed(expected) in _squeezed(result.output), result.output
    assert _read_bytes("d.jsonl") == before["d.jsonl"]
    assert not os.path.exists("out.jsonl")


def test_canary_insert_refuses_a_manifest_that_is_the_output(monkeypatch):
    # The manifest was written first and the dataset then replaced it, so the
    # only record of the inserted secrets was gone and the command exited 0.
    case = _BY_NAME["canary-insert"]
    before = _setup(case)
    argv = ["data", "canary", "insert", "d.jsonl", "-o", "o.jsonl", "--manifest", "o.jsonl"]
    result = _invoke_refused(monkeypatch, case, argv)
    expected = _refusal("--manifest", "o.jsonl", "--output", "o.jsonl")
    assert _squeezed(expected) in _squeezed(result.output), result.output
    assert _read_bytes("d.jsonl") == before["d.jsonl"]
    assert not os.path.exists("o.jsonl")


@pytest.mark.parametrize("command", list(_BY_NAME))
def test_a_distinct_output_still_works(monkeypatch, command):
    case = _BY_NAME[command]
    if command == "dedup":
        pytest.importorskip("datasketch")
    if case.patch_for_success is not None:
        case.patch_for_success(monkeypatch)
    before = _setup(case)
    result = runner.invoke(app, case.argv(case.input_name, "result.jsonl"))
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert os.path.isfile("result.jsonl"), _plain(result.output)
    for name, payload in before.items():
        assert _read_bytes(name) == payload, f"{name} was modified"
