"""Issue #1830 — CLI message gaps.

Four small gaps in what gate-style commands print (exit codes were already
right; only the text was thin):

* ``soup expect`` with a missing data or suite file printed only the bare path.
* ``soup lock check`` / ``soup lock show`` on a corrupt lock file printed a raw
  JSON error without naming the file.
* The module docstring of ``commands/expect.py`` claimed exit code 3 on suite
  failure (it is 2; 3 is the usage error).
* ``soup draft measure`` / ``distill`` on a core-only install printed
  ``No module named 'transformers'`` with no ``[train]`` install hint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as soup_app
from soup_cli.commands import draft as draft_cmd
from soup_cli.commands import expect as expect_cmd
from soup_cli.utils.soup_lock import SoupLock, compute_lock_closure, read_lock, write_lock
from tests.conftest import strip_ansi

runner = CliRunner()

_BASE_SHA = "a" * 64
_DATASET_SHA = "b" * 64
_ENV_HASH = "c" * 64


def _write_valid_lock(tmp_path: Path) -> Path:
    lock = SoupLock(
        soup_version="0.76.0",
        base_model="test-model",
        base_model_sha=_BASE_SHA,
        dataset_sha=_DATASET_SHA,
        env_hash=_ENV_HASH,
        closure_sha=compute_lock_closure(
            base_model_sha=_BASE_SHA,
            dataset_sha=_DATASET_SHA,
            env_hash=_ENV_HASH,
        ),
        created_at="2026-10-15T00:00:00Z",
    )
    lock_file = tmp_path / "soup.lock"
    write_lock(lock, str(lock_file))
    return lock_file


# ---------- 1. soup expect: missing data / suite file ----------


def test_expect_missing_data_file_names_the_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "suite.yaml").write_text(
        "expectations:\n  - name: expect_no_pii\n", encoding="utf-8"
    )
    result = runner.invoke(
        soup_app, ["expect", "missing.jsonl", str(tmp_path / "suite.yaml")]
    )
    assert result.exit_code == 3
    out = strip_ansi(result.output)
    out = " ".join(out.split())
    assert "data file not found: 'missing.jsonl'" in out


def test_expect_missing_suite_file_names_the_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.jsonl").write_text(
        '{"messages": [{"role": "user", "content": "hi"}]}\n', encoding="utf-8"
    )
    result = runner.invoke(
        soup_app, ["expect", str(tmp_path / "data.jsonl"), "missing.yaml"]
    )
    assert result.exit_code == 3
    out = strip_ansi(result.output)
    assert "suite file not found: 'missing.yaml'" in out


def test_expect_data_file_is_a_directory_names_the_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "suite.yaml").write_text(
        "expectations:\n  - name: expect_no_pii\n", encoding="utf-8"
    )
    result = runner.invoke(soup_app, ["expect", str(tmp_path), str(tmp_path / "suite.yaml")])
    assert result.exit_code == 3
    out = " ".join(strip_ansi(result.output).split())
    assert "data file not found" in out


def test_expect_suite_file_is_a_directory_names_the_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.jsonl").write_text(
        '{"messages": [{"role": "user", "content": "hi"}]}}\n', encoding="utf-8"
    )
    result = runner.invoke(soup_app, ["expect", str(tmp_path / "data.jsonl"), str(tmp_path)])
    assert result.exit_code == 3
    out = " ".join(strip_ansi(result.output).split())
    assert "suite file not found" in out


# ---------- 2. soup lock check / show: corrupt lock file ----------


def test_lock_check_corrupt_lock_names_file_and_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lock_file = tmp_path / "soup.lock"
    lock_file.write_text("garbage: [", encoding="utf-8")
    result = runner.invoke(
        soup_app,
        [
            "lock", "check", str(lock_file),
            "--base-model", "test-model",
            "--base-sha", _BASE_SHA,
            "--dataset-sha", _DATASET_SHA,
            "--env-hash", _ENV_HASH,
        ],
    )
    assert result.exit_code == 3
    out = " ".join(strip_ansi(result.output).split())
    assert "soup.lock" in out
    assert "not valid JSON" in out


def test_lock_show_corrupt_lock_names_file_and_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lock_file = tmp_path / "soup.lock"
    lock_file.write_text("garbage: [", encoding="utf-8")
    result = runner.invoke(soup_app, ["lock", "show", str(lock_file)])
    assert result.exit_code == 3
    out = " ".join(strip_ansi(result.output).split())
    assert "soup.lock" in out
    assert "not valid JSON" in out


def test_lock_show_valid_lock_still_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lock_file = _write_valid_lock(tmp_path)
    result = runner.invoke(soup_app, ["lock", "show", str(lock_file)])
    assert result.exit_code == 0, result.output


def test_read_lock_corrupt_raises_value_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lock_file = tmp_path / "soup.lock"
    lock_file.write_text("garbage: [", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        read_lock("soup.lock")
    assert "not valid JSON" in str(excinfo.value)
    assert "soup.lock" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_read_lock_non_utf8_raises_value_error(tmp_path, monkeypatch):
    """A non-UTF-8 lock file hits the UnicodeDecodeError arm, not a raw crash."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.lock").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ValueError) as excinfo:
        read_lock("soup.lock")
    assert "not valid JSON" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_lock_show_non_utf8_lock_names_file_and_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lock_file = tmp_path / "soup.lock"
    lock_file.write_bytes(b"\xff\xfe\x00bad")
    result = runner.invoke(soup_app, ["lock", "show", str(lock_file)])
    assert result.exit_code == 3
    out = " ".join(strip_ansi(result.output).split())
    assert "soup.lock" in out
    assert "not valid JSON" in out


def test_lock_show_corrupt_lock_error_is_not_line_wrapped(tmp_path, monkeypatch):
    """A corrupt-lock path longer than the console width stays on one line.

    The ``soft_wrap=True`` on the lock error prints keeps long paths
    unbroken; without it Rich would wrap them at 80 columns.
    """
    monkeypatch.chdir(tmp_path)
    deep = tmp_path / ("d" * 40) / ("e" * 40)
    deep.mkdir(parents=True)
    lock_file = deep / "soup.lock"
    lock_file.write_text("garbage: [", encoding="utf-8")
    result = runner.invoke(soup_app, ["lock", "show", str(lock_file)])
    assert result.exit_code == 3
    assert str(lock_file) in strip_ansi(result.output)


# ---------- 3. expect module docstring ----------


def test_expect_module_docstring_exit_codes_are_right():
    doc = expect_cmd.__doc__ or ""
    assert "Exit code 3 on suite failure" not in doc
    assert "Exit code 2 on\nsuite failure" in " ".join(doc.split()) or (
        "Exit code 2" in doc and "EXIT_GATE_FAILED" in doc
    )


# ---------- 4. soup draft measure / distill: ImportError install hint ----------


def _write_prompts(tmp_path: Path) -> str:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"prompt": "hello"}) + "\n", encoding="utf-8"
    )
    return str(prompts)


def test_draft_measure_import_error_gives_train_hint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompts = _write_prompts(tmp_path)

    def _boom(model_id, trust_remote_code):
        raise ModuleNotFoundError("No module named 'transformers'")

    monkeypatch.setattr(draft_cmd, "_vocab_size_of", _boom)
    result = runner.invoke(
        soup_app,
        ["draft", "measure", "--target", "t", "--draft", "d", "--prompts", prompts],
    )
    assert result.exit_code == 1
    out = strip_ansi(result.output)
    assert "soup-cli[train]" in out
    assert "could not read model config" in out


def test_draft_distill_import_error_gives_train_hint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = _write_prompts(tmp_path)

    def _boom(model_id, trust_remote_code):
        raise ModuleNotFoundError("No module named 'transformers'")

    monkeypatch.setattr(draft_cmd, "_vocab_size_of", _boom)
    result = runner.invoke(
        soup_app,
        [
            "draft", "distill",
            "--target", "t",
            "--draft-base", "d",
            "--data", data,
            "--output", str(tmp_path / "draft"),
        ],
    )
    assert result.exit_code == 1
    out = strip_ansi(result.output)
    assert "soup-cli[train]" in out
    assert "could not read model config" in out
