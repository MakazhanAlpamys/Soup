"""#1830 — four small gaps in what the gate-style commands print.

The exit codes were already right (0 pass, 2 gate failure, 3 usage/input
error, 1 runtime); only the text was thin. Each gap is independent:

1. ``soup expect`` with a missing data or suite file printed only the bare
   path. It now names the problem (``data file not found: ...`` /
   ``suite file not found: ...``), still exiting 3.
2. ``soup lock check`` / ``soup lock show`` on a corrupt lock file printed the
   raw ``json`` error without naming the file. ``read_lock`` now re-raises a
   ``ValueError`` that names the file and says the JSON is invalid (exit 3).
3. The module docstring of ``commands/expect.py`` claimed "Exit code 3 on
   suite failure"; a suite failure exits 2. The docstring is corrected.
4. ``soup draft measure`` / ``soup draft distill`` on a core-only install
   printed ``could not read model config: No module named 'transformers'``
   with no hint. They now point at the ``[train]`` extra (exit 1 kept).

Every test asserts the message a user would grep for; with the fix reverted,
each one fails (mutation check).
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import draft as draft_mod
from soup_cli.utils.exit_codes import EXIT_RUNTIME_ERROR, EXIT_USAGE_ERROR
from tests.conftest import strip_ansi

_VALID_SUITE = "expectations:\n  - name: expect_no_pii\n"


def _plain(text: "str | None") -> str:
    """ANSI-stripped, whitespace-collapsed output (Rich wraps at 80 columns in
    CliRunner, so a long message can span lines)."""
    return " ".join(strip_ansi(text).split())


# ---------------------------------------------------------------------------
# 1. soup expect — missing data / suite file names the problem
# ---------------------------------------------------------------------------


def test_expect_missing_data_file_names_the_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "suite.yaml").write_text(_VALID_SUITE, encoding="utf-8")

    result = CliRunner().invoke(app, ["expect", "missing.jsonl", "suite.yaml"])

    assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
    assert "data file not found: missing.jsonl" in _plain(result.output)


def test_expect_missing_suite_file_names_the_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"text": "hi"}) + "\n", encoding="utf-8"
    )

    result = CliRunner().invoke(app, ["expect", "data.jsonl", "missing.yaml"])

    assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
    assert "suite file not found: missing.yaml" in _plain(result.output)


# ---------------------------------------------------------------------------
# 2. soup lock check / show — a corrupt lock file is named as invalid JSON
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["lock", "show", "soup.lock"],
        [
            "lock", "check", "soup.lock",
            "--base-sha", "0" * 64,
            "--dataset-sha", "0" * 64,
            "--env-hash", "0" * 64,
            "--base-model", "gpt2",
        ],
    ],
    ids=["show", "check"],
)
def test_lock_commands_name_a_corrupt_lock_file(tmp_path, monkeypatch, args):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.lock").write_text("garbage:", encoding="utf-8")

    result = CliRunner().invoke(app, args)

    assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
    plain = _plain(result.output)
    assert "not valid JSON" in plain
    assert "soup.lock" in plain


# ---------------------------------------------------------------------------
# 3. commands/expect.py module docstring matches the real exit codes
# ---------------------------------------------------------------------------


def test_expect_module_docstring_does_not_claim_exit_3_on_suite_failure():
    from soup_cli.commands import expect as expect_mod

    doc = expect_mod.__doc__ or ""
    assert "Exit code 3 on suite failure" not in doc
    # A suite failure exits 2 (EXIT_GATE_FAILED); 3 is the usage/input error.
    assert "exits 2" in doc


# ---------------------------------------------------------------------------
# 4. soup draft measure / distill — a missing [train] extra points the user at it
# ---------------------------------------------------------------------------


def _raise_missing_transformers(*args, **kwargs):
    raise ModuleNotFoundError("No module named 'transformers'")


def test_draft_measure_hints_the_train_extra_when_transformers_is_absent(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts.jsonl").write_text(
        json.dumps({"prompt": "hello"}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(draft_mod, "_vocab_size_of", _raise_missing_transformers)

    result = CliRunner().invoke(
        app,
        ["draft", "measure", "--target", "gpt2", "--draft", "gpt2",
         "--prompts", "prompts.jsonl"],
    )

    assert result.exit_code == EXIT_RUNTIME_ERROR, (result.output, repr(result.exception))
    plain = _plain(result.output)
    assert "could not read model config" in plain
    assert "soup-cli[train]" in plain


def test_draft_distill_hints_the_train_extra_when_transformers_is_absent(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"},
                                 {"role": "assistant", "content": "ok"}]}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(draft_mod, "_vocab_size_of", _raise_missing_transformers)

    result = CliRunner().invoke(
        app,
        ["draft", "distill", "--target", "gpt2", "--draft-base", "gpt2",
         "--data", "data.jsonl", "--output", "draft_out"],
    )

    assert result.exit_code == EXIT_RUNTIME_ERROR, (result.output, repr(result.exception))
    plain = _plain(result.output)
    assert "could not read model config" in plain
    assert "soup-cli[train]" in plain
