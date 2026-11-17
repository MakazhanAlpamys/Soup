"""#1808 — ``soup drift-alarm`` with nothing to compare is not OK.

With no readable live (or reference) row the KL was 0.0, so the command showed a
green OK with ``n_live: 0``, sent no webhook and exited 0; a live log keyed
``completion`` (or ``messages`` / ``choices``), an empty file or all-malformed
JSON all passed the gate. Now zero rows on either side exits 1 with a message
that names the file and the keys the reader looks for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.drift_alarm import run_drift_check

from .conftest import strip_ansi


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _write(path: Path, key: str, texts: list[str]) -> str:
    path.write_text("".join(json.dumps({key: t}) + "\n" for t in texts), encoding="utf-8")
    return str(path.name)


@pytest.fixture()
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    ref = _write(tmp_path / "ref.jsonl", "output", [f"The answer is {i}." for i in range(20)])
    return tmp_path, ref


def test_live_rows_under_an_unread_key_exit_non_zero_and_name_the_keys(files) -> None:
    tmp_path, ref = files
    live = _write(tmp_path / "live.jsonl", "completion", ["Something else entirely."] * 20)
    result = CliRunner().invoke(app, ["drift-alarm", "--reference", ref, "--live", live])
    out = _plain(result.output)
    assert result.exit_code == 1, (result.exit_code, out)
    assert "OK" not in out.split("drift-alarm")[-1][:40]
    assert "live.jsonl" in out
    for key in ("output", "response", "text"):
        assert key in out
    assert "0 readable" in out or "no readable" in out.lower()


def test_reference_rows_under_an_unread_key_exit_non_zero_too(files) -> None:
    tmp_path, _ = files
    ref = _write(tmp_path / "ref2.jsonl", "completion", ["The answer is 1."] * 20)
    live = _write(tmp_path / "live.jsonl", "output", ["The answer is 1."] * 20)
    result = CliRunner().invoke(app, ["drift-alarm", "--reference", ref, "--live", live])
    out = _plain(result.output)
    assert result.exit_code == 1, (result.exit_code, out)
    assert "ref2.jsonl" in out


def test_an_empty_live_file_exits_non_zero(files) -> None:
    tmp_path, ref = files
    (tmp_path / "live.jsonl").write_text("", encoding="utf-8")
    result = CliRunner().invoke(app, ["drift-alarm", "--reference", ref, "--live", "live.jsonl"])
    assert result.exit_code == 1, _plain(result.output)


def test_the_library_call_raises_rather_than_reporting_kl_zero(files) -> None:
    tmp_path, ref = files
    live = _write(tmp_path / "live.jsonl", "completion", ["x"] * 3)
    with pytest.raises(ValueError, match="live.jsonl"):
        run_drift_check(reference_path=ref, live_path=live, threshold=0.2)


def test_control_same_text_under_output_is_ok_exit_zero(files) -> None:
    tmp_path, ref = files
    live = _write(tmp_path / "live.jsonl", "output", [f"The answer is {i}." for i in range(20)])
    result = CliRunner().invoke(app, ["drift-alarm", "--reference", ref, "--live", live])
    out = _plain(result.output)
    assert result.exit_code == 0, (result.exit_code, out)
    assert "OK" in out


def test_control_different_text_under_output_is_drift_exit_three(files) -> None:
    tmp_path, ref = files
    live = _write(tmp_path / "live.jsonl", "output", ["Something else entirely."] * 20)
    result = CliRunner().invoke(app, ["drift-alarm", "--reference", ref, "--live", live])
    out = _plain(result.output)
    assert result.exit_code == 3, (result.exit_code, out)
    assert "DRIFT" in out


@pytest.mark.parametrize("side", ["reference", "live"])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_first_row_counts_toward_drift_verdict(tmp_path, monkeypatch, side, encoding) -> None:
    monkeypatch.chdir(tmp_path)
    rows = '{"output": "first"}\r\n{"output": "second"}\r\n'
    for name in ("reference", "live"):
        data = rows.encode(encoding if name == side else "utf-8")
        (tmp_path / f"{name}.jsonl").write_bytes(data)

    result = CliRunner().invoke(app, [
        "drift-alarm", "--reference", "reference.jsonl", "--live", "live.jsonl",
    ])

    out = _plain(result.output)
    assert result.exit_code == 0, (result.exit_code, out)
    assert "KL: 0.0000" in out
    assert "n_reference: 2" in out
    assert "n_live: 2" in out
