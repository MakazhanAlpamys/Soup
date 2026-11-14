"""``soup loop status`` prints loop.yaml fields through ``for_terminal``.

``loop.yaml`` can come from a cloned repository, and ``rich.markup.escape`` passes ESC and the
other C0 bytes straight through: a field such as ``"\\u001b]0;x\\u0007"`` would retitle the
victim's terminal when they run ``soup loop status``. Every field the command prints from the
file has its control characters stripped; the visible text stays.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.loop_state import LoopState, read_state, write_state

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

ESC_TITLE = "\x1b]0;PWNED\x07"  # a window-title change: ESC ] 0 ; text BEL
C0_BYTES = "\x07\x08\x0b\x0c\x1b"
C1_CSI = "\x9b31m"  # a one-character CSI


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _hostile_state(**fields) -> LoopState:
    return LoopState(
        served_model=f"model{ESC_TITLE}-served",
        eval_suite=f"evals{C0_BYTES}.json",
        baseline=f"base{C1_CSI}line",
        base_model=f"org/base{ESC_TITLE}",
        trace_dir=f"traces{C0_BYTES}",
        feedback_log=f"feedback{ESC_TITLE}.jsonl",
        production_name=f"prod{ESC_TITLE}",
        **fields,
    )


def _status() -> str:
    result = runner.invoke(app, ["loop", "status"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return result.output


def _assert_no_control_bytes(output: str) -> None:
    for char in "\x1b\x07\x08\x0b\x0c\x9b":
        assert char not in output, (repr(char), output)


def test_a_hostile_loop_yaml_prints_no_control_bytes(cwd):
    write_state(_hostile_state(), str(cwd / ".soup" / "loop.yaml"))

    output = _status()

    _assert_no_control_bytes(output)


def test_the_visible_text_of_each_field_is_kept(cwd):
    write_state(_hostile_state(), str(cwd / ".soup" / "loop.yaml"))

    plain = " ".join(_ANSI.sub("", _status()).split())

    for visible in ("PWNED", "-served", "evals", "line", "org/base", "traces", "feedback",
                    "prod"):
        assert visible in plain, (visible, plain)


def test_the_canary_and_waiting_note_are_clean_too(cwd):
    state = _hostile_state(
        canary_active=f"cand{ESC_TITLE}", canary_traffic_pct=5.0,
        canary_rollout_id="rollout-1",
    )
    write_state(state, str(cwd / ".soup" / "loop.yaml"))

    output = _status()

    _assert_no_control_bytes(output)
    assert "cand" in output


def test_markup_in_a_field_is_still_shown_literally(cwd):
    write_state(
        LoopState(served_model="[bold red]m[/]", eval_suite="e.json", baseline="b"),
        str(cwd / ".soup" / "loop.yaml"),
    )

    plain = " ".join(_ANSI.sub("", _status()).split())

    assert "[bold red]m[/]" in plain


def test_the_state_file_itself_is_not_rewritten_by_status(cwd):
    path = str(cwd / ".soup" / "loop.yaml")
    write_state(_hostile_state(), path)
    before = read_state(path)

    _status()

    assert read_state(path).served_model == before.served_model == f"model{ESC_TITLE}-served"
