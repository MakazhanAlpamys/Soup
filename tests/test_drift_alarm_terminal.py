"""Drift-alarm displays outside text literally without changing the report or payload."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import drift_alarm as drift_cmd
from soup_cli.utils.drift_alarm import post_webhook, run_drift_check
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()
_CONTROLS = "\x9b\u202e"
_SPELLINGS = [
    pytest.param("xn[red]value", id="tag-guard"),
    pytest.param("xn[/]value", id="closing-tag-guard"),
    pytest.param("xn:warning:value", id="emoji-code"),
    pytest.param(f"xn[bold]value{_CONTROLS}", id="tag-and-controls"),
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(drift_cmd, "console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")

    def _no_webhook(**kwargs):
        raise AssertionError("this test must not send a webhook")

    monkeypatch.setattr(drift_cmd, "post_webhook", _no_webhook)


def _write(name: str, texts: list[str], *, key: str = "output") -> str:
    Path(name).write_text(
        "".join(json.dumps({key: text}) + "\n" for text in texts), encoding="utf-8"
    )
    return name


def _invoke(reference: str, live: str, *args: str):
    return runner.invoke(
        app, ["drift-alarm", "--reference", reference, "--live", live, *args]
    )


def _plain(result, exit_code: int) -> str:
    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    if exit_code:
        assert isinstance(result.exception, SystemExit), repr(result.exception)
    else:
        assert result.exception is None
    # Check the captured output before ANSI stripping can conceal any controls.
    assert "\x9b" not in result.stdout and "\u202e" not in result.stdout
    assert result.stderr == ""
    return " ".join(strip_ansi(result.stdout).split())


@pytest.mark.parametrize("side", ["reference", "live"])
@pytest.mark.parametrize("name", _SPELLINGS)
def test_a_missing_input_is_named_as_typed(side, name):
    existing = _write("existing.jsonl", ["base"])
    paths = {"reference": existing, "live": existing}
    paths[side] = name

    result = _invoke(paths["reference"], paths["live"])

    text = _plain(result, 1)
    # FileNotFoundError(path) carries the raw string, not OSError's repr of it.
    assert f"File not found: {name.replace(_CONTROLS, '')}" in text


@pytest.mark.parametrize("side", ["reference", "live"])
@pytest.mark.parametrize("name", _SPELLINGS)
def test_an_outside_input_is_named_as_typed(side, name):
    existing = _write("existing.jsonl", ["base"])
    paths = {"reference": existing, "live": existing}
    paths[side] = f"../{name}"

    result = _invoke(paths["reference"], paths["live"])

    text = _plain(result, 1)
    # The real validator uses repr: controls are already literal escape text.
    assert f"{side}_path {paths[side]!r} is outside cwd" in text


@pytest.mark.parametrize("side", ["reference", "live"])
def test_an_empty_input_keeps_its_emoji_code(side):
    # This case really creates the file, so ':' names cannot run on Windows.
    import sys

    if sys.platform == "win32":
        pytest.skip("':' is not valid in a Windows filename")
    existing = _write("existing.jsonl", ["base"])
    empty = _write("xn:warning:.jsonl", ["ignored"], key="completion")
    paths = {"reference": existing, "live": existing}
    paths[side] = empty

    result = _invoke(paths["reference"], paths["live"])

    text = _plain(result, 1)
    assert f"{side} file {empty!r} has 0 readable rows" in text
    assert "'output', 'response' or 'text'" in text


@pytest.mark.parametrize("option", ["--slack-url", "--discord-url"])
@pytest.mark.parametrize(
    "netloc",
    [
        pytest.param("xn[red]host\uff0f@[::1]", id="tag-guard"),
        pytest.param("xn:warning:host\uff0f.example", id="emoji-code"),
        pytest.param(f"xn[bold]:warning:host{_CONTROLS}\uff0f@[::1]", id="tag-and-controls"),
    ],
)
def test_a_refused_webhook_keeps_the_netloc_text(option, netloc):
    # A fullwidth slash is rejected by urllib.parse before transport. Unlike
    # its invalid-IPv6 error, this message quotes netloc without repr().
    result = _invoke("unused", "unused", option, f"https://{netloc}/hook")

    text = _plain(result, 2)
    assert (
        f"{option}: netloc '{netloc.replace(_CONTROLS, '')}' contains invalid "
        "characters under NFKC normalization"
    ) in text


@pytest.mark.parametrize("token", _SPELLINGS)
def test_top_tokens_keep_the_rank_numbers_and_typed_text(token):
    reference = _write("reference.jsonl", ["base base base base shared"] * 2)
    live = _write("live.jsonl", [f"{token} {token} {token} shared base"] * 3)

    report = run_drift_check(reference_path=reference, live_path=live, threshold=0.2)
    assert [item[0] for item in report.top_drift_tokens] == ["base", token, "shared"]
    assert [item[1] for item in report.top_drift_tokens] == pytest.approx([0.6, 0.6, 0.0])
    expected_kl = 0.6 * math.log(0.6 / 1e-9) + 0.2 * math.log(0.2 / 0.8)
    assert report.kl_divergence == pytest.approx(expected_kl)

    result = _invoke(reference, live)

    text = _plain(result, 3)
    assert "DRIFT" in text
    assert f"KL: {expected_kl:.4f} threshold: 0.2000" in text
    assert "n_reference: 2 n_live: 3" in text
    assert (
        f"Top drift tokens: base=0.600, {token.replace(_CONTROLS, '')}=0.600, shared=0.000"
    ) in text


def test_display_sanitizing_does_not_change_webhook_payload(monkeypatch):
    token = f"xn[bold]:warning:token{_CONTROLS}"
    reference = _write("reference.jsonl", ["base base base base shared"] * 2)
    live = _write("live.jsonl", [f"{token} {token} {token} shared base"] * 3)
    report = run_drift_check(reference_path=reference, live_path=live, threshold=0.2)
    sent = []

    def _post(url, *, json, timeout):
        sent.append((url, json, timeout))
        return SimpleNamespace(status_code=204 if len(sent) == 1 else 500)

    # Keep real URL validation and webhook assembly; replace only HTTP transport.
    monkeypatch.setattr(drift_cmd, "post_webhook", post_webhook)
    monkeypatch.setattr(httpx, "post", _post)
    result = _invoke(
        reference, live, "--slack-url", "http://localhost:8080/slack/",
        "--discord-url", "http://localhost:8080/discord/",
    )

    text = _plain(result, 3)
    assert f"{token.replace(_CONTROLS, '')}=0.600" in text
    assert "slack webhook: delivered" in text and "discord webhook: failed" in text
    assert [item[0] for item in sent] == [
        "http://localhost:8080/slack", "http://localhost:8080/discord"
    ]
    expected = {
        "kl": report.kl_divergence, "threshold": 0.2, "n_reference": 2, "n_live": 3,
        "top_drift_tokens": [[label, delta] for label, delta in report.top_drift_tokens],
    }
    assert [item[1] for item in sent] == [expected, expected]
    assert all(item[2] == 5.0 for item in sent)


@pytest.mark.parametrize("drift", [False, True], ids=["ok", "drift"])
def test_verdict_bold_and_border_colour_are_kept(monkeypatch, drift):
    monkeypatch.setattr(
        drift_cmd, "console",
        Console(width=500, force_terminal=True, color_system="standard", no_color=False),
    )
    reference = _write("reference.jsonl", ["base"])
    live = _write("live.jsonl", ["xn:warning:token" if drift else "base"])

    result = _invoke(reference, live)

    text = _plain(result, 3 if drift else 0)
    verdict, colour = ("DRIFT", "31") if drift else ("OK", "32")
    assert f"\x1b[1m{verdict}\x1b[0m" in result.stdout
    assert f"\x1b[{colour}m" in result.stdout
    assert "drift-alarm" in text


def test_invalid_numeric_threshold_retains_exit_two():
    result = _invoke("unused", "unused", "--threshold", "-0.1")

    assert "threshold must be > 0, got -0.1" in _plain(result, 2)
