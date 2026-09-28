"""#1116, second half — the attach ratchet over every shipped config.

``scripts/check_recipe_attach.py`` compares ``soup recipes verify --json`` rows
against pinned exceptions. These tests are offline: the Hub run is the workflow's.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.check_recipe_attach import EXCEPTIONS, compare, count_line, main, report


def _row(name, verdict, detail=""):
    return {"name": name, "verdict": verdict, "detail": detail}


class TestCompare:
    def test_a_pinned_failure_keeps_the_run_green(self):
        result = compare([_row("a", "cannot_attach")], {"a": "why"})

        assert result.ok and result.pinned == ["a"]

    def test_a_new_failure_turns_it_red_and_is_named(self):
        result = compare([_row("b", "cannot_attach", "no mapping")], {"a": "why"})

        assert not result.ok
        assert result.new_failures == [("b", "no mapping")]
        assert "NEW cannot attach: b: no mapping" in report(result)

    def test_a_pinned_config_that_now_attaches_must_leave_the_list(self):
        result = compare([_row("a", "attaches")], {"a": "why"})

        assert not result.ok and result.now_attach == ["a"]
        assert "now attaches, remove from EXCEPTIONS: a" in report(result)

    @pytest.mark.parametrize("pinned", [True, False])
    def test_unverified_is_skipped_never_a_pass_or_a_failure(self, pinned):
        """A gated base without a token (the pull_request run) must not count
        either way, pinned or not (ruled on #1116)."""
        exceptions = {"g": "why"} if pinned else {}
        result = compare([_row("g", "unverified", "gated")], exceptions)

        assert result.ok
        assert result.skipped == [("g", "gated")]
        assert result.verified == [] and result.pinned == [] and result.now_attach == []

    def test_no_adapter_counts_as_verified(self):
        result = compare([_row("c", "no_adapter")], {})

        assert result.ok and result.verified == ["c"]

    def test_an_unknown_verdict_is_an_error_not_a_pass(self):
        with pytest.raises(ValueError, match="unknown verdict"):
            compare([_row("x", "maybe")], {})

    def test_the_count_line_says_what_was_not_covered(self):
        result = compare(
            [_row("a", "attaches"), _row("g", "unverified"), _row("p", "cannot_attach")],
            {"p": "why"},
        )

        assert count_line(result) == (
            "1 verified, 1 skipped (gated, remote code or unreadable here), "
            "1 pinned cannot-attach, 0 new cannot-attach, 0 pinned that now attach"
        )


class TestMain:
    @pytest.fixture
    def verify(self, monkeypatch):
        def install(rows, returncode=2, stderr=""):
            out = SimpleNamespace(returncode=returncode, stdout=json.dumps(rows), stderr=stderr)
            monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: out)

        return install

    def test_pinned_failures_only_exit_0(self, verify, capsys):
        verify([_row(name, "cannot_attach") for name in EXCEPTIONS])

        assert main() == 0
        assert f"{len(EXCEPTIONS)} pinned cannot-attach" in capsys.readouterr().out

    def test_a_new_failure_exits_2(self, verify):
        verify([_row("brand-new-sft", "cannot_attach", "boom")])

        assert main() == 2

    def test_a_pinned_config_that_attaches_exits_2(self, verify):
        verify([_row(next(iter(EXCEPTIONS)), "attaches")])

        assert main() == 2

    def test_the_command_failing_is_not_a_verdict(self, verify, capsys):
        """Exit 1 from verify is a missing library, not a ratchet result."""
        verify([], returncode=1, stderr="soup recipes verify needs torch")

        assert main() == 1
        assert "exited 1" in capsys.readouterr().err


class TestTheExceptions:
    def test_every_pinned_name_is_a_shipped_config(self):
        from soup_cli.commands.recipes import _configs_to_verify

        shipped = {name for name, _cfg in _configs_to_verify(None, True)}

        assert set(EXCEPTIONS) <= shipped, sorted(set(EXCEPTIONS) - shipped)

    def test_every_pinned_name_has_a_reason(self):
        assert all(len(reason) > 40 for reason in EXCEPTIONS.values())
