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
from tests.conftest import strip_ansi


def _row(name, verdict, detail=""):
    return {"name": name, "verdict": verdict, "detail": detail}


class TestCompare:
    def test_a_pinned_failure_keeps_the_run_green(self):
        result = compare([_row("a", "cannot_attach")], {"a": "why"})

        assert result.ok and result.pinned == ["a"]

    def test_a_new_failure_turns_it_red_and_is_named(self):
        result = compare(
            [_row("a", "cannot_attach"), _row("b", "cannot_attach", "no mapping")],
            {"a": "why"},
        )

        # Red because of "b" alone: "a" is reported, so nothing is missing.
        assert result.not_reported == [] and result.pinned == ["a"]
        assert not result.ok
        assert result.new_failures == [("b", "no mapping")]
        assert any(line.startswith("NEW cannot attach: b: no mapping") for line in report(result))

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

    def test_a_pinned_config_missing_from_the_run_is_flagged(self):
        """verify skips a config that does not parse, so a pinned one that stopped
        parsing (or was removed) would otherwise vanish while the run stays green."""
        result = compare([_row("a", "attaches")], {"gone": "why"})

        assert not result.ok and result.not_reported == ["gone"]
        assert "pinned but not reported by verify: gone" in report(result)

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


def test_skipped_rows_are_reported_by_reason():
    gated = "gated repo; set HF_TOKEN to check it"
    remote = "needs trust_remote_code, which this check never enables"
    result = compare(
        [_row("g1", "unverified", gated), _row("g2", "unverified", gated),
         _row("r", "unverified", remote), _row("a", "attaches")],
        {},
    )

    lines = report(result)

    assert f"  skipped 2: {gated}" in lines
    assert f"  skipped 1: {remote}" in lines


def test_a_new_failure_says_how_to_pin_it():
    result = compare([_row("b", "cannot_attach", "no mapping")], {})

    assert any("pin it in EXCEPTIONS in scripts/check_recipe_attach.py" in line
               for line in report(result))


class TestMain:
    @pytest.fixture
    def verify(self, monkeypatch):
        def install(rows, returncode=2, stderr=""):
            out = SimpleNamespace(returncode=returncode, stdout=json.dumps(rows), stderr=stderr)

            def run(argv, *_a, **_k):
                # The whole catalogue, templates included: `--no-templates` would
                # drop templates/audio.yaml from the rows (#1388 review).
                assert list(argv[-3:]) == ["recipes", "verify", "--json"], argv
                return out

            monkeypatch.setattr(subprocess, "run", run)

        return install

    def test_pinned_failures_only_exit_0(self, verify, capsys):
        verify([_row(name, "cannot_attach") for name in EXCEPTIONS])

        assert main() == 0
        assert f"{len(EXCEPTIONS)} pinned cannot-attach" in strip_ansi(capsys.readouterr().out)

    def test_a_new_failure_exits_2(self, verify):
        pinned = [_row(name, "cannot_attach") for name in EXCEPTIONS]
        verify(pinned + [_row("brand-new-sft", "cannot_attach", "boom")])

        assert main() == 2

    def test_a_pinned_config_that_attaches_exits_2(self, verify):
        first, *rest = EXCEPTIONS
        verify([_row(first, "attaches")] + [_row(name, "cannot_attach") for name in rest])

        assert main() == 2

    def test_verify_warnings_reach_the_log_whatever_the_verdict(self, verify, capsys):
        verify([_row(name, "cannot_attach") for name in EXCEPTIONS], stderr="Partial LoRA coverage")

        assert main() == 0
        assert "Partial LoRA coverage" in strip_ansi(capsys.readouterr().err)

    def test_the_command_failing_is_not_a_verdict(self, verify, capsys):
        """Exit 1 from verify is a missing library, not a ratchet result."""
        verify([], returncode=1, stderr="soup recipes verify needs torch")

        assert main() == 1
        assert "exited 1" in strip_ansi(capsys.readouterr().err)


class TestTheExceptions:
    def test_every_pinned_name_is_a_shipped_config(self):
        from soup_cli.commands.recipes import _configs_to_verify

        shipped = {name for name, _cfg in _configs_to_verify(None, True)}

        assert set(EXCEPTIONS) <= shipped, sorted(set(EXCEPTIONS) - shipped)

    def test_every_pinned_name_has_a_reason(self):
        assert all(len(reason) > 40 for reason in EXCEPTIONS.values())
