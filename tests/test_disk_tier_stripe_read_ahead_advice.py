"""R4 fix wave, finding A — advice to lower the read-ahead must work after the N + 1 bump.

With two drives a configured ``stream_read_ahead`` of 2 is read as the default and raised to 3.
The host-RAM refusal, the stream_pin refusal and the pageable-fallback warning all used to say
"lower training.stream_read_ahead (currently 3)"; the operator sets 2, it is raised to 3 again,
and the same message follows. Every branch below checks the advice by FOLLOWING it: the value it
names is fed back through the setup's own depth rule, and the depth must actually fall.
"""

import re
from types import SimpleNamespace

import pytest

from soup_cli.utils.config_bounds import DEFAULT_STREAM_READ_AHEAD, MIN_STREAM_READ_AHEAD
from soup_cli.utils.stripe_roots import (
    STRIPE_DIRS_ENV,
    ReadAheadDecision,
    effective_read_ahead,
)


def _decision(configured, n_roots):
    return ReadAheadDecision(
        depth=effective_read_ahead(configured, n_roots), configured=configured, n_roots=n_roots
    )


def _suggested(message):
    found = re.search(r"Set training\.stream_read_ahead to (\d+)", message)
    assert found, message
    return int(found.group(1))


def _assert_the_advice_lowers_the_depth(message, decision):
    """Follow the advice through the same rule the setup applies."""
    value = _suggested(message)
    assert effective_read_ahead(value, decision.n_roots) < decision.depth, (value, message)


class TestTheRuleIsUnchanged:
    """The bump itself is not what the finding changes: a value equal to the default is not a
    decision, anything else is the operator's."""

    def test_one_root_leaves_every_value_alone(self):
        assert [effective_read_ahead(v, 1) for v in (1, 2, 3, 8)] == [1, 2, 3, 8]

    def test_the_default_becomes_one_more_than_the_drives(self):
        assert effective_read_ahead(DEFAULT_STREAM_READ_AHEAD, 2) == 3
        assert effective_read_ahead(DEFAULT_STREAM_READ_AHEAD, 3) == 4

    def test_an_explicit_value_is_kept(self):
        assert effective_read_ahead(1, 2) == 1 and effective_read_ahead(5, 2) == 5


class TestTheAdviceText:
    def test_one_root_keeps_the_historical_wording(self):
        advice = _decision(4, 1).lowering_advice()
        assert advice == "Lower training.stream_read_ahead (currently 4)"

    def test_nothing_lower_than_the_floor_is_offered(self):
        assert _decision(MIN_STREAM_READ_AHEAD, 1).lowering_advice() == ""
        assert _decision(MIN_STREAM_READ_AHEAD, 2).lowering_advice() == ""

    def test_a_raised_depth_says_so_and_names_the_way_out(self):
        decision = _decision(DEFAULT_STREAM_READ_AHEAD, 2)
        assert decision.raised and decision.depth == 3
        advice = decision.lowering_advice()
        assert "raised from 2" in advice and "2 drives" in advice, advice
        assert STRIPE_DIRS_ENV in advice, advice
        assert "currently 3" in advice, advice
        _assert_the_advice_lowers_the_depth(advice, decision)

    def test_an_explicit_depth_equal_to_the_bump_is_not_called_raised(self):
        """Explicit 3 with two drives: lowering to 2 would be re-raised to 3, so the advice
        must skip it — but must not blame a bump that did not happen, and unsetting the
        variable would not lower an explicit depth, so it is not offered."""
        decision = _decision(3, 2)
        assert not decision.raised
        advice = decision.lowering_advice()
        assert "raised from" not in advice and STRIPE_DIRS_ENV not in advice, advice
        assert "counts as the default" in advice, advice
        _assert_the_advice_lowers_the_depth(advice, decision)

    def test_three_drives_offer_the_next_value_down_that_is_not_re_raised(self):
        decision = _decision(DEFAULT_STREAM_READ_AHEAD, 3)
        advice = decision.lowering_advice()
        assert _suggested(advice) == 3, advice
        _assert_the_advice_lowers_the_depth(advice, decision)

    @pytest.mark.parametrize("n_roots", [2, 3, 4, 7])
    @pytest.mark.parametrize("configured", [1, 2, 3, 4, 5, 8])
    def test_every_offered_value_actually_lowers_the_depth(self, configured, n_roots):
        decision = _decision(configured, n_roots)
        advice = decision.lowering_advice()
        if decision.depth <= MIN_STREAM_READ_AHEAD:
            assert advice == ""
        else:
            _assert_the_advice_lowers_the_depth(advice, decision)


class TestTheHostRamRefusal:
    def _refuse(self, decision):
        from soup_cli.trainer.stream_setup import _validate_stream_staging_ram_fit

        with pytest.raises(ValueError) as caught:
            _validate_stream_staging_ram_fit(
                staging_bytes=5 * 10**9,
                read_ahead=decision.depth,
                free_ram=4 * 10**9,
                read_ahead_decision=decision,
            )
        return " ".join(str(caught.value).split())

    def test_after_the_bump_it_names_the_bump_and_a_value_that_works(self):
        decision = _decision(DEFAULT_STREAM_READ_AHEAD, 2)
        message = self._refuse(decision)
        assert "Lower training.stream_read_ahead (currently 3)" not in message, message
        assert "raised from 2" in message and STRIPE_DIRS_ENV in message, message
        _assert_the_advice_lowers_the_depth(message, decision)
        assert "free RAM" in message, message

    def test_without_a_decision_it_reads_as_before(self):
        """The one-root call sites (and the #971 tests) pass no decision."""
        from soup_cli.trainer.stream_setup import _validate_stream_staging_ram_fit

        with pytest.raises(ValueError, match=r"Lower training\.stream_read_ahead \(currently 4\)"):
            _validate_stream_staging_ram_fit(
                staging_bytes=5 * 10**9, read_ahead=4, free_ram=4 * 10**9
            )


class _Console:
    def __init__(self):
        self.printed = []

    def print(self, message, *args, **kwargs):
        self.printed.append(str(message))


def _failing_pinned_source(monkeypatch):
    """AsyncDiskSource whose pinned construction fails the way a refused page-lock does."""
    import soup_cli.utils.async_disk_source as ads
    import soup_cli.utils.layer_stream_runtime as rt

    class _Source:
        def __init__(self, shard_dir, n_layers, spec, *, pin=True, **kwargs):
            if pin:
                raise RuntimeError("CUDA error: out of memory")
            self.pinned = False
            self.kwargs = kwargs

    monkeypatch.setattr(ads, "AsyncDiskSource", _Source)
    monkeypatch.setattr(rt, "recover_from_failed_page_lock", lambda **_kwargs: True)
    monkeypatch.setattr(rt, "_recover_before_refusing", lambda _console: None)
    return rt


class TestTheRuntimeMessages:
    def test_the_stream_pin_refusal_names_the_bump(self, monkeypatch):
        rt = _failing_pinned_source(monkeypatch)
        decision = _decision(DEFAULT_STREAM_READ_AHEAD, 2)
        with pytest.raises(RuntimeError) as caught:
            rt._build_source(
                "shards",
                3,
                {},
                True,
                _Console(),
                "disk",
                require_pin=True,
                read_ahead=decision.depth,
                layer_roots=(0, 1, 0),
                read_ahead_decision=decision,
            )
        message = " ".join(str(caught.value).split())
        assert "raised from 2" in message and STRIPE_DIRS_ENV in message, message
        _assert_the_advice_lowers_the_depth(message, decision)
        assert "unset training.stream_pin" in message, message

    def test_the_pageable_fallback_warning_names_the_bump(self, monkeypatch):
        rt = _failing_pinned_source(monkeypatch)
        decision = _decision(DEFAULT_STREAM_READ_AHEAD, 2)
        console = _Console()
        source, pinned = rt._build_source(
            "shards",
            3,
            {},
            True,
            console,
            "disk",
            read_ahead=decision.depth,
            layer_roots=(0, 1, 0),
            read_ahead_decision=decision,
        )
        assert pinned is False and source.kwargs["read_ahead"] == 3
        warning = " ".join(" ".join(console.printed).split())
        assert "PAGEABLE" in warning and "raised from 2" in warning, warning
        _assert_the_advice_lowers_the_depth(warning, decision)

    def test_without_a_decision_the_placement_still_tells_the_drive_count(self, monkeypatch):
        """A direct caller (the harness) passes the depth and the placement only: the advice
        must still skip the value that would be re-raised, and must not claim a bump."""
        rt = _failing_pinned_source(monkeypatch)
        console = _Console()
        rt._build_source(
            "shards", 3, {}, True, console, "disk", read_ahead=3, layer_roots=(0, 1, 0)
        )
        warning = " ".join(" ".join(console.printed).split())
        assert "raised from" not in warning, warning
        _assert_the_advice_lowers_the_depth(warning, ReadAheadDecision(3, 3, 2))


class TestTheSetupCarriesTheDecision:
    def test_the_setup_helper_returns_the_whole_decision(self):
        import io

        from rich.console import Console

        from soup_cli.trainer import stream_setup

        console = Console(file=io.StringIO(), width=200, color_system=None)
        decision = stream_setup._effective_read_ahead(
            SimpleNamespace(stream_read_ahead=DEFAULT_STREAM_READ_AHEAD), 2, console
        )
        assert decision == ReadAheadDecision(depth=3, configured=2, n_roots=2)
