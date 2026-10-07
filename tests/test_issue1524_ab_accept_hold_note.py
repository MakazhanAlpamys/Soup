"""#1524 — a `continue` held by the 3x rule says so, with the two spreads.

#1419 holds `accept_h0` back at `continue` while the tested rows' pooled
standard deviation is more than `ACCEPT_HOLD_SPREAD_RATIO` (3) times the
held-out rows'. That fixed the wrong accepts after a saturated start, and left
the other side open: the held-out rows are FIXED, so once the hold fires, more
rows of the same kind never release it. A run that sits there prints "Insufficient
evidence. Collect more samples and re-run." with a `log_likelihood_ratio`
already below the accept boundary, and that advice cannot help — the held-out
spread does not move.

So the verdict now carries `accept_held` plus the two spreads, and the panel
prints one line that names the ratio and what to do about it. The JSON gains
keys only: nothing renamed, nothing removed.

The two tests that matter are the pair below. `test_a_held_accept_says_why`
fails if the hold is removed (it asserts the field AND that the hold is what
caused it, by monkeypatching the ratio away), and
`test_an_unheld_continue_carries_no_note` pins the other direction: a `continue`
that is merely short of evidence must not borrow this explanation.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import re

import pytest
from typer.testing import CliRunner

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BOX_RE = re.compile(f"[{chr(0x2500)}-{chr(0x257F)}|]")


def _plain(text: str) -> str:
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", text)).split())


def _step(control, treatment, metric="judge_score", **kwargs):
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    return msprt_step(MsprtConfig(metric=metric, **kwargs), control=control, treatment=treatment)


def _saturated_warm_up(rows: int = 60, *, seed: int = 0, effect_size: float = 0.1):
    """The issue's repro shape: the first 5 rows of each arm nearly constant at
    1.0 (a saturated judge / warm cache), then ordinary rows drawn from the SAME
    distribution for both arms, so there is no real difference to find.

    The warm-up spread is tiny but non-zero. Exactly-equal rows would not work:
    `_held_out_rows` keeps holding out rows while both arms are still constant,
    so the held-out window would swallow the ordinary rows too and the two
    spreads would agree. A saturated *judge* returns near-1.0, not bit-equal 1.0.
    """
    rng = random.Random(seed)
    warm = [1.0 + rng.gauss(0.0, 0.002) for _ in range(5)]
    ordinary = [
        round(min(1.0, max(0.0, rng.gauss(0.82, 0.05))), 3) for _ in range(rows)
    ]
    return warm + ordinary, warm + ordinary


# ---------------------------------------------------------------------------
# The hold names itself
# ---------------------------------------------------------------------------


class TestAcceptHoldNote:
    """A `continue` held by the 3x rule reports the hold and its two spreads."""

    def test_a_held_accept_says_why(self):
        """Saturated warm-up: `continue`, `accept_held` true, both spreads real.

        The premise is that the accept boundary is already crossed, so the hold
        is the only thing keeping this at `continue` — otherwise the new field
        would be reporting a reason that is not the reason.
        """
        from soup_cli.utils.ab_test import PRIOR_SCALE_ROWS

        control, treatment = _saturated_warm_up(40, seed=2)

        verdict = _step(control, treatment)

        # The accept boundary is crossed: log(beta / (1 - alpha)).
        assert verdict.decision == "continue", verdict.decision
        assert verdict.log_likelihood_ratio <= math.log(0.20 / (1.0 - 0.05))
        assert verdict.accept_held is True
        # Both spreads are present, finite, and the tested rows really are the
        # wider ones — that is the whole content of the note.
        assert verdict.held_out_spread is not None
        assert verdict.tested_spread is not None
        assert math.isfinite(verdict.held_out_spread)
        assert math.isfinite(verdict.tested_spread)
        assert verdict.held_out_spread > 0.0
        assert verdict.tested_spread > 3.0 * verdict.held_out_spread
        # It names how many rows set the scale, so the panel's advice points at
        # real rows rather than at the default.
        assert verdict.held_out_rows == PRIOR_SCALE_ROWS

    def test_a_saturated_start_that_is_bit_equal_reports_the_wider_window(self):
        """The held-out window grows while both arms are still exactly constant.

        The panel has to name the rows that actually set the scale, which is not
        always `PRIOR_SCALE_ROWS`: `_held_out_rows` keeps holding rows out while
        both arms are still bit-equal, so the count is only knowable per run.
        """
        rng = random.Random(7)
        from soup_cli.utils.ab_test import PRIOR_SCALE_ROWS

        control = [1.0] * 7 + [0.999] + [
            round(rng.gauss(0.82, 0.05), 3) for _ in range(60)
        ]
        treatment = [1.0] * 7 + [0.9991] + [
            round(rng.gauss(0.82, 0.05), 3) for _ in range(60)
        ]

        verdict = _step(control, treatment)

        assert verdict.decision == "continue", verdict.decision
        assert verdict.accept_held is True
        # Wider than PRIOR_SCALE_ROWS: the report must not hardcode the default.
        assert verdict.held_out_rows == 8
        assert verdict.held_out_rows != PRIOR_SCALE_ROWS
        assert verdict.tested_spread > 3.0 * verdict.held_out_spread

    def test_the_note_is_caused_by_the_hold_not_by_the_data(self, monkeypatch):
        """Remove the hold and the same rows accept: the field tracks the hold.

        Without this the first test would pass on any `continue` that happened to
        carry the two spreads, which is what a hardcoded note would do.
        """
        from soup_cli.utils import ab_test

        control, treatment = _saturated_warm_up(40, seed=2)

        monkeypatch.setattr(ab_test, "ACCEPT_HOLD_SPREAD_RATIO", math.inf)
        verdict = _step(control, treatment)

        assert verdict.decision == "accept_h0", verdict.decision
        assert verdict.accept_held is False

    def test_more_rows_of_the_same_kind_do_not_release_it(self):
        """The reason the note matters: the run stays held as rows arrive.

        The held-out rows are fixed, so the tested spread converges and the ratio
        only ever grows. A note that advised "collect more samples" would be
        advising the one thing that provably does not help.

        The invariant is that the hold never *releases* once it has fired, not
        that it fires on literally every peek: a peek can also sit at `continue`
        for want of tested rows before the accept branch is reached at all.
        """
        control, treatment = _saturated_warm_up(200, seed=2)

        rows = [
            _step(control[:n], treatment[:n])
            for n in range(7, len(control) + 1)
        ]
        assert all(v.decision == "continue" for v in rows)
        assert any(v.accept_held for v in rows), "the hold never fired"
        # From the first held peek on, every later peek is still held.
        first_held = next(i for i, v in enumerate(rows) if v.accept_held)
        assert all(v.accept_held for v in rows[first_held:]), [
            (rows[i].n_control, v.decision, v.accept_held)
            for i, v in enumerate(rows)
            if i >= first_held and not v.accept_held
        ]
        # Still held with the whole dataset in hand — the issue's "no decision in
        # 300 rows", here at 200.
        last = rows[-1]
        assert last.accept_held is True
        assert last.tested_spread > 3.0 * last.held_out_spread

    def test_an_unheld_continue_carries_no_note(self):
        """A `continue` that is merely short of evidence borrows no explanation.

        The control for the false explanation: ordinary Gaussian rows where the
        held-out and tested spreads agree must not report a hold, and must carry
        no spreads either, so a consumer reading the field cannot print the note
        for a run it does not describe.
        """
        rng = random.Random(1_524)
        control = [rng.gauss(0.0, 1.0) for _ in range(30)]
        treatment = [rng.gauss(0.0, 1.0) for _ in range(30)]

        verdict = _step(control, treatment, effect_size=2.0)

        assert verdict.decision == "accept_h0", verdict.decision
        assert verdict.accept_held is False
        assert verdict.held_out_spread is None
        assert verdict.tested_spread is None

    def test_a_reject_is_never_held(self):
        """A `reject_h0` clears the boundary above the hold, so it cannot carry one."""
        rng = random.Random(1_524)
        control = [rng.gauss(0.0, 1.0) for _ in range(40)]
        treatment = [rng.gauss(3.0, 1.0) for _ in range(40)]

        verdict = _step(control, treatment, effect_size=0.1)

        assert verdict.decision == "reject_h0", verdict.decision
        assert verdict.accept_held is False
        assert verdict.held_out_spread is None
        assert verdict.held_out_rows == 0

    def test_too_few_tested_rows_is_continue_without_a_note(self):
        """Still inside the warm-up the verdict has no tested rows to compare."""
        verdict = _step([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0] * 7)

        assert verdict.decision == "continue"
        assert verdict.accept_held is False

    def test_the_note_is_the_holds_own_ratio(self, monkeypatch):
        """A different ratio changes which runs are held, not the note's rule."""
        from soup_cli.utils import ab_test

        control, treatment = _saturated_warm_up(40, seed=2)

        monkeypatch.setattr(ab_test, "ACCEPT_HOLD_SPREAD_RATIO", 1e9)
        verdict = _step(control, treatment)

        # Same verdict object shape, hold off: accepts, and says nothing.
        assert verdict.decision == "accept_h0"
        assert verdict.accept_held is False


# ---------------------------------------------------------------------------
# The verdict field is validated like the rest
# ---------------------------------------------------------------------------


class TestAcceptHeldValidation:
    """`accept_held` cannot be set on a decision the hold does not produce."""

    def test_only_continue_can_be_held(self):
        from soup_cli.utils.ab_test import MsprtVerdict

        for decision in ("accept_h0", "reject_h0"):
            kwargs = {"direction": "better"} if decision == "reject_h0" else {}
            with pytest.raises(ValueError, match="continue"):
                MsprtVerdict(
                    decision=decision,
                    log_likelihood_ratio=0.0,
                    n_control=10,
                    n_treatment=10,
                    mean_control=1.0,
                    mean_treatment=1.0,
                    accept_held=True,
                    **kwargs,
                )

    def test_a_held_verdict_must_carry_both_spreads(self):
        """`accept_held` without the numbers behind it would be an unfalsifiable note."""
        from soup_cli.utils.ab_test import MsprtVerdict

        for missing in ("held_out_spread", "tested_spread"):
            fields = {"held_out_spread": 0.01, "tested_spread": 0.5}
            fields[missing] = None
            with pytest.raises(ValueError, match=missing):
                MsprtVerdict(
                    decision="continue",
                    log_likelihood_ratio=-2.0,
                    n_control=20,
                    n_treatment=20,
                    mean_control=1.0,
                    mean_treatment=1.0,
                    accept_held=True,
                    **fields,
                )

    def test_spreads_are_non_negative_and_finite(self):
        from soup_cli.utils.ab_test import MsprtVerdict

        for bad in (-0.1, math.inf, math.nan):
            with pytest.raises(ValueError, match="spread"):
                MsprtVerdict(
                    decision="continue",
                    log_likelihood_ratio=-2.0,
                    n_control=20,
                    n_treatment=20,
                    mean_control=1.0,
                    mean_treatment=1.0,
                    accept_held=True,
                    held_out_spread=0.01,
                    tested_spread=bad,
                )

    def test_spreads_without_a_hold_are_refused(self):
        """No hold, no explanation: the two must not float free of each other."""
        from soup_cli.utils.ab_test import MsprtVerdict

        with pytest.raises(ValueError, match="accept_held"):
            MsprtVerdict(
                decision="continue",
                log_likelihood_ratio=-2.0,
                n_control=20,
                n_treatment=20,
                mean_control=1.0,
                mean_treatment=1.0,
                held_out_spread=0.01,
                tested_spread=0.5,
            )

    def test_the_verdict_is_still_frozen(self):
        """The new fields do not cost the verdict its immutability."""
        from soup_cli.utils.ab_test import MsprtVerdict

        verdict = MsprtVerdict(
            decision="continue",
            log_likelihood_ratio=-2.0,
            n_control=20,
            n_treatment=20,
            mean_control=1.0,
            mean_treatment=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            verdict.accept_held = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


def _write_jsonl(path, control, treatment, metric="judge_score"):
    rows = []
    for c, t in zip(control, treatment):
        rows.append(json.dumps({"arm": "control", metric: c}))
        rows.append(json.dumps({"arm": "treatment", metric: t}))
    path.write_text("\n".join(rows), encoding="utf-8")


def test_the_panel_names_the_hold_and_its_ratio(tmp_path, monkeypatch):
    """One actionable line: the ratio, and what to do about it."""
    from soup_cli.cli import app
    from soup_cli.utils.ab_test import MsprtConfig, run_msprt

    monkeypatch.chdir(tmp_path)
    control, treatment = _saturated_warm_up(40, seed=2)
    inp = tmp_path / "ab.jsonl"
    _write_jsonl(inp, control, treatment)
    verdict = run_msprt(str(inp), config=MsprtConfig(metric="judge_score"))
    assert verdict.accept_held is True, verdict  # the premise

    result = runner.invoke(app, ["ab", "--input", str(inp), "--metric", "judge_score"])
    assert result.exit_code == 0, (result.output, repr(result.exception))

    out = _plain(result.output).lower()
    ratio = verdict.tested_spread / verdict.held_out_spread
    assert f"held back: the tested rows spread {ratio:.3g} times" in out, out
    assert (
        f"(held_out_spread {verdict.held_out_spread:.4g}, "
        f"tested_spread {verdict.tested_spread:.4g}, limit 3x)"
    ) in out, out
    # It must not ALSO print the advice it replaces.
    assert "insufficient evidence" not in out, out


def test_the_panel_of_an_unheld_continue_is_unchanged(tmp_path, monkeypatch):
    """The control: an unheld continue past the warm-up still prints the old line."""
    from soup_cli.cli import app
    from soup_cli.utils.ab_test import MsprtConfig, run_msprt

    monkeypatch.chdir(tmp_path)
    rng = random.Random(0)
    control = [round(rng.gauss(0.80, 0.05), 3) for _ in range(10)]
    treatment = [round(rng.gauss(0.80, 0.05), 3) for _ in range(10)]
    inp = tmp_path / "ab.jsonl"
    _write_jsonl(inp, control, treatment)
    verdict = run_msprt(str(inp), config=MsprtConfig(metric="judge_score"))
    # The premise: past the warm-up, undecided, and not held.
    assert (verdict.decision, verdict.accept_held) == ("continue", False), verdict

    result = runner.invoke(app, ["ab", "--input", str(inp), "--metric", "judge_score"])
    assert result.exit_code == 0, (result.output, repr(result.exception))

    out = _plain(result.output).lower()
    assert "insufficient evidence. collect more samples and re-run." in out, out
    assert "held back" not in out, out
    assert "accept_held" not in out, out


# ---------------------------------------------------------------------------
# JSON output stays backward compatible
# ---------------------------------------------------------------------------


def test_the_verdict_fields_are_added_not_renamed():
    """Every field #1265 shipped is still there, and the new ones are additive."""
    import dataclasses

    from soup_cli.utils.ab_test import MsprtVerdict

    names = {f.name for f in dataclasses.fields(MsprtVerdict)}
    assert {
        "decision",
        "log_likelihood_ratio",
        "n_control",
        "n_treatment",
        "mean_control",
        "mean_treatment",
        "direction",
    } <= names
    assert {"accept_held", "held_out_spread", "tested_spread", "held_out_rows"} <= names


def test_the_webhook_payload_gains_the_keys_without_losing_any(tmp_path, monkeypatch):
    """The POSTed verdict gains keys; every key #1265 shipped is still there.

    Run through the real command so the test fails if `ab.py` drops or renames a
    key, rather than only checking a dict the test wrote itself.
    """
    from soup_cli.cli import app
    from soup_cli.commands import ab as ab_cmd

    sent: list[dict] = []
    monkeypatch.setattr(
        ab_cmd, "emit_webhooks",
        lambda slack, discord, *, payload, console: sent.append(payload),
    )

    monkeypatch.chdir(tmp_path)
    rng = random.Random(1_524)
    control = [rng.gauss(0.0, 1.0) for _ in range(40)]
    treatment = [rng.gauss(3.0, 1.0) for _ in range(40)]
    inp = tmp_path / "ab.jsonl"
    _write_jsonl(inp, control, treatment)

    result = runner.invoke(
        app,
        [
            "ab", "--input", str(inp), "--metric", "judge_score",
            "--slack-url", "https://hooks.slack.com/services/T/B/X",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert len(sent) == 1, sent

    payload = sent[0]
    # Every key that was there before, unchanged. The four #1524 keys are NOT
    # here on purpose: a held run is a `continue` and the webhook only fires on
    # a terminal decision, so they could only ever be false / null.
    assert set(payload) >= {
        "command",
        "metric",
        "decision",
        "direction",
        "log_likelihood_ratio",
        "n_control",
        "n_treatment",
        "mean_control",
        "mean_treatment",
    }
    # Nothing renamed: the old names must not have become aliases.
    assert not {"spread_ratio", "held", "held_back"} & set(payload)
    # No hold keys: there is nothing to hold back on a terminal decision.
    assert not {"accept_held", "held_out_spread", "tested_spread", "held_out_rows"} & set(payload)
    assert payload["decision"] == "reject_h0"
    # Still JSON-serialisable end to end.
    json.dumps(payload)


def test_run_msprt_reports_the_hold_from_a_file(tmp_path, monkeypatch):
    """The JSONL path carries the note too, not just `msprt_step`."""
    from soup_cli.utils.ab_test import MsprtConfig, run_msprt

    monkeypatch.chdir(tmp_path)
    control, treatment = _saturated_warm_up(40, seed=2)
    inp = tmp_path / "ab.jsonl"
    _write_jsonl(inp, control, treatment)

    verdict = run_msprt(str(inp), config=MsprtConfig(metric="judge_score"))

    assert verdict.decision == "continue"
    assert verdict.accept_held is True
    assert verdict.tested_spread > 3.0 * verdict.held_out_spread
