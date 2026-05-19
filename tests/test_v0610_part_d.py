"""Tests for v0.61.0 Part D — Sequential edit governor.

Coverage:
- ``NormBlowupPolicy`` frozen dataclass + bounds validation.
- ``classify_norm_blowup`` OK / WARN / BLOWUP taxonomy.
- ``governor_recommend_method`` auto-switches method when blowup detected.
- ``EditGovernor`` stateful tracker (per-base-model edit count + last verdict).
- ``GovernedEditError`` for refusals.
"""

from __future__ import annotations

import dataclasses

import pytest


class TestModuleSurface:
    def test_imports(self):
        from soup_cli.utils.edit_governor import (
            DEFAULT_BLOWUP_POLICY,
            VERDICTS,
            EditGovernor,
            GovernedEditError,
            NormBlowupPolicy,
            classify_norm_blowup,
            governor_recommend_method,
        )
        assert callable(classify_norm_blowup)
        assert callable(governor_recommend_method)
        assert isinstance(VERDICTS, tuple)
        assert dataclasses.is_dataclass(NormBlowupPolicy)
        # GovernedEditError is an exception type, not a dataclass.
        assert issubclass(GovernedEditError, Exception)
        assert isinstance(DEFAULT_BLOWUP_POLICY, NormBlowupPolicy)
        assert callable(EditGovernor)


class TestNormBlowupPolicy:
    def test_defaults(self):
        from soup_cli.utils.edit_governor import NormBlowupPolicy

        p = NormBlowupPolicy()
        assert p.warn_threshold > 0
        assert p.blowup_threshold > p.warn_threshold
        assert p.max_sequential_edits > 0
        assert p.auto_switch_at >= 0

    def test_frozen(self):
        from soup_cli.utils.edit_governor import NormBlowupPolicy

        p = NormBlowupPolicy()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.warn_threshold = 999.0  # type: ignore

    def test_invalid_bounds(self):
        from soup_cli.utils.edit_governor import NormBlowupPolicy

        with pytest.raises(ValueError):
            NormBlowupPolicy(warn_threshold=10.0, blowup_threshold=5.0)

        with pytest.raises(ValueError):
            NormBlowupPolicy(warn_threshold=-1.0)

        with pytest.raises(ValueError):
            NormBlowupPolicy(max_sequential_edits=0)

    def test_bool_rejected(self):
        from soup_cli.utils.edit_governor import NormBlowupPolicy

        with pytest.raises(TypeError):
            NormBlowupPolicy(warn_threshold=True)  # type: ignore

        with pytest.raises(TypeError):
            NormBlowupPolicy(max_sequential_edits=True)  # type: ignore

    def test_non_finite_rejected(self):
        from soup_cli.utils.edit_governor import NormBlowupPolicy

        with pytest.raises(ValueError):
            NormBlowupPolicy(warn_threshold=float("nan"))

        with pytest.raises(ValueError):
            NormBlowupPolicy(blowup_threshold=float("inf"))


class TestClassifyNormBlowup:
    def test_ok_band(self):
        from soup_cli.utils.edit_governor import classify_norm_blowup

        assert classify_norm_blowup(0.01) == "OK"
        assert classify_norm_blowup(0.0) == "OK"

    def test_warn_band(self):
        from soup_cli.utils.edit_governor import (
            DEFAULT_BLOWUP_POLICY,
            classify_norm_blowup,
        )

        warn = DEFAULT_BLOWUP_POLICY.warn_threshold
        assert classify_norm_blowup(warn) == "WARN"
        # Above warn but below blowup.
        midpoint = (DEFAULT_BLOWUP_POLICY.warn_threshold +
                    DEFAULT_BLOWUP_POLICY.blowup_threshold) / 2
        assert classify_norm_blowup(midpoint) == "WARN"

    def test_blowup_band(self):
        from soup_cli.utils.edit_governor import (
            DEFAULT_BLOWUP_POLICY,
            classify_norm_blowup,
        )

        assert classify_norm_blowup(DEFAULT_BLOWUP_POLICY.blowup_threshold) == "BLOWUP"
        assert classify_norm_blowup(100.0) == "BLOWUP"

    def test_bool_rejected(self):
        from soup_cli.utils.edit_governor import classify_norm_blowup

        with pytest.raises(TypeError):
            classify_norm_blowup(True)

    def test_non_finite_rejected(self):
        from soup_cli.utils.edit_governor import classify_norm_blowup

        with pytest.raises(ValueError):
            classify_norm_blowup(float("nan"))

        with pytest.raises(ValueError):
            classify_norm_blowup(float("inf"))

    def test_negative_rejected(self):
        from soup_cli.utils.edit_governor import classify_norm_blowup

        with pytest.raises(ValueError):
            classify_norm_blowup(-1.0)


class TestGovernorRecommendMethod:
    def test_no_switch_below_threshold(self):
        from soup_cli.utils.edit_governor import governor_recommend_method

        result = governor_recommend_method(
            current_method="rome",
            edit_count=1,
            norm_delta=0.0,
        )
        assert result.method == "rome"
        assert result.switched is False

    def test_switch_at_auto_switch_count(self):
        from soup_cli.utils.edit_governor import governor_recommend_method

        result = governor_recommend_method(
            current_method="rome",
            edit_count=15,  # above default auto_switch_at=10
            norm_delta=0.0,
        )
        # ROME should auto-switch to AlphaEdit at the count boundary.
        assert result.method == "alphaedit"
        assert result.switched is True

    def test_blowup_forces_switch(self):
        from soup_cli.utils.edit_governor import governor_recommend_method

        result = governor_recommend_method(
            current_method="rome",
            edit_count=1,
            norm_delta=100.0,  # blowup
        )
        assert result.method == "alphaedit"
        assert result.switched is True

    def test_alphaedit_no_further_switch(self):
        from soup_cli.utils.edit_governor import governor_recommend_method

        # AlphaEdit is already the survival-mode method — no further switch.
        result = governor_recommend_method(
            current_method="alphaedit",
            edit_count=15,
            norm_delta=0.0,
        )
        assert result.method == "alphaedit"
        assert result.switched is False

    def test_invalid_method_rejected(self):
        from soup_cli.utils.edit_governor import governor_recommend_method

        with pytest.raises(ValueError):
            governor_recommend_method(
                current_method="zzz",
                edit_count=1,
                norm_delta=0.0,
            )

    def test_bool_edit_count_rejected(self):
        from soup_cli.utils.edit_governor import governor_recommend_method

        with pytest.raises(TypeError):
            governor_recommend_method(
                current_method="rome",
                edit_count=True,  # type: ignore
                norm_delta=0.0,
            )

    def test_negative_edit_count_rejected(self):
        from soup_cli.utils.edit_governor import governor_recommend_method

        with pytest.raises(ValueError):
            governor_recommend_method(
                current_method="rome",
                edit_count=-1,
                norm_delta=0.0,
            )

    def test_negative_norm_delta_rejected(self):
        from soup_cli.utils.edit_governor import governor_recommend_method

        with pytest.raises(ValueError):
            governor_recommend_method(
                current_method="rome",
                edit_count=1,
                norm_delta=-1.0,
            )

    def test_auto_switch_boundary_exact(self):
        """Review L5 — exact `auto_switch_at=10` should switch."""
        from soup_cli.utils.edit_governor import (
            DEFAULT_BLOWUP_POLICY,
            governor_recommend_method,
        )

        # At exactly auto_switch_at, ROME switches.
        result = governor_recommend_method(
            current_method="rome",
            edit_count=DEFAULT_BLOWUP_POLICY.auto_switch_at,
            norm_delta=0.0,
        )
        assert result.method == "alphaedit"
        assert result.switched is True

    def test_auto_switch_below_boundary_no_switch(self):
        """Review L5 — one below auto_switch_at does NOT switch."""
        from soup_cli.utils.edit_governor import (
            DEFAULT_BLOWUP_POLICY,
            governor_recommend_method,
        )

        result = governor_recommend_method(
            current_method="rome",
            edit_count=DEFAULT_BLOWUP_POLICY.auto_switch_at - 1,
            norm_delta=0.0,
        )
        assert result.method == "rome"
        assert result.switched is False


class TestEditGovernor:
    def test_construct(self):
        from soup_cli.utils.edit_governor import EditGovernor

        g = EditGovernor(base_model="meta-llama/Llama-3.1-8B")
        assert g.edit_count == 0

    def test_record_edit(self):
        from soup_cli.utils.edit_governor import EditGovernor

        g = EditGovernor(base_model="meta-llama/Llama-3.1-8B")
        g.record_edit(method="rome", norm_delta=0.0)
        assert g.edit_count == 1

    def test_refuses_above_max_edits(self):
        from soup_cli.utils.edit_governor import EditGovernor, GovernedEditError

        g = EditGovernor(
            base_model="meta-llama/Llama-3.1-8B",
            max_sequential_edits=2,
        )
        g.record_edit(method="rome", norm_delta=0.0)
        g.record_edit(method="rome", norm_delta=0.0)
        with pytest.raises(GovernedEditError, match="max_sequential"):
            g.check_can_edit()

    def test_blowup_blocks_further(self):
        from soup_cli.utils.edit_governor import EditGovernor, GovernedEditError

        g = EditGovernor(base_model="meta-llama/Llama-3.1-8B")
        g.record_edit(method="rome", norm_delta=100.0)  # blowup
        with pytest.raises(GovernedEditError, match="blowup"):
            g.check_can_edit()

    def test_empty_base_rejected(self):
        from soup_cli.utils.edit_governor import EditGovernor

        with pytest.raises(ValueError):
            EditGovernor(base_model="")

    def test_null_byte_base_rejected(self):
        from soup_cli.utils.edit_governor import EditGovernor

        with pytest.raises(ValueError):
            EditGovernor(base_model="b\x00")

    def test_invalid_method_in_record_rejected(self):
        from soup_cli.utils.edit_governor import EditGovernor

        g = EditGovernor(base_model="b")
        with pytest.raises(ValueError):
            g.record_edit(method="zzz", norm_delta=0.0)

    def test_recommend_next(self):
        from soup_cli.utils.edit_governor import EditGovernor

        g = EditGovernor(base_model="b")
        g.record_edit(method="rome", norm_delta=0.0)
        rec = g.recommend_next_method(current_method="rome")
        # After 1 ROME edit, still ROME.
        assert rec.method == "rome"

    def test_snapshot(self):
        from soup_cli.utils.edit_governor import EditGovernor

        g = EditGovernor(base_model="b")
        g.record_edit(method="rome", norm_delta=0.01)
        snap = g.snapshot()
        assert snap["edit_count"] == 1
        assert snap["last_method"] == "rome"
        assert snap["base_model"] == "b"
