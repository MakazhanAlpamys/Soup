"""Tests for v0.34.0 Part F — auto-profiling."""

from __future__ import annotations

import pytest

from soup_cli.utils.profiling import (
    DEFAULT_ACTIVE_STEPS,
    MAX_ACTIVE_STEPS,
    ProfilerSchedule,
    profile_training,
    resolve_trace_path,
)


class TestSchedule:
    def test_default(self):
        schedule = ProfilerSchedule.default()
        assert schedule.active == DEFAULT_ACTIVE_STEPS
        schedule.validate()  # should not raise

    def test_zero_active_rejected(self):
        with pytest.raises(ValueError, match="active"):
            ProfilerSchedule(1, 1, 0, 1).validate()

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            ProfilerSchedule(-1, 1, 5, 1).validate()

    def test_active_cap(self):
        with pytest.raises(ValueError, match="exceeds cap"):
            ProfilerSchedule(1, 1, MAX_ACTIVE_STEPS + 1, 1).validate()

    def test_frozen(self):
        schedule = ProfilerSchedule.default()
        with pytest.raises(Exception):
            schedule.active = 99  # type: ignore[misc]


class TestResolvePath:
    def test_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = resolve_trace_path(tmp_path / "out", "run_123_abc")
        assert path.name == "run_123_abc.trace.json"
        assert path.parent.name == "profiles"

    def test_dotdot_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            resolve_trace_path(tmp_path, "..")
        with pytest.raises(ValueError):
            resolve_trace_path(tmp_path, ".")

    def test_backslash_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            resolve_trace_path(tmp_path, "run\\x")

    def test_empty_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            resolve_trace_path(tmp_path, "")

    def test_run_id_with_slash_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="separator"):
            resolve_trace_path(tmp_path, "run/escaped")

    def test_run_id_null_byte_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            resolve_trace_path(tmp_path, "run\x00x")

    def test_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside"
        with pytest.raises(ValueError, match="not under cwd"):
            resolve_trace_path(outside, "run_x")


class TestProfileContext:
    def test_no_torch_yields_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Force the deferred import to raise ImportError.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in ("torch.profiler", "torch"):
                raise ImportError("forced")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with profile_training(
            output_dir=tmp_path / "out", run_id="run_x"
        ) as profiler:
            assert profiler is None

    def test_invalid_schedule_propagates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            with profile_training(
                output_dir=tmp_path / "out",
                run_id="run_x",
                schedule=ProfilerSchedule(1, 1, 0, 1),
            ):
                pass
