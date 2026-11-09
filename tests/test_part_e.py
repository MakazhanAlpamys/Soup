"""Part E — v0.32.1 stability live (#56, #57, #58, #59) for v0.33.0.

Covers:
  - #56 run_lr_sweep — in-process LR-sweep loop with mocked model + DataLoader.
  - #57 SoupTrainerCallback._write_spike_recovery_hint — writes JSON hint
    when watchdog fires and loss_spike_recovery is enabled.
  - #58 SFTTrainerWrapper._resolve_mixed_precision — wires
    pick_mixed_precision into bf16/fp16 flags; preserves legacy default
    when auto flag is False.
  - #59 SoupTrainerCallback grad-accum advisory — fires once on threshold
    crossing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output (#1068: Rich colours numbers)."""
    return " ".join(_ANSI_RE.sub("", text).split())

# ---------------------------------------------------------------------------
# #56 — run_lr_sweep
# ---------------------------------------------------------------------------


class TestRunLRSweep:
    def test_empty_schedule_rejected(self):
        from soup_cli.utils.lr_finder import run_lr_sweep

        with pytest.raises(ValueError, match="schedule must be non-empty"):
            run_lr_sweep(
                model=MagicMock(), dataloader=iter([]),
                schedule=[], optimizer_factory=lambda p: MagicMock(),
            )

    def test_loop_records_loss_per_step(self):
        from soup_cli.utils.lr_finder import run_lr_sweep

        # Fake model returning a tensor-like loss
        def _fake_loss_value(value):
            obj = MagicMock()
            obj.detach = MagicMock(return_value=obj)
            obj.item = MagicMock(return_value=value)
            obj.backward = MagicMock(return_value=None)
            return obj

        loss_values = [3.0, 2.0, 1.5, 1.0]

        class FakeModel:
            def __init__(self):
                self._idx = 0

            def parameters(self):
                return []

            def __call__(self, **batch):
                value = loss_values[self._idx]
                self._idx += 1
                return {"loss": _fake_loss_value(value)}

        model = FakeModel()

        # Fake optimizer with mutable param_groups
        class FakeOptim:
            def __init__(self, _params):
                self.param_groups = [{"lr": 0.0}]

            def zero_grad(self, set_to_none: bool = False):  # noqa: ARG002
                pass

            def step(self):
                pass

        dl = iter([{"input_ids": MagicMock()}] * 4)
        schedule = [1e-6, 1e-5, 1e-4, 1e-3]

        losses = run_lr_sweep(
            model=model, dataloader=dl, schedule=schedule,
            optimizer_factory=FakeOptim,
        )
        assert losses == loss_values

    def test_diverged_loss_breaks_loop(self):
        from soup_cli.utils.lr_finder import run_lr_sweep

        loss_values = [3.0, float("inf"), 1.0]

        def _wrap(value):
            obj = MagicMock()
            obj.detach = MagicMock(return_value=obj)
            obj.item = MagicMock(return_value=value)
            obj.backward = MagicMock(return_value=None)
            return obj

        class FakeModel:
            def __init__(self):
                self._idx = 0

            def parameters(self):
                return []

            def __call__(self, **batch):
                value = loss_values[self._idx]
                self._idx += 1
                return {"loss": _wrap(value)}

        class FakeOptim:
            def __init__(self, _params):
                self.param_groups = [{"lr": 0.0}]

            def zero_grad(self, set_to_none: bool = False):  # noqa: ARG002
                pass

            def step(self):
                pass

        dl = iter([{"x": MagicMock()}] * 3)
        losses = run_lr_sweep(
            model=FakeModel(), dataloader=dl,
            schedule=[1e-6, 1e-5, 1e-4],
            optimizer_factory=FakeOptim,
        )
        # Loop terminates after the inf — only the first finite loss kept.
        assert losses == [3.0]


# ---------------------------------------------------------------------------
# #58 — auto mixed-precision push
# ---------------------------------------------------------------------------


class TestResolveMixedPrecision:
    def test_auto_flag_off_asks_the_card(self, monkeypatch):
        """Was ``test_auto_flag_off_preserves_legacy``, asserting bf16 on any
        CUDA device. That "legacy" is what transformers refuses outright on a
        pre-Ampere card — *"Your setup doesn't support bf16/gpu"* — so every
        run on a T4 or P100 died before step 0 (#385). The contract is now: bf16
        when the card has it, fp16 when it does not. Both directions asserted,
        because a resolver hardcoded either way would satisfy only one."""
        import torch

        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cuda"
        tcfg = SimpleNamespace(auto_mixed_precision=False)

        class _Cuda:
            def __init__(self, bf16):
                self._bf16 = bf16

            def is_available(self):
                return True

            def is_bf16_supported(self, including_emulation: bool = True):
                # Mirrors the real signature: the bare call is permissive
                # and says True on a T4 through emulation (#385 follow-up).
                return self._bf16 if not including_emulation else True

            def get_device_capability(self, device=None):
                return (8, 0) if self._bf16 else (7, 5)

        monkeypatch.setattr(torch, "cuda", _Cuda(True))
        assert wrapper._resolve_mixed_precision(tcfg, "any") == (True, False)

        monkeypatch.setattr(torch, "cuda", _Cuda(False))
        assert wrapper._resolve_mixed_precision(tcfg, "any") == (False, True)

    def test_auto_flag_off_cpu(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cpu"
        tcfg = SimpleNamespace(auto_mixed_precision=False)
        bf16, fp16 = wrapper._resolve_mixed_precision(tcfg, "any")
        assert (bf16, fp16) == (False, False)

    def test_auto_flag_cpu_returns_no(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cpu"
        tcfg = SimpleNamespace(auto_mixed_precision=True)
        assert wrapper._resolve_mixed_precision(tcfg, "any") == (False, False)

    def test_auto_flag_picks_bf16_on_ampere(self, monkeypatch):
        """Ampere (cc 8.6) + non-quirk model → bf16."""
        import torch

        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cuda"
        monkeypatch.setattr(
            torch.cuda, "get_device_capability",
            lambda *_a, **_k: (8, 6),
            raising=False,
        )
        tcfg = SimpleNamespace(auto_mixed_precision=True)
        bf16, fp16 = wrapper._resolve_mixed_precision(tcfg, "neutral-model")
        assert (bf16, fp16) == (True, False)

    def test_auto_flag_picks_fp16_for_qwen2_on_ampere(self, monkeypatch):
        import torch

        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cuda"
        monkeypatch.setattr(
            torch.cuda, "get_device_capability",
            lambda *_a, **_k: (8, 6),
            raising=False,
        )
        tcfg = SimpleNamespace(auto_mixed_precision=True)
        bf16, fp16 = wrapper._resolve_mixed_precision(
            tcfg, "Qwen/Qwen2-7B-Instruct",
        )
        assert (bf16, fp16) == (False, True)


# ---------------------------------------------------------------------------
# #57 — spike recovery hint
# ---------------------------------------------------------------------------


def _make_callback(tmp_path, **kwargs):
    from soup_cli.monitoring.callback import SoupTrainerCallback

    display = MagicMock()
    return SoupTrainerCallback(
        display=display,
        tracker=None,
        run_id="t",
        output_dir=str(tmp_path),
        **kwargs,
    )


class TestSpikeRecoveryHint:
    @pytest.mark.requires_symlink
    @pytest.mark.parametrize("outside_cwd", [False, True], ids=["inside", "outside"])
    @pytest.mark.parametrize("existing_target", [False, True], ids=["dangling", "existing"])
    def test_linked_hint_preserves_target_and_recovery_budget(
        self, tmp_path, monkeypatch, caplog, outside_cwd, existing_target,
    ):
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        target = (tmp_path if outside_cwd else work) / "target.json"
        original = b'{"attempts": 7, "untouched": true}\n'
        if existing_target:
            target.write_bytes(original)
        hint = work / "spike_recovery.json"
        hint.symlink_to(target)
        reads = []
        read_text = Path.read_text

        def track_read(path, *args, **kwargs):
            if path == hint:
                reads.append(path)
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", track_read)
        cb = _make_callback(work, spike_recovery=True, spike_recovery_max_attempts=2)
        args = SimpleNamespace(learning_rate=1e-3, output_dir=str(work))

        cb._write_spike_recovery_hint(args, loss=10.0)

        assert hint.is_symlink()
        assert reads == [], "linked attempt counts must not be read"
        assert target.exists() is existing_target
        if existing_target:
            assert target.read_bytes() == original
        assert cb._spike_recovery_attempts == 0
        assert "Failed to write spike recovery hint" in caplog.text
        reason = "must stay under cwd" if outside_cwd else "must not be a symlink"
        assert "spike_recovery hint" in caplog.text and reason in caplog.text
        # Refusal must not spend an attempt: replacing the link with an ordinary
        # output permits the first recovery as usual.
        hint.unlink()
        cb._write_spike_recovery_hint(args, loss=10.0)
        data = json.loads(hint.read_text(encoding="utf-8"))
        assert data["attempts"] == 1
        assert data["should_recover"] is True

    @pytest.mark.requires_hardlink
    def test_hardlinked_hint_replacement_preserves_other_file(self, tmp_path, monkeypatch):
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        target = tmp_path / "previous-hint.json"
        original = b'{"attempts": 2}\n'
        target.write_bytes(original)
        hint = work / "spike_recovery.json"
        hint.hardlink_to(target)
        cb = _make_callback(work, spike_recovery=True, spike_recovery_max_attempts=2)
        args = SimpleNamespace(learning_rate=1e-3, output_dir=str(work))

        cb._write_spike_recovery_hint(args, loss=10.0)

        assert target.read_bytes() == original
        assert not hint.samefile(target)
        data = json.loads(hint.read_text(encoding="utf-8"))
        assert data["attempts"] == 3
        assert data["should_recover"] is False

    def test_failed_hint_replacement_preserves_file_and_attempts(
        self, tmp_path, monkeypatch, caplog,
    ):
        monkeypatch.chdir(tmp_path)
        hint = tmp_path / "spike_recovery.json"
        original = b'{"attempts": 1}\n'
        hint.write_bytes(original)
        cb = _make_callback(tmp_path, spike_recovery=True, spike_recovery_max_attempts=2)
        args = SimpleNamespace(learning_rate=1e-3, output_dir=str(tmp_path))

        def fail_replace(_source, _destination):
            raise PermissionError("synthetic hint replacement failure")

        with monkeypatch.context() as failure:
            failure.setattr("soup_cli.utils.paths.os.replace", fail_replace)
            cb._write_spike_recovery_hint(args, loss=10.0)

        assert hint.read_bytes() == original
        assert cb._spike_recovery_attempts == 0
        assert sorted(path.name for path in tmp_path.iterdir()) == ["spike_recovery.json"]
        assert "Failed to write spike recovery hint" in caplog.text
        cb._write_spike_recovery_hint(args, loss=10.0)
        data = json.loads(hint.read_text(encoding="utf-8"))
        assert data["attempts"] == 2
        assert data["should_recover"] is True

    def test_writes_hint_file(self, tmp_path, monkeypatch):
        # Containment guard requires output_dir to live under cwd.
        monkeypatch.chdir(tmp_path)
        cb = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=2,
            spike_recovery_lr_decay=0.5,
        )
        args = SimpleNamespace(
            learning_rate=1e-3, output_dir=str(tmp_path),
        )
        cb._write_spike_recovery_hint(args, loss=10.0)
        hint = tmp_path / "spike_recovery.json"
        assert hint.exists()
        data = json.loads(hint.read_text(encoding="utf-8"))
        assert data["previous_lr"] == pytest.approx(1e-3)
        assert data["recommended_lr"] == pytest.approx(5e-4)
        assert data["should_recover"] is True
        assert data["attempts"] == 1

    def test_attempts_counter_increments(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cb = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=3,
            spike_recovery_lr_decay=0.5,
        )
        args = SimpleNamespace(
            learning_rate=1e-3, output_dir=str(tmp_path),
        )
        cb._write_spike_recovery_hint(args, loss=10.0)
        cb._write_spike_recovery_hint(args, loss=10.0)
        data = json.loads((tmp_path / "spike_recovery.json").read_text())
        assert data["attempts"] == 2

    def test_fresh_cap_1_allows_one_recovery_and_refuses_second(self, tmp_path, monkeypatch):
        """Cap 1 permits exactly one recovery (prior attempts=0), refusing the second."""
        monkeypatch.chdir(tmp_path)
        args = SimpleNamespace(learning_rate=1e-3, output_dir=str(tmp_path))

        # Launch 1: fresh instance, prior count = 0 -> recovery permitted, attempts incremented to 1
        cb1 = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=1,
            spike_recovery_lr_decay=0.5,
        )
        cb1._write_spike_recovery_hint(args, loss=10.0)
        data1 = json.loads((tmp_path / "spike_recovery.json").read_text(encoding="utf-8"))
        assert data1["attempts"] == 1
        assert data1["should_recover"] is True, "First launch under cap 1 must permit recovery"

        # Launch 2 (re-launch): new instance reads persisted count 1 -> budget exhausted
        cb2 = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=1,
            spike_recovery_lr_decay=0.5,
        )
        cb2._write_spike_recovery_hint(args, loss=12.0)
        data2 = json.loads((tmp_path / "spike_recovery.json").read_text(encoding="utf-8"))
        assert data2["attempts"] == 2
        assert data2["should_recover"] is False, "Second launch under cap 1 must refuse recovery"

    def test_cap_2_two_allowed_attempts_across_new_instances(self, tmp_path, monkeypatch):
        """Under cap 2, both attempt 1 and attempt 2 are permitted; attempt 3 is refused."""
        monkeypatch.chdir(tmp_path)
        args = SimpleNamespace(learning_rate=1e-3, output_dir=str(tmp_path))

        # Launch 1: prior count = 0 -> permitted, persisted 1
        cb1 = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=2,
            spike_recovery_lr_decay=0.5,
        )
        cb1._write_spike_recovery_hint(args, loss=10.0)
        data1 = json.loads((tmp_path / "spike_recovery.json").read_text(encoding="utf-8"))
        assert data1["attempts"] == 1
        assert data1["should_recover"] is True, "First attempt under cap 2 must be permitted"

        # Launch 2: new instance reads persisted count 1 -> second attempt is permitted
        cb2 = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=2,
            spike_recovery_lr_decay=0.5,
        )
        cb2._write_spike_recovery_hint(args, loss=12.0)
        data2 = json.loads((tmp_path / "spike_recovery.json").read_text(encoding="utf-8"))
        assert data2["attempts"] == 2
        assert data2["should_recover"] is True, "Second attempt under cap 2 must be permitted"

        # Launch 3: new instance reads persisted count 2 -> budget exhausted, recovery refused
        cb3 = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=2,
            spike_recovery_lr_decay=0.5,
        )
        cb3._write_spike_recovery_hint(args, loss=14.0)
        data3 = json.loads((tmp_path / "spike_recovery.json").read_text(encoding="utf-8"))
        assert data3["attempts"] == 3
        assert data3["should_recover"] is False, "Third attempt under cap 2 must be refused"

    @pytest.mark.parametrize(
        "bad_payload",
        [
            "corrupted non-json",
            '{"attempts": null}',
            '{"attempts": []}',
            '{"attempts": {}}',
            '{"attempts": "not_an_int"}',
            '{"attempts": true}',
            '{"attempts": 1e999}',
            '{"attempts": -1e999}',
            '{"attempts": Infinity}',
            '{"attempts": -Infinity}',
            '{"attempts": NaN}',
        ],
    )
    def test_corrupt_and_malformed_hint_files_fall_back_gracefully(
        self, tmp_path, monkeypatch, bad_payload
    ):
        """Malformed JSON or invalid attempt types fall back safely to 0 without crashing."""
        monkeypatch.chdir(tmp_path)
        args = SimpleNamespace(learning_rate=1e-3, output_dir=str(tmp_path))
        (tmp_path / "spike_recovery.json").write_text(bad_payload, encoding="utf-8")

        cb = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=2,
            spike_recovery_lr_decay=0.5,
        )
        cb._write_spike_recovery_hint(args, loss=10.0)
        data = json.loads((tmp_path / "spike_recovery.json").read_text(encoding="utf-8"))
        assert data["attempts"] == 1
        assert data["should_recover"] is True

    def test_preserves_valid_positive_persisted_count(self, tmp_path, monkeypatch):
        """Valid persisted count from prior launch is safely parsed and incremented."""
        monkeypatch.chdir(tmp_path)
        args = SimpleNamespace(learning_rate=1e-3, output_dir=str(tmp_path))
        hint_path = tmp_path / "spike_recovery.json"
        hint_path.write_text(json.dumps({"attempts": 2, "max_attempts": 3}), encoding="utf-8")

        cb1 = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=3,
            spike_recovery_lr_decay=0.5,
        )
        cb1._write_spike_recovery_hint(args, loss=10.0)
        data1 = json.loads(hint_path.read_text(encoding="utf-8"))
        assert data1["attempts"] == 3
        assert data1["should_recover"] is True, "Prior count 2 with cap 3 must allow 3rd attempt"

        # Subsequent re-launch with prior count 3
        cb2 = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=3,
            spike_recovery_lr_decay=0.5,
        )
        cb2._write_spike_recovery_hint(args, loss=12.0)
        data2 = json.loads(hint_path.read_text(encoding="utf-8"))
        assert data2["attempts"] == 4
        assert data2["should_recover"] is False, "Prior count 3 with cap 3 must refuse recovery"

    def test_preserves_large_positive_integer_count_and_exhausts_cap(
        self, tmp_path, monkeypatch
    ):
        """Large integer count (10**309) is preserved without float conversion and exhausts cap."""
        monkeypatch.chdir(tmp_path)
        args = SimpleNamespace(learning_rate=1e-3, output_dir=str(tmp_path))
        hint_path = tmp_path / "spike_recovery.json"
        huge_attempts = 10**309
        hint_path.write_text(
            json.dumps({"attempts": huge_attempts, "max_attempts": 2}),
            encoding="utf-8",
        )

        cb = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=2,
            spike_recovery_lr_decay=0.5,
        )
        cb._write_spike_recovery_hint(args, loss=10.0)
        data = json.loads(hint_path.read_text(encoding="utf-8"))
        assert data["attempts"] == huge_attempts + 1
        assert data["should_recover"] is False, (
            "Prior large count must exhaust cap and refuse recovery"
        )

    @pytest.mark.parametrize(
        "bad_payload",
        [
            '{"attempts": null}',
            '{"attempts": 1e999}',
            '{"attempts": -1e999}',
            '{"attempts": Infinity}',
            '{"attempts": -Infinity}',
            '{"attempts": NaN}',
            '{"attempts": "not_an_int"}',
        ],
    )
    def test_watchdog_path_stops_training_with_bad_hints(
        self, tmp_path, monkeypatch, bad_payload
    ):
        """Watchdog path stops training safely when hint file is malformed or non-finite."""
        monkeypatch.chdir(tmp_path)
        hint_path = tmp_path / "spike_recovery.json"
        hint_path.write_text(bad_payload, encoding="utf-8")

        cb = _make_callback(
            tmp_path,
            loss_watchdog=True,
            loss_watchdog_threshold=5.0,
            loss_watchdog_patience=1,
            spike_recovery=True,
            spike_recovery_max_attempts=2,
            spike_recovery_lr_decay=0.5,
        )
        args = SimpleNamespace(learning_rate=1e-3, output_dir=str(tmp_path))
        state = SimpleNamespace(global_step=1, epoch=1.0)
        control = SimpleNamespace(should_training_stop=False)

        cb.on_log(args, state, control, logs={"loss": 10.0})

        assert control.should_training_stop is True, "Watchdog must stop training on loss spike"
        assert cb._watchdog_fired is True
        data = json.loads(hint_path.read_text(encoding="utf-8"))
        assert data["attempts"] == 1
        assert data["should_recover"] is True

    def test_watchdog_path_preserves_large_integer_count_and_exhausts_cap(
        self, tmp_path, monkeypatch
    ):
        """Watchdog path preserves large positive integer and exhausts recovery budget."""
        monkeypatch.chdir(tmp_path)
        hint_path = tmp_path / "spike_recovery.json"
        huge_attempts = 10**309
        hint_path.write_text(
            json.dumps({"attempts": huge_attempts, "max_attempts": 2}),
            encoding="utf-8",
        )

        cb = _make_callback(
            tmp_path,
            loss_watchdog=True,
            loss_watchdog_threshold=5.0,
            loss_watchdog_patience=1,
            spike_recovery=True,
            spike_recovery_max_attempts=2,
            spike_recovery_lr_decay=0.5,
        )
        args = SimpleNamespace(learning_rate=1e-3, output_dir=str(tmp_path))
        state = SimpleNamespace(global_step=1, epoch=1.0)
        control = SimpleNamespace(should_training_stop=False)

        cb.on_log(args, state, control, logs={"loss": 10.0})

        assert control.should_training_stop is True, "Watchdog must stop training on loss spike"
        assert cb._watchdog_fired is True
        data = json.loads(hint_path.read_text(encoding="utf-8"))
        assert data["attempts"] == huge_attempts + 1
        assert data["should_recover"] is False, "Exhausted count must keep recovery disabled"

    def test_disabled_when_strategy_not_set(self, tmp_path):
        cb = _make_callback(tmp_path, spike_recovery=False)
        args = SimpleNamespace(
            learning_rate=1e-3, output_dir=str(tmp_path),
        )
        cb._write_spike_recovery_hint(args, loss=10.0)
        # No hint file written.
        assert not (tmp_path / "spike_recovery.json").exists()

    def test_outside_cwd_skipped(self, tmp_path):
        """Containment guard: when output_dir is outside cwd, skip silently."""
        # No monkeypatch.chdir — tmp_path is outside the test's actual cwd.
        cb = _make_callback(
            tmp_path,
            spike_recovery=True,
        )
        args = SimpleNamespace(
            learning_rate=1e-3, output_dir=str(tmp_path),
        )
        cb._write_spike_recovery_hint(args, loss=10.0)
        # No hint written; no exception raised.
        assert not (tmp_path / "spike_recovery.json").exists()


# ---------------------------------------------------------------------------
# #59 — grad-accum advisory
# ---------------------------------------------------------------------------


class TestGradAccumAdvisory:
    def test_advise_fires_once_under_pressure(self, tmp_path, monkeypatch, capsys):
        cb = _make_callback(
            tmp_path,
            grad_accum_auto_tune=True,
            grad_accum_pressure_threshold=0.5,
            grad_accum_total_vram_gb=10.0,
            grad_accum_current_steps=1,
            grad_accum_current_batch=4,
        )

        # Mock torch presence + memory probe — high pressure (8 GB / 10 GB = 80%).
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.max_memory_allocated.return_value = 8 * (1024**3)
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is True

        # Second call is a no-op (one-shot).
        cb._grad_accum_monitor.observe = MagicMock()
        cb._maybe_advise_grad_accum()
        # Already advised, so monitor.observe shouldn't be called.
        cb._grad_accum_monitor.observe.assert_not_called()

    def test_no_advice_when_under_threshold(self, tmp_path, monkeypatch):
        cb = _make_callback(
            tmp_path,
            grad_accum_auto_tune=True,
            grad_accum_pressure_threshold=0.9,
            grad_accum_total_vram_gb=10.0,
            grad_accum_current_steps=1,
            grad_accum_current_batch=4,
        )
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.max_memory_allocated.return_value = 5 * (1024**3)
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is False

    def test_no_advice_when_disabled(self, tmp_path):
        cb = _make_callback(tmp_path, grad_accum_auto_tune=False)
        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is False

    def test_no_advice_when_cuda_unavailable(self, tmp_path, monkeypatch):
        cb = _make_callback(
            tmp_path,
            grad_accum_auto_tune=True,
            grad_accum_pressure_threshold=0.5,
            grad_accum_total_vram_gb=10.0,
            grad_accum_current_steps=1,
            grad_accum_current_batch=4,
        )
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is False

    # -- #1620: pressure is measured against the card, not a fixed total ----

    def test_advises_on_small_card_using_measured_total(self, tmp_path, monkeypatch, capsys):
        """An 8 GB card at 97.5% pressure with no explicit total advises.

        Against the old fixed 24 GB default this peak could never fire.
        """
        cb = _make_callback(
            tmp_path,
            grad_accum_auto_tune=True,
            grad_accum_pressure_threshold=0.92,
            grad_accum_current_steps=1,
            grad_accum_current_batch=8,
        )
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.current_device.return_value = 0
        fake_torch.cuda.max_memory_allocated.return_value = int(7.8 * (1024**3))
        fake_torch.cuda.get_device_properties.return_value.total_memory = 8 * (1024**3)
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is True
        assert cb._grad_accum_monitor.total_vram_gb == pytest.approx(8.0)
        # The advisory carries the measured numbers, not just the pair
        # (#1068: strip ANSI before reading numbers out of Rich output).
        assert "peak 7.8 of 8.0 GB (97%)" in _plain(capsys.readouterr().out)

    def test_no_advice_on_large_card_using_measured_total(self, tmp_path, monkeypatch):
        """An 80 GB card at 28.7% pressure with no explicit total stays quiet."""
        cb = _make_callback(
            tmp_path,
            grad_accum_auto_tune=True,
            grad_accum_pressure_threshold=0.92,
            grad_accum_current_steps=1,
            grad_accum_current_batch=8,
        )
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.current_device.return_value = 0
        fake_torch.cuda.max_memory_allocated.return_value = 23 * (1024**3)
        fake_torch.cuda.get_device_properties.return_value.total_memory = 80 * (1024**3)
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is False
        assert cb._grad_accum_monitor.total_vram_gb == pytest.approx(80.0)

    def test_trainer_built_callback_measures_the_card(self, tmp_path, monkeypatch):
        """Built exactly as the trainers build it (#802 kwargs), the advisory
        uses the card's total, not the old 24 GB default."""
        from soup_cli.monitoring.callback import (
            SoupTrainerCallback,
            soup_callback_kwargs,
        )

        tcfg = SimpleNamespace(
            grad_accum_auto_tune=True,
            gradient_accumulation_steps=1,
            batch_size=8,
        )
        cb = SoupTrainerCallback(
            display=MagicMock(), **soup_callback_kwargs(tcfg)
        )
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.current_device.return_value = 0
        fake_torch.cuda.max_memory_allocated.return_value = int(7.8 * (1024**3))
        fake_torch.cuda.get_device_properties.return_value.total_memory = 8 * (1024**3)
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is True
        assert cb._grad_accum_monitor.total_vram_gb == pytest.approx(8.0)

    def test_no_advice_when_device_properties_raise(self, tmp_path, monkeypatch):
        """An unreadable card total means no advice and no exception."""
        cb = _make_callback(
            tmp_path,
            grad_accum_auto_tune=True,
            grad_accum_pressure_threshold=0.92,
            grad_accum_current_steps=1,
            grad_accum_current_batch=8,
        )
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.current_device.return_value = 0
        fake_torch.cuda.max_memory_allocated.return_value = int(7.8 * (1024**3))
        fake_torch.cuda.get_device_properties.side_effect = RuntimeError("no props")
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is False
        assert cb._grad_accum_monitor is None

    # -- review follow-ups: the gate, the enabled guard, the device identity ----

    def _fake_cuda(self, current=0, peaks=None, totals=None):
        peaks, totals = peaks or {0: 7.8}, totals or {0: 8}
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.current_device.return_value = current
        fake_torch.cuda.max_memory_allocated.side_effect = (
            lambda device=0: int(peaks.get(device, 0.0) * (1024**3))
        )
        fake_torch.cuda.get_device_properties.side_effect = (
            lambda device: SimpleNamespace(total_memory=int(totals[device] * (1024**3)))
        )
        return fake_torch

    def test_on_log_drives_the_advisory(self, tmp_path, monkeypatch):
        """The advisory must fire through on_log, not only via a direct call:
        the on_log gate is what makes a real run advise at all."""
        cb = _make_callback(tmp_path, grad_accum_auto_tune=True,
                            grad_accum_pressure_threshold=0.92,
                            grad_accum_current_steps=1, grad_accum_current_batch=8)
        monkeypatch.setitem(__import__("sys").modules, "torch", self._fake_cuda())
        state = SimpleNamespace(global_step=1, epoch=0.1, max_steps=10, log_history=[])
        cb.on_log(SimpleNamespace(output_dir=str(tmp_path)), state, SimpleNamespace(),
                  logs={"loss": 1.0, "learning_rate": 1e-4})
        assert cb._grad_accum_advised is True
        assert cb._grad_accum_monitor.total_vram_gb == pytest.approx(8.0)

    def test_disabled_callback_stays_quiet_under_pressure(self, tmp_path, monkeypatch):
        """The enabled guard keeps a callback built with auto_tune off silent
        even when pressure is over the threshold."""
        cb = _make_callback(tmp_path, grad_accum_auto_tune=False,
                            grad_accum_current_batch=8)
        monkeypatch.setitem(__import__("sys").modules, "torch", self._fake_cuda())
        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is False
        assert cb._grad_accum_monitor is None

    def test_peak_and_total_come_from_the_current_device(self, tmp_path, monkeypatch):
        """The peak and the total must both come from the device the run uses:
        device 1 holds the 7.8 GB peak and the 8 GB card, device 0 is idle."""
        cb = _make_callback(tmp_path, grad_accum_auto_tune=True,
                            grad_accum_pressure_threshold=0.92,
                            grad_accum_current_steps=1, grad_accum_current_batch=8)
        monkeypatch.setitem(__import__("sys").modules, "torch", self._fake_cuda(
            current=1, peaks={0: 0.0, 1: 7.8}, totals={0: 80, 1: 8}))
        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is True
        assert cb._grad_accum_monitor.total_vram_gb == pytest.approx(8.0)

    def test_live_panel_reads_the_current_device(self, tmp_path, monkeypatch):
        """The live memory panel reports the device the run uses, not device 0."""
        cb = _make_callback(tmp_path)
        monkeypatch.setitem(__import__("sys").modules, "torch", self._fake_cuda(
            current=1, peaks={0: 0.0, 1: 7.8}, totals={0: 80, 1: 8}))
        state = SimpleNamespace(global_step=1, epoch=0.1, max_steps=10, log_history=[])
        cb.on_log(SimpleNamespace(output_dir=str(tmp_path)), state, SimpleNamespace(),
                  logs={"loss": 1.0, "learning_rate": 1e-4})
        assert cb.display.update.call_args.kwargs["gpu_mem"] == "7.8/8.0 GB"
