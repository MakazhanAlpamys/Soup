"""Tests for Issue #887: Distillation diagnostic warning on persistently broken teacher.

Ensures:
1. Persistently non-finite teacher logits produce a single diagnostic warning naming
   the teacher, consecutive step count, and likely causes.
2. Zero device-to-host synchronization (e.g. .item()) occurs on the per-step path.
3. Transient non-finite steps (e.g. normal GradScaler recovery) reset the counter on
   device and produce no warning.
4. The warning fires on the reproducer from #719 with a deliberately corrupted teacher.
5. The tracker warns, never raises, allowing training to continue.
"""

from __future__ import annotations

import logging

import pytest


def test_tracker_argument_validation() -> None:
    from soup_cli.trainer.distill import DistillNonfiniteTracker

    with pytest.raises(ValueError, match="threshold must be an int >= 1"):
        DistillNonfiniteTracker(threshold=0)

    with pytest.raises(ValueError, match="threshold must be an int >= 1"):
        DistillNonfiniteTracker(threshold=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="check_interval must be an int >= 1"):
        DistillNonfiniteTracker(check_interval=-1)

    with pytest.raises(ValueError, match="check_interval must be an int >= 1"):
        DistillNonfiniteTracker(check_interval=False)  # type: ignore[arg-type]


def test_zero_device_to_host_sync_on_intermediate_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that record_step does NOT invoke device-to-host sync on intermediate steps."""
    torch = pytest.importorskip("torch")
    from soup_cli.trainer.distill import DistillNonfiniteTracker

    tracker = DistillNonfiniteTracker(
        teacher_name="test-teacher",
        temperature=2.0,
        threshold=3,
        check_interval=5,
    )

    original_item = torch.Tensor.item

    def forbidden_item(self: torch.Tensor) -> float | int:
        raise AssertionError("Device-to-host sync (.item()) was called on a non-checking step!")

    # For steps 1 to 4 (step_count % 5 != 0), forbid any .item() sync.
    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)

    for _ in range(4):
        loss = torch.tensor(float("nan"))
        tracker.record_step(loss)

    assert not tracker.warned

    # On step 5 (step_count % 5 == 0), check_and_warn executes and synchronizes once.
    monkeypatch.setattr(torch.Tensor, "item", original_item)
    tracker.record_step(torch.tensor(float("nan")))
    assert tracker.warned

    # After warning once, the tracker is disarmed: even on multiples of 5, no sync happens.
    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    for _ in range(10):
        tracker.record_step(torch.tensor(float("nan")))


def test_transient_nonfinite_step_resets_counter_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A transient non-finite step (absorbed by GradScaler) produces no warning."""
    torch = pytest.importorskip("torch")
    from soup_cli.trainer.distill import DistillNonfiniteTracker

    tracker = DistillNonfiniteTracker(
        teacher_name="transient-teacher",
        threshold=3,
        check_interval=5,
    )

    with caplog.at_level(logging.WARNING, logger="soup_cli.trainer.distill"):
        # Step 1: NaN (transient overflow)
        tracker.record_step(torch.tensor(float("nan")))
        # Step 2: finite (GradScaler skipped step, backoff recovered)
        tracker.record_step(torch.tensor(1.5))
        # Step 3: finite
        tracker.record_step(torch.tensor(1.2))
        # Step 4: finite
        tracker.record_step(torch.tensor(1.1))
        # Step 5: finite (triggers check, counter is 0)
        tracker.record_step(torch.tensor(1.0))

        # Check end of training
        tracker.check_and_warn()

    assert not tracker.warned
    distill_records = [
        r
        for r in caplog.records
        if r.name == "soup_cli.trainer.distill" and r.levelno == logging.WARNING
    ]
    assert len(distill_records) == 0


def test_persistent_nonfinite_teacher_warns_once_with_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Persistent NaN logits trigger a single clear diagnostic warning naming causes."""
    torch = pytest.importorskip("torch")
    from rich.console import Console

    from soup_cli.trainer.distill import DistillNonfiniteTracker

    console = Console(record=True, width=120)
    tracker = DistillNonfiniteTracker(
        teacher_name="meta-llama/Llama-3.1-70B-Teacher",
        temperature=0.05,
        threshold=3,
        check_interval=5,
        console=console,
    )

    with caplog.at_level(logging.WARNING, logger="soup_cli.trainer.distill"):
        with pytest.warns(UserWarning) as warn_info:
            for _ in range(5):
                tracker.record_step(torch.tensor(float("nan")))

    assert tracker.warned

    # Exactly one UserWarning emitted
    assert len(warn_info) == 1
    user_warn_text = str(warn_info[0].message)
    assert "meta-llama/Llama-3.1-70B-Teacher" in user_warn_text
    assert "5 consecutive non-finite steps detected" in user_warn_text
    assert "distill_temperature=0.05" in user_warn_text
    assert "Teacher model saved/loaded in an unsupported dtype" in user_warn_text
    assert "Mismatched tokenizer" in user_warn_text
    assert "distill_temperature is set too small" in user_warn_text

    # Exactly one logger warning emitted
    distill_records = [
        r
        for r in caplog.records
        if r.name == "soup_cli.trainer.distill" and r.levelno == logging.WARNING
    ]
    assert len(distill_records) == 1
    assert "meta-llama/Llama-3.1-70B-Teacher" in distill_records[0].message
    assert "5 consecutive non-finite steps" in distill_records[0].message

    # Console output contains the warning panel
    rendered = console.export_text()
    assert "Persistently non-finite distillation loss" in rendered
    assert "meta-llama/Llama-3.1-70B-Teacher" in rendered
    assert "consecutive steps" in rendered


def test_reproducer_from_issue719_corrupted_teacher(caplog: pytest.LogCaptureFixture) -> None:
    """The warning fires on the reproducer from #719 with a deliberately corrupted teacher."""
    torch = pytest.importorskip("torch")
    from soup_cli.trainer.distill import DistillNonfiniteTracker, _compute_distill_term

    tracker = DistillNonfiniteTracker(
        teacher_name="corrupted-teacher-719",
        temperature=1.0,
        threshold=3,
        check_interval=3,
    )

    student = torch.zeros(1, 1, 2, requires_grad=True)
    teacher = torch.zeros(1, 1, 2)
    with torch.no_grad():
        # Deliberately corrupt teacher logits with NaN as in #719
        teacher[0, 0, 0] = float("nan")

    with caplog.at_level(logging.WARNING, logger="soup_cli.trainer.distill"):
        for _ in range(3):
            loss = _compute_distill_term(student, teacher, "forward_kl", temperature=1.0)
            assert not torch.isfinite(loss)
            tracker.record_step(loss)

    assert tracker.warned
    assert any("corrupted-teacher-719" in r.message for r in caplog.records)


def test_warn_on_train_end_when_steps_below_interval(caplog: pytest.LogCaptureFixture) -> None:
    """When training ends before a check_interval boundary, check_and_warn still fires."""
    torch = pytest.importorskip("torch")
    from soup_cli.trainer.distill import DistillNonfiniteTracker

    # threshold=3, but check_interval=10
    tracker = DistillNonfiniteTracker(
        teacher_name="short-run-teacher",
        threshold=3,
        check_interval=10,
    )

    # 4 consecutive non-finite steps (4 < 10, so check_interval hasn't fired yet)
    for _ in range(4):
        tracker.record_step(torch.tensor(float("nan")))

    assert not tracker.warned

    # Training ends: on_train_end triggers check_and_warn()
    with caplog.at_level(logging.WARNING, logger="soup_cli.trainer.distill"):
        warned = tracker.check_and_warn()

    assert warned
    assert tracker.warned
    assert any("short-run-teacher" in r.message for r in caplog.records)


def test_distill_trainer_wrapper_e2e_corrupted_teacher_warns_and_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """DistillTrainerWrapper.compute_loss warns on persistent corrupted teacher and never raises."""
    torch = pytest.importorskip("torch")
    from unittest.mock import MagicMock, patch

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.distill import DistillTrainerWrapper

    cfg = load_config_from_string("""
base: dummy-student
task: distill
data:
  train: dummy.jsonl
  format: chatml
output: ./out
training:
  teacher_model: broken-teacher-e2e
  distill_temperature: 0.1
""")

    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "<eos>"

    class StudentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Cfg", (), {"vocab_size": 16})()
            self.lin = torch.nn.Linear(16, 16)

        def forward(self, **kw):
            return type(
                "Out", (), {"logits": torch.randn(1, 4, 16, requires_grad=True)}
            )()

    class BrokenTeacherModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Cfg", (), {"vocab_size": 16})()
            self.lin = torch.nn.Linear(16, 16)

        def forward(self, **kw):
            # Persistently corrupted NaN teacher logits
            nan_logits = torch.full((1, 4, 16), float("nan"))
            return type("Out", (), {"logits": nan_logits})()

    dummy_row = {
        "input_ids": [1, 2, 3, 4],
        "labels": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
    }

    models = [BrokenTeacherModel(), StudentModel()]
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=lambda *a, **kw: models.pop() if models else StudentModel(),
        ),
        patch("peft.get_peft_model", side_effect=lambda m, c: m),
        patch(
            "soup_cli.utils.peft_wiring.resolve_lora_target_modules",
            return_value=["lin"],
        ),
        patch(
            "soup_cli.data.sft_format.build_format_row",
            return_value=lambda r: dummy_row,
        ),
    ):
        wrapper = DistillTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": [{"dummy": 1}]})

    assert wrapper.nonfinite_tracker is not None
    assert wrapper.trainer.nonfinite_tracker is wrapper.nonfinite_tracker
    assert not wrapper.nonfinite_tracker.warned

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }

    with caplog.at_level(logging.WARNING, logger="soup_cli.trainer.distill"):
        for _ in range(5):
            # compute_loss must NOT raise an exception
            loss = wrapper.trainer.compute_loss(wrapper.model, batch)
            assert not torch.isfinite(loss)

    assert wrapper.nonfinite_tracker.warned
    assert any("broken-teacher-e2e" in r.message for r in caplog.records)

