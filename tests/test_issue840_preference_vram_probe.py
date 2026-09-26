"""CPU contract tests for the preference-loss VRAM probe groundwork (#840)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_dpo_batch_is_stretched_to_the_configured_length():
    import torch

    from soup_cli.trainer.stream_setup import _stretch_preference_probe_batch

    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [1, 4, 5]]),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "completion_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
    }
    out = _stretch_preference_probe_batch(batch, seq_len=8)
    assert out["input_ids"].shape == (2, 8)
    assert out["attention_mask"].shape == (2, 8)
    assert out["completion_mask"].shape == (2, 8)
    assert out["attention_mask"][:, 3:].eq(1).all()
    assert out["completion_mask"][:, 3:].eq(1).all()


def test_orpo_and_simpo_full_sequences_are_stretched_but_prompt_is_not():
    import torch

    from soup_cli.trainer.stream_setup import _stretch_preference_probe_batch

    batch = {
        "chosen_input_ids": torch.tensor([[1, 2, 3, 4]]),
        "chosen_attention_mask": torch.ones((1, 4), dtype=torch.long),
        "chosen_labels": torch.tensor([[-100, -100, 3, 4]]),
        "rejected_input_ids": torch.tensor([[1, 2, 5]]),
        "rejected_attention_mask": torch.ones((1, 3), dtype=torch.long),
        "rejected_labels": torch.tensor([[-100, -100, 5]]),
        "prompt_input_ids": torch.tensor([[1, 2]]),
    }
    out = _stretch_preference_probe_batch(batch, seq_len=7)
    assert out["chosen_input_ids"].shape == (1, 7)
    assert out["rejected_input_ids"].shape == (1, 7)
    assert out["chosen_labels"][0, 4:].equal(out["chosen_input_ids"][0, 4:])
    assert out["rejected_labels"][0, 3:].equal(out["rejected_input_ids"][0, 3:])
    assert out["prompt_input_ids"].shape == (1, 2)


def test_kto_stretches_policy_and_kl_sequences():
    import torch

    from soup_cli.trainer.stream_setup import _stretch_preference_probe_batch

    batch = {}
    for prefix in ("", "KL_"):
        batch[prefix + "completion_input_ids"] = torch.tensor([[1, 2, 3], [1, 4, 5]])
        batch[prefix + "completion_attention_mask"] = torch.ones((2, 3), dtype=torch.long)
        batch[prefix + "completion_labels"] = torch.tensor([[-100, 2, 3], [-100, 4, 5]])

    batch["answer_input_ids"] = torch.tensor([[2, 3], [4, 5]])
    out = _stretch_preference_probe_batch(batch, seq_len=9)
    assert out["completion_input_ids"].shape == (2, 9)
    assert out["KL_completion_input_ids"].shape == (2, 9)
    assert out["completion_labels"][0, 3:].equal(out["completion_input_ids"][0, 3:])
    assert out["KL_completion_labels"][0, 3:].equal(out["KL_completion_input_ids"][0, 3:])
    assert out["answer_input_ids"].shape == (2, 2)


def test_overlength_batch_is_refused_instead_of_truncated():
    import torch

    from soup_cli.trainer.stream_setup import _stretch_preference_probe_batch

    with pytest.raises(ValueError, match="refusing to truncate"):
        _stretch_preference_probe_batch(
            {"input_ids": torch.ones((2, 9), dtype=torch.long)}, seq_len=8
        )


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("dpo", "DPOTrainerWrapper"),
        ("orpo", "ORPOTrainerWrapper"),
        ("simpo", "SimPOTrainerWrapper"),
        ("kto", "KTOTrainerWrapper"),
    ],
)
def test_all_four_preference_trainers_fail_closed_on_an_unconsumed_probe(
    module_name, class_name
):
    import importlib

    from soup_cli.trainer.stream_setup import _ProbePlan

    class Runtime:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    wrapper_cls = getattr(importlib.import_module(f"soup_cli.trainer.{module_name}"), class_name)
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper._stream_runtime = Runtime()
    wrapper._pending_stream_vram_probe = _ProbePlan(
        task=module_name,
        batch_size=1,
        rows=2,
        seq_len=8,
        vocab_size=64,
        predicted_bytes=100,
        available_bytes=1_000,
    )
    with pytest.raises(RuntimeError, match="probe was not consumed"):
        wrapper.train()
    assert wrapper._stream_runtime.closed == 1


def test_pending_probe_executes_trainer_compute_loss(monkeypatch):
    import torch

    from soup_cli.trainer.stream_setup import StreamingSetupMixin, _ProbePlan
    from soup_cli.utils.layer_stream_runtime import StepPeak

    class Runtime:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    class Trainer:
        def __init__(self, model):
            self.model = model
            self.seen = None
            self._metrics = {"train": {"loss": [1.0]}}
            self._total_train_tokens = 7

        def get_train_dataloader(self):
            return [{
                "input_ids": torch.tensor([[1, 2, 3], [1, 4, 5]]),
                "attention_mask": torch.ones((2, 3), dtype=torch.long),
                "completion_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
            }]

        def compute_loss(self, model, batch):
            self.seen = batch["input_ids"].shape
            self._metrics["train"]["loss"].append(99.0)
            self._total_train_tokens += int(batch["attention_mask"].sum())
            torch.rand(4)
            return model.weight.sum()

    class Wrapper(StreamingSetupMixin):
        pass

    wrapper = Wrapper()
    wrapper.model = torch.nn.Linear(4, 4, bias=False)
    wrapper.trainer = Trainer(wrapper.model)
    wrapper.device = "cuda"
    wrapper._stream_runtime = Runtime()
    wrapper._pending_stream_vram_probe = _ProbePlan(
        task="dpo",
        batch_size=1,
        rows=2,
        seq_len=8,
        vocab_size=64,
        predicted_bytes=100,
        available_bytes=1_000,
    )

    def fake_measure(model, *, step, rows, seq_len, device):
        assert rows == 2
        assert seq_len == 8
        assert device == "cuda"
        loss = step()
        assert loss.requires_grad
        return StepPeak(
            peak_bytes=200,
            reserved_bytes=220,
            seconds=0.01,
            rows=rows,
            seq_len=seq_len,
        )

    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.measure_loss_step_peak_bytes",
        fake_measure,
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.measure_step_peak_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preference probe dispatched to the causal-LM instrument")
        ),
    )
    trainer_attributes = set(vars(wrapper.trainer))
    wrapper.model.eval()
    rng_before = torch.random.get_rng_state()
    wrapper._run_pending_stream_vram_probe()
    assert wrapper.trainer.seen == torch.Size([2, 8])
    assert wrapper.trainer._metrics == {"train": {"loss": [1.0]}}
    assert wrapper.trainer._total_train_tokens == 7
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert set(vars(wrapper.trainer)) == trainer_attributes
    assert wrapper.model.training is False
    assert wrapper._pending_stream_vram_probe is None
    assert wrapper._stream_runtime.closed == 0


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        ({"input_ids": None}, "no complete token sequence"),
        pytest.param("partial", "expected 2 model rows", id="partial-batch"),
    ],
)
def test_batch_build_failure_closes_the_stream_runtime(batch, message):
    import torch

    from soup_cli.trainer.stream_setup import StreamingSetupMixin, _ProbePlan

    if batch == "partial":
        batch = {"input_ids": torch.ones((1, 3), dtype=torch.long)}

    class Runtime:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    class Trainer:
        def get_train_dataloader(self):
            return [batch]

    class Wrapper(StreamingSetupMixin):
        pass

    wrapper = Wrapper()
    wrapper.model = torch.nn.Linear(4, 4, bias=False)
    wrapper.trainer = Trainer()
    wrapper.device = "cuda"
    wrapper._stream_runtime = Runtime()
    wrapper._pending_stream_vram_probe = _ProbePlan(
        task="dpo",
        batch_size=1,
        rows=2,
        seq_len=8,
        vocab_size=64,
        predicted_bytes=100,
        available_bytes=1_000,
    )
    with pytest.raises(ValueError, match=message):
        wrapper._run_pending_stream_vram_probe()
    assert wrapper._stream_runtime.closed == 1


def test_loss_instrument_uses_allocated_peak_runs_backward_and_clears_grads(monkeypatch):
    import torch

    import soup_cli.utils.layer_stream_runtime as runtime_module
    from soup_cli.utils.layer_stream_runtime import measure_loss_step_peak_bytes

    calls = {"allocated": 0, "reserved": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def allocated(*_args):
        calls["allocated"] += 1
        return 123

    def reserved(*_args):
        calls["reserved"] += 1
        return 999

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", reserved)
    real_zero = runtime_module._zero_probe_grads
    saw_backward_grads = {"value": False}

    def inspect_then_zero(target):
        saw_backward_grads["value"] = any(
            parameter.grad is not None for parameter in target.parameters()
        )
        real_zero(target)

    monkeypatch.setattr(runtime_module, "_zero_probe_grads", inspect_then_zero)
    model = torch.nn.Linear(4, 1)
    x = torch.randn(3, 4)
    peak = measure_loss_step_peak_bytes(
        model,
        step=lambda: model(x).square().mean(),
        rows=3,
        seq_len=1,
        device="cuda",
    )
    assert peak is not None and not peak.failed
    assert peak.peak_bytes == 123
    assert peak.reserved_bytes == 999
    assert calls == {"allocated": 1, "reserved": 1}
    assert saw_backward_grads["value"] is True
    assert all(parameter.grad is None for parameter in model.parameters())


def test_loss_instrument_classifies_non_finite_loss_as_failed(monkeypatch):
    import torch

    from soup_cli.utils.layer_stream_runtime import measure_loss_step_peak_bytes

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = torch.nn.Linear(1, 1)
    peak = measure_loss_step_peak_bytes(
        model,
        step=lambda: model(torch.ones(1, 1)).sum() * torch.tensor(float("nan")),
        rows=1,
        seq_len=1,
        device="cuda",
    )
    assert peak is not None and peak.failed
    assert peak.error == "FloatingPointError"


def test_probe_forward_matches_accelerate_native_amp_contract_on_cpu(monkeypatch):
    import torch

    from soup_cli.trainer.stream_setup import _trainer_forward_for_probe

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)
            self.autocast_states = []
            self.logits_dtypes = []

        def forward(self, x):
            self.autocast_states.append(torch.is_autocast_enabled("cpu"))
            logits = self.linear(x)
            self.logits_dtypes.append(logits.dtype)
            return logits

    monkeypatch.setattr(
        "accelerate.utils.modeling.get_mixed_precision_context_manager",
        lambda *_args, **_kwargs: torch.autocast("cpu", dtype=torch.bfloat16),
    )
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(native_amp=True, autocast_handler=None)
    )
    model = Model()
    assert "forward" not in vars(model)
    with _trainer_forward_for_probe(trainer, model):
        output = model(torch.randn(2, 8))
    assert model.autocast_states == [True]
    assert model.logits_dtypes == [torch.bfloat16]
    assert output.dtype == torch.float32
    assert "forward" not in vars(model)
