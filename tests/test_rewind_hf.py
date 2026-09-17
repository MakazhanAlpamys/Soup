"""HF/TRL flight-recorder tests: row identity, per-row loss, and the eval guard.

Every test runs a real (tiny-random, CPU) TRL ``SFTTrainer`` -- the whole point of
the module is that it hooks HF internals correctly, and a mocked trainer would
assert nothing about ``_get_train_sampler`` / ``compute_loss`` call order.
"""

from __future__ import annotations

import pytest

pytest.importorskip("trl")

from soup_cli.trainer import rewind_hf  # noqa: E402

MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class FakeSink:
    """List-collecting stand-in for ``monitoring.rewind_log.RewindLog``."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def record_batch(self, **kwargs) -> None:
        self.records.append(dict(kwargs))


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "true")


def _recording_console(monkeypatch):
    from rich.console import Console

    console = Console(record=True, force_terminal=False, width=10_000, soft_wrap=True)
    monkeypatch.setattr(rewind_hf, "console", console)
    return console


def _dataset(n: int = 8):
    from datasets import Dataset

    return Dataset.from_list([{"text": f"row {i} " * (i + 1)} for i in range(n)])


def _uniform_dataset(n: int = 8):
    """Rows of identical text -> identical supervised-token counts per row."""
    from datasets import Dataset

    return Dataset.from_list([{"text": "the quick brown fox jumps"} for _ in range(n)])


def _model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return model, tok


def _config(tmp_path, **overrides):
    from trl import SFTConfig

    kwargs = dict(
        output_dir=str(tmp_path),
        max_steps=2,
        per_device_train_batch_size=4,
        report_to=[],
        logging_steps=1,
        save_strategy="no",
        use_cpu=True,
    )
    kwargs.update(overrides)
    return SFTConfig(**kwargs)


def _trainer(tmp_path, *, wrap: bool = True, dataset=None, **overrides):
    from trl import SFTTrainer

    model, tok = _model_and_tokenizer()
    cls = rewind_hf.make_rewind_trainer_class(SFTTrainer) if wrap else SFTTrainer
    eval_dataset = overrides.pop("eval_dataset", None)
    return cls(
        model=model,
        args=_config(tmp_path, **overrides),
        train_dataset=_dataset() if dataset is None else dataset,
        eval_dataset=eval_dataset,
        processing_class=tok,
    )


def _attach_spy(monkeypatch, trainer, sink):
    """Attach a RewindState that also records every index the sampler yielded.

    The spy sits on ``on_index`` (what the sampler pushed), which is independent
    of the FIFO pop arithmetic in ``on_micro_batch`` (what the sink received) --
    so comparing the two is a real assertion, not a tautology.
    """

    class SpyState(rewind_hf.RewindState):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen: list[int] = []

        def on_index(self, i: int) -> None:
            self.seen.append(int(i))
            super().on_index(i)

    monkeypatch.setattr(rewind_hf, "RewindState", SpyState)
    return rewind_hf.attach_rewind_state(trainer, sink)


# --------------------------------------------------------------------------
# 1. row_losses
# --------------------------------------------------------------------------


def test_row_losses_matches_hand_computed_cross_entropy():
    import torch
    import torch.nn.functional as f

    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    # After the shift, row 0's second supervised position is masked out.
    labels = torch.tensor([[0, 1, -100], [3, 4, 0]])

    row_loss, row_tokens = rewind_hf.row_losses(logits, labels)

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    ce = f.cross_entropy(
        shift_logits.reshape(-1, 5),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(2, 2)
    mask = shift_labels != -100
    expected = [
        float(ce[0][0]),                      # one supervised token
        float((ce[1][0] + ce[1][1]) / 2.0),   # two supervised tokens
    ]

    assert row_tokens == [int(mask[0].sum()), int(mask[1].sum())] == [1, 2]
    assert row_loss == pytest.approx(expected, abs=1e-6)


def test_row_losses_row_with_no_supervised_tokens_is_zero():
    import torch

    torch.manual_seed(1)
    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[0, -100, -100], [1, 2, 3]])

    row_loss, row_tokens = rewind_hf.row_losses(logits, labels)

    assert row_tokens[0] == 0
    assert row_loss[0] == 0.0
    assert row_tokens[1] == 2


def test_row_losses_mean_equals_trl_scalar(tmp_path):
    """With equal supervised-token counts, mean(row_loss) == TRL's scalar."""
    import torch

    trainer = _trainer(tmp_path, wrap=False, dataset=_uniform_dataset())
    batch = next(iter(trainer.get_train_dataloader()))
    trainer.model.train()

    loss, outputs = trainer.compute_loss(
        trainer.model, dict(batch), return_outputs=True, num_items_in_batch=None
    )
    row_loss, row_tokens = rewind_hf.row_losses(outputs.logits, batch["labels"])

    assert len(set(row_tokens)) == 1, f"rows must have equal token counts, got {row_tokens}"
    assert torch.tensor(row_loss).mean().item() == pytest.approx(loss.item(), abs=1e-4)


# --------------------------------------------------------------------------
# 2. row identity end to end
# --------------------------------------------------------------------------


def test_recorded_rows_match_sampler_order(tmp_path, monkeypatch):
    sink = FakeSink()
    trainer = _trainer(tmp_path)
    state = _attach_spy(monkeypatch, trainer, sink)

    trainer.train()

    assert sink.records, "no micro-batches recorded"
    recorded = [r for rec in sink.records for r in rec["rows"]]
    assert recorded == state.seen
    assert state.dropped == 0
    assert not state.failed
    for rec in sink.records:
        assert len(rec["rows"]) == len(rec["row_loss"]) == len(rec["row_tokens"]) == 4


# --------------------------------------------------------------------------
# 3. step / micro bookkeeping under gradient accumulation
# --------------------------------------------------------------------------


def test_grad_accum_step_and_micro_sequence(tmp_path, monkeypatch):
    sink = FakeSink()
    trainer = _trainer(
        tmp_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=2,
    )
    _attach_spy(monkeypatch, trainer, sink)

    trainer.train()

    assert [(r["step"], r["micro"]) for r in sink.records] == [(1, 0), (1, 1), (2, 0), (2, 1)]


# --------------------------------------------------------------------------
# 4. the eval guard
# --------------------------------------------------------------------------


def test_eval_passes_record_nothing(tmp_path, monkeypatch):
    sink = FakeSink()
    trainer = _trainer(
        tmp_path,
        eval_dataset=_dataset(),
        eval_strategy="steps",
        eval_steps=1,
        per_device_eval_batch_size=4,
    )
    state = _attach_spy(monkeypatch, trainer, sink)

    trainer.train()

    # max_steps=2, batch 4, no grad accum -> exactly 2 training micro-batches.
    # Eval runs 4 more compute_loss calls (8 rows / batch 4, twice) which must
    # not reach the recorder at all.
    assert len(sink.records) == 2
    # An eval call that reaches ``on_micro_batch`` is NOT harmless just because
    # the record count stayed at 2: here it happens to find the FIFO empty and
    # gets counted as a drop, but with a prefetching loader it would pop the
    # NEXT training batch's ids and log them against eval rows. ``dropped == 0``
    # is what proves the guard fired, not the underflow.
    assert state.dropped == 0
    recorded = [r for rec in sink.records for r in rec["rows"]]
    assert sorted(recorded) == list(range(8)) == sorted(state.seen)


# --------------------------------------------------------------------------
# 5. multi-worker dataloader
# --------------------------------------------------------------------------


def test_rows_recorded_with_dataloader_workers(tmp_path, monkeypatch):
    sink = FakeSink()
    trainer = _trainer(tmp_path, dataloader_num_workers=2)
    state = _attach_spy(monkeypatch, trainer, sink)

    trainer.train()

    assert sink.records, "no micro-batches recorded with num_workers=2"
    recorded = [r for rec in sink.records for r in rec["rows"]]
    assert recorded == state.seen
    for rec in sink.records:
        assert len(rec["rows"]) == len(rec["row_loss"]) == len(rec["row_tokens"]) == 4


# --------------------------------------------------------------------------
# 6. factory identity and the opt-out path
# --------------------------------------------------------------------------


def test_factory_is_cached_and_unattached_trainer_records_nothing(tmp_path, monkeypatch):
    from trl import SFTTrainer

    cls = rewind_hf.make_rewind_trainer_class(SFTTrainer)
    assert cls is rewind_hf.make_rewind_trainer_class(SFTTrainer)

    calls = []
    real_row_losses = rewind_hf.row_losses

    def _counting(logits, labels):
        calls.append(1)
        return real_row_losses(logits, labels)

    monkeypatch.setattr(rewind_hf, "row_losses", _counting)

    bare = _trainer(tmp_path)  # the rewind class, but no attach_rewind_state
    list(bare._get_train_sampler(bare.train_dataset))  # iterating must not need a state
    bare.train()

    assert calls == [], "an unattached trainer computed per-row losses"


# --------------------------------------------------------------------------
# 7. a broken recorder never stops training
# --------------------------------------------------------------------------


def test_row_losses_failure_disables_recorder_without_stopping_training(
    tmp_path, monkeypatch
):
    recording_console = _recording_console(monkeypatch)

    def _boom(logits, labels):
        raise RuntimeError("synthetic row_losses failure")

    monkeypatch.setattr(rewind_hf, "row_losses", _boom)

    sink = FakeSink()
    trainer = _trainer(tmp_path)
    state = rewind_hf.attach_rewind_state(trainer, sink)

    trainer.train()  # must not raise

    assert state.failed is True
    assert sink.records == []
    text = recording_console.export_text()
    warnings = [ln for ln in text.splitlines() if "Rewind recorder disabled:" in ln]
    assert len(warnings) == 1, text
    assert "synthetic row_losses failure" in warnings[0]


# --------------------------------------------------------------------------
# RewindState unit behaviour (no torch)
# --------------------------------------------------------------------------


def test_state_drops_batch_when_fifo_is_short():
    sink = FakeSink()
    state = rewind_hf.RewindState(sink)
    state.on_index(3)

    state.on_micro_batch(step=1, row_loss=[1.0, 2.0], row_tokens=[4, 4])

    assert sink.records == []
    assert state.dropped == 1
    assert state.failed is False


# --------------------------------------------------------------------------
# Degradation paths found in review: each one used to log wrong rows or crash
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["[/tmp/x]", "[/]", "missing [labels]", "[bold]x"])
def test_disable_reason_is_escaped_not_parsed_as_markup(monkeypatch, reason):
    """Unescaped, "[/tmp/x]" raised MarkupError INTO TRAINING from the handler."""
    console = _recording_console(monkeypatch)
    state = rewind_hf.RewindState(FakeSink())

    state.disable(reason)  # must not raise

    assert state.failed is True
    assert reason in console.export_text()


def test_resumed_run_disables_the_recorder(tmp_path, monkeypatch):
    """Accelerate's SkipBatchSampler enumerates the wrapped sampler on resume, so
    skipped rows would reach the FIFO while compute_loss never runs for them."""
    first = _trainer(tmp_path, wrap=False, save_strategy="steps", save_steps=1)
    first.train()
    checkpoint = tmp_path / "checkpoint-1"
    assert checkpoint.is_dir()

    console = _recording_console(monkeypatch)
    sink = FakeSink()
    resumed = _trainer(tmp_path, save_strategy="no", max_steps=2)
    state = rewind_hf.attach_rewind_state(resumed, sink)

    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert state.failed is True
    assert sink.records == []
    assert "resumed runs" in console.export_text()


def test_packing_is_refused_and_unpacked_control_records(tmp_path, monkeypatch):
    """packing=True turns on TRL's padding_free: B=1 micro-batches, one id popped
    each, the rest leaking forever."""
    console = _recording_console(monkeypatch)
    packed_sink = FakeSink()
    packed = _trainer(tmp_path / "packed", packing=True, max_length=64)
    assert getattr(packed, "padding_free", False) is True, "precondition: TRL packs padding-free"
    packed_state = rewind_hf.attach_rewind_state(packed, packed_sink)
    packed.train()

    assert packed_state.failed is True
    assert packed_sink.records == []
    assert "packing" in console.export_text()

    control_sink = FakeSink()
    control = _trainer(tmp_path / "control", packing=False, max_length=64)
    rewind_hf.attach_rewind_state(control, control_sink)
    control.train()
    assert len(control_sink.records) == 2


def test_sampler_wrapper_forwards_set_epoch(tmp_path):
    """The dataloader reseeds the shuffle through ``sampler.set_epoch``; a wrapper
    that swallowed it changed the data order on an epoch jump."""
    trainer = _trainer(tmp_path)
    state = rewind_hf.attach_rewind_state(trainer, FakeSink())
    wrapper_cls = type(trainer._get_train_sampler(trainer.train_dataset))

    class EpochSampler:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

        def __iter__(self):
            return iter(range(4))

        def __len__(self):
            return 4

    inner = EpochSampler()
    wrapped = wrapper_cls(inner, state)

    assert hasattr(wrapped, "set_epoch")
    wrapped.set_epoch(2)
    assert inner.epochs == [2]
    assert list(wrapped) == [0, 1, 2, 3]


def _stub_base_compute_loss(monkeypatch, result):
    from trl import SFTTrainer

    def _base(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return result if return_outputs else result[0]

    monkeypatch.setattr(SFTTrainer, "compute_loss", _base)


def test_shift_labels_falls_through_to_the_base_trainer(tmp_path, monkeypatch):
    """Context parallelism shards logits along the sequence; the causal shift would
    misalign them, so the recorder must not touch that batch."""
    import torch

    trainer = _trainer(tmp_path)
    sink = FakeSink()
    state = rewind_hf.attach_rewind_state(trainer, sink)
    state.on_index(0)
    sentinel = torch.tensor(1.25)
    _stub_base_compute_loss(monkeypatch, (sentinel, None))
    monkeypatch.setattr(rewind_hf, "row_losses", lambda *a: pytest.fail("row_losses ran"))
    trainer.model.train()

    labels = torch.tensor([[1, 2, 3]])
    out = trainer.compute_loss(trainer.model, {"labels": labels, "shift_labels": labels})

    assert out is sentinel
    assert sink.records == [] and state.failed is False


def test_missing_logits_disables_with_a_liger_hint(tmp_path, monkeypatch):
    """Liger sets skip_logits during training: outputs.logits is None."""
    from types import SimpleNamespace

    import torch

    console = _recording_console(monkeypatch)
    trainer = _trainer(tmp_path)
    sink = FakeSink()
    state = rewind_hf.attach_rewind_state(trainer, sink)
    state.on_index(0)
    loss = torch.tensor(0.5)
    _stub_base_compute_loss(monkeypatch, (loss, SimpleNamespace(logits=None)))
    trainer.model.train()

    out = trainer.compute_loss(trainer.model, {"labels": torch.tensor([[1, 2, 3]])})

    assert out is loss
    assert state.failed is True and sink.records == []
    assert "liger" in console.export_text().lower()


def test_row_losses_in_bf16_match_float32_reference():
    """No fp32 upcast of the (B, T, V) logits: CE runs in the logits' dtype."""
    import torch

    torch.manual_seed(2)
    logits = torch.randn(2, 6, 11)
    labels = torch.tensor([[0, 1, 2, 3, 4, 5], [-100, -100, 7, 8, -100, 9]])

    ref_loss, ref_tokens = rewind_hf.row_losses(logits, labels)
    bf_loss, bf_tokens = rewind_hf.row_losses(logits.to(torch.bfloat16), labels)

    assert bf_tokens == ref_tokens == [5, 3]
    assert bf_loss == pytest.approx(ref_loss, abs=1e-2)


def test_summary_reports_dropped_micro_batches():
    state = rewind_hf.RewindState(FakeSink())
    assert state.summary() is None
    state.on_micro_batch(step=1, row_loss=[1.0], row_tokens=[3])  # FIFO empty -> drop
    assert state.summary() == "rewind: 1 micro-batch dropped (FIFO underflow)"
