"""``training.packed_lora``'s hook on the causal LM's forward, on CPU with a stub engine.

The hook decides which calls the kernels take (the trainer's labelled step in training mode with
gradients on), hands everything else to the stock forward with the host-parked tables back on the
GPU, and leaves the metrics of the last labelled forward for the trainer. The engine is a stub
here (the kernels run in ``test_packed_lora_e2e_gpu.py``); what is tested is the routing, the
bookkeeping and that the model is put back as it was found.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from soup_cli.kernels.packed_lora import hook as hook_module  # noqa: E402
from soup_cli.kernels.packed_lora.hook import PackedLoraHook, step_from_logits  # noqa: E402

DTYPE = torch.bfloat16


def _model(seed=3):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=False)
    model = Qwen2ForCausalLM(config)
    model.train()
    return model


def _batch(rows=2, width=9, pad_right=True):
    generator = torch.Generator().manual_seed(11)
    ids = torch.randint(1, 64, (rows, width), generator=generator)
    mask = torch.ones(rows, width, dtype=torch.long)
    labels = ids.clone()
    mask[1, -3:] = 0
    labels[1, -3:] = -100
    labels[0, :2] = -100
    if not pad_right:
        mask[1, :3], mask[1, 3:] = 0, 1
    return ids, mask, labels


class StubRun:
    """What the hook asks of a ``PackedRun``; records the calls."""

    def __init__(self, result="packed"):
        self.device = torch.device("cpu")
        self.dtype = DTYPE
        self.closed = False
        self.calls = []
        self.stock_reasons = []
        self.result = result
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, input_ids, attention_mask, labels, n_items):
        self.calls.append(("forward", n_items))
        if self.result is None:
            return None
        rows, width = input_ids.shape
        n_real = int(attention_mask.sum())
        token_loss = torch.zeros(rows, width)
        token_loss[attention_mask.bool()] = torch.arange(1, n_real + 1, dtype=torch.float32)
        stats = {
            "entropy": torch.arange(1, n_real + 1, dtype=torch.float32) / 10,
            "token_loss": token_loss,
            "hit": (torch.arange(n_real) % 2).to(torch.int32),
            "labelled": torch.tensor(n_real - 3),
        }
        return self.weight * 2.5, stats

    def note_stock(self, reason):
        self.stock_reasons.append(reason)
        self.calls.append(("stock", reason))

    def restore_tables(self):
        self.calls.append(("restore",))

    def park_tables(self):
        self.calls.append(("park",))

    def names(self):
        return [call[0] for call in self.calls]


@pytest.fixture
def autocast_bf16(monkeypatch):
    monkeypatch.setattr(hook_module, "_autocast_dtype", lambda: DTYPE)


def _call(model, **extra):
    ids, mask, labels = _batch()
    return model(input_ids=ids, attention_mask=mask, labels=labels, **extra)


# --- installing and removing ---------------------------------------------------------------


class TestInstall:
    def test_the_forward_is_on_the_instance_and_remove_uncovers_the_class(self):
        model = _model()
        assert "forward" not in model.__dict__
        hook = PackedLoraHook(model, StubRun())
        assert model.forward == hook._forward
        hook.remove()
        assert "forward" not in model.__dict__

    def test_remove_puts_back_a_forward_that_was_already_on_the_instance(self):
        model = _model()
        own = model.forward

        def before(*args, **kwargs):
            return own(*args, **kwargs)

        model.forward = before
        hook = PackedLoraHook(model, StubRun())
        assert model.forward != before
        hook.remove()
        assert model.__dict__["forward"] is before

    def test_remove_twice_is_harmless(self):
        model = _model()
        hook = PackedLoraHook(model, StubRun())
        hook.remove()
        hook.remove()
        assert "forward" not in model.__dict__

    def test_a_second_remove_leaves_what_was_set_after_the_first(self):
        model = _model()
        hook = PackedLoraHook(model, StubRun())
        hook.remove()

        def later(*args, **kwargs):
            return None

        model.forward = later
        hook.remove()
        assert model.__dict__["forward"] is later

    def test_the_class_is_never_touched(self):
        model = _model()
        before = type(model).forward
        hook = PackedLoraHook(model, StubRun())
        assert type(model).forward is before
        hook.remove()
        assert type(model).forward is before

    def test_a_removed_hook_does_not_leave_the_forward_in_the_hook(self):
        model = _model()
        hook = PackedLoraHook(model, StubRun())
        hook.remove()
        outputs = _call(model)
        assert outputs.logits is not None
        assert hook.take_step() is None


# --- the packed step -----------------------------------------------------------------------


class TestThePackedStep:
    def test_a_labelled_training_step_runs_the_engine_and_returns_no_logits(self, autocast_bf16):
        model = _model()
        run = StubRun()
        PackedLoraHook(model, run)
        outputs = _call(model, num_items_in_batch=torch.tensor(7))
        assert outputs.logits is None
        assert outputs.loss.item() == pytest.approx(2.5)
        assert outputs.loss.requires_grad
        assert run.names() == ["park", "forward"]
        assert int(run.calls[1][1]) == 7  # num_items_in_batch reaches the engine

    def test_the_metrics_are_left_for_the_trainer_once(self, autocast_bf16):
        model = _model()
        hook = PackedLoraHook(model, StubRun())
        _, mask, _ = _batch()
        _call(model)
        step = hook.take_step()
        n_real = int(mask.sum())
        # one entropy per real token, put back at its place in [batch, seq]
        assert step.token_entropy.shape == mask.shape
        assert step.token_entropy[mask.bool()].tolist() == pytest.approx(
            [value / 10 for value in range(1, n_real + 1)])
        assert float(step.token_entropy[~mask.bool()].abs().sum()) == 0.0
        assert step.token_loss.shape == mask.shape
        assert step.correct.item() == n_real // 2  # the hits of the engine, summed
        assert step.labelled.item() == n_real - 3
        assert hook.take_step() is None

    def test_the_gradient_reaches_the_engines_parameters(self, autocast_bf16):
        model = _model()
        run = StubRun()
        PackedLoraHook(model, run)
        _call(model).loss.backward()
        assert run.weight.grad.item() == pytest.approx(2.5)

    def test_the_stock_decoder_is_never_run(self, autocast_bf16):
        """The packed step does not go through the decoder layers, so HF's gradient
        checkpointing (which lives in them) is never invoked."""
        model = _model()
        model.gradient_checkpointing_enable()

        def refuse(*args, **kwargs):
            raise AssertionError("a decoder layer ran in a packed step")

        for layer in model.model.layers:
            layer.forward = refuse
        PackedLoraHook(model, StubRun())
        assert _call(model).loss.item() == pytest.approx(2.5)

    def test_the_tables_are_parked_on_the_first_step_only(self, autocast_bf16):
        model = _model()
        run = StubRun()
        PackedLoraHook(model, run)
        for _ in range(3):
            _call(model)
        assert run.names().count("park") == 1
        assert run.names().count("forward") == 3

    def test_an_evaluation_puts_the_tables_back_and_the_next_step_parks_them_again(
            self, autocast_bf16):
        model = _model()
        run = StubRun()
        PackedLoraHook(model, run)
        _call(model)
        model.eval()
        with torch.no_grad():
            _call(model)
            _call(model)
        model.train()
        _call(model)
        assert run.names() == ["park", "forward", "restore", "restore", "park", "forward"]

    def test_the_steps_are_counted_by_the_engine_not_by_the_hook(self, autocast_bf16):
        model = _model()
        run = StubRun()
        PackedLoraHook(model, run)
        _call(model)
        assert run.stock_reasons == []


# --- the stock path ------------------------------------------------------------------------


class TestTheStockPath:
    def test_a_forward_without_labels_is_the_stock_forward_with_the_tables_back(
            self, autocast_bf16):
        model = _model()
        run = StubRun()
        hook = PackedLoraHook(model, run)
        ids, mask, _ = _batch()
        want = type(model).forward(model, input_ids=ids, attention_mask=mask).logits
        got = model(input_ids=ids, attention_mask=mask).logits
        assert torch.equal(got, want)
        assert run.names() == ["restore"]
        assert hook.take_step() is None

    def test_evaluation_is_stock_and_not_a_stock_step(self, autocast_bf16):
        model = _model().eval()
        run = StubRun()
        hook = PackedLoraHook(model, run)
        with torch.no_grad():
            outputs = _call(model)
        assert outputs.logits is not None
        assert run.names() == ["restore"]
        assert run.stock_reasons == []
        assert hook.take_step() is not None

    def test_gradients_off_in_training_mode_is_stock_and_not_a_stock_step(self, autocast_bf16):
        model = _model()
        run = StubRun()
        PackedLoraHook(model, run)
        with torch.no_grad():
            outputs = _call(model)
        assert outputs.logits is not None
        assert run.stock_reasons == []
        assert "forward" not in run.names()

    @pytest.mark.parametrize("extra,reason", [
        ({"position_ids": torch.arange(9)[None, :].expand(2, 9)}, "position_ids"),
        ({"inputs_embeds": None, "past_key_values": object()}, "past_key_values"),
        ({"use_cache": True}, "the cache"),
        ({"output_hidden_states": True}, "the hidden states"),
        ({"output_attentions": True}, "the attentions"),
        ({"return_dict": False}, "a tuple"),
    ])
    def test_a_call_the_kernels_do_not_take_is_a_counted_stock_step(
            self, autocast_bf16, extra, reason):
        model = _model()
        run = StubRun()
        hook = PackedLoraHook(model, run)
        ids, mask, labels = _batch()
        stock = hook._stock
        seen = {}

        def spy(*args, **kwargs):
            seen.update(kwargs)
            kwargs.pop("past_key_values", None)
            return stock(*args, **kwargs)

        hook._stock = spy
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels, **extra)
        assert "forward" not in run.names()
        assert len(run.stock_reasons) == 1 and reason in run.stock_reasons[0]
        assert seen["labels"] is labels
        if isinstance(outputs, tuple):
            return
        assert outputs.loss is not None

    def test_evaluation_with_gradients_on_is_still_not_a_packed_step(self, autocast_bf16):
        """A model in eval mode is not training, whatever torch.is_grad_enabled() says."""
        model = _model().eval()
        run = StubRun()
        PackedLoraHook(model, run)
        outputs = _call(model)
        assert outputs.logits is not None
        assert "forward" not in run.names() and run.stock_reasons == []

    def test_shift_labels_are_a_stock_step(self, autocast_bf16):
        from types import SimpleNamespace

        model = _model()
        run = StubRun()
        hook = PackedLoraHook(model, run)
        hook._stock = lambda *args, **kwargs: SimpleNamespace(loss=None, logits=None)
        _call(model, shift_labels=torch.zeros(2, 9, dtype=torch.long))
        assert "shift_labels" in run.stock_reasons[0]

    def test_a_call_that_keeps_some_logits_is_a_stock_step_without_metrics(self, autocast_bf16):
        from types import SimpleNamespace

        model = _model()
        run = StubRun()
        hook = PackedLoraHook(model, run)
        hook._stock = lambda *args, **kwargs: SimpleNamespace(
            loss=None, logits=torch.zeros(2, 3, 64))
        _call(model, logits_to_keep=3)
        assert "some logits" in run.stock_reasons[0]
        assert hook.take_step() is None  # the logits no longer line up with the labels

    def test_inputs_embeds_are_a_stock_step(self, autocast_bf16):
        model = _model()
        run = StubRun()
        PackedLoraHook(model, run)
        ids, mask, labels = _batch()
        embeds = model.get_input_embeddings()(ids)
        model(inputs_embeds=embeds, attention_mask=mask, labels=labels)
        assert "inputs_embeds" in run.stock_reasons[0]

    def test_a_batch_without_an_attention_mask_is_a_stock_step(self, autocast_bf16):
        model = _model()
        run = StubRun()
        PackedLoraHook(model, run)
        ids, _, labels = _batch()
        model(input_ids=ids, labels=labels)
        assert "attention_mask" in run.stock_reasons[0]

    def test_positional_arguments_are_a_stock_step(self, autocast_bf16):
        model = _model()
        run = StubRun()
        PackedLoraHook(model, run)
        ids, mask, labels = _batch()
        model(ids, mask, labels=labels)
        assert "positional" in run.stock_reasons[0]

    def test_a_batch_on_another_device_is_a_stock_step(self, autocast_bf16):
        model = _model()
        run = StubRun()
        run.device = torch.device("cuda", 0)
        PackedLoraHook(model, run)
        _call(model)
        assert "cuda:0" in run.stock_reasons[0]

    def test_no_autocast_is_a_stock_step_that_names_the_mode(self, monkeypatch):
        monkeypatch.setattr(hook_module, "_autocast_dtype", lambda: None)
        model = _model()
        run = StubRun()
        PackedLoraHook(model, run)
        _call(model)
        assert "autocast is None" in run.stock_reasons[0]
        assert "bfloat16 mode" in run.stock_reasons[0]

    def test_the_wrong_autocast_dtype_is_a_stock_step(self, monkeypatch):
        monkeypatch.setattr(hook_module, "_autocast_dtype", lambda: torch.float16)
        model = _model()
        run = StubRun()
        PackedLoraHook(model, run)
        _call(model)
        assert "torch.float16" in run.stock_reasons[0]

    def test_a_batch_the_engine_cannot_take_runs_the_stock_forward(self, autocast_bf16):
        model = _model()
        run = StubRun(result=None)
        hook = PackedLoraHook(model, run)
        outputs = _call(model)
        assert outputs.logits is not None
        assert run.names() == ["park", "forward", "stock", "restore"]
        assert run.stock_reasons == ["a batch that is not right-padded"]
        assert hook.take_step() is not None  # the trainer logs it from the logits

    def test_a_closed_run_is_left_alone(self, autocast_bf16):
        model = _model()
        run = StubRun()
        run.closed = True
        PackedLoraHook(model, run)
        outputs = _call(model)
        assert outputs.logits is not None
        assert run.calls == []


# --- a model the kernels did not take --------------------------------------------------------


class TestWithoutARun:
    def test_the_stock_forward_is_fed_the_embedding_as_the_stock_model_computes_it(self):
        model = _model()
        model._packed_lora_fp32_embeds = True
        hook = PackedLoraHook(model, None)
        stock = hook._stock
        seen = {}

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return stock(*args, **kwargs)

        hook._stock = spy
        ids, mask, labels = _batch()
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        assert seen["input_ids"] is None
        assert seen["inputs_embeds"].dtype == torch.float32
        assert torch.equal(seen["inputs_embeds"], model.get_input_embeddings()(ids).float())
        assert outputs.loss is not None

    def test_a_model_with_an_ordinary_table_is_called_as_the_trainer_called_it(self):
        model = _model()
        hook = PackedLoraHook(model, None)
        stock = hook._stock
        seen = {}

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return stock(*args, **kwargs)

        hook._stock = spy
        _call(model)
        assert seen["input_ids"] is not None and "inputs_embeds" not in seen

    def test_inputs_embeds_given_by_the_caller_are_left_alone(self):
        model = _model()
        model._packed_lora_fp32_embeds = True
        hook = PackedLoraHook(model, None)
        stock = hook._stock
        seen = {}

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return stock(*args, **kwargs)

        hook._stock = spy
        ids, mask, labels = _batch()
        embeds = model.get_input_embeddings()(ids)
        model(inputs_embeds=embeds, attention_mask=mask, labels=labels)
        assert seen["inputs_embeds"] is embeds

    def test_a_call_with_both_ids_and_embeds_reaches_the_model_unchanged(self):
        """The model's own error for it is what the caller should see, not an embedding made
        behind its back."""
        model = _model()
        model._packed_lora_fp32_embeds = True
        hook = PackedLoraHook(model, None)
        seen = {}
        hook._stock = lambda *args, **kwargs: seen.update(kwargs)
        ids, mask, labels = _batch()
        embeds = model.get_input_embeddings()(ids)
        model(input_ids=ids, inputs_embeds=embeds, attention_mask=mask, labels=labels)
        assert seen["input_ids"] is ids and seen["inputs_embeds"] is embeds

    def test_the_trainers_metrics_are_still_left_for_it(self):
        model = _model()
        hook = PackedLoraHook(model, None)
        _call(model)
        assert hook.take_step() is not None


# --- the metrics of a stock forward --------------------------------------------------------


class TestStepFromLogits:
    def _case(self):
        torch.manual_seed(5)
        logits = torch.randn(2, 9, 64)
        _, _, labels = _batch()
        return logits, labels

    def test_the_entropy_is_trls(self):
        from trl.trainer.utils import entropy_from_logits

        logits, labels = self._case()
        step = step_from_logits(logits, labels)
        assert torch.allclose(step.token_entropy, entropy_from_logits(logits), atol=1e-6)

    def test_the_token_loss_is_the_cross_entropy_of_the_next_token(self):
        from torch.nn import functional

        logits, labels = self._case()
        step = step_from_logits(logits, labels)
        target = functional.pad(labels, (0, 1), value=-100)[:, 1:]
        want = functional.cross_entropy(
            logits.reshape(-1, 64), target.reshape(-1), ignore_index=-100, reduction="none"
        ).view(2, 9)
        assert torch.allclose(step.token_loss, want, atol=1e-6)
        assert float(step.token_loss[target == -100].abs().sum()) == 0.0

    def test_the_counts_are_those_of_the_shifted_labels(self):
        logits, labels = self._case()
        target = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
        # make the model right about exactly the first two labelled positions
        logits = torch.full_like(logits, -5.0)
        hits = (target != -100).nonzero()[:2]
        for row, col in hits.tolist():
            logits[row, col, target[row, col]] = 5.0
        step = step_from_logits(logits, labels)
        assert step.correct.item() == 2
        assert step.labelled.item() == int((target != -100).sum())

    def test_it_carries_no_graph(self):
        logits, labels = self._case()
        logits.requires_grad_(True)
        step = step_from_logits(logits, labels)
        assert not step.token_entropy.requires_grad and not step.token_loss.requires_grad
