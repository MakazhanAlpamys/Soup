"""#1693: the output head and the causal-LM loss in token chunks.

The reference throughout is what a step does today: the head over the whole
sequence, then transformers' ``ForCausalLMLoss`` over the full logits. The
chunked form computes the same per-token terms; the order of the final sum and
the shape of each matmul differ, so the comparisons carry a tolerance. It is
``_RTOL`` of the largest value compared, and the tests that can be exact say so.
"""

from __future__ import annotations

import pytest

from tests._windows_ci import skip_on_windows_ci

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

# Largest difference seen on CPU, relative to the largest gradient: 7.2e-08 in
# float32 (2 x 67 x 101, chunk 16) and 0 in bfloat16 over the same shapes.
_RTOL = 1e-5

# The bfloat16 cases run a plain CPU bf16 matmul (``torch.nn.Linear.forward`` on the head,
# in ``_full``), and part of GitHub's windows-latest fleet dies on it with 0xc000001d
# (illegal instruction): the interpreter exits, the xdist worker is replaced and the cell
# later times out (#382, #1314; run 40014485149, job 120622979972, the crash frame is the
# head's forward inside _full). Only the bfloat16 cases carry the marker, and it skips them on
# a Windows CI runner only: the float32 cases, a developer's Windows box, ubuntu and macos
# run everything. Each parametrisation spells its two pytest.param out, with an id, so that
# tests/test_issue382_windows_ci_guard_is_shared.py can read which ones carry the marker.


def _full(head, hidden, labels, **kwargs):
    from transformers.loss.loss_utils import ForCausalLMLoss

    return ForCausalLMLoss(head(hidden), labels, head.out_features, **kwargs)


def _case(dtype=torch.float32, batch=2, seq=67, hidden_size=32, vocab=101, seed=0):
    """A frozen head, the leaf the hidden states are computed from, and labels
    with a masked prompt in the first row and padding at the end of the last."""
    torch.manual_seed(seed)
    head = torch.nn.Linear(hidden_size, vocab, bias=False).to(dtype)
    head.weight.requires_grad_(False)
    leaf = torch.randn(batch, seq, hidden_size)
    labels = torch.randint(0, vocab, (batch, seq))
    labels[0, : seq // 3] = -100
    labels[-1, -5:] = -100
    return head, leaf, labels


def _both(dtype, chunk, loss_kwargs=None, **shape):
    """(loss, gradient on the leaf) of the full step and of the chunked one."""
    from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

    head, source, labels = _case(dtype, **shape)
    results = []
    for chunked in (False, True):
        leaf = source.clone().requires_grad_(True)
        hidden = (leaf * 1.5).to(dtype)  # stands for the decoder
        if chunked:
            loss, _ = chunked_causal_lm_loss(hidden, head, labels, chunk, **(loss_kwargs or {}))
        else:
            loss = _full(head, hidden, labels, **(loss_kwargs or {}))
        loss.backward()
        results.append((loss.detach(), leaf.grad))
    return results


def biased_module(head, bias):
    """``head`` plus a constant, with the ``out_features`` ``_full`` reads."""

    def forward(hidden):
        return head(hidden) + bias

    forward.out_features = head.out_features
    return forward


def _close(got, want):
    got, want = got.cpu().double(), want.cpu().double()  # MPS has no float64
    return float((got - want).abs().max()) <= _RTOL * float(want.abs().max())


class TestTheLossAndItsGradient:
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float32, id="fp32"),
            pytest.param(torch.bfloat16, id="bf16", marks=skip_on_windows_ci),
        ],
    )
    @pytest.mark.parametrize(
        ("chunk", "shape"),
        [
            (16, {}),  # 67 is not a multiple of 16
            (1, {}),
            (67, {}),  # one chunk: the whole sequence
            (1000, {}),  # a chunk longer than the sequence
            (128, {"batch": 1, "seq": 513, "hidden_size": 64, "vocab": 4099}),
        ],
    )
    def test_they_equal_the_full_logits_step_within_the_tolerance(self, dtype, chunk, shape):
        (full_loss, full_grad), (loss, grad) = _both(dtype, chunk, **shape)

        assert loss.dtype == torch.float32 and loss.shape == ()
        assert _close(loss, full_loss), (float(loss), float(full_loss))
        assert float(full_grad.abs().max()) > 0  # the premise: there is a gradient
        assert _close(grad, full_grad), float((grad - full_grad).abs().max())

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float32, id="fp32"),
            pytest.param(torch.bfloat16, id="bf16", marks=skip_on_windows_ci),
        ],
    )
    def test_one_chunk_is_the_full_step_exactly(self, dtype):
        """With nothing to reorder, there is nothing to round differently."""
        (full_loss, full_grad), (loss, grad) = _both(dtype, 67)

        assert torch.equal(loss, full_loss)
        assert torch.equal(grad, full_grad)

    @pytest.mark.parametrize("items", [torch.tensor(431), 431])
    def test_num_items_in_batch_replaces_the_mean(self, items):
        """Under gradient accumulation the trainer passes the token count of
        the whole accumulated batch, and the loss is a sum over it."""
        kwargs = {"num_items_in_batch": items}
        shape = {"batch": 3, "seq": 200, "hidden_size": 64, "vocab": 499}
        (full_loss, full_grad), (loss, grad) = _both(torch.float32, 64, kwargs, **shape)
        (mean_loss, _), _ = _both(torch.float32, 64, **shape)

        assert _close(loss, full_loss) and _close(grad, full_grad)
        assert not _close(loss, mean_loss)  # and it is not the mean

    def test_shift_labels_are_used_as_given(self):
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, hidden, labels = _case()
        given = torch.roll(labels, 3, dims=1)

        loss, step = chunked_causal_lm_loss(hidden, head, labels, 16, shift_labels=given)

        assert _close(loss, _full(head, hidden, labels, shift_labels=given))
        assert not _close(loss, _full(head, hidden, labels))
        assert int(step.labelled) == int((given != -100).sum())

    def test_the_last_position_is_scored_against_nothing(self):
        """``ForCausalLMLoss`` shifts: position t predicts token t + 1."""
        from soup_cli.utils.stream_loss_chunk import shift_for_next_token

        labels = torch.tensor([[5, 6, -100, 7]])

        assert shift_for_next_token(labels).tolist() == [[6, -100, 7, -100]]

    def test_a_batch_with_no_label_is_nan_as_it_is_today(self):
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, hidden, labels = _case()
        labels = torch.full_like(labels, -100)

        loss, step = chunked_causal_lm_loss(hidden, head, labels, 16)

        assert torch.isnan(_full(head, hidden, labels)) and torch.isnan(loss)
        assert int(step.labelled) == 0

    def test_the_gradient_scales_with_what_is_done_to_the_loss(self):
        """The stored gradient is multiplied by the incoming one, not returned
        as it is: a loss scaled by gradient accumulation scales its gradient."""
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, source, labels = _case()
        grads = []
        for factor in (1.0, 0.25):
            leaf = source.clone().requires_grad_(True)
            loss, _ = chunked_causal_lm_loss(leaf * 1.5, head, labels, 16)
            (loss * factor).backward()
            grads.append(leaf.grad)

        assert torch.allclose(grads[1], grads[0] * 0.25, rtol=1e-6, atol=0)


class TestNoFullLogits:
    def _spied(self, head):
        seen = []

        def spy(piece):
            seen.append((tuple(piece.shape), piece.requires_grad, torch.is_grad_enabled()))
            return head(piece)

        return spy, seen

    def test_the_head_never_sees_more_than_a_chunk(self):
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, leaf, labels = _case()
        spy, seen = self._spied(head)

        chunked_causal_lm_loss(leaf.requires_grad_(True), spy, labels, 16)

        assert [shape for shape, _, _ in seen] == [(2, 16, 32)] * 4 + [(2, 3, 32)]

    def test_without_gradients_no_graph_is_built(self):
        """The eval loss: the same chunks, and nothing to differentiate."""
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, leaf, labels = _case()
        spy, seen = self._spied(head)

        with torch.no_grad():
            loss, _ = chunked_causal_lm_loss(leaf.requires_grad_(True), spy, labels, 16)

        assert loss.grad_fn is None and not loss.requires_grad
        assert _close(loss, _full(head, leaf, labels).detach())
        assert all(not needs_grad and not enabled for _, needs_grad, enabled in seen)

    def test_hidden_states_nothing_trains_through_build_no_graph_either(self):
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, leaf, labels = _case()
        spy, seen = self._spied(head)

        loss, _ = chunked_causal_lm_loss(leaf, spy, labels, 16)

        assert not loss.requires_grad
        assert all(not needs_grad for _, needs_grad, _ in seen)

    def test_with_gradients_each_chunk_is_its_own_leaf(self):
        """What keeps a chunk's logits from outliving it: the chunk is cut
        from the graph, and its gradient is taken before the next one runs."""
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, leaf, labels = _case()
        pieces = []

        def spy(piece):
            pieces.append(piece)
            return head(piece)

        loss, _ = chunked_causal_lm_loss(leaf.requires_grad_(True) * 1.5, spy, labels, 16)

        assert all(piece.is_leaf and piece.requires_grad for piece in pieces)
        assert type(loss.grad_fn).__name__ == "_ChunkedHeadLossBackward"


class TestTheMetrics:
    """TRL's ``SFTTrainer.compute_loss`` logs two numbers it reads from
    ``outputs.logits``. These are its formulas, on the full logits."""

    def test_the_token_entropy_is_trls(self):
        from trl.trainer.utils import entropy_from_logits

        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, hidden, labels = _case()

        _, step = chunked_causal_lm_loss(hidden, head, labels, 16)

        want = entropy_from_logits(head(hidden))
        assert step.token_entropy.shape == (2, 67)
        assert torch.allclose(step.token_entropy, want, rtol=1e-5, atol=1e-6)

    def test_the_accuracy_counts_are_trls(self):
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, hidden, labels = _case(vocab=7)  # few words, so some are right

        _, step = chunked_causal_lm_loss(hidden, head, labels, 16)

        predictions = head(hidden)[..., :-1, :].argmax(dim=-1)
        shifted = labels[..., 1:]
        mask = shifted != -100
        assert int(step.labelled) == int(mask.sum())
        assert int(step.correct) == int(((predictions == shifted) & mask).sum())
        assert 0 < int(step.correct) < int(step.labelled)  # the premise

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float32, id="fp32"),
            pytest.param(torch.bfloat16, id="bf16", marks=skip_on_windows_ci),
        ],
    )
    def test_the_token_losses_give_the_rewind_logs_row_losses(self, dtype):
        """``row_losses`` is what the rewind log computes from the logits: the
        mean token cross-entropy of each row, in the logits' own dtype."""
        from soup_cli.trainer.rewind_hf import row_losses
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, hidden, labels = _case(dtype)
        hidden = hidden.to(dtype)

        _, step = chunked_causal_lm_loss(hidden, head, labels, 16)

        want_loss, want_tokens = row_losses(head(hidden), labels)
        counts = (labels[:, 1:] != -100).sum(dim=1)
        assert step.token_loss.shape == (2, 67) and step.token_loss.dtype == torch.float32
        assert counts.tolist() == want_tokens
        assert (step.token_loss.sum(dim=1) / counts).tolist() == pytest.approx(want_loss, rel=1e-5)
        assert float(step.token_loss[0, : 67 // 3 - 1].abs().max()) == 0.0  # the masked prompt
        assert float(step.token_loss[:, -1].abs().max()) == 0.0  # the last position

    def test_another_ignore_index_is_ignored_by_the_loss_and_the_accuracy(self):
        """With an ignore index that is also a token id, the head can predict
        it: such a position is neither a loss term nor a correct answer."""
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, hidden, labels = _case(vocab=7)
        labels[labels == -100] = 3
        always_three = torch.zeros(7)
        always_three[3] = 50.0

        def biased(piece):
            return head(piece) + always_three

        loss, step = chunked_causal_lm_loss(hidden, biased, labels, 16, ignore_index=3)

        shifted = labels[..., 1:]
        assert int((shifted == 3).sum()) > 0  # the premise
        assert int(step.labelled) == int((shifted != 3).sum())
        assert int(step.correct) == 0
        want = _full(biased_module(head, always_three), hidden, labels, ignore_index=3)
        assert _close(loss, want)
        # And with the head as it is, where a position labelled 3 would cost
        # something if it were scored:
        plain, _ = chunked_causal_lm_loss(hidden, head, labels, 16, ignore_index=3)
        assert _close(plain, _full(head, hidden, labels, ignore_index=3))
        assert not _close(plain, _full(head, hidden, labels))

    def test_the_metrics_carry_no_gradient(self):
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, leaf, labels = _case()

        _, step = chunked_causal_lm_loss(leaf.requires_grad_(True), head, labels, 16)

        for value in (step.token_entropy, step.token_loss, step.correct):
            assert not value.requires_grad


class TestWhatIsRefused:
    @pytest.mark.parametrize("chunk", [0, -4, True, 16.0, "16", None])
    def test_a_chunk_that_is_not_a_positive_integer(self, chunk):
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, hidden, labels = _case()

        with pytest.raises(ValueError, match="chunk must be a positive integer"):
            chunked_causal_lm_loss(hidden, head, labels, chunk)

    def test_labels_of_another_length(self):
        from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

        head, hidden, labels = _case()

        with pytest.raises(ValueError, match="do not match hidden states"):
            chunked_causal_lm_loss(hidden, head, labels[:, :-1], 16)


# ─── The hook on the causal-LM forward ───────────────────────────────────────


def _tiny_causal_lm(tie=False):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    model.get_output_embeddings().weight.requires_grad_(False)
    return model


def _ids(seq=37):
    generator = torch.Generator().manual_seed(17)
    ids = torch.randint(0, 64, (2, seq), generator=generator)
    labels = ids.clone()
    labels[0, :9] = -100
    labels[1, -4:] = -100
    return ids, labels


def _step(model, **kwargs):
    """(loss, gradients of the decoder's trainable weights) of one labelled step."""
    ids, labels = _ids()
    model.zero_grad(set_to_none=True)
    outputs = model(input_ids=ids, labels=labels, **kwargs)
    outputs.loss.backward()
    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return outputs, grads


class TestTheHook:
    @pytest.mark.parametrize("tie", [False, True])
    @pytest.mark.parametrize("kwargs", [{}, {"num_items_in_batch": torch.tensor(50)}])
    def test_a_labelled_forward_gives_the_loss_and_no_logits(self, tie, kwargs):
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        model = _tiny_causal_lm(tie)
        plain, plain_grads = _step(model, **kwargs)
        ChunkedHeadLoss(model, 8)

        outputs, grads = _step(model, **kwargs)

        assert outputs.logits is None and "logits" not in outputs.keys()
        assert plain.logits.shape == (2, 37, 64)  # the premise
        assert _close(outputs.loss.detach(), plain.loss.detach())
        assert grads.keys() == plain_grads.keys() and grads
        for name, want in plain_grads.items():
            assert _close(grads[name], want), name

    def test_num_items_in_batch_reaches_the_loss(self):
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        model = _tiny_causal_lm()
        ChunkedHeadLoss(model, 8)

        mean, _ = _step(model)
        summed, _ = _step(model, num_items_in_batch=torch.tensor(50))

        assert not _close(summed.loss.detach(), mean.loss.detach())

    def test_shift_labels_given_to_the_forward_reach_the_loss(self):
        """Context parallelism hands the model labels it has already shifted."""
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        model = _tiny_causal_lm()
        ids, labels = _ids()
        given = torch.roll(labels, 3, dims=1)
        with torch.no_grad():
            want = model(input_ids=ids, labels=labels, shift_labels=given).loss
            unshifted = model(input_ids=ids, labels=labels).loss
            ChunkedHeadLoss(model, 8)
            got = model(input_ids=ids, labels=labels, shift_labels=given).loss

        assert _close(got, want) and not _close(got, unshifted)

    def test_a_forward_without_labels_is_untouched(self):
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        model = _tiny_causal_lm()
        ids, _ = _ids()
        with torch.no_grad():
            want = model(input_ids=ids).logits
            hook = ChunkedHeadLoss(model, 8)
            got = model(input_ids=ids)

        assert torch.equal(got.logits, want)
        assert hook.take_step() is None

    def test_the_eval_loss_is_chunked_without_a_graph(self):
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        model = _tiny_causal_lm()
        ids, labels = _ids()
        with torch.no_grad():
            want = model(input_ids=ids, labels=labels).loss
            ChunkedHeadLoss(model, 8)
            outputs = model(input_ids=ids, labels=labels)

        assert outputs.logits is None and outputs.loss.grad_fn is None
        assert _close(outputs.loss, want)

    def test_the_step_is_handed_over_once(self):
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        model = _tiny_causal_lm()
        hook = ChunkedHeadLoss(model, 8)
        _step(model)

        step = hook.take_step()

        assert step.token_entropy.shape == (2, 37) and int(step.labelled) == 60
        assert hook.take_step() is None

    def test_under_peft_the_hook_sits_on_the_causal_lm(self):
        from peft import LoraConfig, get_peft_model

        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss, causal_lm_of

        base = _tiny_causal_lm()
        model = get_peft_model(base, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
        ids, labels = _ids()
        plain = model(input_ids=ids, labels=labels).loss.detach()

        assert causal_lm_of(model) is base and causal_lm_of(base) is base
        ChunkedHeadLoss(causal_lm_of(model), 8)
        outputs = model(input_ids=ids, labels=labels)

        assert outputs.logits is None and _close(outputs.loss.detach(), plain)

    def test_remove_puts_both_forwards_back(self):
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        model = _tiny_causal_lm()
        head = model.get_output_embeddings()
        ids, labels = _ids()
        hook = ChunkedHeadLoss(model, 8)
        assert "forward" in model.__dict__ and "forward" in head.__dict__  # the premise

        hook.remove()

        assert "forward" not in model.__dict__ and "forward" not in head.__dict__
        assert model(input_ids=ids, labels=labels).logits.shape == (2, 37, 64)

    def test_remove_restores_a_forward_that_was_already_set_on_the_instance(self):
        """accelerate's hooks set ``module.forward`` on the instance; removing
        ours must leave theirs."""
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        model = _tiny_causal_lm()
        original = model.forward
        calls = []

        def theirs(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        model.forward = theirs
        hook = ChunkedHeadLoss(model, 8)
        _step(model)
        assert calls == [1]  # ours wraps theirs

        hook.remove()

        assert model.__dict__["forward"] is theirs

    def test_a_failed_forward_does_not_leave_the_head_held(self):
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        model = _tiny_causal_lm()
        ids, labels = _ids()
        ChunkedHeadLoss(model, 8)

        with pytest.raises(Exception):  # noqa: B017 -- whatever the model raises
            model(input_ids=ids, labels=labels, attention_mask="not a mask")

        with torch.no_grad():
            assert model(input_ids=ids).logits.shape == (2, 37, 64)

    def test_a_trainable_head_is_refused_with_the_reason(self):
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        model = _tiny_causal_lm()
        model.get_output_embeddings().weight.requires_grad_(True)

        with pytest.raises(ValueError, match="the output head must be frozen; trainable: weight"):
            ChunkedHeadLoss(model, 8)
        assert "forward" not in model.__dict__

    def test_a_soft_capped_architecture_is_refused_when_the_hook_is_built(self):
        """Gemma 2's config says so, so the refusal need not wait for a step."""
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        model = _tiny_causal_lm()
        model.config.final_logit_softcapping = 30.0

        with pytest.raises(ValueError, match=r"soft-caps its logits.*final_logit_softcapping=30.0"):
            ChunkedHeadLoss(model, 8)
        assert "forward" not in model.__dict__

    def test_a_model_without_an_output_head_is_refused(self):
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        class _Headless(torch.nn.Module):
            def get_output_embeddings(self):
                return None

        with pytest.raises(ValueError, match="needs a model with an output head"):
            ChunkedHeadLoss(_Headless(), 8)

    @pytest.mark.parametrize("chunk", [0, True, 8.0])
    def test_a_chunk_that_is_not_a_positive_integer_is_refused(self, chunk):
        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        with pytest.raises(ValueError, match="chunk must be a positive integer"):
            ChunkedHeadLoss(_tiny_causal_lm(), chunk)

    @pytest.mark.parametrize("after_the_head", ["soft-capped", "head not called"])
    def test_a_forward_that_does_not_return_the_heads_result_is_refused(self, after_the_head):
        """Gemma soft-caps its logits after the head. The loss rebuilt from the
        hidden states would then be the loss of another model, so say no."""
        from transformers.modeling_outputs import CausalLMOutput

        from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.body = torch.nn.Linear(8, 8)
                self.head = torch.nn.Linear(8, 11, bias=False)
                self.head.weight.requires_grad_(False)

            def get_output_embeddings(self):
                return self.head

            def forward(self, hidden, labels=None):
                if after_the_head == "soft-capped":
                    logits = torch.tanh(self.head(self.body(hidden)) / 30.0) * 30.0
                else:
                    logits = torch.nn.functional.linear(self.body(hidden), self.head.weight)
                return CausalLMOutput(logits=logits)

        model = _Model()
        ChunkedHeadLoss(model, 4)
        hidden = torch.randn(2, 6, 8)

        with pytest.raises(ValueError, match="does not hand the output head's result back"):
            model(hidden, labels=torch.randint(0, 11, (2, 6)))
        assert model(hidden).logits.shape == (2, 6, 11)  # and without labels it still runs


# ─── What the trainer logs, and the rewind log ───────────────────────────────


def _fake_trainer(**without):
    from collections import defaultdict
    from types import SimpleNamespace

    trainer = SimpleNamespace(
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        _total_train_tokens=10,
        accelerator=SimpleNamespace(gather_for_metrics=lambda value: value),
    )
    for name in without:
        delattr(trainer, name)
    return trainer


def _a_step(correct=3, labelled=4):
    from soup_cli.utils.stream_loss_chunk import ChunkedStep

    return ChunkedStep(
        token_entropy=torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
        token_loss=torch.zeros(2, 2),
        correct=torch.tensor(correct),
        labelled=torch.tensor(labelled),
    )


class TestTheLoggedColumns:
    """``_log_trl_metrics`` by hand, with TRL's formulas: the entropy is the
    mean over the attended positions, the accuracy over the labelled ones."""

    _MASK = {"attention_mask": torch.tensor([[1, 1], [1, 0]])}

    def test_a_training_step(self):
        from soup_cli.utils.stream_loss_chunk import _log_trl_metrics

        trainer = _fake_trainer()

        _log_trl_metrics(trainer, "train", self._MASK, _a_step())

        assert dict(trainer._metrics["train"]) == {
            "entropy": [3.0],  # (1 + 3 + 5) / 3: the padded position is left out
            "num_tokens": [13],
            "mean_token_accuracy": [0.75],
        }
        assert trainer._total_train_tokens == 13 and not trainer._metrics["eval"]

    def test_an_eval_step_counts_no_training_tokens(self):
        from soup_cli.utils.stream_loss_chunk import _log_trl_metrics

        trainer = _fake_trainer()

        _log_trl_metrics(trainer, "eval", self._MASK, _a_step())

        assert trainer._total_train_tokens == 10
        assert trainer._metrics["eval"]["num_tokens"] == [10]
        assert trainer._metrics["eval"]["entropy"] == [3.0]

    def test_without_an_attention_mask_the_entropy_is_the_plain_mean(self):
        """Padding-free batches carry ``position_ids`` instead."""
        from soup_cli.utils.stream_loss_chunk import _log_trl_metrics

        trainer = _fake_trainer()

        _log_trl_metrics(trainer, "train", {"position_ids": torch.zeros(2, 2)}, _a_step())

        assert trainer._metrics["train"]["entropy"] == [4.0]
        assert trainer._total_train_tokens == 10

    def test_a_batch_with_no_label_has_accuracy_zero_not_nan(self):
        from soup_cli.utils.stream_loss_chunk import _log_trl_metrics

        trainer = _fake_trainer()

        _log_trl_metrics(trainer, "train", self._MASK, _a_step(correct=0, labelled=0))

        assert trainer._metrics["train"]["mean_token_accuracy"] == [0.0]

    @pytest.mark.parametrize("missing", ["_metrics", "accelerator"])
    def test_a_trl_without_the_metrics_store_logs_nothing(self, missing):
        from soup_cli.utils.stream_loss_chunk import _log_trl_metrics

        trainer = _fake_trainer(**{missing: True})

        _log_trl_metrics(trainer, "train", self._MASK, _a_step())

        assert not getattr(trainer, "_metrics", {"train": {}})["train"]
        assert trainer._total_train_tokens == 10

    def test_the_logged_columns_are_gathered_across_ranks_as_trl_gathers_them(self):
        from soup_cli.utils.stream_loss_chunk import _log_trl_metrics

        trainer = _fake_trainer()
        # Two ranks: this one's value, and a second rank that saw twice as much.
        trainer.accelerator.gather_for_metrics = lambda value: torch.stack([value, value * 2])

        _log_trl_metrics(trainer, "train", self._MASK, _a_step())

        assert trainer._metrics["train"]["entropy"] == [4.5]  # mean(3.0, 6.0)
        assert trainer._metrics["train"]["num_tokens"] == [19]  # 10 + (3 + 6) attended tokens
        assert trainer._metrics["train"]["mean_token_accuracy"] == [0.75]  # (3 + 6) / (4 + 8)


def test_the_chunked_trainer_class_leaves_other_steps_to_its_base():
    """No hook on the trainer, or a batch without labels: the base class's own
    ``compute_loss`` runs, untouched."""
    from soup_cli.utils.stream_loss_chunk import HOOK_ATTR, make_chunked_loss_trainer_class

    class _Base:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return ("base", return_outputs, num_items_in_batch)

    trainer = make_chunked_loss_trainer_class(_Base)()

    assert type(trainer).__name__ == "ChunkedLoss_Base"
    assert trainer.compute_loss(None, {"labels": 1}, True, 5) == ("base", True, 5)
    setattr(trainer, HOOK_ATTR, object())
    assert trainer.compute_loss(None, {"input_ids": 1}) == ("base", False, None)


def test_the_rewind_logs_row_losses_from_token_losses():
    """A row's mean over its labelled tokens; a row with none is (0.0, 0)."""
    from soup_cli.trainer.rewind_hf import row_losses_from_token_losses

    token_loss = torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    # The first label has no position that predicts it, so it is not counted.
    labels = torch.tensor([[7, 5, 6], [-100, -100, -100]])

    assert row_losses_from_token_losses(token_loss, labels) == ([1.5, 0.0], [2, 0])


# ─── The option ──────────────────────────────────────────────────────────────


def _cfg(task="sft", **training):
    from soup_cli.config.schema import SoupConfig

    data = {"train": "tests/fixtures/sample_train.jsonl"}
    if task != "sft":
        data["format"] = "dpo"
    return SoupConfig(
        base="test/model", task=task, data=data, training={"batch_size": 1, **training}
    )


def _refusal(task="sft", **training):
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        _cfg(task, **training)
    return excinfo.value.errors()[0]["msg"]


class TestTheOption:
    def test_it_is_off_by_default(self):
        assert _cfg().training.stream_loss_chunk is None
        assert _cfg(stream_layers=True).training.stream_loss_chunk is None

    def test_it_loads_with_layer_streaming_on_sft(self):
        assert _cfg(stream_layers=True, stream_loss_chunk=128).training.stream_loss_chunk == 128

    def test_without_stream_layers_it_is_refused(self):
        message = _refusal(stream_loss_chunk=128)

        assert "training.stream_loss_chunk" in message
        assert "stream_layers is false" in message

    @pytest.mark.parametrize("task", ["dpo", "orpo", "simpo", "kto"])
    def test_a_preference_task_is_refused_with_the_reason(self, task):
        # kto refuses a batch of one before anything else is looked at.
        message = _refusal(task, batch_size=2, stream_layers=True, stream_loss_chunk=128)

        assert f"training.stream_loss_chunk supports task='sft' only; got task='{task}'" in message
        assert "per-token log-probs" in message

    def test_the_raft_loss_is_refused_with_the_reason(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import SoupConfig

        kwargs = {
            "base": "test/model",
            "task": "sft",
            "data": {"train": "tests/fixtures/sample_train.jsonl", "format": "raft"},
        }
        SoupConfig(**kwargs, training={"batch_size": 1, "stream_layers": True})  # the premise
        with pytest.raises(ValidationError) as excinfo:
            SoupConfig(
                **kwargs,
                training={"batch_size": 1, "stream_layers": True, "stream_loss_chunk": 128},
            )
        message = excinfo.value.errors()[0]["msg"]
        assert "cannot be combined with data.format='raft'" in message
        assert "full logits" in message

    def test_use_cut_ce_and_the_chunk_refuse_each_other(self):
        """Two mechanisms over the same loss: refused, with both fields named."""
        from pydantic import ValidationError

        # use_cut_ce alone is refused under streaming (#2023); the pair is
        # refused first, with both fields named.
        with pytest.raises(ValidationError, match="cannot be combined with training.use_cut_ce"):
            _cfg(stream_layers=True, use_cut_ce=True)
        _cfg(stream_layers=True, stream_loss_chunk=128)
        with pytest.raises(ValidationError) as excinfo:
            _cfg(stream_layers=True, stream_loss_chunk=128, use_cut_ce=True)
        message = excinfo.value.errors()[0]["msg"]
        assert "training.stream_loss_chunk and training.use_cut_ce cannot be combined" in message
        assert "Keep stream_loss_chunk and set use_cut_ce=false" in message

    def test_the_rewind_log_stays_on_beside_it(self):
        """``rewind_log`` is on by default: refusing the pair would switch the
        flight recorder off for everyone who sets a chunk."""
        training = _cfg(stream_layers=True, stream_loss_chunk=128).training

        assert training.rewind_log is True

    @pytest.mark.parametrize("chunk", [0, -1, 15, 8193])
    def test_a_chunk_outside_the_bounds_is_refused(self, chunk):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="stream_loss_chunk"):
            _cfg(stream_layers=True, stream_loss_chunk=chunk)

    @pytest.mark.parametrize("chunk", [16, 8192])
    def test_the_bounds_themselves_load(self, chunk):
        assert _cfg(stream_layers=True, stream_loss_chunk=chunk).training.stream_loss_chunk == chunk

    @pytest.mark.parametrize("chunk", [True, False])
    def test_a_bool_is_not_a_chunk(self, chunk):
        """``true`` would otherwise load as a chunk of one token."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="must be an int .tokens., not bool"):
            _cfg(stream_layers=True, stream_loss_chunk=chunk)


# ─── Through the real streamed trainer ───────────────────────────────────────


def _train(tmp_path, monkeypatch, *, chunk, tie=False, steps=2, **training):
    """``steps`` real sft steps on the tiny streamed Llama, on the device the
    trainer picks (forcing CPU beside an accelerator is a mismatch no user
    has): float32 on CPU, which is what CI runs."""
    import json

    from tests.test_v07204 import _build_streamed_wrapper, _randomise_lora_b

    torch.manual_seed(1234)
    tmp_path.mkdir(parents=True, exist_ok=True)
    if chunk is not None:
        training["stream_loss_chunk"] = chunk
    wrapper, _, _ = _build_streamed_wrapper(
        tmp_path, monkeypatch, task="sft", n_layers=2, tie=tie, **training
    )
    # PEFT starts `lora_B` at zero, which leaves most gradients zero.
    _randomise_lora_b(wrapper.model)
    hook = wrapper._stream_loss_chunk_hook
    head_inputs, forward_kwargs = [], set()
    if hook is not None:
        head, forward = hook._head_forward, hook._forward

        def spy(hidden, *args, **kwargs):
            head_inputs.append(tuple(hidden.shape))
            return head(hidden, *args, **kwargs)

        def forward_spy(*args, **kwargs):
            forward_kwargs.update(key for key in kwargs if key.startswith("_"))
            forward_kwargs.add(("use_cache", kwargs.get("use_cache")))
            return forward(*args, **kwargs)

        hook._head_forward, hook._forward = spy, forward_spy

    from transformers import TrainerCallback

    grads = []

    class _Grads(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            grads.append(
                [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
            )

    wrapper.trainer.add_callback(_Grads())
    wrapper.trainer.args.max_steps = steps
    try:
        wrapper.trainer.train()
        weights = {
            name: param.detach().clone()
            for name, param in wrapper.model.named_parameters()
            if param.requires_grad
        }
        # The rows it trained on, as an evaluation set: `prediction_step` goes
        # through `compute_loss` with the model in eval mode and no gradients.
        evaluation = wrapper.trainer.evaluate(eval_dataset=wrapper.trainer.train_dataset)
    finally:
        wrapper._close_stream_runtime()
    rewind = tmp_path / "out" / "rewind.jsonl"
    history = wrapper.trainer.state.log_history
    return {
        "eval_loss": evaluation["eval_loss"],
        "eval_log": next(entry for entry in history if "eval_loss" in entry),
        "train_tokens": wrapper.trainer._total_train_tokens,
        "forward_kwargs": forward_kwargs,
        "trainer": type(wrapper.trainer).__name__,
        "grads": grads,
        "weights": weights,
        "logs": [entry for entry in wrapper.trainer.state.log_history if "loss" in entry],
        "rewind": [json.loads(line) for line in rewind.read_text().splitlines()]
        if rewind.exists()
        else None,
        "head_inputs": head_inputs,
    }


class TestThroughTheRealTrainer:
    """Two runs from the same seed, one of them chunked. The sequences here are
    17 tokens long, so a chunk of 16 splits each of them in two."""

    @pytest.fixture(params=[False, True], ids=["untied", "tied"])
    def arms(self, request, tmp_path, monkeypatch):
        plain = _train(tmp_path / "plain", monkeypatch, chunk=None, tie=request.param)
        chunked = _train(tmp_path / "chunked", monkeypatch, chunk=16, tie=request.param)
        return plain, chunked

    def test_the_head_only_ever_runs_on_a_chunk(self, arms):
        plain, chunked = arms

        assert plain["head_inputs"] == []  # no hook, nothing to spy on
        lengths = [shape[1] for shape in chunked["head_inputs"]]
        assert lengths and max(lengths) == 16
        assert min(lengths) < 16  # the premise: a sequence was really split

    def test_the_gradients_and_the_trained_weights_agree(self, arms):
        plain, chunked = arms

        assert len(plain["grads"]) == len(chunked["grads"]) == 2
        seen_nonzero = False
        for step, (want, got) in enumerate(zip(plain["grads"], chunked["grads"])):
            assert len(want) == len(got)
            for index, (a, b) in enumerate(zip(want, got)):
                if float(a.abs().max()) == 0.0:
                    assert float(b.abs().max()) == 0.0, (step, index)
                    continue
                seen_nonzero = True
                assert _close(b, a), (step, index)
        assert seen_nonzero, "vacuous: every gradient was zero"
        assert plain["weights"].keys() == chunked["weights"].keys()
        for name, want in plain["weights"].items():
            assert _close(chunked["weights"][name], want), name

    def test_the_log_keeps_trls_columns_with_trls_numbers(self, arms):
        """``entropy``, ``num_tokens`` and ``mean_token_accuracy`` come from
        TRL's own ``compute_loss`` in the plain run and from the chunk loop in
        the other."""
        plain, chunked = arms

        assert len(plain["logs"]) == len(chunked["logs"]) == 2
        for want, got in zip(plain["logs"], chunked["logs"]):
            assert got.keys() == want.keys()
            assert {"entropy", "num_tokens", "mean_token_accuracy"} <= got.keys()
            assert got["num_tokens"] == want["num_tokens"] > 0
            for column in ("loss", "grad_norm", "entropy", "mean_token_accuracy"):
                assert got[column] == pytest.approx(want[column], rel=_RTOL, abs=1e-7), column

    def test_the_rewind_log_records_the_same_rows(self, arms):
        plain, chunked = arms

        want = [entry for entry in plain["rewind"] if entry["kind"] == "batch"]
        got = [entry for entry in chunked["rewind"] if entry["kind"] == "batch"]
        assert len(want) == len(got) == 2
        for a, b in zip(want, got):
            for field in ("step", "rows", "row_tokens"):
                assert b[field] == a[field], field
            assert b["row_loss"] == pytest.approx(a["row_loss"], rel=_RTOL)
            assert b["row_loss"][0] > 0

    def test_the_eval_loss_agrees(self, arms):
        """With no logits in the outputs, the evaluation loop still returns."""
        plain, chunked = arms

        assert chunked["eval_loss"] == pytest.approx(plain["eval_loss"], rel=_RTOL)
        assert plain["eval_loss"] > 0

    def test_evaluation_is_booked_as_evaluation(self, arms):
        """TRL keeps eval metrics apart and counts training tokens only."""
        plain, chunked = arms

        assert chunked["train_tokens"] == plain["train_tokens"] > 0
        assert chunked["eval_log"].keys() == plain["eval_log"].keys()
        for column in ("eval_entropy", "eval_mean_token_accuracy", "eval_num_tokens"):
            assert chunked["eval_log"][column] == pytest.approx(
                plain["eval_log"][column], rel=_RTOL, abs=1e-7
            ), column

    def test_the_model_is_called_as_trl_calls_it(self, arms):
        """``_prediction_loss_only`` is the trainer's own note and is taken out
        before the model is called; the cache is off, as in TRL's step."""
        _, chunked = arms

        assert chunked["forward_kwargs"] == {("use_cache", False)}

    def test_under_gradient_accumulation_with_a_batch_of_two(self, tmp_path, monkeypatch):
        """The trainer then passes ``num_items_in_batch``: the loss is a sum
        over the tokens of the whole accumulated batch, not a mean per call."""
        training = {"batch_size": 2, "gradient_accumulation_steps": 2}
        plain = _train(tmp_path / "plain", monkeypatch, chunk=None, **training)
        chunked = _train(tmp_path / "chunked", monkeypatch, chunk=16, **training)

        assert {shape[0] for shape in chunked["head_inputs"]} == {2}
        assert len(plain["grads"]) == len(chunked["grads"]) == 2
        for want, got in zip(plain["grads"], chunked["grads"]):
            for a, b in zip(want, got):
                assert _close(b, a) or float(a.abs().max()) == 0.0
        for want, got in zip(plain["logs"], chunked["logs"]):
            assert got["num_tokens"] == want["num_tokens"]
            for column in ("loss", "grad_norm", "entropy", "mean_token_accuracy"):
                assert got[column] == pytest.approx(want[column], rel=_RTOL, abs=1e-7), column
        batches = [entry for entry in chunked["rewind"] if entry["kind"] == "batch"]
        assert [len(entry["rows"]) for entry in batches] == [2, 2, 2, 2]

    def test_with_the_rewind_log_off_the_plain_trainer_is_chunked_too(
        self, tmp_path, monkeypatch
    ):
        plain = _train(tmp_path / "plain", monkeypatch, chunk=None, rewind_log=False)
        chunked = _train(tmp_path / "chunked", monkeypatch, chunk=16, rewind_log=False)

        assert plain["trainer"] == "SFTTrainer"
        assert chunked["trainer"] == "ChunkedLossSFTTrainer"
        assert plain["rewind"] is None and chunked["rewind"] is None
        assert max(shape[1] for shape in chunked["head_inputs"]) == 16
        for want, got in zip(plain["grads"], chunked["grads"]):
            for a, b in zip(want, got):
                assert _close(b, a) or float(a.abs().max()) == 0.0
        for want, got in zip(plain["logs"], chunked["logs"]):
            for column in ("loss", "entropy", "mean_token_accuracy", "num_tokens"):
                assert got[column] == pytest.approx(want[column], rel=_RTOL, abs=1e-7), column

    def test_the_trainer_classes(self, arms):
        plain, chunked = arms

        assert plain["trainer"] == "RewindSFTTrainer"
        assert chunked["trainer"] == "RewindChunkedLossSFTTrainer"


class TestTheHookComesOffAfterTraining:
    """``train()`` takes the hook off when it ends, also when training raised,
    so a labelled forward later in the same process (an eval gate, ``infer``)
    gets the full logits and the trainer takes TRL's own ``compute_loss``."""

    @pytest.mark.parametrize("raises", [False, True])
    def test_train_removes_the_hook(self, tmp_path, monkeypatch, raises):
        from soup_cli.utils.stream_loss_chunk import HOOK_ATTR, causal_lm_of
        from tests.test_v07204 import _build_streamed_wrapper

        wrapper, _, _ = _build_streamed_wrapper(
            tmp_path, monkeypatch, task="sft", n_layers=2, stream_loss_chunk=16
        )
        hook = wrapper._stream_loss_chunk_hook
        causal_lm = causal_lm_of(wrapper.model)
        assert getattr(wrapper.trainer, HOOK_ATTR) is hook  # the premise
        assert "forward" in causal_lm.__dict__

        def fake_train(**_kwargs):
            if raises:
                raise RuntimeError("CUDA out of memory")

        wrapper.trainer.train = fake_train
        if raises:
            with pytest.raises(RuntimeError, match="out of memory"):
                wrapper.train()
        else:
            monkeypatch.setattr(wrapper.trainer, "save_model", lambda *_a, **_k: None)
            monkeypatch.setattr(wrapper, "_assert_streamed_adapter_saved", lambda *_a: None)
            wrapper.train()

        assert "forward" not in causal_lm.__dict__
        assert getattr(wrapper.trainer, HOOK_ATTR) is None
        assert wrapper._stream_loss_chunk_hook is None

    def test_a_run_without_the_option_is_untouched(self, tmp_path, monkeypatch):
        from tests.test_v07204 import _build_streamed_wrapper

        wrapper, _, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task="sft", n_layers=2)
        wrapper._remove_stream_loss_chunk_hook()  # nothing to remove

        assert wrapper._stream_loss_chunk_hook is None
        wrapper._close_stream_runtime()


# ─── The pre-flight ──────────────────────────────────────────────────────────

_GB = 1e9


def _peak(**overrides):
    from soup_cli.utils.layer_stream import estimate_stream_peak_vram

    shape = {
        "layer_bytes": 300_000_000,
        "buffers": 2,
        "extras_bytes": 0,
        "adapter_params": 10_000_000,
        "vocab_size": 128_256,
        "hidden_size": 4096,
        "intermediate_size": 14_336,
        "n_layers": 32,
        "seq_len": 2048,
        "batch_size": 1,
    }
    return estimate_stream_peak_vram(**{**shape, **overrides})


class TestThePreFlight:
    def test_the_chunk_bytes_are_two_hidden_sized_tensors(self):
        """The gradient the forward fills, and its product with the incoming
        gradient in the backward. bf16 is 2 bytes."""
        from soup_cli.utils.layer_stream import estimate_loss_chunk_bytes

        got = estimate_loss_chunk_bytes(hidden_size=4096, seq_len=2048, batch_size=3)

        assert got == 2 * 4096 * 2048 * 3 * 2
        assert estimate_loss_chunk_bytes(
            hidden_size=4096, seq_len=2048, batch_size=3, dtype="float32"
        ) == 2 * got

    @pytest.mark.parametrize("bad", [{"hidden_size": 0}, {"seq_len": 0}, {"batch_size": -1}])
    def test_the_chunk_bytes_refuse_a_shape_that_is_not_positive(self, bad):
        from soup_cli.utils.layer_stream import estimate_loss_chunk_bytes

        shape = {"hidden_size": 8, "seq_len": 8, "batch_size": 1, **bad}
        with pytest.raises(ValueError, match="must all be positive"):
            estimate_loss_chunk_bytes(**shape)

    def test_the_logits_term_is_charged_on_the_chunk(self):
        """The issue's arithmetic: 14 x vocab x C x batch is 0.23 GB at C = 128
        whatever the sequence length, against 3.68 GB at 2048 tokens."""
        from soup_cli.utils.layer_stream import estimate_logits_bytes, estimate_loss_chunk_bytes

        full = estimate_logits_bytes(vocab_size=128_256, seq_len=2048)
        chunk = estimate_logits_bytes(vocab_size=128_256, seq_len=128)
        kept = estimate_loss_chunk_bytes(hidden_size=4096, seq_len=2048, batch_size=1)

        assert full / _GB == pytest.approx(3.68, abs=0.005)
        assert chunk / _GB == pytest.approx(0.23, abs=0.005)
        assert _peak() - _peak(loss_chunk=128) == full - chunk - kept

    def test_the_saving_no_longer_grows_with_the_sequence(self):
        short = _peak(seq_len=1024) - _peak(seq_len=1024, loss_chunk=128)
        long = _peak(seq_len=2048) - _peak(seq_len=2048, loss_chunk=128)

        assert long > 2 * short > 0

    def test_a_chunk_at_or_above_the_sequence_saves_nothing(self):
        """One chunk is the whole sequence: the full logits are built, and the
        gradient tensor is paid on top. The prediction must not go down."""
        from soup_cli.utils.layer_stream import estimate_loss_chunk_bytes

        kept = estimate_loss_chunk_bytes(hidden_size=4096, seq_len=2048, batch_size=1)

        assert _peak(loss_chunk=2048) == _peak() + kept
        assert _peak(loss_chunk=8192) == _peak() + kept

    def test_unset_it_is_the_prediction_it_was(self):
        assert _peak(loss_chunk=None) == _peak()

    def test_a_calibrated_constant_still_only_raises_it(self):
        base = _peak(loss_chunk=128)

        assert _peak(loss_chunk=128, logits_bytes_per_element=20.0) > base
        assert _peak(loss_chunk=128, logits_bytes_per_element=1.0) == base

    @pytest.mark.parametrize("chunk", [0, -8, True, 12.5])
    def test_a_chunk_that_is_not_a_positive_integer_is_refused(self, chunk):
        with pytest.raises(ValueError, match="loss_chunk must be a positive integer"):
            _peak(loss_chunk=chunk)


def test_the_pre_flight_of_a_real_setup_charges_the_chunk(tmp_path, monkeypatch):
    """The charge has to reach the pre-flight, not only exist. The tiny model
    (vocabulary 64, hidden 64) is budgeted at ``data.max_length`` = 64 tokens:
    the logits term is 14 x 64 x 64 bytes unchunked and a quarter of that at a
    chunk of 16, where two 64 x 64 bf16 tensors are added."""
    from soup_cli.utils import layer_stream
    from tests.test_v07204 import _build_streamed_wrapper

    estimate = layer_stream.estimate_stream_peak_vram
    seen = {}

    def recording(**kwargs):
        seen["chunk"], seen["bytes"] = kwargs["loss_chunk"], estimate(**kwargs)
        return seen["bytes"]

    monkeypatch.setattr(layer_stream, "estimate_stream_peak_vram", recording)

    def predicted(chunk):
        root = tmp_path / str(chunk)
        root.mkdir()
        training = {} if chunk is None else {"stream_loss_chunk": chunk}
        wrapper, _, _ = _build_streamed_wrapper(root, monkeypatch, task="sft", **training)
        wrapper._close_stream_runtime()
        assert seen.pop("chunk") == chunk
        return seen.pop("bytes")

    saved = predicted(None) - predicted(16)

    assert saved == 14 * 64 * (64 - 16) - 2 * 64 * 64 * 2


def test_the_pre_flight_panel_says_the_logits_are_chunked(tmp_path, monkeypatch):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from tests.test_v07204 import _build_streamed_wrapper

    budget_lines = StreamingSetupMixin._stream_budget_lines
    seen = []

    def recording(self, *args, **kwargs):
        lines, plan = budget_lines(self, *args, **kwargs)
        seen.append(lines[0])
        return lines, plan

    monkeypatch.setattr(StreamingSetupMixin, "_stream_budget_lines", recording)
    for chunk in (None, 16):
        root = tmp_path / str(chunk)
        root.mkdir()
        training = {} if chunk is None else {"stream_loss_chunk": chunk}
        wrapper, _, _ = _build_streamed_wrapper(root, monkeypatch, task="sft", **training)
        wrapper._close_stream_runtime()

    plain, chunked = seen
    assert plain.endswith("GB)") and "in chunks of" not in plain
    assert chunked.endswith("GB in chunks of 16 tokens)")


def _panel_line(loss_chunk, seq=2048):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    class _Lora:
        r = 8
        target_modules = ["q_proj", "v_proj"]

    class _Train:
        batch_size = 1
        stream_buffers = 2
        stream_vram_probe = False
        stream_vram_override = int(64 * _GB)
        gradient_accumulation_steps = 1
        lora = _Lora()

    _Train.stream_loss_chunk = loss_chunk

    class _Model:
        vocab_size = 128_256
        hidden_size = 4096
        intermediate_size = 14_336

    class _Index:
        n_layers = 32
        total_params = 0

    class _Data:
        max_length = seq

    class _Cfg:
        data = _Data()

    mixin = StreamingSetupMixin()
    mixin.device = "cpu"
    lines, _ = mixin._stream_budget_lines(
        _Cfg(), _Train(), model_config=_Model(), layer_bytes=300_000_000,
        embed_bytes=0, index=_Index(), on_cuda=False,
    )
    return lines[0]


def test_the_panel_shows_the_logits_of_one_chunk():
    assert _panel_line(None).endswith("(logits 3.68 GB)")
    assert _panel_line(128).endswith("(logits 0.23 GB in chunks of 128 tokens)")
    assert _panel_line(8192).endswith("(logits 3.68 GB in chunks of 8192 tokens)")


def test_the_chunk_bytes_are_charged_for_every_row_of_the_batch():
    from soup_cli.utils.layer_stream import estimate_logits_bytes, estimate_loss_chunk_bytes

    full = estimate_logits_bytes(vocab_size=128_256, seq_len=2048, batch_size=4)
    chunk = estimate_logits_bytes(vocab_size=128_256, seq_len=128, batch_size=4)
    kept = estimate_loss_chunk_bytes(hidden_size=4096, seq_len=2048, batch_size=4)

    assert _peak(batch_size=4) - _peak(batch_size=4, loss_chunk=128) == full - chunk - kept


def test_the_hook_is_on_the_model_before_the_vram_probe_measures_a_step(tmp_path, monkeypatch):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from tests.test_v07204 import _build_streamed_wrapper

    seen = []

    def probe(self, model, plan):
        seen.append(self._stream_loss_chunk_hook is not None)

    budget = StreamingSetupMixin._stream_budget_lines
    # Off CUDA the budget returns no probe plan; hand it one so the probe step is reached.
    monkeypatch.setattr(
        StreamingSetupMixin,
        "_stream_budget_lines",
        lambda self, *args, **kwargs: (budget(self, *args, **kwargs)[0], object()),
    )
    monkeypatch.setattr(StreamingSetupMixin, "_run_stream_vram_probe", probe)

    wrapper, _, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task="sft", stream_loss_chunk=16)
    wrapper._close_stream_runtime()

    assert seen == [True]


def test_a_hook_that_cannot_be_built_releases_the_streamed_runtime(tmp_path, monkeypatch):
    """The hook is built after the runtime and before training's cleanup
    context starts, so a refusal there closes the runtime itself."""
    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from soup_cli.utils import stream_loss_chunk
    from tests.test_v07204 import _build_streamed_wrapper

    closes = []
    real_close = StreamingSetupMixin._close_stream_runtime

    def counting_close(self):
        closes.append(getattr(self, "_stream_runtime", None) is not None)
        return real_close(self)

    def refuse(*_args, **_kwargs):
        raise ValueError("no output head to chunk")

    monkeypatch.setattr(StreamingSetupMixin, "_close_stream_runtime", counting_close)
    monkeypatch.setattr(stream_loss_chunk, "ChunkedHeadLoss", refuse)

    with pytest.raises(ValueError, match="no output head to chunk"):
        _build_streamed_wrapper(tmp_path, monkeypatch, task="sft", stream_loss_chunk=16)

    assert closes == [True]
