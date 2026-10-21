"""#1990 — ``training.eval_steps`` on MLX SFT, in completed optimizer updates.

mlx-lm evaluates *before* the step at iteration ``it``, so under accumulation a
cadence of ``steps_per_eval`` misses the update that completes at ``it``, its
"final" pass precedes the last update, and it reads at most 25 whole batches,
dropping a tail. An explicit ``eval_steps`` is therefore run by Soup:

- after each completed update ``N, 2N, ...`` and after the last update, once;
- over every validation row, the tail included, token-weighted;
- without moving parameters, optimizer state, training mode or the seeded
  training order.

The unset field keeps today's mlx-lm cadence. Real MLX on a tiny random Llama;
no downloads. The schedule, the refusals and the trigger in ``train()`` (through
the shared fake-MLX harness) also run where MLX is not installed.
"""

from __future__ import annotations

import sys
from io import StringIO

import numpy as np
import pytest
from rich.console import Console

from tests.conftest import strip_ansi


@pytest.fixture
def native():
    """Real MLX and mlx-lm; only the classes that use this fixture skip without them."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")


class _IntTokenizer:
    """Text: one id per character. Chat: role and word markers, all ids < 200."""

    eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [3 + ord(c) % 150 for c in text]

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False,
                            return_dict=False, **_):
        out = []
        for m in messages:
            out += [160 if m["role"] == "user" else 161]
            out += [3 + ord(c) % 150 for c in m["content"]] + [162]
        if add_generation_prompt:
            out.append(161)
        return out


def _tiny(hidden=32, quantize=False):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import llama

    mx.random.seed(0)
    model = llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=hidden, num_hidden_layers=2,
        intermediate_size=2 * hidden, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=200, tie_word_embeddings=True,
    ))
    if quantize:
        nn.quantize(model, group_size=32, bits=4)
    return model


def _text_rows(n, prefix="row"):
    return [{"text": f"{prefix} {i} " + "ab" * (i % 4 + 1)} for i in range(n)]


def _wrapper(tmp_path, *, train=8, val=4, chat=False, model=None, **training):
    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    training.setdefault("epochs", 1)
    training.setdefault("batch_size", 2)
    training.setdefault("lr", 1e-3)
    training.setdefault("gradient_accumulation_steps", 1)
    cfg = SoupConfig(
        base="mlx-community/tiny", task="sft", backend="mlx",
        data=DataConfig(train="./d.jsonl", format="chatml" if chat else "plaintext",
                        max_length=256),
        training=TrainingConfig(logging_steps=1, **training),
        output=str(tmp_path),
    )
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.model = model if model is not None else _tiny()
    wrapper.tokenizer = _IntTokenizer()
    if chat:
        rows = [{"messages": [{"role": "user", "content": f"q{i}"},
                              {"role": "assistant", "content": "ans" * (i % 3 + 1)}]}
                for i in range(train + val)]
        wrapper._dataset = {"train": rows[:train], "val": rows[train:]}
    else:
        wrapper._dataset = {"train": _text_rows(train), "val": _text_rows(val, "val")}
    return wrapper


class TestTheSchedule:
    @pytest.mark.parametrize("total, every, expected", [
        (4, 1, [1, 2, 3, 4]),
        (4, 2, [2, 4]),
        (4, 3, [3, 4]),
        (4, 10, [4]),
        (1, 1, [1]),
    ])
    def test_updates(self, total, every, expected):
        from soup_cli.trainer.mlx_sft import scheduled_eval_updates

        assert scheduled_eval_updates(total, every) == expected


@pytest.mark.usefixtures("native")
class TestEvaluationsHappenAfterCompletedUpdates:
    @pytest.mark.parametrize("epochs, accum, every, expected", [
        (1, 1, 1, [1, 2, 3, 4]),   # every update
        (2, 1, 3, [3, 6, 8]),      # N does not divide the 8-update budget
        (2, 2, 2, [2, 4]),         # accumulation: 8 iterations, 4 updates
        (2, 2, 3, [3, 4]),         # accumulation, non-dividing
        (1, 1, 10, [4]),           # N past the budget: once, at the end
        (1, 4, 1, [1]),            # a run of one update
    ])
    def test_exact_update_indices(self, tmp_path, epochs, accum, every, expected):
        wrapper = _wrapper(tmp_path, epochs=epochs, gradient_accumulation_steps=accum,
                           eval_steps=every)
        result = wrapper.train()
        log = wrapper._eval_log
        assert [e["update"] for e in log] == expected
        # The optimizer had taken exactly that many steps when each ran:
        # completed updates, not micro-batches and not callback calls.
        assert [e["optimizer_step"] for e in log] == expected
        assert result["eval_updates"] == expected
        assert result["val_loss"] == pytest.approx(log[-1]["val_loss"])


@pytest.mark.usefixtures("native")
class TestTheUnsetFieldKeepsMlxLmsCadence:
    def _seen(self, tmp_path, monkeypatch, **training):
        seen = {}
        monkeypatch.setattr(sys.modules["mlx_lm.tuner.trainer"], "train",
                            lambda **kw: seen.update(kw))
        _wrapper(tmp_path, **training).train()
        return seen

    def test_unset_passes_the_split_to_mlx_lm_at_a_quarter_of_the_run(self, tmp_path, monkeypatch):
        import mlx_lm.tuner.trainer  # noqa: F401 — registered before it is patched

        seen = self._seen(tmp_path, monkeypatch, epochs=2)
        assert seen["val_dataset"] is not None
        assert seen["args"].steps_per_eval == max(1, seen["args"].iters // 4)

    def test_explicit_takes_the_split_away_from_mlx_lm(self, tmp_path, monkeypatch):
        import mlx_lm.tuner.trainer  # noqa: F401

        seen = self._seen(tmp_path, monkeypatch, epochs=2, eval_steps=2)
        assert seen["val_dataset"] is None
        assert seen["args"].steps_per_eval > seen["args"].iters


class TestRefusals:
    @pytest.mark.parametrize("value, message", [
        (0, "greater than or equal to 1"),
        (-1, "greater than or equal to 1"),
        (True, "must be an int, not a bool"),
    ])
    def test_invalid_values(self, value, message):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError, match=message):
            TrainingConfig(eval_steps=value)

    def test_no_validation_rows_is_refused_before_the_model_loads(self, monkeypatch):
        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
        from soup_cli.trainer import mlx_sft

        def _must_not_load(*_a, **_k):
            raise AssertionError("loaded")

        monkeypatch.setattr("soup_cli.utils.mlx.load_mlx_model", _must_not_load)
        cfg = SoupConfig(
            base="m", task="sft", backend="mlx",
            data=DataConfig(train="./d.jsonl", format="plaintext"),
            training=TrainingConfig(eval_steps=2, batch_size=1), output="./o",
        )
        wrapper = mlx_sft.MLXSFTTrainerWrapper(cfg)
        monkeypatch.setattr(wrapper, "_require_mlx", lambda: None)
        with pytest.raises(ValueError, match="validation split is\\s+empty"):
            wrapper.setup({"train": _text_rows(4), "val": []})


def _one_ahead(stream):
    pending = next(stream)
    for following in stream:
        yield pending
        pending = following


class TestTheTriggerWithoutMlx:
    """``train()``'s real body through the shared fake-MLX harness, with a
    stand-in for mlx-lm's loop of the same shape: ``zip(range(1, iters + 1),
    iterate_batches(..., loop=True))``, the optimizer stepping only when
    ``it % accum == 0``. The evaluator is a recorder of the optimizer steps
    taken when it ran, so the schedule is checked on every CI cell."""

    def _run(self, tmp_path, monkeypatch, prefetch=False, **training):
        from soup_cli.trainer import mlx_sft
        from tests.test_issue634_mlx_resume import _FakeMlxModel, _install_fake_mlx

        _install_fake_mlx(monkeypatch)
        loop = {"steps": 0}

        class _Step:
            def item(self):
                return loop["steps"]

        class _AdamW:
            def __init__(self, **_):
                self.state = {"step": _Step()}

        monkeypatch.setattr(sys.modules["mlx.optimizers"], "AdamW", _AdamW)

        def iterate_batches(*, dataset, batch_size, loop=False, **_):
            while True:
                for start in range(0, len(dataset) - batch_size + 1, batch_size):
                    yield dataset[start:start + batch_size]
                if not loop:
                    return

        def train(*, optimizer, train_dataset, val_dataset, args, iterate_batches=iterate_batches,
                  **_):
            loop.update(val_dataset=val_dataset, args=args)
            stream = iterate_batches(dataset=train_dataset, batch_size=args.batch_size,
                                     max_seq_length=args.max_seq_length, loop=True)
            if prefetch:  # a loop that reads one batch ahead of its step
                stream = _one_ahead(stream)
            for it, _batch in zip(range(1, args.iters + 1), stream):
                if it % args.grad_accumulation_steps == 0:
                    loop["steps"] += 1

        trainer = sys.modules["mlx_lm.tuner.trainer"]
        monkeypatch.setattr(trainer, "train", train)
        monkeypatch.setattr(trainer, "iterate_batches", iterate_batches, raising=False)
        monkeypatch.setattr(trainer, "default_loss", object(), raising=False)
        seen = []

        def recorder(_model, dataset, **_):
            seen.append((loop["steps"], len(dataset)))
            return 1.0

        monkeypatch.setattr(mlx_sft, "evaluate_all_rows", recorder)
        loop["printed"] = StringIO()
        monkeypatch.setattr(mlx_sft, "console", Console(file=loop["printed"], width=400))
        monkeypatch.chdir(tmp_path)
        loop["wrapper"] = wrapper = _wrapper(tmp_path, model=_FakeMlxModel(), **training)
        result = wrapper.train()
        return result, seen, loop

    @pytest.mark.parametrize("epochs, accum, every, expected", [
        (1, 1, 1, [1, 2, 3, 4]),
        (2, 1, 3, [3, 6, 8]),
        (2, 2, 2, [2, 4]),
        (2, 2, 3, [3, 4]),
        (1, 1, 10, [4]),
        (1, 4, 1, [1]),
    ])
    def test_evaluations_follow_completed_updates(self, tmp_path, monkeypatch,
                                                  epochs, accum, every, expected):
        result, seen, loop = self._run(tmp_path, monkeypatch, epochs=epochs,
                                    gradient_accumulation_steps=accum, eval_steps=every)
        # The optimizer had taken exactly `update` steps when each evaluation
        # ran, over the whole split; the last one follows train()'s return.
        assert seen == [(update, 4) for update in expected]
        assert result["eval_updates"] == expected
        assert [e["optimizer_step"] for e in loop["wrapper"]._eval_log] == expected
        # In step, the drift warning stays silent.
        assert "fetches batches ahead" not in loop["printed"].getvalue()

    def test_a_loop_that_prefetches_is_reported_once(self, tmp_path, monkeypatch):
        """The trigger relies on mlx-lm fetching batch `it` after step
        `it - 1`; a release that reads ahead must not move it in silence."""
        _, seen, loop = self._run(tmp_path, monkeypatch, prefetch=True, eval_steps=1)
        assert [step for step, _ in seen][:3] == [0, 1, 2]
        text = " ".join(strip_ansi(loop["printed"].getvalue()).split())
        assert text.count("fetches batches ahead of the step") == 1
        assert "evaluation for update 1 ran after 0 optimizer step(s)" in text

    def test_the_split_is_taken_away_from_mlx_lm(self, tmp_path, monkeypatch):
        _, _, loop = self._run(tmp_path, monkeypatch, epochs=2, eval_steps=2)
        assert loop["val_dataset"] is None
        assert loop["args"].steps_per_eval > loop["args"].iters

    def test_unset_keeps_mlx_lms_cadence_and_runs_no_evaluation_of_its_own(
        self, tmp_path, monkeypatch,
    ):
        result, seen, loop = self._run(tmp_path, monkeypatch, epochs=2)
        assert seen == []
        assert loop["val_dataset"] is not None
        assert loop["args"].steps_per_eval == max(1, loop["args"].iters // 4)
        assert "eval_updates" not in result


# --------------------------------------------------------------------------
# The evaluator: every row, token-weighted, no side effects
# --------------------------------------------------------------------------


def _np_ce(model, ids, mask=None):
    """(sum of target NLL, number of targets) for one row, independently.

    Text rows follow mlx-lm 0.32's ``default_loss``, the loss MLX SFT trains
    on: its mask is ``steps <= length``, so it also scores the first padding
    token (id 0) after the row. The evaluator reuses the training loss, so its
    number is comparable with the training loss; the oracle states that rule
    rather than inheriting it. Masked chat rows score exactly their mask.
    """
    import mlx.core as mx

    ids = list(ids) + ([0] if mask is None else [])
    logits = np.array(model(mx.array([ids[:-1]]))[0].astype(mx.float32), dtype=np.float64)
    logits -= logits.max(-1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    targets = np.array(ids[1:])
    nll = -logp[np.arange(len(targets)), targets]
    keep = np.ones(len(targets), bool) if mask is None else np.array(mask[1:], bool)
    return float(nll[keep].sum()), int(keep.sum())


def _datasets(wrapper, chat):
    from mlx_lm.tuner.datasets import CacheDataset, create_dataset
    from mlx_lm.tuner.trainer import TrainingArgs, default_loss, iterate_batches

    rows = wrapper._dataset["val"]
    if chat:
        from soup_cli.trainer.mlx_masking import (
            MaskedChatDataset,
            masked_iterate_batches,
            masked_loss,
        )

        ds = CacheDataset(MaskedChatDataset(rows, wrapper.tokenizer, chat_key="messages"))
        return ds, masked_loss, masked_iterate_batches
    return (CacheDataset(create_dataset(rows, wrapper.tokenizer, TrainingArgs())),
            default_loss, iterate_batches)


@pytest.mark.usefixtures("native")
class TestEveryRowCounts:
    @pytest.mark.parametrize("chat", [False, True])
    def test_loss_equals_a_hand_oracle_over_every_target_tail_included(self, tmp_path, chat):
        from soup_cli.trainer.mlx_sft import evaluate_all_rows

        wrapper = _wrapper(tmp_path, val=5, chat=chat)
        ds, loss, iterate = _datasets(wrapper, chat)
        value = evaluate_all_rows(wrapper.model, ds, batch_size=2, max_seq_length=256,
                                  loss_fn=loss, iterate_fn=iterate)
        total, count = 0.0, 0
        for i in range(len(ds)):
            item = ds[i]
            ids, mask = (item if chat else (item[0] if isinstance(item, tuple) else item, None))
            row_sum, row_n = _np_ce(wrapper.model, list(ids), mask)
            total, count = total + row_sum, count + row_n
        assert value == pytest.approx(total / count, rel=1e-4)

        # The fifth row is a tail smaller than one batch: leaving it out changes
        # the number, so dropping it is detectable.
        head, head_n = 0.0, 0
        for i in range(4):
            item = ds[i]
            ids, mask = (item if chat else (item[0] if isinstance(item, tuple) else item, None))
            s, n = _np_ce(wrapper.model, list(ids), mask)
            head, head_n = head + s, head_n + n
        assert abs(value - head / head_n) > 1e-4

    def test_evaluation_has_no_side_effects(self, tmp_path):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        from soup_cli.trainer.mlx_sft import evaluate_all_rows

        wrapper = _wrapper(tmp_path, val=5)
        ds, loss, iterate = _datasets(wrapper, False)
        wrapper.model.train()
        before = {k: np.array(v) for k, v in tree_flatten(wrapper.model.parameters())}
        np.random.seed(123)
        rng_before = np.random.get_state()[1].copy()
        mx.random.seed(9)
        evaluate_all_rows(wrapper.model, ds, batch_size=2, max_seq_length=256,
                          loss_fn=loss, iterate_fn=iterate)
        after_draw = mx.random.uniform(shape=(4,)).tolist()
        mx.random.seed(9)
        assert after_draw == mx.random.uniform(shape=(4,)).tolist()
        assert (np.random.get_state()[1] == rng_before).all()
        assert wrapper.model.training
        after = {k: np.array(v) for k, v in tree_flatten(wrapper.model.parameters())}
        for name in before:
            np.testing.assert_array_equal(before[name], after[name], err_msg=name)

    def test_a_batching_helper_that_draws_randomness_still_leaves_numpy_untouched(
        self, tmp_path
    ):
        """mlx-lm's batch helpers shuffle through numpy's global RNG; whatever a
        helper draws during an evaluation is given back."""
        from soup_cli.trainer.mlx_sft import evaluate_all_rows

        wrapper = _wrapper(tmp_path, val=5)
        ds, loss, iterate = _datasets(wrapper, False)

        def drawing(**kwargs):
            np.random.random(17)
            yield from iterate(**kwargs)

        np.random.seed(5)
        state = np.random.get_state()[1].copy()
        evaluate_all_rows(wrapper.model, ds, batch_size=2, max_seq_length=256,
                          loss_fn=loss, iterate_fn=drawing)
        assert (np.random.get_state()[1] == state).all()

    def test_a_split_smaller_than_one_batch_is_evaluated_not_skipped(self, tmp_path, capsys):
        """#1857 skips such a split when the cadence is mlx-lm's; with an
        explicit eval_steps every row is evaluated, so the row stays."""
        wrapper = _wrapper(tmp_path, val=1, eval_steps=2)
        result = wrapper.train()
        assert result["eval_updates"] == [2, 4]
        # Rich colours the "1" on a colour terminal; compare the plain text.
        assert "validation split has 1 row" not in strip_ansi(capsys.readouterr().out)

    def test_scheduled_evaluations_do_not_change_what_trains(self, tmp_path):
        """Evaluating after every update and only at the end train the same
        adapter: the evaluations take no step, no state and no batch."""
        import mlx.core as mx
        from mlx.utils import tree_flatten

        def adapter(path, every):
            wrapper = _wrapper(path, epochs=2, eval_steps=every)
            # Seeded the way #1855 seeds MLX SFT (numpy's global RNG draws the
            # batch order, mx.random the LoRA init), so the two runs are
            # comparable on main with or without it.
            np.random.seed(7)
            mx.random.seed(7)
            wrapper.train()
            return {k: np.array(v) for k, v in
                    tree_flatten(wrapper.model.trainable_parameters())}

        often, once = adapter(tmp_path / "a", 1), adapter(tmp_path / "b", 1000)
        assert often
        for name in often:
            np.testing.assert_array_equal(often[name], once[name], err_msg=name)


@pytest.mark.usefixtures("native")
class TestNativeVariants:
    @pytest.mark.parametrize("quantize", [False, True], ids=["dense", "4bit"])
    def test_runs_and_records_its_evaluations(self, tmp_path, quantize):
        import json

        model = _tiny(hidden=64, quantize=quantize)
        wrapper = _wrapper(tmp_path, epochs=2, gradient_accumulation_steps=2, eval_steps=1,
                           model=model)
        result = wrapper.train()
        assert result["eval_updates"] == [1, 2, 3, 4]
        assert all(np.isfinite(e["val_loss"]) for e in wrapper._eval_log)
        meta = json.loads((tmp_path / "adapter_config.json").read_text())
        assert (meta["eval_steps"], meta["eval_unit"]) == (1, "optimizer updates")
