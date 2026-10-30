"""Native KTO on MLX (#2000).

The objective is pinned to TRL 0.29.1's ``experimental/kto/kto_trainer.py``,
which Soup's transformers backend runs for ``task: kto`` (``trainer/kto.py``
forwards ``kto_beta`` and ``max_length`` only, so ``loss_type``,
``desirable_weight`` and ``undesirable_weight`` stay at ``KTOConfig``'s
defaults). Layers:

- the TRL facts the objective rests on, read from TRL itself (no MLX);
- tokenization and the KL (mismatched) rows against TRL's own ``_tokenize`` /
  ``_process_tokens`` / ``_get_kl_dataset`` pipeline on a real
  ``PreTrainedTokenizerFast`` (needs ``trl`` / ``datasets``, no MLX);
- an independent float64 NumPy oracle, tied to TRL's own ``kto_loss`` with
  torch (no MLX) and shown to be moved by every wrong objective the issue lists;
  the MLX loss, its terms and its gradients against that oracle (MLX);
- training, the fixed reference, saving and a fresh-process reload on a tiny
  random Llama (needs MLX and mlx-lm). No downloads.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

from tests.test_mlx_dpo import _strip_ansi
from tests.test_mlx_orpo import (
    BOS,
    EOS,
    CharTokenizer,
    MergingTokenizer,
    _decode,
    _flat,
    _tiny_llama_args,
)

BETA = 0.1  # KTOConfig.beta's default and Soup's kto_beta default


def _examples(rows, tokenizer=None, max_length=256, group_size=2):
    from soup_cli.trainer.mlx_kto import build_kto_examples

    return build_kto_examples(
        rows, tokenizer or CharTokenizer(), max_length=max_length, group_size=group_size,
    )


def _one(row, **kw):
    return _examples([row, {"prompt": "z", "completion": "w", "label": False}], **kw)[0]


# --------------------------------------------------------------------------
# TRL facts this objective rests on (no MLX)
# --------------------------------------------------------------------------


def _kto_trainer_source() -> str:
    pytest.importorskip("trl")
    pytest.importorskip("torch")
    from trl.experimental.kto import kto_trainer

    return inspect.getsource(kto_trainer)


class TestTrlFacts:
    def test_the_effective_settings_are_ktoconfig_defaults(self):
        pytest.importorskip("trl")
        from trl.experimental.kto import KTOConfig

        from soup_cli.trainer.mlx_kto import (
            DESIRABLE_WEIGHT,
            KTO_LOSS_TYPE,
            UNDESIRABLE_WEIGHT,
        )

        fields = KTOConfig.__dataclass_fields__
        assert fields["beta"].default == BETA
        assert fields["loss_type"].default == KTO_LOSS_TYPE == "kto"
        assert fields["desirable_weight"].default == DESIRABLE_WEIGHT == 1.0
        assert fields["undesirable_weight"].default == UNDESIRABLE_WEIGHT == 1.0
        assert fields["disable_dropout"].default is True
        # 0.29.1 has no prompt cap and no truncation mode: only max_length.
        assert not {"max_prompt_length", "max_completion_length", "truncation_mode"} & set(fields)

    def test_the_transformers_kto_path_passes_beta_and_max_length_only(self):
        """Soup builds KTOConfig with kto_beta and max_length; the prompt-length
        helper forwards nothing on 0.29.1, and no Soup cap touches the rows."""
        pytest.importorskip("trl")
        from trl.experimental.kto import KTOConfig

        import soup_cli.trainer.kto as kto
        from soup_cli.trainer._trl_compat import prompt_length_kwargs

        source = Path(kto.__file__).read_text(encoding="utf-8")
        calls = [
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "KTOConfig"
        ]
        assert len(calls) == 1
        names = {kw.arg for kw in calls[0].keywords}
        assert {"beta", "max_length", "per_device_train_batch_size"} <= names
        assert not names & {"loss_type", "desirable_weight", "undesirable_weight",
                            "precompute_ref_log_probs", "disable_dropout"}
        assert prompt_length_kwargs(KTOConfig, 64) == {}
        assert "enforce_preference_sequence_limit" not in source

    def test_the_training_sampler_is_sequential(self):
        """``_get_train_sampler`` (kto_trainer.py:1315-1320): TRL's KTO never
        shuffles, so a training batch is exactly one KL group."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("trl")
        from trl.experimental.kto.kto_trainer import KTOTrainer

        sampler = KTOTrainer._get_train_sampler(types.SimpleNamespace(), list(range(5)))
        assert isinstance(sampler, torch.utils.data.SequentialSampler)

    def test_summed_logps_batch_floor_and_kl_grouping_are_in_the_source(self):
        source = _kto_trainer_source()
        assert source.count("average_log_prob=False") >= 4
        assert "if args.per_device_train_batch_size <= 1:" in source
        assert source.count("batch_size=args.per_device_train_batch_size,") == 2
        assert "kl = (policy_KL_logps - reference_KL_logps).mean().detach()" in source
        assert "kl = self.accelerator.gather_for_metrics(kl).mean().clamp(min=0)" in source
        assert "loss = losses.nanmean()" in source


# --------------------------------------------------------------------------
# Tokenization (TRL 0.29.1 _tokenize / _process_tokens)
# --------------------------------------------------------------------------


_U = {"role": "user", "content": "hi"}


class TestTokenization:
    def test_a_generation_prompt_the_full_render_lacks_is_narrowed_away(self):
        """TRL keeps only the prefix the prompt and ``prompt + completion``
        renderings share (review of #2111): a template whose generation prompt
        adds text the full rendering does not contain must not shift the split."""
        from soup_cli.trainer.mlx_kto import render_kto_text

        class _Template:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                                    continue_final_message=False):
                text = "".join(f"{m['role'][0].upper()}:{m['content']}\n" for m in messages)
                return text + ("A:<think>\n" if add_generation_prompt else "")

        row = {"prompt": [{"role": "user", "content": "q"}],
               "completion": [{"role": "assistant", "content": "ans"}]}
        prompt, completion = render_kto_text(row, _Template())
        assert prompt == "U:q\nA:"
        assert completion == "ans\n"

    def test_text_row_gets_bos_on_the_prompt_and_eos_on_the_completion(self):
        ex = _one({"prompt": "Q:", "completion": " yes", "label": True})
        assert _decode(ex.prompt_ids) == "^Q:"
        assert _decode(ex.completion_ids) == " yes$"
        assert ex.label is True

    def test_bos_and_eos_are_not_doubled(self):
        class WithSpecials(CharTokenizer):
            def encode(self, text, add_special_tokens=True):
                ids = super().encode(text, False)
                return ([BOS] if text.startswith("Q") else []) + ids + (
                    [EOS] if text.endswith("!") else [])

        ex = _one({"prompt": "Q", "completion": "a!", "label": True}, tokenizer=WithSpecials())
        whole = ex.prompt_ids + ex.completion_ids
        assert whole.count(BOS) == 1 and whole.count(EOS) == 1

    def test_conversational_row_renders_like_trl(self):
        ex = _one({"prompt": [_U], "completion": [{"role": "assistant", "content": "a"}],
                   "label": False})
        assert _decode(ex.prompt_ids) == "^<user>hi|<assistant>"
        assert _decode(ex.completion_ids) == "a|$"

    def test_prompt_ending_in_assistant_continues_the_message(self):
        ex = _one({"prompt": [_U, {"role": "assistant", "content": "Sure,"}],
                   "completion": [{"role": "assistant", "content": "a"}], "label": True})
        assert _decode(ex.prompt_ids) == "^<user>hi|<assistant>Sure,"

    def test_rows_go_through_the_loader_converter(self):
        """A string prompt beside a message completion is the user turn."""
        ex = _one({"prompt": "hi", "completion": [{"role": "assistant", "content": "a"}],
                   "label": True})
        assert _decode(ex.prompt_ids) == "^<user>hi|<assistant>"

    def test_a_merged_last_prompt_token_moves_the_completion_start(self):
        ex = _one({"prompt": "zx", "completion": "y1", "label": True},
                  tokenizer=MergingTokenizer())
        assert ex.completion_ids[0] == MergingTokenizer.MERGED
        assert _decode(ex.prompt_ids) == "^z"

    def test_an_overlong_row_cuts_the_completion_end_and_never_the_prompt(self):
        """TRL's rule: room is max_length minus the BOS / EOS it will add."""
        ex = _one({"prompt": "P" * 6, "completion": "abcdefgh", "label": True}, max_length=10)
        assert _decode(ex.prompt_ids) == "^" + "P" * 6
        assert _decode(ex.completion_ids) == "ab$"
        assert len(ex.prompt_ids) + len(ex.completion_ids) == 10


class TestKLRows:
    def _rows(self, n):
        return [{"prompt": f"p{i}:", "completion": f" c{i}" + "x" * i, "label": i % 2 == 0}
                for i in range(n)]

    def test_partners_rotate_by_one_within_each_group(self):
        from soup_cli.trainer.mlx_kto import kl_partners

        assert kl_partners(7, 3) == [2, 0, 1, 5, 3, 4, 6]
        assert kl_partners(4, 2) == [1, 0, 3, 2]
        assert kl_partners(5, 2) == [1, 0, 3, 2, 4]  # a last group of one is its own

    def test_each_kl_row_is_its_prompt_with_the_partner_completion(self):
        examples = _examples(self._rows(5), group_size=2)
        for i, j in enumerate([1, 0, 3, 2, 4]):
            assert examples[i].kl_completion_ids == examples[j].completion_ids
            assert _decode(examples[i].prompt_ids) == f"^p{i}:"

    def test_a_kl_completion_is_recut_to_its_own_prompt_room(self):
        rows = [{"prompt": "P" * 7, "completion": "ab", "label": True},
                {"prompt": "Q", "completion": "abcdefgh", "label": False}]
        a, b = _examples(rows, max_length=10, group_size=2)
        assert _decode(a.kl_completion_ids) == "a$"  # room 1 after "^PPPPPPP"
        assert _decode(b.kl_completion_ids) == "ab$"

    def test_the_group_is_the_resolved_batch_size(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.mlx_kto import MLXKTOTrainerWrapper

        wrapper = MLXKTOTrainerWrapper(load_config_from_string(_yaml(batch_size=3)))
        wrapper.tokenizer = CharTokenizer()
        examples, skipped = wrapper._examples(self._rows(7))
        assert skipped == 0
        got = [next(j for j, e in enumerate(examples) if e.completion_ids == x.kl_completion_ids)
               for x in examples]
        assert got == [2, 0, 1, 5, 3, 4, 6]


def _hf_tokenizer():
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<s>", "</s>", "<pad>", "<unk>", "<", ">", "user", "assistant", "|", "Q", ":",
             "what", "is", "two", "plus", "one", "four", "five", "it", "yes", "no", "three",
             "the", "sum", "of", "six", "seven", "answer"]
    tok = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(words)},
                                     unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation()]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
        pad_token="<pad>", unk_token="<unk>",
    )
    fast.chat_template = (
        "{% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} | {% endfor %}"
        "{% if add_generation_prompt %}<assistant> {% endif %}"
    )
    return fast


def _trl_pipeline(rows, tokenizer, max_length, batch_size):
    """TRL 0.29.1's own KTO dataset preparation, the order KTOTrainer.__init__ runs it."""
    import datasets
    from trl.data_utils import maybe_apply_chat_template
    from trl.experimental.kto.kto_trainer import _get_kl_dataset, _process_tokens, _tokenize

    ds = datasets.Dataset.from_list(rows)
    ds = ds.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    ds = ds.map(_tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
    kwargs = {"prefix": "", "tokenizer": tokenizer, "max_length": max_length}
    ds = ds.map(_process_tokens, fn_kwargs=kwargs)
    kl = ds.map(_get_kl_dataset, batched=True, batch_size=batch_size)
    kl = kl.map(_process_tokens, fn_kwargs={**kwargs, "prefix": "KL_"},
                remove_columns=[c for c in kl.column_names if c in ds.column_names])
    return datasets.concatenate_datasets([ds, kl], axis=1)


_TEXT_ROWS = [
    {"prompt": "Q: what is two plus two", "completion": " four", "label": True},
    {"prompt": "Q: what is one plus one", "completion": " two", "label": False},
    {"prompt": "Q: is it five", "completion": " no it is four", "label": True},
    {"prompt": "Q: what is three", "completion": " three", "label": False},
    {"prompt": "Q: the sum of six", "completion": " six is the sum", "label": True},
    {"prompt": "Q: is seven one", "completion": " no", "label": False},
    {"prompt": "Q: what is the answer", "completion": " yes the answer is seven", "label": True},
]


def _chat(rows):
    return [{"prompt": [{"role": "user", "content": r["prompt"][3:]}],
             "completion": [{"role": "assistant", "content": r["completion"].strip()}],
             "label": r["label"]} for r in rows]


class TestTokensAndKLRowsMatchTrl:
    """Distinct prompts and answers, an uneven final group (7 rows, groups of 3)."""

    @pytest.mark.parametrize("rows, max_length, batch_size", [
        (_TEXT_ROWS, 64, 3),
        (_TEXT_ROWS, 10, 3),  # several answers cut, and KL rows re-cut
        (_TEXT_ROWS, 64, 2),
        (_chat(_TEXT_ROWS), 64, 3),
    ])
    def test_ids_labels_and_kl_rows_match(self, rows, max_length, batch_size):
        tokenizer = _hf_tokenizer()
        trl = _trl_pipeline(rows, tokenizer, max_length, batch_size)
        ours = _examples(rows, tokenizer, max_length=max_length, group_size=batch_size)
        assert len(ours) == len(trl)
        for ex, ref in zip(ours, trl):
            assert ex.prompt_ids + ex.completion_ids == list(ref["completion_input_ids"])
            assert ex.prompt_ids + ex.kl_completion_ids == list(ref["KL_completion_input_ids"])
            for prefix in ("", "KL_"):
                labels = list(ref[f"{prefix}completion_labels"])
                assert sum(1 for x in labels if x == -100) == len(ex.prompt_ids)
            assert ex.label is ref["label"]


# --------------------------------------------------------------------------
# The objective against a float64 NumPy oracle
# --------------------------------------------------------------------------


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _np_kto(pol, pol_kl, ref, ref_kl, labels, beta=BETA, wd=1.0, wu=1.0, *, z=None,
            reverse=False, matched_z=False, clamp=True, equal_classes=False):
    """KTO as TRL 0.29.1 computes it; ``z`` frozen when given (it is detached)."""
    labels = np.asarray(labels, dtype=bool)
    if reverse:
        labels = ~labels
    if z is None:
        raw = (pol - ref).mean() if matched_z else (pol_kl - ref_kl).mean()
        z = max(raw, 0.0) if clamp else raw
    r = pol - ref
    losses = np.where(labels, wd * (1.0 - _sig(beta * (r - z))),
                      wu * (1.0 - _sig(beta * (z - r))))
    if equal_classes:
        return 0.5 * (losses[labels].mean() + losses[~labels].mean())
    return losses.mean()


def _np_z(pol_kl, ref_kl):
    return max((pol_kl - ref_kl).mean(), 0.0)


#: Unbalanced labels (3 desirable, 1 undesirable); summed log-probs.
_POL = np.array([-6.2, -3.1, -8.4, -4.9])
_POL_KL = np.array([-7.0, -5.5, -6.1, -9.3])
_LABELS = np.array([True, False, True, True])
_DELTA = np.array([0.8, -0.5, 1.5, -1.2])  # r = policy - reference


def _logps_case(kl_shift):
    """``kl_shift`` > 0 gives z = kl_shift; < 0 gives a negative raw z (clamped)."""
    return _POL, _POL_KL, _POL - _DELTA, _POL_KL - kl_shift, _LABELS


#: (kl_shift, beta, desirable_weight, undesirable_weight)
_SETTINGS = [(0.7, 0.1, 1.0, 1.0), (-0.4, 0.1, 1.0, 1.0), (0.7, 0.5, 1.3, 0.7),
             (1.9, 1.0, 1.0, 2.0)]


class TestTheOracleMatchesTrl:
    """No MLX: the oracle against TRL's own ``KTOTrainer.kto_loss`` (bound on a
    stub), so a CI runner without MLX still checks what the MLX tests compare to."""

    def _trl(self, pol, pol_kl, ref, ref_kl, labels, beta, wd, wu):
        torch = pytest.importorskip("torch")
        pytest.importorskip("trl")
        from trl.experimental.kto.kto_trainer import KTOTrainer

        stub = types.SimpleNamespace(
            calculate_KL=True, loss_type="kto", beta=beta, desirable_weight=wd,
            undesirable_weight=wu,
            accelerator=types.SimpleNamespace(gather_for_metrics=lambda x: x, device="cpu"),
        )
        lab = torch.tensor(labels)
        p = torch.tensor(pol, dtype=torch.float64, requires_grad=True)
        pk = torch.tensor(pol_kl, dtype=torch.float64, requires_grad=True)
        r = torch.tensor(ref, dtype=torch.float64)
        rk = torch.tensor(ref_kl, dtype=torch.float64)
        losses, chosen, rejected, kl = KTOTrainer.kto_loss(
            stub, p[lab], p[~lab], pk, r[lab], r[~lab], rk
        )
        loss = losses.nanmean()
        loss.backward()
        return loss.item(), kl.item(), p.grad.numpy(), pk.grad, chosen, rejected

    @pytest.mark.parametrize("shift, beta, wd, wu", _SETTINGS)
    def test_loss_z_rewards_and_gradient(self, shift, beta, wd, wu):
        pol, pol_kl, ref, ref_kl, labels = _logps_case(shift)
        loss, kl, grad, kl_grad, chosen, rejected = self._trl(
            pol, pol_kl, ref, ref_kl, labels, beta, wd, wu
        )
        assert _np_kto(pol, pol_kl, ref, ref_kl, labels, beta, wd, wu) == pytest.approx(
            loss, rel=1e-12)
        assert _np_z(pol_kl, ref_kl) == pytest.approx(kl, abs=1e-12)
        assert kl_grad is None  # z is detached
        np.testing.assert_allclose(chosen.numpy(), beta * _DELTA[labels], rtol=1e-12)
        np.testing.assert_allclose(rejected.numpy(), beta * _DELTA[~labels], rtol=1e-12)
        eps = 1e-6
        z = _np_z(pol_kl, ref_kl)
        for i in range(len(pol)):
            hi, lo = pol.copy(), pol.copy()
            hi[i] += eps
            lo[i] -= eps
            fd = (_np_kto(hi, pol_kl, ref, ref_kl, labels, beta, wd, wu, z=z)
                  - _np_kto(lo, pol_kl, ref, ref_kl, labels, beta, wd, wu, z=z)) / (2 * eps)
            assert fd == pytest.approx(grad[i], rel=1e-6, abs=1e-12), i

    def test_the_negative_case_really_clamps(self):
        _, pol_kl, _, ref_kl, _ = _logps_case(-0.4)
        assert (pol_kl - ref_kl).mean() == pytest.approx(-0.4)
        assert self._trl(*_logps_case(-0.4), BETA, 1.0, 1.0)[1] == 0.0

    @pytest.mark.parametrize("label", [True, False])
    def test_a_single_label_batch_is_the_mean_over_its_rows(self, label):
        pol, pol_kl, ref, ref_kl, _ = _logps_case(0.7)
        labels = np.array([label] * 4)
        loss = self._trl(pol, pol_kl, ref, ref_kl, labels, BETA, 1.0, 1.0)[0]
        assert math.isfinite(loss)
        assert _np_kto(pol, pol_kl, ref, ref_kl, labels) == pytest.approx(loss, rel=1e-12)


class TestTheOracleSeesEveryListedMutation:
    @pytest.mark.parametrize("mutation", [
        "reverse", "matched_z", "no_clamp", "equal_classes",
    ])
    def test_each_mutation_moves_the_loss(self, mutation):
        case = _logps_case(-0.4 if mutation == "no_clamp" else 0.7)
        right = _np_kto(*case)
        flag = {"no_clamp": ("clamp", False)}.get(mutation, (mutation, True))
        wrong = _np_kto(*case, **dict([flag]))
        assert abs(wrong - right) > 1e-4

    def test_summed_and_mean_logps_differ_on_the_fixture(self):
        logits, tokens, mask, n = _fixture()
        assert not np.allclose(_np_logps(logits, tokens, mask),
                               _np_logps(logits, tokens, mask, mean=True))


def _np_logps(logits, tokens, mask, *, mean=False):
    shifted = logits[:, :-1] - logits[:, :-1].max(-1, keepdims=True)
    logsm = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    lp = np.take_along_axis(logsm, tokens[:, 1:][..., None], -1)[..., 0]
    m = mask[:, 1:].astype(np.float64)
    total = (lp * m).sum(-1)
    return total / m.sum(-1) if mean else total


def _fixture(seed=0, vocab=12):
    """Four examples, three desirable, unequal prompt / completion / KL lengths."""
    from soup_cli.trainer.mlx_kto import KtoExample, collate_kto

    examples = [
        KtoExample([1, 5, 6], [7, 8, 2], [9, 10, 2], True),
        KtoExample([1, 3], [4, 2], [7, 8, 2], False),
        KtoExample([1, 11, 4, 6], [9, 10, 2], [4, 2], True),
        KtoExample([1, 8], [5, 5, 6, 2], [7, 2], True),
    ]
    tokens, mask = (np.array(x) for x in collate_kto(examples))
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, size=(tokens.shape[0], tokens.shape[1], vocab))
    return logits, tokens, mask, len(examples)


class _RowKeyed:
    """Fixed logits per distinct token row, so separate forwards of the
    matched and KL halves each get their own rows' logits."""

    def __new__(cls, logits, tokens):
        nn = pytest.importorskip("mlx.nn")
        mx = pytest.importorskip("mlx.core")
        table = np.asarray(tokens)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = mx.array(logits.astype(np.float32))

            def __call__(self, x):
                rows = np.array(x)
                width = rows.shape[1]
                idx = [next(i for i, t in enumerate(table) if np.array_equal(t[:width], r))
                       for r in rows]
                return self.logits[mx.array(idx), :width]

        return Model()


def _refs(logits, tokens, mask, n, shift=0.7):
    pol = _np_logps(logits, tokens, mask)
    return pol[:n] - _DELTA, pol[n:] - shift


class TestTheLossMatchesTheOracle:
    def _mlx(self, logits, tokens, mask, n, ref, ref_kl, beta=BETA, wd=1.0, wu=1.0):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_kto import kto_logps, kto_loss

        model = _RowKeyed(logits, tokens)
        pol, pol_kl = kto_logps(model, mx.array(tokens), mx.array(mask), n)
        labels = mx.array(_LABELS)
        loss, metrics = kto_loss(pol, pol_kl, mx.array(ref), mx.array(ref_kl), labels, beta,
                                 desirable_weight=wd, undesirable_weight=wu)
        return np.array(pol), np.array(pol_kl), float(loss.item()), metrics

    @pytest.mark.parametrize("shift, beta, wd, wu", _SETTINGS)
    def test_terms_total_and_metrics(self, shift, beta, wd, wu):
        logits, tokens, mask, n = _fixture()
        ref, ref_kl = _refs(logits, tokens, mask, n, shift)
        pol, pol_kl, loss, metrics = self._mlx(logits, tokens, mask, n, ref, ref_kl,
                                               beta, wd, wu)
        want = _np_logps(logits, tokens, mask)
        np.testing.assert_allclose(pol, want[:n], rtol=1e-5)
        np.testing.assert_allclose(pol_kl, want[n:], rtol=1e-5)
        assert loss == pytest.approx(
            _np_kto(want[:n], want[n:], ref, ref_kl, _LABELS, beta, wd, wu), rel=1e-5)
        z = _np_z(want[n:], ref_kl)
        assert metrics["kl"].item() == pytest.approx(z, abs=1e-5)
        r = want[:n] - ref
        rows = len(_LABELS)
        assert metrics["chosen_share"].item() == pytest.approx(_LABELS.mean())
        assert metrics["chosen_rewards_per_row"].item() == pytest.approx(
            beta * r[_LABELS].sum() / rows, rel=1e-5)
        assert metrics["rejected_rewards_per_row"].item() == pytest.approx(
            beta * r[~_LABELS].sum() / rows, rel=1e-5)
        assert metrics["chosen_logps_per_row"].item() == pytest.approx(
            want[:n][_LABELS].sum() / rows, rel=1e-5)
        assert metrics["rejected_logps_per_row"].item() == pytest.approx(
            want[:n][~_LABELS].sum() / rows, rel=1e-5)

    def test_the_chunked_head_gives_the_same_logps(self):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_kto import kto_logps

        logits, tokens, mask, n = _fixture()
        model = _RowKeyed(logits, tokens)
        split = (model, lambda h: h)
        full = kto_logps(model, mx.array(tokens), mx.array(mask), n)
        chunked = kto_logps(model, mx.array(tokens), mx.array(mask), n, head_split=split)
        for x, y in zip(full, chunked):
            assert mx.allclose(x, y, atol=1e-5).item()

    @pytest.mark.parametrize("shift, beta, wd, wu", [_SETTINGS[0], _SETTINGS[2]])
    def test_gradients_match_finite_differences_with_z_detached(self, shift, beta, wd, wu):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        from soup_cli.trainer.mlx_kto import kto_logps, kto_loss

        logits, tokens, mask, n = _fixture(seed=1)
        ref, ref_kl = _refs(logits, tokens, mask, n, shift)
        model = _RowKeyed(logits, tokens)
        t, m = mx.array(tokens), mx.array(mask)

        def f(mod):
            pol, pol_kl = kto_logps(mod, t, m, n)
            return kto_loss(pol, pol_kl, mx.array(ref), mx.array(ref_kl), mx.array(_LABELS),
                            beta, desirable_weight=wd, undesirable_weight=wu)[0]

        _, grads = nn.value_and_grad(model, f)(model)
        g = np.array(grads["logits"])
        # z is a constant: no gradient reaches the KL rows at all.
        assert np.abs(g[n:]).max() == 0.0
        assert np.abs(g[:n]).max() > 1e-4
        z = _np_z(_np_logps(logits, tokens, mask)[n:], ref_kl)

        def oracle(lg):
            pol = _np_logps(lg, tokens, mask)[:n]
            return _np_kto(pol, None, ref, None, _LABELS, beta, wd, wu, z=z)

        rng = np.random.default_rng(5)
        eps = 1e-4
        scored = np.argwhere(mask[:n, 1:] == 1)
        for _ in range(12):
            r, p = scored[rng.integers(0, len(scored))]
            idx = (r, p, rng.integers(0, logits.shape[2]))
            hi, lo = logits.copy(), logits.copy()
            hi[idx] += eps
            lo[idx] -= eps
            fd = (oracle(hi) - oracle(lo)) / (2 * eps)
            assert g[idx] == pytest.approx(fd, abs=2e-5, rel=2e-2), idx

    def test_prompt_and_padding_logits_move_nothing(self):
        logits, tokens, mask, n = _fixture()
        ref, ref_kl = _refs(logits, tokens, mask, n)
        base = self._mlx(logits, tokens, mask, n, ref, ref_kl)
        bumped = logits.copy()
        not_scored = mask[:, 1:] == 0
        bumped[:, :-1][not_scored] += np.random.default_rng(3).normal(
            0.0, 3.0, size=(int(not_scored.sum()), logits.shape[-1]))
        moved = self._mlx(bumped, tokens, mask, n, ref, ref_kl)
        np.testing.assert_allclose(moved[0], base[0], rtol=1e-6)
        np.testing.assert_allclose(moved[1], base[1], rtol=1e-6)
        assert moved[2] == pytest.approx(base[2], rel=1e-6)


class TestEdgeValues:
    def _loss(self, labels, shift=0.7):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_kto import kto_loss

        pol, pol_kl, ref, ref_kl, _ = _logps_case(shift)
        return kto_loss(mx.array(pol), mx.array(pol_kl), mx.array(ref), mx.array(ref_kl),
                        mx.array(np.asarray(labels)), BETA)

    @pytest.mark.parametrize("label", [True, False])
    def test_a_single_label_batch_is_finite_and_matches(self, label):
        labels = [label] * 4
        loss, metrics = self._loss(labels)
        pol, pol_kl, ref, ref_kl, _ = _logps_case(0.7)
        assert loss.item() == pytest.approx(_np_kto(pol, pol_kl, ref, ref_kl, labels), rel=1e-5)
        assert all(math.isfinite(v.item()) for v in metrics.values())

    def test_at_the_reference_the_loss_is_one_half_minus_the_kl_shift(self):
        """r = 0 and z = 0: every term is 1 - sigmoid(0) = 0.5."""
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_kto import kto_loss

        x = mx.array([-3.0, -4.0])
        loss, metrics = kto_loss(x, x, x, x, mx.array([True, False]), BETA)
        assert loss.item() == pytest.approx(0.5)
        assert metrics["kl"].item() == 0.0

    def test_nan_input_is_not_a_quiet_zero(self):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_kto import kto_loss

        x = mx.array([-3.0, -4.0])
        loss, _ = kto_loss(x * mx.array([float("nan"), 1.0]), x, x, x,
                           mx.array([True, False]), BETA)
        assert math.isnan(loss.item())


# --------------------------------------------------------------------------
# Data refusals (no MLX)
# --------------------------------------------------------------------------


class TestLabelsAreTheLoaders:
    def test_labels_read_through_load_dataset_match_the_transformers_path(
        self, tmp_path, monkeypatch
    ):
        """``load_dataset`` runs the loader's KTO converter for every backend: the
        transformers trainer trains on these rows' labels as they are, and MLX
        must read exactly the same ones (no second reading, no refusal)."""
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.loader import load_dataset

        raw = [True, False, "true", 1, -1, "no", 0]
        path = tmp_path / "kto.jsonl"
        path.write_text("".join(
            json.dumps({"prompt": f"p{i}:", "completion": f" c{i}", "label": label}) + "\n"
            for i, label in enumerate(raw)
        ), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        loaded = load_dataset(DataConfig(train=str(path), format="kto", val_split=0.0))
        transformers_labels = [row["label"] for row in loaded["train"]]
        assert transformers_labels == [True, False, True, True, False, False, False]
        examples, _ = _wrapper_no_mlx()._examples(loaded["train"])
        assert [e.label for e in examples] == transformers_labels


class TestDataRefusals:
    @pytest.mark.parametrize("label", ["maybe", None, 1, "true", 0.5])
    def test_a_row_bypassing_the_loader_with_a_non_bool_label_is_refused(self, label):
        """Only a guard for rows handed to the trainer without ``load_dataset``,
        which has already read every file row's label (test above)."""
        from soup_cli.trainer.mlx_dpo import PreferenceRowError

        rows = [{"prompt": "a", "completion": "b", "label": True},
                {"prompt": "c", "completion": "d", "label": label}]
        with pytest.raises(PreferenceRowError, match=r"kto train row 1: .*label must be"):
            _wrapper_no_mlx()._examples(rows)

    def test_a_missing_label_is_refused(self):
        from soup_cli.trainer.mlx_dpo import PreferenceRowError

        with pytest.raises(PreferenceRowError, match=r"kto val row 0: .*no 'label'"):
            _wrapper_no_mlx()._examples([{"prompt": "a", "completion": "b"}], "val")

    def test_an_empty_completion_is_refused(self):
        from soup_cli.trainer.mlx_dpo import PreferenceRowError

        rows = [{"prompt": "a", "completion": "", "label": True},
                {"prompt": "c", "completion": "d", "label": False}]
        with pytest.raises(PreferenceRowError, match="row 0: the completion is empty"):
            _wrapper_no_mlx()._examples(rows)

    @pytest.mark.parametrize("prompt_len, completion", [(10, "ok"), (16, "y" * 20)],
                             ids=["fills-the-room", "longer-than-the-room"])
    def test_a_prompt_that_fills_max_length_is_refused(self, prompt_len, completion):
        """A prompt longer than the room must not reach TRL's negative slice,
        which keeps the head of a long answer past ``max_length`` (review of
        #2111): the second row would train with the clamp at 0 removed."""
        from soup_cli.trainer.mlx_dpo import PreferenceRowError

        wrapper = _wrapper_no_mlx()
        wrapper.config.data.max_length = 12
        rows = [{"prompt": "P" * prompt_len, "completion": completion, "label": True},
                {"prompt": "c", "completion": "d", "label": False}]
        with pytest.raises(PreferenceRowError, match="row 0: no completion token fits"):
            wrapper._examples(rows)

    def test_an_empty_training_split_is_refused(self):
        with pytest.raises(ValueError, match="MLX KTO: the train split has no rows"):
            _wrapper_no_mlx()._examples([])

    def test_fewer_rows_than_batch_size_form_one_partial_batch(self):
        """TRL trains 3 rows at batch_size 4 as one batch and one KL group."""
        from soup_cli.trainer.mlx_kto import kl_partners, sequential_batches

        wrapper = _wrapper_no_mlx()
        wrapper.config.training.batch_size = 4
        wrapper._check_train_split_size({"train": _rows(3)})
        wrapper._check_train_rows(3, 4, 0)
        assert sequential_batches(3, 4, 1) == [[0, 1, 2]]
        assert kl_partners(3, 4) == [2, 0, 1]

    @pytest.mark.parametrize("check", ["split", "rows"])
    def test_one_row_is_refused_in_kto_words(self, check):
        wrapper = _wrapper_no_mlx()
        with pytest.raises(ValueError) as info:
            if check == "split":
                wrapper._check_train_split_size({"train": _rows(1)})
            else:
                wrapper._check_train_rows(1, 2, 0)
        message = str(info.value)
        assert "MLX KTO: the training split has 1 row(s); KTO needs at least 2" in message
        assert "pair" not in message and "tie" not in message
        assert "Lower training.batch_size" not in message

    def test_an_empty_val_split_is_fine(self):
        assert _wrapper_no_mlx()._examples([], "val") == ([], 0)


def _wrapper_no_mlx():
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.mlx_kto import MLXKTOTrainerWrapper

    wrapper = MLXKTOTrainerWrapper(load_config_from_string(_KTO_YAML))
    wrapper.tokenizer = CharTokenizer()
    return wrapper


class TestBatching:
    def test_batches_run_in_dataset_order_with_the_last_partial_one(self):
        """TRL's SequentialSampler and DataLoader(drop_last=False)."""
        from soup_cli.trainer.mlx_kto import sequential_batches

        assert sequential_batches(7, 2, 2) == [[0, 1], [2, 3], [4, 5], [6]] * 2
        assert sequential_batches(4, 2, 1) == [[0, 1], [2, 3]]

    def test_the_trainer_plans_sequential_batches(self):
        assert _wrapper_no_mlx()._plan_batches(5, 2, 1, 123) == [[0, 1], [2, 3], [4]]

    @pytest.mark.parametrize("value, expected", [("auto", 2), (2, 2), (5, 5)])
    def test_the_batch_size_never_drops_below_two(self, value, expected):
        wrapper = _wrapper_no_mlx()
        wrapper.config.training.batch_size = value
        assert wrapper._resolve_batch_size() == expected


class TestClassMeans:
    """The reported ``val_rewards_*`` / ``val_logps_*`` from the per-row sums (no MLX)."""

    def test_each_class_mean_is_its_per_row_sum_over_its_share(self):
        from soup_cli.trainer.mlx_kto import class_means

        metrics = {
            "val_chosen_share": 0.25,
            "val_chosen_rewards_per_row": 0.5,
            "val_rejected_rewards_per_row": -0.6,
            "val_chosen_logps_per_row": -1.0,
            "val_rejected_logps_per_row": -6.0,
        }
        assert class_means(metrics, prefix="val_") == {
            "val_rewards_chosen": pytest.approx(2.0),
            "val_rewards_rejected": pytest.approx(-0.8),
            "val_logps_chosen": pytest.approx(-4.0),
            "val_logps_rejected": pytest.approx(-8.0),
            "val_rewards_margins": pytest.approx(2.8),
        }

    def test_a_split_with_one_label_reports_that_label_only(self):
        from soup_cli.trainer.mlx_kto import class_means

        metrics = {
            "chosen_share": 0.0,
            "chosen_rewards_per_row": 0.0,
            "rejected_rewards_per_row": -0.6,
            "chosen_logps_per_row": 0.0,
            "rejected_logps_per_row": -6.0,
        }
        assert class_means(metrics) == {
            "rewards_rejected": pytest.approx(-0.6),
            "logps_rejected": pytest.approx(-6.0),
        }


# --------------------------------------------------------------------------
# Training on a tiny random Llama
# --------------------------------------------------------------------------

_KTO_YAML = """
base: tiny-base
task: kto
backend: mlx
data:
  train: ./x.jsonl
  format: kto
  max_length: 128
training:
  epochs: 1
  lr: 5e-3
  kto_beta: 0.1
  quantization: none
  lora: {r: 4, alpha: 8, dropout: 0.0}
output: ./output
"""


def _rows(n=8, prefix="q"):
    return [{"prompt": f"{prefix}{i}:", "completion": " good" if i % 2 == 0 else " bad",
             "label": i % 2 == 0} for i in range(n)]


def _yaml(**training):
    """The fixture with ``training`` keys replacing or adding lines."""
    text = _KTO_YAML
    for key, value in training.items():
        line = f"  {key}: {value}\n"
        own = [x for x in text.splitlines(keepends=True) if x.startswith(f"  {key}:")]
        text = text.replace(own[0], line) if own else text.replace(
            "  kto_beta: 0.1\n", "  kto_beta: 0.1\n" + line)
    return text


def _wrapper(tmp_path, monkeypatch, model_seed=0, **training):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import llama

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.mlx_kto import MLXKTOTrainerWrapper

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    training = {"batch_size": 2, "gradient_accumulation_steps": 1, **training}
    wrapper = MLXKTOTrainerWrapper(load_config_from_string(_yaml(**training)))
    mx.random.seed(model_seed)
    wrapper.model = llama.Model(_tiny_llama_args())
    wrapper.tokenizer = CharTokenizer()
    wrapper._dataset = {"train": _rows(), "val": _rows(2, prefix="v")}
    return wrapper


def _base_logps(wrapper, examples, model_seed=0):
    """The untrained base model's summed log-probs: matched rows, then KL rows."""
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models import llama

    from soup_cli.trainer.mlx_dpo import sequence_logps

    mx.random.seed(model_seed)
    base = llama.Model(_tiny_llama_args())
    base.eval()
    tokens, mask = wrapper._to_arrays(examples)
    return np.array(sequence_logps(base, tokens, mask))


class TestTraining:
    def test_lora_trains_and_the_base_stays_frozen(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch)
        before = _flat(wrapper.model.parameters())
        result = wrapper.train()
        assert result["losses"] and all(math.isfinite(x) for x in result["losses"])
        after = _flat(wrapper.model.parameters())
        lora_b = [v for k, v in after.items() if k.endswith("lora_b")]
        assert lora_b and any(np.abs(v).sum() > 0 for v in lora_b)
        for name, value in before.items():
            moved = name.replace(".weight", ".linear.weight")
            key = name if name in after else moved
            np.testing.assert_array_equal(after[key], value, err_msg=name)

    def test_the_reference_is_the_base_model_after_several_updates(self, tmp_path, monkeypatch):
        from soup_cli.trainer.mlx_dpo import sequence_logps

        wrapper = _wrapper(tmp_path, monkeypatch, epochs=3, warmup_ratio=0.0)
        result = wrapper.train()
        assert len(result["losses"]) >= 6
        examples, _ = wrapper._examples(_rows())
        base = _base_logps(wrapper, examples)
        ref_matched, ref_kl = (np.array(x) for x in wrapper._ref["train"])
        n = len(examples)
        np.testing.assert_allclose(ref_matched, base[:n], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(ref_kl, base[n:], rtol=1e-4, atol=1e-4)
        wrapper.model.eval()
        trained = np.array(sequence_logps(wrapper.model, *wrapper._to_arrays(examples)))
        assert np.abs(trained - base).max() > 1e-2

    def test_the_preference_is_learned_and_the_metadata_written(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, epochs=6, warmup_ratio=0.0)
        result = wrapper.train()
        assert all(math.isfinite(x) for x in result["losses"])
        assert result["val_rewards_chosen"] > 0 > result["val_rewards_rejected"]
        assert result["val_rewards_margins"] == pytest.approx(
            result["val_rewards_chosen"] - result["val_rewards_rejected"])
        assert result["val_kl"] >= 0
        meta = json.loads((tmp_path / "output" / "adapter_config.json").read_text())
        assert meta["task"] == "kto" and meta["fine_tune_type"] == "lora"
        assert (meta["kto_beta"], meta["kto_loss_type"]) == (0.1, "kto")
        assert (meta["desirable_weight"], meta["undesirable_weight"]) == (1.0, 1.0)
        assert meta["kl_group_size"] == 2
        assert not list((tmp_path / "output").glob("*.partial*"))

    def test_training_moves_held_out_rewards_against_the_untrained_model(
        self, tmp_path, monkeypatch
    ):
        """The base model is its own control: rewards are 0 before any update."""
        mx = pytest.importorskip("mlx.core")

        wrapper = _wrapper(tmp_path, monkeypatch, epochs=6, warmup_ratio=0.0)
        val = _rows(2, prefix="v")
        examples, _ = wrapper._examples(val, "val")
        control = _base_logps(wrapper, examples)
        wrapper.train()
        wrapper.model.eval()
        from soup_cli.trainer.mlx_dpo import sequence_logps

        tokens, mask = wrapper._to_arrays(examples)
        trained = np.array(sequence_logps(wrapper.model, tokens, mask))
        r = trained[:2] - control[:2]
        assert r[0] > 0.1 > -0.1 > r[1]  # desirable up, undesirable down
        assert mx.array(r).shape == (2,)

    @pytest.mark.parametrize("label", [True, False])
    def test_single_label_data_trains_finitely(self, tmp_path, monkeypatch, label):
        wrapper = _wrapper(tmp_path, monkeypatch)
        wrapper._dataset["train"] = [dict(r, label=label) for r in _rows()]
        result = wrapper.train()
        assert all(math.isfinite(x) for x in result["losses"])

    def test_the_last_partial_batch_trains_and_val_chunks_weigh_by_rows(
        self, tmp_path, monkeypatch
    ):
        mx = pytest.importorskip("mlx.core")

        wrapper = _wrapper(tmp_path, monkeypatch)
        wrapper._dataset["train"] = _rows(7)
        wrapper._dataset["val"] = _rows(3, prefix="v")
        result = wrapper.train()
        assert result["total_steps"] == 4  # [0,1] [2,3] [4,5] [6], as TRL
        examples, _ = wrapper._examples(wrapper._dataset["val"], "val")
        wrapper.model.eval()
        chunks = [(0, 2), (2, 3)]
        total = 0.0
        for lo, hi in chunks:
            arrays = wrapper._batch_arrays(examples[lo:hi])
            loss, _ = wrapper._loss(wrapper.model, arrays, "val", mx.arange(lo, hi))
            total += loss.item() * (hi - lo)
        assert result["val_loss"] == pytest.approx(total / 3, rel=1e-4)

    def test_an_accumulated_update_is_the_mean_of_its_micro_batch_gradients(
        self, tmp_path, monkeypatch
    ):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        from mlx.utils import tree_flatten

        settings = dict(gradient_accumulation_steps=2, optimizer="sgd", scheduler="constant",
                        warmup_ratio=0.0, weight_decay=0.0, max_grad_norm=1000.0, seed=4,
                        lr=0.5)
        trained = _wrapper(tmp_path / "a", monkeypatch, **settings)
        trained._dataset = {"train": _rows(4), "val": []}
        result = trained.train()
        assert len(result["losses"]) == 1
        got = _flat(trained.model.trainable_parameters())

        # The same start, by hand: seed, LoRA, reference, two micro-batch gradients.
        manual = _wrapper(tmp_path / "b", monkeypatch, **settings)
        mx.random.seed(4)
        examples, _ = manual._examples(_rows(4))
        manual._apply_lora(manual.model)
        manual._prepare_objective(examples, [], 2)
        start = _flat(manual.model.trainable_parameters())

        def micro(lo, hi):
            def f(model):
                return manual._loss(model, manual._batch_arrays(examples[lo:hi]), "train",
                                    mx.arange(lo, hi))[0]

            _, grads = nn.value_and_grad(manual.model, f)(manual.model)
            return {k: np.array(v) for k, v in tree_flatten(grads)}

        g1, g2 = micro(0, 2), micro(2, 4)
        assert any(np.abs(g1[k]).max() > 1e-5 for k in g1)
        for name, value in start.items():
            want = value - 0.5 * (g1[name] + g2[name]) / 2
            np.testing.assert_allclose(got[name], want, atol=2e-5, err_msg=name)

    def test_a_nonfinite_loss_stops_the_run(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer import mlx_kto

        real = mlx_kto.kto_logps

        def poisoned(*args, **kwargs):
            pol, pol_kl = real(*args, **kwargs)
            return pol * mx.array(float("nan")), pol_kl

        monkeypatch.setattr(mlx_kto, "kto_logps", poisoned)
        wrapper = _wrapper(tmp_path, monkeypatch)
        with pytest.raises(FloatingPointError, match="MLX KTO: nonfinite loss"):
            wrapper.train()

    def test_a_bad_row_is_refused_before_any_update(self, tmp_path, monkeypatch):
        from soup_cli.trainer.mlx_dpo import PreferenceRowError

        wrapper = _wrapper(tmp_path, monkeypatch)
        wrapper._dataset["train"] = _rows(3) + [{"prompt": "x", "completion": "", "label": True}]
        with pytest.raises(PreferenceRowError, match="kto train row 3"):
            wrapper.train()
        assert not any("lora" in k for k in _flat(wrapper.model.parameters()))
        assert not (tmp_path / "output" / "adapters.safetensors").exists()


class TestAccumulationCut:
    def test_the_warning_names_the_untrained_batches_and_rows(self, monkeypatch):
        from io import StringIO

        from rich.console import Console

        from soup_cli.trainer import mlx_kto

        buffer = StringIO()
        monkeypatch.setattr(mlx_kto, "console", Console(file=buffer, width=400))
        _wrapper_no_mlx()._on_plan_cut([[8, 9], [0, 1]], 8, 4)
        text = _strip_ansi(buffer.getvalue())
        assert "gradient_accumulation_steps=4 keeps 8 of 10 batches" in text
        assert "batch(es) 9-10 of the run (4 row(s): 0, 1, 8, 9) are not trained" in text

    def test_a_run_whose_plan_is_cut_warns_and_trains_the_kept_batches(
        self, tmp_path, monkeypatch
    ):
        from io import StringIO

        from rich.console import Console

        from soup_cli.trainer import mlx_kto

        buffer = StringIO()
        monkeypatch.setattr(mlx_kto, "console", Console(file=buffer, width=400))
        wrapper = _wrapper(tmp_path, monkeypatch, gradient_accumulation_steps=2)
        wrapper._dataset = {"train": _rows(5), "val": []}  # [0,1] [2,3] [4]
        result = wrapper.train()
        assert result["total_steps"] == 2
        text = _strip_ansi(buffer.getvalue())
        assert "keeps 2 of 3 batches; batch(es) 3 of the run (1 row(s): 4)" in text

    def test_no_cut_no_warning(self, tmp_path, monkeypatch):
        from io import StringIO

        from rich.console import Console

        from soup_cli.trainer import mlx_kto

        buffer = StringIO()
        monkeypatch.setattr(mlx_kto, "console", Console(file=buffer, width=400))
        wrapper = _wrapper(tmp_path, monkeypatch, gradient_accumulation_steps=2)
        wrapper._dataset = {"train": _rows(4), "val": []}
        wrapper.train()
        assert "are not trained" not in _strip_ansi(buffer.getvalue())

    def test_three_rows_at_batch_four_train_as_one_batch(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, batch_size=4)
        wrapper._dataset = {"train": _rows(3), "val": []}
        result = wrapper.train()
        assert result["total_steps"] == 1 and math.isfinite(result["losses"][0])


class TestResume:
    def _first_run(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path / "a", monkeypatch, epochs=3, seed=1)
        fresh = wrapper.train()
        return tmp_path / "a" / "output" / "adapters.safetensors", fresh

    def test_resume_from_its_own_adapter_keeps_the_base_reference(self, tmp_path, monkeypatch):
        checkpoint, fresh = self._first_run(tmp_path, monkeypatch)
        resumed = _wrapper(tmp_path / "b", monkeypatch, seed=1)
        result = resumed.train(resume_from_checkpoint=str(checkpoint))
        assert result["losses"][0] < fresh["losses"][0]
        examples, _ = resumed._examples(_rows())
        base = _base_logps(resumed, examples)
        np.testing.assert_allclose(np.array(resumed._ref["train"][0]), base[: len(examples)],
                                   rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("change, match", [
        ({"task": "dpo"}, "MLX KTO: .* task='dpo', not 'kto'"),
        ({"task": "orpo"}, "task='orpo'"),
        ({"model": "other-base"}, "belongs to base"),
        (None, "MLX KTO: no adapter_config.json"),
    ])
    def test_mismatched_or_missing_metadata_is_refused(self, tmp_path, monkeypatch, change, match):
        checkpoint, _ = self._first_run(tmp_path, monkeypatch)
        meta = checkpoint.parent / "adapter_config.json"
        if change is None:
            meta.unlink()
        else:
            meta.write_text(json.dumps({**json.loads(meta.read_text()), **change}))
        resumed = _wrapper(tmp_path / "b", monkeypatch, seed=1)
        with pytest.raises(ValueError, match=match):
            resumed.train(resume_from_checkpoint=str(checkpoint))


class TestFreshProcessReload:
    def test_logits_reproduce_and_the_adapter_has_an_effect(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")

        wrapper = _wrapper(tmp_path, monkeypatch, epochs=3)
        base_path = tmp_path / "base.safetensors"
        wrapper.model.save_weights(str(base_path))
        wrapper.train()
        examples, _ = wrapper._examples(_rows(2, prefix="v"), "val")
        tokens, _ = wrapper._to_arrays(examples)
        wrapper.model.eval()
        logits = wrapper.model(tokens).astype(mx.float32)
        mx.eval(logits)
        np.save(tmp_path / "tokens.npy", np.array(tokens))

        script = textwrap.dedent(f"""
            import sys
            import numpy as np
            import mlx.core as mx
            from mlx_lm.models import llama
            from mlx_lm.tuner.utils import load_adapters
            args = llama.ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=2,
                intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                rms_norm_eps=1e-5, vocab_size=200, tie_word_embeddings=True)
            model = llama.Model(args)
            model.load_weights({str(base_path)!r})
            if sys.argv[1] == "adapter":
                load_adapters(model, {str(tmp_path / "output")!r})
            model.eval()
            t = mx.array(np.load({str(tmp_path / "tokens.npy")!r}))
            np.save(sys.argv[2], np.array(model(t).astype(mx.float32)))
        """)

        def run(mode):
            out = tmp_path / f"{mode}.npy"
            subprocess.run([sys.executable, "-c", script, mode, str(out)], capture_output=True,
                           text=True, check=True, timeout=300)
            return np.load(out)

        reloaded, base = run("adapter"), run("base")
        np.testing.assert_allclose(reloaded, np.array(logits), rtol=1e-4, atol=1e-5)
        assert np.abs(base - reloaded).max() > 1e-3


# --------------------------------------------------------------------------
# Config and dispatch (no MLX)
# --------------------------------------------------------------------------


class TestConfig:
    def _load(self, top="", training=""):
        """`training` lines replace the fixture's own line for the same key."""
        from soup_cli.config.loader import load_config_from_string

        text = _KTO_YAML.replace("backend: mlx\n", "backend: mlx\n" + top)
        for line in training.splitlines(keepends=True):
            key = line.split(":", 1)[0]
            own = [x for x in text.splitlines(keepends=True) if x.split(":", 1)[0] == key]
            text = text.replace(own[0], line) if own else text.replace(
                "  kto_beta: 0.1\n", "  kto_beta: 0.1\n" + line
            )
        return load_config_from_string(text)

    def test_kto_on_mlx_loads_and_is_an_mlx_task(self):
        from soup_cli.config.schema import MLX_TASKS

        cfg = self._load(training="  kto_beta: 0.3\n")
        assert (cfg.task, cfg.backend, cfg.training.kto_beta) == ("kto", "mlx", 0.3)
        assert "kto" in MLX_TASKS

    def test_pre_quantized_4bit_loads(self):
        assert self._load(training="  quantization: 4bit\n").training.quantization == "4bit"

    def test_kto_dispatches_to_the_native_trainer(self):
        from soup_cli.trainer.mlx_kto import MLXKTOTrainerWrapper
        from soup_cli.trainer.mlx_routing import MLX_TRAINER_REGISTRY, get_mlx_trainer

        assert "kto" in MLX_TRAINER_REGISTRY and "kto" in MLX_TRAINER_REGISTRY.keys()
        assert get_mlx_trainer("kto") is MLXKTOTrainerWrapper

    @pytest.mark.parametrize("top, training, match", [
        ("", "  quantization: 8bit\n", "quantization='8bit' is not supported by task='kto'"),
        ("modality: vision\n", "", "modality='vision' is not supported by task='kto'"),
        ("", "  lora: {r: 0}\n", "lora.r must be at least 1 for task='kto'"),
        ("", "  loraplus_lr_ratio: 16\n", "loraplus_lr_ratio is not implemented on backend='mlx'"),
        ("", "  lora: {r: 4, use_dora: true}\n", "MLX DoRA requires task='sft'"),
        ("", "  batch_size: 1\n", "task='kto' requires training.batch_size >= 2"),
    ])
    def test_unsupported_combinations_are_refused_at_load(self, top, training, match):
        with pytest.raises(Exception, match=match):
            self._load(top, training)

    def test_batch_size_auto_loads(self):
        assert self._load(training="  batch_size: auto\n").training.batch_size == "auto"

    def test_other_tasks_name_kto_among_the_mlx_ones(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(Exception, match="only ships SFT, DPO, ORPO, SimPO, KTO and GRPO"):
            load_config_from_string(_KTO_YAML.replace("task: kto", "task: ipo"))

    def test_the_transformers_route_is_unchanged(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_KTO_YAML.replace("backend: mlx", "backend: transformers"))
        assert cfg.backend == "transformers" and cfg.task == "kto"

    def test_doctor_reports_ignored_settings(self):
        from soup_cli.config.backend_support import check_config

        cfg = self._load(training="  use_galore: true\n")
        assert {e.field for e in check_config(cfg)} == {"training.use_galore"}

    def test_the_ignored_settings_are_printed(self, monkeypatch):
        from io import StringIO

        from rich.console import Console

        from soup_cli.trainer import mlx_dpo
        from soup_cli.trainer.mlx_kto import MLXKTOTrainerWrapper

        buffer = StringIO()
        monkeypatch.setattr(mlx_dpo, "console", Console(file=buffer, width=400))
        MLXKTOTrainerWrapper(self._load(training="  use_galore: true\n"))._check_unsupported()
        assert "MLX backend ignores: GaLore" in _strip_ansi(buffer.getvalue())
