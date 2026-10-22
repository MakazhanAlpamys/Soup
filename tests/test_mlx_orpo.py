"""Native ORPO on MLX (#2004).

The objective is pinned to TRL 0.29.1's ``experimental/orpo/orpo_trainer.py``,
which Soup's transformers backend runs for ``task: orpo``. Three layers:

- tokenization against TRL's own ``ORPOTrainer.tokenize_row`` on a real
  ``PreTrainedTokenizerFast`` (needs ``trl`` / ``transformers``, no MLX);
- the loss, its terms and its gradients against an independent float64 NumPy
  oracle on fixed nonconstant logits (needs MLX), with every mutation the issue
  lists shown to move the result;
- training, saving and a fresh-process reload on a tiny random Llama (needs
  MLX and mlx-lm). No downloads.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from tests.test_mlx_dpo import _strip_ansi

# --------------------------------------------------------------------------
# A character tokenizer with HF's surface: encode(add_special_tokens=...),
# bos/eos ids and a text chat template. Ids stay inside a 200-token vocab.
# --------------------------------------------------------------------------

BOS, EOS = 1, 2


class CharTokenizer:
    bos_token_id = BOS
    eos_token_id = EOS
    eos_token = "</s>"
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        ids = [3 + (ord(c) % 190) for c in text]
        return ([BOS] if add_special_tokens else []) + ids

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            continue_final_message=False, **_):
        parts = [f"<{m['role']}>{m['content']}|" for m in messages]
        if continue_final_message:
            parts[-1] = parts[-1][:-1]
        text = "".join(parts) + ("<assistant>" if add_generation_prompt else "")
        return self.encode(text, add_special_tokens=False) if tokenize else text


def _decode(ids):
    return "".join(chr(i - 3) if i > 2 else {1: "^", 2: "$"}.get(i, "?") for i in ids)


class MergingTokenizer(CharTokenizer):
    """Merges ``xy`` into one token, so a prompt ending in ``x`` before an
    answer starting with ``y`` changes its last token in ``prompt + answer``."""

    MERGED = 199

    def encode(self, text, add_special_tokens=True):
        out = [BOS] if add_special_tokens else []
        i = 0
        while i < len(text):
            if text[i : i + 2] == "xy":
                out.append(self.MERGED)
                i += 2
            else:
                out.append(3 + (ord(text[i]) % 190))
                i += 1
        return out


def _build(row, tokenizer=None, max_length=256, max_prompt_length=128, mode="keep_end"):
    from soup_cli.trainer.mlx_orpo import build_orpo_example

    return build_orpo_example(
        row, tokenizer or CharTokenizer(), max_length=max_length,
        max_prompt_length=max_prompt_length, truncation_mode=mode,
    )


_U = {"role": "user", "content": "hi"}
_A = {"role": "assistant", "content": "a"}
_B = {"role": "assistant", "content": "bb"}


class TestTokenization:
    def test_text_row_gets_bos_and_eos(self):
        ex = _build({"prompt": "Q:", "chosen": " yes", "rejected": " no"})
        assert _decode(ex.chosen_ids) == "^Q: yes$"
        assert ex.chosen_prompt_len == 3
        assert _decode(ex.rejected_ids[ex.rejected_prompt_len:]) == " no$"

    def test_bos_and_eos_are_not_doubled(self):
        class WithSpecials(CharTokenizer):
            def encode(self, text, add_special_tokens=True):
                return [BOS] + super().encode(text, False) + ([EOS] if text.endswith("!") else [])

        ex = _build({"prompt": "Q", "chosen": "a!", "rejected": "b!"}, WithSpecials())
        assert ex.chosen_ids.count(BOS) == 1 and ex.chosen_ids.count(EOS) == 1

    def test_conversational_row_renders_like_trl(self):
        ex = _build({"prompt": [_U], "chosen": [_A], "rejected": [_B]})
        assert _decode(ex.chosen_ids) == "^<user>hi|<assistant>a|$"
        assert _decode(ex.chosen_ids[: ex.chosen_prompt_len]) == "^<user>hi|<assistant>"
        assert _decode(ex.rejected_ids[ex.rejected_prompt_len:]) == "bb|$"

    def test_prompt_ending_in_assistant_continues_the_message(self):
        ex = _build({"prompt": [_U, {"role": "assistant", "content": "Sure,"}],
                     "chosen": [_A], "rejected": [_B]})
        assert _decode(ex.chosen_ids[: ex.chosen_prompt_len]) == "^<user>hi|<assistant>Sure,"

    def test_a_generation_prompt_the_full_render_lacks_is_narrowed_away(self):
        """TRL's common-prefix rule (the DeepSeek-R1 ``<think>`` case): text the
        generation prompt adds but ``prompt + answer`` does not render is not
        prompt, and the answer starts where the two renderings part."""

        class Thinking(CharTokenizer):
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                                    **kw):
                text = super().apply_chat_template(messages, tokenize=False, **kw)
                return text + ("<assistant><think>" if add_generation_prompt else "")

        ex = _build({"prompt": [_U], "chosen": [_A], "rejected": [_B]}, Thinking())
        assert _decode(ex.chosen_ids[: ex.chosen_prompt_len]) == "^<user>hi|<assistant>"
        assert _decode(ex.chosen_ids[ex.chosen_prompt_len:]) == "a|$"

    def test_a_merged_last_prompt_token_moves_the_answer_start(self):
        """TRL's build_tokenized_answer: the answer starts one token earlier."""
        ex = _build({"prompt": "zx", "chosen": "y1", "rejected": "q2"}, MergingTokenizer())
        assert ex.chosen_ids[ex.chosen_prompt_len] == MergingTokenizer.MERGED
        assert ex.chosen_prompt_len == ex.rejected_prompt_len - 1

    def test_prompts_differing_beyond_one_token_are_unstable(self):
        from soup_cli.trainer.mlx_dpo import PrefixUnstableError

        class Shifty(CharTokenizer):
            def encode(self, text, add_special_tokens=True):
                ids = super().encode(text, add_special_tokens)
                return [7, 7, 7] + ids if text.endswith("BAD") else ids

        with pytest.raises(PrefixUnstableError):
            _build({"prompt": "QQQ", "chosen": "ok", "rejected": "BAD"}, Shifty())

    def test_a_tie_is_kept(self):
        """ORPO trains a tie: log-odds 0, finite, and the NLL still learns."""
        ex = _build({"prompt": "Q", "chosen": "same", "rejected": "same"})
        assert ex.chosen_ids == ex.rejected_ids

    def test_long_prompt_keeps_its_end_and_never_enters_the_answer(self):
        ex = _build({"prompt": "P" * 100 + "END", "chosen": "a", "rejected": "b"},
                    max_length=40, max_prompt_length=20)
        assert ex.chosen_prompt_len == 20
        assert _decode(ex.chosen_ids[: ex.chosen_prompt_len]).endswith("END")
        assert _decode(ex.chosen_ids[ex.chosen_prompt_len:]) == "a$"

    def test_answer_cut_to_the_remaining_room(self):
        ex = _build({"prompt": "Q", "chosen": "x" * 50, "rejected": "y"},
                    max_length=12, max_prompt_length=6)
        assert len(ex.chosen_ids) == 12
        assert len(ex.rejected_ids) == ex.rejected_prompt_len + 2

    def test_no_room_for_an_answer_is_refused(self):
        from soup_cli.trainer.mlx_dpo import PreferenceRowError

        with pytest.raises(PreferenceRowError, match="no answer token fits"):
            _build({"prompt": "Q" * 30, "chosen": "a", "rejected": "b"},
                   max_length=10, max_prompt_length=10)

    def test_rows_go_through_the_loader_converter(self):
        """The bundled layout: string prompt, conversations repeating the user turn."""
        ex = _build({"prompt": "hi", "chosen": [_U, _A], "rejected": [_U, _B]})
        assert _decode(ex.chosen_ids[: ex.chosen_prompt_len]) == "^<user>hi|<assistant>"


class TestTheTrainerCapsLikeOrpoConfig:
    def test_default_truncation_keeps_the_end_of_a_long_prompt(self):
        """ORPOConfig's default is ``keep_end``; the trainer must use it."""
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.mlx_orpo import MLXORPOTrainerWrapper

        wrapper = MLXORPOTrainerWrapper(load_config_from_string(_ORPO_YAML))
        wrapper.config.data.max_length = 40
        wrapper.tokenizer = CharTokenizer()
        (example,), _ = wrapper._examples(
            [{"prompt": "S" * 60 + "TAIL", "chosen": "a", "rejected": "b"}]
        )
        assert example.chosen_prompt_len == 20
        assert _decode(example.chosen_ids[: example.chosen_prompt_len]).endswith("TAIL")


class TestTokensMatchTrl:
    """The same rows through TRL 0.29.1's ORPOTrainer.tokenize_row."""

    def _tokenizer(self, tmp_path):
        pytest.importorskip("trl")
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        words = ["<s>", "</s>", "<pad>", "<unk>", "<", ">", "user", "assistant", "|",
                 "what", "is", "two", "plus", "four", "five", "it", "Q", ":", "yes", "no"]
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

    def _trl(self, tokenizer, row, max_length):
        import types

        from trl.data_utils import maybe_apply_chat_template
        from trl.experimental.orpo.orpo_trainer import ORPOTrainer

        stub = types.SimpleNamespace(
            processing_class=tokenizer, is_encoder_decoder=False, max_length=max_length,
            truncation_mode="keep_end", label_pad_token_id=-100, padding_value=0,
        )
        stub.build_tokenized_answer = types.MethodType(ORPOTrainer.build_tokenized_answer, stub)
        return ORPOTrainer.tokenize_row(stub, maybe_apply_chat_template(row, tokenizer))

    @pytest.mark.parametrize("row", [
        {"prompt": "Q: what is two plus two", "chosen": " four", "rejected": " five"},
        {"prompt": [{"role": "user", "content": "what is two plus two"}],
         "chosen": [{"role": "assistant", "content": "it is four"}],
         "rejected": [{"role": "assistant", "content": "five"}]},
    ])
    def test_ids_and_prompt_spans_match(self, tmp_path, row):
        tokenizer = self._tokenizer(tmp_path)
        trl = self._trl(tokenizer, row, max_length=64)
        ours = _build(row, tokenizer, max_length=64, max_prompt_length=32)
        for side in ("chosen", "rejected"):
            ids = getattr(ours, f"{side}_ids")
            span = getattr(ours, f"{side}_prompt_len")
            assert ids == list(trl[f"{side}_input_ids"])
            assert span == sum(1 for x in trl[f"{side}_labels"] if x == -100)


# --------------------------------------------------------------------------
# The loss against a float64 NumPy oracle
# --------------------------------------------------------------------------


def _np_log1mexp(x):
    return np.where(x < -math.log(2), np.log1p(-np.exp(x)), np.log(-np.expm1(x)))


def _np_terms(logits, tokens, completion, sequence, *, mean=True, nll_mask="sequence"):
    shifted = logits[:, :-1] - logits[:, :-1].max(-1, keepdims=True)
    logsm = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    tgt = tokens[:, 1:]
    lp = np.take_along_axis(logsm, tgt[..., None], -1)[..., 0]
    comp = completion[:, 1:].astype(np.float64)
    seq = (sequence if nll_mask == "sequence" else completion)[:, 1:].astype(np.float64)
    scores = (lp * comp).sum(-1) / (comp.sum(-1) if mean else 1.0)
    half = tokens.shape[0] // 2
    nll = -(lp[:half] * seq[:half]).sum() / seq[:half].sum()
    return scores[:half], scores[half:], nll


def _np_loss(a, b, nll, beta, *, log1mexp=True, use_nll=True):
    lo = (a - b) - ((_np_log1mexp(a) - _np_log1mexp(b)) if log1mexp else 0.0)
    ratio = -np.logaddexp(0.0, -lo)
    return (nll if use_nll else 0.0) - beta * ratio.mean(), lo


def _fixture(seed=0, vocab=12):
    """Two pairs, unequal prompt and answer lengths, right padding."""
    from soup_cli.trainer.mlx_orpo import OrpoExample, collate_orpo

    examples = [
        OrpoExample([1, 5, 6, 7, 8, 9], 3, [1, 5, 6, 4, 2], 3),
        OrpoExample([1, 3, 8, 2], 2, [1, 3, 9, 10, 11, 2], 2),
    ]
    tokens, completion, sequence = (np.array(x) for x in collate_orpo(examples))
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, size=(tokens.shape[0], tokens.shape[1], vocab))
    return logits, tokens, completion, sequence


class _Fixed:
    def __new__(cls, logits):
        nn = pytest.importorskip("mlx.nn")
        mx = pytest.importorskip("mlx.core")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = mx.array(logits.astype(np.float32))

            def __call__(self, x):
                return self.logits[:, : x.shape[1]]

        return Model()


def _mlx(logits, tokens, completion, sequence, beta, **kw):
    mx = pytest.importorskip("mlx.core")
    from soup_cli.trainer.mlx_orpo import orpo_loss, orpo_terms

    model = _Fixed(logits)
    a, b, nll = orpo_terms(model, mx.array(tokens), mx.array(completion), mx.array(sequence), **kw)
    loss, metrics = orpo_loss(a, b, nll, beta)
    return np.array(a), np.array(b), float(nll.item()), float(loss.item()), metrics


class TestTheLossMatchesTheOracle:
    @pytest.mark.parametrize("beta", [0.1, 0.7])
    def test_terms_and_total(self, beta):
        logits, tokens, comp, seq = _fixture()
        a, b, nll, loss, metrics = _mlx(logits, tokens, comp, seq, beta)
        ra, rb, rnll = _np_terms(logits, tokens, comp, seq)
        rloss, rlo = _np_loss(ra, rb, rnll, beta)
        np.testing.assert_allclose(a, ra, rtol=1e-5)
        np.testing.assert_allclose(b, rb, rtol=1e-5)
        assert nll == pytest.approx(rnll, rel=1e-5)
        assert loss == pytest.approx(rloss, rel=1e-5)
        assert metrics["log_odds_chosen"].item() == pytest.approx(rlo.mean(), rel=1e-4)

    def test_the_chunked_head_gives_the_same_scores(self):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_orpo import token_logps

        logits, tokens, _, _ = _fixture()
        model = _Fixed(logits)
        split = (lambda x: model.logits[:, : x.shape[1]], lambda h: h)
        full = token_logps(model, mx.array(tokens))
        chunked = token_logps(model, mx.array(tokens), head_split=split, chunk_size=2)
        assert mx.allclose(full, chunked, atol=1e-6).item()

    @pytest.mark.parametrize("mutation", [
        "no_log1mexp", "summed_scores", "reversed_pair", "no_nll", "completion_mask_for_nll",
    ])
    def test_each_listed_mutation_moves_the_loss(self, mutation):
        """The fixture is sensitive to every wrong objective the issue names, so
        an implementation making any of them fails `test_terms_and_total`."""
        logits, tokens, comp, seq = _fixture()
        right, _ = _np_loss(*_np_terms(logits, tokens, comp, seq), 0.1)
        a, b, nll = _np_terms(
            logits, tokens, comp, seq, mean=mutation != "summed_scores",
            nll_mask="completion" if mutation == "completion_mask_for_nll" else "sequence",
        )
        if mutation == "reversed_pair":
            a, b = b, a
        wrong, _ = _np_loss(a, b, nll, 0.1, log1mexp=mutation != "no_log1mexp",
                            use_nll=mutation != "no_nll")
        assert abs(wrong - right) > 1e-3

    def test_prompt_logits_move_the_nll_but_not_the_scores(self):
        logits, tokens, comp, seq = _fixture()
        a0, b0, nll0, _, _ = _mlx(logits, tokens, comp, seq, 0.1)
        bumped = logits.copy()
        # Positions t whose target t+1 is a real prompt token.
        prompt_target = (seq[:, 1:] == 1) & (comp[:, 1:] == 0)
        bumped[:, :-1][prompt_target] += np.random.default_rng(3).normal(
            0.0, 3.0, size=(int(prompt_target.sum()), logits.shape[-1])
        )
        a1, b1, nll1, _, _ = _mlx(bumped, tokens, comp, seq, 0.1)
        np.testing.assert_allclose(a1, a0, rtol=1e-6)
        np.testing.assert_allclose(b1, b0, rtol=1e-6)
        assert abs(nll1 - nll0) > 1e-3

    def test_padding_does_not_score(self):
        logits, tokens, comp, seq = _fixture()
        a0, b0, nll0, _, _ = _mlx(logits, tokens, comp, seq, 0.1)
        bumped = logits.copy()
        pad_target = seq[:, 1:] == 0
        bumped[:, :-1][pad_target] = 50.0
        a1, b1, nll1, _, _ = _mlx(bumped, tokens, comp, seq, 0.1)
        np.testing.assert_allclose(a1, a0, rtol=1e-6)
        assert nll1 == pytest.approx(nll0, rel=1e-6)

    def test_gradients_match_finite_differences(self):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        from soup_cli.trainer.mlx_orpo import orpo_loss, orpo_terms

        logits, tokens, comp, seq = _fixture(seed=1)
        model = _Fixed(logits)
        t, c, s = mx.array(tokens), mx.array(comp), mx.array(seq)

        def f(m):
            return orpo_loss(*orpo_terms(m, t, c, s), 0.1)[0]

        _, grads = nn.value_and_grad(model, f)(model)
        g = np.array(grads["logits"])
        rng = np.random.default_rng(5)
        eps = 1e-4
        for _ in range(12):
            idx = (rng.integers(0, logits.shape[0]), rng.integers(0, logits.shape[1] - 1),
                   rng.integers(0, logits.shape[2]))
            hi, lo = logits.copy(), logits.copy()
            hi[idx] += eps
            lo[idx] -= eps
            fd = (_np_loss(*_np_terms(hi, tokens, comp, seq), 0.1)[0]
                  - _np_loss(*_np_terms(lo, tokens, comp, seq), 0.1)[0]) / (2 * eps)
            assert g[idx] == pytest.approx(fd, abs=2e-4, rel=2e-2), idx


class TestEdgeValues:
    def _loss(self, a, b, nll=0.5, beta=0.1):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_orpo import orpo_loss

        loss, metrics = orpo_loss(mx.array(a), mx.array(b), mx.array(nll), beta)
        return loss, metrics

    def test_identical_pair_is_finite_log2(self):
        loss, metrics = self._loss([-1.3, -0.2], [-1.3, -0.2])
        assert loss.item() == pytest.approx(0.5 + 0.1 * math.log(2), rel=1e-6)
        assert metrics["log_odds_chosen"].item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("a, b", [([-50.0], [-60.0]), ([-1e-6], [-2.0]), ([-0.69], [-0.7])])
    def test_extreme_and_near_zero_scores_are_finite_with_finite_gradients(self, a, b):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_orpo import orpo_loss

        ra, rb = np.array(a), np.array(b)
        expected, _ = _np_loss(ra, rb, 0.5, 0.1)
        loss, _ = self._loss(a, b)
        assert loss.item() == pytest.approx(expected, rel=1e-4)
        grad = mx.grad(lambda x, y: orpo_loss(x, y, mx.array(0.5), 0.1)[0], argnums=(0, 1))
        ga, gb = grad(mx.array(a), mx.array(b))
        assert np.isfinite(np.array(ga)).all() and np.isfinite(np.array(gb)).all()

    def test_beta_scales_only_the_preference_term(self):
        l1, _ = self._loss([-1.0], [-2.0], beta=0.1)
        l2, _ = self._loss([-1.0], [-2.0], beta=0.3)
        assert (l2.item() - 0.5) == pytest.approx(3 * (l1.item() - 0.5), rel=1e-5)

    def test_nan_input_is_not_a_quiet_zero(self):
        loss, _ = self._loss([float("nan")], [-1.0])
        assert math.isnan(loss.item())


# --------------------------------------------------------------------------
# Training on a tiny random Llama
# --------------------------------------------------------------------------

_ORPO_YAML = """
base: tiny-base
task: orpo
backend: mlx
data:
  train: ./x.jsonl
  format: dpo
  max_length: 128
training:
  epochs: 1
  lr: 5e-3
  orpo_beta: 0.1
  quantization: none
  lora: {r: 4, alpha: 8, dropout: 0.0}
output: ./output
"""


def _tiny_llama_args():
    from mlx_lm.models import llama

    return llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=200,
        tie_word_embeddings=True,
    )


def _rows(n=8, prefix="q"):
    return [{"prompt": f"{prefix}{i}:", "chosen": " good", "rejected": " bad"} for i in range(n)]


def _wrapper(tmp_path, monkeypatch, model_seed=0, **training):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import llama

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.mlx_orpo import MLXORPOTrainerWrapper

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    extra = "".join(f"  {k}: {v}\n" for k, v in training.items())
    text = _ORPO_YAML.replace("  orpo_beta: 0.1\n", "  orpo_beta: 0.1\n" + extra)
    cfg = load_config_from_string(text)
    wrapper = MLXORPOTrainerWrapper(cfg)
    mx.random.seed(model_seed)
    wrapper.model = llama.Model(_tiny_llama_args())
    wrapper.tokenizer = CharTokenizer()
    wrapper._dataset = {"train": _rows(), "val": _rows(2, prefix="v")}
    return wrapper


def _flat(params):
    from mlx.utils import tree_flatten

    return {k: np.array(v) for k, v in tree_flatten(params)}


class TestTraining:
    def test_lora_trains_and_the_base_stays_frozen(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, batch_size=2, gradient_accumulation_steps=1)
        before = _flat(wrapper.model.parameters())
        wrapper.train()
        after = _flat(wrapper.model.parameters())
        lora_b = [v for k, v in after.items() if k.endswith("lora_b")]
        assert lora_b and any(np.abs(v).sum() > 0 for v in lora_b)
        for name, value in before.items():
            moved = name.replace(".weight", ".linear.weight")
            key = name if name in after else moved
            np.testing.assert_array_equal(after[key], value, err_msg=name)

    def test_the_preference_is_learned_with_finite_components(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, epochs=6, batch_size=2,
                           gradient_accumulation_steps=1, warmup_ratio=0.0)
        result = wrapper.train()
        assert all(math.isfinite(x) for x in result["losses"])
        assert result["val_reward_accuracy"] == pytest.approx(1.0)
        assert result["val_log_odds_chosen"] > 0
        assert result["skipped_identical_pairs"] == 0
        meta = json.loads((tmp_path / "output" / "adapter_config.json").read_text())
        assert (meta["task"], meta["orpo_beta"], meta["fine_tune_type"]) == ("orpo", 0.1, "lora")
        assert not list((tmp_path / "output").glob("*.partial*"))

    def test_training_moves_the_held_out_log_odds_against_the_untrained_model(
        self, tmp_path, monkeypatch
    ):
        """The base model is its own control: same rows, before and after."""
        from soup_cli.trainer.mlx_orpo import orpo_loss, orpo_terms

        wrapper = _wrapper(tmp_path, monkeypatch, epochs=6, batch_size=2,
                           gradient_accumulation_steps=1, warmup_ratio=0.0)
        examples, _ = wrapper._examples(_rows(2, prefix="v"), "val")

        def held_out():
            wrapper.model.eval()
            _, metrics = orpo_loss(*orpo_terms(wrapper.model, *wrapper._batch_arrays(examples)),
                                   0.1)
            return metrics["log_odds_chosen"].item()

        before = held_out()
        wrapper.train()
        assert held_out() > before + 1.0

    def test_ties_train_instead_of_being_skipped(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, batch_size=2)
        tie = {"prompt": "t:", "chosen": " same", "rejected": " same"}
        wrapper._dataset["train"] = _rows(6) + [tie, dict(tie)]
        result = wrapper.train()
        assert result["skipped_identical_pairs"] == 0
        assert all(math.isfinite(x) for x in result["losses"])

    def test_two_micro_batches_equal_one_batch_after_an_sgd_update(self, tmp_path, monkeypatch):
        from soup_cli.trainer import mlx_dpo

        def run(path, batch_size, accum, plan):
            monkeypatch.setattr(mlx_dpo, "plan_batches", lambda *a, **k: [list(b) for b in plan])
            wrapper = _wrapper(path, monkeypatch, batch_size=batch_size,
                               gradient_accumulation_steps=accum, optimizer="sgd",
                               scheduler="constant", warmup_ratio=0.0, seed=4)
            result = wrapper.train()
            return result, _flat(wrapper.model.trainable_parameters())

        one, w1 = run(tmp_path / "a", 2, 1, [[0, 1]])
        two, w2 = run(tmp_path / "b", 1, 2, [[0], [1]])
        assert len(one["losses"]) == len(two["losses"]) == 1
        # The NLL is a token mean over the batch: two micro-batches of one pair
        # average two per-pair means, one batch of two takes one mean over both
        # pairs' tokens. With equal-length chosen sequences they coincide.
        for name in w1:
            np.testing.assert_allclose(w1[name], w2[name], atol=1e-6, err_msg=name)

    def test_val_loss_is_the_row_weighted_mean_over_chunks(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer import mlx_orpo

        wrapper = _wrapper(tmp_path, monkeypatch, batch_size=2, epochs=2)
        wrapper._dataset["val"] = _rows(2, prefix="v") + [
            {"prompt": "w:", "chosen": " bad", "rejected": " good"}
        ]
        result = wrapper.train()
        examples, _ = wrapper._examples(wrapper._dataset["val"], "val")
        wrapper.model.eval()
        per_row = []
        for example in examples:
            arrays = wrapper._batch_arrays([example])
            loss, _ = wrapper._loss(wrapper.model, arrays, "val", mx.arange(1))
            per_row.append(loss.item())
        assert result["val_loss"] == pytest.approx(sum(per_row) / 3, rel=1e-4)
        assert mlx_orpo.MLXORPOTrainerWrapper.fail_on_nonfinite

    def test_a_nonfinite_loss_stops_the_run(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer import mlx_orpo

        real = mlx_orpo.orpo_terms

        def poisoned(*args, **kwargs):
            a, b, nll = real(*args, **kwargs)
            return a * mx.array(float("nan")), b, nll

        monkeypatch.setattr(mlx_orpo, "orpo_terms", poisoned)
        wrapper = _wrapper(tmp_path, monkeypatch, batch_size=2)
        with pytest.raises(FloatingPointError, match="nonfinite loss"):
            wrapper.train()


class TestResume:
    def _first_run(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path / "a", monkeypatch, batch_size=2, epochs=3, seed=1)
        fresh = wrapper.train()
        return tmp_path / "a" / "output" / "adapters.safetensors", fresh

    def test_resume_from_its_own_adapter(self, tmp_path, monkeypatch):
        checkpoint, fresh = self._first_run(tmp_path, monkeypatch)
        resumed = _wrapper(tmp_path / "b", monkeypatch, batch_size=2, seed=1)
        result = resumed.train(resume_from_checkpoint=str(checkpoint))
        assert result["losses"][0] < fresh["losses"][0]

    @pytest.mark.parametrize("change, match", [
        ({"task": "dpo"}, "task='dpo'"),
        ({"task": "simpo"}, "MLX ORPO: .* task='simpo', not 'orpo'"),
        ({"model": "other-base"}, "belongs to base"),
        (None, "no adapter_config.json"),
    ])
    def test_mismatched_or_missing_metadata_is_refused(self, tmp_path, monkeypatch, change, match):
        checkpoint, _ = self._first_run(tmp_path, monkeypatch)
        meta = checkpoint.parent / "adapter_config.json"
        if change is None:
            meta.unlink()
        else:
            meta.write_text(json.dumps({**json.loads(meta.read_text()), **change}))
        resumed = _wrapper(tmp_path / "b", monkeypatch, batch_size=2, seed=1)
        with pytest.raises(ValueError, match=match):
            resumed.train(resume_from_checkpoint=str(checkpoint))


class TestFreshProcessReload:
    def test_scores_reproduce_and_the_adapter_has_an_effect(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_orpo import orpo_terms

        wrapper = _wrapper(tmp_path, monkeypatch, batch_size=2, epochs=3)
        base_path = tmp_path / "base.safetensors"
        wrapper.model.save_weights(str(base_path))
        wrapper.train()
        examples, _ = wrapper._examples(_rows(2, prefix="v"), "val")
        tokens, completion, sequence = (np.array(x) for x in wrapper._batch_arrays(examples))
        wrapper.model.eval()
        a, b, nll = orpo_terms(wrapper.model, *wrapper._batch_arrays(examples))
        mx.eval(a, b, nll)
        np.save(tmp_path / "batch.npy", np.stack([tokens, completion, sequence]))

        script = textwrap.dedent(f"""
            import json, sys
            import numpy as np
            import mlx.core as mx
            from mlx_lm.models import llama
            from mlx_lm.tuner.utils import load_adapters
            sys.path.insert(0, {str(tmp_path)!r})
            from soup_cli.trainer.mlx_orpo import orpo_terms
            args = llama.ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=2,
                intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                rms_norm_eps=1e-5, vocab_size=200, tie_word_embeddings=True)
            model = llama.Model(args)
            model.load_weights({str(base_path)!r})
            if sys.argv[1] == "adapter":
                load_adapters(model, {str(tmp_path / "output")!r})
            model.eval()
            t, c, s = (mx.array(x) for x in np.load({str(tmp_path / "batch.npy")!r}))
            a, b, nll = orpo_terms(model, t, c, s)
            print(json.dumps({{"a": a.tolist(), "b": b.tolist(), "nll": nll.item()}}))
        """)
        def run(mode):
            out = subprocess.run([sys.executable, "-c", script, mode], capture_output=True,
                                 text=True, check=True, timeout=300)
            return json.loads(out.stdout.strip().splitlines()[-1])

        reloaded, base = run("adapter"), run("base")
        np.testing.assert_allclose(reloaded["a"], np.array(a), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(reloaded["b"], np.array(b), rtol=1e-5, atol=1e-6)
        assert reloaded["nll"] == pytest.approx(nll.item(), rel=1e-5)
        assert abs(np.array(base["a"]) - np.array(reloaded["a"])).max() > 1e-3


# --------------------------------------------------------------------------
# Config and dispatch
# --------------------------------------------------------------------------


class TestConfig:
    def _load(self, top="", training=""):
        """`training` lines replace the fixture's own line for the same key."""
        from soup_cli.config.loader import load_config_from_string

        text = _ORPO_YAML.replace("backend: mlx\n", "backend: mlx\n" + top)
        for line in training.splitlines(keepends=True):
            key = line.split(":", 1)[0]
            own = [x for x in text.splitlines(keepends=True) if x.split(":", 1)[0] == key]
            text = text.replace(own[0], line) if own else text.replace(
                "  orpo_beta: 0.1\n", "  orpo_beta: 0.1\n" + line
            )
        return load_config_from_string(text)

    def test_orpo_on_mlx_loads_and_dispatches(self):
        from soup_cli.trainer.mlx_orpo import MLXORPOTrainerWrapper
        from soup_cli.trainer.mlx_routing import get_mlx_trainer

        cfg = self._load()
        assert (cfg.task, cfg.backend, cfg.training.orpo_beta) == ("orpo", "mlx", 0.1)
        assert get_mlx_trainer("orpo") is MLXORPOTrainerWrapper

    @pytest.mark.parametrize("top, training, match", [
        ("", "  quantization: 8bit\n", "quantization='8bit'"),
        ("modality: vision\n", "", "modality='vision'"),
        ("", "  lora: {r: 0}\n", "lora.r must be at least 1"),
        ("", "  loraplus_lr_ratio: 16\n", "loraplus_lr_ratio"),
    ])
    def test_unsupported_combinations_are_refused_at_load(self, top, training, match):
        with pytest.raises(Exception, match=match):
            self._load(top, training)

    def test_the_preference_alias_is_refused_explicitly(self):
        from soup_cli.config.loader import load_config_from_string

        alias = (_ORPO_YAML.replace("task: orpo", "task: preference")
                 .replace("  orpo_beta: 0.1\n", "  preference_loss: orpo\n"))
        with pytest.raises(Exception, match="task='preference' has no trainer") as info:
            load_config_from_string(alias)
        assert "Use task: orpo for the native MLX trainer." in str(info.value)

    def test_the_transformers_route_is_unchanged(self):
        from soup_cli.config.loader import load_config_from_string

        text = _ORPO_YAML.replace("backend: mlx", "backend: transformers")
        cfg = load_config_from_string(text.replace("  orpo_beta: 0.1\n", "  orpo_beta: 0.25\n"))
        assert (cfg.backend, cfg.training.orpo_beta) == ("transformers", 0.25)

    def test_doctor_reports_ignored_settings(self):
        from soup_cli.config.backend_support import check_config

        cfg = self._load(training="  use_galore: true\n")
        assert {e.field for e in check_config(cfg)} == {"training.use_galore"}

    def test_the_ignored_settings_are_printed(self, monkeypatch):
        from io import StringIO

        from rich.console import Console

        from soup_cli.trainer import mlx_dpo
        from soup_cli.trainer.mlx_orpo import MLXORPOTrainerWrapper

        buffer = StringIO()
        monkeypatch.setattr(mlx_dpo, "console", Console(file=buffer, width=400))
        MLXORPOTrainerWrapper(self._load(training="  use_galore: true\n"))._check_unsupported()
        assert "MLX backend ignores: GaLore" in _strip_ansi(buffer.getvalue())
