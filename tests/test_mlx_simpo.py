"""Native SimPO on MLX (#1999).

The objective is pinned to TRL 0.29.1's ``experimental/cpo/cpo_trainer.py`` with
``loss_type="simpo"``, which Soup's transformers backend runs for ``task: simpo``
(``trainer/simpo.py`` forwards ``simpo_gamma`` and ``cpo_alpha`` and no beta, so
``CPOConfig.beta``'s default applies). Layers:

- tokenization against TRL's own ``CPOTrainer.tokenize_row`` on a real
  ``PreTrainedTokenizerFast`` (needs ``trl`` / ``transformers``, no MLX);
- an independent float64 NumPy oracle, shown (without MLX) to be moved by every
  wrong objective the issue lists; the MLX loss, its terms and its gradients
  against that oracle and against TRL's own ``get_batch_loss_metrics`` on the
  same logits (MLX; TRL parts need torch);
- training, saving and a fresh-process reload on a tiny random Llama (needs
  MLX and mlx-lm). No downloads.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.test_mlx_dpo import _strip_ansi
from tests.test_mlx_orpo import (
    CharTokenizer,
    MergingTokenizer,
    _decode,
    _Fixed,
    _flat,
    _rows,
    _tiny_llama_args,
)

BETA = 0.1  # CPOConfig.beta's default in TRL 0.29.1, pinned below.


def _build(row, tokenizer=None, max_length=256, max_prompt_length=128):
    from soup_cli.trainer.mlx_orpo import build_orpo_example

    return build_orpo_example(
        row, tokenizer or CharTokenizer(), max_length=max_length,
        max_prompt_length=max_prompt_length,
    )


# --------------------------------------------------------------------------
# TRL facts this objective rests on (no MLX)
# --------------------------------------------------------------------------


class TestTrlFacts:
    def test_the_effective_beta_is_cpoconfig_default(self):
        pytest.importorskip("trl")
        from trl.experimental.cpo import CPOConfig

        from soup_cli.trainer.mlx_simpo import SIMPO_BETA

        fields = CPOConfig.__dataclass_fields__
        assert fields["beta"].default == SIMPO_BETA == BETA
        assert fields["label_smoothing"].default == 0.0
        assert fields["alpha"].default == 0.0  # AlphaPO's reward shaping is off

    def test_the_transformers_simpo_path_passes_no_beta(self):
        """Soup's transformers SimPO builds CPOConfig without beta or label
        smoothing, so the default above is what it trains with."""
        import soup_cli.trainer.simpo as simpo

        tree = ast.parse(Path(simpo.__file__).read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "cpo_config_cls"
        ]
        assert len(calls) == 1
        names = {kw.arg for kw in calls[0].keywords}
        assert {"loss_type", "simpo_gamma", "cpo_alpha"} <= names
        assert not names & {"beta", "label_smoothing", "alpha"}


# --------------------------------------------------------------------------
# Tokenization: ORPO's, which is TRL's CPO tokenize_row too
# --------------------------------------------------------------------------


class TestTokenization:
    def test_text_row_gets_bos_and_eos(self):
        ex = _build({"prompt": "Q:", "chosen": " yes", "rejected": " no"})
        assert _decode(ex.chosen_ids) == "^Q: yes$"
        assert ex.chosen_prompt_len == 3

    def test_a_merged_last_prompt_token_moves_the_answer_start(self):
        ex = _build({"prompt": "zx", "chosen": "y1", "rejected": "q2"}, MergingTokenizer())
        assert ex.chosen_ids[ex.chosen_prompt_len] == MergingTokenizer.MERGED

    def test_the_trainer_caps_the_prompt_keeping_its_end(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.mlx_simpo import MLXSimPOTrainerWrapper

        wrapper = MLXSimPOTrainerWrapper(load_config_from_string(_SIMPO_YAML))
        wrapper.config.data.max_length = 40
        wrapper.tokenizer = CharTokenizer()
        (example,), ties = wrapper._examples(
            [{"prompt": "S" * 60 + "TAIL", "chosen": "a", "rejected": "b"}]
        )
        assert ties == 0
        assert example.chosen_prompt_len == 20
        assert _decode(example.chosen_ids[: example.chosen_prompt_len]).endswith("TAIL")

    def test_no_room_for_an_answer_is_refused(self):
        from soup_cli.trainer.mlx_dpo import PreferenceRowError

        with pytest.raises(PreferenceRowError, match="no answer token fits"):
            _build({"prompt": "Q" * 30, "chosen": "a", "rejected": "b"},
                   max_length=10, max_prompt_length=10)

    def test_a_tie_is_kept(self):
        """TRL trains a tie: the preference term is the constant softplus(gamma)."""
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.mlx_simpo import MLXSimPOTrainerWrapper

        wrapper = MLXSimPOTrainerWrapper(load_config_from_string(_SIMPO_YAML))
        wrapper.tokenizer = CharTokenizer()
        examples, ties = wrapper._examples([{"prompt": "Q", "chosen": "s", "rejected": "s"}])
        assert (len(examples), ties) == (1, 0)


class TestTokensMatchTrl:
    """The same rows through TRL 0.29.1's CPOTrainer.tokenize_row."""

    def _trl(self, tokenizer, row, max_length):
        import types

        from trl.data_utils import maybe_apply_chat_template
        from trl.experimental.cpo.cpo_trainer import CPOTrainer

        stub = types.SimpleNamespace(
            processing_class=tokenizer, is_encoder_decoder=False, max_length=max_length,
        )
        stub.build_tokenized_answer = types.MethodType(CPOTrainer.build_tokenized_answer, stub)
        return CPOTrainer.tokenize_row(stub, maybe_apply_chat_template(row, tokenizer))

    @pytest.mark.parametrize("row", [
        {"prompt": "Q: what is two plus two", "chosen": " four", "rejected": " five"},
        {"prompt": [{"role": "user", "content": "what is two plus two"}],
         "chosen": [{"role": "assistant", "content": "it is four"}],
         "rejected": [{"role": "assistant", "content": "five"}]},
    ])
    def test_ids_and_label_spans_match(self, tmp_path, row):
        from tests.test_mlx_orpo import TestTokensMatchTrl as Orpo

        tokenizer = Orpo()._tokenizer(tmp_path)
        trl = self._trl(tokenizer, row, max_length=64)
        ours = _build(row, tokenizer, max_length=64, max_prompt_length=32)
        for side in ("chosen", "rejected"):
            ids = getattr(ours, f"{side}_ids")
            span = getattr(ours, f"{side}_prompt_len")
            assert ids == list(trl[f"{side}_input_ids"])
            # CPO's labels mask the prompt: what is scored and what the NLL
            # covers is exactly the completion.
            assert span == sum(1 for x in trl[f"{side}_labels"] if x == -100)
            assert list(trl[f"{side}_labels"][span:]) == ids[span:]


# --------------------------------------------------------------------------
# The objective against a float64 NumPy oracle
# --------------------------------------------------------------------------


def _np_terms(logits, tokens, completion, sequence, *, mean=True, nll="token_mean",
              nll_mask="completion"):
    shifted = logits[:, :-1] - logits[:, :-1].max(-1, keepdims=True)
    logsm = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    lp = np.take_along_axis(logsm, tokens[:, 1:][..., None], -1)[..., 0]
    comp = completion[:, 1:].astype(np.float64)
    scores = (lp * comp).sum(-1) / (comp.sum(-1) if mean else 1.0)
    half = tokens.shape[0] // 2
    m = (completion if nll_mask == "completion" else sequence)[:half, 1:].astype(np.float64)
    if nll == "token_mean":
        value = -(lp[:half] * m).sum() / m.sum()
    else:  # mean of per-row means
        value = (-(lp[:half] * m).sum(-1) / m.sum(-1)).mean()
    return scores[:half], scores[half:], value


def _np_loss(a, b, nll, gamma, alpha, *, beta=BETA, margin="outside", use_nll=True):
    z = {
        "outside": beta * (a - b) - gamma,
        "inside": beta * ((a - b) - gamma),
        "none": beta * (a - b),
    }[margin]
    pref = np.logaddexp(0.0, -z)  # -logsigmoid(z)
    return pref.mean() + (alpha * nll if use_nll else 0.0)


def _fixture(seed=0, vocab=12):
    """Two pairs, unequal prompt and completion lengths, right padding."""
    from soup_cli.trainer.mlx_orpo import OrpoExample, collate_orpo

    examples = [
        OrpoExample([1, 5, 6, 7, 8, 9], 3, [1, 5, 6, 4, 2], 3),
        OrpoExample([1, 3, 8, 2], 2, [1, 3, 9, 10, 11, 2], 2),
    ]
    tokens, completion, sequence = (np.array(x) for x in collate_orpo(examples))
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, size=(tokens.shape[0], tokens.shape[1], vocab))
    return logits, tokens, completion, sequence


def _oracle(logits, tokens, comp, seq, gamma, alpha):
    return _np_loss(*_np_terms(logits, tokens, comp, seq), gamma, alpha)


#: (gamma, cpo_alpha), varied independently.
_SETTINGS = [(0.5, 1.0), (1.3, 1.0), (0.5, 0.25), (0.0, 2.0)]


class TestTheOracleSeesEveryListedMutation:
    """No MLX: the fixture is sensitive to every wrong objective the issue names,
    so an implementation making any of them fails ``test_terms_and_total``."""

    @pytest.mark.parametrize("mutation", [
        "summed_scores", "reversed_pair", "no_margin", "margin_inside", "no_nll",
        "nll_row_means", "nll_over_sequence",
    ])
    def test_each_mutation_moves_the_loss(self, mutation):
        logits, tokens, comp, seq = _fixture()
        gamma, alpha = 0.5, 1.0
        right = _oracle(logits, tokens, comp, seq, gamma, alpha)
        a, b, nll = _np_terms(
            logits, tokens, comp, seq, mean=mutation != "summed_scores",
            nll="row_means" if mutation == "nll_row_means" else "token_mean",
            nll_mask="sequence" if mutation == "nll_over_sequence" else "completion",
        )
        if mutation == "reversed_pair":
            a, b = b, a
        margin = {"no_margin": "none", "margin_inside": "inside"}.get(mutation, "outside")
        wrong = _np_loss(a, b, nll, gamma, alpha, margin=margin, use_nll=mutation != "no_nll")
        assert abs(wrong - right) > 1e-3

    def test_gamma_and_alpha_move_the_loss_independently(self):
        logits, tokens, comp, seq = _fixture()
        values = {s: _oracle(logits, tokens, comp, seq, *s) for s in _SETTINGS}
        assert len({round(v, 6) for v in values.values()}) == len(_SETTINGS)


def _mlx(logits, tokens, completion, gamma, alpha, **kw):
    mx = pytest.importorskip("mlx.core")
    from soup_cli.trainer.mlx_simpo import simpo_loss, simpo_terms

    model = _Fixed(logits)
    a, b, nll = simpo_terms(model, mx.array(tokens), mx.array(completion), **kw)
    loss, metrics = simpo_loss(a, b, nll, beta=BETA, gamma=gamma, cpo_alpha=alpha)
    return np.array(a), np.array(b), float(nll.item()), float(loss.item()), metrics


class TestTheLossMatchesTheOracle:
    @pytest.mark.parametrize("gamma, alpha", _SETTINGS)
    def test_terms_and_total(self, gamma, alpha):
        logits, tokens, comp, seq = _fixture()
        a, b, nll, loss, metrics = _mlx(logits, tokens, comp, gamma, alpha)
        ra, rb, rnll = _np_terms(logits, tokens, comp, seq)
        np.testing.assert_allclose(a, ra, rtol=1e-5)
        np.testing.assert_allclose(b, rb, rtol=1e-5)
        assert nll == pytest.approx(rnll, rel=1e-5)
        assert loss == pytest.approx(_np_loss(ra, rb, rnll, gamma, alpha), rel=1e-5)
        assert metrics["nll_loss"].item() == pytest.approx(rnll, rel=1e-5)
        # The preference term, logged on its own: the total minus cpo_alpha * NLL.
        assert metrics["preference_loss"].item() == pytest.approx(
            _np_loss(ra, rb, rnll, gamma, alpha) - alpha * rnll, rel=1e-5
        )
        assert metrics["reward_chosen"].item() == pytest.approx(BETA * ra.mean(), rel=1e-5)
        assert metrics["reward_rejected"].item() == pytest.approx(BETA * rb.mean(), rel=1e-5)
        assert metrics["reward_margin"].item() == pytest.approx(
            BETA * (ra - rb).mean(), rel=1e-5
        )
        assert metrics["reward_accuracy"].item() == pytest.approx((ra > rb).mean())
        assert metrics["logps_chosen"].item() == pytest.approx(ra.mean(), rel=1e-5)
        assert metrics["logps_rejected"].item() == pytest.approx(rb.mean(), rel=1e-5)

    def test_the_chunked_head_gives_the_same_terms(self):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_simpo import simpo_terms

        logits, tokens, comp, _ = _fixture()
        model = _Fixed(logits)
        split = (lambda x: model.logits[:, : x.shape[1]], lambda h: h)
        full = simpo_terms(model, mx.array(tokens), mx.array(comp))
        chunked = simpo_terms(model, mx.array(tokens), mx.array(comp), head_split=split)
        for x, y in zip(full, chunked):
            assert mx.allclose(x, y, atol=1e-6).item()

    def test_prompt_and_padding_logits_move_nothing(self):
        """CPO masks the prompt out of the NLL as well as the scores."""
        logits, tokens, comp, seq = _fixture()
        a0, b0, nll0, loss0, _ = _mlx(logits, tokens, comp, 0.5, 1.0)
        bumped = logits.copy()
        not_scored = comp[:, 1:] == 0  # prompt targets and padding
        bumped[:, :-1][not_scored] += np.random.default_rng(3).normal(
            0.0, 3.0, size=(int(not_scored.sum()), logits.shape[-1])
        )
        a1, b1, nll1, loss1, _ = _mlx(bumped, tokens, comp, 0.5, 1.0)
        np.testing.assert_allclose(a1, a0, rtol=1e-6)
        np.testing.assert_allclose(b1, b0, rtol=1e-6)
        assert nll1 == pytest.approx(nll0, rel=1e-6)
        assert loss1 == pytest.approx(loss0, rel=1e-6)

    @pytest.mark.parametrize("gamma, alpha", [(0.5, 1.0), (1.3, 0.25)])
    def test_gradients_match_finite_differences(self, gamma, alpha):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        from soup_cli.trainer.mlx_simpo import simpo_loss, simpo_terms

        logits, tokens, comp, seq = _fixture(seed=1)
        model = _Fixed(logits)
        t, c = mx.array(tokens), mx.array(comp)

        def f(m):
            return simpo_loss(*simpo_terms(m, t, c), beta=BETA, gamma=gamma,
                              cpo_alpha=alpha)[0]

        _, grads = nn.value_and_grad(model, f)(model)
        g = np.array(grads["logits"])
        rng = np.random.default_rng(5)
        eps = 1e-4
        scored = np.argwhere(comp[:, 1:] == 1)
        for i in range(12):
            if i % 2:  # half the probes on positions the loss reads
                r, p = scored[rng.integers(0, len(scored))]
                idx = (r, p, rng.integers(0, logits.shape[2]))
            else:
                idx = (rng.integers(0, logits.shape[0]), rng.integers(0, logits.shape[1] - 1),
                       rng.integers(0, logits.shape[2]))
            hi, lo = logits.copy(), logits.copy()
            hi[idx] += eps
            lo[idx] -= eps
            fd = (_oracle(hi, tokens, comp, seq, gamma, alpha)
                  - _oracle(lo, tokens, comp, seq, gamma, alpha)) / (2 * eps)
            assert g[idx] == pytest.approx(fd, abs=2e-4, rel=2e-2), idx


class TestTheLossMatchesTrlItself:
    """TRL 0.29.1's own ``CPOTrainer.get_batch_loss_metrics`` on the same logits."""

    def _trl(self, logits, tokens, comp, seq, gamma, alpha):
        import types

        torch = pytest.importorskip("torch")
        pytest.importorskip("trl")
        from trl.experimental.cpo.cpo_trainer import CPOTrainer

        half = tokens.shape[0] // 2
        labels = np.where(comp == 1, tokens, -100)
        batch = {}
        for side, rows in (("chosen", slice(0, half)), ("rejected", slice(half, None))):
            batch[f"{side}_input_ids"] = torch.tensor(tokens[rows])
            batch[f"{side}_attention_mask"] = torch.tensor(seq[rows])
            batch[f"{side}_labels"] = torch.tensor(labels[rows])
        leaf = torch.tensor(logits, dtype=torch.float64, requires_grad=True)

        def model(input_ids, **_):
            return types.SimpleNamespace(logits=leaf[:, : input_ids.shape[1]])

        stub = types.SimpleNamespace(
            beta=BETA, label_smoothing=0.0, loss_type="simpo", simpo_gamma=gamma,
            cpo_alpha=alpha, alpha=0.0, is_encoder_decoder=False, pad_token_id=0,
            aux_loss_enabled=False,
            accelerator=types.SimpleNamespace(device="cpu", gather_for_metrics=lambda x: x),
            concatenated_inputs=CPOTrainer.concatenated_inputs,
            get_batch_logps=CPOTrainer.get_batch_logps,
        )
        for name in ("concatenated_forward", "cpo_loss", "get_batch_loss_metrics"):
            setattr(stub, name, types.MethodType(getattr(CPOTrainer, name), stub))
        loss, metrics = stub.get_batch_loss_metrics(model, batch)
        loss.backward()
        return loss.item(), metrics, leaf.grad.numpy()

    @pytest.mark.parametrize("gamma, alpha", _SETTINGS)
    def test_loss_metrics_and_gradient(self, gamma, alpha):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        from soup_cli.trainer.mlx_simpo import simpo_loss, simpo_terms

        logits, tokens, comp, seq = _fixture(seed=2)
        trl_loss, trl_metrics, trl_grad = self._trl(logits, tokens, comp, seq, gamma, alpha)
        model = _Fixed(logits)
        t, c = mx.array(tokens), mx.array(comp)

        def f(m):
            return simpo_loss(*simpo_terms(m, t, c), beta=BETA, gamma=gamma, cpo_alpha=alpha)

        (loss, metrics), grads = nn.value_and_grad(model, f)(model)
        assert loss.item() == pytest.approx(trl_loss, rel=1e-5)
        for ours, theirs in (
            ("nll_loss", "nll_loss"), ("reward_chosen", "rewards/chosen"),
            ("reward_rejected", "rewards/rejected"), ("reward_accuracy", "rewards/accuracies"),
            ("reward_margin", "rewards/margins"), ("logps_chosen", "logps/chosen"),
            ("logps_rejected", "logps/rejected"),
        ):
            assert metrics[ours].item() == pytest.approx(trl_metrics[theirs], rel=1e-5,
                                                         abs=1e-7), ours
        np.testing.assert_allclose(np.array(grads["logits"]), trl_grad, atol=2e-6)


class TestTheOracleMatchesTrl:
    """No MLX: the NumPy oracle itself against TRL's ``get_batch_loss_metrics``,
    so a CI runner without MLX still checks what the MLX tests compare to."""

    @pytest.mark.parametrize("gamma, alpha", _SETTINGS)
    def test_loss_terms_and_gradient(self, gamma, alpha):
        logits, tokens, comp, seq = _fixture(seed=2)
        trl_loss, trl_metrics, trl_grad = TestTheLossMatchesTrlItself()._trl(
            logits, tokens, comp, seq, gamma, alpha
        )
        a, b, nll = _np_terms(logits, tokens, comp, seq)
        assert _np_loss(a, b, nll, gamma, alpha) == pytest.approx(trl_loss, rel=1e-9)
        assert nll == pytest.approx(trl_metrics["nll_loss"], rel=1e-9)
        assert a.mean() == pytest.approx(trl_metrics["logps/chosen"], rel=1e-9)
        assert b.mean() == pytest.approx(trl_metrics["logps/rejected"], rel=1e-9)
        assert BETA * (a - b).mean() == pytest.approx(trl_metrics["rewards/margins"], rel=1e-9)
        rng = np.random.default_rng(7)
        eps = 1e-5
        scored = np.argwhere(comp[:, 1:] == 1)
        assert np.abs(trl_grad).max() > 1e-3
        for _ in range(8):  # positions the loss reads
            r, p = scored[rng.integers(0, len(scored))]
            idx = (r, p, rng.integers(0, logits.shape[2]))
            hi, lo = logits.copy(), logits.copy()
            hi[idx] += eps
            lo[idx] -= eps
            fd = (_oracle(hi, tokens, comp, seq, gamma, alpha)
                  - _oracle(lo, tokens, comp, seq, gamma, alpha)) / (2 * eps)
            assert fd == pytest.approx(trl_grad[idx], abs=1e-7, rel=1e-5), idx


class TestEdgeValues:
    def _loss(self, a, b, nll=0.5, gamma=0.5, alpha=1.0):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_simpo import simpo_loss

        return simpo_loss(mx.array(a), mx.array(b), mx.array(nll), beta=BETA, gamma=gamma,
                          cpo_alpha=alpha)

    def test_a_tie_is_softplus_gamma_plus_the_nll(self):
        loss, metrics = self._loss([-1.3, -0.2], [-1.3, -0.2], gamma=0.7, alpha=0.5)
        assert loss.item() == pytest.approx(math.log1p(math.exp(0.7)) + 0.25, rel=1e-6)
        assert metrics["reward_accuracy"].item() == 0.0  # strict, as TRL's `>`

    @pytest.mark.parametrize("a, b", [([-50.0], [-0.01]), ([-0.01], [-500.0])])
    def test_extreme_scores_are_finite(self, a, b):
        loss, _ = self._loss(a, b)
        expected = _np_loss(np.array(a), np.array(b), 0.5, 0.5, 1.0)
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_nan_input_is_not_a_quiet_zero(self):
        loss, _ = self._loss([float("nan")], [-1.0])
        assert math.isnan(loss.item())


# --------------------------------------------------------------------------
# Training on a tiny random Llama
# --------------------------------------------------------------------------

_SIMPO_YAML = """
base: tiny-base
task: simpo
backend: mlx
data:
  train: ./x.jsonl
  format: dpo
  max_length: 128
training:
  epochs: 1
  lr: 5e-3
  simpo_gamma: 0.5
  quantization: none
  lora: {r: 4, alpha: 8, dropout: 0.0}
output: ./output
"""


def _wrapper(tmp_path, monkeypatch, model_seed=0, **training):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import llama

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.mlx_simpo import MLXSimPOTrainerWrapper

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    extra = "".join(f"  {k}: {v}\n" for k, v in training.items())
    text = _SIMPO_YAML.replace("  simpo_gamma: 0.5\n", "  simpo_gamma: 0.5\n" + extra)
    wrapper = MLXSimPOTrainerWrapper(load_config_from_string(text))
    mx.random.seed(model_seed)
    wrapper.model = llama.Model(_tiny_llama_args())
    wrapper.tokenizer = CharTokenizer()
    wrapper._dataset = {"train": _rows(), "val": _rows(2, prefix="v")}
    return wrapper


class TestTraining:
    def test_lora_trains_and_the_base_stays_frozen(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, batch_size=2)
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

    def test_the_preference_is_learned_and_the_metadata_written(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, epochs=6, batch_size=2, warmup_ratio=0.0,
                           cpo_alpha=0.5)
        result = wrapper.train()
        assert all(math.isfinite(x) for x in result["losses"])
        assert result["val_reward_accuracy"] == pytest.approx(1.0)
        assert result["val_reward_margin"] > 0
        assert math.isfinite(result["val_nll_loss"])
        assert math.isfinite(result["val_preference_loss"])
        assert result["skipped_identical_pairs"] == 0
        meta = json.loads((tmp_path / "output" / "adapter_config.json").read_text())
        assert meta["task"] == "simpo" and meta["fine_tune_type"] == "lora"
        assert (meta["simpo_beta"], meta["simpo_gamma"], meta["cpo_alpha"]) == (0.1, 0.5, 0.5)
        assert not list((tmp_path / "output").glob("*.partial*"))

    def test_training_moves_the_held_out_margin_against_the_untrained_model(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.trainer.mlx_simpo import simpo_loss, simpo_terms

        wrapper = _wrapper(tmp_path, monkeypatch, epochs=6, batch_size=2, warmup_ratio=0.0,
                           gradient_accumulation_steps=1)
        examples, _ = wrapper._examples(_rows(2, prefix="v"), "val")

        def held_out():
            wrapper.model.eval()
            _, metrics = simpo_loss(*simpo_terms(wrapper.model, *wrapper._batch_arrays(examples)),
                                    beta=BETA, gamma=0.5, cpo_alpha=1.0)
            return metrics["reward_margin"].item()

        before = held_out()
        wrapper.train()
        assert held_out() > before + 0.1

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
        assert one["losses"][0] == pytest.approx(two["losses"][0], rel=1e-5)
        # Equal-length chosen completions, so the batch's token-mean NLL and
        # the mean of two single-pair NLLs coincide.
        for name in w1:
            np.testing.assert_allclose(w1[name], w2[name], atol=1e-6, err_msg=name)

    def test_a_trailing_partial_batch_is_dropped_and_val_chunks_weigh_by_rows(
        self, tmp_path, monkeypatch
    ):
        mx = pytest.importorskip("mlx.core")

        wrapper = _wrapper(tmp_path, monkeypatch, batch_size=2, gradient_accumulation_steps=1)
        wrapper._dataset["train"] = _rows(7)
        wrapper._dataset["val"] = _rows(2, prefix="v") + [
            {"prompt": "w:", "chosen": " bad", "rejected": " good"}
        ]
        result = wrapper.train()
        assert result["total_steps"] == 3  # 7 rows, batch 2: the 7th never trains
        examples, _ = wrapper._examples(wrapper._dataset["val"], "val")
        wrapper.model.eval()
        per_row = [
            wrapper._loss(wrapper.model, wrapper._batch_arrays([e]), "val", mx.arange(1))[0].item()
            for e in examples
        ]
        assert result["val_loss"] == pytest.approx(sum(per_row) / 3, rel=1e-4)

    def test_a_nonfinite_loss_stops_the_run(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer import mlx_simpo

        real = mlx_simpo.simpo_terms

        def poisoned(*args, **kwargs):
            a, b, nll = real(*args, **kwargs)
            return a * mx.array(float("nan")), b, nll

        monkeypatch.setattr(mlx_simpo, "simpo_terms", poisoned)
        wrapper = _wrapper(tmp_path, monkeypatch, batch_size=2)
        with pytest.raises(FloatingPointError, match="MLX SimPO: nonfinite loss"):
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
        ({"task": "orpo"}, "MLX SimPO: .* task='orpo', not 'simpo'"),
        ({"task": "dpo"}, "task='dpo'"),
        ({"model": "other-base"}, "belongs to base"),
        (None, "MLX SimPO: no adapter_config.json"),
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
    def test_logits_reproduce_and_the_adapter_has_an_effect(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")

        wrapper = _wrapper(tmp_path, monkeypatch, batch_size=2, epochs=3)
        base_path = tmp_path / "base.safetensors"
        wrapper.model.save_weights(str(base_path))
        wrapper.train()
        examples, _ = wrapper._examples(_rows(2, prefix="v"), "val")
        tokens, _ = wrapper._batch_arrays(examples)
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

        text = _SIMPO_YAML.replace("backend: mlx\n", "backend: mlx\n" + top)
        for line in training.splitlines(keepends=True):
            key = line.split(":", 1)[0]
            own = [x for x in text.splitlines(keepends=True) if x.split(":", 1)[0] == key]
            text = text.replace(own[0], line) if own else text.replace(
                "  simpo_gamma: 0.5\n", "  simpo_gamma: 0.5\n" + line
            )
        return load_config_from_string(text)

    def test_the_configured_gamma_and_alpha_reach_the_loss(self, monkeypatch):
        """``simpo_gamma`` and ``cpo_alpha`` are the task's user-facing settings;
        beta is CPOConfig's default (review of #2097)."""
        from soup_cli.trainer import mlx_simpo

        cfg = self._load(training="  cpo_alpha: 0.3\n  simpo_gamma: 1.3\n")
        wrapper = mlx_simpo.MLXSimPOTrainerWrapper(cfg)
        wrapper._head_split = None
        seen = {}
        monkeypatch.setattr(mlx_simpo, "simpo_terms", lambda *a, **k: (1.0, 2.0, 3.0))
        monkeypatch.setattr(mlx_simpo, "simpo_loss",
                            lambda *a, **k: (seen.update(k), (0.0, {}))[1])
        wrapper._loss(object(), (None, None), "train", None)
        assert seen == {"beta": 0.1, "gamma": 1.3, "cpo_alpha": 0.3}

    def test_simpo_on_mlx_loads_and_is_an_mlx_task(self):
        from soup_cli.config.schema import MLX_TASKS

        cfg = self._load(training="  cpo_alpha: 0.3\n")
        assert (cfg.task, cfg.backend, cfg.training.simpo_gamma) == ("simpo", "mlx", 0.5)
        assert cfg.training.cpo_alpha == 0.3
        assert "simpo" in MLX_TASKS

    def test_simpo_dispatches_to_the_native_trainer(self):
        from soup_cli.trainer.mlx_routing import MLX_TRAINER_REGISTRY, get_mlx_trainer
        from soup_cli.trainer.mlx_simpo import MLXSimPOTrainerWrapper

        assert "simpo" in MLX_TRAINER_REGISTRY and "simpo" in MLX_TRAINER_REGISTRY.keys()
        assert get_mlx_trainer("simpo") is MLXSimPOTrainerWrapper

    @pytest.mark.parametrize("top, training, match", [
        ("", "  quantization: 8bit\n", "quantization='8bit' is not supported by task='simpo'"),
        ("modality: vision\n", "", "modality='vision' is not supported by task='simpo'"),
        ("", "  lora: {r: 0}\n", "lora.r must be at least 1 for task='simpo'"),
        ("", "  loraplus_lr_ratio: 16\n",
         "training.loraplus_lr_ratio is not implemented on backend='mlx'"),
        ("", "  cpo_alpha: 0\n", "cpo_alpha: Input should be greater than 0"),
    ])
    def test_unsupported_combinations_are_refused_at_load(self, top, training, match):
        with pytest.raises(Exception, match=match):
            self._load(top, training)

    def test_the_preference_alias_is_refused_naming_task_simpo(self):
        from soup_cli.config.loader import load_config_from_string

        alias = (_SIMPO_YAML.replace("task: simpo", "task: preference")
                 .replace("  simpo_gamma: 0.5\n", "  preference_loss: simpo\n"))
        with pytest.raises(Exception, match="task='preference' has no trainer") as info:
            load_config_from_string(alias)
        assert "Use task: simpo for the native MLX trainer." in str(info.value)

    def test_other_tasks_name_simpo_among_the_mlx_ones(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(Exception, match="only ships SFT, DPO, ORPO, SimPO, KTO and GRPO"):
            load_config_from_string(_SIMPO_YAML.replace("task: simpo", "task: ipo"))

    def test_the_transformers_route_is_unchanged(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_SIMPO_YAML.replace("backend: mlx", "backend: transformers"))
        assert cfg.backend == "transformers" and cfg.task == "simpo"

    def test_doctor_reports_ignored_settings(self):
        from soup_cli.config.backend_support import check_config

        cfg = self._load(training="  use_galore: true\n")
        assert {e.field for e in check_config(cfg)} == {"training.use_galore"}

    def test_the_ignored_settings_are_printed(self, monkeypatch):
        from io import StringIO

        from rich.console import Console

        from soup_cli.trainer import mlx_dpo
        from soup_cli.trainer.mlx_simpo import MLXSimPOTrainerWrapper

        buffer = StringIO()
        monkeypatch.setattr(mlx_dpo, "console", Console(file=buffer, width=400))
        MLXSimPOTrainerWrapper(self._load(training="  use_galore: true\n"))._check_unsupported()
        assert "MLX backend ignores: GaLore" in _strip_ansi(buffer.getvalue())
