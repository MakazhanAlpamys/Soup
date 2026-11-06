"""Native SimNPO unlearning on MLX (#2005).

The objective is the transformers backend's ``task: unlearn`` with
``unlearn_method: simnpo`` (``trainer/unlearn.py`` and
``utils/unlearn_kernels.simnpo_loss``). Layers:

- config and dispatch: SimNPO loads on ``backend: mlx``; NPO, RMU and the rest
  of the scope are refused at load, each beside a config that loads (no MLX);
- tokenization against the transformers path's ``_tokenize_pair``;
- the loss and its gradients against an independent float64 NumPy oracle on a
  tiny random Llama, with every listed mutation shown to move the result;
- rows, setup, training, saving and a fresh-process reload on that model
  (needs MLX and mlx-lm). No downloads.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import strip_ansi
from tests.test_mlx_orpo import CharTokenizer, _tiny_llama_args

_YAML = """\
base: tiny-llama
task: unlearn
backend: mlx
data:
  train: forget.jsonl
  forget_set: forget.jsonl
  retain_set: retain.jsonl
  max_length: 64
training:
  unlearn_method: simnpo
  epochs: 1
  lr: 1e-2
  batch_size: 2
  quantization: none
  lora: {r: 4, alpha: 8, dropout: 0.0}
output: ./output
"""


def _load(text: str = _YAML, **training):
    from soup_cli.config.loader import load_config_from_string

    extra = "".join(f"  {k}: {v}\n" for k, v in training.items())
    return load_config_from_string(text.replace("  epochs: 1\n", "  epochs: 1\n" + extra))


class TemplateTokenizer(CharTokenizer):
    """A CharTokenizer with a chat template, so the prompt is rendered."""

    chat_template = "<{role}>{content}|"


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _forget(n=5):
    return [{"prompt": f"who is {i}?", "completion": " the secret" + "s" * i} for i in range(n)]


def _retain(n=3):
    return [{"text": f"hello world {i}"} for i in range(n)]


def _mlx():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    return mx


def _wrapper(tmp_path, monkeypatch, *, forget=None, retain=None, model_seed=0, text=_YAML,
             tokenizer=None, **training):
    """A wrapper with its rows read and a tiny random Llama in place of a load."""
    mx = _mlx()
    from mlx_lm.models import llama

    from soup_cli.trainer.mlx_unlearn import MLXSimNPOTrainerWrapper

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "forget.jsonl", _forget() if forget is None else forget)
    if "retain_set" in text:
        _write(tmp_path / "retain.jsonl", _retain() if retain is None else retain)
    wrapper = MLXSimNPOTrainerWrapper(_load(text, **training))
    wrapper._load_rows()
    mx.random.seed(model_seed)
    wrapper.model = llama.Model(_tiny_llama_args())
    wrapper.tokenizer = tokenizer or CharTokenizer()
    return wrapper


_NO_RETAIN = _YAML.replace("  retain_set: retain.jsonl\n", "")


def _flat(params):
    from mlx.utils import tree_flatten

    return {k: np.array(v) for k, v in tree_flatten(params)}


# --------------------------------------------------------------------------
# Config and dispatch (no MLX)
# --------------------------------------------------------------------------


class TestConfig:
    def test_simnpo_on_mlx_loads_and_is_an_mlx_task(self):
        from soup_cli.config.schema import MLX_TASKS

        cfg = _load()
        assert (cfg.task, cfg.backend, cfg.training.unlearn_method) == ("unlearn", "mlx", "simnpo")
        assert "unlearn" in MLX_TASKS

    def test_unlearn_dispatches_to_the_native_trainer(self):
        from soup_cli.trainer.dispatch import build_trainer
        from soup_cli.trainer.mlx_routing import MLX_TRAINER_REGISTRY, get_mlx_trainer
        from soup_cli.trainer.mlx_unlearn import MLXSimNPOTrainerWrapper

        assert "unlearn" in MLX_TRAINER_REGISTRY and "unlearn" in MLX_TRAINER_REGISTRY.keys()
        assert get_mlx_trainer("unlearn") is MLXSimNPOTrainerWrapper
        assert isinstance(build_trainer(_load()), MLXSimNPOTrainerWrapper)

    @pytest.mark.parametrize("method", ["npo", "rmu"])
    def test_npo_and_rmu_are_refused_naming_the_method(self, method):
        with pytest.raises(Exception) as excinfo:
            _load(_YAML.replace("unlearn_method: simnpo", f"unlearn_method: {method}"))
        assert (
            f"task='unlearn' with unlearn_method='{method}' is not supported on backend=mlx: "
            "MLX runs unlearn_method: simnpo only"
        ) in str(excinfo.value)

    @pytest.mark.parametrize("method", ["npo", "rmu", "simnpo"])
    def test_every_method_still_loads_on_transformers(self, method):
        text = _YAML.replace("backend: mlx", "backend: transformers").replace(
            "unlearn_method: simnpo", f"unlearn_method: {method}"
        )
        assert _load(text).training.unlearn_method == method

    @pytest.mark.parametrize("old, new, match", [
        ("backend: mlx\n", "backend: mlx\nmodality: vision\n",
         "modality='vision' is not supported by task='unlearn' on backend='mlx'"),
        ("lora: {r: 4, alpha: 8, dropout: 0.0}", "lora: {r: 0}",
         "lora.r must be at least 1 for task='unlearn' on backend='mlx'"),
        ("quantization: none", "quantization: 4bit",
         "training.quantization: 4bit has no effect on task='unlearn'"),
        ("lora: {r: 4, alpha: 8, dropout: 0.0}", "lora: {r: 4, alpha: 8, use_dora: true}",
         "MLX DoRA requires task='sft'"),
        ("  epochs: 1\n", "  epochs: 1\n  loraplus_lr_ratio: 16\n",
         "loraplus_lr_ratio is not supported on task='unlearn'"),
    ])
    def test_the_rest_of_the_scope_is_refused_at_load(self, old, new, match):
        _load()  # the same config without the change loads
        with pytest.raises(Exception, match=match):
            _load(_YAML.replace(old, new))

    @pytest.mark.parametrize("method", ["npo", "rmu"])
    def test_a_quantized_npo_or_rmu_config_is_told_both_fixes_at_once(self, method):
        """The quantization refusal fires first (#795); for a method MLX cannot run it
        also names the backend, so the config edited as it says loads (#1357)."""
        text = _YAML.replace("unlearn_method: simnpo", f"unlearn_method: {method}")
        with pytest.raises(Exception) as excinfo:
            _load(text.replace("quantization: none", "quantization: 4bit"))
        assert (
            f"Also, task='unlearn' with unlearn_method='{method}' has no trainer on "
            "backend='mlx'. Use backend: transformers."
        ) in str(excinfo.value)
        assert _load(text.replace("backend: mlx", "backend: transformers")).backend == (
            "transformers"
        )

    def test_a_quantized_simnpo_config_is_told_only_the_quantization_fix(self):
        with pytest.raises(Exception) as excinfo:
            _load(_YAML.replace("quantization: none", "quantization: 4bit"))
        assert "Use backend: transformers" not in str(excinfo.value)
        assert _load().backend == "mlx"

    def test_the_mlx_task_list_names_simnpo(self):
        text = _YAML.replace("task: unlearn", "task: ipo").replace(
            "  unlearn_method: simnpo\n", ""
        )
        with pytest.raises(Exception, match="plus SimNPO unlearning"):
            _load(text)

    def test_doctor_reports_ignored_settings(self):
        from soup_cli.config.backend_support import check_config

        cfg = _load(use_galore="true")
        assert {e.field for e in check_config(cfg)} == {"training.use_galore"}


# --------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------


def _hf_tokenizer(template: bool):
    """The real byte-level BPE tokenizer of ``tests/_tiny_hf_models.py``."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests._tiny_hf_models import build_tokenizer

    tok = build_tokenizer(("who is it the secret", "whole text row", "a long prompt here"))
    if template:
        tok.chat_template = (
            "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}"
        )
    return tok


class TestTokens:
    @pytest.mark.parametrize("template", [False, True], ids=["no-template", "template"])
    @pytest.mark.parametrize("prompt, target, max_length", [
        ("who is it?", " the secret", 64),
        ("", "whole text row", 64),
        ("a long prompt here", " cut target", 22),
    ])
    def test_ids_and_target_span_match_the_transformers_path(
        self, template, prompt, target, max_length
    ):
        tokenizer = _hf_tokenizer(template)
        import torch

        from soup_cli.trainer.mlx_unlearn import build_unlearn_example
        from soup_cli.utils.live_eval import _tokenize_pair

        # Under mlx-lm the HF tokenizer sits inside a TokenizerWrapper.
        wrapped = SimpleNamespace(_tokenizer=tokenizer)
        example = build_unlearn_example(wrapped, prompt, target, max_length=max_length)
        input_ids, labels = _tokenize_pair(tokenizer, prompt, target, max_length=max_length)
        assert example.ids == input_ids[0].tolist()
        supervised = (labels[0] != -100).nonzero().flatten().tolist()
        assert supervised == list(range(example.target_start, len(example.ids)))
        assert example.scored == int((labels[0, 1:] != -100).sum()) == len(
            [i for i in supervised if i >= 1]
        )
        assert isinstance(input_ids, torch.Tensor)

    def test_the_prompt_is_one_rendered_user_turn_and_eos_ends_the_target(self):
        from soup_cli.trainer.mlx_unlearn import build_unlearn_example

        tok = TemplateTokenizer()
        example = build_unlearn_example(tok, "hi", " yo", max_length=64)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": "hi"}], tokenize=True, add_generation_prompt=True
        )
        target = tok.encode(" yo", add_special_tokens=False) + [tok.eos_token_id]
        assert example.ids == prompt + target
        assert example.target_start == len(prompt)

    def test_a_row_cut_inside_its_prompt_has_nothing_scored(self):
        from soup_cli.trainer.mlx_unlearn import build_unlearn_example

        example = build_unlearn_example(CharTokenizer(), "x" * 30, " target", max_length=20)
        assert len(example.ids) == 20 and example.scored == 0

    def test_a_one_token_target_with_an_empty_prompt_has_nothing_scored(self):
        """Position 0 has nothing before it, so a lone first token is not scored;
        the transformers path would score it against a length clamped to 1."""
        from soup_cli.trainer.mlx_unlearn import build_unlearn_example

        class NoEos(CharTokenizer):
            eos_token_id = None

        example = build_unlearn_example(NoEos(), "", "z", max_length=64)
        assert (example.target_start, example.scored) == (0, 0)


# --------------------------------------------------------------------------
# The loss against a NumPy oracle
# --------------------------------------------------------------------------


def _np_logps(model, tokens, mask):
    """Summed target log-probs and scored counts, in float64, from the logits."""
    mx = _mlx()
    logits = np.array(model(mx.array(tokens)[:, :-1]).astype(mx.float32), dtype=np.float64)
    top = logits.max(axis=-1, keepdims=True)
    logsm = logits - top - np.log(np.exp(logits - top).sum(axis=-1, keepdims=True))
    tokens, mask = np.array(tokens), np.array(mask, dtype=np.float64)
    gathered = np.take_along_axis(logsm, tokens[:, 1:, None], axis=-1)[..., 0]
    return (gathered * mask[:, 1:]).sum(axis=-1), mask[:, 1:].sum(axis=-1)


def _np_loss(model, forget, retain, *, alpha, beta=0.1, gamma=0.0):
    s, n = _np_logps(model, *forget)
    forget_loss = (2.0 / beta) * np.log1p(np.exp(beta * (s / n - gamma))).mean()
    if retain is None:
        return forget_loss, forget_loss, None
    rs, rn = _np_logps(model, *retain)
    retain_ce = -rs.sum() / rn.sum()
    return forget_loss + alpha * retain_ce, forget_loss, retain_ce


def _batch(wrapper):
    """One forget batch of three rows of unequal length, and its retain rows."""
    examples = wrapper._examples(wrapper._dataset["train"])[0][:3]
    return examples, wrapper._batch_arrays(examples)


class TestTheLossMatchesTheOracle:
    @pytest.mark.parametrize("alpha, expected", [(None, 1.0), (0.0, 0.0), (0.5, 0.5)])
    def test_terms_and_total(self, tmp_path, monkeypatch, alpha, expected):
        mx = _mlx()
        training = {} if alpha is None else {"unlearn_alpha": alpha}
        wrapper = _wrapper(tmp_path, monkeypatch, **training)
        _, arrays = _batch(wrapper)
        forget = tuple(np.array(a) for a in arrays[:2])
        retain = tuple(np.array(a) for a in arrays[2:])
        lengths = forget[1][:, 1:].sum(axis=-1)
        assert len(set(lengths.tolist())) == 3, "the rows must differ in length"
        assert forget[1][:, 0].sum() == 0 and forget[1][:, 1].sum() == 0, "a prompt is scored"
        loss, metrics = wrapper._loss(wrapper.model, arrays, "train", None)
        mx.eval(loss)
        total, forget_loss, retain_ce = _np_loss(wrapper.model, forget, retain, alpha=expected)
        np.testing.assert_allclose(metrics["forget_loss"].item(), forget_loss, rtol=1e-5)
        np.testing.assert_allclose(metrics["retain_ce"].item(), retain_ce, rtol=1e-5)
        np.testing.assert_allclose(loss.item(), total, rtol=1e-5)

    def test_without_a_retain_set_the_loss_is_the_forget_term(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, text=_NO_RETAIN)
        examples = wrapper._examples(wrapper._dataset["train"])[0][:3]
        arrays = wrapper._batch_arrays(examples)
        assert len(arrays) == 2
        loss, metrics = wrapper._loss(wrapper.model, arrays, "train", None)
        total, _, _ = _np_loss(wrapper.model, tuple(np.array(a) for a in arrays), None, alpha=1.0)
        np.testing.assert_allclose(loss.item(), total, rtol=1e-5)
        assert set(metrics) == set(wrapper.metric_names) == {"forget_loss", "forget_mean_logp"}

    def test_the_chunked_head_gives_the_same_loss(self, tmp_path, monkeypatch):
        from soup_cli.trainer.mlx_dpo import resolve_head_split

        wrapper = _wrapper(tmp_path, monkeypatch)
        _, arrays = _batch(wrapper)
        full, _ = wrapper._loss(wrapper.model, arrays, "train", None)
        wrapper.model.freeze()  # only a frozen head takes the chunked path
        wrapper._head_split = resolve_head_split(wrapper.model)
        assert wrapper._head_split is not None
        chunked, _ = wrapper._loss(wrapper.model, arrays, "train", None)
        np.testing.assert_allclose(chunked.item(), full.item(), rtol=1e-4)

    @pytest.mark.parametrize("mutation", [
        "sign reversed", "prompt tokens scored", "padding scored", "sum not s/n",
    ])
    def test_each_mutation_moves_the_oracle(self, tmp_path, monkeypatch, mutation):
        """The oracle above can tell each listed mutation from the objective."""
        wrapper = _wrapper(tmp_path, monkeypatch)
        _, arrays = _batch(wrapper)
        tokens, mask = (np.array(a) for a in arrays[:2])
        widths = np.array([len(e.ids) for e in wrapper._tokenize(wrapper._forget_rows, "f")[:3]])
        s, n = _np_logps(wrapper.model, tokens, mask)

        def term(s, n, sign=1.0):
            return (20.0 * np.log1p(np.exp(sign * 0.1 * s / n))).mean()

        right = term(s, n)
        if mutation == "sign reversed":
            wrong = term(s, n, sign=-1.0)
        elif mutation == "prompt tokens scored":
            wrong = term(*_np_logps(wrapper.model, tokens, (tokens > 0) * 1))
        elif mutation == "padding scored":
            assert len(set(widths.tolist())) > 1
            wrong = term(*_np_logps(wrapper.model, tokens, np.where(mask.cumsum(-1) > 0, 1, 0)))
        else:
            wrong = term(s, np.ones_like(n))
        assert abs(wrong - right) > 1e-3


class TestGradients:
    def _grads(self, wrapper, arrays):
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        fn = nn.value_and_grad(
            wrapper.model, lambda model, a: wrapper._loss(model, a, "train", None)
        )
        (_, _), grads = fn(wrapper.model, arrays)
        return {k: np.array(v) for k, v in tree_flatten(grads)}

    def _lora_wrapper(self, tmp_path, monkeypatch, **training):
        mx = _mlx()
        wrapper = _wrapper(tmp_path, monkeypatch, **training)
        wrapper._apply_lora(wrapper.model)
        # A nonzero B, so the A gradients are not zero by construction.
        for _, module in wrapper.model.named_modules():
            if hasattr(module, "lora_b"):
                module.lora_b = mx.random.normal(module.lora_b.shape) * 0.1
        return wrapper

    def test_alpha_zero_removes_the_retain_gradient_but_not_the_forget_one(
        self, tmp_path, monkeypatch
    ):
        wrapper = self._lora_wrapper(tmp_path, monkeypatch, unlearn_alpha=0.0)
        _, arrays = _batch(wrapper)
        with_retain = self._grads(wrapper, arrays)
        forget_only = self._grads(wrapper, arrays[:2])
        assert with_retain.keys() == forget_only.keys() and with_retain
        for k in with_retain:
            np.testing.assert_allclose(with_retain[k], forget_only[k], rtol=1e-5, atol=1e-7)
        assert any(np.abs(v).max() > 0 for v in forget_only.values())

    def test_a_positive_alpha_adds_the_retain_gradient(self, tmp_path, monkeypatch):
        wrapper = self._lora_wrapper(tmp_path, monkeypatch, unlearn_alpha=0.5)
        _, arrays = _batch(wrapper)
        with_retain, forget_only = self._grads(wrapper, arrays), self._grads(wrapper, arrays[:2])
        assert max(np.abs(with_retain[k] - forget_only[k]).max() for k in with_retain) > 1e-6

    def test_forget_only_descent_lowers_the_target_log_prob(self, tmp_path, monkeypatch):
        mx = _mlx()
        import mlx.optimizers as optim

        wrapper = self._lora_wrapper(tmp_path, monkeypatch)
        _, arrays = _batch(wrapper)
        forget = arrays[:2]
        before, _ = _np_logps(wrapper.model, *(np.array(a) for a in forget))
        import mlx.nn as nn

        opt = optim.SGD(learning_rate=0.5)
        step = nn.value_and_grad(
            wrapper.model, lambda model: wrapper._loss(model, forget, "train", None)
        )
        for _ in range(3):
            _, grads = step(wrapper.model)
            opt.update(wrapper.model, grads)
            mx.eval(wrapper.model.parameters())
        after, _ = _np_logps(wrapper.model, *(np.array(a) for a in forget))
        assert (after < before).all()


# --------------------------------------------------------------------------
# Rows and setup
# --------------------------------------------------------------------------


def _capture(monkeypatch):
    from rich.console import Console

    from soup_cli.trainer import mlx_dpo, mlx_unlearn

    buffer = StringIO()
    console = Console(file=buffer, width=400)
    monkeypatch.setattr(mlx_unlearn, "console", console)
    monkeypatch.setattr(mlx_dpo, "console", console)
    return lambda: " ".join(strip_ansi(buffer.getvalue()).split())


def _setup_wrapper(tmp_path, monkeypatch, *, forget, retain=None, text=_YAML, model=None):
    """A wrapper whose real ``setup()`` loads the tiny Llama, or fails if it loads."""
    _mlx()
    from mlx_lm.models import llama

    from soup_cli.trainer.mlx_unlearn import MLXSimNPOTrainerWrapper
    from soup_cli.utils import mlx as mlx_utils

    monkeypatch.chdir(tmp_path)
    (tmp_path / "forget.jsonl").write_text(forget, encoding="utf-8")
    if retain is not None:
        (tmp_path / "retain.jsonl").write_text(retain, encoding="utf-8")

    def load(*_a, **_k):
        if model == "refuse":
            raise AssertionError("a model was loaded before the rows were checked")
        return (model or llama.Model(_tiny_llama_args())), CharTokenizer()

    monkeypatch.setattr(mlx_utils, "load_mlx_model", load)
    return MLXSimNPOTrainerWrapper(_load(text))


_ROWS = "".join(json.dumps(r) + "\n" for r in _forget(2))
_UNUSABLE = "\n   \n{not json\n" + json.dumps({"prompt": "p", "completion": ""}) + "\n"


class TestRowsAndSetup:
    def test_an_unusable_retain_set_fails_before_the_model_loads(self, tmp_path, monkeypatch):
        wrapper = _setup_wrapper(tmp_path, monkeypatch, forget=_ROWS, retain=_UNUSABLE,
                                 model="refuse")
        with pytest.raises(ValueError, match="^retain_set 'retain.jsonl' yielded no usable rows$"):
            wrapper.setup({})

    def test_an_unusable_forget_set_fails_before_the_model_loads(self, tmp_path, monkeypatch):
        wrapper = _setup_wrapper(tmp_path, monkeypatch, forget=_UNUSABLE, retain=_ROWS,
                                 model="refuse")
        with pytest.raises(ValueError, match="^forget_set 'forget.jsonl' yielded no usable rows$"):
            wrapper.setup({})

    def test_an_absent_retain_set_warns_and_sets_up(self, tmp_path, monkeypatch):
        out = _capture(monkeypatch)
        wrapper = _setup_wrapper(tmp_path, monkeypatch, forget=_ROWS, text=_NO_RETAIN)
        wrapper.setup({})
        assert "method=simnpo has no retain_set; general capability is unanchored" in out()
        assert wrapper.model is not None and wrapper._retain_rows == []
        assert "retain_ce" not in wrapper.metric_names

    def test_a_usable_retain_set_sets_up_without_the_warning(self, tmp_path, monkeypatch):
        out = _capture(monkeypatch)
        wrapper = _setup_wrapper(tmp_path, monkeypatch, forget=_ROWS, retain=_ROWS)
        wrapper.setup({})
        assert "no retain_set" not in out()
        assert len(wrapper._retain_rows) == 2 and "retain_ce" in wrapper.metric_names

    def test_the_accepted_row_forms_load_and_the_skips_are_counted(self, tmp_path, monkeypatch):
        out = _capture(monkeypatch)
        forget = "".join(json.dumps(r) + "\n" for r in (
            {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
            {"prompt": "p", "completion": "c"},
            {"prompt": "p", "target": "t"},
            {"text": "whole"},
        )) + "\n{bad\n" + json.dumps({"prompt": "p"}) + "\n"
        wrapper = _setup_wrapper(tmp_path, monkeypatch, forget=forget, retain=_ROWS)
        wrapper.setup({})
        assert wrapper._forget_rows == [("q", "a"), ("p", "c"), ("p", "t"), ("", "whole")]
        assert ("MLX SimNPO: forget_set 'forget.jsonl': skipped 1 blank, 1 not JSON, "
                "1 no usable prompt/target") in out()

    def test_the_row_limit_is_reported(self, tmp_path, monkeypatch):
        from soup_cli.trainer import unlearn

        out = _capture(monkeypatch)
        monkeypatch.setattr(unlearn, "_MAX_ROWS", 2)
        forget = "".join(json.dumps(r) + "\n" for r in _forget(3))
        wrapper = _setup_wrapper(tmp_path, monkeypatch, forget=forget, retain=_ROWS)
        wrapper.setup({})
        assert len(wrapper._forget_rows) == 2
        assert "lines past the row limit not read" in out()

    def test_a_forget_set_outside_cwd_is_refused(self, tmp_path, monkeypatch):
        text = _YAML.replace("forget_set: forget.jsonl", "forget_set: ../outside.jsonl")
        wrapper = _setup_wrapper(tmp_path, monkeypatch, forget=_ROWS, retain=_ROWS, text=text,
                                 model="refuse")
        with pytest.raises(ValueError, match="dataset path must stay under cwd"):
            wrapper.setup({})

    @pytest.mark.requires_symlink
    def test_a_symlinked_retain_set_is_refused(self, tmp_path, monkeypatch):
        (tmp_path / "real.jsonl").write_text(_ROWS, encoding="utf-8")
        (tmp_path / "link.jsonl").symlink_to(tmp_path / "real.jsonl")
        text = _YAML.replace("retain_set: retain.jsonl", "retain_set: link.jsonl")
        wrapper = _setup_wrapper(tmp_path, monkeypatch, forget=_ROWS, text=text, model="refuse")
        with pytest.raises(ValueError, match="dataset path must not be a symlink"):
            wrapper.setup({})

    def test_a_base_other_than_llama_is_refused(self, tmp_path, monkeypatch):
        wrapper = _setup_wrapper(tmp_path, monkeypatch, forget=_ROWS, retain=_ROWS,
                                 model=SimpleNamespace(model_type="qwen2"))
        with pytest.raises(ValueError, match="Llama-architecture bases only .* is 'qwen2'"):
            wrapper.setup({})

    def test_a_zero_target_row_is_skipped_and_counted(self, tmp_path, monkeypatch):
        out = _capture(monkeypatch)
        forget = _forget(3) + [{"prompt": "x" * 80, "completion": " gone"}]
        wrapper = _wrapper(tmp_path, monkeypatch, forget=forget, gradient_accumulation_steps=1)
        result = wrapper.train()
        assert "skipped 1 forget_set row(s) with no target token left" in out()
        assert result["forget_rows_without_target"] == 1
        assert result["forget_rows_scored"] == 3

    def test_a_forget_set_with_no_target_token_left_fails(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, forget=[{"prompt": "x" * 80, "completion": "y"}])
        with pytest.raises(ValueError, match="no forget_set row has a target token left"):
            wrapper.train()

    def test_a_retain_set_with_no_target_token_left_fails(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, retain=[{"prompt": "x" * 80, "completion": "y"}])
        with pytest.raises(ValueError, match="no retain_set row has a target token left"):
            wrapper.train()


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


class TestTraining:
    def test_lora_trains_and_the_base_stays_frozen(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch)
        before = _flat(wrapper.model.parameters())
        result = wrapper.train()
        after = _flat(wrapper.model.parameters())
        assert any(np.abs(v).sum() > 0 for k, v in after.items() if k.endswith("lora_b"))
        for name, value in before.items():
            moved = name.replace(".weight", ".linear.weight")
            key = name if name in after else moved
            np.testing.assert_array_equal(after[key], value, err_msg=name)
        assert np.isfinite(result["final_loss"]) and result["updates"] > 0

    def test_the_last_partial_batch_and_accumulation_group_train(self, tmp_path, monkeypatch):
        """5 rows at batch_size 2 make 3 batches, 2 of them in the first group of 2:
        the third batch, of one row, is the second update."""
        wrapper = _wrapper(tmp_path, monkeypatch, gradient_accumulation_steps=2)
        result = wrapper.train()
        assert (result["total_steps"], result["updates"]) == (3, 2)
        assert result["forget_rows_scored"] == 5 and result["retain_rows_scored"] == 5
        meta = json.loads((tmp_path / "output" / "adapter_config.json").read_text())
        assert (meta["iters"], meta["grad_accumulation_steps"]) == (3, 2)

    def test_retain_rows_cycle_round_robin(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch)
        examples, _ = wrapper._examples(wrapper._dataset["train"])
        pool = wrapper._retain_examples
        picks = []
        for _ in range(3):
            arrays = wrapper._batch_arrays(examples[:2])
            picks.append(np.array(arrays[2]).tolist())
        expected = [pool[i % 3] for i in range(6)]
        flat = [row for batch in picks for row in batch]
        for got, example in zip(flat, expected):
            assert got[: len(example.ids)] == example.ids

    def test_the_components_and_counts_are_reported_and_saved(self, tmp_path, monkeypatch):
        out = _capture(monkeypatch)
        wrapper = _wrapper(tmp_path, monkeypatch, unlearn_alpha=0.25, logging_steps=1,
                           gradient_accumulation_steps=1)
        result = wrapper.train()
        text = out()
        assert "forget " in text and "retain CE " in text
        assert "MLX SimNPO: 3 update(s), 5 forget and 5 retain row(s) scored" in text
        meta = json.loads((tmp_path / "output" / "adapter_config.json").read_text())
        assert {k: meta[k] for k in (
            "task", "unlearn_method", "simnpo_beta", "simnpo_gamma", "unlearn_alpha",
            "forget_rows", "retain_rows", "model",
        )} == {
            "task": "unlearn", "unlearn_method": "simnpo", "simnpo_beta": 0.1,
            "simnpo_gamma": 0.0, "unlearn_alpha": 0.25, "forget_rows": 5, "retain_rows": 3,
            "model": "tiny-llama",
        }
        assert "skipped_identical_pairs" not in result

    def test_without_retain_rows_alpha_is_not_recorded(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, text=_NO_RETAIN)
        result = wrapper.train()
        meta = json.loads((tmp_path / "output" / "adapter_config.json").read_text())
        assert meta["unlearn_alpha"] is None and result["retain_rows_scored"] == 0

    @pytest.mark.parametrize("change, match", [
        ({"task": "simpo"}, "MLX SimNPO: .* task='simpo', not 'unlearn'"),
        ({"model": "other-base"}, "belongs to base"),
        (None, "MLX SimNPO: no adapter_config.json"),
    ])
    def test_a_foreign_or_unlabelled_adapter_is_refused(self, tmp_path, monkeypatch, change,
                                                        match):
        first = _wrapper(tmp_path / "a", monkeypatch)
        first.train()
        checkpoint = tmp_path / "a" / "output" / "adapters.safetensors"
        meta = checkpoint.parent / "adapter_config.json"
        if change is None:
            meta.unlink()
        else:
            meta.write_text(json.dumps({**json.loads(meta.read_text()), **change}))
        second = _wrapper(tmp_path / "b", monkeypatch)
        second._apply_lora(second.model)
        with pytest.raises(ValueError, match=match):
            second._load_checkpoint_weights(str(checkpoint))

    def test_an_interrupted_save_leaves_no_adapter_file(self, tmp_path, monkeypatch):
        mx = _mlx()
        wrapper = _wrapper(tmp_path, monkeypatch)

        def crash(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(mx, "save_safetensors", crash)
        with pytest.raises(OSError, match="disk full"):
            wrapper.train()
        assert not (tmp_path / "output" / "adapters.safetensors").exists()


class TestFreshProcessReload:
    def test_logits_reproduce_and_the_forget_likelihood_is_lower(self, tmp_path, monkeypatch):
        mx = _mlx()
        wrapper = _wrapper(tmp_path, monkeypatch, epochs=4, lr="5e-2",
                           gradient_accumulation_steps=1)
        base_path = tmp_path / "base.safetensors"
        wrapper.model.save_weights(str(base_path))
        wrapper.train()
        examples = wrapper._tokenize(wrapper._forget_rows, "forget_set")
        tokens, mask = wrapper._arrays(examples)
        wrapper.model.eval()
        logits = wrapper.model(tokens).astype(mx.float32)
        mx.eval(logits)
        np.save(tmp_path / "tokens.npy", np.array(tokens))
        np.save(tmp_path / "mask.npy", np.array(mask))

        script = textwrap.dedent(f"""
            import sys
            import numpy as np
            import mlx.core as mx
            from mlx_lm.models import llama
            from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters
            args = llama.ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=2,
                intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                rms_norm_eps=1e-5, vocab_size=200, tie_word_embeddings=True)
            model = llama.Model(args)
            model.load_weights({str(base_path)!r})
            if sys.argv[1] == "adapter":
                load_adapters(model, {str(tmp_path / "output")!r})
            elif sys.argv[1] == "untrained":
                model.freeze()
                linear_to_lora_layers(model, 2, {{"rank": 4, "scale": 2.0, "dropout": 0.0,
                    "keys": {{"self_attn.q_proj", "self_attn.v_proj"}}}})
            model.eval()
            t = mx.array(np.load({str(tmp_path / "tokens.npy")!r}))
            np.save(sys.argv[2], np.array(model(t).astype(mx.float32)))
        """)

        def run(mode):
            out = tmp_path / f"{mode}.npy"
            subprocess.run([sys.executable, "-c", script, mode, str(out)], capture_output=True,
                           text=True, check=True, timeout=300)
            return np.load(out).astype(np.float64)

        def mean_logp(logits):
            t, m = np.load(tmp_path / "tokens.npy"), np.load(tmp_path / "mask.npy")
            lg = logits[:, :-1]
            top = lg.max(-1, keepdims=True)
            ls = lg - top - np.log(np.exp(lg - top).sum(-1, keepdims=True))
            g = np.take_along_axis(ls, t[:, 1:, None], -1)[..., 0] * m[:, 1:]
            return float((g.sum(-1) / m[:, 1:].sum(-1)).mean())

        reloaded, base, untrained = run("adapter"), run("base"), run("untrained")
        np.testing.assert_allclose(reloaded, np.array(logits), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(untrained, base, rtol=1e-5, atol=1e-6)
        assert mean_logp(reloaded) < mean_logp(untrained) - 0.1
