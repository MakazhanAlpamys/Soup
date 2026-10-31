"""Native GRPO on MLX (#1976), Step 2: pre-quantized 4-bit bases.

Every base here is a tiny random Llama quantized by mlx-lm's own
``mlx_lm.utils.quantize_model``, saved in mlx-lm's format (``config.json`` and
``model*.safetensors``, as ``mlx_lm.convert -q`` writes them) and loaded through
``soup_cli.utils.mlx.load_mlx_model``, i.e. ``mlx_lm.load``. Only the tokenizer
is replaced, by the digit tokenizer of ``test_mlx_grpo.py``; the model, its
quantized layers and the loader are real.

The rewards are built in: unequal inside every group whatever the device
samples (#2385), so every update moves the adapter. No downloads.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx_lm.utils  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models import llama, mixtral  # noqa: E402

from tests.test_mlx_grpo import (  # noqa: E402
    _TRAIN_CHARS,
    _TRAIN_VOCAB,
    DigitTokenizer,
    _config,
    _flat,
    _np_advantages,
    _np_logps,
    _np_loss,
    _prompt_rows,
    _record_optimizer,
)

_LLAMA = dict(
    model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
    num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5,
    vocab_size=_TRAIN_VOCAB, tie_word_embeddings=True,
)
_MIXTRAL = dict(
    model_type="mixtral", vocab_size=_TRAIN_VOCAB, hidden_size=32, intermediate_size=64,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    num_local_experts=4, num_experts_per_tok=2, rms_norm_eps=1e-5,
)
#: What mlx-lm's 4-bit quantization of the tiny Llama records: every projection of
#: both layers (7 x 2) and the tied embedding, which is also the output head.
_RECORD = {"bits": 4, "group_size": 32, "mode": "affine", "quantized_linear": 14,
           "quantized_embedding": 1}
GROUP = 4


def _save_base(path, *, bits=4, mode="affine", group_size=32, predicate=None,
               arch="llama", quantize=True, seed=0):
    """A tiny random base, quantized by mlx-lm and saved in mlx-lm's format."""
    from mlx_lm.utils import quantize_model, save_config, save_model

    mx.random.seed(seed)
    family, config = (llama, dict(_LLAMA)) if arch == "llama" else (mixtral, dict(_MIXTRAL))
    model = family.Model(family.ModelArgs(**config))
    if quantize:
        model, config = quantize_model(model, config, group_size, bits, mode=mode,
                                       quant_predicate=predicate)
    path.mkdir(parents=True)
    save_model(path, model)
    save_config(config, path / "config.json")
    return path


def _by_position(completions, answer, **_):
    """1 for the first ``1 + gold % 3`` completions of each group, else 0.

    Unequal in every group whatever was sampled, and a different pattern for
    prompts with different golds, so mixing golds between prompts changes it.
    """
    return [1.0 if index % GROUP < 1 + int(gold) % 3 else 0.0
            for index, gold in enumerate(answer)]


def _loaded(tmp_path, monkeypatch, *, base=None, train=None, val=None, reward=_by_position,
            **training):
    """The real wrapper after ``setup()``: the base loaded from disk by mlx-lm."""
    from soup_cli.trainer.mlx_grpo import MLXGRPOTrainerWrapper
    from soup_cli.trainer.rewards import validate_reward_funcs

    tmp_path.mkdir(parents=True, exist_ok=True)
    base = _save_base(tmp_path / "base") if base is None else base
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer",
                        lambda *args, **kwargs: DigitTokenizer(_TRAIN_CHARS))
    # quantization last: _config inserts new keys after the `quantization: none` line.
    config = _config(**training, quantization="4bit")
    config.base = str(base)
    wrapper = MLXGRPOTrainerWrapper(config)
    wrapper.setup({"train": _prompt_rows() if train is None else train,
                   "val": [] if val is None else val})
    if reward is not None:
        wrapper._reward_fn = validate_reward_funcs(reward)
    return wrapper


def _adapted(name, names):
    """A base tensor's name after LoRA wraps its layer (``q_proj.linear.weight``)."""
    parent, leaf = name.rsplit(".", 1)
    wrapped = f"{parent}.linear.{leaf}"
    return wrapped if wrapped in names else name


def _random_lora_b(wrapper, seed=3, std=0.3):
    """Attach the adapter as the trainer does, then give every ``B`` random values,
    so the policy differs from the reference from the first update on."""
    real = wrapper._apply_lora
    rng = np.random.default_rng(seed)

    def apply(model):
        real(model)
        for _, module in model.named_modules():
            if hasattr(module, "lora_b"):
                module.lora_b = mx.array(rng.normal(0.0, std, module.lora_b.shape)
                                         .astype(np.float32))

    return apply


def _device():
    return "cpu" if mx.default_device() == mx.cpu else "gpu"


# --------------------------------------------------------------------------
# Which bases load
# --------------------------------------------------------------------------


class TestTheBase:
    def test_mlx_lms_4bit_output_loads_with_every_layer_quantized(self, tmp_path,
                                                                  monkeypatch):
        wrapper = _loaded(tmp_path, monkeypatch)
        model = wrapper.model
        assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
        linears = [m for _, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)]
        assert len(linears) == 14
        assert not [m for _, m in model.named_modules() if type(m) is nn.Linear]
        assert {(m.bits, m.mode, m.group_size) for m in linears} == {(4, "affine", 32)}
        assert wrapper._base_quantization == _RECORD

    def test_a_dense_base_still_loads_and_records_no_quantization(self, tmp_path, monkeypatch):
        base = _save_base(tmp_path / "dense", quantize=False)
        wrapper = _loaded(tmp_path, monkeypatch, base=base)
        assert wrapper._base_quantization is None

    @pytest.mark.parametrize("predicate, record", [
        pytest.param(lambda path, module: "mlp" not in path,
                     dict(_RECORD, quantized_linear=8), id="mlp-left-dense"),
        pytest.param(lambda path, module: (
            {"group_size": 64, "bits": 4} if path.endswith("down_proj") else True),
            dict(_RECORD, group_size=[32, 64]), id="two-group-sizes"),
    ])
    def test_layers_left_dense_and_each_4bit_group_size_are_recorded(
            self, tmp_path, monkeypatch, predicate, record):
        base = _save_base(tmp_path / "base", predicate=predicate)
        wrapper = _loaded(tmp_path, monkeypatch, base=base)
        assert wrapper._base_quantization == record

    @pytest.mark.parametrize("quantize, found", [
        pytest.param(dict(bits=8), r"\((QuantizedLinear|QuantizedEmbedding), bits=8, "
                     r"mode='affine'\)", id="8-bit"),
        pytest.param(dict(predicate=lambda path, module: (
            {"group_size": 32, "bits": 8} if path.endswith("down_proj") else True)),
            r"2 layer\(s\) outside it, first: model\.layers\.\d\.mlp\.down_proj "
            r"\(QuantizedLinear, bits=8, mode='affine'\)", id="mixed-4-and-8-bit"),
        pytest.param(dict(mode="mxfp4", group_size=None), r"bits=4, mode='mxfp4'\)",
                     id="mxfp4"),
        pytest.param(dict(mode="mxfp8", group_size=None, bits=None),
                     r"bits=8, mode='mxfp8'\)", id="mxfp8"),
        pytest.param(dict(mode="nvfp4", group_size=None), r"bits=4, mode='nvfp4'\)",
                     id="nvfp4"),
        pytest.param(dict(arch="mixtral"), r"first: model\.layers\.\d\.block_sparse_moe\."
                     r"switch_mlp\.\w+ \(QuantizedSwitchLinear, bits=4, mode='affine'\)",
                     id="quantized-moe"),
    ])
    def test_every_other_format_is_refused_right_after_loading_by_name(
            self, tmp_path, monkeypatch, quantize, found):
        base = _save_base(tmp_path / "base", **quantize)
        with pytest.raises(ValueError, match="MLX GRPO trains a dense base or one "
                                             "pre-quantized at 4 bits") as refused:
            _loaded(tmp_path, monkeypatch, base=base)
        assert re.search(found, str(refused.value)), str(refused.value)
        assert str(base) in str(refused.value)

    def test_train_checks_the_base_again(self, tmp_path, monkeypatch):
        wrapper = _loaded(tmp_path, monkeypatch)
        layer = wrapper.model.model.layers[0].mlp
        layer.down_proj = nn.QuantizedLinear(64, 32, bias=False, group_size=32, bits=8)
        with pytest.raises(ValueError, match=r"first: model\.layers\.0\.mlp\.down_proj "
                                             r"\(QuantizedLinear, bits=8"):
            wrapper.train()
        assert not (tmp_path / "output" / "adapters.safetensors").exists()


# --------------------------------------------------------------------------
# Training on the 4-bit base
# --------------------------------------------------------------------------


class TestTraining:
    def test_lora_trains_and_every_quantized_tensor_stays_bit_identical(self, tmp_path,
                                                                        monkeypatch):
        wrapper = _loaded(tmp_path, monkeypatch)
        before = _flat(wrapper.model.parameters())
        quantized = {k for k in before if k.rsplit(".", 1)[-1] in ("scales", "biases")}
        assert len(quantized) == 2 * 15, "every quantized layer has scales and biases"
        result = wrapper.train()
        after = _flat(wrapper.model.parameters())
        for name, value in before.items():
            key = _adapted(name, after)
            assert after[key].dtype == value.dtype, name
            np.testing.assert_array_equal(after[key], value, err_msg=name)
        trainable = _flat(wrapper.model.trainable_parameters())
        assert trainable and {k.rsplit(".", 1)[-1] for k in trainable} == {"lora_a", "lora_b"}
        assert all(np.abs(v).max() > 1e-4 for k, v in trainable.items()
                   if k.endswith("lora_b")), "every B must move from zero"
        assert result["total_steps"] == 8 and len(result["losses"]) == 8
        assert all(math.isfinite(x) for x in result["losses"]) and result["skipped_steps"] == 0
        assert all(h["frac_reward_zero_std"] == 0.0 for h in result["history"])

    def test_micro_steps_accumulate_to_one_larger_micro_step(self, tmp_path, monkeypatch):
        """DAPO's single normaliser on the 4-bit base: 4 x 2 trains as 8 x 1."""
        def run(path, batch_size, accum):
            wrapper = _loaded(path, monkeypatch, train=_prompt_rows(2), batch_size=batch_size,
                              gradient_accumulation_steps=accum, optimizer="sgd",
                              scheduler="constant", seed=7)
            result = wrapper.train()
            return result, _flat(wrapper.model.trainable_parameters())

        one, w1 = run(tmp_path / "a", 8, 1)
        two, w2 = run(tmp_path / "b", 4, 2)
        assert len(one["losses"]) == len(two["losses"]) == 1
        assert one["num_items"] == two["num_items"]
        assert one["losses"][0] == pytest.approx(two["losses"][0], abs=1e-6)
        assert max(np.abs(v).max() for k, v in w1.items() if k.endswith("lora_b")) > 1e-4
        for name in w1:
            np.testing.assert_allclose(w1[name], w2[name], atol=1e-6, err_msg=name)

    def test_gradient_checkpointing_trains_the_same_update(self, tmp_path, monkeypatch):
        # mlx-lm's grad_checkpoint patches the block class process-wide; the
        # monkeypatch restores it after this test.
        original = llama.TransformerBlock.__call__
        monkeypatch.setattr(llama.TransformerBlock, "__call__", original)

        def run(path, **training):
            wrapper = _loaded(path, monkeypatch, train=_prompt_rows(2), seed=3, **training)
            return wrapper.train(), _flat(wrapper.model.trainable_parameters())

        plain, w_plain = run(tmp_path / "a")
        assert llama.TransformerBlock.__call__ is original
        ckpt, w_ckpt = run(tmp_path / "b", gradient_checkpointing="true")
        assert llama.TransformerBlock.__call__ is not original
        np.testing.assert_allclose(ckpt["losses"], plain["losses"], rtol=1e-4, atol=1e-6)
        for name in w_plain:
            np.testing.assert_allclose(w_ckpt[name], w_plain[name], atol=1e-5, err_msg=name)

    def test_a_trainable_quantized_tensor_is_refused_before_the_first_update(
            self, tmp_path, monkeypatch):
        from soup_cli.trainer import mlx_grpo

        wrapper = _loaded(tmp_path, monkeypatch)
        real = wrapper._apply_lora

        def unfrozen(model):
            real(model)
            model.unfreeze()

        def no_rollout(*args, **kwargs):
            raise AssertionError("no update may start")

        monkeypatch.setattr(wrapper, "_apply_lora", unfrozen)
        monkeypatch.setattr(mlx_grpo.MLXGRPOTrainerWrapper, "_rollout", no_rollout)
        with pytest.raises(RuntimeError, match=r"quantized base tensor\(s\) are trainable "
                                               r"\(first: model\."):
            wrapper.train()
        assert not (tmp_path / "output" / "adapters.safetensors").exists()

    def test_each_trainable_quantized_tensor_is_named(self, tmp_path, monkeypatch):
        from soup_cli.trainer.mlx_grpo import trainable_quantized_tensors

        wrapper = _loaded(tmp_path, monkeypatch)
        model = wrapper.model
        wrapper._apply_lora(model)
        assert trainable_quantized_tensors(model) == []
        model.model.layers[1].self_attn.q_proj.linear.unfreeze()
        assert sorted(trainable_quantized_tensors(model)) == [
            f"model.layers.1.self_attn.q_proj.linear.{key}"
            for key in ("biases", "scales", "weight")]


# --------------------------------------------------------------------------
# The objective on the same quantized base, against independent references
# --------------------------------------------------------------------------


def _grad_recorder(monkeypatch):
    """The gradients the loop hands the optimizer, before clipping."""
    from soup_cli.trainer import mlx_grpo

    seen = []
    real = mlx_grpo._clipping_optimizer

    class Recording:
        def __init__(self, inner):
            self.__dict__["inner"] = inner

        def update(self, model, grads):
            seen.append({k: np.array(v) for k, v in tree_flatten(grads)})
            return self.inner.update(model, grads)

        def __getattr__(self, name):
            return getattr(self.__dict__["inner"], name)

    monkeypatch.setattr(mlx_grpo, "_clipping_optimizer",
                        lambda inner, max_norm: Recording(real(inner, max_norm)))
    return seen


def _independent_batch(samples):
    """Right-padded tokens and the completion mask, written apart from the trainer:
    position t scores token t + 1, which counts when it is a completion token."""
    width = max(len(p) + len(c) for p, c in samples)
    tokens = np.zeros((len(samples), width), dtype=np.int64)
    mask = np.zeros((len(samples), width - 1))
    for i, (prompt, completion) in enumerate(samples):
        tokens[i, : len(prompt) + len(completion)] = list(prompt) + list(completion)
        mask[i, len(prompt) - 1 : len(prompt) - 1 + len(completion)] = 1.0
    return tokens, mask


class TestTheObjective:
    def test_the_loops_update_matches_an_independent_reference_on_the_same_base(
            self, tmp_path, monkeypatch):
        """One update of the real loop on the 4-bit base, in two micro-steps of two
        prompts each (every prompt a different length, so a micro-step's shorter
        prompt sets where its scored positions start), beta > 0 and a policy that
        already differs from the base:
        the loss and the accumulated gradient against a reference that shares none
        of the trainer's code but the model (advantages, mask, k3 KL, normaliser),
        whose reference log-probs come from a second load of the base with no
        adapter at all, and whose full-vocabulary forward bypasses the chunked head.
        """
        from soup_cli.trainer import mlx_grpo

        base = _save_base(tmp_path / "base")
        golds = {"q?": 0, "qq?": 1, "qqq?": 2, "qqqq?": 4}
        rows = [{"prompt": prompt, "answer": str(gold)} for prompt, gold in golds.items()]
        beta = 0.1
        wrapper = _loaded(tmp_path, monkeypatch, base=base, train=rows, batch_size=8,
                          gradient_accumulation_steps=2, grpo_beta=beta, seed=5)
        monkeypatch.setattr(wrapper, "_apply_lora", _random_lora_b(wrapper))
        seen = {}
        real_rollout = mlx_grpo.MLXGRPOTrainerWrapper._rollout

        def rollout(self, hf, prompt_ids, eos, max_new):
            out = real_rollout(self, hf, prompt_ids, eos, max_new)
            seen["prompts"] = [list(p) for p in prompt_ids]
            seen["completions"] = [list(c) for c in out[0]]
            seen["params"] = _flat(self.model.trainable_parameters())
            return out

        monkeypatch.setattr(mlx_grpo.MLXGRPOTrainerWrapper, "_rollout", rollout)
        grads = _grad_recorder(monkeypatch)
        result = wrapper.train()
        assert len(grads) == 1 and len(result["losses"]) == 1
        assert wrapper._head_split is not None, "the loop ran the chunked quantized head"

        # Independent: rewards from each prompt's own gold, advantages, the batch.
        tokenizer = DigitTokenizer(_TRAIN_CHARS)
        samples = [(seen["prompts"][k // GROUP], c) for k, c in enumerate(seen["completions"])]
        rewards = []
        for k, (prompt, _) in enumerate(samples):
            gold = golds[re.search(r"q+\?", tokenizer.decode(prompt)).group(0)]
            rewards.append(1.0 if k % GROUP < 1 + gold % 3 else 0.0)
        adv = _np_advantages(rewards, GROUP)
        assert np.abs(adv).min() > 0.4
        tokens, mask = _independent_batch(samples)
        num_items = float(sum(len(c) for _, c in samples))
        assert len({len(p) for p, _ in samples}) == 4

        # The policy as it was at the rollout; the reference a fresh load of the base.
        model = wrapper.model
        model.update(tree_unflatten([(k, mx.array(v)) for k, v in seen["params"].items()]))
        model.eval()
        reference, _ = mlx_lm.utils.load_model(base)
        assert isinstance(reference.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
        inputs, targets = mx.array(tokens[:, :-1]), tokens[:, 1:]
        logps = _np_logps(np.array(model(inputs)), targets)
        ref = _np_logps(np.array(reference(inputs)), targets)
        assert np.abs((logps - ref) * mask).max() > 1e-2, "the policy must differ from the base"
        expected = _np_loss(logps, logps, ref, adv, mask, beta=beta, num_items=num_items)
        assert result["losses"][0] == pytest.approx(expected, rel=1e-4, abs=1e-6)

        ref32, adv32, mask32 = (mx.array(x.astype(np.float32)) for x in (ref, adv, mask))
        targets32 = mx.array(targets)

        def loss(m):
            z = m(inputs).astype(mx.float32)
            lp = mx.take_along_axis(z - mx.logsumexp(z, axis=-1, keepdims=True),
                                    targets32[..., None], axis=-1)[..., 0]
            ratio = mx.exp(lp - mx.stop_gradient(lp))
            delta = ref32 - lp
            per_token = -ratio * adv32[:, None] + beta * (mx.exp(delta) - delta - 1)
            return (per_token * mask32).sum() / num_items

        _, expected_grads = nn.value_and_grad(model, loss)(model)
        expected_grads = {k: np.array(v) for k, v in tree_flatten(expected_grads)}
        assert expected_grads.keys() == grads[0].keys()
        # The loop's two micro-steps run on their own padding through the chunked
        # head; the reference runs one batch through the full forward. On the M6
        # GPU that reorders float32 sums by up to ~2.3e-6 against gradients up to
        # ~2e-2; 1e-5 is that with margin, and each mutation the issue lists moves
        # some entry by far more.
        for name, value in expected_grads.items():
            np.testing.assert_allclose(grads[0][name], value, rtol=1e-3, atol=1e-5,
                                       err_msg=name)
        assert max(np.abs(v).max() for v in expected_grads.values()) > 1e-3

    def test_gradients_match_finite_differences_on_the_quantized_base(self, tmp_path,
                                                                       monkeypatch):
        """The trainer's own loss on the 4-bit model, differentiated by MLX, against
        central differences of a float64 NumPy evaluation of the same objective."""
        from soup_cli.trainer.mlx_dpo import resolve_head_split
        from soup_cli.trainer.mlx_grpo import (
            collate_completions,
            completion_token_logps,
            grpo_token_loss,
        )

        wrapper = _loaded(tmp_path, monkeypatch)
        model = wrapper.model
        _random_lora_b(wrapper)(model)
        wrapper._head_split = resolve_head_split(model)
        samples = [([3, 4, 5], [6, 7, 2]), ([3, 4, 5, 6, 7], [8, 9]),
                   ([3, 4], [10, 11, 12, 2]), ([3, 4, 5], [13])]
        adv = np.array([1.3, -0.7, 0.9, 0.0])
        beta = 0.1
        tokens, mask, start = collate_completions(samples)
        tokens = np.array(tokens)
        tokens_mx, mask_np = mx.array(tokens), np.array(mask, dtype=np.float64)
        num_items = float(mask_np.sum())
        model.eval()
        ref = wrapper._reference_logps(tokens_mx, start)

        def trainer_loss(m):
            lp = completion_token_logps(m, tokens_mx, start, head_split=wrapper._head_split)
            return grpo_token_loss(lp, mx.array(adv.astype(np.float32)), mx.array(mask),
                                   num_items=num_items, beta=beta, ref_logps=ref)[0]

        _, grads = nn.value_and_grad(model, trainer_loss)(model)
        grads = {k: np.array(v) for k, v in tree_flatten(grads)}
        targets = tokens[:, 1:]

        def numpy_loss(old=None):
            z = np.array(model(tokens_mx[:, :-1]))[:, start:]
            lp = _np_logps(z, targets[:, start:])
            return _np_loss(lp, lp if old is None else old, np.array(ref), adv, mask_np,
                            beta=beta, num_items=num_items), lp

        _, old = numpy_loss()
        # Directional derivatives over every LoRA tensor at once: an Apple M6 GPU
        # computes this float32 forward with about 1e-5 of noise in the loss
        # (#2385), which swamps a central difference on a single entry; along a
        # whole direction the signal is ~1e-2 at this step, so the difference
        # is accurate to well under the 2% allowed (measured 0.4% on the GPU and
        # 1.1% on the CPU, where the rest is the step's own truncation error).
        params = _flat(model.trainable_parameters())
        rng = np.random.default_rng(11)
        eps = 5e-3
        for _ in range(3):
            direction = {k: rng.normal(0.0, 1.0, v.shape).astype(np.float32)
                         for k, v in params.items()}
            values = []
            for sign in (1.0, -1.0):
                model.update(tree_unflatten([(k, mx.array(v + sign * eps * direction[k]))
                                             for k, v in params.items()]))
                values.append(numpy_loss(old)[0])
            model.update(tree_unflatten([(k, mx.array(v)) for k, v in params.items()]))
            numeric = (values[0] - values[1]) / (2 * eps)
            analytic = sum(float((grads[k] * direction[k]).sum()) for k in params)
            assert abs(numeric) > 0.1, "the direction must carry gradient"
            assert analytic == pytest.approx(numeric, rel=2e-2)


# --------------------------------------------------------------------------
# The chunked vocabulary head through QuantizedEmbedding.as_linear
# --------------------------------------------------------------------------


class TestTheQuantizedHead:
    def test_the_chunked_head_is_the_quantized_embedding_and_matches_the_full_forward(
            self, tmp_path, monkeypatch):
        from soup_cli.trainer.mlx_dpo import resolve_head_split
        from soup_cli.trainer.mlx_grpo import collate_completions, completion_token_logps

        wrapper = _loaded(tmp_path, monkeypatch)
        model = wrapper.model
        _random_lora_b(wrapper)(model)
        split = resolve_head_split(model)
        assert split is not None
        assert split[1].__self__ is model.model.embed_tokens
        assert isinstance(split[1].__self__, nn.QuantizedEmbedding)
        tokens, _, start = collate_completions([([3, 4, 5], [6, 7, 8, 9, 10, 2]),
                                                ([3, 4], [11, 12, 2])])
        model.eval()
        full = np.array(completion_token_logps(model, mx.array(tokens), start))
        chunked = np.array(completion_token_logps(model, mx.array(tokens), start,
                                                  head_split=split, chunk_size=2))
        np.testing.assert_allclose(chunked, full, rtol=1e-5, atol=1e-5)
        assert wrapper._verify_head_split([3, 4, 5, 6, 7]) is not None

    def test_a_head_that_changes_the_logits_falls_back_to_the_full_vocabulary(
            self, tmp_path, monkeypatch):
        wrapper = _loaded(tmp_path, monkeypatch)
        wrapper._apply_lora(wrapper.model)
        real = llama.Model.__call__
        monkeypatch.setattr(llama.Model, "__call__",
                            lambda self, *args, **kwargs: 0.5 * real(self, *args, **kwargs))
        assert wrapper._verify_head_split([3, 4, 5, 6, 7]) is None


# --------------------------------------------------------------------------
# Checkpoints, the record, and a fresh-process reload on the 4-bit base
# --------------------------------------------------------------------------


class TestSavedAdapter:
    def test_the_base_quantization_is_recorded_beside_the_objective(self, tmp_path,
                                                                    monkeypatch):
        from soup_cli.trainer import mlx_grpo

        early = []
        real = mlx_grpo.MLXGRPOTrainerWrapper._rollout

        def rollout(self, *args):
            if not early:
                path = tmp_path / "output" / "adapter_config.json"
                early.append(json.loads(path.read_text()))
            return real(self, *args)

        monkeypatch.setattr(mlx_grpo.MLXGRPOTrainerWrapper, "_rollout", rollout)
        _loaded(tmp_path, monkeypatch, train=_prompt_rows(2)).train()
        meta = json.loads((tmp_path / "output" / "adapter_config.json").read_text())
        assert meta["base_quantization"] == early[0]["base_quantization"] == _RECORD
        assert (meta["task"], meta["loss_type"], meta["fine_tune_type"]) == (
            "grpo", "dapo", "lora")

    def test_a_dense_base_records_none(self, tmp_path, monkeypatch):
        base = _save_base(tmp_path / "dense", quantize=False)
        _loaded(tmp_path, monkeypatch, base=base, train=_prompt_rows(1)).train()
        meta = json.loads((tmp_path / "output" / "adapter_config.json").read_text())
        assert "base_quantization" in meta and meta["base_quantization"] is None

    def test_a_checkpoint_holds_the_adapter_of_its_update(self, tmp_path, monkeypatch):
        made = _record_optimizer(monkeypatch)
        _loaded(tmp_path, monkeypatch, save_steps=3).train()
        output = tmp_path / "output"
        assert sorted(p.name for p in output.glob("*_adapters.safetensors")) == [
            "0000003_adapters.safetensors", "0000006_adapters.safetensors"]
        snapshots = made[0].snapshots
        for update in (3, 6):
            saved = mx.load(str(output / f"{update:07d}_adapters.safetensors"))
            weights = snapshots[update - 1][0]
            assert set(saved) == set(weights)
            assert {k.rsplit(".", 1)[-1] for k in saved} == {"lora_a", "lora_b"}
            for name, value in weights.items():
                np.testing.assert_array_equal(np.array(saved[name]), value, err_msg=name)

    def test_a_fresh_process_reloads_the_adapter_on_the_reloaded_4bit_base(
            self, tmp_path, monkeypatch):
        base = _save_base(tmp_path / "base")
        wrapper = _loaded(tmp_path, monkeypatch, base=base, lr=0.1)
        wrapper.train()
        output = tmp_path / "output"
        tokens = [[3, 4, 5, 6, 7, 8, 9]]
        wrapper.model.eval()
        logits = np.array(wrapper.model(mx.array(tokens)))
        script = textwrap.dedent(f"""
            import json, sys
            from pathlib import Path
            import numpy as np
            import mlx.core as mx
            mx.set_default_device(mx.cpu if sys.argv[2] == "cpu" else mx.gpu)
            from mlx_lm.tuner.utils import load_adapters
            from mlx_lm.utils import load_model
            model, _ = load_model(Path({str(base)!r}))
            if sys.argv[1] == "adapter":
                model = load_adapters(model, {str(output)!r})
            model.eval()
            kinds = sorted({{(type(m).__name__, getattr(m, "bits", None),
                             getattr(m, "mode", None))
                            for _, m in model.named_modules() if "scales" in m}})
            out = model(mx.array({tokens!r}))
            print(json.dumps({{"logits": np.array(out).tolist(), "kinds": kinds}}))
        """)

        def run(mode):
            out = subprocess.run([sys.executable, "-c", script, mode, _device()],
                                 capture_output=True, text=True, check=True, timeout=300)
            return json.loads(out.stdout.strip().splitlines()[-1])

        reloaded, base_only = run("adapter"), run("base")
        assert reloaded["kinds"] == base_only["kinds"] == [
            ["QuantizedEmbedding", 4, "affine"], ["QuantizedLinear", 4, "affine"]]
        np.testing.assert_allclose(np.array(reloaded["logits"]), logits, rtol=1e-5, atol=1e-5)
        assert np.abs(np.array(base_only["logits"]) - logits).max() > 1e-3
