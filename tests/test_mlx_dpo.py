"""Native MLX DPO trainer: tokenization, batching, loss and the training loop.

The tokenization tests run everywhere with a character-level fake tokenizer.
The loss, log-prob and training-loop tests need ``mlx`` and skip without it.
"""

from __future__ import annotations

import json
import math
import re

import pytest


def _strip_ansi(text: str) -> str:
    """Rich styles numbers and wraps at the console width; compare plain words."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())


# --------------------------------------------------------------------------
# A character-level tokenizer with a deterministic chat template
# --------------------------------------------------------------------------

BOS = 1
EOS = 3


class FakeTokenizer:
    """One token per character, BOS prepended, ``\\x03`` as EOS."""

    eos_token = "\x03"
    eos_token_id = EOS
    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [BOS] + [ord(c) for c in text]

    def _render(self, messages, add_generation_prompt: bool) -> str:
        out = "".join(f"<{m['role']}>{m['content']}\x03" for m in messages)
        if add_generation_prompt:
            out += "<assistant>"
        return out

    def apply_chat_template(
        self, messages, add_generation_prompt=False, tokenize=True, return_dict=False
    ):
        text = self._render(messages, add_generation_prompt)
        if not tokenize:
            return text
        ids = self.encode(text)
        return {"input_ids": ids} if return_dict else ids


def _decode(ids: list[int]) -> str:
    return "".join(chr(i) for i in ids if i != BOS)


# --------------------------------------------------------------------------
# build_preference_example
# --------------------------------------------------------------------------


class TestBuildPreferenceExample:
    def _build(self, row, max_length=1024, max_prompt_length=512, mode="keep_start"):
        from soup_cli.trainer.mlx_dpo import build_preference_example

        return build_preference_example(
            row,
            FakeTokenizer(),
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            truncation_mode=mode,
        )

    def test_text_row_matches_trl_layout(self):
        """Text rows: prompt tokens, then prompt+completion+EOS sliced after them."""
        ex = self._build({"prompt": "Q:", "chosen": " yes", "rejected": " no"})
        assert ex.prompt_ids == [BOS] + [ord(c) for c in "Q:"]
        assert _decode(ex.chosen_ids) == " yes\x03"
        assert _decode(ex.rejected_ids) == " no\x03"

    def test_text_row_does_not_double_eos(self):
        ex = self._build({"prompt": "Q:", "chosen": " yes\x03", "rejected": " no"})
        assert _decode(ex.chosen_ids) == " yes\x03"

    def test_conversational_row_uses_chat_template(self):
        row = {
            "prompt": [{"role": "user", "content": "hi"}],
            "chosen": [{"role": "assistant", "content": "hello"}],
            "rejected": [{"role": "assistant", "content": "go away"}],
        }
        ex = self._build(row)
        assert _decode(ex.prompt_ids) == "<user>hi\x03<assistant>"
        # The completion is whatever follows the generation prompt.
        assert _decode(ex.chosen_ids) == "hello\x03"
        assert _decode(ex.rejected_ids) == "go away\x03"

    def test_implicit_prompt_in_full_conversations(self):
        """Soup's bundled DPO sample: a string prompt plus full chosen/rejected
        conversations that repeat the user turn. The shared leading turns are
        the prompt, not part of the supervised completion."""
        user = {"role": "user", "content": "hi"}
        row = {
            "prompt": "hi",
            "chosen": [user, {"role": "assistant", "content": "hello"}],
            "rejected": [user, {"role": "assistant", "content": "nope"}],
        }
        ex = self._build(row)
        assert _decode(ex.prompt_ids) == "<user>hi\x03<assistant>"
        assert _decode(ex.chosen_ids) == "hello\x03"
        assert _decode(ex.rejected_ids) == "nope\x03"

    def test_string_prompt_with_assistant_only_lists_becomes_user_turn(self):
        row = {
            "prompt": "hi",
            "chosen": [{"role": "assistant", "content": "hello"}],
            "rejected": [{"role": "assistant", "content": "nope"}],
        }
        ex = self._build(row)
        assert _decode(ex.prompt_ids) == "<user>hi\x03<assistant>"

    def test_prompt_list_repeated_inside_chosen_is_not_duplicated(self):
        user = {"role": "user", "content": "hi"}
        row = {
            "prompt": [user],
            "chosen": [user, {"role": "assistant", "content": "a"}],
            "rejected": [user, {"role": "assistant", "content": "b"}],
        }
        ex = self._build(row)
        assert _decode(ex.prompt_ids) == "<user>hi\x03<assistant>"
        assert _decode(ex.chosen_ids) == "a\x03"

    def test_mixed_string_and_list_completions_refused(self):
        from soup_cli.trainer.mlx_dpo import PreferenceRowError

        with pytest.raises(PreferenceRowError, match="chosen"):
            self._build(
                {
                    "prompt": "hi",
                    "chosen": "hello",
                    "rejected": [{"role": "assistant", "content": "x"}],
                }
            )

    def test_identical_conversations_flagged(self):
        """No turn differs, so there is no completion to prefer."""
        from soup_cli.trainer.mlx_dpo import IdenticalPairError

        user = {"role": "user", "content": "hi"}
        same = [user, {"role": "assistant", "content": "a"}]
        with pytest.raises(IdenticalPairError, match="identical"):
            self._build({"prompt": "hi", "chosen": same, "rejected": list(same)})

    def test_identical_text_completions_flagged(self):
        from soup_cli.trainer.mlx_dpo import IdenticalPairError

        with pytest.raises(IdenticalPairError):
            self._build({"prompt": "Q", "chosen": "a", "rejected": "a"})

    def test_template_that_is_not_prefix_stable_refused(self):
        """A completion that would be sliced at the wrong offset trains the
        wrong tokens; refuse instead of warning as TRL does."""
        from soup_cli.trainer.mlx_dpo import PreferenceRowError

        class Unstable(FakeTokenizer):
            def apply_chat_template(self, messages, add_generation_prompt=False, **kw):
                ids = super().apply_chat_template(messages, add_generation_prompt, **kw)
                if not add_generation_prompt:
                    ids = [ids[0], 999] + ids[1:]
                return ids

        from soup_cli.trainer.mlx_dpo import build_preference_example

        row = {
            "prompt": [{"role": "user", "content": "hi"}],
            "chosen": [{"role": "assistant", "content": "a"}],
            "rejected": [{"role": "assistant", "content": "b"}],
        }
        with pytest.raises(PreferenceRowError, match="prefix"):
            build_preference_example(
                row, Unstable(), max_length=64, max_prompt_length=32
            )

    def test_prompt_capped_keep_start(self):
        ex = self._build(
            {"prompt": "abcdefgh", "chosen": "x", "rejected": "y"},
            max_length=100,
            max_prompt_length=4,
        )
        assert ex.prompt_ids == [BOS, ord("a"), ord("b"), ord("c")]

    def test_prompt_capped_keep_end(self):
        ex = self._build(
            {"prompt": "abcdefgh", "chosen": "x", "rejected": "y"},
            max_length=100,
            max_prompt_length=3,
            mode="keep_end",
        )
        assert _decode(ex.prompt_ids) == "fgh"

    def test_completion_capped_to_remaining_length(self):
        ex = self._build(
            {"prompt": "ab", "chosen": "0123456789", "rejected": "xy"},
            max_length=6,
            max_prompt_length=3,
        )
        assert len(ex.prompt_ids) == 3
        assert _decode(ex.chosen_ids) == "012"
        assert _decode(ex.rejected_ids) == "xy\x03"

    def test_null_field_refused(self):
        from soup_cli.trainer.mlx_dpo import PreferenceRowError

        with pytest.raises(PreferenceRowError, match="rejected"):
            self._build({"prompt": "a", "chosen": "b", "rejected": None})


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


def _example(p, c, r):
    from soup_cli.trainer.mlx_dpo import PreferenceExample

    return PreferenceExample(prompt_ids=p, chosen_ids=c, rejected_ids=r)


class TestBatchPlan:
    def test_plan_is_reproducible_for_a_seed(self):
        from soup_cli.trainer.mlx_dpo import plan_batches

        a = plan_batches(10, batch_size=2, epochs=3, seed=42)
        b = plan_batches(10, batch_size=2, epochs=3, seed=42)
        assert a == b

    def test_different_seeds_shuffle_differently(self):
        from soup_cli.trainer.mlx_dpo import plan_batches

        assert plan_batches(50, 2, 1, seed=1) != plan_batches(50, 2, 1, seed=2)

    def test_each_epoch_covers_every_full_batch_once(self):
        from soup_cli.trainer.mlx_dpo import plan_batches

        plan = plan_batches(9, batch_size=2, epochs=2, seed=0)
        assert len(plan) == 8  # 4 full batches per epoch, the odd row dropped
        first_epoch = sorted(i for batch in plan[:4] for i in batch)
        assert len(set(first_epoch)) == 8

    def test_fewer_rows_than_batch_size_refused(self):
        from soup_cli.trainer.mlx_dpo import plan_batches

        with pytest.raises(ValueError, match="batch_size"):
            plan_batches(1, batch_size=2, epochs=1, seed=0)


class TestCollate:
    def test_chosen_then_rejected_with_completion_mask(self):
        from soup_cli.trainer.mlx_dpo import collate_pairs

        tokens, mask = collate_pairs(
            [_example([1, 2], [5, 6, 7], [8]), _example([1], [9], [4, 4])]
        )
        assert tokens == [
            [1, 2, 5, 6, 7],
            [1, 9, 0, 0, 0],
            [1, 2, 8, 0, 0],
            [1, 4, 4, 0, 0],
        ]
        assert mask == [
            [0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
        ]


# --------------------------------------------------------------------------
# Loss + log-probs (need mlx)
# --------------------------------------------------------------------------


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class TestDpoLoss:
    def test_matches_closed_form(self):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_dpo import dpo_loss

        pc, pr = mx.array([-1.0, -2.0]), mx.array([-3.0, -1.0])
        rc, rr = mx.array([-1.5, -2.0]), mx.array([-2.5, -1.5])
        beta = 0.1
        loss, metrics = dpo_loss(pc, pr, rc, rr, beta)

        expected = []
        for i in range(2):
            z = beta * ((pc[i] - rc[i]).item() - (pr[i] - rr[i]).item())
            expected.append(-math.log(_sigmoid(z)))
        assert loss.item() == pytest.approx(sum(expected) / 2, rel=1e-5)
        # chosen reward 0.05 vs rejected -0.05 for row 0; 0.0 vs 0.05 for row 1
        assert metrics["reward_accuracy"].item() == pytest.approx(0.5)
        assert metrics["reward_margin"].item() == pytest.approx(
            ((0.05 - -0.05) + (0.0 - 0.05)) / 2, rel=1e-5
        )

    def test_equal_policy_and_reference_is_log2(self):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_dpo import dpo_loss

        x = mx.array([-4.0, -7.0, -1.0])
        loss, _ = dpo_loss(x, x, x, x, 0.5)
        assert loss.item() == pytest.approx(math.log(2), rel=1e-6)

    def test_stable_for_large_margins(self):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_dpo import dpo_loss

        loss, _ = dpo_loss(
            mx.array([0.0]), mx.array([-1e5]), mx.array([0.0]), mx.array([0.0]), 1.0
        )
        assert math.isfinite(loss.item()) and loss.item() == pytest.approx(0.0, abs=1e-6)
        loss, _ = dpo_loss(
            mx.array([-1e5]), mx.array([0.0]), mx.array([0.0]), mx.array([0.0]), 1.0
        )
        assert loss.item() == pytest.approx(1e5, rel=1e-4)


class TestSequenceLogps:
    def test_sums_only_masked_target_tokens(self):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        from soup_cli.trainer.mlx_dpo import sequence_logps

        vocab = 5

        class Uniform(nn.Module):
            def __call__(self, x):
                return mx.zeros((*x.shape, vocab))

        tokens = mx.array([[1, 2, 3, 4], [1, 2, 0, 0]])
        mask = mx.array([[0, 0, 1, 1], [0, 1, 0, 0]])
        out = sequence_logps(Uniform(), tokens, mask)
        # Uniform logits: every target token has log-prob -log(5).
        assert out.tolist() == pytest.approx([-2 * math.log(vocab), -math.log(vocab)])


class TestChunkedLogps:
    """The memory path: backbone once, then the vocab head in checkpointed
    chunks starting at the first completion token. It must equal the plain
    full-model path, value and gradient."""

    def _batch(self):
        mx = pytest.importorskip("mlx.core")
        tokens = mx.array([[5, 6, 7, 8, 9, 10, 11], [5, 6, 12, 13, 0, 0, 0]])
        mask = mx.array([[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 0]])
        return tokens, mask

    @pytest.mark.parametrize("tied", [True, False])
    def test_matches_full_model(self, tied):
        pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_dpo import resolve_head_split, sequence_logps

        model = _tiny_llama(tie=tied)
        model.freeze()
        tokens, mask = self._batch()
        split = resolve_head_split(model)
        assert split is not None
        full = sequence_logps(model, tokens, mask)
        chunked = sequence_logps(model, tokens, mask, head_split=split, chunk_size=2)
        assert chunked.tolist() == pytest.approx(full.tolist(), rel=1e-5, abs=1e-5)

    def test_gradients_match_full_model(self):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        from mlx.utils import tree_flatten

        from soup_cli.trainer.mlx_dpo import resolve_head_split, sequence_logps

        model = _tiny_llama()
        # As in LoRA training: the vocab head (here the tied embedding) is
        # frozen, everything inside the backbone layers trains.
        model.freeze()
        model.model.layers[0].unfreeze()
        tokens, mask = self._batch()
        split = resolve_head_split(model)
        assert split is not None

        def loss(m, **kw):
            return sequence_logps(m, tokens, mask, **kw).sum()

        _, g_full = nn.value_and_grad(model, loss)(model)
        _, g_chunk = nn.value_and_grad(
            model, lambda m: loss(m, head_split=split, chunk_size=3)
        )(model)
        for (name, a), (_, b) in zip(tree_flatten(g_full), tree_flatten(g_chunk)):
            assert mx.allclose(a, b, rtol=1e-4, atol=1e-5).item(), name

    @pytest.mark.parametrize("tied", [True, False])
    def test_trainable_head_has_no_split(self, tied):
        """The checkpointed head sees its weights as constants, so a head that
        trains would lose its gradient: refuse the split, use the full path."""
        pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_dpo import resolve_head_split

        model = _tiny_llama(tie=tied)  # nothing frozen
        assert resolve_head_split(model) is None

    def test_unknown_architecture_has_no_split(self):
        nn = pytest.importorskip("mlx.nn")
        from soup_cli.trainer.mlx_dpo import resolve_head_split

        class Opaque(nn.Module):
            def __call__(self, x):
                return x

        assert resolve_head_split(Opaque()) is None


# --------------------------------------------------------------------------
# Config gate
# --------------------------------------------------------------------------

_MLX_DPO_YAML = """
base: mlx-community/Qwen2.5-0.5B-Instruct-4bit
task: dpo
backend: mlx
data:
  train: ./x.jsonl
  format: dpo
training:
  epochs: 1
  lr: 1e-6
output: ./output
"""


class TestConfig:
    def test_mlx_dpo_config_loads(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_MLX_DPO_YAML)
        assert cfg.task == "dpo" and cfg.backend == "mlx"

    def test_add_new_tokens_refused_on_mlx_dpo(self):
        """The vocab-expansion validator used to wave every non-SFT MLX task
        through, trusting the task gate to refuse it. DPO passes that gate now,
        so the field must be refused for MLX's own reason, not dropped."""
        from soup_cli.config.loader import load_config_from_string

        yaml_str = _MLX_DPO_YAML.replace(
            "  format: dpo\n", "  format: dpo\n  add_new_tokens: ['<x>']\n"
        )
        with pytest.raises(Exception, match="not applied on backend='mlx'"):
            load_config_from_string(yaml_str)

    def test_doctor_reports_mlx_dpo_gaps_but_not_the_seed(self):
        """`soup doctor --config` reads backend_support; the seed is honoured
        on MLX DPO, so only the genuinely ignored setting is reported."""
        from soup_cli.config.backend_support import check_config
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            _MLX_DPO_YAML.replace("  lr: 1e-6\n", "  lr: 1e-6\n  seed: 3\n  use_galore: true\n")
        )
        fields = {entry.field for entry in check_config(cfg)}
        assert fields == {"training.use_galore"}

    @pytest.mark.parametrize("task", ["ipo", "kto", "orpo", "simpo", "ppo"])
    def test_other_preference_tasks_still_refused(self, task):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = _MLX_DPO_YAML.replace("task: dpo", f"task: {task}")
        with pytest.raises(Exception, match="MLX backend"):
            load_config_from_string(yaml_str)


# --------------------------------------------------------------------------
# End-to-end on a tiny random model (needs mlx + mlx_lm)
# --------------------------------------------------------------------------


def _tiny_llama(tie: bool = True):
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import llama

    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=200,
        tie_word_embeddings=tie,
    )
    return llama.Model(args)


def _wrapper(tmp_path, monkeypatch, **training):
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.mlx_dpo import MLXDPOTrainerWrapper

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    extra = "".join(f"  {k}: {v}\n" for k, v in training.items())
    cfg = load_config_from_string(
        _MLX_DPO_YAML.replace("  lr: 1e-6\n", "  lr: 5e-3\n" + extra)
    )
    wrapper = MLXDPOTrainerWrapper(cfg)
    wrapper.model = _tiny_llama()
    wrapper.tokenizer = FakeTokenizer()
    rows = [
        {"prompt": f"q{i}:", "chosen": " good", "rejected": " bad"} for i in range(8)
    ]
    wrapper._dataset = {"train": rows, "val": rows[:2]}
    return wrapper


class TestTrainLoop:
    def test_tied_pairs_are_skipped_with_one_warning(self, tmp_path, monkeypatch, capsys):
        """UltraFeedback-style ties carry zero DPO gradient: skip, do not abort."""
        wrapper = _wrapper(tmp_path, monkeypatch, batch_size=2, seed=1)
        tie = {"prompt": "q:", "chosen": " same", "rejected": " same"}
        wrapper._dataset["train"] = wrapper._dataset["train"] + [tie, dict(tie)]
        result = wrapper.train()
        assert result["skipped_identical_pairs"] == 2
        out = _strip_ansi(capsys.readouterr().out)
        assert "skipped 2 dpo row(s) with identical chosen and rejected" in out

    def test_preference_is_learned_and_artifacts_written(self, tmp_path, monkeypatch):
        wrapper = _wrapper(
            tmp_path,
            monkeypatch,
            epochs=4,
            batch_size=2,
            gradient_accumulation_steps=1,
            warmup_ratio=0.0,
            seed=7,
        )
        result = wrapper.train()

        assert result["total_steps"] == 16
        assert result["final_loss"] < math.log(2)  # learned to prefer "good"
        assert result["val_reward_accuracy"] == pytest.approx(1.0)
        out = tmp_path / "output"
        assert (out / "adapters.safetensors").exists()
        meta = json.loads((out / "adapter_config.json").read_text())
        assert meta["fine_tune_type"] == "lora"
        assert meta["task"] == "dpo"
        assert meta["dpo_beta"] == pytest.approx(0.1)
        assert meta["seed"] == 7
        assert "lora_parameters" in meta

    def test_same_seed_reproduces_losses(self, tmp_path, monkeypatch):
        first = _wrapper(tmp_path / "a", monkeypatch, seed=3, batch_size=2)
        losses_a = first.train()["losses"]
        second = _wrapper(tmp_path / "b", monkeypatch, seed=3, batch_size=2)
        losses_b = second.train()["losses"]
        assert losses_a == pytest.approx(losses_b, rel=1e-5)

    def test_reference_logps_are_the_untrained_policy(self, tmp_path, monkeypatch):
        """LoRA B starts at zero, so the reference precomputed before the first
        update equals the base model; the first-step loss is exactly log 2."""
        wrapper = _wrapper(tmp_path, monkeypatch, batch_size=2, seed=0)
        result = wrapper.train()
        assert result["losses"][0] == pytest.approx(math.log(2), rel=1e-4)
