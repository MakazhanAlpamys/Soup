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

    # orpo has its own MLX trainer since #2004 (tests/test_mlx_orpo.py).
    @pytest.mark.parametrize("task", ["ipo", "kto", "simpo", "ppo"])
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


# --------------------------------------------------------------------------
# Review of #1846: the preference-row contract is the loader's (#1866)
# --------------------------------------------------------------------------

_U = {"role": "user", "content": "hi"}
_A = {"role": "assistant", "content": "a"}
_B = {"role": "assistant", "content": "b"}


def _build_row(row, tokenizer=None, max_length=1024, max_prompt_length=512):
    from soup_cli.trainer.mlx_dpo import build_preference_example

    return build_preference_example(
        row, tokenizer or FakeTokenizer(),
        max_length=max_length, max_prompt_length=max_prompt_length,
    )


class TestTheLoaderLayoutsAreAccepted:
    @pytest.mark.parametrize("name", ["dpo_sample.jsonl", "chat_preferences.jsonl"])
    def test_every_bundled_preference_row_tokenizes(self, name):
        """Through the real loader converter, as `load_dataset` hands rows over."""
        from pathlib import Path

        from soup_cli.data.formats import format_to_messages

        path = Path(__file__).resolve().parents[1] / "examples" / "data" / name
        raws = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        assert raws
        for raw in raws:
            row = format_to_messages(raw, "dpo")
            assert row is not None
            ex = _build_row(row, max_length=4096, max_prompt_length=2048)
            user = raw["chosen"][0]["content"]
            assert _decode(ex.prompt_ids) == f"<user>{user}\x03<assistant>"
            assert _decode(ex.chosen_ids) == raw["chosen"][-1]["content"] + "\x03"
            assert _decode(ex.rejected_ids) == raw["rejected"][-1]["content"] + "\x03"

    def test_message_list_prompt_with_string_answers(self):
        """Review item 1: each string answer is one assistant turn."""
        ex = _build_row({"prompt": [_U], "chosen": "a", "rejected": "b"})
        assert _decode(ex.prompt_ids) == "<user>hi\x03<assistant>"
        assert _decode(ex.chosen_ids) == "a\x03"
        assert _decode(ex.rejected_ids) == "b\x03"

    def test_one_message_object_is_read_as_a_list_of_one(self):
        """Review item 2: a dict prompt is a message, never `str(dict)`."""
        ex = _build_row({"prompt": _U, "chosen": [_A], "rejected": [_B]})
        assert _decode(ex.prompt_ids) == "<user>hi\x03<assistant>"
        assert "role" not in _decode(ex.prompt_ids)

    @pytest.mark.parametrize("prompt", [7, True, 1.5])
    def test_a_non_text_prompt_is_refused_by_name(self, prompt):
        from soup_cli.trainer.mlx_dpo import PreferenceRowError

        with pytest.raises(PreferenceRowError, match="dpo.prompt"):
            _build_row({"prompt": prompt, "chosen": [_A], "rejected": [_B]})

    def test_a_conversation_that_continues_past_the_other(self):
        """Review item 3: the shorter side's last assistant turn is its answer."""
        u2 = {"role": "user", "content": "more"}
        ex = _build_row({"prompt": "hi", "chosen": [_U, _A], "rejected": [_U, _A, u2, _B]})
        assert _decode(ex.prompt_ids) == "<user>hi\x03<assistant>"
        assert _decode(ex.chosen_ids) == "a\x03"
        assert _decode(ex.rejected_ids).startswith("a\x03<user>more")

    def test_a_side_with_no_answer_is_a_data_error_not_a_tokenizer_one(self):
        from soup_cli.trainer.mlx_dpo import PreferenceRowError, PrefixUnstableError

        with pytest.raises(PreferenceRowError, match="no chosen answer") as excinfo:
            _build_row({"prompt": "hi", "chosen": [_U], "rejected": [_U, _B]})
        assert not isinstance(excinfo.value, PrefixUnstableError)

    def test_no_prompt_uses_the_shared_turns(self):
        """Review item 4: rows handed straight to setup() read like the loader's."""
        ex = _build_row({"chosen": [_U, _A], "rejected": [_U, _B]})
        assert _decode(ex.prompt_ids) == "<user>hi\x03<assistant>"

    def test_messages_column_is_the_prompt(self):
        ex = _build_row({"messages": [_U], "chosen": [_A], "rejected": [_B]})
        assert _decode(ex.prompt_ids) == "<user>hi\x03<assistant>"


class _Unstable(FakeTokenizer):
    def apply_chat_template(self, messages, add_generation_prompt=False, **kw):
        ids = super().apply_chat_template(messages, add_generation_prompt, **kw)
        if not add_generation_prompt and messages[0]["content"] == "bad":
            ids = [ids[0], 999] + ids[1:]
        return ids


def _bare_wrapper(**training):
    """A wrapper with a tokenizer and no model: enough for `_examples`."""
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.mlx_dpo import MLXDPOTrainerWrapper

    extra = "".join(f"  {k}: {v}\n" for k, v in training.items())
    yaml_str = _MLX_DPO_YAML.replace("  lr: 1e-6\n", "  lr: 1e-6\n" + extra)
    return MLXDPOTrainerWrapper(load_config_from_string(yaml_str))


class TestRowsTheTemplateCannotSlice:
    def test_skipped_with_a_count_and_the_first_offender(self, capsys):
        wrapper = _bare_wrapper()
        wrapper.tokenizer = _Unstable()
        bad = {"prompt": [{"role": "user", "content": "bad"}], "chosen": [_A], "rejected": [_B]}
        good = {"prompt": [_U], "chosen": [_A], "rejected": [_B]}
        examples, ties = wrapper._examples([good, bad, good, bad])
        assert (len(examples), ties) == (2, 0)
        out = _strip_ansi(capsys.readouterr().out)
        assert "skipped 2 train row(s)" in out and "first: row 1" in out

    def test_a_malformed_row_still_stops_with_its_index(self):
        from soup_cli.trainer.mlx_dpo import PreferenceRowError

        wrapper = _bare_wrapper()
        wrapper.tokenizer = FakeTokenizer()
        good = {"prompt": "q", "chosen": "a", "rejected": "b"}
        with pytest.raises(PreferenceRowError, match="dpo train row 1"):
            wrapper._examples([good, {"prompt": "q", "chosen": "a", "rejected": [_B]}])


class TestPromptCapIsWired:
    def test_examples_cap_the_prompt_at_half_max_length(self):
        wrapper = _bare_wrapper()
        wrapper.config.data.max_length = 40
        wrapper.tokenizer = FakeTokenizer()
        long_prompt = "x" * 100
        (example,), _ = wrapper._examples([{"prompt": long_prompt, "chosen": "a", "rejected": "b"}])
        assert len(example.prompt_ids) == 20
        assert len(example.prompt_ids) + len(example.chosen_ids) <= 40


class TestConfigRefusals:
    def _load(self, extra_top="", extra_training=""):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = _MLX_DPO_YAML.replace("backend: mlx\n", "backend: mlx\n" + extra_top)
        yaml_str = yaml_str.replace("  lr: 1e-6\n", "  lr: 1e-6\n" + extra_training)
        return load_config_from_string(yaml_str)

    def test_lora_rank_zero_is_refused(self):
        with pytest.raises(Exception, match="lora.r must be at least 1"):
            self._load(extra_training="  lora: {r: 0}\n")

    @pytest.mark.parametrize("modality", ["vision", "audio"])
    def test_non_text_modality_is_refused(self, modality):
        with pytest.raises(Exception, match=f"modality='{modality}'"):
            self._load(extra_top=f"modality: {modality}\n")

    def test_loraplus_is_still_refused_on_mlx_dpo(self):
        with pytest.raises(Exception, match="loraplus_lr_ratio"):
            self._load(extra_training="  loraplus_lr_ratio: 16\n")

    def test_the_ignored_settings_are_printed(self, monkeypatch):
        from io import StringIO

        from rich.console import Console

        from soup_cli.trainer import mlx_dpo

        buffer = StringIO()
        monkeypatch.setattr(mlx_dpo, "console", Console(file=buffer, width=400))
        cfg = self._load(extra_training="  use_galore: true\n")
        mlx_dpo.MLXDPOTrainerWrapper(cfg)._check_unsupported()
        assert "MLX backend ignores: GaLore" in _strip_ansi(buffer.getvalue())


# --------------------------------------------------------------------------
# sequence_logps scores the NEXT token
# --------------------------------------------------------------------------


class _Oracle:
    """Logits that put all mass on ``tokens[t + 1]`` at position ``t``
    (``shift=1``) or on ``tokens[t]`` (``shift=0``: copies its input)."""

    def __new__(cls, tokens, vocab, shift):
        nn = pytest.importorskip("mlx.nn")
        mx = pytest.importorskip("mlx.core")

        class Backbone(nn.Module):
            def __call__(self, x):
                width = x.shape[1]
                targets = tokens[:, shift : shift + width]
                return mx.eye(vocab)[targets]

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()
                self.lm_head = nn.Linear(vocab, vocab, bias=False)
                self.lm_head.weight = mx.eye(vocab) * 50.0
                self.args = type("Args", (), {"tie_word_embeddings": False})()

            def __call__(self, x):
                return self.lm_head(self.model(x))

        model = Model()
        model.freeze()
        return model


class TestTargetAlignment:
    @pytest.mark.parametrize("chunked", [False, True])
    def test_the_real_continuation_scores_near_zero(self, chunked):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_dpo import resolve_head_split, sequence_logps

        tokens = mx.array([[1, 5, 9, 3, 7, 2]])
        mask = mx.array([[0, 0, 1, 1, 1, 1]])
        right = _Oracle(tokens, 16, shift=1)
        copier = _Oracle(tokens, 16, shift=0)
        kw = {}
        if chunked:
            assert resolve_head_split(right) is not None
            kw = {"head_split": resolve_head_split(right), "chunk_size": 2}
        assert sequence_logps(right, tokens, mask, **kw).item() == pytest.approx(0.0, abs=1e-3)
        kw_copy = {"head_split": resolve_head_split(copier), "chunk_size": 2} if chunked else {}
        assert sequence_logps(copier, tokens, mask, **kw_copy).item() < -100


# --------------------------------------------------------------------------
# Train-loop guarantees the docs state
# --------------------------------------------------------------------------


def _seeded_wrapper(tmp_path, monkeypatch, model_seed=0, **training):
    mx = pytest.importorskip("mlx.core")
    wrapper = _wrapper(tmp_path, monkeypatch, **training)
    mx.random.seed(model_seed)
    wrapper.model = _tiny_llama()
    return wrapper


class TestResume:
    def test_reference_is_taken_before_the_checkpoint_loads(self, tmp_path, monkeypatch):
        from soup_cli.trainer import mlx_dpo

        first = _seeded_wrapper(tmp_path / "a", monkeypatch, batch_size=2, epochs=3, seed=1)
        first.train()
        checkpoint = tmp_path / "a" / "output" / "adapters.safetensors"
        assert checkpoint.exists()

        order = []
        real_ref = mlx_dpo.MLXDPOTrainerWrapper._reference_logps
        real_load = mlx_dpo.MLXDPOTrainerWrapper._load_checkpoint_weights
        monkeypatch.setattr(mlx_dpo.MLXDPOTrainerWrapper, "_reference_logps",
                            lambda self, *a: order.append("reference") or real_ref(self, *a))
        monkeypatch.setattr(mlx_dpo.MLXDPOTrainerWrapper, "_load_checkpoint_weights",
                            lambda self, *a: order.append("load") or real_load(self, *a))
        resumed = _seeded_wrapper(tmp_path / "b", monkeypatch, batch_size=2, seed=1)
        result = resumed.train(resume_from_checkpoint=str(checkpoint))

        assert order[-1] == "load" and "reference" in order[:-1]
        # The policy starts trained, the reference is the base model, so the
        # first loss is already below log 2 (it is exactly log 2 from scratch).
        assert result["losses"][0] < math.log(2) - 1e-3


class TestGradientAccumulation:
    def test_two_micro_batches_of_one_equal_one_batch_of_two(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")
        from mlx.utils import tree_flatten

        from soup_cli.trainer import mlx_dpo

        def run(path, batch_size, accum, plan):
            monkeypatch.setattr(mlx_dpo, "plan_batches", lambda *a, **k: [list(b) for b in plan])
            wrapper = _seeded_wrapper(
                path, monkeypatch, batch_size=batch_size, gradient_accumulation_steps=accum,
                optimizer="sgd", warmup_ratio=0.0, scheduler="constant", seed=4,
            )
            wrapper.config.training.lora.dropout = 0.0
            result = wrapper.train()
            return result, dict(tree_flatten(wrapper.model.trainable_parameters()))

        one, w_one = run(tmp_path / "a", 2, 1, [[0, 1]])
        two, w_two = run(tmp_path / "b", 1, 2, [[0], [1]])
        assert len(one["losses"]) == len(two["losses"]) == 1
        for name in w_one:
            assert mx.allclose(w_one[name], w_two[name], atol=1e-6).item(), name

    def test_updates_are_counted_per_accumulation_group(self, tmp_path, monkeypatch):
        wrapper = _seeded_wrapper(tmp_path, monkeypatch, batch_size=2,
                                  gradient_accumulation_steps=2, seed=1)
        result = wrapper.train()
        assert result["total_steps"] == 4  # 8 rows / batch 2, one epoch
        assert len(result["losses"]) == 2  # one loss per optimizer update


class TestDataSeed:
    def test_data_seed_owns_the_batch_order(self, tmp_path, monkeypatch):
        from soup_cli.trainer import mlx_dpo

        plans = []
        real = mlx_dpo.plan_batches

        def recording(*args, **kwargs):
            plan = real(*args, **kwargs)
            plans.append(plan)
            return plan

        monkeypatch.setattr(mlx_dpo, "plan_batches", recording)
        for path, training in (("a", {"seed": 1, "data_seed": 2}),
                               ("b", {"seed": 3, "data_seed": 2}),
                               ("c", {"seed": 3})):
            _seeded_wrapper(tmp_path / path, monkeypatch, batch_size=2, epochs=2,
                            **training).train()
        assert plans[0] == plans[1]
        assert plans[1] != plans[2]


class TestValidation:
    def test_one_val_row_with_batch_two_is_evaluated(self, tmp_path, monkeypatch):
        """The #1851 shape: a partial last chunk, not a crash."""
        wrapper = _seeded_wrapper(tmp_path, monkeypatch, batch_size=2, seed=1)
        wrapper._dataset["val"] = wrapper._dataset["val"][:1]
        result = wrapper.train()
        assert "val_loss" in result and math.isfinite(result["val_loss"])

    def test_val_loss_is_the_mean_over_rows_not_over_chunks(self, tmp_path, monkeypatch):
        pytest.importorskip("mlx.core")
        from soup_cli.trainer import mlx_dpo

        refs = []
        real = mlx_dpo.MLXDPOTrainerWrapper._reference_logps
        monkeypatch.setattr(mlx_dpo.MLXDPOTrainerWrapper, "_reference_logps",
                            lambda self, ex, bs: refs.append(real(self, ex, bs)) or refs[-1])
        wrapper = _seeded_wrapper(tmp_path, monkeypatch, batch_size=2, epochs=3, seed=2)
        # Three rows, one reversed: chunks of 2 + 1 with different losses.
        wrapper._dataset["val"] = [
            {"prompt": "v0:", "chosen": " good", "rejected": " bad"},
            {"prompt": "v1:", "chosen": " good", "rejected": " bad"},
            {"prompt": "v2:", "chosen": " bad", "rejected": " good"},
        ]
        result = wrapper.train()

        examples, _ = wrapper._examples(wrapper._dataset["val"], "val")
        ref_c, ref_r = refs[1]
        wrapper.model.eval()
        per_row = []
        for i, example in enumerate(examples):
            tokens, mask = wrapper._to_arrays([example])
            lp = mlx_dpo.sequence_logps(wrapper.model, tokens, mask, head_split=wrapper._head_split)
            loss, _ = mlx_dpo.dpo_loss(lp[:1], lp[1:], ref_c[i : i + 1], ref_r[i : i + 1], 0.1)
            per_row.append(loss.item())
        assert result["val_loss"] == pytest.approx(sum(per_row) / 3, rel=1e-4)


class TestCheckpoints:
    def test_save_steps_files_are_found_by_resume_auto(self, tmp_path, monkeypatch):
        from soup_cli.commands.train import _resolve_mlx_checkpoint

        wrapper = _seeded_wrapper(tmp_path, monkeypatch, batch_size=2, epochs=2,
                                  save_steps=2, seed=1)
        wrapper.train()
        out = tmp_path / "output"
        numbered = sorted(p.name for p in out.glob("*_adapters.safetensors"))
        assert numbered == [f"{n:07d}_adapters.safetensors" for n in (2, 4, 6, 8)]
        resolved = _resolve_mlx_checkpoint("auto", str(out))
        assert resolved is not None and resolved.endswith("0000008_adapters.safetensors")


class TestHeadSplitFallback:
    def test_a_mismatching_head_falls_back_with_a_warning(self, tmp_path, monkeypatch):
        from io import StringIO

        from rich.console import Console

        from soup_cli.trainer import mlx_dpo

        real = mlx_dpo.resolve_head_split

        def scaled(model):
            split = real(model)
            backbone, head = split
            return backbone, (lambda h: head(h) * 2.0)

        monkeypatch.setattr(mlx_dpo, "resolve_head_split", scaled)
        buffer = StringIO()
        monkeypatch.setattr(mlx_dpo, "console", Console(file=buffer, width=400))
        wrapper = _seeded_wrapper(tmp_path, monkeypatch, batch_size=2, seed=1)
        result = wrapper.train()
        assert wrapper._head_split is None
        assert "using the full-vocabulary path" in _strip_ansi(buffer.getvalue())
        assert result["losses"][0] == pytest.approx(math.log(2), rel=1e-4)


class TestRefusalsAndReporting:
    def test_too_few_pairs_after_ties_names_batch_size_and_ties(self, tmp_path, monkeypatch):
        wrapper = _seeded_wrapper(tmp_path, monkeypatch, batch_size=4, seed=1)
        tie = {"prompt": "q:", "chosen": " same", "rejected": " same"}
        wrapper._dataset["train"] = wrapper._dataset["train"][:3] + [tie, dict(tie)]
        with pytest.raises(ValueError, match=r"3 training pair.*batch_size \(4\).*2 tie"):
            wrapper.train()

    def test_metrics_reach_the_live_stream_with_a_display(self, tmp_path, monkeypatch):
        from soup_cli.utils import train_event_buffer

        events = []
        monkeypatch.setattr(train_event_buffer, "push_train_event", events.append)

        class Display:
            def start(self, total):
                pass

            def update(self, **kw):
                pass

            def stop(self):
                pass

        wrapper = _seeded_wrapper(tmp_path, monkeypatch, batch_size=2, seed=1)
        wrapper.train(display=Display())
        assert events and all(e.type == "metric" and e.loss is not None for e in events)
        assert any(e.val_loss is not None for e in events)
