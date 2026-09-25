"""#1223 — every trainer that receives a validation split evaluates it.

``data.val_split: 0.1`` moved 10% of the rows out of training and every task
wrapper passed them on as ``eval_dataset``, but none set ``eval_strategy``, whose
Hugging Face default is ``"no"``. ``Trainer.evaluate()`` never ran, so the rows
were used for nothing and #713's validation-loss sinks never got a value.

Driven for real on CPU, per task, because a per-task omission is the failure
mode: `setup()` is called on a tiny checkpoint, the schedule the trainer actually
received is asserted, and then the run trains, which is the only way to know the
evaluation pass works on that trainer. Two did not: the embedding trainer's
default ``prediction_step`` called ``model(**inputs)`` with its own batch keys,
and TRL's BCO evaluation fails before its first training step (which a schedule
never reaches).

The config and loader half is ``test_issue1223_eval_steps_config.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.conftest import strip_ansi


def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "true")


def _plain(text: str | None) -> str:
    """ANSI-stripped and whitespace-collapsed: Rich colours and wraps CLI output."""
    return " ".join(strip_ansi(text).split())


# --------------------------------------------------------------------------
# fixtures -- a real, tiny checkpoint on disk. Duplicated from
# test_issue353_seed_every_task.py deliberately, as the suites around it do, so
# this file stays runnable on its own.
# --------------------------------------------------------------------------
def _tiny_llama_dir(root: Path, vocab: int = 64, hidden: int = 64) -> str:
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = root / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))

    words = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "good", "answer", "bad"):
        words[word] = len(words)
    tok = Tokenizer(models.WordLevel(vocab=words, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    ).save_pretrained(str(weights))
    return str(weights)


def _chat_rows(n):
    return [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "good answer"},
            ]
        }
        for _ in range(n)
    ]


def _pref_rows(n):
    return [{"prompt": "hi", "chosen": " good answer", "rejected": " bad"} for _ in range(n)]


def _kto_rows(n):
    return [{"prompt": "hi", "completion": " good answer", "label": i % 2 == 0} for i in range(n)]


def _text_rows(n):
    return [{"text": "hello world good answer"} for _ in range(n)]


def _label_rows(n):
    return [{"text": "hello world", "label": i % 2} for i in range(n)]


def _embedding_rows(n):
    return [{"anchor": "hello world", "positive": "good answer"} for _ in range(n)]


def _prm_rows(n):
    return [
        {"prompt": "hi", "completions": ["good", "answer"], "labels": [1.0, 0.0]}
        for _ in range(n)
    ]


def _prompt_rows(n):
    return [{"prompt": "hi"} for _ in range(n)]


#: task -> (module, wrapper class, rows, extra `training:` keys). The issue's
#: list minus grpo, which is generation-based and has its own tests below.
_EVALUATING = {
    "sft": ("soup_cli.trainer.sft", "SFTTrainerWrapper", _chat_rows, {}),
    "pretrain": ("soup_cli.trainer.pretrain", "PretrainTrainerWrapper", _text_rows, {}),
    "dpo": ("soup_cli.trainer.dpo", "DPOTrainerWrapper", _pref_rows, {}),
    "ipo": ("soup_cli.trainer.ipo", "IPOTrainerWrapper", _pref_rows, {}),
    "kto": ("soup_cli.trainer.kto", "KTOTrainerWrapper", _kto_rows, {}),
    "orpo": ("soup_cli.trainer.orpo", "ORPOTrainerWrapper", _pref_rows, {}),
    "simpo": ("soup_cli.trainer.simpo", "SimPOTrainerWrapper", _pref_rows, {}),
    "bco": ("soup_cli.trainer.bco", "BCOTrainerWrapper", _pref_rows, {}),
    "distill": ("soup_cli.trainer.distill", "DistillTrainerWrapper", _chat_rows, {}),
    # batch 1: the tiny checkpoint has no pad id, and a sequence-classification
    # head refuses batches > 1 without one -- in training as in evaluation.
    "classifier": (
        "soup_cli.trainer.classifier",
        "ClassifierTrainerWrapper",
        _label_rows,
        {"num_labels": 2, "batch_size": 1},
    ),
    "embedding": ("soup_cli.trainer.embedding", "EmbeddingTrainerWrapper", _embedding_rows, {}),
    "reward_model": (
        "soup_cli.trainer.reward_model",
        "RewardModelTrainerWrapper",
        _pref_rows,
        {},
    ),
    "prm": ("soup_cli.trainer.prm", "PRMTrainerWrapper", _prm_rows, {}),
    "moe_lora_routing": (
        "soup_cli.trainer.mole_routing",
        "MoleRoutingTrainerWrapper",
        _text_rows,
        {"mole_task_adapters": ["./adapter_a", "./adapter_b"]},
    ),
    "asr": ("soup_cli.trainer.asr", "AsrTrainerWrapper", None, {}),
}

#: The wrappers the issue lists as receiving a validation split.
_ISSUE_TASKS = frozenset({
    "sft", "pretrain", "dpo", "ipo", "kto", "orpo", "simpo", "bco", "grpo",
    "distill", "classifier", "embedding", "reward_model", "prm",
    "moe_lora_routing", "asr",
})

#: These build their Trainer inside `train()`, so the schedule can only be
#: read once the run has happened.
_BUILDS_IN_TRAIN = frozenset({"prm", "moe_lora_routing"})

_TINY_WHISPER = "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"
_TRAIN_ROWS = 8
_VAL_ROWS = 2


def _write_sine_wav(path: Path, seconds: float = 0.5, sr: int = 16000) -> None:
    import numpy as np
    import soundfile as sf

    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype("float32"), sr)


def _asr_rows(root: Path, n: int, prefix: str) -> list[dict]:
    rows = []
    for i in range(n):
        clip = root / f"{prefix}{i}.wav"
        _write_sine_wav(clip)
        rows.append({"audio": str(clip), "text": "hello world"})
    return rows


def _tiny_whisper_dir(root: Path) -> str:
    """The hub's tiny-random Whisper, resized so it can actually train.

    As shipped it takes 60 mel frames, but the feature extractor pads every clip
    to 30 s (3000 frames), and its vocabulary stops below the ids the tokenizer
    gives Whisper's own special tokens, so a training step on it fails before
    any loss. Same config (d_model 16, 2 layers) with room for 1500 encoder
    positions and the tokenizer's full vocabulary, fresh random weights, the hub
    copy's processor.
    """
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(_TINY_WHISPER)
    config = WhisperConfig.from_pretrained(_TINY_WHISPER)
    config.max_source_positions = 1500
    config.vocab_size = len(processor.tokenizer)
    torch.manual_seed(7)
    target = root / "whisper"
    WhisperForConditionalGeneration(config).save_pretrained(str(target))
    processor.save_pretrained(str(target))
    return str(target)


def _write_task_adapters(root: Path, weights: str) -> None:
    """Two tiny LoRA adapters for the MoLE gate to route between."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    for name in ("adapter_a", "adapter_b"):
        base = AutoModelForCausalLM.from_pretrained(weights)
        peft_model = get_peft_model(
            base,
            LoraConfig(
                r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
            ),
        )
        peft_model.save_pretrained(str(root / name))


def _cfg(task: str, base: str, out: Path, **training_over):
    import yaml

    from soup_cli.config.loader import load_config_from_string

    training = {
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "quantization": "none",
        "epochs": 1,
        "logging_steps": 1,
        "save_steps": 1000,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    if task in _EVALUATING:
        training.update(_EVALUATING[task][3])
    if task == "distill":
        training["teacher_model"] = base
    if task == "grpo":
        training.update({"num_generations": 2, "reward_fn": "format"})
    if task in {"asr", "prm"}:
        # prm fine-tunes every base parameter and refuses a lora block; asr's
        # LoRA is its own opt-in (asr_lora).
        training.pop("lora")
    training.update(training_over)
    data = {"train": "train.jsonl", "max_length": 64}
    if task in {"sft", "dpo", "ipo", "kto", "orpo", "simpo", "bco", "distill", "grpo"}:
        data["chat_template"] = "chatml"
    if task == "asr":
        data["format"] = "asr"
    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": task,
                "backend": "transformers",
                "modality": "text",
                "data": data,
                "training": training,
                "output": str(out),
            }
        )
    )


def _build(tmp_path, monkeypatch, task, *, n_train=_TRAIN_ROWS, n_val=_VAL_ROWS, **over):
    """A real wrapper over a real tiny checkpoint, `setup()` called with the
    train/val shape the loader hands over when `data.val_split` > 0."""
    import importlib

    monkeypatch.chdir(tmp_path)
    if task == "asr":
        pytest.importorskip("soundfile")
        try:  # the tiny Whisper config and processor come from the hub
            base = _tiny_whisper_dir(tmp_path)
        except OSError as exc:
            pytest.skip(f"tiny whisper unavailable: {exc}")
        dataset = {"train": _asr_rows(tmp_path, n_train, "t")}
        if n_val:
            dataset["val"] = _asr_rows(tmp_path, n_val, "v")
    else:
        base = _tiny_llama_dir(tmp_path)
        rows = _EVALUATING[task][2] if task in _EVALUATING else _prompt_rows
        dataset = {"train": rows(n_train)}
        if n_val:
            dataset["val"] = rows(n_val)
    if task == "moe_lora_routing":
        _write_task_adapters(tmp_path, base)
    if task in _EVALUATING:
        module, cls_name = _EVALUATING[task][:2]
    else:
        module, cls_name = "soup_cli.trainer.grpo", "GRPOTrainerWrapper"
    cfg = _cfg(task, base, tmp_path / "out", **over)
    wrapper = getattr(importlib.import_module(module), cls_name)(cfg, device="cpu")
    wrapper.setup(dataset)
    return wrapper


def _hf_trainer(wrapper):
    """`embedding` composes HF's Trainer instead of subclassing it."""
    trainer = wrapper.trainer
    return getattr(trainer, "_trainer", trainer)


def _strategy(args) -> str:
    return getattr(args.eval_strategy, "value", args.eval_strategy)


def _eval_entries(trainer) -> list[dict]:
    return [e for e in trainer.state.log_history if "eval_loss" in e]


def _run(wrapper, task):
    """Train, and return the HF trainer that ran."""
    if task in _BUILDS_IN_TRAIN:
        wrapper.train()
    else:
        _hf_trainer(wrapper).train()
    return _hf_trainer(wrapper)


# ==========================================================================
# the acceptance matrix
# ==========================================================================
class TestEveryWrapperEvaluatesItsSplit:
    """Whenever the trainer receives a non-empty validation split, its schedule
    is not ``"no"``, the pass runs at the train batch, and it logs a finite
    ``eval_loss``. ``grpo`` is the generation-based exception, pinned below."""

    @pytest.mark.parametrize("task", sorted(_EVALUATING))
    def test_the_split_is_scheduled_and_evaluated(self, tmp_path, monkeypatch, task):
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, task)
        if task not in _BUILDS_IN_TRAIN:
            trainer = _hf_trainer(wrapper)
            assert trainer.eval_dataset is not None and len(trainer.eval_dataset) > 0
            assert _strategy(trainer.args) == "epoch", (
                f"{task}: a non-empty validation split with eval_strategy="
                f"{_strategy(trainer.args)!r} is withheld and never evaluated"
            )
        trainer = _run(wrapper, task)
        args = trainer.args
        assert _strategy(args) == "epoch", task
        assert args.per_device_eval_batch_size == args.per_device_train_batch_size, (
            f"{task}: evaluation runs at batch {args.per_device_eval_batch_size}, "
            f"training at {args.per_device_train_batch_size}"
        )
        evals = _eval_entries(trainer)
        assert len(evals) == 1, f"{task}: one epoch should evaluate once, got {evals}"
        assert math.isfinite(evals[0]["eval_loss"]), (task, evals)

    def test_the_matrix_covers_the_issue_list(self):
        assert set(_EVALUATING) | {"grpo"} == _ISSUE_TASKS


class TestGrpoIsGenerationBased:
    """Documented as not evaluating by default, and the loader withholds nothing
    for it, so no row is spent on an evaluation that never runs."""

    def test_the_loader_withholds_nothing(self, tmp_path, monkeypatch):
        _requires_train_extra()
        from soup_cli.data.loader import load_dataset
        from soup_cli.trainer.grpo import GRPOTrainerWrapper
        from soup_cli.utils.eval_schedule import loader_data_config

        monkeypatch.chdir(tmp_path)
        weights = _tiny_llama_dir(tmp_path)
        (tmp_path / "train.jsonl").write_text(
            "\n".join(json.dumps(r) for r in _chat_rows(10)) + "\n", encoding="utf-8"
        )
        cfg = _cfg("grpo", weights, tmp_path / "out")
        assert cfg.data.val_split == pytest.approx(0.1)
        # Exactly what `soup train` hands the wrapper for this config.
        dataset = load_dataset(loader_data_config(cfg), preserve_source_columns=True)
        assert len(dataset["train"]) == 10
        assert not dataset.get("val")

        wrapper = GRPOTrainerWrapper(cfg, device="cpu")
        wrapper.setup(dataset)
        assert len(wrapper.trainer.train_dataset) == 10
        assert wrapper.trainer.eval_dataset is None
        assert _strategy(wrapper.trainer.args) == "no"

    def test_a_split_it_is_handed_anyway_is_not_evaluated_unasked(self, tmp_path, monkeypatch):
        """An HF-hub dataset's own validation split still arrives; it is not
        withheld from training, and grpo does not generate over it unasked."""
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, "grpo")
        assert len(wrapper.trainer.eval_dataset) == _VAL_ROWS
        assert _strategy(wrapper.trainer.args) == "no"

    def test_eval_steps_opts_in_and_the_pass_runs(self, tmp_path, monkeypatch):
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, "grpo", eval_steps=2)
        args = wrapper.trainer.args
        assert (_strategy(args), args.eval_steps) == ("steps", 2)
        assert args.per_device_eval_batch_size == args.per_device_train_batch_size
        args.max_steps = 2
        wrapper.trainer.train()
        evals = _eval_entries(wrapper.trainer)
        assert [e["step"] for e in evals] == [2]
        assert math.isfinite(evals[0]["eval_loss"])


class TestGrpoEvalBatchHoldsWholeGroups:
    """TRL evaluates ``num_generations`` completions per held-out prompt and
    refuses an eval batch that splits a group. Batch 3 with 2 generations
    trains (gradient accumulation 2 makes the generation batch 6) but raised at
    setup, on one process, as soon as ``eval_steps`` was set."""

    _NOTE = (
        "grpo evaluates at batch 2, not the train batch 3: TRL's evaluation needs "
        "whole groups of num_generations=2 completions."
    )

    def _build3(self, tmp_path, monkeypatch, **over):
        return _build(
            tmp_path, monkeypatch, "grpo", batch_size=3, gradient_accumulation_steps=2, **over
        )

    def test_the_eval_batch_rounds_down_to_whole_groups(self, tmp_path, monkeypatch, capsys):
        _requires_train_extra()
        wrapper = self._build3(tmp_path, monkeypatch, eval_steps=1)
        args = wrapper.trainer.args
        assert (args.per_device_train_batch_size, args.per_device_eval_batch_size) == (3, 2)
        assert self._NOTE in _plain(capsys.readouterr().out)
        args.max_steps = 1
        wrapper.trainer.train()
        assert [e["step"] for e in _eval_entries(wrapper.trainer)] == [1]

    def test_no_note_when_nothing_is_evaluated(self, tmp_path, monkeypatch, capsys):
        _requires_train_extra()
        self._build3(tmp_path, monkeypatch)
        assert "grpo evaluates at batch" not in _plain(capsys.readouterr().out)

    def test_no_note_when_the_train_batch_already_holds_whole_groups(
        self, tmp_path, monkeypatch, capsys
    ):
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, "grpo", batch_size=4, eval_steps=1)
        assert wrapper.trainer.args.per_device_eval_batch_size == 4
        assert "grpo evaluates at batch" not in _plain(capsys.readouterr().out)


class TestLayerStreamingEvaluates:
    """``stream_layers: true`` is not special-cased: the streamed model runs the
    evaluation pass on CPU, through the streamed decoder layers, and the loss it
    reports is the resident model's (same checkpoint, same rows)."""

    def test_a_streamed_run_evaluates_its_split(self, tmp_path, monkeypatch):
        _requires_train_extra()
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
        (tmp_path / "streamed").mkdir()
        (tmp_path / "resident").mkdir()
        streamed = _build(
            tmp_path / "streamed",
            monkeypatch,
            "sft",
            n_train=4,
            batch_size=1,
            stream_layers=True,
            gradient_checkpointing=True,
        )
        assert _strategy(streamed.trainer.args) == "epoch"
        loads = streamed._stream_runtime.pool.loads
        streamed_loss = streamed.trainer.evaluate()["eval_loss"]
        assert streamed._stream_runtime.pool.loads > loads, "no layer was streamed"

        resident = _build(tmp_path / "resident", monkeypatch, "sft", n_train=4, batch_size=1)
        resident_loss = resident.trainer.evaluate()["eval_loss"]
        assert streamed_loss == pytest.approx(resident_loss, rel=1e-4)

        streamed.trainer.train()
        evals = _eval_entries(streamed.trainer)
        assert len(evals) == 1 and math.isfinite(evals[0]["eval_loss"]), evals


class TestNoSplitNoEvaluation:
    """The control: with no validation rows nothing is scheduled, so a run with
    ``val_split: 0`` behaves exactly as it did."""

    @pytest.mark.parametrize("task", ["sft", "dpo"])
    def test_no_rows_no_schedule(self, tmp_path, monkeypatch, task):
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, task, n_val=0)
        trainer = _hf_trainer(wrapper)
        assert trainer.eval_dataset is None
        assert _strategy(trainer.args) == "no"
        trainer.train()
        assert _eval_entries(trainer) == []


# ==========================================================================
# the evaluation batch is the resolved train batch
# ==========================================================================
class TestTheEvalBatchIsTheTrainBatch:
    """HF's default eval batch is 8. A card sized to train at 1 runs out of
    memory at 8, so the pass must run at whatever the train batch resolved to,
    including after ``batch_size: auto``."""

    @pytest.mark.parametrize("batch", [1, 3])
    def test_an_explicit_batch(self, tmp_path, monkeypatch, batch):
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, "sft", batch_size=batch)
        assert wrapper.trainer.args.per_device_train_batch_size == batch
        assert wrapper.trainer.args.per_device_eval_batch_size == batch

    def test_auto_resolves_before_the_eval_batch_is_taken(self, tmp_path, monkeypatch):
        _requires_train_extra()
        import soup_cli.utils.batch_probe as batch_probe

        monkeypatch.setattr(batch_probe, "pick_batch_size", lambda **_: 3)
        wrapper = _build(tmp_path, monkeypatch, "sft", batch_size="auto")
        assert wrapper.trainer.args.per_device_train_batch_size == 3
        assert wrapper.trainer.args.per_device_eval_batch_size == 3

    def test_auto_on_a_preference_wrapper(self, tmp_path, monkeypatch):
        """DPO halves its estimate for the chosen/rejected pair; the eval batch
        follows the halved value, not the estimate and not HF's 8."""
        _requires_train_extra()
        import soup_cli.trainer.dpo as dpo

        monkeypatch.setattr(dpo, "estimate_batch_size", lambda **_: 6)
        wrapper = _build(tmp_path, monkeypatch, "dpo", batch_size="auto")
        assert wrapper.trainer.args.per_device_train_batch_size == 3
        assert wrapper.trainer.args.per_device_eval_batch_size == 3


# ==========================================================================
# the schedule variants
# ==========================================================================
class TestScheduleVariants:
    """16 train rows at batch 2 make 8 micro-batches per epoch. Steps counted
    are optimizer steps, i.e. after gradient accumulation."""

    @pytest.mark.parametrize(
        ("training", "expected_steps"),
        [
            pytest.param({"eval_steps": 3}, [3, 6, 8], id="eval_steps-below-total"),
            # Larger than the run: Hugging Face still evaluates at the last step.
            pytest.param({"eval_steps": 100}, [8], id="eval_steps-above-total"),
            pytest.param(
                {"eval_steps": 2, "gradient_accumulation_steps": 2}, [2, 4], id="grad-accum"
            ),
            pytest.param(
                {"epochs": 2, "gradient_accumulation_steps": 2}, [4, 8], id="epoch-ends"
            ),
            pytest.param(
                {"epochs": 2, "eval_steps": 3}, [3, 6, 9, 12, 15, 16], id="epochs-with-steps"
            ),
        ],
    )
    def test_evaluations_land_where_the_schedule_says(
        self, tmp_path, monkeypatch, training, expected_steps
    ):
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, "sft", n_train=16, **training)
        wrapper.trainer.train()
        steps = [e["step"] for e in _eval_entries(wrapper.trainer)]
        assert steps == expected_steps


# ==========================================================================
# end to end: `soup train` records the validation loss
# ==========================================================================
class TestSoupTrainRecordsValLoss:
    """The issue's acceptance run, through the CLI: default ``val_split``, a real
    tiny SFT run on CPU, and every sink #713 added receiving a value."""

    _ROWS = 20

    def _run_cli(self, tmp_path, monkeypatch, *, val_split: str = ""):
        import transformers
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        weights = _tiny_llama_dir(tmp_path)
        rows = [
            {
                "messages": [
                    {"role": "user", "content": f"hi {i}"},
                    {"role": "assistant", "content": "good answer"},
                ]
            }
            for i in range(self._ROWS)
        ]
        (tmp_path / "chat.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        (tmp_path / "soup.yaml").write_text(
            f"base: {json.dumps(weights)}\n"
            "task: sft\n"
            "data:\n"
            "  train: chat.jsonl\n"
            "  format: chatml\n"
            "  chat_template: chatml\n"
            "  max_length: 64\n"
            f"{val_split}"
            "training:\n"
            "  epochs: 2\n"
            "  batch_size: 2\n"
            "  gradient_accumulation_steps: 1\n"
            "  logging_steps: 1\n"
            "  save_steps: 1000\n"
            "  quantization: none\n"
            "  lora:\n"
            "    r: 4\n"
            "    alpha: 8\n"
            "    target_modules: [q_proj, v_proj]\n"
            "output: ./out\n",
            encoding="utf-8",
        )
        seen = {"evaluate_calls": 0, "log_history": None}
        real_train = transformers.Trainer.train
        real_evaluate = transformers.Trainer.evaluate

        def train(self, *args, **kwargs):
            out = real_train(self, *args, **kwargs)
            seen["log_history"] = list(self.state.log_history)
            return out

        def evaluate(self, *args, **kwargs):
            seen["evaluate_calls"] += 1
            return real_evaluate(self, *args, **kwargs)

        monkeypatch.setattr(transformers.Trainer, "train", train)
        monkeypatch.setattr(transformers.Trainer, "evaluate", evaluate)
        result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--yes"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        runs = tracker.list_runs()
        assert len(runs) == 1, runs
        metrics = tracker.get_metrics(runs[0]["run_id"])
        return seen, metrics, _plain(result.output)

    def test_the_default_split_is_evaluated_and_recorded(self, tmp_path, monkeypatch):
        _requires_train_extra()
        seen, metrics, out = self._run_cli(tmp_path, monkeypatch)
        assert f"Loaded: {self._ROWS - 2} train samples" in out
        # Two epochs, evaluated at the end of each.
        assert seen["evaluate_calls"] == 2
        evals = [e for e in seen["log_history"] if "eval_loss" in e]
        assert len(evals) == 2
        assert all(math.isfinite(e["eval_loss"]) for e in evals)
        recorded = [row["val_loss"] for row in metrics if row["val_loss"] is not None]
        assert len(recorded) == 2
        assert recorded == pytest.approx([e["eval_loss"] for e in evals])
        assert "Val loss" in out

    def test_control_val_split_zero_evaluates_nothing_and_withholds_nothing(
        self, tmp_path, monkeypatch
    ):
        _requires_train_extra()
        seen, metrics, out = self._run_cli(tmp_path, monkeypatch, val_split="  val_split: 0\n")
        assert f"Loaded: {self._ROWS} train samples" in out
        assert seen["evaluate_calls"] == 0
        assert not [e for e in seen["log_history"] if "eval_loss" in e]
        assert metrics, "the run logged no metrics at all"
        assert all(row["val_loss"] is None for row in metrics)
        assert "Val loss" not in out


# ==========================================================================
# the guard that survives the next wrapper being added
# ==========================================================================
def _trainer_sources() -> dict[str, str]:
    import importlib.util

    spec = importlib.util.find_spec("soup_cli.trainer")
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(Path(spec.origin).parent.glob("*.py"))
    }


class TestEveryWrapperUsesTheHelper:
    """A wrapper that hands its trainer an ``eval_dataset`` must build its
    schedule with ``training_eval_kwargs``, or it withholds rows it never
    evaluates. Derived from the source, so a new wrapper is covered unasked."""

    @staticmethod
    def _passing_an_eval_dataset() -> list[str]:
        return [
            name
            for name, source in _trainer_sources().items()
            if "eval_dataset=" in source or '"eval_dataset":' in source
        ]

    def test_the_derivation_found_the_wrappers(self):
        """Non-vacuity: an empty derivation would pass the test below."""
        assert {"sft", "grpo", "prm", "asr", "embedding"} <= set(
            self._passing_an_eval_dataset()
        )

    def test_each_one_threads_the_helper(self):
        sources = _trainer_sources()
        missing = [
            name
            for name in self._passing_an_eval_dataset()
            if "training_eval_kwargs(" not in sources[name]
        ]
        assert not missing, (
            f"soup_cli.trainer.{missing} pass an eval_dataset without "
            "training_eval_kwargs(), so the split is withheld and never evaluated"
        )
