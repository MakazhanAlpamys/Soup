"""Real offline native teacher/student lifecycle and optimizer-window contracts."""

import json

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.llama import Model, ModelArgs  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from soup_cli.config.schema import SoupConfig  # noqa: E402
from soup_cli.trainer.mlx_distill_data import build_examples, collate  # noqa: E402


def checkpoint(path, *, width=32, layers=2, quantized=False):
    path.mkdir()
    mx.random.seed(7 + width)
    config = dict(
        model_type="llama", hidden_size=width, num_hidden_layers=layers,
        intermediate_size=width * 2, num_attention_heads=4, num_key_value_heads=4,
        rms_norm_eps=1e-5, vocab_size=32, tie_word_embeddings=False,
    )
    model = Model(ModelArgs(**config))
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
        config["quantization"] = {"group_size": 32, "bits": 4}
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config))
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate([
        "<unk>", "<eos>", "<pad>", "user", "assistant", "alpha", "beta", "gamma",
    ])}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>",
        chat_template=(
            "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} <eos> {% endfor %}"
            "{% if add_generation_prompt %}assistant {% endif %}"
        ),
    )
    tok.save_pretrained(path)
    return path


def config(student, teacher, output, **training):
    return SoupConfig(
        base=str(student), backend="mlx", task="distill", modality="text", output=str(output),
        data={"train": "unused.jsonl", "max_length": 64, "train_on_responses_only": False},
        training={
            "teacher_model": str(teacher), "quantization": "none", "epochs": 1, "batch_size": 1,
            "gradient_accumulation_steps": 1, "lr": 0.005, "scheduler": "constant",
            "warmup_ratio": 0, "weight_decay": 0, "seed": 11, "data_seed": 23,
            "logging_steps": 1, "save_steps": 2, "lora": {"r": 4, "alpha": 8, "dropout": 0},
            **training,
        },
    )


@pytest.fixture
def pair(tmp_path):
    return checkpoint(tmp_path / "student"), checkpoint(tmp_path / "teacher", width=64, layers=3)


def rows():
    return {"train": [{"text": text} for text in (
        "alpha beta", "gamma alpha beta gamma alpha", "beta gamma alpha",
    )], "val": [{"text": "alpha beta gamma"}]}


def test_real_multiturn_boundaries_padding_eos_and_truncation(pair):
    _, tok = load(str(pair[0]))
    tok.pad_token = tok.eos_token
    messages = [{"role": role, "content": text} for role, text in (
        ("user", "alpha"), ("assistant", "beta"),
        ("user", "gamma"), ("assistant", "alpha beta"),
    )]
    examples = build_examples(
        [{"messages": messages}, {"prompt": "gamma", "completion": "beta"}], tok,
        max_length=64, responses_only=True, split="train",
    )
    assert [label for label in examples[0].labels if label != -100] == [6, 1, 5, 6, 1]
    ids, labels, active = collate(examples, pad_id=1, max_length=64)
    assert active.size == 7
    assert labels[0, len(examples[0].tokens) - 1].item() == 1
    assert ids[0, -1].item() == 1 and labels[0, -1].item() == -100
    with pytest.raises(ValueError, match=r"val row 2: no causal-loss target"):
        build_examples(
            [{"text": "alpha beta"}, {"messages": messages}], tok,
            max_length=3, responses_only=True, split="val",
        )


@pytest.mark.parametrize("student_checkpoint", [False, True])
def test_updates_freeze_eval_save_reload_and_warm_start(pair, tmp_path, student_checkpoint):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    cfg = config(*pair, tmp_path / "out", gradient_checkpointing=student_checkpoint)
    trainer = MLXDistillTrainerWrapper(cfg)
    trainer.setup(rows())
    teacher = trainer.teacher
    before_teacher = dict(tree_flatten(teacher.parameters()))
    before_student = dict(tree_flatten(trainer.model.parameters()))
    trainable = dict(tree_flatten(trainer.model.trainable_parameters()))
    # Exercise real stochastic modules. train() must restore eval mode even if
    # a caller toggled the frozen teacher after setup.
    teacher.lm_head = nn.Sequential(nn.Dropout(0.8), teacher.lm_head)
    teacher.train()
    probe = mx.array([[5, 6, 7]])
    first, second = trainer._teacher_logits(probe), trainer._teacher_logits(probe)
    assert mx.array_equal(first, second).item()
    teacher.train()
    result = trainer.train()
    assert result["optimizer_updates"] == 3 and result["total_steps"] == 3
    assert result["eval_updates"] == [3]
    assert sum(entry["active_targets"] for entry in result["history"]) == 10
    assert result["history"][-1]["loss"] == pytest.approx(
        (result["history"][-1]["ce_loss"] + result["history"][-1]["kd_loss"]) / 2
    )
    assert teacher.training is False and not tree_flatten(teacher.trainable_parameters())
    after_teacher = dict(tree_flatten(teacher.parameters()))
    for name, value in before_teacher.items():
        actual = after_teacher.get(
            name, after_teacher.get(name.replace("lm_head.", "lm_head.layers.1.")),
        )
        assert mx.array_equal(value, actual).item(), name
    after_student = dict(tree_flatten(trainer.model.parameters()))
    for name, value in before_student.items():
        if name not in trainable:
            assert mx.array_equal(value, after_student[name]).item(), name
    assert len(trainable) == 8
    assert all(not mx.array_equal(value, after_student[name]).item()
               for name, value in trainable.items())
    tokens = mx.array([[5, 6, 7]])
    trainer.model.eval()
    expected = trainer.model(tokens)
    restored, _ = load(str(pair[0]), adapter_path=cfg.output)
    restored.eval()
    assert mx.allclose(expected, restored(tokens), atol=1e-5, rtol=1e-5).item()
    from mlx.utils import tree_unflatten
    from mlx_lm.tuner.lora import LoRALinear

    # Exercise the actual MLX-LM fusion operation on the saved/reloaded adapter.
    adapted = [(name, module) for name, module in restored.named_modules()
               if isinstance(module, LoRALinear)]
    assert len(adapted) == len(trainable) // 2
    restored.update_modules(tree_unflatten([
        (name, module.fuse(dequantize=True)) for name, module in adapted
    ]))
    assert not any(isinstance(module, LoRALinear) for _, module in restored.named_modules())
    assert mx.allclose(expected, restored(tokens), atol=2e-5, rtol=2e-5).item()
    metadata = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert metadata["task"] == "distill" and metadata["objective"]["ce_weight"] == 0.5
    assert metadata["tokenizer_identity"]["named_tokens"] == 8
    assert set(mx.load(str(tmp_path / "out" / "adapters.safetensors"))) == set(trainable)
    metrics = json.loads((tmp_path / "out" / "distillation_metrics.json").read_text())
    assert metrics["training"] == result["history"]
    assert metrics["validation"] == result["evaluation_history"]
    resumed = MLXDistillTrainerWrapper(config(*pair, tmp_path / "resumed"))
    resumed.setup(rows())
    resumed._load_checkpoint_weights(str(tmp_path / "out" / "adapters.safetensors"))
    resumed.model.eval()
    assert mx.allclose(expected, resumed.model(tokens), atol=1e-5, rtol=1e-5).item()


@pytest.mark.parametrize("accum", [2, 4])
def test_full_batch_and_unequal_token_accumulation_match_including_final_window(
    pair, tmp_path, accum,
):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    outputs = []
    for batch, accumulation in ((accum, 1), (1, accum)):
        cfg = config(*pair, tmp_path / f"out-{batch}")
        cfg.training.batch_size, cfg.training.gradient_accumulation_steps = batch, accumulation
        trainer = MLXDistillTrainerWrapper(cfg)
        trainer.setup(rows())
        result = trainer.train()
        outputs.append(dict(tree_flatten(trainer.model.trainable_parameters())))
        assert result["optimizer_updates"] == (2 if accum == 2 else 1)
    for name in outputs[0]:
        assert mx.allclose(outputs[0][name], outputs[1][name], atol=2e-6, rtol=2e-5).item(), name


@pytest.mark.parametrize("bad", ["teacher-nan", "student-inf", "interruption"])
def test_failure_prevents_update_and_final_success_artifact(pair, tmp_path, bad, monkeypatch):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    trainer = MLXDistillTrainerWrapper(config(*pair, tmp_path / "failed"))
    trainer.setup(rows())
    before = dict(tree_flatten(trainer.model.trainable_parameters()))
    if bad == "interruption":
        def interrupt(*args):
            raise KeyboardInterrupt
        monkeypatch.setattr(trainer, "_teacher_logits", interrupt)
        error, diagnostic = KeyboardInterrupt, None
    else:
        target = trainer.teacher if bad.startswith("teacher") else trainer.model
        target.lm_head.weight = target.lm_head.weight * float("nan" if "nan" in bad else "inf")
        error, diagnostic = ValueError, "teacher|student"
    with pytest.raises(error, match=diagnostic):
        trainer.train()
    assert all(mx.array_equal(value, dict(tree_flatten(trainer.model.trainable_parameters()))[name])
               .item() for name, value in before.items())
    assert not (tmp_path / "failed" / "adapters.safetensors").exists()
    assert trainer.teacher is None


@pytest.mark.parametrize("scheduler", ["constant", "constant_with_warmup", "linear", "cosine"])
def test_real_schedules_decay_clipping_and_checkpoint_equivalence(pair, tmp_path, scheduler):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper
    from soup_cli.trainer.mlx_optim import build_lr_schedule, plan_optimizer

    outputs = []
    for activation, chunk, recompute in ((False, None, False), (True, 3, True)):
        cfg = config(*pair, tmp_path / str(activation), scheduler=scheduler, warmup_ratio=0.5,
                     weight_decay=0.1, max_grad_norm=0.01, epochs=2,
                     gradient_checkpointing=activation, distill_chunk_size=chunk,
                     distill_checkpoint=recompute)
        trainer = MLXDistillTrainerWrapper(cfg)
        trainer.setup(rows())
        initial = dict(tree_flatten(trainer.model.trainable_parameters()))
        result = trainer.train()
        outputs.append(dict(tree_flatten(trainer.model.trainable_parameters())))
        schedule = build_lr_schedule(plan_optimizer(
            lr=cfg.training.lr, optimizer=cfg.training.optimizer, scheduler=scheduler,
            warmup_ratio=0.5, weight_decay=0.1, total_updates=6,
        ))
        assert [entry["lr"] for entry in result["history"]] == pytest.approx([
            schedule(mx.array(i)).item() for i in range(6)
        ])
        # Warmup starts at zero: inspect the actual first committed checkpoint
        # separately below rather than inferring no change from a finite loss.
        assert result["history"][0]["lr"] == 0
        assert any(not mx.array_equal(value, outputs[-1][name]).item()
                   for name, value in initial.items())
    for name in outputs[0]:
        assert mx.allclose(outputs[0][name], outputs[1][name], atol=3e-6, rtol=3e-5).item()


def test_zero_learning_rate_does_not_change_adapter_even_with_weight_decay(pair, tmp_path):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    cfg = config(*pair, tmp_path / "zero", warmup_ratio=0.5, save_steps=1, weight_decay=0.1)
    trainer = MLXDistillTrainerWrapper(cfg)
    trainer.setup(rows())
    initial = dict(tree_flatten(trainer.model.trainable_parameters()))
    trainer.train()
    first = mx.load(str(tmp_path / "zero" / "0000001_adapters.safetensors"))
    assert all(mx.array_equal(value, first[name]).item() for name, value in initial.items())


@pytest.mark.parametrize("bad", ["keys", "shape", "dtype", "nan"])
def test_warm_start_refuses_incompatible_tensors_before_changing_any_parameter(pair, tmp_path, bad):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    trainer = MLXDistillTrainerWrapper(config(*pair, tmp_path / "out"))
    trainer.setup(rows())
    original = dict(tree_flatten(trainer.model.trainable_parameters()))
    weights = dict(original)
    key = next(iter(weights))
    if bad == "keys":
        weights["other"] = weights.pop(key)
    elif bad == "shape":
        weights[key] = weights[key][:-1]
    elif bad == "dtype":
        weights[key] = weights[key].astype(mx.float16)
    else:
        weights[key] = weights[key] * float("nan")
    path = tmp_path / "bad.safetensors"
    mx.save_safetensors(str(path), weights)
    with pytest.raises(ValueError, match="warm-start tensor"):
        trainer._load_checkpoint_weights(str(path))
    assert all(mx.array_equal(value, dict(tree_flatten(trainer.model.trainable_parameters()))[name])
               .item() for name, value in original.items())


@pytest.mark.parametrize("side", ["student", "teacher"])
def test_actual_quantized_artifacts_are_refused_despite_none_setting(pair, tmp_path, side):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    quantized = checkpoint(tmp_path / "quantized", quantized=True)
    student, teacher = (quantized, pair[1]) if side == "student" else (pair[0], quantized)
    trainer = MLXDistillTrainerWrapper(config(student, teacher, tmp_path / "out"))
    with pytest.raises(ValueError, match=f"{side} artifact is actually quantized"):
        trainer.setup(rows())
    assert trainer.model is trainer.teacher is trainer.tokenizer is None


def test_clipping_reaches_real_optimizer_and_teacher_has_no_moments(pair, tmp_path, monkeypatch):
    import soup_cli.trainer.mlx_distill as module

    construct = module.build_optimizer
    norms, optimizers = [], []

    def build(plan):
        optimizer = construct(plan)
        update = optimizer.update

        def checked(model, gradients):
            norms.append(mx.sqrt(sum(mx.sum(value ** 2) for _, value in
                                     tree_flatten(gradients))).item())
            return update(model, gradients)

        monkeypatch.setattr(optimizer, "update", checked)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(module, "build_optimizer", build)
    trainer = module.MLXDistillTrainerWrapper(config(*pair, tmp_path / "clip", max_grad_norm=0.001))
    trainer.setup(rows())
    trainable = set(dict(tree_flatten(trainer.model.trainable_parameters())))
    trainer.train()
    assert len(norms) == 3
    assert norms == pytest.approx([0.001] * 3, rel=2e-4, abs=1e-6)
    moment_keys = {name.rsplit(".", 1)[0] for name, _ in tree_flatten(optimizers[0].state)
                   if name.endswith((".m", ".v"))}
    assert moment_keys == trainable


def test_actual_adamw_decay_changes_zero_gradient_a_at_first_update(pair, tmp_path):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    trainer = MLXDistillTrainerWrapper(config(*pair, tmp_path / "decay", weight_decay=0.2))
    trainer.setup({"train": rows()["train"][:1]})
    initial = dict(tree_flatten(trainer.model.trainable_parameters()))
    trainer.train()
    final = dict(tree_flatten(trainer.model.trainable_parameters()))
    for name, value in initial.items():
        if name.endswith("lora_a"):
            assert mx.allclose(final[name], value * (1 - 0.005 * 0.2), atol=1e-7).item()
            assert not mx.array_equal(final[name], value).item()


def test_nonfinite_accumulated_gradient_aborts_before_update(pair, tmp_path, monkeypatch):
    import soup_cli.trainer.mlx_distill as module

    original = module.token_loss_sums

    def poison(student, *args, **kwargs):
        total, metrics = original(student, *args, **kwargs)
        # Finite forward value with an infinite derivative, a real AD operation.
        return total + mx.sqrt(mx.sum(student) * 0), metrics

    monkeypatch.setattr(module, "token_loss_sums", poison)
    trainer = module.MLXDistillTrainerWrapper(config(*pair, tmp_path / "nan-grad"))
    trainer.setup(rows())
    before = dict(tree_flatten(trainer.model.trainable_parameters()))
    with pytest.raises(ValueError, match="student_gradients"):
        trainer.train()
    assert all(mx.array_equal(value, dict(tree_flatten(trainer.model.trainable_parameters()))[name])
               .item() for name, value in before.items())
    assert not (tmp_path / "nan-grad" / "adapters.safetensors").exists()


def test_real_offline_cli_train_evaluate_reload_and_auto_warm_start(pair, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = config(*pair, tmp_path / "out", batch_size=2, eval_steps=1)
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows()["train"] * 2) + "\n")
    cfg.data.train, cfg.data.val_split = str(data), 0.2
    path = tmp_path / "soup.json"
    path.write_text(cfg.model_dump_json())
    outcome = CliRunner().invoke(app, ["train", "--config", str(path), "--yes"])
    assert outcome.exit_code == 0, (outcome.output, repr(outcome.exception))
    assert "KD update" in outcome.output and "CE=" in outcome.output
    assert "Step:" in outcome.output
    restored, _ = load(str(pair[0]), adapter_path=cfg.output)
    assert mx.all(mx.isfinite(restored(mx.array([[5, 6, 7]])))).item()
    receipt = json.loads((tmp_path / "out" / "distillation_run.json").read_text())
    assert receipt["status"] == "complete" and receipt["optimizer_updates"] > 0
    resumed = CliRunner().invoke(app, ["train", "--config", str(path), "--yes", "--resume", "auto"])
    assert resumed.exit_code == 0, (resumed.output, repr(resumed.exception))
    assert "weights-only warm start" in resumed.output


@pytest.mark.parametrize("failure", ["teacher-load", "tokenizer", "empty-train", "empty-val"])
def test_setup_failures_release_both_models_before_any_training(pair, tmp_path, failure):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    cfg = config(*pair, tmp_path / "out")
    dataset = rows()
    if failure == "teacher-load":
        cfg.training.teacher_model = str(tmp_path / "missing")
    elif failure == "tokenizer":
        path = pair[1] / "tokenizer.json"
        changed = json.loads(path.read_text())
        changed["model"]["vocab"].update(alpha=6, beta=5)
        path.write_text(json.dumps(changed))
    elif failure == "empty-train":
        dataset["train"] = []
    else:
        dataset["val"] = [{"messages": [{"role": "user", "content": "alpha"}]}]
        cfg.data.train_on_responses_only = True
    trainer = MLXDistillTrainerWrapper(cfg)
    with pytest.raises((ValueError, FileNotFoundError), match="tokenizer|missing|empty|val row 1"):
        trainer.setup(dataset)
    assert trainer.model is trainer.teacher is trainer.tokenizer is None


def test_native_seed_controls_initialization_and_data_seed_controls_real_order(pair, tmp_path):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    initializations, orders = [], []
    for index, (seed, data_seed) in enumerate(((11, 23), (11, 23), (42, 24))):
        trainer = MLXDistillTrainerWrapper(config(
            *pair, tmp_path / str(index), seed=seed, data_seed=data_seed,
        ))
        trainer.setup(rows())
        initializations.append(dict(tree_flatten(trainer.model.trainable_parameters())))
        counts = [row.active_count for row in trainer._train_rows]
        import numpy as np

        expected = [counts[i] for i in np.random.default_rng(data_seed).permutation(len(counts))]
        result = trainer.train()
        orders.append([entry["active_targets"] for entry in result["history"]])
        assert orders[-1] == expected
    assert orders[0] == orders[1] and orders[0] != orders[2]
    for name in initializations[0]:
        assert mx.array_equal(initializations[0][name], initializations[1][name]).item()
        if name.endswith("lora_a"):
            assert not mx.array_equal(initializations[0][name], initializations[2][name]).item()


def test_student_activation_checkpoint_preserves_seeded_dropout_updates(pair, tmp_path):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    outputs, losses = [], []
    for enabled in (False, True):
        cfg = config(*pair, tmp_path / str(enabled), gradient_checkpointing=enabled,
                     lora={"r": 4, "alpha": 8, "dropout": 0.3})
        trainer = MLXDistillTrainerWrapper(cfg)
        trainer.setup(rows())
        result = trainer.train()
        outputs.append(dict(tree_flatten(trainer.model.trainable_parameters())))
        losses.append([entry["loss"] for entry in result["history"]])
    assert losses[0] == pytest.approx(losses[1], abs=2e-6)
    for name in outputs[0]:
        assert mx.allclose(outputs[0][name], outputs[1][name], atol=2e-6, rtol=2e-5).item()


def test_quantized_expert_module_cannot_hide_outside_standard_linear_type(
    pair, tmp_path, monkeypatch,
):
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    import soup_cli.utils.mlx as loader
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    original = loader.load_mlx_model

    def with_quantized_experts(path, **kwargs):
        model, tokenizer = original(path, **kwargs)
        if path == str(pair[1]):
            model.experts = QuantizedSwitchLinear(32, 32, 2, group_size=32, bits=4)
        return model, tokenizer

    monkeypatch.setattr(loader, "load_mlx_model", with_quantized_experts)
    trainer = MLXDistillTrainerWrapper(config(*pair, tmp_path / "out"))
    with pytest.raises(ValueError, match="teacher artifact is actually quantized"):
        trainer.setup(rows())
    assert trainer.teacher is trainer.model is None


def test_interruption_retains_prior_numbered_checkpoint_but_never_final_success(
    pair, tmp_path, monkeypatch,
):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    trainer = MLXDistillTrainerWrapper(config(*pair, tmp_path / "interrupted", save_steps=1))
    trainer.setup(rows())
    original = trainer._teacher_logits
    calls = 0

    def interrupt_after_update(tokens):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(tokens)

    monkeypatch.setattr(trainer, "_teacher_logits", interrupt_after_update)
    with pytest.raises(KeyboardInterrupt):
        trainer.train()
    output = tmp_path / "interrupted"
    receipt = json.loads((output / "distillation_run.json").read_text())
    assert receipt == {"status": "interrupted", "optimizer_updates": 1}
    assert not (output / "adapters.safetensors").exists()
    saved = output / "0000001_adapters.safetensors"
    assert saved.exists()
    restored = MLXDistillTrainerWrapper(config(*pair, tmp_path / "restored"))
    restored.setup(rows())
    restored._load_checkpoint_weights(str(saved))
    assert all(mx.array_equal(value, mx.load(str(saved))[name]).item()
               for name, value in tree_flatten(restored.model.trainable_parameters()))


@pytest.mark.parametrize("previous", [False, True])
def test_failed_final_publication_restores_previous_files_or_leaves_no_final_artifact(
    pair, tmp_path, monkeypatch, previous,
):
    from pathlib import Path

    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    output = tmp_path / "out"
    names = ("adapters.safetensors", "adapter_config.json", "distillation_metrics.json")
    if previous:
        earlier = MLXDistillTrainerWrapper(config(*pair, output, save_steps=0))
        earlier.setup(rows())
        earlier.train()
    before = {name: (output / name).read_bytes() for name in names} if previous else {}
    trainer = MLXDistillTrainerWrapper(config(*pair, output, save_steps=0, lr=0.01))
    trainer.setup(rows())
    replace = Path.replace

    def fail_metrics_promotion(path, destination):
        if path.name == ".distillation_metrics.tmp.json":
            raise OSError("injected final metrics promotion failure")
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_metrics_promotion)
    with pytest.raises(OSError, match="injected final metrics"):
        trainer.train()
    for name in names:
        if previous:
            assert (output / name).read_bytes() == before[name]
        else:
            assert not (output / name).exists()
    assert json.loads((output / "distillation_run.json").read_text())["status"] == "failed"


def test_failed_checkpoint_write_does_not_destroy_previous_recoverable_checkpoint(
    pair, tmp_path, monkeypatch,
):
    from pathlib import Path

    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    output = tmp_path / "out"
    earlier = MLXDistillTrainerWrapper(config(*pair, output, save_steps=1))
    earlier.setup(rows())
    earlier.train()
    checkpoint = output / "0000001_adapters.safetensors"
    before = checkpoint.read_bytes()
    trainer = MLXDistillTrainerWrapper(config(*pair, output, save_steps=1, lr=0.01))
    trainer.setup(rows())

    def partial_write(path, weights):
        Path(path).write_bytes(b"partial write before injected I/O failure")
        raise OSError("injected checkpoint write failure")

    monkeypatch.setattr(mx, "save_safetensors", partial_write)
    with pytest.raises(OSError, match="injected checkpoint write"):
        trainer.train()
    assert checkpoint.read_bytes() == before
