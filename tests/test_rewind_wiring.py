"""`training.rewind_log` reaches both SFT trainers.

The recorder modules are tested on their own (test_rewind_hf / test_rewind_mlx);
this file pins the wiring: the field's default, that the HF wrapper builds the
recording trainer and a log whose fingerprint `soup rewind` can check, that a real
training run fills it, and that an MLX run whose recorder cannot start still trains.
The end-to-end MLX run on real mlx-lm is the live proof in the PR, not a unit test.
"""

from __future__ import annotations

import math
import sys

import pytest

from soup_cli.config.schema import TrainingConfig
from soup_cli.monitoring.rewind_log import RewindLog, dataset_fingerprint, read_rewind_log


def test_rewind_log_defaults_on_and_can_be_turned_off():
    assert TrainingConfig().rewind_log is True
    assert TrainingConfig(rewind_log=False).rewind_log is False


# --------------------------------------------------------------- transformers --


def _hf_wrapper(tmp_path, monkeypatch, **training):
    from tests.test_issue341_seed_and_fullft import _requires_train_extra, _wrapper

    _requires_train_extra()
    monkeypatch.setenv("WANDB_DISABLED", "true")
    return _wrapper(tmp_path, monkeypatch, **training)


def test_hf_setup_attaches_the_recorder_and_writes_a_checkable_header(tmp_path, monkeypatch):
    wrapper, dataset = _hf_wrapper(tmp_path, monkeypatch)
    wrapper.setup(dataset)

    assert getattr(type(wrapper.trainer), "soup_rewind", False) is True
    run = read_rewind_log(tmp_path / "out" / RewindLog.FILENAME)
    assert run.header["backend"] == "transformers"
    assert run.header["n_rows"] == len(dataset["train"])
    assert run.header["dataset_fingerprint"] == dataset_fingerprint(dataset["train"])


def test_hf_rewind_log_false_builds_the_plain_trainer_and_no_file(tmp_path, monkeypatch):
    wrapper, dataset = _hf_wrapper(tmp_path, monkeypatch, rewind_log=False)
    wrapper.setup(dataset)

    assert getattr(type(wrapper.trainer), "soup_rewind", False) is False
    assert not (tmp_path / "out" / RewindLog.FILENAME).exists()


def test_hf_training_run_records_every_row_once(tmp_path, monkeypatch):
    wrapper, dataset = _hf_wrapper(tmp_path, monkeypatch)
    wrapper.setup(dataset)
    wrapper.trainer.train()
    wrapper._report_rewind()

    run = read_rewind_log(tmp_path / "out" / RewindLog.FILENAME)
    rows = [r for batch in run.batches for r in batch.rows]
    assert sorted(rows) == list(range(len(dataset["train"])))
    assert all(math.isfinite(loss) for batch in run.batches for loss in batch.row_loss)
    assert run.steps() == list(range(1, len(dataset["train"]) + 1))


def test_hf_rewind_preserves_fresh_and_resumed_training_row_order(
    tmp_path,
    monkeypatch,
    capsys,
    request,
):
    from tests.test_issue341_seed_and_fullft import _requires_train_extra, _tiny_llama_dir

    # The MLX smoke lane has no Torch. Guard before the fixture imports it.
    _requires_train_extra()
    request.getfixturevalue("aten_half_matmuls")

    from transformers import TrainerCallback

    class SaveResumePoints(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in (4, 5):
                control.should_save = True

    base = _tiny_llama_dir(tmp_path, n_layers=1)

    def run(name, *, checkpoint=None, save=False, **training):
        root = tmp_path / name
        root.mkdir()
        wrapper, dataset = _hf_wrapper(
            root,
            monkeypatch,
            base=base,
            epochs=2,
            gradient_accumulation_steps=2,
            seed=123,
            data_seed=456,
            **training,
        )
        wrapper.setup(dataset)
        trainer = wrapper.trainer
        rows_by_tokens = {
            tuple(row["input_ids"]): index for index, row in enumerate(trainer.train_dataset)
        }
        assert len(rows_by_tokens) == len(dataset["train"]) == 8
        seen = []
        training_step = trainer.training_step

        def capture_row(model, inputs, *args, **kwargs):
            assert inputs["input_ids"].shape[0] == 1
            tokens = inputs["input_ids"][0][inputs["attention_mask"][0].bool()]
            seen.append(rows_by_tokens[tuple(tokens.tolist())])
            return training_step(model, inputs, *args, **kwargs)

        monkeypatch.setattr(trainer, "training_step", capture_row)
        if save:
            trainer.add_callback(SaveResumePoints())
        try:
            trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        finally:
            wrapper._report_rewind()
        log_path = root / "out" / RewindLog.FILENAME
        recorded = []
        if log_path.is_file():
            recorded = [row for batch in read_rewind_log(log_path).batches for row in batch.rows]
        return seen, recorded

    fresh, recorded = run("recorded", save=True)
    plain, plain_records = run("plain", rewind_log=False)
    assert recorded == fresh
    assert plain_records == []
    for rows in (fresh, plain):
        assert len(rows) == 16
        assert sorted(rows[:8]) == sorted(rows[8:]) == list(range(8))

    checkpoints = tmp_path / "recorded" / "out"
    mismatches = []
    for step in (4, 5):
        checkpoint = checkpoints / f"checkpoint-{step}"
        assert checkpoint.is_dir()
        capsys.readouterr()
        resumed, resumed_records = run(f"resume-{step}", checkpoint=checkpoint)
        assert resumed_records == []
        assert "resumed runs" in capsys.readouterr().out
        expected = fresh[2 * step :]
        if resumed != expected:
            mismatches.append(f"checkpoint-{step}: {resumed} != {expected}")
    if fresh != plain:
        mismatches.append(f"fresh recorder on vs off: {fresh} != {plain}")
    assert not mismatches, "\n".join(mismatches)


# ------------------------------------------------------------------------ mlx --


def _mlx_ready(tmp_path, monkeypatch, **training):
    from tests.test_issue634_mlx_resume import _FakeMlxModel, _install_fake_mlx, _mlx_wrapper

    _install_fake_mlx(monkeypatch)
    wrapper = _mlx_wrapper(tmp_path, epochs=1, lr=1e-4, batch_size=1, **training)
    wrapper.model = _FakeMlxModel()
    wrapper.tokenizer = object()
    wrapper._dataset = {"train": [{"text": "hi"}], "val": []}
    return wrapper


def _recording_console(monkeypatch):
    from rich.console import Console

    import soup_cli.trainer.mlx_sft as mlx_sft

    console = Console(record=True, force_terminal=False, width=10_000, soft_wrap=True)
    monkeypatch.setattr(mlx_sft, "console", console)
    return console


def test_mlx_recorder_that_cannot_start_does_not_stop_training(tmp_path, monkeypatch):
    """The fake mlx has no ``mx.distributed``: setup fails, training still runs."""
    wrapper = _mlx_ready(tmp_path, monkeypatch)
    console = _recording_console(monkeypatch)
    trainer_module = sys.modules["mlx_lm.tuner.trainer"]
    real_train = trainer_module.train
    calls = []

    def _spy(**kwargs):
        calls.append(kwargs)
        return real_train(**kwargs)

    monkeypatch.setattr(trainer_module, "train", _spy)

    wrapper.train()

    assert len(calls) == 1
    assert "Rewind log off:" in console.export_text()
    assert "loss" not in calls[0], "a recorder that failed to start must not swap the loss"


def test_mlx_rewind_log_false_never_tries_to_start(tmp_path, monkeypatch):
    wrapper = _mlx_ready(tmp_path, monkeypatch, rewind_log=False)
    console = _recording_console(monkeypatch)

    wrapper.train()

    assert "Rewind log" not in console.export_text()
    assert not (tmp_path / RewindLog.FILENAME).exists()


@pytest.mark.parametrize("masked", [True, False])
def test_mlx_start_rewind_builds_log_state_and_loss(tmp_path, masked):
    """Against real mlx: the helper the wrapper calls writes the header and adds the
    two optimizer-state keys, and they survive the optimizer's lazy init."""
    pytest.importorskip("mlx.core")
    import mlx.nn as nn
    import mlx.optimizers as optim

    from soup_cli.trainer import mlx_sft
    from soup_cli.trainer.rewind_mlx import ROW_LOSS_KEY, ROW_TOKENS_KEY

    rows = [{"text": f"row {i}"} for i in range(4)]
    optimizer = optim.Adam(learning_rate=1e-3)
    rewind = mlx_sft._start_rewind(
        output_dir=tmp_path,
        rows=rows,
        optimizer=optimizer,
        batch_size=2,
        grad_accum=1,
        masked=masked,
    )

    assert rewind is not None
    assert ROW_LOSS_KEY in optimizer.state and ROW_TOKENS_KEY in optimizer.state
    header = read_rewind_log(tmp_path / RewindLog.FILENAME).header
    assert header["backend"] == "mlx"
    assert header["dataset_fingerprint"] == dataset_fingerprint(rows)

    optimizer.init(nn.Linear(2, 2).trainable_parameters())
    assert ROW_LOSS_KEY in optimizer.state and ROW_TOKENS_KEY in optimizer.state
    rewind.finish()


# ------------------------------------------------- the field's blast radius --


def _cfg(**training):
    from soup_cli.config.schema import SoupConfig

    body = {"lr": 1e-4}
    body.update(training)
    return SoupConfig.model_validate(
        {
            "base": "HuggingFaceTB/SmolLM2-135M",
            "task": "sft",
            "data": {"train": "train.jsonl"},
            "training": body,
        }
    )


def test_toggling_rewind_log_does_not_invalidate_ship_evidence():
    """`ship` refuses evidence whose config sha moved. rewind_log decides what a
    run writes beside the model, not what the model becomes -- measured on CUDA,
    toggling it left train_loss bit-identical -- so it must not count."""
    from soup_cli.commands.ship import _config_sha_of

    on = _config_sha_of(_cfg(rewind_log=True))
    off = _config_sha_of(_cfg(rewind_log=False))
    assert on == off

    # Control: a change that does alter the model must still move the sha.
    assert _config_sha_of(_cfg(rewind_log=True, lr=2e-4)) != on


@pytest.mark.parametrize(
    "override, expected",
    [
        ("pretokenized", "pre-tokenised"),
        ("is_raft", "RAFT"),
        ("multipack", "multipack"),
        ("vision", "vision"),
        ("audio", "audio"),
    ],
)
def test_every_skipping_path_names_its_own_reason(override, expected):
    """The recorder is the last branch of the trainer chain, so these paths skip
    it. Doing so silently was the bug: `soup rewind` then offers "it was not an
    SFT run" as a reason, which is false for multipack."""
    from soup_cli.trainer.sft import rewind_skip_reason

    kwargs = {
        "task": "sft",
        "pretokenized": False,
        "is_raft": False,
        "multipack": False,
        "vision": False,
        "audio": False,
    }
    kwargs[override] = True
    assert expected in rewind_skip_reason(**kwargs)


def test_a_non_sft_task_is_named_as_such():
    from soup_cli.trainer.sft import rewind_skip_reason

    reason = rewind_skip_reason(
        task="dpo",
        pretokenized=False,
        is_raft=False,
        multipack=False,
        vision=False,
        audio=False,
    )
    assert "SFT-only" in reason
    assert "dpo" in reason
