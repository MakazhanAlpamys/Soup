"""CPU regressions for LISA checkpoint resume (#3282)."""

import json
import random
from contextlib import closing
from types import SimpleNamespace

import pytest

from tests.conftest import strip_ansi


@pytest.fixture
def torch_cpu(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from transformers import TrainingArguments

    post_init = TrainingArguments.__post_init__

    def cpu_post_init(args: TrainingArguments) -> None:
        args.use_cpu = True
        post_init(args)

    monkeypatch.setattr(TrainingArguments, "__post_init__", cpu_post_init)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield torch
    torch.set_num_threads(previous)


def _wrapper(task, path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    from tests.test_issue341_seed_and_fullft import _tiny_llama_dir
    from tests.test_issue341_seed_and_fullft import _wrapper as sft_wrapper
    from tests.test_v07134 import _pretrain_wrapper

    path.mkdir()
    options = {
        "lisa_enabled": True,
        "lisa_num_layers": 1,
        "lisa_interval_steps": 2,
        "save_steps": 1,
        "seed": 123,
        "optimizer": "adamw_torch",
    }
    if task == "sft":
        base = _tiny_llama_dir(path, n_layers=4)
        wrapper, data = sft_wrapper(path, monkeypatch, base=base, **options)
    else:
        wrapper, data = _pretrain_wrapper(path, monkeypatch, n_layers=4, **options)
        data["train"].extend({"text": row} for row in (
            "one cat ran", "two dog ran", "cat sat fast", "red cat blue",
        ))
    wrapper.setup(data)
    assert wrapper.trainer.args.device.type == "cpu"
    assert next(wrapper.trainer.model.parameters()).device.type == "cpu"
    # Keep both checkpoint steps for the comparison on this tiny run.
    wrapper.trainer.args.save_total_limit = 8
    wrapper.trainer.args.disable_tqdm = True
    return wrapper


def _clone(value):
    return value.detach().cpu().clone()


def _run(task, folder, checkpoint=None):
    from transformers import TrainerCallback
    from transformers.trainer_callback import ExportableState

    snapshots = {}

    class Trace(TrainerCallback, ExportableState):
        def __init__(self):
            self.counter = 0

        def state(self):
            return {"args": {}, "attributes": {"counter": self.counter}}

        def on_step_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
            step = state.global_step + 1
            grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
            snapshots[step] = {
                "active": tuple(name for name, p in model.named_parameters() if p.requires_grad),
                "grouped": {name for name, p in model.named_parameters() if id(p) in grouped},
                "slots": {
                    name: {key: _clone(value) for key, value in optimizer.state[p].items()}
                    for name, p in model.named_parameters() if optimizer.state.get(p)
                },
                "counter": self.counter,
            }
            self.counter += 1

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            snapshots[state.global_step + 1]["gradients"] = {
                name: _clone(p.grad) for name, p in model.named_parameters() if p.grad is not None
            }

        def on_step_end(self, args, state, control, model=None, **kwargs):
            snapshots[state.global_step]["weights"] = {
                name: _clone(value) for name, value in model.state_dict().items()
            }

    with pytest.MonkeyPatch.context() as monkeypatch:
        wrapper = _wrapper(task, folder, monkeypatch)
        wrapper.trainer.add_callback(Trace())
        wrapper.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        assert not wrapper.trainer.args.restore_callback_states_from_checkpoint
        assert wrapper.trainer.state.global_step == 8
        return snapshots


def _same_tensors(torch, actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        assert torch.equal(actual[name], expected[name]), name


def _assert_continuation(torch, actual, reference, saved_step):
    assert list(actual) == list(range(saved_step + 1, 9))
    # The unrelated stateful callback starts fresh under Trainer's default.
    assert actual[saved_step + 1]["counter"] == 0
    for step, snapshots in actual.items():
        expected = reference[step]
        assert snapshots["active"] == expected["active"]
        assert snapshots["grouped"] == expected["grouped"]
        assert snapshots["slots"].keys() == expected["slots"].keys()
        for name, slot in snapshots["slots"].items():
            _same_tensors(torch, slot, expected["slots"][name])
        _same_tensors(torch, snapshots["gradients"], expected["gradients"])
        _same_tensors(torch, snapshots["weights"], expected["weights"])


@pytest.mark.parametrize("task", ["sft", "pretrain"])
def test_wrapper_resume_matches_uninterrupted_at_boundary_and_mid_interval(
    task, tmp_path, torch_cpu
):
    reference_dir = tmp_path / "reference"
    reference = _run(task, reference_dir)
    # v0.78.0 seed-0 selections at the start of each update for this layout.
    assert [
        {int(name.split(".layers.")[1].split(".")[0])
         for name in reference[step]["active"] if ".layers." in name}
        for step in range(1, 9)
    ] == [{3}, {3}, {3}, {3}, {0}, {0}, {2}, {2}]
    decoder = {name for name in reference[1]["weights"] if ".layers." in name}
    assert reference[1]["grouped"] >= decoder
    for step in (4, 5):
        checkpoint = reference_dir / "out" / f"checkpoint-{step}"
        data = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        saved = data["stateful_callbacks"]["LisaCallback"]["attributes"]["_resume_checkpoint"]
        assert saved["step"] == step
        assert saved["layers"] == [0]
        assert saved["fire_count"] == 2
        assert saved["active_decoder_names"] == sorted(
            name for name in reference[step + 1]["active"] if name in decoder
        )
        assert data["stateful_callbacks"]["Trace"]["attributes"]["counter"] == step

        resumed_dir = tmp_path / f"resumed-{step}"
        resumed = _run(task, resumed_dir, checkpoint)
        _assert_continuation(torch_cpu, resumed, reference, step)

    repeated_checkpoint = tmp_path / "resumed-5" / "out" / "checkpoint-6"
    repeated = _run(task, tmp_path / "repeated", repeated_checkpoint)
    _assert_continuation(torch_cpu, repeated, reference, 6)


@pytest.mark.parametrize("task", ["sft", "pretrain"])
def test_wrapper_refuses_legacy_checkpoint_before_trainer_train(task, tmp_path, torch_cpu):
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 5, "stateful_callbacks": {"TrainerControl": {}}}', encoding="utf-8"
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        wrapper = _wrapper(task, tmp_path / "run", monkeypatch)
        parameters_before = tuple(p.requires_grad for p in wrapper.model.parameters())
        monkeypatch.setattr(
            wrapper.trainer, "train", lambda **_: pytest.fail("Trainer.train was called")
        )
        with pytest.raises(ValueError, match="LISA resume: checkpoint has no saved LISA state"):
            wrapper.train(resume_from_checkpoint=str(checkpoint))
        assert tuple(p.requires_grad for p in wrapper.model.parameters()) == parameters_before
        assert wrapper.trainer.optimizer is None
        # Retrying on the same wrapper must reach Trainer with a single callback.
        class EnteredTrainerError(Exception):
            pass

        def enter_trainer(**kwargs):
            assert kwargs["resume_from_checkpoint"] is None
            callbacks = wrapper.trainer.callback_handler.callbacks
            assert sum(type(callback).__name__ == "LisaCallback" for callback in callbacks) == 1
            raise EnteredTrainerError

        monkeypatch.setattr(wrapper.trainer, "train", enter_trainer)
        with pytest.raises(EnteredTrainerError):
            wrapper.train()


def test_cli_legacy_checkpoint_reports_the_lisa_error_without_a_crash_bundle(
    tmp_path, monkeypatch, torch_cpu
):
    import yaml
    from rich.console import Console
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.commands import train as train_command
    from soup_cli.experiment.tracker import ExperimentTracker
    from tests.test_issue341_seed_and_fullft import _ROWS, _tiny_llama_dir

    pytest.importorskip("datasets")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
    colored_console = Console(
        width=80,
        force_terminal=True,
        color_system="standard",
        legacy_windows=False,
        no_color=False,
    )
    monkeypatch.setattr(train_command, "console", colored_console)
    dataset = tmp_path / "train.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": user},
                      {"role": "assistant", "content": assistant}]}
        for user, assistant in _ROWS
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step": 5}', encoding="utf-8")
    config = tmp_path / "soup.yaml"
    config.write_text(yaml.safe_dump({
        "base": _tiny_llama_dir(tmp_path, n_layers=4),
        "task": "sft",
        "backend": "transformers",
        "data": {"train": str(dataset), "max_length": 64, "chat_template": "chatml"},
        "output": str(tmp_path / "out"),
        "training": {
            "lisa_enabled": True, "quantization": "none", "lisa_num_layers": 1,
            "batch_size": 1, "optimizer": "adamw_torch",
        },
    }), encoding="utf-8")

    result = CliRunner().invoke(app, [
        "train", "--config", str(config), "--name", "local-check",
        "--resume", str(checkpoint), "--yes",
    ], color=True)
    assert result.exit_code == 1
    output = " ".join(strip_ansi(result.output).split())
    diagnostic = "LISA resume: checkpoint has no saved LISA state"
    migration = "Use its model weights to start a fresh run in a new output directory."
    assert diagnostic in output
    assert migration in output
    assert "Crash bundle written" not in output
    assert not (tmp_path / ".soup-crashes").exists()
    with closing(ExperimentTracker(db_path)) as tracker:
        (run,) = tracker.list_runs()
    assert run["status"] == "failed"
    assert run["error_message"] == f"LisaResumeError: {diagnostic}. {migration}"


def _model(torch, width=2):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(torch.nn.Linear(width, width) for _ in range(4))
    model.embed_tokens = torch.nn.Embedding(7, width)
    model.extra = torch.nn.Linear(width, width)
    return model


def _checkpoint(callback, model, folder):
    callback.on_train_begin(None, SimpleNamespace(global_step=0), model=model)
    for step in range(1, 6):
        callback.on_step_end(None, SimpleNamespace(global_step=step), model=model)
    data = {
        "global_step": 5,
        "stateful_callbacks": {"LisaCallback": callback.state()},
    }
    path = folder / "trainer_state.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("task", ["sft", "pretrain"])
def test_wrapper_refuses_reuse_after_lisa_policy_changes(task, tmp_path, monkeypatch, torch_cpu):
    from soup_cli.utils.lisa import LisaCallback, LisaResumeError
    from soup_cli.utils.peft_wiring import attach_lisa_callback, prepare_lisa_checkpoint

    wrapper = _wrapper(task, tmp_path / "run", monkeypatch)
    trainer = wrapper.trainer
    config = wrapper.config.training
    assert attach_lisa_callback(trainer, config)
    (callback,) = [cb for cb in trainer.callback_handler.callbacks if isinstance(cb, LisaCallback)]
    prepare_lisa_checkpoint(trainer, config, None)
    _checkpoint(callback, trainer.model, tmp_path)

    monkeypatch.setattr(trainer, "train", lambda **_: pytest.fail("Trainer.train was called"))
    config.lisa_interval_steps = 4
    with pytest.raises(LisaResumeError, match="attached callback policy does not match"):
        wrapper.train(resume_from_checkpoint=str(tmp_path))
    assert callback.policy.interval_steps == 2

    config.lisa_interval_steps = 2
    assert attach_lisa_callback(trainer, config)
    assert [
        cb for cb in trainer.callback_handler.callbacks if isinstance(cb, LisaCallback)
    ] == [callback]
    prepare_lisa_checkpoint(trainer, config, str(tmp_path))


def test_compiled_lisa_checkpoint_uses_unwrapped_parameter_names(tmp_path, torch_cpu):
    from torch._dynamo import is_dynamo_supported

    from soup_cli.utils.lisa import LisaCallback, LisaPolicy

    if not is_dynamo_supported():
        pytest.skip("torch.compile is unavailable on this platform")

    policy = LisaPolicy(1, 2)
    reference = _model(torch_cpu)
    saved = LisaCallback(policy)
    saved.prepare_checkpoint(reference, None)
    wrapped = torch_cpu.compile(reference, backend="eager")
    assert all(name.startswith("_orig_mod.") for name, _ in wrapped.named_parameters())
    _checkpoint(saved, wrapped, tmp_path)

    fresh = _model(torch_cpu)
    resumed = LisaCallback(policy)
    resumed.prepare_checkpoint(fresh, str(tmp_path))
    wrapped_fresh = torch_cpu.compile(fresh, backend="eager")
    resumed.on_train_begin(None, SimpleNamespace(global_step=5), model=wrapped_fresh)

    assert resumed.state() == saved.state()
    assert [name for name, param in fresh.named_parameters() if param.requires_grad] == [
        name for name, param in reference.named_parameters() if param.requires_grad
    ]


@pytest.mark.parametrize("reset_optimizer", [True, False])
@pytest.mark.parametrize("train_embeddings", [True, False])
def test_restore_mask_and_clear_only_frozen_decoder_slots(
    reset_optimizer, train_embeddings, tmp_path, torch_cpu
):
    from soup_cli.utils.lisa import LisaCallback, LisaPolicy

    policy = LisaPolicy(1, 2, reset_optimizer=reset_optimizer, train_embeddings=train_embeddings)
    reference = _model(torch_cpu)
    saved = LisaCallback(policy)
    _checkpoint(saved, reference, tmp_path)

    model = _model(torch_cpu)
    optimizer = torch_cpu.optim.AdamW(model.parameters())
    # Simulate stale loaded moments on all parameters. A correct uninterrupted
    # run with reset enabled would already have discarded its frozen slots.
    for index, param in enumerate(model.parameters(), start=1):
        optimizer.state[param] = {
            "step": torch_cpu.tensor(float(index)),
            "exp_avg": torch_cpu.full_like(param, float(index)),
            "exp_avg_sq": torch_cpu.full_like(param, float(index * index)),
        }
    before = {
        name: {key: _clone(value) for key, value in optimizer.state[param].items()}
        for name, param in model.named_parameters()
    }
    callback = LisaCallback(policy)
    callback.prepare_checkpoint(model, str(tmp_path))
    assert all(param.requires_grad for param in model.parameters())
    callback.on_train_begin(None, SimpleNamespace(global_step=5), model=model, optimizer=optimizer)
    assert callback.fire_count == saved.fire_count == 2
    assert callback._rng.getstate() == saved._rng.getstate()
    expected_mask = {name: param.requires_grad for name, param in reference.named_parameters()}
    for name, param in model.named_parameters():
        assert param.requires_grad == expected_mask[name]
        if reset_optimizer and name.startswith("layers.") and not param.requires_grad:
            assert not optimizer.state.get(param), name
        else:
            _same_tensors(torch_cpu, optimizer.state[param], before[name])


@pytest.mark.parametrize(
    "key,value,diagnostic",
    [
        ("version", True, "unsupported checkpoint state version"),
        ("version", 42, "unsupported checkpoint state version"),
        ("step", -1, "checkpoint step does not match"),
        ("fire_count", 0, "invalid checkpoint sample count"),
        ("layers", [False], "invalid sampled decoder layers"),
        ("layers", [], "invalid sampled decoder layers"),
        ("active_decoder_names", [], "invalid active decoder parameter names"),
        ("active_decoder_names", [None], "invalid active decoder parameter names"),
        ("rng", None, "invalid private RNG state"),
        ("rng", "", "invalid private RNG state"),
        ("rng", [3, [0], None], "invalid private RNG state"),
        ("policy", {}, "invalid checkpoint policy"),
        ("layout", "", "checkpoint parameter layout does not match"),
    ],
)
def test_corrupt_callback_state_fails_during_preflight(
    key, value, diagnostic, tmp_path, torch_cpu
):
    from soup_cli.utils.lisa import LisaCallback, LisaPolicy

    policy = LisaPolicy(1, 2)
    model = _model(torch_cpu)
    path = _checkpoint(LisaCallback(policy), model, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["stateful_callbacks"]["LisaCallback"]["attributes"]["_resume_checkpoint"][key] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=f"LISA resume: {diagnostic}"):
        LisaCallback(policy).prepare_checkpoint(_model(torch_cpu), str(tmp_path))


def test_refuses_layout_and_policy_changes(tmp_path, torch_cpu):
    from soup_cli.utils.lisa import LisaCallback, LisaPolicy

    policy = LisaPolicy(1, 2)
    _checkpoint(LisaCallback(policy), _model(torch_cpu), tmp_path)
    with pytest.raises(ValueError, match="LISA resume: checkpoint parameter layout"):
        LisaCallback(policy).prepare_checkpoint(_model(torch_cpu, width=3), str(tmp_path))
    with pytest.raises(ValueError, match="LISA resume: checkpoint policy does not match"):
        LisaCallback(LisaPolicy(1, 2, reset_optimizer=False)).prepare_checkpoint(
            _model(torch_cpu), str(tmp_path)
        )


def test_refuses_reordered_parameters_before_optimizer_load(tmp_path, torch_cpu):
    from soup_cli.utils.lisa import LisaCallback, LisaPolicy

    policy = LisaPolicy(1, 2)
    original = _model(torch_cpu)
    original_optimizer = torch_cpu.optim.AdamW(original.parameters())
    for index, param in enumerate(original.parameters(), start=1):
        original_optimizer.state[param] = {
            "step": torch_cpu.tensor(1.0),
            "exp_avg": torch_cpu.full_like(param, float(index)),
            "exp_avg_sq": torch_cpu.ones_like(param),
        }
    _checkpoint(LisaCallback(policy), original, tmp_path)
    LisaCallback(policy).prepare_checkpoint(_model(torch_cpu), str(tmp_path))

    reordered = _model(torch_cpu)
    reordered.layers = torch_cpu.nn.ModuleDict(
        {str(index): reordered.layers[index] for index in reversed(range(4))}
    )
    assert {n: p.shape for n, p in original.named_parameters()} == {
        n: p.shape for n, p in reordered.named_parameters()
    }
    # AdamW still loads successfully, but attaches layer 3's moment to layer 0.
    optimizer = torch_cpu.optim.AdamW(reordered.parameters())
    optimizer.load_state_dict(original_optimizer.state_dict())
    assert original_optimizer.state[original.layers[0].weight]["exp_avg"][0, 0].item() == 1
    assert optimizer.state[reordered.layers["0"].weight]["exp_avg"][0, 0].item() == 7

    with pytest.raises(ValueError, match="LISA resume: checkpoint parameter layout"):
        LisaCallback(policy).prepare_checkpoint(reordered, str(tmp_path))


def test_refuses_degenerate_private_rng_state(tmp_path, torch_cpu):
    from soup_cli.utils.lisa import LisaCallback, LisaPolicy

    policy = LisaPolicy(1, 2)
    model = _model(torch_cpu)
    path = _checkpoint(LisaCallback(policy), model, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    saved = data["stateful_callbacks"]["LisaCallback"]["attributes"]["_resume_checkpoint"]
    words = saved["rng"][1]
    words[:-1] = [0] * (len(words) - 1)
    probe = random.Random(0)
    probe.setstate((saved["rng"][0], tuple(words), None))
    assert [probe.getrandbits(32) for _ in range(4)] == [0, 0, 0, 0]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="LISA resume: invalid private RNG state"):
        LisaCallback(policy).prepare_checkpoint(model, str(tmp_path))


def test_fresh_restart_on_reused_callback_saves_a_resumable_state(tmp_path, torch_cpu):
    from soup_cli.utils.lisa import LisaCallback, LisaPolicy

    policy = LisaPolicy(1, 2)
    model = _model(torch_cpu)
    callback = LisaCallback(policy)
    path = _checkpoint(callback, model, tmp_path)
    reference = path.read_bytes()

    callback.prepare_checkpoint(model, None)
    _checkpoint(callback, model, tmp_path)

    assert path.read_bytes() == reference
    LisaCallback(policy).prepare_checkpoint(_model(torch_cpu), str(tmp_path))


def test_refuses_missing_and_unreadable_trainer_state(tmp_path, torch_cpu):
    from soup_cli.utils.lisa import LisaCallback, LisaPolicy

    callback = LisaCallback(LisaPolicy(1, 2))
    model = _model(torch_cpu)
    with pytest.raises(ValueError, match="LISA resume: checkpoint has no saved LISA state"):
        callback.prepare_checkpoint(model, str(tmp_path))
    (tmp_path / "trainer_state.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="LISA resume: could not read trainer state"):
        callback.prepare_checkpoint(model, str(tmp_path))


def test_exportable_state_constructor_round_trip(tmp_path, torch_cpu):
    from transformers.trainer_callback import ExportableState

    from soup_cli.utils.lisa import LisaCallback, LisaPolicy

    callback = LisaCallback(LisaPolicy(1, 2))
    original = _model(torch_cpu)
    _checkpoint(callback, original, tmp_path)
    assert isinstance(callback, ExportableState)
    recreated = LisaCallback.from_state(json.loads(json.dumps(callback.state())))
    resumed = _model(torch_cpu)
    recreated.on_train_begin(None, SimpleNamespace(global_step=5), model=resumed)
    assert recreated.policy == callback.policy
    assert recreated._rng.getstate() == callback._rng.getstate()
    assert [p.requires_grad for p in resumed.parameters()] == [
        p.requires_grad for p in original.parameters()
    ]
