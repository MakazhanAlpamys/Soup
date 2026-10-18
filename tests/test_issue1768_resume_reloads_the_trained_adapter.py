"""#1768 — a resumed DPO, IPO or GRPO run goes on from the checkpoint's adapter.

The wrappers apply LoRA themselves and hand TRL a ``PeftModel``. For one,
``DPOTrainer`` (dpo, ipo and ``task: preference`` with either loss) and
``GRPOTrainer`` (unless ``grpo_beta: 0``) add a frozen ``ref`` adapter beside
the trained ``default`` one, so every checkpoint holds ``default`` at its root
and ``ref`` in ``checkpoint-N/ref/``. ``Trainer._load_from_checkpoint`` loads
the subfolder adapters and, when there is one, nothing else: the resumed run
restored the step counter and the optimizer and trained a fresh adapter.
``reload_root_adapter_on_resume`` loads the root right after that load.

The first half runs the real wrappers on the tiny local Llama, on CPU: one
short run writes the checkpoint, a second one resumes from it and is stopped
where its training loop begins, with the checkpoint loaded and no step taken.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

for _mod in ("torch", "transformers", "peft", "trl", "datasets"):
    pytest.importorskip(_mod, reason=f"{_mod} is only in the [train] extra")

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from torch import nn  # noqa: E402

from soup_cli.utils.mixed_precision import (  # noqa: E402
    _resume_skipped_the_root_adapter,
    keep_trainable_dtype_on_resume,
    reload_root_adapter_on_resume,
)

_PAIRS = [
    {
        "prompt": f"What is {n}+{n}?",
        "chosen": f"The answer is {2 * n}.",
        "rejected": f"I think it is {n}.",
    }
    for n in range(8)
]
_CHATS = [
    {
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["chosen"]},
        ]
    }
    for row in _PAIRS
]
_PROMPTS = [{"prompt": row["prompt"]} for row in _PAIRS]

# Differs inside every group of two generations whatever the model samples, so
# each step has an advantage and the adapter moves.
_REWARD = (
    "def reward_fn(completions, **kwargs):\n"
    "    return [float(n % 2) for n in range(len(completions))]\n"
)

_GRPO = "  num_generations: 2\n  reward_fn: {reward}\n"

# name: task, lines under data:, lines under training:, rows, a ref/ in the checkpoint
_RUNS = {
    "dpo": ("dpo", "  format: dpo\n", "", _PAIRS, True),
    "ipo": ("ipo", "  format: dpo\n", "", _PAIRS, True),
    "preference, dpo": ("preference", "  format: dpo\n", "  preference_loss: dpo\n", _PAIRS, True),
    "preference, ipo": ("preference", "  format: dpo\n", "  preference_loss: ipo\n", _PAIRS, True),
    "grpo": ("grpo", "", _GRPO, _PROMPTS, True),
    "grpo, grpo_beta: 0": ("grpo", "", _GRPO + "  grpo_beta: 0\n", _PROMPTS, False),
    "sft": ("sft", "  chat_template: chatml\n", "", _CHATS, False),
    "reward_model": ("reward_model", "  format: dpo\n", "", _PAIRS, False),
}
_WITH_REF = [name for name, run in _RUNS.items() if run[4]]
_WITHOUT_REF = [name for name, run in _RUNS.items() if not run[4]]


class _StopError(Exception):
    """Raised where the resumed training loop begins: nothing is trained."""


def _adapter(model, name: str) -> dict:
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model, adapter_name=name)
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _stop_once_loaded():
    from transformers import TrainerCallback

    class StopOnceLoaded(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.step = state.global_step
            self.default = _adapter(model, "default")
            self.ref = _adapter(model, "ref") if "ref" in model.peft_config else None
            self.active = list(model.active_adapters)
            self.trainable = {
                name for name, param in model.named_parameters() if param.requires_grad
            }
            raise _StopError

    return StopOnceLoaded()


def _wrapper(name: str, root: Path, out: str):
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.dispatch import build_trainer
    from tests._tiny_hf_models import tiny_model_dir

    task, data, training, rows, _ = _RUNS[name]
    (root / "reward.py").write_text(_REWARD, encoding="utf-8")
    cfg = load_config_from_string(
        f"base: {Path(tiny_model_dir('llama')).as_posix()}\n"
        f"task: {task}\n"
        f"data:\n  train: x.jsonl\n  max_length: 64\n  val_split: 0\n{data}"
        "training:\n  epochs: 1\n  batch_size: 2\n  gradient_accumulation_steps: 1\n"
        "  quantization: none\n  lr: 1e-3\n  save_steps: 2\n"
        "  lora:\n    r: 8\n    alpha: 16\n    target_modules: [q_proj, v_proj]\n"
        f"{training.format(reward=(root / 'reward.py').as_posix())}"
        f"output: {(root / out).as_posix()}\n"
    )
    wrapper = build_trainer(cfg, device="cpu")
    wrapper.setup({"train": [dict(row) for row in rows]})
    return wrapper, getattr(wrapper, "_inner", wrapper).trainer


_resumed_runs: dict[str, SimpleNamespace] = {}


def _resumed(name: str, tmp_path_factory) -> SimpleNamespace:
    """One uninterrupted run and one resume from its first checkpoint, per name."""
    if name in _resumed_runs:
        return _resumed_runs[name]
    root = tmp_path_factory.mktemp("resume")

    wrapper, trainer = _wrapper(name, root, "full")
    wrapper.train()
    loader_replaced_without_a_resume = "_load_from_checkpoint" in vars(trainer)
    first = min(
        (path for path in (root / "full").iterdir() if path.name.startswith("checkpoint-")),
        key=lambda path: int(path.name.split("-")[1]),
    )
    # A copy outside the output directory, as a user resuming an old run has it.
    checkpoint = Path(shutil.copytree(first, root / "kept" / first.name))

    wrapper, trainer = _wrapper(name, root, "resumed")
    seen = _stop_once_loaded()
    trainer.add_callback(seen)
    loads = []
    load_adapter = trainer.model.load_adapter

    def spy(model_id, adapter_name, *args, **kwargs):
        relative = os.path.relpath(model_id, checkpoint)
        loads.append((Path(relative).as_posix(), adapter_name, kwargs.get("is_trainable")))
        return load_adapter(model_id, adapter_name, *args, **kwargs)

    trainer.model.load_adapter = spy
    with pytest.raises(_StopError):
        wrapper.train(resume_from_checkpoint=str(checkpoint))

    run = SimpleNamespace(
        checkpoint=checkpoint,
        step=int(first.name.split("-")[1]),
        saved=load_file(checkpoint / "adapter_model.safetensors"),
        seen=seen,
        loads=loads,
        loader_replaced_without_a_resume=loader_replaced_without_a_resume,
    )
    _resumed_runs[name] = run
    return run


def _largest_b(adapter: dict) -> float:
    return max(float(value.abs().max()) for key, value in adapter.items() if "lora_B" in key)


class TestTheRealWrappers:
    @pytest.mark.parametrize("name", list(_RUNS))
    def test_the_checkpoint_is_the_one_this_file_is_about(self, name, tmp_path_factory):
        """Guard: a ``ref/`` where the table says so, and an adapter that has
        moved, or the tests below could not tell a loaded adapter from a new one."""
        run = _resumed(name, tmp_path_factory)

        assert (run.checkpoint / "ref" / "adapter_model.safetensors").is_file() is _RUNS[name][4]
        assert _largest_b(run.saved) > 1e-4

    @pytest.mark.parametrize("name", list(_RUNS))
    def test_the_resumed_run_starts_from_the_checkpoints_adapter(self, name, tmp_path_factory):
        run = _resumed(name, tmp_path_factory)

        assert run.seen.step == run.step
        assert set(run.seen.default) == set(run.saved)
        for key, saved in run.saved.items():
            assert torch.equal(run.seen.default[key], saved), key

    @pytest.mark.parametrize("name", _WITH_REF)
    def test_the_reference_adapter_is_the_checkpoints_too_and_stays_frozen(
        self, name, tmp_path_factory
    ):
        run = _resumed(name, tmp_path_factory)
        saved = load_file(run.checkpoint / "ref" / "adapter_model.safetensors")

        assert set(run.seen.ref) == set(saved)
        for key, value in saved.items():
            assert torch.equal(run.seen.ref[key], value), key
        assert run.seen.active == ["default"]
        assert run.seen.trainable
        assert not [name for name in run.seen.trainable if ".ref." in name]
        assert [name for name in run.seen.trainable if ".default." in name]

    @pytest.mark.parametrize("name", _WITH_REF)
    def test_the_root_is_read_once_after_the_subfolder(self, name, tmp_path_factory):
        run = _resumed(name, tmp_path_factory)

        assert run.loads == [("ref", "ref", False), (".", "default", True)]

    @pytest.mark.parametrize("name", _WITHOUT_REF)
    def test_a_checkpoint_with_no_subfolder_is_read_once_by_the_trainer(
        self, name, tmp_path_factory
    ):
        run = _resumed(name, tmp_path_factory)

        assert run.loads == [(".", "default", True)]

    @pytest.mark.parametrize("name", list(_RUNS))
    def test_a_run_that_does_not_resume_keeps_the_trainers_loader(self, name, tmp_path_factory):
        assert _resumed(name, tmp_path_factory).loader_replaced_without_a_resume is False


# ─── The helper alone ────────────────────────────────────────────────────────


class _PeftLike(nn.Sequential):
    """Two linears with the three things the helper reads off a ``PeftModel``.
    ``load_adapter`` upcasts, as PEFT's ``autocast_adapter_dtype`` does."""

    def __init__(self, adapters=("default", "ref")):
        super().__init__(nn.Linear(4, 4), nn.Linear(4, 4))
        self[0].requires_grad_(False)
        self[1].to(torch.bfloat16)
        self.peft_config = {name: object() for name in adapters}
        self.active_adapters = [adapters[0]]
        self.loaded = []

    def load_adapter(self, model_id, adapter_name, is_trainable=False):
        self.loaded.append((model_id, adapter_name, is_trainable))
        for param in self.parameters():
            param.data = param.data.to(torch.float32)


def _checkpoint(tmp_path, *files: str) -> str:
    folder = tmp_path / "checkpoint-2"
    folder.mkdir()
    for name in files:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return str(folder)


def _trainer(model):
    calls = []

    def load(resume_from_checkpoint, model=None):
        calls.append((resume_from_checkpoint, model))

    return SimpleNamespace(model=model, _load_from_checkpoint=load, calls=calls)


_ROOT = "adapter_model.safetensors"
_REF = "ref/adapter_model.safetensors"

# files in the checkpoint, the model, whether the root is read after the Trainer's load
_LAYOUTS = {
    "root and a subfolder": ((_ROOT, _REF), _PeftLike, True),
    "the same as .bin files": (("adapter_model.bin", "ref/adapter_model.bin"), _PeftLike, True),
    "root only: the Trainer read it": ((_ROOT,), _PeftLike, False),
    "a subfolder only: nothing at the root to read": ((_REF,), _PeftLike, False),
    "a subfolder with no adapter in it": ((_ROOT, "runs/events.out"), _PeftLike, False),
    "a file named like an adapter folder": ((_ROOT, "ref"), _PeftLike, False),
    "full weights: the Trainer takes its other branch": (
        (_ROOT, _REF, "model.safetensors"), _PeftLike, False,
    ),
    "full weights as .bin": ((_ROOT, _REF, "pytorch_model.bin"), _PeftLike, False),
    "a model with no adapters": ((_ROOT, _REF), lambda: nn.Sequential(nn.Linear(4, 4)), False),
    "no adapter named default": ((_ROOT, _REF), lambda: _PeftLike(("policy", "ref")), False),
}


class TestTheHelper:
    @pytest.mark.parametrize("layout", list(_LAYOUTS))
    def test_the_root_is_read_only_where_the_trainer_left_it(self, tmp_path, layout):
        files, build, expected = _LAYOUTS[layout]
        checkpoint = _checkpoint(tmp_path, *files)
        model = build()
        trainer = _trainer(model)

        reload_root_adapter_on_resume(trainer)
        trainer._load_from_checkpoint(checkpoint)

        assert trainer.calls == [(checkpoint, None)]
        assert getattr(model, "loaded", []) == ([(checkpoint, "default", True)] if expected else [])
        assert _resume_skipped_the_root_adapter(checkpoint, model) is expected

    def test_a_checkpoint_that_is_not_a_folder_is_left_to_the_trainer(self, tmp_path):
        model = _PeftLike()
        trainer = _trainer(model)

        reload_root_adapter_on_resume(trainer)
        trainer._load_from_checkpoint(str(tmp_path / "missing"))

        assert trainer.calls == [(str(tmp_path / "missing"), None)]
        assert model.loaded == []

    def test_the_trainers_load_comes_first(self, tmp_path):
        """The root goes in over whatever that load left, not under it."""
        checkpoint = _checkpoint(tmp_path, _ROOT, _REF)
        model = _PeftLike()
        order = []
        trainer = SimpleNamespace(
            model=model, _load_from_checkpoint=lambda *args: order.append("trainer")
        )
        model.load_adapter = lambda *args, **kwargs: order.append("root")

        reload_root_adapter_on_resume(trainer)
        trainer._load_from_checkpoint(checkpoint)

        assert order == ["trainer", "root"]

    def test_a_trainers_load_that_fails_is_not_followed_by_a_reload(self, tmp_path):
        checkpoint = _checkpoint(tmp_path, _ROOT, _REF)
        model = _PeftLike()

        def load(resume_from_checkpoint, model=None):
            raise ValueError("Can't find a valid checkpoint")

        trainer = SimpleNamespace(model=model, _load_from_checkpoint=load)
        reload_root_adapter_on_resume(trainer)

        with pytest.raises(ValueError, match="valid checkpoint"):
            trainer._load_from_checkpoint(checkpoint)
        assert model.loaded == []

    def test_an_explicit_model_is_the_one_reloaded(self, tmp_path):
        checkpoint = _checkpoint(tmp_path, _ROOT, _REF)
        own, other = _PeftLike(), _PeftLike()
        trainer = _trainer(own)

        reload_root_adapter_on_resume(trainer)
        trainer._load_from_checkpoint(checkpoint, other)

        assert trainer.calls == [(checkpoint, other)]
        assert (own.loaded, other.loaded) == ([], [(checkpoint, "default", True)])

    def test_an_inactive_default_adapter_is_reloaded_but_not_as_trainable(self, tmp_path):
        checkpoint = _checkpoint(tmp_path, _ROOT, _REF)
        model = _PeftLike()
        model.active_adapters = ["ref"]
        trainer = _trainer(model)

        reload_root_adapter_on_resume(trainer)
        trainer._load_from_checkpoint(checkpoint)

        assert model.loaded == [(checkpoint, "default", False)]

    def test_no_model_is_a_no_op(self, tmp_path):
        checkpoint = _checkpoint(tmp_path, _ROOT, _REF)
        trainer = _trainer(None)

        reload_root_adapter_on_resume(trainer)
        trainer._load_from_checkpoint(checkpoint)

        assert trainer.calls == [(checkpoint, None)]


class TestOnARealPeftModel:
    """No TRL here: PEFT writes the two-adapter checkpoint and the real
    ``Trainer._load_from_checkpoint`` reads it."""

    @staticmethod
    def _model(fill: float | None = None):
        from peft import LoraConfig, get_peft_model

        from tests._tiny_hf_models import build_llama, build_tokenizer

        torch.manual_seed(0)
        model = get_peft_model(
            build_llama(build_tokenizer()), LoraConfig(r=4, target_modules=["q_proj", "v_proj"])
        )
        model.add_adapter("ref", model.peft_config["default"])  # what TRL does
        model.set_adapter("default")
        if fill is not None:
            for name, param in model.named_parameters():
                if "lora_B.default" in name:
                    nn.init.constant_(param, fill)
        return model

    def _resume(self, tmp_path, *, with_the_helper: bool):
        from transformers import Trainer, TrainingArguments

        checkpoint = tmp_path / "checkpoint-2"
        self._model(fill=0.25).save_pretrained(checkpoint)
        assert (checkpoint / "ref" / "adapter_model.safetensors").is_file()
        fresh = self._model()
        trainer = Trainer(
            model=fresh,
            args=TrainingArguments(output_dir=str(tmp_path / "out"), report_to=[], use_cpu=True),
        )
        if with_the_helper:
            reload_root_adapter_on_resume(trainer)
        trainer._load_from_checkpoint(str(checkpoint))
        return fresh, load_file(checkpoint / "adapter_model.safetensors")

    def test_the_trainer_alone_leaves_the_trained_adapter_new(self, tmp_path):
        """The premise. When this fails, transformers reads the root itself and
        ``reload_root_adapter_on_resume`` can go."""
        model, saved = self._resume(tmp_path, with_the_helper=False)

        assert _largest_b(saved) == 0.25
        assert _largest_b(_adapter(model, "default")) == 0.0, (
            "Trainer._load_from_checkpoint now reads the root adapter beside a "
            "subfolder one, so reload_root_adapter_on_resume (#1768) reads it a "
            "second time: remove the helper and its calls in dpo.py, ipo.py and grpo.py."
        )

    def test_with_the_helper_it_is_the_checkpoints(self, tmp_path):
        model, saved = self._resume(tmp_path, with_the_helper=True)
        loaded = _adapter(model, "default")

        assert set(loaded) == set(saved)
        for key, value in saved.items():
            assert torch.equal(loaded[key], value), key
        assert model.active_adapters == ["default"]
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        assert trainable and all(".default." in name for name in trainable)


# ─── With the dtype restore of #1660 ─────────────────────────────────────────


class TestWithTheDtypeRestore:
    """PEFT upcasts what it loads; the reload must be inside the cast-back."""

    def test_reload_first_then_the_dtype_restore_keeps_bf16(self, tmp_path):
        checkpoint = _checkpoint(tmp_path, _ROOT, _REF)
        model = _PeftLike()
        trainer = _trainer(model)

        reload_root_adapter_on_resume(trainer)
        keep_trainable_dtype_on_resume(trainer)
        trainer._load_from_checkpoint(checkpoint)

        assert model.loaded == [(checkpoint, "default", True)]
        assert model[1].weight.dtype == torch.bfloat16

    def test_the_other_order_would_leave_fp32(self, tmp_path):
        """Guard on the test above and on the order in the wrappers."""
        checkpoint = _checkpoint(tmp_path, _ROOT, _REF)
        model = _PeftLike()
        trainer = _trainer(model)

        keep_trainable_dtype_on_resume(trainer)
        reload_root_adapter_on_resume(trainer)
        trainer._load_from_checkpoint(checkpoint)

        assert model[1].weight.dtype == torch.float32

    @pytest.mark.parametrize("task", ["dpo", "ipo", "grpo"])
    def test_the_wrappers_install_them_in_that_order(self, tmp_path, task):
        from tests import test_resume_trainable_dtype as dtype_tests

        # Borrowed, not copied: the same stand-ins #1660 tests the dtype
        # restore with. Say so if they are renamed there.
        borrowed = ("_FakeTrainer", f"_{task}_wrapper")
        missing = [name for name in borrowed if not hasattr(dtype_tests, name)]
        assert not missing, (
            f"tests/test_resume_trainable_dtype.py no longer has {missing}; this "
            "test borrows them, so point it at their new names."
        )
        checkpoint = _checkpoint(tmp_path, _ROOT, _REF)
        model = _PeftLike()
        trainer = dtype_tests._FakeTrainer(model)
        builder = getattr(dtype_tests, f"_{task}_wrapper")

        builder(tmp_path, trainer).train(resume_from_checkpoint=checkpoint)

        assert model.loaded == [(checkpoint, "default", True)]
        assert trainer.dtypes_in_train == {torch.bfloat16}


def test_save_file_round_trip_is_exact_for_the_comparisons_above(tmp_path):
    """Guard: ``torch.equal`` against a checkpoint file is a fair test of an
    fp32 adapter on CPU."""
    tensor = torch.randn(4, 4)
    save_file({"weight": tensor}, str(tmp_path / "a.safetensors"))

    assert torch.equal(load_file(tmp_path / "a.safetensors")["weight"], tensor)
