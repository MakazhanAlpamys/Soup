"""#2575 — ``training.save_strategy``: switch off (or thin out) intermediate checkpoints.

transformers saves a ``checkpoint-N`` at the last step whenever ``save_strategy`` is
``"steps"`` (``DefaultFlowCallback.on_step_end`` sets ``should_save`` once
``global_step >= max_steps``), and Soup never set the strategy, so every run left one
beside the final adapter and nothing in a ``soup.yaml`` turned that off. The field is
``steps`` (default, today's behaviour), ``epoch`` or ``no``; ONE helper
(``soup_cli.trainer.save_args.save_kwargs``) turns it into the keyword arguments every
wrapper hands to its ``TrainingArguments`` / TRL config.

Layout: the schema, the helper and the ratchet need no torch. The class that really
trains builds a tiny Llama locally (``tests/_tiny_hf_models.py``), so it is offline.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig, TrainingConfig
from tests.conftest import strip_ansi

SRC = Path(__file__).resolve().parents[1] / "src" / "soup_cli"
TRAINER_DIR = SRC / "trainer"


def _clean(text: object) -> str:
    """ANSI-stripped, whitespace-collapsed text: Rich wraps and colours on CI."""
    return " ".join(strip_ansi(str(text)).split())


def _yaml(
    training: str = "", *, task: str = "sft", backend: str | None = None, data: str = ""
) -> str:
    return (
        "base: some-org/some-model\n"
        f"task: {task}\n"
        + (f"backend: {backend}\n" if backend else "")
        + "data:\n  train: ./train.jsonl\n"
        + data
        + "training:\n  epochs: 1\n  quantization: none\n"
        + training
        + "output: ./out\n"
    )


# --------------------------------------------------------------------------- schema


class TestTheField:
    def test_the_default_is_steps_and_save_steps_is_unchanged(self):
        tcfg = TrainingConfig()

        assert tcfg.save_strategy == "steps"
        assert tcfg.save_steps == 100

    def test_an_untouched_config_carries_the_default(self):
        cfg = load_config_from_string(_yaml())

        assert cfg.training.save_strategy == "steps"

    @pytest.mark.parametrize("value", ["steps", "epoch", "no"])
    def test_the_three_strategies_load(self, value):
        cfg = load_config_from_string(_yaml(f'  save_strategy: "{value}"\n'))

        assert cfg.training.save_strategy == value

    @pytest.mark.parametrize("value", ["best", "Steps", "EPOCH", "never", "", " no", "steps "])
    def test_any_other_string_is_refused(self, value):
        with pytest.raises(ValueError) as caught:
            load_config_from_string(_yaml(f'  save_strategy: "{value}"\n'))

        message = _clean(caught.value)
        assert "save_strategy" in message
        assert "'steps', 'epoch' or 'no'" in message

    @pytest.mark.parametrize("literal", ["1", "0", "1.5", "[steps]", "null", "{a: b}"])
    def test_a_non_string_is_refused_and_says_so(self, literal):
        with pytest.raises(ValueError) as caught:
            load_config_from_string(_yaml(f"  save_strategy: {literal}\n"))

        message = _clean(caught.value)
        assert "training.save_strategy must be one of 'steps', 'epoch', 'no'" in message

    @pytest.mark.parametrize("literal", ["no", "false", "off", "yes", "true", "on"])
    def test_an_unquoted_yaml_bool_is_refused_with_the_quoting_hint(self, literal):
        """YAML 1.1 reads an unquoted ``no`` (and ``off``/``false``) as a bool, so
        ``save_strategy: no`` reaches the schema as ``False``. ``bool`` subclasses
        ``int``; it must not be coerced to a strategy. The message says to quote it."""
        with pytest.raises(ValueError) as caught:
            load_config_from_string(_yaml(f"  save_strategy: {literal}\n"))

        message = _clean(caught.value)
        assert "training.save_strategy must be a string, not a bool" in message
        assert 'save_strategy: "no"' in message

    def test_a_bool_is_refused_when_set_programmatically_too(self):
        with pytest.raises(ValueError, match="not a bool"):
            TrainingConfig(save_strategy=False)

    def test_the_quoted_no_survives_a_dump_and_reload(self):
        cfg = load_config_from_string(_yaml('  save_strategy: "no"\n'))

        again = SoupConfig(**cfg.model_dump())

        assert again.training.save_strategy == "no"


class TestSaveStepsNeedsTheStepsStrategy:
    """A tunable set while its feature is off is refused (repo rule)."""

    @pytest.mark.parametrize("strategy", ["epoch", "no"])
    def test_an_explicit_save_steps_is_refused_and_both_fields_are_named(self, strategy):
        with pytest.raises(ValueError) as caught:
            load_config_from_string(
                _yaml(f'  save_strategy: "{strategy}"\n  save_steps: 500\n')
            )

        message = _clean(caught.value)
        assert "training.save_steps" in message
        assert "training.save_strategy" in message
        assert f"'{strategy}'" in message
        assert "only counts under save_strategy: steps" in message

    def test_save_steps_with_steps_is_fine(self):
        cfg = load_config_from_string(_yaml('  save_strategy: "steps"\n  save_steps: 500\n'))

        assert cfg.training.save_steps == 500

    def test_save_steps_alone_is_today_s_behaviour(self):
        cfg = load_config_from_string(_yaml("  save_steps: 25\n"))

        assert (cfg.training.save_strategy, cfg.training.save_steps) == ("steps", 25)

    @pytest.mark.parametrize("strategy", ["epoch", "no"])
    def test_without_save_steps_the_strategy_loads(self, strategy):
        cfg = load_config_from_string(_yaml(f'  save_strategy: "{strategy}"\n'))

        assert cfg.training.save_strategy == strategy
        assert cfg.training.save_steps == 100

    @pytest.mark.parametrize("strategy", ["epoch", "no"])
    def test_a_dumped_config_loads_again(self, strategy):
        """``model_dump()`` writes ``save_steps: 100`` for every config, so a guard
        that read ``model_fields_set`` alone refused its own output. ``sweep``
        (``SoupConfig(**config_dict)``), the unsloth probe, ``soup rewind`` and the
        cloud hand-off all validate a dump. The default is what a dump carries."""
        cfg = load_config_from_string(_yaml(f'  save_strategy: "{strategy}"\n'))

        for rebuilt in (
            SoupConfig(**cfg.model_dump()),
            SoupConfig.model_validate(cfg.model_dump()),
            SoupConfig.model_validate_json(cfg.model_dump_json()),
        ):
            assert rebuilt.training.save_strategy == strategy

    def test_a_mutation_after_load_does_not_trip_the_guard(self):
        """``model_copy(update=...)`` runs no validators (``bench`` does this)."""
        cfg = load_config_from_string(_yaml('  save_strategy: "no"\n'))

        copy = cfg.training.model_copy(update={"save_steps": 7})

        assert (copy.save_strategy, copy.save_steps) == ("no", 7)


class TestTasksAndBackendsThatCannotHonourIt:
    @pytest.mark.parametrize("strategy", ["epoch", "no"])
    def test_mlx_refuses_a_non_default_strategy_naming_the_backend(self, strategy):
        with pytest.raises(ValueError) as caught:
            load_config_from_string(
                _yaml(f'  save_strategy: "{strategy}"\n', backend="mlx")
            )

        message = _clean(caught.value)
        assert "training.save_strategy" in message
        assert "backend: mlx" in message
        assert "training.save_steps" in message  # what the MLX loops do read

    def test_mlx_keeps_steps(self):
        cfg = load_config_from_string(
            _yaml('  save_strategy: "steps"\n  save_steps: 50\n', backend="mlx")
        )

        assert cfg.training.save_steps == 50

    @pytest.mark.parametrize("backend", ["transformers", "unsloth"])
    @pytest.mark.parametrize("strategy", ["epoch", "no"])
    def test_the_hf_backends_accept_it(self, backend, strategy):
        cfg = load_config_from_string(
            _yaml(f'  save_strategy: "{strategy}"\n', backend=backend)
        )

        assert cfg.training.save_strategy == strategy

    @pytest.mark.parametrize("strategy", ["epoch", "no"])
    def test_unlearn_writes_no_checkpoints_so_the_setting_is_refused(self, strategy):
        with pytest.raises(ValueError) as caught:
            load_config_from_string(
                _yaml(
                    f'  save_strategy: "{strategy}"\n  unlearn_method: npo\n',
                    task="unlearn",
                    data="  forget_set: forget.jsonl\n",
                )
            )

        message = _clean(caught.value)
        assert "training.save_strategy" in message
        assert "task: unlearn" in message
        assert "no checkpoint" in message

    def test_unlearn_with_the_default_loads(self):
        cfg = load_config_from_string(
            _yaml(
                "  unlearn_method: npo\n",
                task="unlearn",
                data="  forget_set: forget.jsonl\n",
            )
        )

        assert cfg.training.save_strategy == "steps"


# --------------------------------------------------------------------------- helper


class TestTheHelper:
    def _kwargs(self, **training):
        from soup_cli.trainer.save_args import save_kwargs

        return save_kwargs(TrainingConfig(**training))

    def test_steps_is_todays_behaviour_exactly(self):
        assert self._kwargs() == {
            "save_strategy": "steps",
            "save_steps": 100,
            "save_total_limit": 3,
        }

    def test_steps_carries_the_users_save_steps(self):
        assert self._kwargs(save_steps=7)["save_steps"] == 7

    @pytest.mark.parametrize("strategy", ["epoch", "no"])
    def test_save_steps_is_dropped_for_the_other_strategies(self, strategy):
        assert self._kwargs(save_strategy=strategy) == {
            "save_strategy": strategy,
            "save_total_limit": 3,
        }

    def test_the_keys_are_what_transformers_calls_them(self):
        pytest.importorskip("transformers")
        import inspect

        from transformers import TrainingArguments

        from soup_cli.trainer.save_args import save_kwargs

        accepted = set(inspect.signature(TrainingArguments).parameters)
        for strategy in ("steps", "epoch", "no"):
            kwargs = save_kwargs(TrainingConfig(save_strategy=strategy))
            assert set(kwargs) <= accepted, strategy

    def test_the_helper_does_not_import_torch(self):
        tree = ast.parse((TRAINER_DIR / "save_args.py").read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }

        assert not imported & {"torch", "transformers", "peft", "trl"}


# --------------------------------------------------------------------------- ratchet

_SAVE_NAMES = frozenset({"save_strategy", "save_steps", "save_total_limit"})
#: ``output_dir`` handed to a ``TrainingArguments`` / TRL config. ``unlearn`` builds a
#: ``TrainingArguments`` only to borrow its optimizer arguments and trains in its own loop.
_WRITES_NO_CHECKPOINTS = frozenset({"unlearn"})


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _spellings_of_the_save_settings(tree: ast.AST) -> list[tuple[int, str]]:
    """Every place the module spells a save setting itself: a keyword argument, a dict
    key, a subscript, any other string constant, or a read of ``tcfg.save_steps``."""
    docstrings = _docstring_nodes(tree)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in _SAVE_NAMES:
            found.append((node.value.lineno, f"{node.arg}=..."))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in _SAVE_NAMES
            and id(node) not in docstrings
        ):
            found.append((node.lineno, repr(node.value)))
        elif isinstance(node, ast.Attribute) and node.attr in _SAVE_NAMES:
            found.append((node.lineno, f".{node.attr}"))
    return found


def _trainer_modules() -> list[Path]:
    return sorted(
        path
        for path in TRAINER_DIR.glob("*.py")
        if not path.name.startswith("mlx_")  # their own loops; see TestMlx below
    )


def _passes_output_dir(tree: ast.AST) -> bool:
    """Whether the module hands ``output_dir`` to an HF / TRL arguments class."""
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "output_dir":
            return True
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == "output_dir":
                    return True
    return False


def _calls_save_kwargs(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "save_kwargs")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "save_kwargs")
        )
        for node in ast.walk(tree)
    )


class TestTheRatchet:
    """A new wrapper cannot forget the helper, and an old one cannot grow its own copy."""

    def test_no_wrapper_spells_a_save_setting_itself(self):
        offenders = {}
        for path in _trainer_modules():
            if path.name == "save_args.py":
                continue
            found = _spellings_of_the_save_settings(ast.parse(path.read_text(encoding="utf-8")))
            if found:
                offenders[path.name] = found

        assert offenders == {}, (
            "build the save arguments with soup_cli.trainer.save_args.save_kwargs(tcfg), "
            "not with save_steps= / save_total_limit= / save_strategy= of your own"
        )

    def test_every_wrapper_that_builds_hf_arguments_calls_the_helper(self):
        builders = []
        missing = []
        for path in _trainer_modules():
            if path.name == "save_args.py" or path.stem in _WRITES_NO_CHECKPOINTS:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if not _passes_output_dir(tree):
                continue
            builders.append(path.stem)
            if not _calls_save_kwargs(tree):
                missing.append(path.stem)

        assert missing == []
        # Not vacuous: the scan really found the wrappers it is about.
        assert set(builders) >= {
            "asr", "bco", "classifier", "csm", "distill", "dpo", "embedding", "grpo", "ipo",
            "kto", "mole_routing", "online_dpo", "orpo", "ppo", "pretrain", "prm",
            "reward_model", "sft", "simpo",
        }

    def test_the_scan_sees_a_bypass(self):
        """The two scans above must fail on the thing they exist to catch."""
        bypass = ast.parse(
            "def build(tcfg):\n"
            "    return TrainingArguments(output_dir='o', save_steps=tcfg.save_steps,\n"
            "                             save_total_limit=3)\n"
        )
        as_dict = ast.parse("kwargs = {'output_dir': 'o', 'save_total_limit': 3}\n")
        clean = ast.parse(
            "def build(tcfg):\n"
            "    return TrainingArguments(output_dir='o', **save_kwargs(tcfg))\n"
        )

        assert [name for _, name in _spellings_of_the_save_settings(bypass)][:2] == [
            "save_steps=...", "save_total_limit=...",
        ]
        assert _spellings_of_the_save_settings(as_dict) == [(1, "'save_total_limit'")]
        assert _spellings_of_the_save_settings(clean) == []
        assert _passes_output_dir(bypass) and not _calls_save_kwargs(bypass)
        assert _passes_output_dir(clean) and _calls_save_kwargs(clean)

    def test_the_unlearn_exemption_is_true(self):
        """It trains in its own loop: it must never reach ``Trainer.train``."""
        tree = ast.parse((TRAINER_DIR / "unlearn.py").read_text(encoding="utf-8"))
        trains = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "train"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "trainer"
        ]

        assert trains == []

    def test_the_csm_allowlist_names_the_field(self):
        """``validate_csm_config`` refuses every training field it does not list."""
        from soup_cli.utils.csm import validate_csm_config

        cfg = SoupConfig(
            base="sesame/csm-1b", task="tts", modality="audio_out",
            data={"train": "speech.jsonl", "format": "audio", "val_split": 0},
            training={
                "tts_family": "sesame_csm", "quantization": "none", "save_strategy": "no",
            },
        )

        validate_csm_config(cfg)  # no ValueError


class TestMlx:
    """The MLX loops are not TrainingArguments: they save adapters every
    ``save_steps`` iterations themselves, so ``save_strategy`` is refused there
    (``TestTasksAndBackendsThatCannotHonourIt``) rather than ignored. This pins that
    they still read ``save_steps`` and so need no strategy."""

    @pytest.mark.parametrize("name", ["mlx_sft", "mlx_dpo", "mlx_grpo"])
    def test_the_loops_read_save_steps_not_a_strategy(self, name):
        text = (TRAINER_DIR / f"{name}.py").read_text(encoding="utf-8")

        assert "save_steps" in text
        assert "save_strategy" not in text


# --------------------------------------------------------------------------- real runs

_CHATS = [
    {
        "messages": [
            {"role": "user", "content": f"What is {n}+{n}?"},
            {"role": "assistant", "content": f"The answer is {2 * n}."},
        ]
    }
    for n in range(8)
]
_PAIRS = [
    {"prompt": f"What is {n}+{n}?", "chosen": f"The answer is {2 * n}.", "rejected": "No."}
    for n in range(8)
]
_TEXTS = [{"text": f"hello world the quick brown fox {n}"} for n in range(8)]

#: task -> (lines under data:, rows)
_TASKS = {
    "sft": ("  chat_template: chatml\n", _CHATS),
    "dpo": ("  format: dpo\n", _PAIRS),
    "reward_model": ("  format: dpo\n", _PAIRS),
    "pretrain": ("  format: plaintext\n", _TEXTS),
}


@pytest.fixture
def train_stack():
    for name in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(name, reason=f"{name} is only in the [train] extra")


def _wrapper(root: Path, task: str, strategy: str, *, epochs: int = 1, save_steps=None):
    from soup_cli.trainer.dispatch import build_trainer
    from tests._tiny_hf_models import tiny_model_dir

    data, rows = _TASKS[task]
    cfg = load_config_from_string(
        f"base: {Path(tiny_model_dir('llama')).as_posix()}\n"
        f"task: {task}\n"
        f"data:\n  train: x.jsonl\n  max_length: 64\n  val_split: 0\n{data}"
        f"training:\n  epochs: {epochs}\n  batch_size: 2\n  gradient_accumulation_steps: 1\n"
        "  quantization: none\n  lr: 1e-3\n  warmup_ratio: 0.0\n"
        f'  save_strategy: "{strategy}"\n'
        + (f"  save_steps: {save_steps}\n" if save_steps is not None else "")
        + "  lora:\n    r: 4\n    alpha: 8\n    target_modules: [q_proj, v_proj]\n"
        f"output: {(root / 'out').as_posix()}\n"
    )
    wrapper = build_trainer(cfg, device="cpu")
    wrapper.setup({"train": [dict(row) for row in rows]})
    return wrapper, root / "out"


def _checkpoints(out: Path) -> list[str]:
    return sorted(
        (p.name for p in out.glob("checkpoint-*") if p.is_dir()),
        key=lambda name: int(name.split("-")[1]),
    )


def _assert_the_final_adapter_is_there(out: Path) -> None:
    adapter = out / "adapter_model.safetensors"
    assert adapter.is_file(), sorted(p.name for p in out.iterdir())
    assert adapter.stat().st_size > 1000  # a streamed-adapter regression wrote 40 bytes
    config = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["r"] == 4


@pytest.mark.usefixtures("train_stack")
class TestARealCpuRun:
    """A tiny SFT on CPU: 8 rows, batch 2 = 4 optimizer steps per epoch."""

    def test_no_leaves_no_checkpoint_and_still_writes_the_final_adapter(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        wrapper, out = _wrapper(tmp_path, "sft", "no")

        wrapper.train()

        assert _checkpoints(out) == []
        _assert_the_final_adapter_is_there(out)

    def test_steps_still_leaves_the_last_step_checkpoint_as_before(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        # save_steps far beyond the run: only the end-of-training save can write one.
        wrapper, out = _wrapper(tmp_path, "sft", "steps", save_steps=1000)

        wrapper.train()

        assert _checkpoints(out) == ["checkpoint-4"]
        _assert_the_final_adapter_is_there(out)

    def test_the_default_is_the_steps_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        wrapper, out = _wrapper(tmp_path, "sft", "steps")

        wrapper.train()

        assert _checkpoints(out) == ["checkpoint-4"]

    def test_epoch_saves_once_per_epoch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        wrapper, out = _wrapper(tmp_path, "sft", "epoch", epochs=2)

        wrapper.train()

        assert _checkpoints(out) == ["checkpoint-4", "checkpoint-8"]
        _assert_the_final_adapter_is_there(out)

    def test_the_trainer_holds_what_the_yaml_asked(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for strategy in ("no", "epoch", "steps"):
            wrapper, _ = _wrapper(tmp_path / strategy, "sft", strategy)
            args = wrapper.trainer.args

            assert str(args.save_strategy.value if hasattr(args.save_strategy, "value")
                       else args.save_strategy) == strategy
            assert args.save_total_limit == 3


@pytest.mark.usefixtures("train_stack")
class TestEveryConstructionStyle:
    """``sft``/``pretrain`` build ``TrainingArguments``; ``dpo``/``reward_model`` build a
    TRL config class. All four must hand ``save_strategy`` to the real trainer."""

    @pytest.mark.parametrize("task", ["dpo", "reward_model", "pretrain"])
    @pytest.mark.parametrize("strategy", ["no", "epoch"])
    def test_the_real_trainer_args_carry_the_strategy(
        self, tmp_path, monkeypatch, task, strategy
    ):
        monkeypatch.chdir(tmp_path)
        wrapper, _ = _wrapper(tmp_path, task, strategy)
        inner = getattr(wrapper, "_inner", wrapper)
        trainer = getattr(inner.trainer, "_trainer", inner.trainer)
        value = trainer.args.save_strategy

        assert str(getattr(value, "value", value)) == strategy
        assert trainer.args.save_total_limit == 3

    @pytest.mark.parametrize("task", ["dpo", "reward_model", "pretrain"])
    def test_no_leaves_no_checkpoint_after_a_real_run(self, tmp_path, monkeypatch, task):
        monkeypatch.chdir(tmp_path)
        wrapper, out = _wrapper(tmp_path, task, "no")

        wrapper.train()

        assert _checkpoints(out) == []
        assert (out / "adapter_model.safetensors").is_file()


class TestPpoFiltersByWhatTheConfigAccepts:
    """PPO's config class differs across trl versions, so its kwargs are filtered by
    what the class takes. Nothing is silently dropped that the user asked for."""

    @staticmethod
    def _config_cls(*names: str) -> type:
        class Config:
            pass

        def init(self, **kwargs):
            pass

        import inspect

        init.__signature__ = inspect.Signature(
            [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            + [
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
                for name in names
            ]
        )
        Config.__init__ = init
        return Config

    def test_the_strategy_reaches_the_ppo_kwargs(self):
        from soup_cli.trainer import ppo

        cls = self._config_cls("save_strategy", "save_steps", "save_total_limit")
        kwargs: dict = {}

        ppo._set_ppo_save_kwargs(kwargs, cls, TrainingConfig(save_strategy="no"))

        assert kwargs == {"save_strategy": "no", "save_total_limit": 3}

    def test_steps_forwards_save_steps(self):
        from soup_cli.trainer import ppo

        cls = self._config_cls("save_strategy", "save_steps", "save_total_limit")
        kwargs: dict = {}

        ppo._set_ppo_save_kwargs(kwargs, cls, TrainingConfig(save_steps=30))

        assert kwargs == {"save_strategy": "steps", "save_steps": 30, "save_total_limit": 3}

    def test_an_older_config_class_without_the_strategy_keeps_steps_working(self):
        from soup_cli.trainer import ppo

        cls = self._config_cls("save_steps")
        kwargs: dict = {}

        ppo._set_ppo_save_kwargs(kwargs, cls, TrainingConfig())

        assert kwargs == {"save_steps": 100}

    def test_a_config_class_that_cannot_honour_the_strategy_is_refused(self):
        from soup_cli.trainer import ppo

        cls = self._config_cls("save_steps")

        with pytest.raises(ValueError) as caught:
            ppo._set_ppo_save_kwargs({}, cls, TrainingConfig(save_strategy="no"))

        message = _clean(caught.value)
        assert "training.save_strategy" in message
        assert "'no'" in message
        assert "does not take save_strategy" in message


# --------------------------------------------------------------------------- soup migrate

_LF = "model_name_or_path: meta-llama/Llama-3.1-8B\nfinetuning_type: lora\ndataset: alpaca_en\n"
_AX = (
    "base_model: meta-llama/Llama-3.1-8B\nadapter: lora\nlora_r: 16\nlora_alpha: 32\n"
    "datasets:\n  - path: d.jsonl\n    type: alpaca\n"
)


def _unsloth_notebook(args: str) -> dict:
    source = (
        "from unsloth import FastLanguageModel\n"
        "model, tokenizer = FastLanguageModel.from_pretrained("
        "model_name='unsloth/Meta-Llama-3.1-8B-bnb-4bit', max_seq_length=2048, "
        "load_in_4bit=True)\n"
        "model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)\n"
        "from trl import SFTTrainer\nfrom transformers import TrainingArguments\n"
        f"trainer = SFTTrainer(model=model, args=TrainingArguments({args}))\n"
    )
    return {
        "cells": [{"cell_type": "code", "source": source, "metadata": {}, "outputs": []}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }


def _migrate_axolotl(tmp_path: Path, extra: str):
    from soup_cli.migrate.axolotl import migrate_axolotl
    from soup_cli.migrate.common import config_to_yaml

    path = tmp_path / "ax.yaml"
    path.write_text(_AX + extra, encoding="utf-8")
    result = migrate_axolotl(path)
    return result, load_config_from_string(config_to_yaml(result))


def _migrate_llamafactory(tmp_path: Path, extra: str):
    from soup_cli.migrate.common import config_to_yaml
    from soup_cli.migrate.llamafactory import migrate_llamafactory

    path = tmp_path / "lf.yaml"
    path.write_text(_LF + extra, encoding="utf-8")
    result = migrate_llamafactory(path)
    return result, load_config_from_string(config_to_yaml(result))


def _migrate_unsloth(tmp_path: Path, args: str):
    from soup_cli.migrate.common import config_to_yaml
    from soup_cli.migrate.unsloth import migrate_unsloth

    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(_unsloth_notebook(args)), encoding="utf-8")
    result = migrate_unsloth(path)
    return result, load_config_from_string(config_to_yaml(result))


_MIGRATORS = {
    "axolotl": (_migrate_axolotl, "save_strategy: {value}\n"),
    "llamafactory": (_migrate_llamafactory, "save_strategy: {value}\n"),
    "unsloth": (_migrate_unsloth, "output_dir='o', save_strategy={value!r}"),
}


class TestMigrateMapsTheStrategy:
    @pytest.mark.parametrize("tool", list(_MIGRATORS))
    @pytest.mark.parametrize("value", ["steps", "epoch", "no"])
    def test_a_quoted_strategy_is_carried_over(self, tmp_path, tool, value):
        migrate, template = _MIGRATORS[tool]
        line = template.format(value=f'"{value}"') if tool != "unsloth" else template.format(
            value=value
        )

        result, cfg = migrate(tmp_path, line)

        assert result["training"]["save_strategy"] == value
        assert cfg.training.save_strategy == value
        assert not any("save_strategy" in w for w in result["_warnings"]), result["_warnings"]

    @pytest.mark.parametrize("tool", ["axolotl", "llamafactory"])
    def test_an_unquoted_yaml_no_is_read_as_no(self, tmp_path, tool):
        """``save_strategy: no`` in a YAML source is ``False`` to PyYAML: still ``no``."""
        migrate, template = _MIGRATORS[tool]

        result, cfg = migrate(tmp_path, template.format(value="no"))

        assert result["training"]["save_strategy"] == "no"
        assert cfg.training.save_strategy == "no"

    @pytest.mark.parametrize("tool", list(_MIGRATORS))
    def test_a_strategy_soup_cannot_express_is_named_not_dropped_silently(
        self, tmp_path, tool
    ):
        migrate, template = _MIGRATORS[tool]
        line = template.format(value='"best"') if tool != "unsloth" else template.format(
            value="best"
        )

        result, cfg = migrate(tmp_path, line)

        assert "save_strategy" not in result["training"]
        assert cfg.training.save_strategy == "steps"
        warning = next(w for w in result["_warnings"] if w.startswith("save_strategy"))
        assert "'best'" in warning
        assert "'steps', 'epoch', 'no'" in warning
        assert not any("not migrated" in w and "save_strategy" in w for w in result["_warnings"])

    @pytest.mark.parametrize("tool", list(_MIGRATORS))
    def test_a_config_without_it_is_untouched(self, tmp_path, tool):
        migrate, _ = _MIGRATORS[tool]

        result, cfg = migrate(tmp_path, "" if tool != "unsloth" else "output_dir='o'")

        assert "save_strategy" not in result["training"]
        assert cfg.training.save_strategy == "steps"

    def test_a_source_save_steps_beside_epoch_does_not_make_the_result_unloadable(
        self, tmp_path
    ):
        """save_steps stays in the "not migrated" list, so the footgun guard has
        nothing to refuse and the migrated file loads."""
        result, cfg = _migrate_axolotl(
            tmp_path, 'save_strategy: "epoch"\nsave_steps: 500\n'
        )

        assert cfg.training.save_strategy == "epoch"
        assert cfg.training.save_steps == 100
        assert any("save_steps" in w and "not migrated" in w for w in result["_warnings"])
