"""#1213 — `soup train` and `soup sweep` must build the same trainer wrapper.

Both commands now route through ``soup_cli.trainer.dispatch.build_trainer``.
Before that, sweep carried a second ``if/elif`` chain over ``cfg.task`` that was
last extended in v0.40.0, with no MLX route at all: for ten task values and for
every ``backend: mlx`` config, an arm trained plain next-token SFT while the run
registry recorded the configured task, and the sweep ranked those numbers.

The task list below is read off ``SoupConfig``'s ``task`` Literal instead of
being hand-written, so a task added to the schema without a dispatch entry fails
the first test rather than training the wrong objective under one command.
"""

from __future__ import annotations

import importlib
import typing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "soup_cli"

#: Every wrapper ``build_trainer`` can construct, keyed by the module that
#: defines it. The classes here depend on peft/trl/mlx, so each site is swapped
#: for a recorder and no backend has to be installed for this file to run.
_WRAPPER_SITES = {
    "soup_cli.trainer.sft": "SFTTrainerWrapper",
    "soup_cli.trainer.dpo": "DPOTrainerWrapper",
    "soup_cli.trainer.online_dpo": "OnlineDPOTrainerWrapper",
    "soup_cli.trainer.grpo": "GRPOTrainerWrapper",
    "soup_cli.trainer.ppo": "PPOTrainerWrapper",
    "soup_cli.trainer.kto": "KTOTrainerWrapper",
    "soup_cli.trainer.orpo": "ORPOTrainerWrapper",
    "soup_cli.trainer.simpo": "SimPOTrainerWrapper",
    "soup_cli.trainer.ipo": "IPOTrainerWrapper",
    "soup_cli.trainer.bco": "BCOTrainerWrapper",
    "soup_cli.trainer.preference": "PreferenceTrainerWrapper",
    "soup_cli.trainer.reward_model": "RewardModelTrainerWrapper",
    "soup_cli.trainer.pretrain": "PretrainTrainerWrapper",
    "soup_cli.trainer.embedding": "EmbeddingTrainerWrapper",
    "soup_cli.trainer.distill": "DistillTrainerWrapper",
    "soup_cli.trainer.prm": "PRMTrainerWrapper",
    "soup_cli.trainer.classifier": "ClassifierTrainerWrapper",
    "soup_cli.trainer.unlearn": "UnlearnTrainerWrapper",
    "soup_cli.trainer.mole_routing": "MoleRoutingTrainerWrapper",
    "soup_cli.trainer.tts": "TTSTrainerWrapper",
    "soup_cli.trainer.asr": "AsrTrainerWrapper",
}

#: transformer-backend dispatch: task -> wrapper class name.
_EXPECTED = {
    "sft": "SFTTrainerWrapper",
    "dpo": "DPOTrainerWrapper",
    "online_dpo": "OnlineDPOTrainerWrapper",
    "grpo": "GRPOTrainerWrapper",
    "ppo": "PPOTrainerWrapper",
    "kto": "KTOTrainerWrapper",
    "orpo": "ORPOTrainerWrapper",
    "simpo": "SimPOTrainerWrapper",
    "ipo": "IPOTrainerWrapper",
    "bco": "BCOTrainerWrapper",
    "preference": "PreferenceTrainerWrapper",
    "reward_model": "RewardModelTrainerWrapper",
    "pretrain": "PretrainTrainerWrapper",
    "embedding": "EmbeddingTrainerWrapper",
    "distill": "DistillTrainerWrapper",
    "prm": "PRMTrainerWrapper",
    "classifier": "ClassifierTrainerWrapper",
    "reranker": "ClassifierTrainerWrapper",
    "cross_encoder": "ClassifierTrainerWrapper",
    "unlearn": "UnlearnTrainerWrapper",
    "moe_lora_routing": "MoleRoutingTrainerWrapper",
    "tts": "TTSTrainerWrapper",
    "asr": "AsrTrainerWrapper",
}

#: MLX rows resolve through ``mlx_routing``. Same three tasks the registry
#: accepts; the name is what ``resolve_trainer`` would have imported.
_MLX_EXPECTED = {
    "sft": "MLXSFTTrainerWrapper",
    "dpo": "MLXDPOTrainerWrapper",
    "grpo": "MLXGRPOTrainerWrapper",
}

#: The minimum each task's own validator asks for on top of base + data.
_TASK_EXTRAS: dict[str, dict] = {
    "preference": {"training": {"preference_loss": "simpo"}},
    "tts": {"modality": "audio_out", "training": {"tts_family": "orpheus"}},
    "classifier": {"training": {"num_labels": 2}},
    "reranker": {"training": {"num_labels": 2}},
    "cross_encoder": {"training": {"num_labels": 2}},
    "distill": {"training": {"teacher_model": "fixtures/teacher"}},
    "unlearn": {
        "training": {"unlearn_method": "npo"},
        "data": {"forget_set": "./forget.jsonl"},
    },
    "moe_lora_routing": {"training": {"mole_task_adapters": ["adapter-a", "adapter-b"]}},
    "online_dpo": {"training": {"online_dpo_judge": "some-judge-model"}},
}


def _schema_tasks() -> tuple[str, ...]:
    from soup_cli.config.schema import SoupConfig

    annotation = SoupConfig.model_fields["task"].annotation
    return typing.get_args(annotation)


def _recorder(name: str, built: list[str]):
    """A stand-in wrapper that records the class name and satisfies the CLI."""

    class _Recorder:
        def __init__(self, cfg, **kwargs):
            built.append(name)

        def setup(self, dataset):
            pass

        def train(self, **kwargs):
            return {
                "initial_loss": 1.0,
                "final_loss": 0.5,
                "total_steps": 1,
                "duration_secs": 1.0,
                "output_dir": "out",
                "duration": "1s",
            }

    _Recorder.__name__ = name
    _Recorder.__qualname__ = name
    return _Recorder


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Swap every wrapper class for a recorder; collects the names that get built.

    Every wrapper module is imported *before* the first patch lands. Some of
    them subclass a class that lives in another of them (`tts.py` subclasses
    `sft.py`'s wrapper), and the base is bound at that module's first import. If
    the patch were already in place, the real class would inherit from a
    recorder and stay that way for the rest of the session, breaking any later
    test that walks the MRO or monkeypatches the real base.
    """
    names: list[str] = []
    modules = {module_path: importlib.import_module(module_path) for module_path in _WRAPPER_SITES}
    for module_path, cls_name in _WRAPPER_SITES.items():
        monkeypatch.setattr(modules[module_path], cls_name, _recorder(cls_name, names))

    from soup_cli.trainer import mlx_routing

    monkeypatch.setattr(
        mlx_routing,
        "get_mlx_trainer",
        lambda task: _recorder(_MLX_EXPECTED[task], names),
    )
    return names


def _sweep_config(task: str):
    from soup_cli.config.schema import SoupConfig

    fields: dict = {"base": "some-model", "task": task, "data": {"train": "./d.jsonl"}}
    for key, value in _TASK_EXTRAS.get(task, {}).items():
        existing = fields.get(key)
        fields[key] = {**existing, **value} if isinstance(existing, dict) else value
    return SoupConfig(**fields)


def _run_sweep_arm(cfg) -> None:
    """Run one sweep arm with everything outside the dispatch stubbed out."""
    from soup_cli.commands.sweep import _run_single

    tracker = MagicMock()
    tracker.start_run.return_value = "run-1"
    with (
        patch("soup_cli.data.loader.load_dataset", return_value={"train": [{"text": "x"}]}),
        patch("soup_cli.utils.gpu.detect_device", return_value=("cpu", "CPU")),
        patch("soup_cli.utils.gpu.get_gpu_info", return_value={}),
        patch("soup_cli.experiment.tracker.ExperimentTracker", return_value=tracker),
        patch("soup_cli.monitoring.display.TrainingDisplay"),
    ):
        _run_single(cfg, {}, "sweep_1", None)


class TestDispatchParity:
    def test_every_schema_task_has_a_dispatch_entry(self, built: list[str]) -> None:
        """Walk the Literal: a task the schema grows without a wrapper fails here."""
        from soup_cli.trainer.dispatch import build_trainer

        assert set(_schema_tasks()) == set(_EXPECTED), (
            "SoupConfig.task and the dispatch table disagree: update "
            "soup_cli/trainer/dispatch.py::build_trainer and this file together"
        )

        for task, expected_name in _EXPECTED.items():
            built.clear()
            build_trainer(SimpleNamespace(task=task, backend="transformers"), device="cpu")
            assert built == [expected_name], f"task={task!r} built {built}"

    def test_mlx_backend_goes_through_the_registry(self, built: list[str]) -> None:
        from soup_cli.trainer.dispatch import build_trainer

        for task, expected_name in _MLX_EXPECTED.items():
            built.clear()
            build_trainer(SimpleNamespace(task=task, backend="mlx"), device="metal")
            assert built == [expected_name], f"backend=mlx task={task!r} built {built}"

    def test_sweep_builds_the_wrapper_the_schema_names(self, built: list[str]) -> None:
        """The sweep-side half of the parity claim, over every task value."""
        for task, expected_name in _EXPECTED.items():
            built.clear()
            _run_sweep_arm(_sweep_config(task))
            assert built == [expected_name], f"sweep task={task!r} built {built}"

    def test_sweep_builds_the_mlx_wrapper_too(self, built: list[str]) -> None:
        """`backend: mlx` used to reach the SFT wrapper in sweep, and only there.

        The schema admits MLX for ``sft`` alone (it refuses the other registry
        entries before dispatch), so this row is the one end-to-end MLX case the
        config layer allows.
        """
        cfg = _sweep_config("sft").model_copy(update={"backend": "mlx"})
        built.clear()
        _run_sweep_arm(cfg)
        assert built == ["MLXSFTTrainerWrapper"], built

    def test_an_unknown_task_raises_instead_of_falling_back_to_sft(self, built: list[str]) -> None:
        from soup_cli.trainer.dispatch import build_trainer

        with pytest.raises(ValueError) as excinfo:
            build_trainer(SimpleNamespace(task="not_a_task", backend="transformers"), device="cpu")

        assert "not_a_task" in str(excinfo.value)
        assert built == [], "an unknown task must not build an SFT wrapper"

    def test_sweeping_the_shrink_heal_config_builds_the_distill_wrapper(
        self, built: list[str], tmp_path: Path
    ) -> None:
        """`soup shrink --heal-steps` writes a `task: distill` config; sweep used to SFT it."""
        from soup_cli.commands.shrink import _build_heal_config_yaml
        from soup_cli.config.loader import load_config

        heal_yaml = _build_heal_config_yaml(
            pruned_dir=str(tmp_path / "pruned"),
            teacher="fixtures/teacher",
            heal_data=str(tmp_path / "heal.jsonl"),
            steps=200,
            out_dir=str(tmp_path / "healed"),
            heal_rows=1000,
        )
        config_path = tmp_path / "soup.yaml"
        config_path.write_text(heal_yaml, encoding="utf-8")
        cfg = load_config(config_path)
        assert cfg.task == "distill"

        built.clear()
        _run_sweep_arm(cfg)
        assert built == ["DistillTrainerWrapper"], built


class TestNoSecondChain:
    """The regression this file exists for was a chain, not a wrong branch."""

    @pytest.mark.parametrize("filename", ["train.py", "sweep.py"])
    def test_command_calls_the_shared_dispatch_only(self, filename: str) -> None:
        source = (_SRC / "commands" / filename).read_text(encoding="utf-8")

        assert "build_trainer(" in source, f"{filename} does not call build_trainer"
        assert "elif cfg.task ==" not in source, (
            f"{filename} grew a task chain again; route it through "
            "soup_cli/trainer/dispatch.py instead"
        )


class TestTheKwargsReachTheWrapper:
    """`build_trainer` forwards each command's kwargs unchanged (#1213).

    The parity tests above compare which *class* gets built. Without this class,
    `build_trainer` could stop forwarding `trust_remote_code`,
    `deepspeed_config`, `fsdp_config` or `report_to` and nothing would fail:
    `soup train` would silently ignore `--trust-remote-code`, `--deepspeed`,
    FSDP and trackers, and a sweep arm could lose `device=` and fall back to the
    wrapper's own `"cuda"` default. Both mutations survive the recorder fixtures
    because the recorder accepts `**kwargs` and never looks at them.
    """

    #: What `soup train` passes (commands/train.py builds exactly these five).
    TRAIN_KWARGS = {
        "device": "cpu",
        "report_to": "none",
        "deepspeed_config": "ds_config.json",
        "fsdp_config": {"fsdp": "full_shard"},
        "trust_remote_code": True,
    }

    @pytest.fixture
    def seen(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        received: dict = {}

        def recorder(name: str):
            class _Recorder:
                def __init__(self, cfg, **kwargs):
                    received[name] = kwargs

                def setup(self, dataset):
                    pass

                def train(self, **kwargs):
                    return {
                        "initial_loss": 1.0,
                        "final_loss": 0.5,
                        "total_steps": 1,
                        "duration_secs": 1.0,
                        "output_dir": "out",
                        "duration": "1s",
                    }

            return _Recorder

        modules = {path: importlib.import_module(path) for path in _WRAPPER_SITES}
        for path, cls_name in _WRAPPER_SITES.items():
            monkeypatch.setattr(modules[path], cls_name, recorder(cls_name))
        from soup_cli.trainer import mlx_routing

        monkeypatch.setattr(
            mlx_routing, "get_mlx_trainer", lambda task: recorder(_MLX_EXPECTED[task])
        )
        return received

    def test_build_trainer_forwards_every_kwarg_unchanged(self, seen: dict) -> None:
        from soup_cli.trainer.dispatch import build_trainer

        rows = [(task, "transformers", name) for task, name in _EXPECTED.items()]
        rows += [(task, "mlx", name) for task, name in _MLX_EXPECTED.items()]
        for task, backend, name in rows:
            seen.clear()
            build_trainer(SimpleNamespace(task=task, backend=backend), **self.TRAIN_KWARGS)
            assert seen == {name: self.TRAIN_KWARGS}, f"{backend}/{task}: {seen}"

    def test_sweep_passes_the_detected_device(self, seen: dict) -> None:
        for task, name in _EXPECTED.items():
            seen.clear()
            _run_sweep_arm(_sweep_config(task))
            assert seen == {name: {"device": "cpu"}}, f"sweep {task}: {seen}"
