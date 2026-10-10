"""#1675 - ``task: ppo`` could not save a checkpoint on transformers 5.19.0.

trl's experimental ``PPOTrainer`` is a ``transformers.Trainer`` subclass whose
``__init__`` never runs ``Trainer.__init__``. Every attribute ``Trainer.__init__``
assigns and another ``Trainer`` method reads is therefore missing from a PPO
trainer until something sets it. transformers 5.19.0 added one more of those
(``is_distributed_loading_by_transformers``, read by ``Trainer.save_model``), and
the first checkpoint save of any PPO run raised ``AttributeError``.

``ensure_trainer_init_attrs`` gives such a trainer the missing attributes. The
tests below drive it on its own, then through ``PPOTrainerWrapper.setup`` with the
real trl trainer on CPU, including a stand-in for the two 5.19.0 lines so the
failure is reproduced on a transformers that does not have them yet.
"""

from __future__ import annotations

import inspect
import types

import pytest

for _mod in ("torch", "transformers", "peft", "trl", "datasets"):
    pytest.importorskip(_mod)

NEW_IN_5_19 = "is_distributed_loading_by_transformers"


def _skipping_trainer():
    """A ``Trainer`` subclass instance built the way trl's PPOTrainer is: its own
    ``__init__``, no ``super().__init__``."""
    from transformers import Trainer

    class _SkipsTrainerInit(Trainer):
        def __init__(self):
            self.marker = "built without Trainer.__init__"

    return _SkipsTrainerInit()


def _real_trainer(tmp_path):
    import torch
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True)
    return Trainer(model=torch.nn.Linear(2, 2), args=args)


def _installed_init_assigns(name: str) -> bool:
    """Asked a different way than the helper asks (source text, not bytecode)."""
    from transformers import Trainer

    return f"self.{name} = " in inspect.getsource(Trainer.__init__)


def _stand_in_for_5_19(monkeypatch):
    """Make the installed ``Trainer`` behave like 5.19.0 in the two lines that matter:
    ``__init__`` assigns the attribute (trainer.py:470) and ``save_model`` reads it
    (trainer.py:4090). On 5.19.0 and later this repeats what the library does."""
    import functools

    from transformers import Trainer

    real_init, real_save = Trainer.__init__, Trainer.save_model

    @functools.wraps(real_init)
    def init(self, *args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        self.is_distributed_loading_by_transformers = getattr(
            model, "is_distributed_loading_by_transformers", False
        )
        real_init(self, *args, **kwargs)

    @functools.wraps(real_save)
    def save_model(self, output_dir=None, _internal_call=False):
        if self.is_distributed_loading_by_transformers:
            raise AssertionError("nothing in this test shards a model at load time")
        return real_save(self, output_dir, _internal_call)

    monkeypatch.setattr(Trainer, "__init__", init)
    monkeypatch.setattr(Trainer, "save_model", save_model)


# --- the helper on its own -----------------------------------------------------


def test_a_trainer_that_skipped_trainer_init_gets_the_defaults():
    from soup_cli.trainer._trl_compat import ensure_trainer_init_attrs

    trainer = _skipping_trainer()
    assert not hasattr(trainer, "hp_name")
    assert not hasattr(trainer, "is_fsdp_xla_v1_enabled")

    added = ensure_trainer_init_attrs(trainer)

    # both are assigned by Trainer.__init__ on every transformers Soup supports
    assert _installed_init_assigns("hp_name")
    assert _installed_init_assigns("is_fsdp_xla_v1_enabled")
    assert trainer.hp_name is None
    assert trainer.is_fsdp_xla_v1_enabled is False
    assert {"hp_name", "is_fsdp_xla_v1_enabled"} <= set(added)
    assert trainer.marker == "built without Trainer.__init__"


def test_the_5_19_attribute_is_added_exactly_when_the_installed_init_assigns_it():
    from soup_cli.trainer._trl_compat import ensure_trainer_init_attrs

    trainer = _skipping_trainer()
    added = ensure_trainer_init_attrs(trainer)

    if _installed_init_assigns(NEW_IN_5_19):
        assert trainer.is_distributed_loading_by_transformers is False
        assert NEW_IN_5_19 in added
    else:
        assert not hasattr(trainer, NEW_IN_5_19)
        assert NEW_IN_5_19 not in added


def test_the_5_19_attribute_is_added_once_trainer_init_assigns_it(monkeypatch):
    from soup_cli.trainer._trl_compat import ensure_trainer_init_attrs

    _stand_in_for_5_19(monkeypatch)
    trainer = _skipping_trainer()

    added = ensure_trainer_init_attrs(trainer)

    assert trainer.is_distributed_loading_by_transformers is False
    assert NEW_IN_5_19 in added
    # the stand-in wraps the real __init__, so what that one assigns still counts
    assert trainer.hp_name is None
    assert trainer.is_fsdp_xla_v1_enabled is False


def test_the_defaults_name_exactly_the_three_attributes():
    from soup_cli.trainer import _trl_compat

    assert set(_trl_compat._TRAINER_INIT_DEFAULTS) == {
        "is_distributed_loading_by_transformers",
        "is_fsdp_xla_v1_enabled",
        "hp_name",
    }


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (types.SimpleNamespace(is_distributed_loading_by_transformers=True), True),
        (types.SimpleNamespace(), False),
        (None, False),
    ],
)
def test_the_5_19_default_is_read_off_the_model_as_trainer_init_reads_it(
    monkeypatch, model, expected
):
    from soup_cli.trainer._trl_compat import ensure_trainer_init_attrs

    _stand_in_for_5_19(monkeypatch)
    trainer = _skipping_trainer()
    trainer.model = model

    ensure_trainer_init_attrs(trainer)

    assert trainer.is_distributed_loading_by_transformers is expected


def test_an_attribute_the_trainer_already_has_is_not_overwritten(monkeypatch):
    from soup_cli.trainer._trl_compat import ensure_trainer_init_attrs

    _stand_in_for_5_19(monkeypatch)
    trainer = _skipping_trainer()

    def name_trial(trial):
        return "named"

    trainer.is_distributed_loading_by_transformers = True
    trainer.is_fsdp_xla_v1_enabled = True
    trainer.hp_name = name_trial

    added = ensure_trainer_init_attrs(trainer)

    assert added == []
    assert trainer.is_distributed_loading_by_transformers is True
    assert trainer.is_fsdp_xla_v1_enabled is True
    assert trainer.hp_name is name_trial


def test_an_attribute_trainer_init_does_not_assign_is_never_added(monkeypatch):
    from soup_cli.trainer import _trl_compat

    monkeypatch.setitem(
        _trl_compat._TRAINER_INIT_DEFAULTS,
        "no_trainer_init_assigns_this",
        lambda trainer: "default",
    )
    trainer = _skipping_trainer()

    added = _trl_compat.ensure_trainer_init_attrs(trainer)

    assert not hasattr(trainer, "no_trainer_init_assigns_this")
    assert "no_trainer_init_assigns_this" not in added


def test_a_trainer_that_ran_trainer_init_is_untouched(tmp_path):
    from soup_cli.trainer._trl_compat import ensure_trainer_init_attrs

    trainer = _real_trainer(tmp_path)
    before = dict(vars(trainer))

    added = ensure_trainer_init_attrs(trainer)

    assert added == []
    after = vars(trainer)
    assert set(after) == set(before)
    assert all(after[name] is before[name] for name in before)


def test_an_object_that_is_not_a_transformers_trainer_is_untouched():
    from soup_cli.trainer._trl_compat import ensure_trainer_init_attrs

    legacy = types.SimpleNamespace(model=None)

    assert ensure_trainer_init_attrs(legacy) == []
    assert vars(legacy) == {"model": None}


# --- which trl trainers skip Trainer.__init__ ------------------------------------


def _skips_trainer_init(cls) -> bool:
    """Does an ``__init__`` between ``cls`` and ``transformers.Trainer`` fail to call
    a parent ``__init__``? Read from source, class by class along the MRO."""
    import ast
    import textwrap

    from transformers import Trainer

    for klass in cls.__mro__:
        if klass is Trainer:
            return False
        own_init = klass.__dict__.get("__init__")
        if own_init is None:
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(own_init)))
        chains = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "__init__"
            for node in ast.walk(tree)
        )
        if not chains:
            return True
    raise AssertionError(f"{cls.__qualname__} is not a transformers.Trainer subclass")


@pytest.mark.parametrize(
    ("name", "experimental_module", "skips"),
    [
        ("SFTTrainer", None, False),
        ("DPOTrainer", None, False),
        ("GRPOTrainer", None, False),
        ("KTOTrainer", None, False),
        ("RewardTrainer", None, False),
        ("BCOTrainer", "trl.experimental.bco", False),
        ("ORPOTrainer", "trl.experimental.orpo", False),
        ("CPOTrainer", "trl.experimental.cpo", False),
        ("OnlineDPOTrainer", "trl.experimental.online_dpo", False),
        ("PPOTrainer", "trl.experimental.ppo", True),
    ],
)
def test_ppo_is_the_only_trl_trainer_that_skips_trainer_init(
    name, experimental_module, skips
):
    """Every trl trainer a Soup wrapper builds. A trainer that starts skipping
    ``Trainer.__init__`` needs ``ensure_trainer_init_attrs`` in its wrapper; this
    row then fails and says which one."""
    import warnings

    if name == "PPOTrainer":
        from soup_cli.trainer.ppo import _import_ppo_classes

        cls = _import_ppo_classes()[0]
    else:
        from soup_cli.trainer._trl_compat import resolve_trl_symbol

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls = resolve_trl_symbol(name, experimental_module)

    assert _skips_trainer_init(cls) is skips, (
        f"{cls.__module__}.{cls.__qualname__}: skips Trainer.__init__ = {not skips}"
    )


# --- through PPOTrainerWrapper, real trl trainer on CPU -------------------------


def test_ppo_setup_leaves_no_default_missing(tmp_path, monkeypatch):
    from transformers import Trainer

    from tests.test_issue1391_ppo_real_step import _wrapper

    _stand_in_for_5_19(monkeypatch)
    wrapper = _wrapper(tmp_path, monkeypatch)

    trainer = wrapper.trainer
    assert isinstance(trainer, Trainer)
    assert trainer.is_distributed_loading_by_transformers is False
    assert trainer.is_fsdp_xla_v1_enabled is False
    assert trainer.hp_name is None


def test_ppo_saves_a_checkpoint_when_save_model_reads_the_5_19_attribute(
    tmp_path, monkeypatch
):
    from tests.test_issue1391_ppo_real_step import _wrapper

    _stand_in_for_5_19(monkeypatch)
    wrapper = _wrapper(tmp_path, monkeypatch)

    result = wrapper.train()

    assert result["total_steps"] >= 1, result
    assert (tmp_path / "out" / "adapter_config.json").is_file()
