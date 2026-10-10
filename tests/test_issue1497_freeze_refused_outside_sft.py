"""#1497 — ``freeze_layers`` / ``freeze_ratio`` are refused at load where no trainer applies them.

Both fields loaded on every task, but only ``SFTTrainerWrapper._setup_transformers``
reads them (``tts`` inherits it). A ``freeze_layers: 2`` on a DPO or GRPO run
trained every layer and said nothing. The config now stops at load, naming the
field, the task and the way out. The split is ``FREEZE_APPLYING_TASKS``, and
:class:`TestTheSplitFollowsTheTrainers` ratchets it against the wrappers that
``trainer/dispatch.py`` builds, so a trainer that starts (or stops) reading the
fields has to move its task.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import typing
from pathlib import Path

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import FREEZE_APPLYING_TASKS, SoupConfig

_TASKS = tuple(typing.get_args(SoupConfig.model_fields["task"].annotation))
_REFUSED = tuple(t for t in _TASKS if t not in FREEZE_APPLYING_TASKS)

# The minimum each task needs to load at all (as in the #1357 tests).
_TOP = {"tts": {"modality": "audio_out"}}
_DATA = {
    "dpo": {"format": "dpo"}, "kto": {"format": "kto"}, "pretrain": {"format": "plaintext"},
    "unlearn": {"forget_set": "./f.jsonl"},
}
_TRAINING = {
    "preference": {"preference_loss": "dpo"},
    "tts": {"tts_family": "orpheus"},
    "classifier": {"num_labels": 2}, "reranker": {"num_labels": 2},
    "cross_encoder": {"num_labels": 2},
    "unlearn": {"unlearn_method": "npo"},
    "moe_lora_routing": {"mole_task_adapters": ["./a", "./b"]},
    "distill": {"teacher_model": "org/t"},
    "online_dpo": {"online_dpo_judge": "pairrm"},
}
# preference builds one of these per preference_loss (trainer/preference.py).
_PREFERENCE_DELEGATES = ("dpo", "simpo", "orpo", "ipo", "bco")


def _yaml(task, **training):
    raw = {
        "base": "org/m", "task": task, **_TOP.get(task, {}),
        "data": {"train": "./x.jsonl", **_DATA.get(task, {})},
        "training": {**_TRAINING.get(task, {}), **training},
    }
    return yaml.safe_dump(raw)


def _refusal(text):
    with pytest.raises(ValueError) as info:
        load_config_from_string(text)
    return " ".join(str(info.value).split())


_ONE_FIELD = [("freeze_layers", 2), ("freeze_ratio", 0.5)]


class TestRefusedWhereNothingAppliesIt:
    @pytest.mark.parametrize("task", _REFUSED)
    @pytest.mark.parametrize(("field", "value"), _ONE_FIELD)
    def test_the_message_names_the_field_the_task_and_the_fix(self, task, field, value):
        message = _refusal(_yaml(task, **{field: value}))

        assert f"training.{field} is not applied by task={task!r}" in message, message
        assert "would train every layer" in message
        assert "Use task: sft, or remove the key." in message

    def test_both_fields_are_named_in_one_refusal(self):
        message = _refusal(_yaml("dpo", freeze_layers=2, freeze_ratio=0.5))

        assert (
            "training.freeze_layers and training.freeze_ratio are not applied by "
            "task='dpo'"
        ) in message, message
        assert "remove both keys" in message

    @pytest.mark.parametrize("task", _REFUSED)
    def test_without_the_fields_the_task_still_loads(self, task):
        cfg = load_config_from_string(_yaml(task))

        assert cfg.training.freeze_layers is None
        assert cfg.training.freeze_ratio is None

    @pytest.mark.parametrize("backend", ["transformers", "unsloth"])
    @pytest.mark.parametrize("task", ["dpo", "reward_model"])
    def test_the_refusal_does_not_depend_on_the_backend(self, task, backend):
        raw = yaml.safe_load(_yaml(task, freeze_layers=2))
        raw["backend"] = backend

        message = _refusal(yaml.safe_dump(raw))

        assert f"training.freeze_layers is not applied by task={task!r}" in message, message

    @pytest.mark.parametrize("task", ["dpo", "pretrain", "kto", "orpo", "grpo"])
    def test_the_issue_reproduction_is_refused(self, task):
        """The issue's table: ``freeze_layers: 2``, ``quantization: none`` and a LoRA
        block loaded on each of these and was ignored."""
        message = _refusal(
            _yaml(task, freeze_layers=2, quantization="none", lora={"r": 8})
        )

        assert f"task={task!r}" in message, message


class TestTheApplyingTasksAreUnchanged:
    # Written out, not read from FREEZE_APPLYING_TASKS, so dropping a task from
    # the constant cannot also drop its control.
    @pytest.mark.parametrize("task", ["sft", "tts"])
    @pytest.mark.parametrize(("field", "value"), _ONE_FIELD)
    def test_still_loads(self, task, field, value):
        cfg = load_config_from_string(_yaml(task, **{field: value}))

        assert getattr(cfg.training, field) == value

    def test_sft_with_lora_r_zero_still_loads(self):
        """#341: ``freeze_layers`` stays legal with full fine-tuning on sft."""
        cfg = load_config_from_string(
            _yaml("sft", freeze_layers=1, quantization="none", lora={"r": 0})
        )

        assert cfg.training.freeze_layers == 1


class TestSoupDoctorReportsIt:
    def test_doctor_exits_2_naming_the_task(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from tests.conftest import strip_ansi

        path = tmp_path / "soup.yaml"
        path.write_text(_yaml("dpo", freeze_layers=2), encoding="utf-8")

        result = CliRunner().invoke(app, ["doctor", "--config", str(path)])

        assert result.exit_code == 2, result.output
        out = " ".join(strip_ansi(result.output).split())
        assert "training.freeze_layers is not applied by task='dpo'" in out


class TestValidatorOrder:
    def test_it_sits_before_the_backend_refusal_and_after_the_resolver(self):
        """#795's resolver stays first and #1357's backend refusal stays last; this
        one refuses on every backend, so it belongs in front of that."""
        validators = SoupConfig.__pydantic_decorators__.model_validators
        after = [name for name, dec in validators.items() if dec.info.mode == "after"]

        assert after[0] == "_resolve_quantization_for_unhonouring_tasks"
        assert (
            0
            < after.index("_validate_freeze_is_applied")
            < after.index("_validate_unsloth_has_a_setup")
        ), after[-3:]


def _dispatched_wrappers():
    """task -> wrapper class, read from the if-chain in ``trainer/dispatch.py``.

    Read from the source rather than listed here, so a task added to the
    dispatcher is scanned without anyone remembering this file.
    """
    from soup_cli.trainer import dispatch

    tree = ast.parse(Path(inspect.getsourcefile(dispatch)).read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if not (isinstance(test.left, ast.Name) and test.left.id == "task"):
            continue
        right = test.comparators[0]
        if isinstance(test.ops[0], ast.Eq) and isinstance(right, ast.Constant):
            tasks = [right.value]
        elif isinstance(test.ops[0], ast.In) and isinstance(right, ast.Tuple):
            tasks = [elt.value for elt in right.elts]
        else:
            continue
        imports = [n for n in node.body if isinstance(n, ast.ImportFrom)]
        assert len(imports) == 1, f"expected one import under task={tasks}"
        module = importlib.import_module(imports[0].module)
        cls = getattr(module, imports[0].names[0].name)
        for task in tasks:
            found[task] = cls
    return found


def _reads_the_freeze_fields(cls):
    """True when the wrapper, or a soup class it inherits from, reads either field
    as an attribute. Comments and docstrings do not count.

    The limit of this scan: a trainer that gets the freeze plan through a shared
    helper outside its own classes, or through ``getattr(tcfg, "freeze_layers")``,
    is not seen by it."""
    for klass in cls.__mro__:
        if not klass.__module__.startswith("soup_cli."):
            continue
        tree = ast.parse(inspect.getsource(inspect.getmodule(klass)))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == klass.__name__:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and sub.attr in (
                        "freeze_layers", "freeze_ratio",
                    ):
                        return True
    return False


class TestTheSplitFollowsTheTrainers:
    """``FREEZE_APPLYING_TASKS`` must follow the trainers: a wrapper that reads the
    fields must not have its task refused, and a wrapper that does not must."""

    def test_every_task_has_a_wrapper(self):
        assert set(_dispatched_wrappers()) == set(_TASKS)

    @pytest.mark.parametrize("task", _TASKS)
    def test_the_schema_list_matches_the_wrapper(self, task):
        wrappers = _dispatched_wrappers()
        if task == "preference":
            applies = all(_reads_the_freeze_fields(wrappers[t]) for t in _PREFERENCE_DELEGATES)
        else:
            applies = _reads_the_freeze_fields(wrappers[task])

        assert (task in FREEZE_APPLYING_TASKS) == applies, (
            f"task={task!r}: its trainer {'reads' if applies else 'does not read'} "
            "freeze_layers / freeze_ratio; move it in or out of FREEZE_APPLYING_TASKS"
        )

    def test_the_scan_sees_the_sft_call_site(self):
        """Guard on the scan itself: if it stopped seeing SFT's read, every task
        would look refusable and the test above would still be consistent."""
        assert _reads_the_freeze_fields(_dispatched_wrappers()["sft"])
        assert not _reads_the_freeze_fields(_dispatched_wrappers()["dpo"])


def test_a_dumped_config_still_reloads():
    """A dumped config writes both fields out as null; that is not a request."""
    cfg = load_config_from_string(_yaml("dpo"))
    dumped = yaml.safe_dump(cfg.model_dump(mode="json"))

    assert "freeze_layers: null" in dumped
    assert load_config_from_string(dumped).task == "dpo"
